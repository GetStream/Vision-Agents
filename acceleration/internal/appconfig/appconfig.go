// Package appconfig reads the configuration a request is measured against -- the API key
// it presented, the app and organization behind it, their policies, and the agents,
// router presets and voices they have written down -- and keeps it in front of Postgres.
//
// It exists because that configuration is read on every request and written a few times a
// week. The key alone was a join on every authenticated call, which is a round trip to
// Postgres before anything the caller asked for had begun.
//
// Two tiers, one connection. Redis holds the value, and the client-side cache rueidis
// keeps alongside it answers without a round trip at all; Redis invalidates that local
// copy itself when the key is deleted, so every replica of the router forgets a revoked
// key at once rather than each on its own timer. Postgres remains the only writer and the
// only truth: every value here was loaded from it, and every write goes to it first and
// deletes the key afterwards.
//
// A deployment with no Redis reads Postgres directly, and so does one whose Redis is
// unreachable. Nothing here fails a request because a cache is missing.
package appconfig

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/redis/rueidis"
	"github.com/redis/rueidis/rueidisaside"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tracing"
)

var tracer = tracing.Tracer("appconfig")

// DefaultTTL is how long a value lives in Redis when nothing deletes it first. It is long
// because every write deletes the keys it changed, and the expiry is only the backstop for
// a deletion that did not land.
const DefaultTTL = time.Hour

// Options configures the store.
type Options struct {
	// Store is where the configuration is read from and written to. Required.
	Store *store.Store
	// Address is a host:port, for example localhost:6379. An empty one is a deployment
	// with no Redis, which reads Postgres every time.
	Address  string
	Username string
	Password string
	// TTL is how long a cached value lives. DefaultTTL when unset.
	TTL    time.Duration
	Logger *slog.Logger
}

// Store reads configuration, through Redis and the local cache where there is one.
type Store struct {
	db     *store.Store
	cache  rueidisaside.CacheAsideClient
	ttl    time.Duration
	logger *slog.Logger
	// touched is when this process last recorded each key's use, keyed by key id. It is
	// per process rather than shared because it guards a write nobody reads in anger.
	touched sync.Map
}

// New returns a store reading through Redis, or straight from Postgres when no address
// was given.
func New(options Options) (*Store, error) {
	if options.Store == nil {
		return nil, errors.New("appconfig: a store is required")
	}
	if options.TTL == 0 {
		options.TTL = DefaultTTL
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	cached := &Store{db: options.Store, ttl: options.TTL, logger: options.Logger}
	if options.Address == "" {
		return cached, nil
	}

	cache, err := rueidisaside.NewClient(rueidisaside.ClientOption{
		ClientOption: rueidis.ClientOption{
			InitAddress: []string{options.Address},
			Username:    options.Username,
			Password:    options.Password,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("appconfig: connect to redis: %w", err)
	}
	cached.cache = cache
	return cached, nil
}

// DB is the Postgres store underneath, for the reads and writes that are not configuration.
func (s *Store) DB() *store.Store { return s.db }

// Close releases the Redis connection.
func (s *Store) Close() {
	if s.cache != nil {
		s.cache.Close()
	}
}

// key names one cached value. Every key is prefixed, so what this package put in Redis can
// be told from what the rest of the router did.
func key(parts ...string) string { return "appcfg:" + strings.Join(parts, ":") }

// read returns what load produced, from the local cache, from Redis, or from Postgres, in
// that order.
//
// An error from load is the caller's answer, unchanged, so that ErrNoAPIKey and its
// siblings still read as themselves. An error from Redis is not: the value is loaded from
// Postgres instead, because a cache nobody can reach is a slow deployment and not a broken
// one.
func read[T any](ctx context.Context, s *Store, name string, load func(context.Context) (T, error)) (T, error) {
	var value T
	if s.cache == nil {
		return load(ctx)
	}

	ctx, span := tracer.Start(ctx, "appconfig.read")
	defer span.End()

	var refused error
	encoded, err := s.cache.Get(ctx, s.ttl, name, func(ctx context.Context, _ string) (string, error) {
		loaded, err := load(ctx)
		if err != nil {
			refused = err
			return "", err
		}
		raw, err := json.Marshal(loaded)
		return string(raw), err
	})
	switch {
	case refused != nil:
		return value, refused
	case err != nil:
		s.logger.Error("could not read configuration from redis, reading postgres",
			"key", name, "error", err)
		return load(ctx)
	}

	if err := json.Unmarshal([]byte(encoded), &value); err != nil {
		return value, fmt.Errorf("appconfig: decode %s: %w", name, err)
	}
	return value, nil
}

// forget drops keys from Redis, which drops them from every replica's local cache too.
//
// A deletion that does not land is logged rather than returned: the write it followed has
// already happened, and refusing it afterwards would tell the caller their change was
// rejected when it was not. The TTL is what covers the difference.
func (s *Store) forget(ctx context.Context, names ...string) {
	if s.cache == nil {
		return
	}
	for _, name := range names {
		if err := s.cache.Del(ctx, name); err != nil {
			s.logger.Error("could not drop a cached configuration key",
				"key", name, "error", err)
		}
	}
}
