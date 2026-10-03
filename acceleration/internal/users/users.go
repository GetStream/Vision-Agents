// Package users records the end users an app is seen acting for.
//
// Every authenticated request names one, and nearly every one of them names somebody
// already written down, so the work is almost always a write that changes nothing. Three
// tiers keep it off Postgres: an LRU of the last ten thousand users this process saw, a
// key in Redis that the other replicas share, and the row itself.
//
// Redis is told to track the keys this process read, so a user whose row changes under it
// -- a guest who was claimed -- is dropped from every replica's LRU at once rather than
// each on its own timer.
//
// A deployment with no Redis keeps the LRU and writes to Postgres on a miss, and so does
// one whose Redis is unreachable. Nothing here fails a request because a cache is missing,
// and nothing here fails a request at all: recording who called is not what the caller
// asked for.
package users

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// DefaultTTL is how long a user's key lives in Redis. It is a backstop rather than a
// policy: the key says the row exists, the row is never deleted on its own, and claiming
// drops the key itself.
const DefaultTTL = time.Hour

// trackingCacheBytes is how much rueidis keeps per connection. It is small because the
// LRU here is the in-memory tier; rueidis only needs a cache at all because reading
// through it is what turns Redis's tracking on.
const trackingCacheBytes = 1 << 20

// keyPrefix names what this package put in Redis, so it can be told from the rest.
const keyPrefix = "users:"

// Options configures a Recorder.
type Options struct {
	// Store is where users are written. Required.
	Store *store.Store
	// Address is a host:port, for example localhost:6379. An empty one is a deployment
	// with no Redis, which writes through the LRU alone.
	Address  string
	Username string
	Password string
	// TTL is how long a user's key lives in Redis. DefaultTTL when unset.
	TTL time.Duration
	// Capacity is how many users this process holds in memory across every app.
	// DefaultCapacity when unset.
	Capacity int
	Logger   *slog.Logger
}

// Recorder writes down the users an app acts for, once each.
type Recorder struct {
	db     *store.Store
	redis  rueidis.Client
	seen   *seen
	ttl    time.Duration
	logger *slog.Logger
}

// New returns a Recorder reading through Redis, or one holding the LRU alone when no
// address was given.
func New(options Options) (*Recorder, error) {
	if options.Store == nil {
		return nil, errors.New("users: a store is required")
	}
	if options.TTL == 0 {
		options.TTL = DefaultTTL
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	recorder := &Recorder{
		db:     options.Store,
		seen:   newSeen(options.Capacity),
		ttl:    options.TTL,
		logger: options.Logger,
	}
	if options.Address == "" {
		return recorder, nil
	}

	client, err := rueidis.NewClient(rueidis.ClientOption{
		InitAddress:       []string{options.Address},
		Username:          options.Username,
		Password:          options.Password,
		CacheSizeEachConn: trackingCacheBytes,
		OnInvalidations:   recorder.invalidate,
	})
	if err != nil {
		return nil, fmt.Errorf("users: connect to redis: %w", err)
	}
	recorder.redis = client
	return recorder, nil
}

// Close releases the Redis connection.
func (r *Recorder) Close() {
	if r.redis != nil {
		r.redis.Close()
	}
}

// Seen records that an app acted for a user, and does nothing for one already written
// down.
//
// The kind is what the credential proved them to be, so only a verified caller belongs
// here: an anonymous one goes by a name nobody checked, and recording it would fill the
// table with whatever names were asked for.
func (r *Recorder) Seen(ctx context.Context, customerID, userID, kind string) error {
	if customerID == "" || userID == "" || kind == "" {
		return nil
	}
	if r.seen.Get(customerID, userID) {
		return nil
	}

	name := key(customerID, userID)
	if r.redis != nil {
		// Read through the client-side cache, which is what puts the key under Redis's
		// tracking: a later claim deleting it drops this entry on every replica holding
		// one. A Redis that cannot be reached is a slow deployment and not a broken one,
		// so the answer there is the same as a miss.
		found, err := r.redis.DoCache(ctx, r.redis.B().Get().Key(name).Cache(), r.ttl).ToString()
		switch {
		case err == nil && found != "":
			r.seen.Add(customerID, userID)
			return nil
		case err != nil && !rueidis.IsRedisNil(err):
			r.logger.Error("could not read a user from redis, writing postgres",
				"customer", customerID, "user", userID, "error", err)
		}
	}

	user := &store.User{CustomerID: customerID, ID: userID, Kind: kind}
	if err := r.db.RecordUser(ctx, user); err != nil {
		return err
	}

	if r.redis != nil {
		set := r.redis.B().Set().Key(name).Value("1").Ex(r.ttl).Build()
		if err := r.redis.Do(ctx, set).Error(); err != nil {
			r.logger.Error("could not cache a user in redis",
				"customer", customerID, "user", userID, "error", err)
		}
	}
	r.seen.Add(customerID, userID)
	return nil
}

// Forget drops a user from the caches, for one whose row has changed under them. Deleting
// the key is what tells the other replicas; this process's own entry goes directly,
// because a process is not sent an invalidation for its own delete.
func (r *Recorder) Forget(ctx context.Context, customerID, userID string) {
	r.seen.Remove(customerID, userID)
	if r.redis == nil {
		return
	}
	if err := r.redis.Do(ctx, r.redis.B().Del().Key(key(customerID, userID)).Build()).Error(); err != nil {
		r.logger.Error("could not drop a cached user",
			"customer", customerID, "user", userID, "error", err)
	}
}

// invalidate drops what Redis says has gone stale. A nil message is Redis saying it
// cannot name the keys -- a flush, or a restart -- so everything goes.
func (r *Recorder) invalidate(messages []rueidis.RedisMessage) {
	if messages == nil {
		r.seen.Clear()
		return
	}
	for _, message := range messages {
		name, err := message.ToString()
		if err != nil {
			continue
		}
		customerID, userID, named := split(name)
		if named {
			r.seen.Remove(customerID, userID)
		}
	}
}

// key names one user's key. The customer comes first because it carries no colon, which
// is what lets split read the pair back out of an invalidation naming only the key.
func key(customerID, userID string) string {
	return keyPrefix + customerID + ":" + userID
}

func split(name string) (customerID, userID string, ok bool) {
	rest, found := strings.CutPrefix(name, keyPrefix)
	if !found {
		return "", "", false
	}
	customerID, userID, found = strings.Cut(rest, ":")
	return customerID, userID, found
}
