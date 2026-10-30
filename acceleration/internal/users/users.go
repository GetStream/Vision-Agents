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
//
// The email a backend says its user signs in with is said on every request too, and is
// held in an LRU of its own, by user and address, for emailTTL: a changed address is written
// at once.
package users

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// DefaultTTL is how long a user's key lives in Redis. It is a backstop rather than a
// policy: the key says the row exists, the row is never deleted on its own, and claiming
// drops the key itself.
const DefaultTTL = time.Hour

// emailTTL is how long an email this process wrote is not written again. It is short, and
// not the TTL, because the row is what the first consent for an owner_email connector is
// checked against (api ownerEmailRefusal): an address another replica wrote under this one
// is the one that consent admits until this replica writes its own again, so a held address
// leaves the row stale for at most this long.
const emailTTL = time.Minute

// emailStripes is how many locks the email writes are spread over, so that one slow write
// holds up only the users that hash to its lock. Any count is correct; 64 is a judgement,
// not a measurement.
const emailStripes = 64

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
	emails *seen
	ttl    time.Duration
	logger *slog.Logger

	// emailTTL is emailTTL; a test shortens it.
	emailTTL time.Duration
	// emailLocks keep each user's write and hold together, so that two requests saying two
	// addresses at once leave the held one the one the row holds.
	emailLocks [emailStripes]sync.Mutex
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
		emails: newSeen(options.Capacity),
		ttl:    options.TTL,
		logger: options.Logger,

		emailTTL: emailTTL,
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

// Seen records that an app acted for a user, as the kind their own token proved them to
// be, and does nothing for one already written down from a sighting that settles it.
//
// The kind is what the credential proved them to be, so only a verified caller belongs
// here: an anonymous one goes by a name nobody checked, and recording it would fill the
// table with whatever names were asked for.
//
// A user's own full token is the one sighting that changes a row already there
// (store.RecordAccount), so it is settled only by having been written: one held from a
// guest token, or from a backend naming them, is written again.
func (r *Recorder) Seen(ctx context.Context, customerID, userID, kind string) error {
	if kind == "" {
		return nil
	}
	return r.record(ctx, customerID, userID, kind)
}

// Named records a user a backend said it acts for, as an authenticated user when there is
// no row yet. A backend may name any user, a guest of its own included, so this proves
// nothing about which kind they are and changes no row already there.
func (r *Recorder) Named(ctx context.Context, customerID, userID string) error {
	return r.record(ctx, customerID, userID, named)
}

// named is what a user a backend named is held with: there is a row, of a kind this does
// not know.
const named = "named"

// record writes a sighting down unless one held for the user settles it, and holds the
// user with the sighting.
func (r *Recorder) record(ctx context.Context, customerID, userID, sighting string) error {
	if customerID == "" || userID == "" {
		return nil
	}
	if held, found := r.seen.Value(customerID, userID); found && settles(held, sighting) {
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
		case err == nil && found != "" && settles(found, sighting):
			r.seen.Keep(customerID, userID, found, time.Time{})
			return nil
		case err != nil && !rueidis.IsRedisNil(err):
			r.logger.Error("could not read a user from redis, writing postgres",
				"customer", customerID, "user", userID, "error", err)
		}
	}

	var err error
	switch sighting {
	case store.UserKindAuthenticated:
		err = r.db.RecordAccount(ctx, customerID, userID)
	case named:
		err = r.db.RecordUser(ctx, &store.User{CustomerID: customerID, ID: userID, Kind: store.UserKindAuthenticated})
	default:
		err = r.db.RecordUser(ctx, &store.User{CustomerID: customerID, ID: userID, Kind: sighting})
	}
	if err != nil {
		return err
	}

	if r.redis != nil {
		set := r.redis.B().Set().Key(name).Value(sighting).Ex(r.ttl).Build()
		if err := r.redis.Do(ctx, set).Error(); err != nil {
			r.logger.Error("could not cache a user in redis",
				"customer", customerID, "user", userID, "error", err)
		}
	}
	r.seen.Keep(customerID, userID, sighting, time.Time{})
	return nil
}

// settles reports whether a user held with held needs no write for sighting. Only a user's
// own full token changes a row already there, so only having written that settles it; any
// other sighting is settled by the row being there at all.
func settles(held, sighting string) bool {
	return sighting != store.UserKindAuthenticated || held == store.UserKindAuthenticated
}

// Email records the address a backend said a user signs in with (store.SetUserEmail), and
// does nothing for the address this process last wrote for them within emailTTL. A backend
// may say it on every request. A changed address is written at once; one another replica
// wrote under this one is written again here once emailTTL ends, so a row lags for at most
// that long. A failed write is not held, so the next request tries again. Like Seen, a
// request naming no app or no user records nothing.
func (r *Recorder) Email(ctx context.Context, customerID, userID, email string) error {
	if customerID == "" || userID == "" || email == "" {
		return nil
	}
	if r.emails.Holds(customerID, userID, email, time.Now()) {
		return nil
	}

	lock := r.emailLock(customerID, userID)
	lock.Lock()
	defer lock.Unlock()
	now := time.Now()
	if err := r.db.SetUserEmail(ctx, customerID, userID, email); err != nil {
		return err
	}
	r.emails.Keep(customerID, userID, email, now.Add(r.emailTTL))
	return nil
}

// emailLock is the lock one user's email writes take.
func (r *Recorder) emailLock(customerID, userID string) *sync.Mutex {
	hash := fnv.New32a()
	hash.Write([]byte(key(customerID, userID)))
	return &r.emailLocks[hash.Sum32()%emailStripes]
}

// Forget drops a user from the caches, for one whose row has changed under them. Deleting
// the key is what tells the other replicas and, once Redis sends it, this process's own
// copy too; this process's entry goes directly as well, so it does not wait for that.
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
