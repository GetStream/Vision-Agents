package appconfig

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// lastUsedInterval throttles how often a key's use is recorded. Writing on every request
// would double the writes of a busy key, and recording nothing means nobody can answer
// whether a key is still in use, so nobody ever revokes one.
const lastUsedInterval = time.Minute

// APIKey returns the app a live key belongs to, with the sealed secret to verify the
// caller's token with.
//
// It is the read on every authenticated request, so it is the one the cache is for. The
// expiry is checked here as well as in the query: a key cached while it was live must stop
// working when it lapses, not when the entry does.
func (s *Store) APIKey(ctx context.Context, id string) (store.APIKeyOwner, error) {
	owner, err := read(ctx, s, key("key", id), func(ctx context.Context) (store.APIKeyOwner, error) {
		return s.db.LiveAPIKey(ctx, id)
	})
	if err != nil {
		return store.APIKeyOwner{}, err
	}
	if owner.ExpiresAt != nil && !owner.ExpiresAt.After(time.Now().UTC()) {
		return store.APIKeyOwner{}, store.ErrNoAPIKey
	}
	return owner, nil
}

// TouchAPIKey records that a key was used, at most once an interval per process.
//
// The store already refuses to write more often than that, but it took a round trip to
// Postgres to find out, which on a busy key is a second query on every request for a
// column nothing reads in anger. Remembering here what this process last wrote makes the
// common case no query at all.
func (s *Store) TouchAPIKey(ctx context.Context, id string) error {
	now := time.Now().UTC()
	if last, ok := s.touched.Load(id); ok && now.Sub(last.(time.Time)) < lastUsedInterval {
		return nil
	}
	s.touched.Store(id, now)
	if err := s.db.TouchAPIKey(ctx, id, lastUsedInterval); err != nil {
		s.touched.Delete(id)
		return err
	}
	return nil
}

// RevokeAPIKey stops a key working, here and on every other replica.
//
// Creating one has no counterpart here: a fresh id is one nothing can have cached, since
// a lookup that found no key caches nothing.
func (s *Store) RevokeAPIKey(ctx context.Context, id, by string) error {
	if err := s.db.RevokeAPIKey(ctx, id, by); err != nil {
		return err
	}
	s.forget(ctx, key("key", id))
	s.touched.Delete(id)
	return nil
}
