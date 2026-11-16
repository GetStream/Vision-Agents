package appconfig

import (
	"context"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
)

// Lookup resolves an API key to the app holding it, which is what the api_key mode of
// authentication is built on.
//
// It lives here rather than beside the authenticator so that the router and its test
// harness resolve a key the same way, down to the cache in front of it: a measurement of
// what authentication costs is only worth anything if it is measuring the real thing.
func (s *Store) Lookup(sealer *auth.Sealer) auth.Lookup {
	return func(ctx context.Context, key string) (auth.App, error) {
		// The shape of the key is checked before the database is, so a truncated paste
		// costs nothing to reject.
		if !auth.ValidKey(key) {
			return auth.App{}, auth.ErrUnauthenticated
		}
		owner, err := s.APIKey(ctx, key)
		if err != nil {
			return auth.App{}, auth.ErrUnauthenticated
		}
		secret, err := sealer.Open(owner.Sealed)
		if err != nil {
			return auth.App{}, fmt.Errorf("unseal key %s: %w", key, err)
		}
		if err := s.TouchAPIKey(ctx, key); err != nil {
			s.logger.Debug("could not record key use", "key", key, "error", err)
		}
		// The app's settings came back on the same row, so which levels of end user it
		// admits costs nothing beyond the lookup that was already happening. They are
		// inverted on the way across because auth measures a caller against a zero value
		// in the three modes that resolve no app at all, and that zero value has to admit
		// everybody.
		return auth.App{
			OrganizationID: owner.OrganizationID,
			AppID:          owner.AppID,
			Secret:         secret,
			Levels: auth.Levels{
				NoAnonymous: !owner.Settings.AnonymousAllowed(),
				NoGuest:     !owner.Settings.GuestAllowed(),
			},
		}, nil
	}
}
