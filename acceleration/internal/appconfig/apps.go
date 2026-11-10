package appconfig

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// AppSettings returns what an app has turned on or off, read on every session an end user
// starts. Stream writes them in Postgres and no operation of the router does, so nothing
// forgets the key: a change is seen once the TTL has passed, as the settings an API key is
// read with (Lookup) are.
func (s *Store) AppSettings(ctx context.Context, appID string) (store.AppSettings, error) {
	return read(ctx, s, key("app-settings", appID), func(ctx context.Context) (store.AppSettings, error) {
		return s.db.AppSettingsFor(ctx, appID)
	})
}
