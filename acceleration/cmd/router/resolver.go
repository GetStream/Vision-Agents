package main

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// newConnectorResolver is the one door to a connection's credential: the connections in db,
// their stored credentials sealed by sealer (pgsealed), retrieved through registry's schemes.
// It is nil when connectors are off, which leaves no sealer, or there is no database.
func newConnectorResolver(registry core.Registry, db *store.Store, sealer *auth.Sealer) (*resolver.Resolver, error) {
	if db == nil || sealer == nil {
		return nil, nil
	}
	credentials, err := pgsealed.New(db, sealer)
	if err != nil {
		return nil, err
	}
	return resolver.New(resolver.Config{Store: db, Credentials: credentials, Schemes: registry.Schemes})
}
