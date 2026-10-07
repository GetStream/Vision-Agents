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

// newConnectorTransports builds each connection's outbound client with egress.NewClient over
// connectors (core.Transports): every request asks it for the access credential. One request,
// redirects included, is bounded by connectorHTTPTimeout, the bound the schemes' own requests
// have. It is nil when there is no resolver, which is when connectors are off.
func newConnectorTransports(connectors *resolver.Resolver) (*core.Transports, error) {
	if connectors == nil {
		return nil, nil
	}
	return core.NewTransports(core.TransportsConfig{Resolver: connectors, Timeout: connectorHTTPTimeout})
}
