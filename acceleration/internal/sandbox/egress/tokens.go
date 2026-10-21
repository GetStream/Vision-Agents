package egress

import (
	"context"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Tokens is grant's token as its connection gives it now. An error wrapping
// sandbox.ErrNoCredential means the connection gives none until it is reconnected.
type Tokens func(ctx context.Context, grant Grant) (string, error)

// ResolverTokens asks connections for the credential on every call, so nothing keeps a copy
// and a revoked connection stops at its next request, and reads the token as grant's scheme
// sends it.
func ResolverTokens(connections core.Resolver, schemes map[string]core.Scheme) Tokens {
	return func(ctx context.Context, grant Grant) (string, error) {
		scheme, found := schemes[grant.Scheme]
		if !found {
			return "", stack.Wrap(fmt.Errorf("%w: egress: there is no scheme %q", sandbox.ErrNoCredential, grant.Scheme))
		}
		credential, err := connections.Resolve(ctx, grant.Connection, core.CredentialRequest{})
		if errors.Is(err, resolver.ErrNotConnected) || errors.Is(err, store.ErrNoConnectorConnection) {
			return "", fmt.Errorf("%w: %w", sandbox.ErrNoCredential, err)
		}
		if err != nil {
			return "", err
		}
		return sandbox.BearerToken(ctx, scheme, credential, grant.Hosts[0])
	}
}
