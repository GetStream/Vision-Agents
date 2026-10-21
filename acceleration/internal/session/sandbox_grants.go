package session

import (
	"context"
	"errors"
	"fmt"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// sandboxGrant adds an opened binding with sandbox set to d's grants, reading the sandbox
// block of the revision its connection was saved on. A connection saved before the connector
// had one gives the sandbox nothing, which fails a required binding and is a warning
// otherwise.
func (m *Manager) sandboxGrant(binding store.ConnectorBinding, connection store.ConnectorConnection,
	manifest core.ResolvedManifest, scheme core.Scheme, d *dispatcher) string {
	if !binding.Sandbox {
		return ""
	}
	if manifest.Sandbox == nil {
		if binding.Required {
			return unavailableSandbox
		}
		m.logger.Warn("a connector binding gives the sandbox no access: reconnect to use sandbox access", "connector", binding.Name,
			"connection", connection.ID, "revision", connection.DefinitionRevision)
		return ""
	}
	d.grants = append(d.grants, sandbox.Grant{
		Connection: core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID},
		Binding:    binding.Name, Env: manifest.Sandbox.Env, Hosts: manifest.Sandbox.Hosts, Required: binding.Required,
	})
	if d.schemes == nil {
		d.schemes = map[string]core.Scheme{}
	}
	d.schemes[binding.Name] = scheme
	return ""
}

// broker is the code_sandbox.broker setting's broker for the session's grants.
func (m *Manager) broker(spec Spec, d *dispatcher) (sandbox.CredentialBroker, error) {
	if m.options.SandboxSecretPrefix == "" {
		return nil, stack.Wrap(errors.New("session: a connector binding sets sandbox, and this router has no " +
			"code_sandbox.secret_prefix (ROUTER_CODE_SANDBOX_SECRET_PREFIX) to name its secrets with"))
	}
	if m.options.SandboxBroker != daytona.BrokerName {
		return nil, stack.Wrap(fmt.Errorf("session: there is no sandbox broker called %q", m.options.SandboxBroker))
	}
	return daytona.NewSecrets(daytona.SecretsOptions{Prefix: m.options.SandboxSecretPrefix, Session: spec.ID,
		Tokens: m.sandboxTokens(d.schemes), Logger: m.logger})
}

// sandboxTokens resolves a grant's credential and reads it as its scheme sends it: the token
// of the Authorization header Wrap puts on a request to the grant's first host. A connection
// that is not connected, or was deleted, gives sandbox.ErrNoCredential; a renewal that failed
// just now (resolver.ErrTemporarilyUnavailable) does not.
func (m *Manager) sandboxTokens(schemes map[string]core.Scheme) sandbox.Tokens {
	return func(ctx context.Context, grant sandbox.Grant) (sandbox.Token, error) {
		if m.options.Connectors.Resolver == nil {
			return sandbox.Token{}, stack.Wrap(fmt.Errorf("%w: connectors are off on this deployment", sandbox.ErrNoCredential))
		}
		credential, err := m.options.Connectors.Resolver.Resolve(ctx, grant.Connection, core.CredentialRequest{})
		if errors.Is(err, resolver.ErrNotConnected) || errors.Is(err, store.ErrNoConnectorConnection) {
			return sandbox.Token{}, fmt.Errorf("%w: %w", sandbox.ErrNoCredential, err)
		}
		if err != nil {
			return sandbox.Token{}, err
		}
		token, err := sandbox.BearerToken(ctx, schemes[grant.Binding], credential, grant.Hosts[0])
		if err != nil {
			return sandbox.Token{}, err
		}
		return sandbox.Token{Value: token, Revision: credential.Revision}, nil
	}
}
