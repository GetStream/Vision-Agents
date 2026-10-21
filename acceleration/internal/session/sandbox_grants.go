package session

import (
	"context"
	"errors"
	"fmt"
	"net/netip"
	"net/url"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/egress"
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
		Binding:    binding.Name, Scheme: connection.AuthScheme, Env: manifest.Sandbox.Env, Hosts: manifest.Sandbox.Hosts, Required: binding.Required,
	})
	if d.schemes == nil {
		d.schemes = map[string]core.Scheme{}
	}
	d.schemes[binding.Name] = scheme
	return ""
}

// broker is the code_sandbox.broker setting's broker for the session's grants.
func (m *Manager) broker(spec Spec, d *dispatcher) (sandbox.CredentialBroker, error) {
	if m.options.SandboxBroker == egress.BrokerName {
		if err := behindProxy(spec.SandboxOptions, m.options.SandboxProxy.URL); err != nil {
			return nil, err
		}
		options := m.options.SandboxProxy
		options.Session, options.Tokens, options.Logger = spec.ID, m.sandboxTokens(d.schemes), m.logger
		return egress.NewBroker(options)
	}
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

// behindProxy refuses a sandbox config the egress proxy cannot serve. HTTPS_PROXY sends every
// request there and the proxy refuses a host no grant names, so an open network would reach
// nothing but the grants. Daytona takes one allow list, so with CIDR ranges the proxy must be an
// address in them.
func behindProxy(config sandbox.Config, proxy string) error {
	if config.Network && len(config.AllowedDomains) == 0 && len(config.NetworkAllowList) == 0 {
		return stack.Wrap(errors.New("session: the agent config's sandbox_options.network is true, and this router's " +
			"code_sandbox.broker is proxy, which sends every request from the sandbox through the egress proxy, and the " +
			"proxy reaches only connectors' hosts: list the domains code needs in sandbox_options.allowed_domains instead"))
	}
	parsed, err := url.Parse(proxy)
	if len(config.NetworkAllowList) > 0 && (err != nil || !isAddress(parsed.Hostname())) {
		return stack.Wrap(errors.New("session: the agent config's sandbox_options.network_allow_list must hold the " +
			"egress proxy's address, and this router's code_sandbox.proxy_public_url names it by a host name: list " +
			"sandbox_options.allowed_domains instead, or give the proxy's IP address"))
	}
	return nil
}

// isAddress says whether host is an IP address, which a CIDR range can hold.
func isAddress(host string) bool {
	_, err := netip.ParseAddr(host)
	return err == nil
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
