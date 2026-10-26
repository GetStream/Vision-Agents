package session

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
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
		Binding:    binding.Name, Config: d.spec.ConfigID, Connector: binding.ConnectorID, Selection: binding.Connection.Type,
		Scheme: connection.AuthScheme, Env: manifest.Sandbox.Env, Hosts: manifest.Sandbox.Hosts, Required: binding.Required,
	})
	if d.schemes == nil {
		d.schemes = map[string]core.Scheme{}
	}
	d.schemes[binding.Name] = scheme
	return ""
}

// lateGrant gives the sandbox an env var for a session binding with sandbox set that waits for
// its person to log in (offerLogin): a late grant, read from the sandbox block of the
// connector's latest revision, which a consent runs on. Once the login opens the binding, the
// broker binds the grant before the next run (boundLate). A connector with no sandbox block, or
// one deleted while the config still binds it, gives the sandbox nothing.
func (m *Manager) lateGrant(ctx context.Context, spec Spec, binding store.ConnectorBinding, d *dispatcher) error {
	if !binding.Sandbox {
		return nil
	}
	definition, err := m.options.Store.LatestConnectorDefinition(ctx, spec.CustomerID, binding.ConnectorID)
	if err != nil && !errors.Is(err, store.ErrNoConnectorDefinition) {
		return err
	}
	if err != nil || definition.Manifest.Sandbox == nil {
		m.logger.Warn("a connector binding gives the sandbox no access: its connector offers none", "connector", binding.Name)
		return nil
	}
	rule := definition.Manifest.Sandbox
	d.grants = append(d.grants, sandbox.Grant{Binding: binding.Name, Config: spec.ConfigID, Connector: binding.ConnectorID,
		Selection: binding.Connection.Type, Env: rule.Env, Hosts: rule.Hosts, Late: true})
	return nil
}

// boundLate is a late grant's connection once its login has opened the binding: the grant the
// binding opened with there. A late grant is made only beside its login (lateGrant). Only this
// session's own login is read, which opens on a consent this session began, for a connection
// its caller owns (loginFinished, mayUse). A connection whose own revision names another env
// var or other hosts than the late grant's stays unbound, as its revision decides what the
// sandbox may reach (sandboxGrant): it needs a reconnect on the latest revision.
func (d *dispatcher) boundLate(_ context.Context, grant sandbox.Grant) (sandbox.Grant, bool) {
	l := d.logins.byAlias[grant.Binding]
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.opened == nil || len(l.opened.grants) == 0 {
		return sandbox.Grant{}, false
	}
	opened := l.opened.grants[0]
	if opened.Env != grant.Env || !slices.Equal(opened.Hosts, grant.Hosts) {
		d.logger.Warn("a connector binding gives the sandbox no access: reconnect to use sandbox access", "connector", grant.Binding,
			"connection", opened.Connection.ConnectionID)
		return sandbox.Grant{}, false
	}
	return opened, true
}

// connecting is the session's sandbox while one of its grants is late: each run's result names
// the late grants whose user has not connected them (Result.Unconnected), as the model reads
// a tool's result more surely than its description, and, once, those whose token is still on
// its way (Result.Connected).
type connecting struct {
	sandbox.Sandbox
	d *dispatcher
	// delayed is a broker whose bound token reaches code some time after the run began: a
	// Daytona secret's update took 18 to 24 seconds (probe against Daytona's API, 2026-10-25),
	// where the egress proxy reads it on every request.
	delayed bool
	mu      sync.Mutex
	// opened are the late grants' bindings a run found connected, which no result names again.
	opened map[string]bool
}

func (c *connecting) Grants() []sandbox.Grant { return c.d.grants }

// Run reads the logins after the run, whose BeforeRun bound each grant connected by then: one
// connected since is named as connected, not as unconnected.
func (c *connecting) Run(ctx context.Context, code string, outputs []string) (sandbox.Result, error) {
	result, err := c.Sandbox.Run(ctx, code, outputs)
	if err != nil {
		return result, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, grant := range c.d.grants {
		if !grant.Late || c.opened[grant.Binding] {
			continue
		}
		if !c.d.logins.byAlias[grant.Binding].loggedIn() {
			result.Unconnected = append(result.Unconnected, grant.Binding)
			continue
		}
		c.opened[grant.Binding] = true
		if !c.delayed {
			continue
		}
		if _, bound := c.d.boundLate(ctx, grant); bound {
			result.Connected = append(result.Connected, grant.Binding)
		}
	}
	return result, nil
}

// loggedIn is whether the login has opened the binding. A connection whose revision gives the
// sandbox nothing, or another env var or hosts, is logged in and still unbound (boundLate): a
// login cannot bind it, a reconnect on the latest revision can.
func (l *login) loggedIn() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.opened != nil
}

// broker is the code_sandbox.broker setting's broker for the session's grants.
func (m *Manager) broker(spec Spec, d *dispatcher) (sandbox.CredentialBroker, error) {
	if m.options.SandboxBroker == egress.BrokerName {
		if err := behindProxy(spec.SandboxOptions); err != nil {
			return nil, err
		}
		options := m.options.SandboxProxy
		options.Session, options.Allowed = spec.ID, spec.SandboxOptions.AllowedDomains
		options.Tokens, options.Bind, options.Logger = m.sandboxTokens(d.schemes), d.boundLate, m.logger
		return egress.NewBroker(options)
	}
	if m.options.SandboxSecretPrefix == "" {
		return nil, stack.Wrap(errors.New("session: a connector binding sets sandbox, and this router has no " +
			"code_sandbox.secret_prefix (ROUTER_CODE_SANDBOX_SECRET_PREFIX) to name its secrets with"))
	}
	if m.options.SandboxBroker != metered.SandboxSecretsBroker {
		return nil, stack.Wrap(fmt.Errorf("session: there is no sandbox broker called %q", m.options.SandboxBroker))
	}
	return metered.SandboxSecrets(m.options.SandboxSecretPrefix, spec.ID, m.sandboxTokens(d.schemes), d.boundLate, m.logger)
}

// behindProxy refuses a sandbox config the egress proxy cannot serve. Every request goes there,
// and the proxy reaches the grants' hosts and the config's allowed domains by name only, so an
// open network or CIDR ranges would reach nothing but the grants.
func behindProxy(config sandbox.Config) error {
	if config.Network && len(config.AllowedDomains) == 0 && len(config.NetworkAllowList) == 0 {
		return stack.Wrap(errors.New("session: the agent config's sandbox_options.network is true, and this router's " +
			"code_sandbox.broker is proxy, which sends every request from the sandbox through the egress proxy, and the " +
			"proxy reaches only connectors' hosts and sandbox_options.allowed_domains: list the domains code needs in sandbox_options.allowed_domains instead"))
	}
	if len(config.NetworkAllowList) > 0 {
		return stack.Wrap(errors.New("session: the agent config's sandbox_options.network_allow_list is set, and this " +
			"router's code_sandbox.broker is proxy, which reaches hosts by name only: list the domains code needs in " +
			"sandbox_options.allowed_domains instead"))
	}
	return nil
}

// sandboxTokens resolves a grant's credential and reads it as its scheme sends it: the token
// of the Authorization header Wrap puts on a request to the grant's first host. A binding the
// agent config dropped (SandboxBound), or a connection that is not connected, or was deleted,
// gives sandbox.ErrNoCredential; a renewal that failed just now
// (resolver.ErrTemporarilyUnavailable) does not. A late grant's connection was opened after
// the session, by its login, so its scheme is read from the registry by name.
func (m *Manager) sandboxTokens(schemes map[string]core.Scheme) sandbox.Tokens {
	return func(ctx context.Context, grant sandbox.Grant) (sandbox.Token, error) {
		if m.options.Connectors.Resolver == nil {
			return sandbox.Token{}, stack.Wrap(fmt.Errorf("%w: connectors are off on this deployment", sandbox.ErrNoCredential))
		}
		if err := SandboxBound(ctx, m.options.Store, grant); err != nil {
			return sandbox.Token{}, err
		}
		credential, err := m.options.Connectors.Resolver.Resolve(ctx, grant.Connection, core.CredentialRequest{})
		if errors.Is(err, resolver.ErrNotConnected) || errors.Is(err, store.ErrNoConnectorConnection) {
			return sandbox.Token{}, fmt.Errorf("%w: %w", sandbox.ErrNoCredential, err)
		}
		if err != nil {
			return sandbox.Token{}, err
		}
		scheme, found := schemes[grant.Binding]
		if !found {
			scheme = m.options.Connectors.Registry.Schemes[grant.Scheme]
		}
		token, err := sandbox.BearerToken(ctx, scheme, credential, grant.Hosts[0])
		if err != nil {
			return sandbox.Token{}, err
		}
		return sandbox.Token{Value: token, Revision: credential.Revision}, nil
	}
}

// SandboxBound is nil while grant's agent config still binds grant.Binding as the session
// opened it: with sandbox set, to the same connector, picking its connection the same way, and
// the same connection when that is fixed. Once it does not, or the config is gone, it wraps
// sandbox.ErrNoCredential, so the broker withdraws the credential before the next run and the
// egress proxy sends it with no further request, as recheck refuses a tool call. A config that
// cannot be read just now is any other error. One read by primary key.
func SandboxBound(ctx context.Context, configs *store.Store, grant sandbox.Grant) error {
	config, err := configs.AgentConfig(ctx, grant.Connection.CustomerID, grant.Config)
	if errors.Is(err, store.ErrNoAgentConfig) {
		return fmt.Errorf("%w: %w", sandbox.ErrNoCredential, err)
	}
	if err != nil {
		return err
	}
	index := slices.IndexFunc(config.Connectors, func(b store.ConnectorBinding) bool { return b.Name == grant.Binding })
	if index < 0 {
		return stack.Wrap(fmt.Errorf("%w: the agent config no longer binds %q", sandbox.ErrNoCredential, grant.Binding))
	}
	current := config.Connectors[index]
	if !current.Sandbox || current.ConnectorID != grant.Connector || current.Connection.Type != grant.Selection ||
		(current.Connection.Type == selectionFixed && current.Connection.ConnectionID != grant.Connection.ConnectionID) {
		return stack.Wrap(fmt.Errorf("%w: the agent config binds %q differently now", sandbox.ErrNoCredential, grant.Binding))
	}
	return nil
}

// SandboxProxyTokens are the egress proxy's tokens: on every request it checks that the agent
// config still binds the grant (SandboxBound) and asks connections for the credential.
func SandboxProxyTokens(connections core.Resolver, configs *store.Store, schemes map[string]core.Scheme) egress.Tokens {
	return egress.ResolverTokens(connections, schemes, func(ctx context.Context, grant sandbox.Grant) error {
		return SandboxBound(ctx, configs, grant)
	})
}
