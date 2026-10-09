package session

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Connectors is the connector layer a session calls tools through: the registry's schemes
// and tool sources, and each connection's outbound client (core.Transports), which applies
// the resolver's credential, the scheme and the egress policy to every request.
type Connectors struct {
	Registry   core.Registry
	Transports *core.Transports
	// Consents begins a consent for the caller's own connection when a tool call needs one
	// (connector_login.go). Nil leaves such a binding out of the session, as it was before.
	Consents Consents
	// Limiter holds a connector's calls after its provider answered 429, until Retry-After
	// passes. Nil, as without Redis, limits nothing.
	Limiter *core.Limiter
}

// defaultConnectorTimeout bounds one connector tool call whose binding sets no timeout_ms.
// 5 s is the prototype's defaultConnectorToolTimeout (internal/session/connector_tools.go:19
// on codex/connector-support at cf62af0d), which gives no reason for it: unverified. A
// binding may set 1 to 30000 ms (api.AgentConnectorBinding.TimeoutMs).
const defaultConnectorTimeout = 5 * time.Second

// The two ways a binding picks its connection, as store.ConnectionBinding.Type holds them
// (api.AgentConnectorSelectionType).
const (
	selectionFixed   = "fixed"
	selectionSession = "session"
)

// Why a binding was left out of a session: ConnectorUnavailable.Reason, and the reason a
// required binding fails the session with. Stable codes for a program to branch on; none
// names a credential or says whose a connection is.
const (
	// unavailableNoSelection: a session binding the caller picked no connection for, and does
	// not have exactly one connected connection to the connector of (impliedSelection).
	unavailableNoSelection = "no_selection"
	// unavailableShared: a session binding in a conversation more than one verified person
	// writes in (Spec.Shared), which uses the app's connections only.
	unavailableShared = "shared_session"
	// unavailableUnverified: a session binding, and a caller whose name nobody vouched for.
	unavailableUnverified = "caller_unverified"
	// unavailableConnection: no live connection by that id that the binding may use. One code
	// for a missing connection and for another person's, so it says nothing of whose an id is.
	unavailableConnection = "connection_unavailable"
	// unavailableProvider: the connection is to another connector than the binding names.
	unavailableProvider = "provider_mismatch"
	// unavailableReauthorize: the provider no longer takes the connection's credential, or the
	// connection reads a definition revision a later one marked broken.
	unavailableReauthorize = "needs_reauthorization"
	// unavailableCredentialRejected: the provider no longer takes the token or key a
	// core.Static scheme holds (AI-990). Only new credentials help, never a consent.
	unavailableCredentialRejected = "credential_rejected"
	// unavailableNotConnected: the connection has no credential yet, or was disconnected.
	unavailableNotConnected = "not_connected"
	// unavailableOpenFailed: the provider could not be reached or listed nothing, or this
	// deployment has connectors off.
	unavailableOpenFailed = "open_failed"
	// unavailableTool: the provider no longer offers a granted tool with the schema it was
	// granted against, or for a grant by name, pinned at for the connection (pinGrants). The
	// tools it still offers are kept.
	unavailableTool = "tool_unavailable"
	// unavailableDropped: a fork's or a reopened chat's selection for an alias its config no
	// longer declares as a session binding.
	unavailableDropped = "selection_dropped"
)

// unavailableWhy is each reason in words, for the error a required binding fails with.
var unavailableWhy = map[string]string{
	unavailableNoSelection: "the session named no connection for it in connector_bindings",
	unavailableShared:      "more than one person writes in this conversation, so it uses the app's connections only",
	unavailableUnverified:  "it is the caller's own connection, and the caller is anonymous, a guest or a backend acting for nobody",
	unavailableConnection:  "there is no such connection that it may use",
	unavailableProvider:    "the connection is to another connector",
	unavailableReauthorize: "the provider no longer takes the connection's credential; reconnect it",
	unavailableCredentialRejected: "the provider rejected the connection's token or key; replace it with " +
		"PUT /v1/agents/connections/{id}/credentials",
	unavailableNotConnected: "the connection is not connected",
	unavailableOpenFailed:   "its tools could not be listed",
	unavailableTool:         "the provider no longer offers a granted tool with the schema it was granted against",
}

// attachConnectors opens the connector bindings the session may use and returns the
// dispatcher over their tools, the tools, and why each optional binding was left out. A
// required binding that cannot be used fails the session with why. The dispatcher is nil when
// the config binds nothing.
//
// A selection for an alias the config does not declare as a session binding fails the session,
// since the caller asked for something it cannot have. A fork, or a chat reopened from what
// it chose before (Spec.Reopened), drops it from spec instead, with an event, so the new row
// does not keep it: both re-resolve the earlier selections against the config as it is now
// and the principal asking. Example: a chat opened with crm, whose config dropped crm since,
// is answered without it.
func (m *Manager) attachConnectors(ctx context.Context, spec *Spec) (*dispatcher, []harness.Tool, []ConnectorUnavailable, error) {
	selected, unavailable, err := selections(*spec)
	if err != nil {
		return nil, nil, nil, err
	}
	// A new slice, since the one a fork holds is its parent's.
	var kept []ConnectorSelection
	for _, selection := range spec.ConnectorSelections {
		if _, found := selected[selection.Name]; found {
			kept = append(kept, selection)
		}
	}
	spec.ConnectorSelections = kept
	if len(spec.ConnectorBindings) == 0 {
		return nil, nil, unavailable, nil
	}
	// The dispatcher reads the config again before every call, so a binding is only ever
	// one a stored config still holds.
	if spec.ConfigID == "" {
		return nil, nil, nil, stack.Wrap(errors.New("session: connector bindings come from a stored agent config, and this session names none"))
	}
	d := &dispatcher{store: m.options.Store, spec: *spec, routes: map[string]route{}, invocations: m.invocations,
		limiter: m.options.Connectors.Limiter}
	// Opened on the context a call runs on (dispatcher.correlated): a refresh while the tools
	// are listed is audited as one during a call is, with no request or session id for an
	// incognito session. The MCP client keeps the values of the context it connected on for
	// what it sends outside a call, the standalone stream and the DELETE that Close sends
	// (connCtx, xcontext.Detach, in go-sdk v1.8.0 mcp/streamable.go:2074 and :2782), so this
	// covers those too.
	opening := d.correlated(ctx)
	for _, binding := range spec.ConnectorBindings {
		selection := selected[binding.Name]
		if selection == "" {
			selection, err = m.impliedSelection(opening, *spec, binding)
			if err != nil {
				d.Close()
				return nil, nil, nil, err
			}
		}
		reason, err := m.openBinding(opening, *spec, binding, selection, d)
		if err != nil {
			d.Close()
			return nil, nil, nil, err
		}
		if reason == "" {
			continue
		}
		if binding.Required {
			d.Close()
			return nil, nil, nil, stack.Wrap(fmt.Errorf("session: required connector %q cannot be used: %s: %s",
				binding.Name, reason, unavailableWhy[reason]))
		}
		m.logger.Warn("opening the session without a connector", "connector", binding.Name, "reason", reason)
		unavailable = append(unavailable, ConnectorUnavailable{Name: binding.Name, ConnectorID: binding.ConnectorID, Reason: reason})
		m.offerLogin(*spec, binding, reason, selection, d)
	}
	return d, d.tools, unavailable, nil
}

// impliedSelection is the connection a session binding the session named none for uses: the
// verified caller's one connected connection to the binding's connector, of the session's
// customer (Kanat's decision of 2026-10-09, AI-994), so an app that creates sessions without
// connector_bindings, as it did with user_plugins, needs no new login after the plugin rows
// move. Empty with none, or with more than one, which the caller has to choose between; a
// connection pending, needing reauthorization or disconnected is not counted, since only a
// connected one opens without a login. Not kept as the session's selection: a fork or a
// reopened chat implies again from the connections as they are then. Nothing is implied for
// a fixed binding, a shared conversation, an unverified caller, or with connectors off.
//
// Example: Alice's only connected Linear connection is the one the plugin migration moved, so
// a session of a config binding linear as session opens on it with no connector_bindings.
// Once she connects a second Linear account, a session has to name one.
func (m *Manager) impliedSelection(ctx context.Context, spec Spec, binding store.ConnectorBinding) (string, error) {
	if binding.Connection.Type != selectionSession || spec.Shared() || !verifiedCaller(spec) ||
		m.options.Store == nil || m.options.Connectors.Transports == nil {
		return "", nil
	}
	// Limit 1 lists at most two, enough to tell one from more than one.
	connected, err := m.options.Store.ConnectorConnectionsByOwner(ctx, spec.CustomerID, store.ConnectionFilter{
		OwnerType: store.OwnerUser, OwnerID: spec.Caller.UserID, ConnectorID: binding.ConnectorID,
		Status: store.ConnectionConnected, Limit: 1,
	})
	if err != nil || len(connected) != 1 {
		return "", err
	}
	m.logger.Debug("a session binding uses the caller's only connected connection", "connector", binding.Name,
		"connection", connected[0].ID)
	return connected[0].ID, nil
}

// selections are the caller's connections by alias. One for an alias the config does not
// declare as a session binding is refused, and on a fork or a reopened chat dropped with an
// event.
func selections(spec Spec) (map[string]string, []ConnectorUnavailable, error) {
	selected := make(map[string]string, len(spec.ConnectorSelections))
	var dropped []ConnectorUnavailable
	for _, selection := range spec.ConnectorSelections {
		if selection.Name == "" || selection.ConnectionID == "" {
			return nil, nil, stack.Wrap(errors.New("session: a connector binding needs a name and a connection_id"))
		}
		if _, twice := selected[selection.Name]; twice {
			return nil, nil, stack.Wrap(fmt.Errorf("session: connector %q is chosen twice", selection.Name))
		}
		index := slices.IndexFunc(spec.ConnectorBindings, func(b store.ConnectorBinding) bool { return b.Name == selection.Name })
		if index >= 0 && spec.ConnectorBindings[index].Connection.Type == selectionSession {
			selected[selection.Name] = selection.ConnectionID
			continue
		}
		switch {
		case spec.ForkedFrom != "" || !spec.Reopened.IsZero():
			connectorID := ""
			if index >= 0 {
				connectorID = spec.ConnectorBindings[index].ConnectorID
			}
			dropped = append(dropped, ConnectorUnavailable{Name: selection.Name, ConnectorID: connectorID, Reason: unavailableDropped})
		case index >= 0:
			return nil, nil, stack.Wrap(fmt.Errorf("session: connector %q has the agent config's fixed connection, which a session cannot replace", selection.Name))
		default:
			return nil, nil, stack.Wrap(fmt.Errorf("session: the agent config has no connector binding %q to choose a connection for", selection.Name))
		}
	}
	return selected, dropped, nil
}

// openBinding opens one binding's tools into d, or says why the binding cannot be used. An
// error is the store failing, not the binding.
func (m *Manager) openBinding(ctx context.Context, spec Spec, binding store.ConnectorBinding, selection string, d *dispatcher) (string, error) {
	id := binding.Connection.ConnectionID
	if binding.Connection.Type == selectionSession {
		switch {
		case spec.Shared():
			return unavailableShared, nil
		case !verifiedCaller(spec):
			return unavailableUnverified, nil
		case selection == "":
			return unavailableNoSelection, nil
		}
		id = selection
	}
	registry, transports := m.options.Connectors.Registry, m.options.Connectors.Transports
	if m.options.Store == nil || transports == nil {
		m.logger.Warn("a connector binding cannot be opened: connectors are off on this deployment", "connector", binding.Name)
		return unavailableOpenFailed, nil
	}
	connection, err := m.options.Store.ConnectorConnection(ctx, spec.CustomerID, id)
	if errors.Is(err, store.ErrNoConnectorConnection) {
		return unavailableConnection, nil
	}
	if err != nil {
		return "", err
	}
	if !mayUse(spec, binding, connection) {
		return unavailableConnection, nil
	}
	if connection.ConnectorID != binding.ConnectorID {
		return unavailableProvider, nil
	}
	switch connection.Status {
	case store.ConnectionConnected:
	case store.ConnectionNeedsReauthorization:
		if core.IsStatic(registry.Schemes, connection.AuthScheme) {
			return unavailableCredentialRejected, nil
		}
		return unavailableReauthorize, nil
	default:
		return unavailableNotConnected, nil
	}
	scheme, found := registry.Schemes[connection.AuthScheme]
	if !found {
		m.logger.Warn("a connector binding cannot be opened: no such scheme", "connector", binding.Name, "scheme", connection.AuthScheme)
		return unavailableOpenFailed, nil
	}
	definition, err := m.options.Store.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return "", err
	}
	// The resolver gives a connection on a revision marked broken no credential, so it is
	// one that needs a reconnect: a binding of the caller's own waits for their login, whose
	// consent runs on the latest revision.
	_, broken, err := m.options.Store.BrokenConnectorRevision(ctx, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return "", err
	}
	if broken {
		return unavailableReauthorize, nil
	}
	manifest, err := definition.Manifest.Resolve(connection.AuthScheme, connection.Inputs, connection.Metadata)
	if err != nil {
		m.logger.Warn("a connector binding cannot be opened", "connector", binding.Name, "error", err)
		return unavailableOpenFailed, nil
	}
	ref := core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}
	resolved := core.ResolvedBinding{
		// No Timeout: the dispatcher bounds every call itself, the one envelope for every
		// source.
		Binding:    core.Binding{Name: binding.Name, ConnectorID: binding.ConnectorID, Selection: binding.Connection.Type, ConnectionID: connection.ID, Required: binding.Required},
		Connection: coreConnection(connection),
		Manifest:   manifest,
		HTTP:       transports.Client(ref, scheme),
	}
	grants := make([]core.ToolGrant, 0, len(binding.Tools))
	digests := make(map[string]string, len(binding.Tools))
	for _, grant := range binding.Tools {
		grants = append(grants, core.ToolGrant{Name: grant.Name, SchemaDigest: grant.SchemaDigest})
		digests[grant.Name] = grant.SchemaDigest
	}
	if binding.Connection.Type == selectionSession {
		reason, err := m.pinGrants(ctx, binding.Name, connection, resolved, grants)
		if reason != "" || err != nil {
			return reason, err
		}
	}
	timeout := defaultConnectorTimeout
	if binding.TimeoutMs > 0 {
		timeout = time.Duration(binding.TimeoutMs) * time.Millisecond
	}
	offered := 0
	for _, kind := range sourceKinds(manifest) {
		source, found := registry.ToolSources[kind]
		if !found {
			m.logger.Warn("a connector binding cannot be opened: no such tool source", "connector", binding.Name, "source", kind)
			return unavailableOpenFailed, nil
		}
		toolset, err := source.Open(ctx, resolved, grants)
		if err != nil {
			m.logger.Warn("a connector binding cannot be opened", "connector", binding.Name, "error", err)
			return unavailableOpenFailed, nil
		}
		d.toolsets = append(d.toolsets, toolset)
		for _, tool := range toolset.Tools() {
			name, found := strings.CutPrefix(tool.Name, binding.Name+mcp.Separator)
			if _, taken := d.routes[tool.Name]; taken || !found {
				continue
			}
			d.routes[tool.Name] = route{binding: binding, connection: connection, toolset: toolset,
				tool: name, digest: digests[name], timeout: timeout,
				limit: manifest.RateLimitKey(connection.CustomerID, resolved.Connection)}
			d.tools = append(d.tools, harness.Tool{Name: tool.Name, Description: tool.Description, Parameters: tool.Parameters})
			offered++
		}
	}
	if offered < len(binding.Tools) {
		var changed []string
		for _, grant := range binding.Tools {
			if _, found := d.routes[binding.Name+mcp.Separator+grant.Name]; !found && grant.SchemaDigest == "" {
				changed = append(changed, grant.Name)
			}
		}
		if len(changed) > 0 {
			m.logger.Warn("a tool granted by name is not offered: the connection no longer lists it with the "+
				"schema it was pinned at, and a reconnect approves it again", "connector", binding.Name,
				"connection", connection.ID, "tools", changed)
		}
		return unavailableTool, nil
	}
	return "", nil
}

// pinGrants gives each of a session binding's grants that names its tool alone the digest the
// connection's tool is pinned at for its current grant (store.ConnectorToolPin). A tool with no
// pin yet is pinned first, at the digest the provider lists for it now: trust on first use, so
// a tool whose schema changes after is not offered until the connection is connected again. A
// tool the provider does not list is left with no digest, which Open does not offer. It says
// why the binding cannot be used when the provider cannot list its tools; an error is the store
// failing.
//
// Example: Slack names the signed-in user in slack_send_message's description, so Alice's and
// Bob's digests differ and no one digest in the config fits both. A grant of the name alone
// pins Alice's on her connection and Bob's on his.
func (m *Manager) pinGrants(ctx context.Context, alias string, connection store.ConnectorConnection, resolved core.ResolvedBinding, grants []core.ToolGrant) (string, error) {
	var named []int
	for i, grant := range grants {
		if grant.SchemaDigest == "" {
			named = append(named, i)
		}
	}
	if len(named) == 0 {
		return "", nil
	}
	pins, err := m.options.Store.ConnectorToolPins(ctx, connection.ID, connection.ConnectedAt)
	if err != nil {
		return "", err
	}
	if slices.ContainsFunc(named, func(i int) bool { _, pinned := pins[grants[i].Name]; return !pinned }) {
		listed := map[string]string{}
		for _, kind := range sourceKinds(resolved.Manifest) {
			source, found := m.options.Connectors.Registry.ToolSources[kind]
			if !found {
				m.logger.Warn("a connector binding cannot be opened: no such tool source", "connector", alias, "source", kind)
				return unavailableOpenFailed, nil
			}
			specs, err := source.Discover(ctx, resolved)
			if err != nil {
				m.logger.Warn("a connector binding cannot be opened", "connector", alias, "error", err)
				return unavailableOpenFailed, nil
			}
			for _, spec := range specs {
				listed[spec.Name] = spec.SchemaDigest
			}
		}
		first := map[string]string{}
		for _, i := range named {
			name := grants[i].Name
			if _, pinned := pins[name]; pinned {
				continue
			}
			if digest, found := listed[name]; found {
				first[name] = digest
			}
		}
		pins, err = m.options.Store.PinConnectorTools(ctx, connection.ID, connection.ConnectedAt, first)
		if err != nil {
			return "", err
		}
	}
	for _, i := range named {
		grants[i].SchemaDigest = pins[grants[i].Name]
	}
	return "", nil
}

// mayUse is the owner rule. A fixed binding uses the app's own connection it names. A session
// binding uses the verified caller's own connection, never an anonymous caller's or a
// guest's: the prototype's rule (internal/session/connector_tools.go:119-127 on
// codex/connector-support at cf62af0d).
func mayUse(spec Spec, binding store.ConnectorBinding, connection store.ConnectorConnection) bool {
	switch binding.Connection.Type {
	case selectionFixed:
		return connection.OwnerType == store.OwnerApp && connection.ID == binding.Connection.ConnectionID
	case selectionSession:
		return connection.OwnerType == store.OwnerUser && verifiedCaller(spec) && connection.OwnerID == spec.Caller.UserID
	}
	return false
}

// verifiedCaller reports whether the caller's name was vouched for: an end user whose token
// was verified, or the app's backend naming the user it acts for, as the connection
// endpoints take it (api.mayReach). A guest is refused, though its token was verified: its
// account is a temporary one the app does not know.
func verifiedCaller(spec Spec) bool {
	return spec.Caller.UserID != "" && (spec.CallerKind == auth.KindAuthenticated || spec.CallerKind == auth.KindServer)
}

// connectorCollision is a connector tool called what another of the session's tools is.
// A config cannot name a plugin or an MCP server what one of its bindings is
// (api.pluginAliasComplaint), unless the binding is to that plugin's own connector, whose
// entry the spec drops (Spec.withoutBoundPlugins); a caller's own tool still can.
func connectorCollision(connectors, others []harness.Tool) error {
	for _, tool := range connectors {
		if slices.ContainsFunc(others, func(other harness.Tool) bool { return other.Name == tool.Name }) {
			return stack.Wrap(fmt.Errorf("session: the tool %q is both a connector's and another of the session's", tool.Name))
		}
	}
	return nil
}

// sourceKinds are the kinds of tool source a manifest lists, each once, in its order.
func sourceKinds(m core.ResolvedManifest) []string {
	var kinds []string
	for _, rule := range m.Sources {
		if !slices.Contains(kinds, rule.Kind) {
			kinds = append(kinds, rule.Kind)
		}
	}
	return kinds
}

// coreConnection is a stored connection as a tool source reads it.
func coreConnection(connection store.ConnectorConnection) core.Connection {
	return core.Connection{
		ID:                 connection.ID,
		ConnectorID:        connection.ConnectorID,
		DefinitionRevision: connection.DefinitionRevision,
		OwnerType:          connection.OwnerType,
		OwnerID:            connection.OwnerID,
		AccountID:          connection.AccountID,
		Inputs:             maps.Clone(connection.Inputs),
		Metadata:           maps.Clone(connection.Metadata),
		Status:             connection.Status,
	}
}
