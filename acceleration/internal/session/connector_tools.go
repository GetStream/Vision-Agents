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
	// unavailableNoSelection: a session binding the caller picked no connection for.
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
	// unavailableReauthorize: the provider no longer takes the connection's credential.
	unavailableReauthorize = "needs_reauthorization"
	// unavailableNotConnected: the connection has no credential yet, or was disconnected.
	unavailableNotConnected = "not_connected"
	// unavailableOpenFailed: the provider could not be reached or listed nothing, or this
	// deployment has connectors off.
	unavailableOpenFailed = "open_failed"
	// unavailableTool: the provider no longer offers a granted tool with the schema it was
	// granted against. The tools it still offers are kept.
	unavailableTool = "tool_unavailable"
	// unavailableDropped: a fork's or a reopened chat's selection for an alias its config no
	// longer declares as a session binding.
	unavailableDropped = "selection_dropped"
)

// unavailableWhy is each reason in words, for the error a required binding fails with.
var unavailableWhy = map[string]string{
	unavailableNoSelection:  "the session named no connection for it in connector_bindings",
	unavailableShared:       "more than one person writes in this conversation, so it uses the app's connections only",
	unavailableUnverified:   "it is the caller's own connection, and the caller is anonymous, a guest or a backend acting for nobody",
	unavailableConnection:   "there is no such connection that it may use",
	unavailableProvider:     "the connection is to another connector",
	unavailableReauthorize:  "the provider no longer takes the connection's credential; reconnect it",
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
		reason, err := m.openBinding(opening, *spec, binding, selected[binding.Name], d)
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
		m.offerLogin(*spec, binding, reason, selected[binding.Name], d)
	}
	return d, d.tools, unavailable, nil
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
		return unavailableTool, nil
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
// (api.pluginAliasComplaint); a caller's own tool still can.
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
