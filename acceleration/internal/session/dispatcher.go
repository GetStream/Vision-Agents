package session

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// dispatcher runs a session's connector tools: the agent.ToolRunner in front of the rest of
// the session's chain. Its authority is the map from the name the model is offered to the
// binding, connection, toolset and tool it was opened for, never the name itself: a name it
// did not open under a bound alias is refused, and anything else goes to next.
//
// Every call is checked again against the config and the connection as they are now, then
// run inside one envelope for every source: the binding's timeout, a cancel when the turn is
// interrupted, and the result cap.
//
// Every call it opened, refused or run, leaves one row in the invocation log once it has
// answered (store.ConnectorInvocation), queued so the model never waits on the write.
type dispatcher struct {
	store *store.Store
	// spec is what the session was opened with: whose it is, its config and its selections.
	spec   Spec
	routes map[string]route
	tools  []harness.Tool
	// toolsets are every toolset opened, closed with the session.
	toolsets  []core.Toolset
	closeOnce sync.Once
	next      agent.ToolRunner
	// logins are the session bindings waiting for their person to log in (connector_login.go).
	logins *logins
	// invocations writes the log; nil records nothing.
	invocations *invocationRecorder
	// stepUps asks the caller for more access when a provider wants it (step_up.go); nil
	// asks nobody.
	stepUps *stepUps
}

// route is what one offered name was opened for.
type route struct {
	// binding and connection are as they were when the session opened.
	binding    store.ConnectorBinding
	connection store.ConnectorConnection
	toolset    core.Toolset
	// tool is the tool's name at the provider, and digest the schema it was granted against.
	tool    string
	digest  string
	timeout time.Duration
}

// Run calls a connector tool, or hands a name it does not own to next.
func (d *dispatcher) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	found, ok := d.routes[call.Name]
	if !ok {
		// A name under a bound alias that was not opened is a tool the grant does not
		// allow, or one the provider stopped offering. It goes nowhere else.
		if alias, _, cut := strings.Cut(call.Name, mcp.Separator); cut && d.spec.boundAlias(alias) {
			// A binding waiting for its person to log in answers with the login.
			if waiting, ok := d.logins.waiting(alias); ok {
				return d.runLogin(ctx, waiting, call)
			}
			return nil, errUnknownTool(call.Name)
		}
		if d.next != nil {
			return d.next.Run(ctx, call)
		}
		return nil, errUnknownTool(call.Name)
	}
	started := time.Now()
	if err := d.recheck(ctx, found); err != nil {
		d.record(found, started, store.InvocationDenied)
		return nil, err
	}
	parts, failure, err := d.call(ctx, found, call)
	d.record(found, started, failure)
	return parts, err
}

// record queues the row of one call that started at started and ended now, as failure says.
// The row names the binding, the connection and the tool, never what the call was asked or
// answered, and an incognito session's names no session.
func (d *dispatcher) record(r route, started time.Time, failure string) {
	sessionID := d.spec.ID
	if d.spec.Incognito {
		sessionID = ""
	}
	d.invocations.Record(store.ConnectorInvocation{
		CustomerID: r.connection.CustomerID, ConnectionID: r.connection.ID, ConnectorID: r.binding.ConnectorID,
		ConfigID: d.spec.ConfigID, Binding: r.binding.Name, Tool: r.tool, SessionID: sessionID,
		StartedAt: started, LatencyMs: time.Since(started).Milliseconds(), ErrorType: failure,
	})
}

// correlated is ctx naming this session for the audit rows its calls cause, such as a
// refresh (core.Correlation), and nothing at all for an incognito session. Not even the
// request id: a session's turns run on the context of the request that created it
// (Manager.Create), so that id is the same for the whole session and names it in the access
// log as well as a session id would.
func (d *dispatcher) correlated(ctx context.Context) context.Context {
	if d.spec.Incognito {
		return core.WithCorrelation(ctx, core.Correlation{})
	}
	correlation := core.CorrelationOf(ctx)
	correlation.SessionID = d.spec.ID
	return core.WithCorrelation(ctx, correlation)
}

// Close ends every toolset. A second call does nothing.
func (d *dispatcher) Close() {
	d.closeOnce.Do(func() {
		for _, toolset := range d.toolsets {
			toolset.Close()
		}
	})
}

// call runs one tool inside the envelope.
//
// The binding's timeout bounds it, and it is always the first bound to end a call: the mcp
// source sends with a copy of the connection's client whose timeout is at least 35 s
// (mcp.defaultCallTimeout), past the longest timeout_ms of 30000, where the connection's own
// client has 10 s (connectorHTTPTimeout in the router). When the deadline runs out the request may have reached the provider and done its
// work, so the model reads outcome_unknown rather than an error it would retry: the
// architecture doc's SourceContract («a timed-out write returns outcome_unknown, not an error
// the model retries», connectors/planning). A turn that is
// interrupted cancels ctx, and the MCP SDK sends notifications/cancelled for the call in
// flight (cancelCall in go-sdk v1.8.0 mcp/transport.go).
//
// It also says how the call failed, for its row: empty when it answered, else one of the
// store.Invocation* values.
func (d *dispatcher) call(ctx context.Context, r route, call llm.ToolCall) ([]llm.ContentPart, string, error) {
	bounded, cancel := context.WithTimeout(d.correlated(ctx), r.timeout)
	defer cancel()
	observed, exchange := core.WithExchange(bounded)
	result, err := r.toolset.Call(observed, call)
	if err != nil {
		if ctx.Err() == nil && errors.Is(bounded.Err(), context.DeadlineExceeded) {
			return llm.TextParts(outcomeUnknown(call.Name)), store.InvocationOutcomeUnknown, nil
		}
		if asked, ok := exchange.ScopeRequired(); ok {
			if text, asking := d.stepUp(ctx, r, call.TurnID, asked); asking {
				return llm.TextParts(text), failed(ctx, exchange), nil
			}
		}
		return nil, failed(ctx, exchange), err
	}
	// A source cuts its own results; the cap holds whichever source answered.
	if text := llm.TextOf(result.Parts); len(text) > core.MaxResultBytes {
		return llm.TextParts(core.CutResult(text)), "", nil
	}
	return result.Parts, "", nil
}

// failed is how a call that failed before the binding's deadline failed, from what its
// requests saw (core.Exchange): a tool source's error does not say whether anything was sent,
// and the MCP client reports a 401 only as the words of its status.
//
//	the connection has no credential to give (resolver.ErrNotConnected), or
//	  the provider answered 401 or 403                      customer_auth
//	the provider could not renew the credential just now    external_server
//	the turn ended (an interruption) after a request left    outcome_unknown
//	  before any left                                        client_timeout
//	the connection's client stopped waiting for a request    client_timeout
//	nothing was sent: the source refused the arguments, or
//	  the connection went between the check and the call    denied
//	anything else the provider or the way to it answered    external_server
//
// Example: an MCP tool that answers isError, or a 502, is external_server; a connection whose
// refresh token the provider revoked is customer_auth.
func failed(turn context.Context, seen *core.Exchange) string {
	credential, status := seen.CredentialError(), seen.Status()
	switch {
	case errors.Is(credential, resolver.ErrNotConnected) || status == http.StatusUnauthorized || status == http.StatusForbidden:
		return store.InvocationCustomerAuth
	case errors.Is(credential, resolver.ErrTemporarilyUnavailable):
		return store.InvocationExternalServer
	case turn.Err() != nil && seen.Sent():
		return store.InvocationOutcomeUnknown
	case turn.Err() != nil, seen.TimedOut():
		return store.InvocationClientTimeout
	case !seen.Sent():
		return store.InvocationDenied
	default:
		return store.InvocationExternalServer
	}
}

// outcomeUnknown is what the model reads of a call a timeout cut off.
func outcomeUnknown(name string) string {
	return fmt.Sprintf("outcome_unknown: %s did not answer in time. It may or may not have done "+
		"what was asked; check before calling it again.", name)
}

// recheck refuses a call that the session's config and connection no longer allow, before
// anything is sent: the prototype's authorizeConnectorTool
// (internal/session/connector_tools.go:229-288 on codex/connector-support at cf62af0d). The
// config must still hold the binding, with the same connector and connection, and grant the
// tool at the digest it was opened with. The connection must still be live, connected, one
// the binding may use, and reach where it reached when the session opened: the same
// definition revision, inputs and captured metadata, which are what its endpoints are built
// from. Two reads by primary key per call.
func (d *dispatcher) recheck(ctx context.Context, r route) error {
	refuse := func(why string) error {
		return stack.Wrap(fmt.Errorf("session: connector %q: %s, so %s was not called",
			r.binding.Name, why, r.binding.Name+mcp.Separator+r.tool))
	}
	config, err := d.store.AgentConfig(ctx, d.spec.CustomerID, d.spec.ConfigID)
	if err != nil {
		return refuse("the agent config could not be read")
	}
	index := slices.IndexFunc(config.Connectors, func(b store.ConnectorBinding) bool { return b.Name == r.binding.Name })
	if index < 0 {
		return refuse("the agent config no longer binds it")
	}
	current := config.Connectors[index]
	switch {
	case current.ConnectorID != r.binding.ConnectorID || current.Connection.Type != r.binding.Connection.Type:
		return refuse("the agent config binds it differently now")
	case !slices.Contains(current.Tools, store.ToolGrant{Name: r.tool, SchemaDigest: r.digest}):
		return refuse("the tool is no longer granted")
	}
	connection, err := d.store.ConnectorConnection(ctx, d.spec.CustomerID, r.connection.ID)
	switch {
	case errors.Is(err, store.ErrNoConnectorConnection):
		return refuse("the connection is gone")
	case err != nil:
		return refuse("the connection could not be read")
	case !mayUse(d.spec, current, connection) || connection.ConnectorID != current.ConnectorID:
		return refuse("the binding may no longer use the connection")
	case connection.Status != store.ConnectionConnected:
		return refuse("the connection is " + connection.Status)
	case connection.DefinitionRevision != r.connection.DefinitionRevision ||
		!maps.Equal(connection.Inputs, r.connection.Inputs) || !maps.Equal(connection.Metadata, r.connection.Metadata):
		return refuse("the connection changed where it reaches since the session opened")
	}
	return nil
}

// boundAlias reports whether a connector binding of the spec is called alias.
func (s Spec) boundAlias(alias string) bool {
	return slices.ContainsFunc(s.ConnectorBindings, func(b store.ConnectorBinding) bool { return b.Name == alias })
}
