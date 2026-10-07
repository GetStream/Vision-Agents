package session

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
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
			return nil, errUnknownTool(call.Name)
		}
		if d.next != nil {
			return d.next.Run(ctx, call)
		}
		return nil, errUnknownTool(call.Name)
	}
	if err := d.recheck(ctx, found); err != nil {
		return nil, err
	}
	return d.call(ctx, found, call)
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
// The binding's timeout bounds it, and so does the connection's client, which bounds each
// request it sends (core.TransportsConfig.Timeout, connectorHTTPTimeout in the router, 10 s
// against a timeout_ms of up to 30000). When either runs out the request may have reached
// the provider and done its work, so the model reads outcome_unknown rather than an error it
// would retry: the architecture doc's SourceContract («a timed-out write returns
// outcome_unknown, not an error the model retries», connectors/planning). The client's bound
// is not raised to the binding's instead: the transport is one per connection, shared with
// the validate endpoint and the channel bridge, whose bound would change with it. A turn that is
// interrupted cancels ctx, and the MCP SDK sends notifications/cancelled for the call in
// flight (cancelCall in go-sdk v1.8.0 mcp/transport.go).
func (d *dispatcher) call(ctx context.Context, r route, call llm.ToolCall) ([]llm.ContentPart, error) {
	bounded, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	result, err := r.toolset.Call(bounded, call)
	if err != nil {
		if ctx.Err() == nil && (errors.Is(bounded.Err(), context.DeadlineExceeded) || timedOut(err)) {
			return llm.TextParts(outcomeUnknown(call.Name)), nil
		}
		return nil, err
	}
	// A source cuts its own results; the cap holds whichever source answered.
	if text := llm.TextOf(result.Parts); len(text) > core.MaxResultBytes {
		return llm.TextParts(core.CutResult(text)), nil
	}
	return result.Parts, nil
}

// outcomeUnknown is what the model reads of a call a timeout cut off.
func outcomeUnknown(name string) string {
	return fmt.Sprintf("outcome_unknown: %s did not answer in time. It may or may not have done "+
		"what was asked; check before calling it again.", name)
}

// timedOut reports whether err is a request that ran out of time: net/http's client timeout
// is an error whose Timeout method says so (url.Error, net.Error).
func timedOut(err error) bool {
	var timeout interface{ Timeout() bool }
	return errors.As(err, &timeout) && timeout.Timeout()
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
