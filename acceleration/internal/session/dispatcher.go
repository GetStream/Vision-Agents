package session

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
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
// interrupted (as the binding's policy says), and the result cap.
//
// Every call it opened, refused or run, leaves one row in the invocation log once it has
// answered (store.ConnectorInvocation), queued so the model never waits on the write.
type dispatcher struct {
	store *store.Store
	// spec is what the session was opened with: whose it is, its config and its selections.
	spec   Spec
	routes map[string]route
	tools  []harness.Tool
	// leftOut are the offered names of granted tools a binding left out, whose schema the
	// provider changed since they were approved (unavailableTool).
	leftOut []string
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
	// limiter holds calls after a provider's 429 (Connectors.Limiter); nil limits nothing.
	limiter *core.Limiter
	// slots caps each connection's calls in flight (Manager.slots); nil caps nothing.
	slots *callSlots
	// logger is the session's (Manager.logger): a call that failed or was refused says so on
	// it (record).
	logger *slog.Logger
	// grants are the sandbox bindings opened, for the session's sandbox (sandbox_grants.go),
	// and schemes their connections' schemes by binding name.
	grants  []sandbox.Grant
	schemes map[string]core.Scheme
	// approvals are the calls held for their person to allow them (connector_approval.go).
	approvals approvals
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
	// limit is the key the provider's rate limit counts the call under
	// (core.ResolvedManifest.RateLimitKey), "" when its manifest names none.
	limit string
	// maxInFlight is how many of the connection's calls may be in flight at once
	// (rate_limit.max_concurrent_calls), 0 for no cap.
	maxInFlight int
	// declared are the argument names the tool's input schema lists under properties: the only
	// names a call's row keeps (argumentShape).
	declared map[string]bool
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
			if slices.Contains(d.leftOut, call.Name) {
				return llm.TextParts(toolLeftOut(alias, strings.TrimPrefix(call.Name, alias+mcp.Separator))), nil
			}
			return nil, errUnknownTool(call.Name)
		}
		if d.next != nil {
			return d.next.Run(ctx, call)
		}
		return nil, errUnknownTool(call.Name)
	}
	return d.checkedCall(ctx, found, call)
}

// checkedCall runs call on r once its binding and connection are read again (recheck), and,
// for a tool the binding's require_approval names as the session opened it or as the config is
// now, once the session's user allows the call (holdForApproval). A person can answer minutes
// later, so an allowed call is read again before it is sent, and its row counts its latency
// from the answer. It records the call's row, refused or run.
func (d *dispatcher) checkedCall(ctx context.Context, r route, call llm.ToolCall) ([]llm.ContentPart, error) {
	started := time.Now()
	current, err := d.recheck(ctx, r)
	if err != nil {
		d.record(r, call, started, failure{kind: store.InvocationDenied}, err)
		return nil, err
	}
	if r.binding.RequiresApproval(r.tool) || current.RequiresApproval(r.tool) {
		allowed, parts, how, err := d.holdForApproval(ctx, r, call)
		if !allowed {
			d.record(r, call, started, how, err)
			return parts, err
		}
		started = time.Now()
		if _, err := d.recheck(ctx, r); err != nil {
			d.record(r, call, started, failure{kind: store.InvocationDenied}, err)
			return nil, err
		}
	}
	parts, how, err := d.call(ctx, r, call)
	d.record(r, call, started, how, err)
	return parts, err
}

// failure is how a call failed, for its row and its log line (record).
type failure struct {
	// kind is one of the store.Invocation* values, empty when the call answered.
	kind string
	// status is the provider's HTTP status for the call's last answered request
	// (core.Exchange.Status), 0 when none was answered.
	status int
	// host is where the call's last request went, and body, challenge and retryAfter the
	// start of the body, the WWW-Authenticate and the Retry-After of its answer when that was a
	// 4xx or a 5xx (core.Exchange); "" when there was none.
	host, body, challenge, retryAfter string
	// message is what the router says of a call it held itself, with no error and no answer
	// to say it (heldByTheRouter); "" for every other call.
	message string
	// redact is the call's core.Exchange.Redact, for what the error the call returned says
	// (providerMessage); nil for a call that sent nothing through an Exchange.
	redact func(string) string
}

// answerOf is what the call's requests saw (core.Exchange), with no kind.
func answerOf(exchange *core.Exchange) failure {
	return failure{status: exchange.Status(), host: exchange.Host(), body: exchange.ErrorBody(),
		challenge: exchange.WWWAuthenticate(), retryAfter: exchange.RetryAfterHeader(), redact: exchange.Redact}
}

// record queues the row of one call that started at started and ended now, as how says.
// The row names the binding, the connection and the tool, and the shape of call's arguments
// (argumentShape), never what the call was asked or answered, except the start of the
// provider's error for a call that failed (error_message, below). An incognito session's names no
// session and keeps no shape: the lengths come from the conversation.
//
// A call that failed or was refused also leaves one Warn line, "connector call failed", on the
// session's logger: the row's names and id, the provider's HTTP status, and what the provider
// or the refusal said (providerMessage of err, the error the call returned, without what the
// call's Exchange redacts: an echoed token or request path, AI-1244). It also names the
// host the last request went to (provider_host) and the start of a 4xx's or a 5xx's body
// (provider_body, through providerBody), and, when that answer had them, its WWW-Authenticate
// (www_authenticate) and Retry-After (retry_after), each cut to providerMessageBytes (AI-1243).
// A call the model
// reads connector_rate_limited, outcome_unknown or a step-up of returned no error, so its line
// has no message, except one the limiter held before sending, whose message is
// heldByTheRouter's (AI-1244). An incognito session's line carries the message like any other's, by the
// owner's decision (AI-1242, 2026-10-14), but names no session, as its row does (Spec.Incognito).
// A call that answered logs nothing. The id is given here rather than by the store, so the line
// names the row it wrote.
//
// A failed call's row keeps the status, nil when none was answered, and an error_message: the
// line's provider_body when the provider answered a 4xx or a 5xx with one, since the MCP
// client's error for it often says only the status, else its provider_message, so the row holds
// nothing the line does not (AI-1244). Example: a 403 whose body says why keeps that body; an
// MCP isError, which comes in a 200, keeps the tool's words.
func (d *dispatcher) record(r route, call llm.ToolCall, started time.Time, how failure, err error) {
	sessionID := d.spec.ID
	var arguments []store.ArgumentShape
	if d.spec.Incognito {
		sessionID = ""
	} else {
		arguments = argumentShape(call.Arguments, r.declared)
	}
	row := store.ConnectorInvocation{
		ID:         store.NewID(),
		CustomerID: r.connection.CustomerID, ConnectionID: r.connection.ID, ConnectorID: r.binding.ConnectorID,
		ConfigID: d.spec.ConfigID, Binding: r.binding.Name, Tool: r.tool, SessionID: sessionID,
		StartedAt: started, LatencyMs: time.Since(started).Milliseconds(), ErrorType: how.kind,
		Arguments: arguments,
	}
	message, body := how.message, providerBody(how.body)
	if message == "" {
		message = providerMessage(err, how.redact)
	}
	// A call that answered has no failure to say: no status, no message. The body is "" unless
	// the last answer was a 4xx or a 5xx (core.Exchange.ErrorBody).
	row.ErrorMessage = cmp.Or(body, message)
	if how.status != 0 {
		row.ProviderStatus = &how.status
	}
	d.invocations.Record(row)
	if how.kind == "" {
		return
	}
	fields := []any{"customer", row.CustomerID, "connection", row.ConnectionID,
		"connector", row.ConnectorID, "binding", row.Binding, "tool", row.Tool, "session", row.SessionID,
		"invocation", row.ID, "error_type", row.ErrorType, "latency_ms", row.LatencyMs,
		"provider_status", how.status, "provider_message", message,
		"provider_host", how.host, "provider_body", body}
	if how.challenge != "" {
		fields = append(fields, "www_authenticate", core.CutAtRune(how.challenge, providerMessageBytes))
	}
	if how.retryAfter != "" {
		fields = append(fields, "retry_after", core.CutAtRune(how.retryAfter, providerMessageBytes))
	}
	d.logger.Warn("connector call failed", fields...)
}

// argumentShape is each top-level argument of a call's arguments with its JSON type and, for a
// string or an array, its length, sorted by name, so a call can be told apart afterwards
// without any value it was asked being kept (AI-990 F40): «thread_ts» sent as "" shows as a
// string of length 0. Only names in declared are kept: a tool that takes additionalProperties
// lets the model choose a key, which can be content (an email address), so the keys that are
// not declared are one entry named undeclaredArguments, an object whose length is their count. Empty arguments are the empty object, as the MCP source sends them
// (mcp.Toolset.Call). Arguments that are not a JSON object have no shape: nil.
func argumentShape(arguments string, declared map[string]bool) []store.ArgumentShape {
	raw := strings.TrimSpace(arguments)
	if raw == "" {
		raw = "{}"
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal([]byte(raw), &fields) != nil || fields == nil {
		return nil
	}
	shape := make([]store.ArgumentShape, 0, len(fields))
	undeclared := 0
	for name, value := range fields {
		if !declared[name] {
			undeclared++
			continue
		}
		argument := store.ArgumentShape{Name: name}
		// RFC 8259 section 3: a value's first character says its type.
		switch value[0] {
		case '"':
			var text string
			_ = json.Unmarshal(value, &text)
			length := utf8.RuneCountInString(text)
			argument.Type, argument.Length = "string", &length
		case '[':
			var items []json.RawMessage
			_ = json.Unmarshal(value, &items)
			length := len(items)
			argument.Type, argument.Length = "array", &length
		case '{':
			argument.Type = "object"
		case 't', 'f':
			argument.Type = "boolean"
		case 'n':
			argument.Type = "null"
		default:
			argument.Type = "number"
		}
		shape = append(shape, argument)
	}
	if undeclared > 0 {
		shape = append(shape, store.ArgumentShape{Name: undeclaredArguments, Type: "object", Length: &undeclared})
	}
	slices.SortFunc(shape, func(a, b store.ArgumentShape) int { return strings.Compare(a.Name, b.Name) })
	return shape
}

// undeclaredArguments is the name the row keeps for the arguments a tool's schema does not declare.
const undeclaredArguments = "(undeclared)"

// declaredArguments is the names in a tool's input schema under properties.
func declaredArguments(parameters map[string]any) map[string]bool {
	properties, _ := parameters["properties"].(map[string]any)
	declared := make(map[string]bool, len(properties))
	for name := range properties {
		declared[name] = true
	}
	return declared
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
// The binding's policy (store.BindingPolicy) changes what an interruption does. A wait
// binding's call does not see it, and the turn waits for its answer. A binding that is not
// cancellable stops waiting at the interruption and leaves the call running to its answer or
// deadline, so the interruption never sends the provider the cancel; the binding's timeout
// still does, as for every call. That holds for a call already sent: one still waiting for a
// slot (below) is not sent, whatever the binding says (Kanat, 2026-10-20, AI-1540 review). A
// binding with no policy is the paragraph above.
//
// A provider that answered a call on the same rate limit key with 429 and Retry-After is not
// sent the call until that passes, on any router sharing the limiter's Redis: the model reads
// connector_rate_limited instead, and so it does for the 429 itself, with the wait the router
// holds: the Retry-After, at most core.MaxBlock. The router never sends the call again by
// itself (Kanat, 2026-10-07, D8).
//
// A connection whose manifest caps its calls in flight (rate_limit.max_concurrent_calls)
// sends a call over the cap once one of them ends (callSlots). The wait is part of the
// binding's timeout: a call still waiting when it runs out is not sent, and the model reads
// connector_busy, so a runaway loop slows down instead of reaching the provider. A call that
// waited reads the limiter again once it has its slot: the call it waited behind may have
// been answered 429.
//
// It also says how the call failed, for its row and its log line: no kind when it answered,
// else one of the store.Invocation* values.
func (d *dispatcher) call(ctx context.Context, r route, call llm.ToolCall) ([]llm.ContentPart, failure, error) {
	policy := r.binding.Policy
	if policy != nil && policy.OnInterrupt == store.InterruptWait {
		ctx = context.WithoutCancel(ctx)
	}
	deadline := time.Now().Add(r.timeout)
	release, parts, how, err := d.admit(ctx, deadline, r, call)
	if release == nil {
		return parts, how, err
	}
	if policy == nil || policy.OnInterrupt == store.InterruptWait || policy.Cancellable == nil || *policy.Cancellable {
		defer release()
		return d.send(ctx, deadline, r, call)
	}
	type answer struct {
		parts []llm.ContentPart
		how   failure
		err   error
	}
	answered := make(chan answer, 1)
	go func() {
		defer release()
		parts, how, err := d.send(context.WithoutCancel(ctx), deadline, r, call)
		answered <- answer{parts, how, err}
	}()
	select {
	case got := <-answered:
		return got.parts, got.how, got.err
	case <-ctx.Done():
		// The call goes on, so whether the provider does what was asked is unknown here.
		return nil, failure{kind: store.InvocationOutcomeUnknown}, stack.Wrap(fmt.Errorf(
			"session: %s was left running at the provider: %w", call.Name, ctx.Err()))
	}
}

// admit waits until call may be sent: the provider holds no 429 on its rate limit key, and
// the connection has a slot under its cap before deadline, or ctx ends. It returns what frees
// the slot, or nil and the answer of a call that is not sent.
func (d *dispatcher) admit(ctx context.Context, deadline time.Time, r route, call llm.ToolCall) (func(), []llm.ContentPart, failure, error) {
	if wait := d.limiter.Wait(ctx, r.limit); wait > 0 {
		return nil, llm.TextParts(rateLimited(call.Name, wait)), failure{kind: store.InvocationDenied, message: heldByTheRouter(wait)}, nil
	}
	if r.maxInFlight == 0 {
		return func() {}, nil, failure{}, nil
	}
	waiting, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	release, err := d.slots.take(waiting, r.connection.ID, r.maxInFlight)
	if err != nil {
		if ctx.Err() == nil {
			return nil, llm.TextParts(busy(call.Name, r.maxInFlight)), failure{kind: store.InvocationDenied, message: heldByTheCap(r.maxInFlight)}, nil
		}
		return nil, nil, failure{kind: store.InvocationClientTimeout}, stack.Wrap(fmt.Errorf("session: %s was not sent: %w", call.Name, err))
	}
	if wait := d.limiter.Wait(ctx, r.limit); wait > 0 {
		release()
		return nil, llm.TextParts(rateLimited(call.Name, wait)), failure{kind: store.InvocationDenied, message: heldByTheRouter(wait)}, nil
	}
	return release, nil, failure{}, nil
}

// send runs one tool admit let go, inside the binding's deadline and the result cap,
// cancelled with ctx.
func (d *dispatcher) send(ctx context.Context, deadline time.Time, r route, call llm.ToolCall) ([]llm.ContentPart, failure, error) {
	bounded, cancel := context.WithDeadline(d.correlated(ctx), deadline)
	defer cancel()
	observed, exchange := core.WithExchange(bounded)
	result, err := r.toolset.Call(observed, call)
	if err != nil {
		if ctx.Err() == nil && errors.Is(bounded.Err(), context.DeadlineExceeded) {
			how := answerOf(exchange)
			how.kind = store.InvocationOutcomeUnknown
			return llm.TextParts(outcomeUnknown(call.Name)), how, nil
		}
		if held := d.limiter.Block(ctx, r.limit, exchange.RetryAfter()); held > 0 {
			return llm.TextParts(rateLimited(call.Name, held)), failed(ctx, exchange), nil
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
		return llm.TextParts(core.CutResult(text)), failure{}, nil
	}
	return result.Parts, failure{}, nil
}

// failed is how a call that failed before the binding's deadline failed, from what its
// requests saw (core.Exchange): a tool source's error does not say whether anything was sent,
// and the MCP client reports a 401 only as the words of its status. The status goes with it,
// for the call's log line, with the rest of what the requests saw (answerOf).
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
func failed(turn context.Context, exchange *core.Exchange) failure {
	how := answerOf(exchange)
	how.kind = failedAs(turn, exchange, how.status)
	return how
}

// failedAs is the kind of failed.
func failedAs(turn context.Context, seen *core.Exchange, status int) string {
	credential := seen.CredentialError()
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

// rateLimited is what the model reads of a call its provider's rate limit holds: the
// connector_rate_limited result with retry_after_seconds (Kanat, 2026-10-07, D8), in whole
// seconds rounded up, as Retry-After's delay-seconds are whole (RFC 9110 section 10.2.3).
func rateLimited(name string, wait time.Duration) string {
	return fmt.Sprintf("connector_rate_limited: the provider limits how often %s may be called, and it was "+
		"not run. retry_after_seconds: %d. Do not call it again before then.", name, retryAfterSeconds(wait))
}

// heldByTheRouter is the error_message, and the log line's provider_message, of a call the
// limiter held before sending (AI-1244): its error_type stays denied, as on the base, and this
// says why, with the seconds the model reads. The form is fixed, for a reader to match on:
// "rate limited by the router: retry after <seconds>s".
func heldByTheRouter(wait time.Duration) string {
	return fmt.Sprintf("rate limited by the router: retry after %ds", retryAfterSeconds(wait))
}

// busy is what the model reads of a call that waited its whole timeout behind the
// connection's limit of calls in flight, and was not sent.
func busy(name string, limit int) string {
	return fmt.Sprintf("connector_busy: %s was not run: the connector's limit of calls at once (%d) was reached, "+
		"and none of them ended in time. Call it again once one has answered.", name, limit)
}

// heldByTheCap is the error_message, and the log line's provider_message, of a call busy
// answered. The form is fixed, for a reader to match on: "held by the router: in-flight
// limit <limit> reached on the connection".
func heldByTheCap(limit int) string {
	return fmt.Sprintf("held by the router: in-flight limit %d reached on the connection", limit)
}

// retryAfterSeconds is wait in whole seconds rounded up, as Retry-After's delay-seconds are
// whole (RFC 9110 section 10.2.3).
func retryAfterSeconds(wait time.Duration) int64 {
	return int64((wait + time.Second - 1) / time.Second)
}

// recheck refuses a call that the session's config and connection no longer allow, before
// anything is sent: the prototype's authorizeConnectorTool
// (internal/session/connector_tools.go:229-288 on codex/connector-support at cf62af0d). The
// config must still hold the binding, with the same connector and connection, and grant the
// tool at the digest it was opened with. A binding opened through tool_patterns is granted
// while the current patterns still match the tool: Open already kept digest == pin. The
// connection must still be live, connected, one
// the binding may use, and reach where it reached when the session opened: the same
// definition revision, inputs and captured metadata, which are what its endpoints are built
// from. Two reads by primary key per call. It returns the binding as the config holds it now.
func (d *dispatcher) recheck(ctx context.Context, r route) (store.ConnectorBinding, error) {
	refuse := func(why string) (store.ConnectorBinding, error) {
		return store.ConnectorBinding{}, stack.Wrap(fmt.Errorf("session: connector %q: %s, so %s was not called",
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
	case r.binding.ToolPatterns != nil && current.ToolPatterns != nil:
		if !plugins.Offered(*current.ToolPatterns, r.tool) {
			return refuse("the tool is no longer granted")
		}
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
	return current, nil
}

// boundAlias reports whether a connector binding of the spec is called alias.
func (s Spec) boundAlias(alias string) bool {
	return slices.ContainsFunc(s.ConnectorBindings, func(b store.ConnectorBinding) bool { return b.Name == alias })
}

// toolPolicy is what the binding of the tool offered as name asks of the agent
// (store.BindingPolicy): what to say while it runs, and whether its call goes on after an
// interruption. The binding is the one bound under the name's alias, so a binding waiting
// for a login is found by the two tools it offers until then. Nothing for a name under no
// bound alias, or a binding with no policy.
func (d *dispatcher) toolPolicy(name string) agent.ToolPolicy {
	alias, _, cut := strings.Cut(name, mcp.Separator)
	index := slices.IndexFunc(d.spec.ConnectorBindings, func(b store.ConnectorBinding) bool { return b.Name == alias })
	if !cut || index < 0 || d.spec.ConnectorBindings[index].Policy == nil {
		return agent.ToolPolicy{}
	}
	policy := d.spec.ConnectorBindings[index].Policy
	return agent.ToolPolicy{PreSpeech: policy.PreSpeech, Waits: policy.OnInterrupt == store.InterruptWait,
		Cancels: policy.OnInterrupt == store.InterruptCancel}
}
