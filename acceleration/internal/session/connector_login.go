package session

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A session binding the caller has no usable connection for, because none was chosen or the
// chosen one needs reauthorization, is offered as two tools whose names do not depend on the
// connection, as a user plugin is (plugins.UserTools): a session's tools are fixed when it
// opens. The first call asks the person to log in, in the conversation; once they have, the
// same two tools reach their account through the dispatcher, and the session carries on.
const (
	loginListTools = "list_tools"
	loginCallTool  = "call_tool"
)

// Consents starts the provider's consent for a person's own connection, T17's attempt
// (api.ConnectorConsents): single use, ten minutes, bound to the first browser that hands it
// off.
type Consents func(ctx context.Context, request ConsentRequest) (Consent, error)

// ConsentRequest is whose connection a consent is for.
type ConsentRequest struct {
	CustomerID  string
	ConnectorID string
	// UserID is the verified caller, who owns the connection.
	UserID string
	// ConnectionID is the connection to reconnect. Empty for none chosen: the caller's newest
	// connection to the connector in any status, or a new one.
	ConnectionID string
	// StepUp is what the provider asked for on a call of ConnectionID, a scope_required
	// outcome (step_up.go): the consent is a step-up that asks for that access. Nil for a
	// login.
	StepUp *core.Outcome
	// ReturnURL is the session's (Spec.ReturnURL): where the consent goes back to. Empty is
	// the dashboard.
	ReturnURL string
}

// Consent is a consent begun for one connection, as a client is handed it.
type Consent struct {
	ConnectionID string
	// AuthorizationID is the attempt, the last segment of LaunchURL.
	AuthorizationID string
	LaunchURL       string
	HandoffToken    string
	ExpiresAt       time.Time
	// Name is the connector's, for what the person and the model are told.
	Name string
}

// loginReasons are why a session binding was left out that a login in the conversation
// fixes. Every other reason stays as it was: a shared conversation or an unverified caller
// has nobody to log in as. A rejected token or key is offered as a login too, whose tools
// begin no consent and tell the model it must be replaced (credentialRejected), and which
// opens once it was (AI-990).
var loginReasons = []string{unavailableNoSelection, unavailableReauthorize, unavailableCredentialRejected}

// logins are a dispatcher's bindings waiting for their person to log in, by alias, and how
// it asks them to.
type logins struct {
	byAlias  map[string]*login
	consents Consents
	// ask shows a consent on the reply being written, in the session's own conversation
	// (conversation.AskToConnect).
	ask func(owner string, found persistent.ConnectorAuthorization) bool
	// canAsk reports whether ask would show owner a login now (conversation.CanAskToConnect),
	// checked before a consent begins so none is begun that nobody would see.
	canAsk func(owner string) bool
	// chose keeps a connection a login opened as the session's selection (Session.chose).
	chose func(name, connectionID string)
	// shows reports whether the reply being written already shows an attempt's login, and
	// connected marks one finished (conversation.ShowsLogin, conversation.ConnectorConnected).
	shows     func(authorizationID string) bool
	connected func(authorizationID string) bool
	// await shows a consent on the step of the call that needs it, which then waits for the
	// person (conversation.AwaitAuthorization), and ended settles that step
	// (conversation.AuthorizationEnded). waits reports whether the step still waits, which it
	// stops doing when its reply ends (conversation.AwaitsAuthorization). Without await the
	// call answers at once that the person was shown a button.
	await func(owner, callID string, found persistent.Authorization) bool
	ended func(callID, status string) bool
	waits func(callID string) bool
	// open is Manager.openBinding, which opens a binding once its connection is usable.
	open func(ctx context.Context, spec Spec, binding store.ConnectorBinding, selection string, d *dispatcher) (string, error)
	// implied is Manager.impliedSelection: the caller's one connected connection, which a login
	// with none chosen opens on once the person has connected one outside the conversation.
	implied func(ctx context.Context, spec Spec, binding store.ConnectorBinding) (string, error)
	logger  *slog.Logger
}

// login is one binding waiting for its person to log in, and once they have, the binding
// opened on the connection they logged into.
type login struct {
	binding store.ConnectorBinding
	// rejected is a chosen connection whose token or key the provider rejected, or that reads
	// a revision marked broken (unavailableCredentialRejected): no consent fixes it, so none is
	// begun.
	rejected bool

	mu sync.Mutex
	// connectionID is the connection the consents are for, empty before the first one when
	// none was chosen.
	connectionID string
	// begun are the attempts asked for, newest last, so a button pressed on an earlier reply
	// still carries on.
	begun []string
	name  string
	// last is the newest consent begun, reused while the reply being written shows it.
	last Consent
	// asked is whether a consent was begun, and connectedBefore the connection's connected_at
	// when it was: the connection opens without a hand-back only once a consent since then
	// connected it (a later connected_at), never because it was connected already.
	asked           bool
	connectedBefore *time.Time
	// opened holds the binding's routes once the login is made. Nil before.
	opened *dispatcher
	// held are the calls waiting for the person to answer a consent on their step, by call id.
	held map[string]waiting
}

// waiting is a call waiting for the person to answer the consent of authorizationID, which is
// sent how it ended.
type waiting struct {
	authorizationID string
	answer          chan string
}

// maxBegun bounds the attempts a login remembers. Each lives ten minutes
// (api.attemptLifetime), so an older one is one nobody is waiting on; the conversation keeps
// as many asking replies (conversation.maxAsked).
const maxBegun = 10

// loginPoll is how often a call waiting for a login reads its connection again. A consent
// finished on another router is never handed back to this one (ConnectorConsentFinished), so
// the store is the only place it shows.
const loginPoll = 3 * time.Second

// waiting is the login of alias, if alias is a binding waiting for one.
func (w *logins) waiting(alias string) (*login, bool) {
	if w == nil {
		return nil, false
	}
	l, ok := w.byAlias[alias]
	return l, ok
}

// offerLogin makes binding a login of d when reason is one a login fixes, the binding is the
// caller's to choose, it is optional, and the session has a conversation to ask in. It
// reports whether it did. selection is the connection chosen, empty for none.
func (m *Manager) offerLogin(spec Spec, binding store.ConnectorBinding, reason, selection string, d *dispatcher) bool {
	if m.options.Connectors.Consents == nil || m.options.Connectors.Transports == nil || !spec.PersistConversation ||
		binding.Required || binding.Connection.Type != selectionSession || !slices.Contains(loginReasons, reason) {
		return false
	}
	if d.logins == nil {
		d.logins = &logins{byAlias: map[string]*login{}, consents: m.options.Connectors.Consents,
			open: m.openBinding, implied: m.impliedSelection, logger: m.logger}
	}
	d.logins.byAlias[binding.Name] = &login{binding: binding, connectionID: selection,
		rejected: reason == unavailableCredentialRejected}
	d.tools = append(d.tools, loginTools(binding.Name)...)
	return true
}

// loginTools are the two tools a waiting binding is offered as.
func loginTools(alias string) []harness.Tool {
	list, call := alias+mcp.Separator+loginListTools, alias+mcp.Separator+loginCallTool
	return []harness.Tool{
		{
			Name: list,
			Description: fmt.Sprintf("List what the user's own %s account can do. Call this before %s. "+
				"If the user has not connected it yet, they are shown a button to connect it; tell them "+
				"to press it, and you are told once they have.", alias, call),
			Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
		},
		{
			Name:        call,
			Description: fmt.Sprintf("Run one of the tools %s lists, on the user's own %s account.", list, alias),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"tool":      map[string]any{"type": "string", "description": "The tool's name, as listed."},
					"arguments": map[string]any{"type": "object", "description": "The tool's arguments, as its input schema says."},
				},
				"required": []string{"tool"},
			},
		},
	}
}

// askIn has the logins ask in conv, the session's own conversation, and keep what they open
// with chose.
func (d *dispatcher) askIn(conv *persistent.Conversation, chose func(name, connectionID string)) {
	if conv != nil {
		d.approvals.asksIn(conv)
	}
	if d.logins != nil && conv != nil {
		d.logins.ask, d.logins.canAsk = conv.AskToConnect, conv.CanAskToConnect
		d.logins.shows, d.logins.connected = conv.ShowsLogin, conv.ConnectorConnected
		d.logins.await, d.logins.ended, d.logins.waits = conv.AwaitAuthorization, conv.AuthorizationEnded, conv.AwaitsAuthorization
		d.logins.chose = chose
	}
}

// runLogin answers a call to a waiting binding's tools: the login's account once it is made,
// and until then a consent the person is shown in the conversation, which the call waits for.
func (d *dispatcher) runLogin(ctx context.Context, l *login, call llm.ToolCall) ([]llm.ContentPart, error) {
	verb := strings.TrimPrefix(call.Name, l.binding.Name+mcp.Separator)
	if verb != loginListTools && verb != loginCallTool {
		return nil, errUnknownTool(call.Name)
	}
	opened, instead, err := d.openOrAsk(ctx, l, call.ID)
	if err != nil {
		return nil, err
	}
	if opened == nil {
		return llm.TextParts(instead), nil
	}
	if verb == loginListTools {
		return llm.TextParts(listed(opened.tools, l.binding.Name)), nil
	}
	var asked struct {
		Tool      string          `json:"tool"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(call.Arguments), &asked); err != nil || asked.Tool == "" {
		return nil, fmt.Errorf("session: %s needs the name of the tool to run", call.Name)
	}
	name := l.binding.Name + mcp.Separator + asked.Tool
	found, ok := opened.routes[name]
	if !ok {
		return nil, errUnknownTool(name)
	}
	// Logged as Run logs a call, one row for each, refused or run.
	started := time.Now()
	arguments := string(asked.Arguments)
	if arguments == "" || arguments == "null" {
		arguments = "{}"
	}
	run := llm.ToolCall{ID: call.ID, Name: name, Arguments: arguments}
	current, err := d.recheck(ctx, found)
	if err != nil {
		d.record(found, run, started, failure{kind: store.InvocationDenied}, err)
		return nil, err
	}
	parts, how, err := d.call(ctx, found, current, run)
	d.record(found, run, started, how, err)
	return parts, err
}

// openOrAsk is the login's opened binding, or else what the model reads instead. A call whose
// step shows the consent waits for the person to answer it (waitForLogin).
func (d *dispatcher) openOrAsk(ctx context.Context, l *login, callID string) (*dispatcher, string, error) {
	l.mu.Lock()
	if l.opened != nil || d.openConnected(ctx, l) {
		defer l.mu.Unlock()
		return l.opened, "", nil
	}
	answer, instead := d.askToLogIn(ctx, l, callID)
	expires := l.last.ExpiresAt
	l.mu.Unlock()
	if answer == nil {
		return nil, instead, nil
	}
	return d.waitForLogin(ctx, l, callID, answer, expires)
}

// openConnected opens the login's binding on a connection the store says is usable now, and
// reports whether it did. l.mu is held.
//
// A consent finished on another router, or one whose hand-back failed, still connected the
// connection in the store, so the connection is read before anybody is asked again: once
// it is connected and the binding may use it, the binding opens here. The plugin path does
// the same by reading its login on every call (userPluginRunner.connect).
//
// A connection the chat asked to consent on again, which was connected already, opens only
// once that consent connected it anew (connectedBefore): the person consents in this chat
// before the session uses it, and a consent that came back with another account
// (account_mismatch) leaves connected_at as it was.
func (d *dispatcher) openConnected(ctx context.Context, l *login) bool {
	if l.connectionID != "" && d.store != nil {
		connection, err := d.store.ConnectorConnection(ctx, d.spec.CustomerID, l.connectionID)
		if err == nil && connection.Status == store.ConnectionConnected &&
			connection.ConnectorID == l.binding.ConnectorID && mayUse(d.spec, l.binding, connection) &&
			(!l.asked || connectedSince(connection.ConnectedAt, l.connectedBefore)) &&
			d.openLogin(ctx, l, connection.ID) {
			for _, begun := range l.begun {
				if d.logins.connected != nil {
					d.logins.connected(begun)
				}
			}
			return true
		}
	}
	// None chosen yet: the person may have connected an account since the session opened, from
	// their app with a consent its backend began rather than in this chat. That one opens as it
	// would have had it been connected when the session opened, with no second consent.
	if l.connectionID == "" && d.logins.implied != nil {
		id, err := d.logins.implied(ctx, d.spec, l.binding)
		if err != nil {
			d.logins.logger.Warn("could not read the caller's connections for a login", "connector", l.binding.Name, "error", err)
		} else if id != "" && d.openLogin(ctx, l, id) {
			return true
		}
	}
	return false
}

// waitForLogin holds the call until the person answers the consent on its step. Once they
// connect, it carries on with the binding the login opened, and when the consent is declined,
// expires or fails, the model is told so. A turn interrupted first, or the session closing,
// ends the call with nothing run, and so does its reply ending while the call goes on.
func (d *dispatcher) waitForLogin(ctx context.Context, l *login, callID string, answer <-chan string, expires time.Time) (*dispatcher, string, error) {
	defer func() {
		l.mu.Lock()
		delete(l.held, callID)
		l.mu.Unlock()
	}()
	expiry := time.NewTimer(time.Until(expires))
	defer expiry.Stop()
	poll := time.NewTicker(loginPoll)
	defer poll.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil, "", stack.Wrap(fmt.Errorf("session: %s stopped waiting for its login: %w", l.binding.Name, ctx.Err()))
		case status := <-answer:
			if status == persistent.AuthorizationConnected && !d.stillAsks(callID) {
				return nil, loginStopped(l.name), nil
			}
			return d.loginEnded(l, callID, status)
		case <-expiry.C:
			return d.loginEnded(l, callID, persistent.AuthorizationExpired)
		case <-poll.C:
			if !d.stillAsks(callID) {
				return nil, loginStopped(l.name), nil
			}
			l.mu.Lock()
			opened := l.opened != nil || d.openConnected(ctx, l)
			l.mu.Unlock()
			if opened {
				return d.loginEnded(l, callID, persistent.AuthorizationConnected)
			}
		}
	}
}

// stillAsks reports whether the step of the call still waits for the person. A reply can end
// and leave its calls running, as an interruption that lets tools finish does (agent.stopTools),
// and one waiting for a login would then run the stopped request once they connect for another.
func (d *dispatcher) stillAsks(callID string) bool {
	return d.logins.waits == nil || d.logins.waits(callID)
}

// loginEnded settles the step of a call that waited for its login as status says, and is what
// the call carries on with.
func (d *dispatcher) loginEnded(l *login, callID, status string) (*dispatcher, string, error) {
	if d.logins.ended != nil {
		d.logins.ended(callID, status)
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	switch {
	case status == persistent.AuthorizationConnected && l.opened != nil:
		return l.opened, "", nil
	case status == persistent.AuthorizationDeclined:
		return nil, loginDeclined(l.name), nil
	case status == persistent.AuthorizationFailed:
		return nil, loginFailed(l.name), nil
	default:
		return nil, loginExpired(l.name), nil
	}
}

// openLogin opens the login's binding on connectionID and records it as the session's
// selection for the binding, so a fork or a reopened chat chooses it again rather than ask
// for another login. l.mu is held.
func (d *dispatcher) openLogin(ctx context.Context, l *login, connectionID string) bool {
	opened := &dispatcher{store: d.store, spec: d.spec, routes: map[string]route{}, logger: d.logger}
	// Opened as attachConnectors opens a binding, on the context a call runs on, so a refresh
	// while the tools are listed names this session, whether the turn or the consent's
	// callback asked for it.
	reason, err := d.logins.open(d.correlated(ctx), d.spec, l.binding, connectionID, opened)
	if err != nil || reason != "" {
		opened.Close()
		d.logins.logger.Warn("a login in the conversation is on a connection the session cannot use",
			"connector", l.binding.Name, "reason", reason, "error", err)
		return false
	}
	l.opened = opened
	if d.logins.chose != nil {
		d.logins.chose(l.binding.Name, connectionID)
	}
	return true
}

// connectedSince reports whether a connection connected at now was connected anew after
// before, its connected_at when a consent was begun (nil for never).
func connectedSince(now, before *time.Time) bool {
	return now != nil && (before == nil || now.After(*before))
}

// askToLogIn shows the caller a consent for their own connection on the reply being written,
// and on the step of callID, which then waits for their answer on the channel it returns.
// Without a channel it returns what the model reads instead, which names no URL and no token,
// since those are for the person. l.mu is held.
//
// The config is read first, as the dispatcher does before every call (recheck), so a binding
// the config no longer declares as the caller's own begins no consent. A second call in the
// reply that already shows a consent still open reuses it rather than begin another.
func (d *dispatcher) askToLogIn(ctx context.Context, l *login, callID string) (chan string, string) {
	if !d.stillWaits(ctx, l.binding) {
		return nil, loginUnavailable(l.binding.Name)
	}
	if l.rejected {
		return nil, credentialRejected(l.binding.Name)
	}
	if d.logins.canAsk == nil || !d.logins.canAsk(d.spec.Caller.UserID) {
		return nil, loginUnavailable(l.binding.Name)
	}
	consent := l.last
	if consent.AuthorizationID == "" || !time.Now().Before(consent.ExpiresAt) ||
		d.logins.shows == nil || !d.logins.shows(consent.AuthorizationID) {
		begun, ok := d.beginConsent(ctx, l)
		if !ok {
			return nil, loginUnavailable(l.binding.Name)
		}
		consent = begun
	}
	expires := consent.ExpiresAt.UTC()
	if d.logins.await == nil || !d.logins.await(d.spec.Caller.UserID, callID, persistent.Authorization{
		Type: "oauth", Provider: l.binding.ConnectorID, ProviderName: consent.Name,
		Title:   "Connect " + consent.Name,
		Message: "You sign in at " + consent.Name + ", and the agent never sees your password.",
		URL:     consent.LaunchURL, HandoffToken: consent.HandoffToken, ExpiresAt: &expires,
	}) {
		return nil, loginRequired(consent.Name)
	}
	answer := make(chan string, 1)
	if l.held == nil {
		l.held = map[string]waiting{}
	}
	l.held[callID] = waiting{authorizationID: consent.AuthorizationID, answer: answer}
	return answer, ""
}

// beginConsent begins a consent for the caller's own connection and shows it on the reply
// being written. It reports false when none could be begun or shown. l.mu is held.
func (d *dispatcher) beginConsent(ctx context.Context, l *login) (Consent, bool) {
	consent, err := d.logins.consents(ctx, ConsentRequest{
		CustomerID: d.spec.CustomerID, ConnectorID: l.binding.ConnectorID,
		UserID: d.spec.Caller.UserID, ConnectionID: l.connectionID, ReturnURL: d.spec.ReturnURL,
	})
	if err != nil {
		d.logins.logger.Warn("could not begin a consent in the conversation", "connector", l.binding.Name, "error", err)
		return Consent{}, false
	}
	shown := d.logins.ask != nil && d.logins.ask(d.spec.Caller.UserID, persistent.ConnectorAuthorization{
		Name: l.binding.Name, ConnectorID: l.binding.ConnectorID, ConnectionID: consent.ConnectionID,
		AuthorizationID: consent.AuthorizationID, Title: "Connect " + consent.Name,
		LaunchURL: consent.LaunchURL, HandoffToken: consent.HandoffToken, ExpiresAt: consent.ExpiresAt,
	})
	if !shown {
		return Consent{}, false
	}
	if !l.asked {
		l.asked = true
		if connection, err := d.store.ConnectorConnection(ctx, d.spec.CustomerID, consent.ConnectionID); err == nil {
			l.connectedBefore = connection.ConnectedAt
		}
	}
	l.connectionID, l.name, l.last = consent.ConnectionID, consent.Name, consent
	l.begun = append(l.begun, consent.AuthorizationID)
	if len(l.begun) > maxBegun {
		l.begun = l.begun[len(l.begun)-maxBegun:]
	}
	return consent, true
}

// stillWaits reports whether the stored config still declares binding as the caller's own
// connection to the same connector.
func (d *dispatcher) stillWaits(ctx context.Context, binding store.ConnectorBinding) bool {
	if d.store == nil {
		return false
	}
	config, err := d.store.AgentConfig(ctx, d.spec.CustomerID, d.spec.ConfigID)
	if err != nil {
		return false
	}
	return slices.ContainsFunc(config.Connectors, func(current store.ConnectorBinding) bool {
		return current.Name == binding.Name && current.ConnectorID == binding.ConnectorID &&
			current.Connection.Type == selectionSession
	})
}

// loginRequired is what the model reads while the person is shown a consent to press.
func loginRequired(name string) string {
	raw, _ := json.Marshal(struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}{
		Status: authorizationRequired,
		Message: fmt.Sprintf("The user has not connected %s. They have been shown a button to connect it. "+
			"Tell them to press it; you are told once they have, and then carry on.", name),
	})
	return string(raw)
}

// loginDeclined is what the model reads when the person chose not to connect name.
func loginDeclined(name string) string {
	raw, _ := json.Marshal(struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}{
		Status: persistent.AuthorizationDeclined,
		Message: fmt.Sprintf("The user chose not to connect %s, so nothing was done with it. Tell them, and that "+
			"they can connect it next time they ask for something that needs it.", name),
	})
	return string(raw)
}

// loginExpired is what the model reads when the person did not connect name in time.
func loginExpired(name string) string {
	raw, _ := json.Marshal(struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}{
		Status: persistent.AuthorizationExpired,
		Message: fmt.Sprintf("The user did not connect %s in time, so nothing was done with it. Tell them, and "+
			"that they can ask again to connect it.", name),
	})
	return string(raw)
}

// loginFailed is what the model reads when the consent for name ended without connecting it.
func loginFailed(name string) string {
	raw, _ := json.Marshal(struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}{
		Status: persistent.AuthorizationFailed,
		Message: fmt.Sprintf("Connecting %s did not go through, so nothing was done with it. Tell them, and that "+
			"they can ask again to try once more.", name),
	})
	return string(raw)
}

// loginStopped is what the model reads of a call whose reply was stopped before the person
// connected name.
func loginStopped(name string) string {
	raw, _ := json.Marshal(struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}{
		Status:  "stopped",
		Message: fmt.Sprintf("The reply was stopped before the user connected %s, so nothing was done with it.", name),
	})
	return string(raw)
}

// authorizationRequired is the status a waiting binding's tools answer with until the person
// logs in, the word a user plugin's do (plugins.AuthorizationRequired).
const authorizationRequired = "authorization_required"

// credentialRejected is what the model reads for a connection whose token or key the provider
// rejected, or that was saved on a connector revision marked broken (AI-1002): nobody can log
// in to fix it here, whoever set it up must save it again.
func credentialRejected(alias string) string {
	raw, _ := json.Marshal(struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}{
		Status: unavailableCredentialRejected,
		Message: fmt.Sprintf("%s cannot use the token or key stored for the user's connection: the provider rejected "+
			"it, or the connector changed since it was saved. It cannot be used until whoever set it up saves it again. "+
			"Tell the user that; do not offer to connect it here.", alias),
	})
	return string(raw)
}

// loginUnavailable is what the model reads when nobody can be asked to log in here.
func loginUnavailable(alias string) string {
	raw, _ := json.Marshal(struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}{
		Status: "unavailable",
		Message: fmt.Sprintf("%s cannot be connected here right now, so the user cannot use it. "+
			"Tell them it is not available; do not offer to set it up.", alias),
	})
	return string(raw)
}

// listed is the tools of an opened binding as the model reads them through list_tools: each
// one's name at the provider, what it does and its input schema.
func listed(tools []harness.Tool, alias string) string {
	type entry struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		InputSchema map[string]any `json:"input_schema,omitempty"`
	}
	entries := make([]entry, 0, len(tools))
	for _, tool := range tools {
		entries = append(entries, entry{Name: strings.TrimPrefix(tool.Name, alias+mcp.Separator),
			Description: tool.Description, InputSchema: tool.Parameters})
	}
	raw, _ := json.Marshal(struct {
		Tools []entry `json:"tools"`
	}{entries})
	return string(raw)
}

// loginFinished opens the binding whose consent this attempt was, on the connection it
// connected, and returns the connector's name and how many calls waiting for the login it
// handed it to. It reports false when none of d's logins asked for it, or the connection
// cannot be used: the binding then stays waiting.
func (d *dispatcher) loginFinished(ctx context.Context, authorizationID, connectionID string) (string, bool, int) {
	if d.logins == nil {
		return "", false, 0
	}
	for _, l := range d.logins.byAlias {
		l.mu.Lock()
		if l.opened != nil || l.connectionID != connectionID || !slices.Contains(l.begun, authorizationID) {
			l.mu.Unlock()
			continue
		}
		opened := d.openLogin(ctx, l, connectionID)
		handed := 0
		for id, call := range l.held {
			// One whose reply ended ends on its own, and the chat carries on as after any
			// login.
			if !opened || !d.stillAsks(id) {
				continue
			}
			select {
			case call.answer <- persistent.AuthorizationConnected:
				handed++
			default:
			}
		}
		name := l.name
		l.mu.Unlock()
		return name, opened, handed
	}
	return "", false, 0
}

// loginEndedWithout ends, as status says, the wait of the calls waiting for this attempt's
// consent on this connection. The consent ended without connecting and the attempt is spent,
// so a call after this begins another. It reports false when none of d's logins began it.
func (d *dispatcher) loginEndedWithout(authorizationID, connectionID, status string) bool {
	if d.logins == nil {
		return false
	}
	for _, l := range d.logins.byAlias {
		l.mu.Lock()
		if l.opened != nil || l.connectionID != connectionID || !slices.Contains(l.begun, authorizationID) {
			l.mu.Unlock()
			continue
		}
		if l.last.AuthorizationID == authorizationID {
			l.last = Consent{}
		}
		// A call of a later reply may wait on a newer consent, which is still open.
		for _, call := range l.held {
			if call.authorizationID != authorizationID {
				continue
			}
			select {
			case call.answer <- status:
			default:
			}
		}
		l.mu.Unlock()
		return true
	}
	return false
}

// declineLogin ends the wait of the call id for its login, as the person declined it. It
// reports whether the call was waiting.
func (d *dispatcher) declineLogin(callID string) bool {
	if d.logins == nil {
		return false
	}
	for _, l := range d.logins.byAlias {
		l.mu.Lock()
		call, ok := l.held[callID]
		l.mu.Unlock()
		if ok {
			select {
			case call.answer <- persistent.AuthorizationDeclined:
			default:
			}
			return true
		}
	}
	return false
}

// closeLogins ends the toolsets the logins opened.
func (d *dispatcher) closeLogins() {
	if d.logins == nil {
		return
	}
	for _, l := range d.logins.byAlias {
		l.mu.Lock()
		if l.opened != nil {
			l.opened.Close()
		}
		l.mu.Unlock()
	}
}

// ConnectorConsentFinished hands a consent the router's callback finished back to the session
// that asked for it in its conversation: the binding opens on the connection, the button says
// it is done, and the session carries on with what the login was asked for, so nobody has to
// ask again. Only the session whose dispatcher began this attempt, for this connection, of
// this customer, is handed it.
func (m *Manager) ConnectorConsentFinished(ctx context.Context, customerID, connectionID, authorizationID string) {
	m.mu.Lock()
	held := make([]*Session, 0, len(m.sessions))
	for _, s := range m.sessions {
		if s.connectors != nil && s.spec.CustomerID == customerID {
			held = append(held, s)
		}
	}
	m.mu.Unlock()
	for _, s := range held {
		name, ok, handed := s.connectors.loginFinished(ctx, authorizationID, connectionID)
		if !ok {
			continue
		}
		if s.persisted != nil {
			s.persisted.ConnectorConnected(authorizationID)
		}
		// A call waiting for the login carries on with what it was asked, in the reply that
		// asked, so there is nothing to ask again.
		if handed > 0 {
			return
		}
		text := name + " is connected now. Carry on with what I asked for before you needed it."
		if err := s.FollowUp(context.WithoutCancel(ctx), "login_"+store.NewID(), text); err != nil {
			m.logger.Warn("could not carry on after a login in the conversation", "session", s.id, "error", err)
		}
		return
	}
}

// ConnectorConsentEnded hands a consent the router's callback ended without connecting back to
// the session that asked for it in its conversation. Its calls waiting for the login stop
// waiting with status, declined when the person refused at the provider and failed otherwise,
// rather than wait out an attempt nobody can finish any more.
func (m *Manager) ConnectorConsentEnded(customerID, connectionID, authorizationID, status string) {
	m.mu.Lock()
	held := make([]*Session, 0, len(m.sessions))
	for _, s := range m.sessions {
		if s.connectors != nil && s.spec.CustomerID == customerID {
			held = append(held, s)
		}
	}
	m.mu.Unlock()
	for _, s := range held {
		if s.connectors.loginEndedWithout(authorizationID, connectionID, status) {
			return
		}
	}
}

// DeclineLogin ends the wait of a connector call for the person's login, as they declined it.
// The call runs nothing, and the model is told they did not connect. It reports whether the
// call was waiting.
func (s *Session) DeclineLogin(callID string) bool {
	return s.connectors != nil && s.connectors.declineLogin(callID)
}
