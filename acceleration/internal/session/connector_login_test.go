//go:build integration

package session

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"testing"
	"testing/fstest"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/none"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectorLoginSuite is which session bindings wait for their person to log in in the
// conversation, and what the dispatcher does with one before and after. The consent itself
// is the router's (api.ConnectorConsents), run end to end in the api package's
// ChatLoginsSuite; here a session is handed the consent its router would have begun.
type ConnectorLoginSuite struct {
	connectorFixture
	// begun is the consent the router hands a login, and shown who the conversation showed
	// it to.
	begun Consent
	shown []string
	// begins is how many consents the router began, and requested what each was asked for.
	begins    int
	requested []ConsentRequest
	// steps are the sign-ins the conversation showed on calls' steps, and ended how each
	// one's ended, by call id (held).
	mu    sync.Mutex
	steps map[string]persistent.Authorization
	ended map[string]string
}

func TestConnectorLoginSuite(t *testing.T) {
	suite.Run(t, new(ConnectorLoginSuite))
}

func (s *ConnectorLoginSuite) SetupTest() {
	s.connectorFixture.SetupTest()
	s.begun, s.shown, s.begins, s.requested = Consent{}, nil, 0, nil
	s.steps, s.ended = map[string]persistent.Authorization{}, map[string]string{}
}

// TestWithoutConsentsABindingWithNoSelectionIsLeftOutAsBefore: a deployment with connectors
// off, or a router that has no consents to begin, opens the session as it did before.
func (s *ConnectorLoginSuite) TestWithoutConsentsABindingWithNoSelectionIsLeftOutAsBefore() {
	spec := s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil))

	d, tools, unavailable, err := s.attach(spec)

	s.Require().NoError(err)
	s.Empty(tools)
	s.Nil(d.logins)
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableNoSelection}}, unavailable)
}

// TestAConfigWithNoBindingsOpensNoDispatcher: consents change nothing for a config that binds
// no connector.
func (s *ConnectorLoginSuite) TestAConfigWithNoBindingsOpensNoDispatcher() {
	d, tools, unavailable, err := s.attachWithConsents(s.persisted(s.spec(s.config(), "alice", nil)))

	s.Require().NoError(err)
	s.Nil(d)
	s.Empty(tools)
	s.Empty(unavailable)
}

// TestOnlyAnOptionalBindingOfTheCallersOwnWaitsForALogin: none chosen or one that needs
// reauthorization waits; every other reason, a required binding, the app's connection and a
// session with no conversation to ask in stay as they were.
func (s *ConnectorLoginSuite) TestOnlyAnOptionalBindingOfTheCallersOwnWaitsForALogin() {
	waits := []string{"crm__list_tools", "crm__call_tool"}
	stale := s.connection("alice", "primary")
	s.setState(stale, func(state *core.CredentialState) { state.Status = store.ConnectionNeedsReauthorization })
	staleApp := s.connection("", "secondary")
	s.setState(staleApp, func(state *core.CredentialState) { state.Status = store.ConnectionNeedsReauthorization })
	optional := s.config(s.chosen("crm", "whoami"))
	shared := s.persisted(s.spec(optional, "alice", nil))
	shared.ConversationID = "agent:" + persistent.ThreadChannelPrefix + "0199"
	shared.Thread = true
	unverified := s.persisted(s.spec(optional, "", nil))
	unverified.Caller, unverified.CallerKind = routing.Caller{}, auth.KindServer

	cases := map[string]struct {
		spec   Spec
		offers []string
		reason string
	}{
		"none chosen":           {s.persisted(s.spec(optional, "alice", nil)), waits, unavailableNoSelection},
		"needs reauthorization": {s.persisted(s.spec(optional, "alice", map[string]string{"crm": stale})), waits, unavailableReauthorize},
		"the app's connection":  {s.persisted(s.spec(s.config(s.fixed("crm", staleApp, "whoami")), "alice", nil)), nil, unavailableReauthorize},
		"a shared conversation": {shared, nil, unavailableShared},
		"an unverified caller":  {unverified, nil, unavailableUnverified},
		"no conversation":       {s.spec(optional, "alice", nil), nil, unavailableNoSelection},
	}
	for name, c := range cases {
		_, tools, unavailable, err := s.attachWithConsents(c.spec)

		s.Require().NoError(err, name)
		s.Equal(c.offers, names(tools), name)
		s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: c.reason}}, unavailable, name)
	}
	_, _, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(required(s.chosen("crm", "whoami"))), "alice", nil)))
	s.ErrorContains(err, `required connector "crm" cannot be used: no_selection`, "a required binding still fails the session")
}

// TestABindingWaitingForALoginKeepsItsPolicy: the two tools a binding offers until its person
// logs in are its own, so its pre_speech and on_interrupt apply to them as to the tools the
// login opens.
func (s *ConnectorLoginSuite) TestABindingWaitingForALoginKeepsItsPolicy() {
	binding := s.chosen("crm", "whoami")
	binding.Policy = &store.BindingPolicy{PreSpeech: "Let me open the CRM.", OnInterrupt: store.InterruptWait}

	d, tools, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(binding), "alice", nil)))

	s.Require().NoError(err)
	s.Equal([]string{"crm__list_tools", "crm__call_tool"}, names(tools))
	for _, name := range names(tools) {
		s.Equal(agent.ToolPolicy{PreSpeech: "Let me open the CRM.", Waits: true}, d.toolPolicy(name), name)
	}
}

// TestALoginOpensTheBindingOnlyForTheConsentItBegan: before the login the tools ask for it;
// a consent the login did not begin, or one for another connection, opens nothing; the one
// it began opens the binding, and its granted tools reach the account through call_tool.
func (s *ConnectorLoginSuite) TestALoginOpensTheBindingOnlyForTheConsentItBegan() {
	mine, other := s.pending("alice", "primary"), s.pending("alice", "secondary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", LaunchURL: "https://router.example/launch", Name: "CRM"}
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil)))
	s.Require().NoError(err)

	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.Contains(asked, `"status":"authorization_required"`)
	s.NotContains(asked, s.begun.LaunchURL)
	s.Equal([]string{"alice"}, s.shown, "asked of the caller")

	_, unknown, _ := d.loginFinished(s.ctx, "attempt-bob", mine)
	_, elsewhere, _ := d.loginFinished(s.ctx, "attempt-alice", other)
	still, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.connect(mine, "primary")
	name, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	said, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	listed, err := s.call(d, "crm__list_tools", `{}`)
	s.Require().NoError(err)
	_, ungranted := s.call(d, "crm__call_tool", `{"tool":"secret"}`)

	s.False(unknown, "an attempt the login did not begin")
	s.False(elsewhere, "a consent for another connection")
	s.Contains(still, `"status":"authorization_required"`)
	s.True(finished)
	s.Equal("CRM", name)
	s.Equal("primary", said, "the account the login connected")
	s.JSONEq(`{"tools":[{"name":"whoami","description":"Says which account this is.","input_schema":{"type":"object","properties":{}}}]}`, listed)
	s.Error(ungranted, "a tool the grant does not name")
}

// TestALoginTheConversationCannotShowIsNotAskedFor: with no reply to show it on, the model is
// told the binding is not available rather than that a button was shown.
func (s *ConnectorLoginSuite) TestALoginTheConversationCannotShowIsNotAskedFor() {
	s.begun = Consent{ConnectionID: "unused", AuthorizationID: "attempt-alice", Name: "CRM"}
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "bob", nil)))
	s.Require().NoError(err)
	d.logins.canAsk = func(string) bool { return false }

	said, err := s.call(d, "crm__list_tools", `{}`)

	s.Require().NoError(err)
	s.Contains(said, `"status":"unavailable"`)
	s.Zero(s.begins, "no consent is begun that nobody would see")
	_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", "unused")
	s.False(finished, "nothing was asked, so nothing carries on")
}

// TestALoginFinishedWithoutAHandBackIsPickedUpOnTheNextCall: the consent finished on another
// router, or its hand-back failed, so this session was never told. The next call reads the
// connection, finds it connected, and runs, with no second consent.
func (s *ConnectorLoginSuite) TestALoginFinishedWithoutAHandBackIsPickedUpOnTheNextCall() {
	mine := s.pending("alice", "primary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM"}
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil)))
	s.Require().NoError(err)
	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.Require().Contains(asked, `"status":"authorization_required"`)

	s.connect(mine, "primary")
	said, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Equal(1, s.begins, "no second consent")
	s.Equal([]string{"alice"}, s.shown, "no second button")
	_, handedBack, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	s.False(handedBack, "a hand-back that arrives after is not run again")
}

// TestALoginWithNoneChosenOpensOnAnAccountConnectedOutsideTheChat: the session opened before
// the caller had any connection, and they then connected one from their app, with a consent
// their backend began. The next call opens on it, as the session would have had it been
// connected then, and nobody is asked to log in again.
func (s *ConnectorLoginSuite) TestALoginWithNoneChosenOpensOnAnAccountConnectedOutsideTheChat() {
	s.begun = Consent{ConnectionID: "unused", AuthorizationID: "attempt-alice", Name: "CRM"}
	d, tools, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil)))
	s.Require().NoError(err)

	s.connection("alice", "primary")
	said, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

	s.Require().NoError(err)
	s.Equal([]string{"crm__list_tools", "crm__call_tool"}, names(tools), "it opened waiting for a login")
	s.Equal("primary", said)
	s.Zero(s.begins, "no consent begun")
	s.Empty(s.shown, "no button shown")
}

// TestALoginWithNoneChosenStillAsksWhenTheCallerHasTwoAccounts: with two connected, which one
// the session uses is the person's to say, so the login asks rather than picks.
func (s *ConnectorLoginSuite) TestALoginWithNoneChosenStillAsksWhenTheCallerHasTwoAccounts() {
	s.begun = Consent{ConnectionID: "unused", AuthorizationID: "attempt-alice", Name: "CRM"}
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil)))
	s.Require().NoError(err)

	s.connection("alice", "primary")
	s.connection("alice", "secondary")
	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

	s.Require().NoError(err)
	s.Contains(asked, `"status":"authorization_required"`)
	s.Equal(1, s.begins)
}

// TestAReconnectFinishedWithoutAHandBackIsPickedUpOnTheNextCall: the same for a connection
// the caller chose that needed reauthorization.
func (s *ConnectorLoginSuite) TestAReconnectFinishedWithoutAHandBackIsPickedUpOnTheNextCall() {
	stale := s.connection("alice", "primary")
	s.setState(stale, func(state *core.CredentialState) { state.Status = store.ConnectionNeedsReauthorization })
	s.begun = Consent{ConnectionID: stale, AuthorizationID: "attempt-alice", Name: "CRM"}
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", map[string]string{"crm": stale})))
	s.Require().NoError(err)
	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.Require().Contains(asked, `"status":"authorization_required"`)

	s.setState(stale, func(state *core.CredentialState) { state.Status = store.ConnectionConnected })
	said, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Equal(1, s.begins)
}

// TestASecondCallInTheReplyShowingTheConsentReusesIt: list_tools then call_tool in one reply
// begin one consent; a later reply that no longer shows it begins a fresh one.
func (s *ConnectorLoginSuite) TestASecondCallInTheReplyShowingTheConsentReusesIt() {
	s.begun = Consent{ConnectionID: s.pending("alice", "primary"), AuthorizationID: "attempt-alice", Name: "CRM",
		ExpiresAt: time.Now().Add(10 * time.Minute)}
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil)))
	s.Require().NoError(err)
	showing := true
	d.logins.shows = func(id string) bool { return showing && id == "attempt-alice" }

	_, err = s.call(d, "crm__list_tools", `{}`)
	s.Require().NoError(err)
	reused, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.Equal(1, s.begins)
	s.Contains(reused, `"status":"authorization_required"`)

	showing = false
	_, err = s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.Equal(2, s.begins, "a later reply is shown a consent of its own")
}

// TestAConfigThatDroppedTheBindingBeginsNoConsent: the config is read before a consent begins,
// as before every call.
func (s *ConnectorLoginSuite) TestAConfigThatDroppedTheBindingBeginsNoConsent() {
	s.begun = Consent{ConnectionID: s.pending("alice", "primary"), AuthorizationID: "attempt-alice", Name: "CRM"}
	config := s.config(s.chosen("crm", "whoami"))
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(config, "alice", nil)))
	s.Require().NoError(err)
	s.rebind(config)

	said, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

	s.Require().NoError(err)
	s.Contains(said, `"status":"unavailable"`)
	s.Zero(s.begins)
	s.Empty(s.shown)
}

// TestAConnectionConnectedBeforeOpensOnlyOnceTheChatsConsentConnectsItAnew: the router asks
// on the caller's connection that is connected already (a new chat with no selection, and two
// connected, so none is implied). The session does not use it on its own: only a consent
// since then, a later connected_at, opens it.
func (s *ConnectorLoginSuite) TestAConnectionConnectedBeforeOpensOnlyOnceTheChatsConsentConnectsItAnew() {
	mine := s.connection("alice", "primary")
	s.connection("alice", "secondary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM"}
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil)))
	s.Require().NoError(err)

	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	still, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.setState(mine, func(state *core.CredentialState) { state.ConnectedAt = time.Now().UTC() })
	said, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

	s.Require().NoError(err)
	s.Contains(asked, `"status":"authorization_required"`)
	s.Contains(still, `"status":"authorization_required"`, "connected before the chat asked is not a consent in it")
	s.Equal("primary", said)
}

// TestAConnectionOnARevisionMarkedBrokenWaitsForALogin: the caller's connection opens while
// its revision is unmarked. Once a later revision of the built-in marks it broken, the binding
// waits for a login as one that needs reauthorization does, and the consent that connects it
// on the latest revision opens it.
func (s *ConnectorLoginSuite) TestAConnectionOnARevisionMarkedBrokenWaitsForALogin() {
	s.connectorID, s.revision = "crm"+strings.ReplaceAll(uuid.NewString(), "-", ""), 1
	s.seedCRM(1, "")
	mine := s.connection("alice", "primary")
	spec := s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", map[string]string{"crm": mine}))
	_, before, unmarked, err := s.attachWithConsents(spec)
	s.Require().NoError(err)
	s.Equal([]string{"crm__whoami"}, names(before), "unmarked, it opens")
	s.Empty(unmarked)

	s.seedCRM(2, "broken_revisions:\n  - revisions: [1]\n    reason: reads the wrong path\n")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM"}
	d, tools, unavailable, err := s.attachWithConsents(spec)
	s.Require().NoError(err)
	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.setState(mine, func(state *core.CredentialState) { state.DefinitionRevision, state.ConnectedAt = 2, time.Now().UTC() })
	said, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

	s.Equal([]string{"crm__list_tools", "crm__call_tool"}, names(tools))
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableReauthorize}}, unavailable)
	s.Contains(asked, `"status":"authorization_required"`)
	s.Require().NoError(err)
	s.Equal("primary", said, "the login's consent moved it to the latest revision")
}

// TestAnImpliedConnectionOnARevisionMarkedBrokenIsTheOneTheLoginReconnects: the caller's one
// connected connection, implied, needs a reconnect. The login asks for that one, not for the
// newest of the caller's connections, which a consent with none named would take.
func (s *ConnectorLoginSuite) TestAnImpliedConnectionOnARevisionMarkedBrokenIsTheOneTheLoginReconnects() {
	s.connectorID, s.revision = "crm"+strings.ReplaceAll(uuid.NewString(), "-", ""), 1
	s.seedCRM(1, "")
	mine := s.connection("alice", "primary")
	s.pending("alice", "secondary")
	s.seedCRM(2, "broken_revisions:\n  - revisions: [1]\n    reason: reads the wrong path\n")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM"}
	d, tools, unavailable, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil)))
	s.Require().NoError(err)

	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

	s.Require().NoError(err)
	s.Equal([]string{"crm__list_tools", "crm__call_tool"}, names(tools))
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableReauthorize}}, unavailable)
	s.Contains(asked, `"status":"authorization_required"`)
	s.Require().Len(s.requested, 1)
	s.Equal(mine, s.requested[0].ConnectionID)
}

// TestANoneConnectionThatCannotBeUsedIsRejectedNotAskedToLogIn: a none connection takes no
// consent (begin refuses it), so one on a revision marked broken, or one the resolver
// invalidated, is unavailable as credential_rejected, and the login that waits for it begins no
// consent.
func (s *ConnectorLoginSuite) TestANoneConnectionThatCannotBeUsedIsRejectedNotAskedToLogIn() {
	broken := func(mine string) {
		s.seedCRM(2, "broken_revisions:\n  - revisions: [1]\n    reason: reads the wrong path\n")
	}
	invalidated := func(mine string) {
		s.setState(mine, func(state *core.CredentialState) { state.Status = store.ConnectionNeedsReauthorization })
	}
	for name, spoil := range map[string]func(string){"a broken revision": broken, "needs reauthorization": invalidated} {
		s.connectorID, s.revision = "crm"+strings.ReplaceAll(uuid.NewString(), "-", ""), 1
		s.seedCRM(1, "")
		mine := s.noneConnection("alice", "primary")
		spoil(mine)
		s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM"}
		d, _, unavailable, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", map[string]string{"crm": mine})))
		s.Require().NoError(err, name)

		told, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

		s.Require().NoError(err, name)
		s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableCredentialRejected}}, unavailable, name)
		s.Contains(told, `"status":"credential_rejected"`, name)
		s.Zero(s.begins, name)
	}
}

// TestAConnectionOnAnOutdatedRevisionStillOpens: a later revision that marks nothing broken
// leaves the connection's binding open.
func (s *ConnectorLoginSuite) TestAConnectionOnAnOutdatedRevisionStillOpens() {
	s.connectorID, s.revision = "crm"+strings.ReplaceAll(uuid.NewString(), "-", ""), 1
	s.seedCRM(1, "")
	mine := s.connection("alice", "primary")
	s.seedCRM(2, "")

	_, tools, unavailable, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", map[string]string{"crm": mine})))

	s.Require().NoError(err)
	s.Equal([]string{"crm__whoami"}, names(tools))
	s.Empty(unavailable)
}

// seedCRM seeds the test's connector as a built-in at revision, with more manifest YAML in
// extra, as a router start with that file does.
func (s *ConnectorLoginSuite) seedCRM(revision int, extra string) {
	s.Require().NoError(s.store.SeedConnectorDefinitions(s.ctx,
		fstest.MapFS{s.connectorID + ".yaml": {Data: []byte(s.crmManifest(s.connectorID, revision) + extra)}}))
}

// TestACallThroughALoginLeavesOneInvocationRow: call_tool on a binding a login opened is a
// connector call like any other, and leaves one row. Asking for the login leaves none: no
// connection was called.
func (s *ConnectorLoginSuite) TestACallThroughALoginLeavesOneInvocationRow() {
	mine := s.pending("alice", "primary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM"}
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil)))
	s.Require().NoError(err)
	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.Require().Contains(asked, `"status":"authorization_required"`)
	s.connect(mine, "primary")
	_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	s.Require().True(finished)

	said, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

	s.Require().NoError(err)
	s.Equal("primary", said)
	rows := s.invocations(mine, 1)
	s.Equal("whoami", rows[0].Tool)
	s.Equal("crm", rows[0].Binding)
	s.Empty(rows[0].ErrorType)
}

// TestACallThroughALoginRecordsTheShapeOfTheToolsArguments: the row is the inner tool's
// arguments, not the {tool, arguments} of the login's call_tool verb; so is a refused one's.
func (s *ConnectorLoginSuite) TestACallThroughALoginRecordsTheShapeOfTheToolsArguments() {
	mine := s.pending("alice", "primary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM"}
	config := s.config(s.chosen("crm", "echo"))
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(config, "alice", nil)))
	s.Require().NoError(err)
	_, err = s.call(d, "crm__call_tool", `{"tool":"echo"}`)
	s.Require().NoError(err)
	s.connect(mine, "primary")
	_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	s.Require().True(finished)
	length := 2
	inner := []store.ArgumentShape{{Name: "note", Type: "string", Length: &length}}

	_, err = s.call(d, "crm__call_tool", `{"tool":"echo","arguments":{"note":"hi"}}`)
	s.Require().NoError(err)
	s.rebind(config, s.chosen("crm"))
	_, err = s.call(d, "crm__call_tool", `{"tool":"echo","arguments":{"note":"hi"}}`)

	s.Require().Error(err)
	rows := s.invocations(mine, 2)
	sort.Slice(rows, func(i, j int) bool { return rows[i].ErrorType < rows[j].ErrorType })
	s.Empty(rows[0].ErrorType)
	s.Equal(inner, rows[0].Arguments)
	s.Equal(store.InvocationDenied, rows[1].ErrorType)
	s.Equal(inner, rows[1].Arguments)
}

// TestARefusedCallThroughALoginLeavesADeniedRow: the config stopped granting the tool after
// the login opened the binding.
func (s *ConnectorLoginSuite) TestARefusedCallThroughALoginLeavesADeniedRow() {
	mine := s.pending("alice", "primary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM"}
	config := s.config(s.chosen("crm", "whoami"))
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(config, "alice", nil)))
	s.Require().NoError(err)
	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.Require().Contains(asked, `"status":"authorization_required"`)
	s.connect(mine, "primary")
	_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	s.Require().True(finished)
	s.rebind(config, s.chosen("crm"))

	_, err = s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

	s.Require().ErrorContains(err, "the tool is no longer granted")
	s.Equal(store.InvocationDenied, s.invocations(mine, 1)[0].ErrorType)
	s.Zero(s.provider.calls("primary"))
}

// TestAFailedCallThroughALoginLogsTheToolsWords and TestARefusedCallThroughALoginLogsTheReason:
// call_tool on a binding a login opened leaves the "connector call failed" line of the inner
// tool (AI-1242), as a call that bypassed the login does.
func (s *ConnectorLoginSuite) TestAFailedCallThroughALoginLogsTheToolsWords() {
	d, logged, mine := s.openedLoginLogging(s.chosen("crm", "fails"))

	_, err := s.call(d, "crm__call_tool", `{"tool":"fails"}`)

	s.Require().Error(err)
	line := s.onlyFailedLine(logged)
	s.Contains(line, "binding=crm")
	s.Contains(line, "connection="+mine)
	s.Contains(line, "tool=fails")
	s.Contains(line, "error_type=external_server")
	s.Contains(line, "provider_status=200")
	s.Contains(line, `provider_message="the record is locked"`)
}

func (s *ConnectorLoginSuite) TestARefusedCallThroughALoginLogsTheReason() {
	config := s.config(s.chosen("crm", "whoami"))
	d, logged, mine := s.openedLoginLoggingOf(config)
	s.rebind(config, s.chosen("crm"))

	_, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)

	s.Require().Error(err)
	line := s.onlyFailedLine(logged)
	s.Contains(line, "connection="+mine)
	s.Contains(line, "tool=whoami")
	s.Contains(line, "error_type=denied")
	s.Contains(line, `provider_message="session: connector \"crm\": the tool is no longer granted, so crm__whoami was not called"`)
}

// openedLoginLogging is a session of one binding, chosen, whose login alice's consent opened,
// with the session's log.
func (s *ConnectorLoginSuite) openedLoginLogging(binding store.ConnectorBinding) (*dispatcher, *bytes.Buffer, string) {
	return s.openedLoginLoggingOf(s.config(binding))
}

func (s *ConnectorLoginSuite) openedLoginLoggingOf(config store.AgentConfig) (*dispatcher, *bytes.Buffer, string) {
	mine := s.pending("alice", "primary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM"}
	manager := s.managerWith(fixtureRequestTimeout)
	var logged bytes.Buffer
	manager.logger = slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelDebug}))
	d, _, _, err := s.attachWithConsentsOn(manager, s.persisted(s.spec(config, "alice", nil)))
	s.Require().NoError(err)
	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.Require().Contains(asked, `"status":"authorization_required"`)
	s.connect(mine, "primary")
	_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	s.Require().True(finished)
	return d, &logged, mine
}

// onlyFailedLine is the one "connector call failed" line in logged.
func (s *ConnectorLoginSuite) onlyFailedLine(logged *bytes.Buffer) string {
	var lines []string
	for _, line := range strings.Split(logged.String(), "\n") {
		if strings.Contains(line, `msg="connector call failed"`) {
			lines = append(lines, line)
		}
	}
	s.Require().Len(lines, 1, logged.String())
	return lines[0]
}

// TestABindingALoginOpensAuditsWithTheSession: the consent's callback hands the login back on
// its own request's context; the provider refuses the token while the tools are listed, and
// the revocation names the session, as one during a call does.
func (s *ConnectorLoginSuite) TestABindingALoginOpensAuditsWithTheSession() {
	mine := s.pending("alice", "primary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM"}
	spec := s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil))
	spec.ID = uuid.NewString()
	d, _, _, err := s.attachWithConsents(spec)
	s.Require().NoError(err)
	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.Require().Contains(asked, `"status":"authorization_required"`)
	// Connected with another account's token, which the provider refuses.
	s.connect(mine, "secondary")
	callback := core.WithCorrelation(s.ctx, core.Correlation{RequestID: "callback-request"})

	_, opened, _ := d.loginFinished(callback, "attempt-alice", mine)

	s.False(opened, "the provider refused the token")
	var rows []store.ConnectorAuditEvent
	s.Require().Eventually(func() bool {
		rows, err = s.store.ConnectorAuditEvents(s.ctx, s.customerID, store.AuditFilter{ConnectionID: mine})
		return err == nil && len(rows) == 1
	}, 5*time.Second, 20*time.Millisecond)
	s.Equal(store.AuditGrantRevoked, rows[0].Action)
	s.Equal(spec.ID, rows[0].SessionID)
	s.Equal("callback-request", rows[0].RequestID)
}

// TestAnIncognitoSessionIsOfferedNoLogin: an incognito session keeps no conversation
// (Spec.Normalize), and a login is asked only in one, so no consent is begun and no binding
// is opened by one.
func (s *ConnectorLoginSuite) TestAnIncognitoSessionIsOfferedNoLogin() {
	s.begun = Consent{ConnectionID: s.pending("alice", "primary"), AuthorizationID: "attempt-alice", Name: "CRM"}
	spec := s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil))
	spec.Incognito = true
	s.Require().NoError(spec.Normalize())

	d, tools, unavailable, err := s.attachWithConsents(spec)

	s.Require().NoError(err)
	s.Nil(d.logins)
	s.Empty(tools)
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableNoSelection}}, unavailable)
	s.Zero(s.begins)
}

// invocations is connection's call log, once it holds count rows and a moment more.
func (s *ConnectorLoginSuite) invocations(connection string, count int) []store.ConnectorInvocation {
	read := func() []store.ConnectorInvocation {
		rows, err := s.store.ConnectorInvocations(s.ctx, s.customerID, connection, 0, nil)
		s.Require().NoError(err)
		return rows
	}
	s.Require().Eventually(func() bool { return len(read()) >= count }, 5*time.Second, 20*time.Millisecond)
	s.Never(func() bool { return len(read()) > count }, 300*time.Millisecond, 50*time.Millisecond)
	return read()
}

// pending is a connection of user's to account that was never connected.
func (s *ConnectorLoginSuite) pending(user, account string) string {
	connection := &store.ConnectorConnection{
		CustomerID: s.customerID, ConnectorID: s.connectorID, DefinitionRevision: s.revision,
		OwnerType: store.OwnerUser, OwnerID: user, AuthScheme: bearer.Name, Inputs: map[string]string{"account": account},
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, s.registry, connection))
	return connection.ID
}

// noneConnection is a connected none connection of user's to account, which holds no secret.
func (s *ConnectorLoginSuite) noneConnection(user, account string) string {
	connection := &store.ConnectorConnection{
		CustomerID: s.customerID, ConnectorID: s.connectorID, DefinitionRevision: s.revision,
		OwnerType: store.OwnerUser, OwnerID: user, AuthScheme: none.Name, Inputs: map[string]string{"account": account},
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, s.registry, connection))
	stored, _, err := none.New().Complete(s.ctx, core.CompleteInput{})
	s.Require().NoError(err)
	s.setState(connection.ID, func(state *core.CredentialState) {
		state.Credentials, state.Status = stored, store.ConnectionConnected
	})
	return connection.ID
}

// connect is a consent finishing on id: account's token stored, the connection connected.
func (s *ConnectorLoginSuite) connect(id, account string) {
	stored, _, err := bearer.New().Complete(s.ctx, core.CompleteInput{Supplied: map[string]string{bearer.SuppliedToken: tokenOf(account)}})
	s.Require().NoError(err)
	s.setState(id, func(state *core.CredentialState) {
		state.Credentials, state.Status = stored, store.ConnectionConnected
	})
}

// attachWithConsents is attach on a router that begins consents: s.begun, which the session's
// conversation shows to whoever it is asked for.
func (s *ConnectorLoginSuite) attachWithConsents(spec Spec) (*dispatcher, []harness.Tool, []ConnectorUnavailable, error) {
	return s.attachWithConsentsOn(s.managerWith(fixtureRequestTimeout), spec)
}

// attachWithConsentsOn is attachWithConsents on manager.
func (s *ConnectorLoginSuite) attachWithConsentsOn(manager *Manager, spec Spec) (*dispatcher, []harness.Tool, []ConnectorUnavailable, error) {
	manager.options.Connectors.Consents = func(_ context.Context, request ConsentRequest) (Consent, error) {
		if s.begun.AuthorizationID == "" || request.UserID == "" {
			return Consent{}, errors.New("no consent to begin")
		}
		s.begins++
		s.requested = append(s.requested, request)
		return s.begun, nil
	}
	d, tools, unavailable, err := manager.attachConnectors(s.ctx, &spec)
	if d != nil {
		s.T().Cleanup(d.Close)
		s.T().Cleanup(d.closeLogins)
		if d.logins != nil {
			d.logins.canAsk = func(string) bool { return true }
			d.logins.ask = func(owner string, _ persistent.ConnectorAuthorization) bool {
				s.shown = append(s.shown, owner)
				return true
			}
		}
	}
	return d, tools, unavailable, err
}

// persisted is spec kept in a conversation, which a login is asked in.
func (s *ConnectorLoginSuite) persisted(spec Spec) Spec {
	spec.PersistConversation, spec.Text = true, true
	return spec
}

// TestACallWaitsForTheLoginOnItsStepAndRunsOnceConnected: the consent is shown on the call's
// own step, the call waits, and once the person connects it runs on their account and answers
// as though they had been connected all along.
func (s *ConnectorLoginSuite) TestACallWaitsForTheLoginOnItsStepAndRunsOnceConnected() {
	mine := s.pending("alice", "primary")
	expires := time.Now().Add(10 * time.Minute).UTC()
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", LaunchURL: "https://router.example/launch/attempt-alice",
		HandoffToken: "handoff-alice", Name: "CRM", ExpiresAt: expires}
	d := s.held()

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().Eventually(func() bool { return s.stepOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	s.Never(func() bool { return len(answer) > 0 }, 200*time.Millisecond, 20*time.Millisecond, "it waits for the person")
	s.connect(mine, "primary")
	_, finished, handed := d.loginFinished(s.ctx, "attempt-alice", mine)
	got := s.answerOf(answer)

	s.True(finished)
	s.Equal(1, handed, "handed to the call waiting")
	s.Require().NoError(got.err)
	s.Equal("primary", got.said)
	s.Equal(persistent.Authorization{
		Type: "oauth", Provider: s.connectorID, ProviderName: "CRM", Title: "Connect CRM",
		Message: "You sign in at CRM, and the agent never sees your password.",
		URL:     "https://router.example/launch/attempt-alice", HandoffToken: "handoff-alice", ExpiresAt: &expires,
	}, *s.stepOf("call-1"))
	s.Equal(persistent.AuthorizationConnected, s.endedOf("call-1"))
}

// TestADeclinedLoginEndsTheCallWithoutRunningIt: "Not now" answers the call at once, with
// words for the model and nothing sent to the provider.
func (s *ConnectorLoginSuite) TestADeclinedLoginEndsTheCallWithoutRunningIt() {
	mine := s.pending("alice", "primary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM", ExpiresAt: time.Now().Add(10 * time.Minute)}
	d := s.held()

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().Eventually(func() bool { return s.stepOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	s.False(d.declineLogin("call-2"), "a call that is not waiting")
	s.True(d.declineLogin("call-1"))
	got := s.answerOf(answer)

	s.Require().NoError(got.err)
	s.Contains(got.said, `"status":"declined"`)
	s.Equal(persistent.AuthorizationDeclined, s.endedOf("call-1"))
	s.Never(func() bool {
		rows, err := s.store.ConnectorInvocations(s.ctx, s.customerID, mine, 0, nil)
		return err != nil || len(rows) > 0
	}, 300*time.Millisecond, 50*time.Millisecond, "nothing reached the provider")
}

// TestAConsentThatEndsWithoutConnectingEndsTheCall: the provider refused, or the person came
// back as another account. The attempt is spent, so the call stops waiting, and the next call
// begins another consent rather than wait on that one.
func (s *ConnectorLoginSuite) TestAConsentThatEndsWithoutConnectingEndsTheCall() {
	mine := s.pending("alice", "primary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM", ExpiresAt: time.Now().Add(10 * time.Minute)}
	d := s.held()
	d.logins.shows = func(id string) bool { return id == "attempt-alice" }

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().Eventually(func() bool { return s.stepOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	s.False(d.loginEndedWithout("attempt-bob", mine, persistent.AuthorizationFailed), "an attempt the login did not begin")
	s.True(d.loginEndedWithout("attempt-alice", mine, persistent.AuthorizationFailed))
	got := s.answerOf(answer)
	s.Require().NoError(got.err)
	s.Contains(got.said, `"status":"failed"`)
	s.Equal(persistent.AuthorizationFailed, s.endedOf("call-1"))

	again := s.callInBackground(s.ctx, d, "call-2", "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().Eventually(func() bool { return s.stepOf("call-2") != nil }, 5*time.Second, 10*time.Millisecond)
	s.Equal(2, s.begins, "a spent attempt is not shown again")
	s.True(d.declineLogin("call-2"))
	s.answerOf(again)
}

// TestALoginNotFinishedInTimeEndsTheCall: the consent lives ten minutes in production; once it
// is spent, the call stops waiting.
func (s *ConnectorLoginSuite) TestALoginNotFinishedInTimeEndsTheCall() {
	s.begun = Consent{ConnectionID: s.pending("alice", "primary"), AuthorizationID: "attempt-alice", Name: "CRM",
		ExpiresAt: time.Now().Add(300 * time.Millisecond)}
	d := s.held()

	got := s.answerOf(s.callInBackground(s.ctx, d, "call-1", "crm__call_tool", `{"tool":"whoami"}`))

	s.Require().NoError(got.err)
	s.Contains(got.said, `"status":"expired"`)
	s.Equal(persistent.AuthorizationExpired, s.endedOf("call-1"))
}

// TestALoginFinishedOnAnotherRouterIsPickedUpWhileTheCallWaits: the callback ran elsewhere,
// so nothing is handed back here; the waiting call reads the connection again and carries on.
func (s *ConnectorLoginSuite) TestALoginFinishedOnAnotherRouterIsPickedUpWhileTheCallWaits() {
	mine := s.pending("alice", "primary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", Name: "CRM", ExpiresAt: time.Now().Add(10 * time.Minute)}
	d := s.held()

	answer := s.callInBackground(s.ctx, d, "call-1", "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().Eventually(func() bool { return s.stepOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	s.connect(mine, "primary")
	got := s.answerOf(answer)

	s.Require().NoError(got.err)
	s.Equal("primary", got.said)
	s.Equal(persistent.AuthorizationConnected, s.endedOf("call-1"))
	_, handedBack, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	s.False(handedBack, "a hand-back that arrives after is not run again")
}

// TestATurnThatEndsStopsTheWait: an interruption, or the session closing, ends the call rather
// than leave it waiting out the consent.
func (s *ConnectorLoginSuite) TestATurnThatEndsStopsTheWait() {
	s.begun = Consent{ConnectionID: s.pending("alice", "primary"), AuthorizationID: "attempt-alice", Name: "CRM",
		ExpiresAt: time.Now().Add(10 * time.Minute)}
	d := s.held()
	turn, end := context.WithCancel(s.ctx)
	defer end()

	answer := s.callInBackground(turn, d, "call-1", "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().Eventually(func() bool { return s.stepOf("call-1") != nil }, 5*time.Second, 10*time.Millisecond)
	end()
	got := s.answerOf(answer)

	s.ErrorIs(got.err, context.Canceled)
	s.Empty(s.endedOf("call-1"), "the reply that ended settles its own steps")
}

// held is a dispatcher for alice's crm binding whose conversation shows the consent on the
// steps of its calls, which then wait for her: s.steps records what each step was shown, and
// s.ended how each one's sign-in ended.
func (s *ConnectorLoginSuite) held() *dispatcher {
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil)))
	s.Require().NoError(err)
	d.logins.await = func(owner, callID string, found persistent.Authorization) bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.steps[callID] = found
		return owner == "alice"
	}
	d.logins.ended = func(callID, status string) bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.ended[callID] = status
		return true
	}
	return d
}

// answered is what a call run in the background said.
type answered struct {
	said string
	err  error
}

// callInBackground runs the call id on turn, and hands back what it said once it answers.
func (s *ConnectorLoginSuite) callInBackground(turn context.Context, d *dispatcher, id, name, arguments string) chan answered {
	answer := make(chan answered, 1)
	go func() {
		parts, err := d.Run(turn, llm.ToolCall{ID: id, Name: name, Arguments: arguments})
		answer <- answered{said: llm.TextOf(parts), err: err}
	}()
	return answer
}

// answerOf waits for a call run in the background to answer, past loginPoll.
func (s *ConnectorLoginSuite) answerOf(answer chan answered) answered {
	select {
	case got := <-answer:
		return got
	case <-time.After(2*loginPoll + 5*time.Second):
		s.FailNow("the call never answered")
		return answered{}
	}
}

func (s *ConnectorLoginSuite) stepOf(callID string) *persistent.Authorization {
	s.mu.Lock()
	defer s.mu.Unlock()
	found, ok := s.steps[callID]
	if !ok {
		return nil
	}
	return &found
}

func (s *ConnectorLoginSuite) endedOf(callID string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ended[callID]
}
