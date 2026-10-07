//go:build integration

package session

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
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
}

func TestConnectorLoginSuite(t *testing.T) {
	suite.Run(t, new(ConnectorLoginSuite))
}

func (s *ConnectorLoginSuite) SetupTest() {
	s.connectorFixture.SetupTest()
	s.begun, s.shown = Consent{}, nil
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

// TestALoginOpensTheBindingOnlyForTheConsentItBegan: before the login the tools ask for it;
// a consent the login did not begin, or one for another connection, opens nothing; the one
// it began opens the binding, and its granted tools reach the account through call_tool.
func (s *ConnectorLoginSuite) TestALoginOpensTheBindingOnlyForTheConsentItBegan() {
	mine, other := s.connection("alice", "primary"), s.connection("alice", "secondary")
	s.begun = Consent{ConnectionID: mine, AuthorizationID: "attempt-alice", LaunchURL: "https://router.example/launch", Name: "CRM"}
	d, _, _, err := s.attachWithConsents(s.persisted(s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil)))
	s.Require().NoError(err)

	asked, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	s.Contains(asked, `"status":"authorization_required"`)
	s.NotContains(asked, s.begun.LaunchURL)
	s.Equal([]string{"alice"}, s.shown, "asked of the caller")

	_, unknown := d.loginFinished(s.ctx, s.manager, "attempt-bob", mine)
	_, elsewhere := d.loginFinished(s.ctx, s.manager, "attempt-alice", other)
	still, err := s.call(d, "crm__call_tool", `{"tool":"whoami"}`)
	s.Require().NoError(err)
	name, finished := d.loginFinished(s.ctx, s.manager, "attempt-alice", mine)
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
	d.logins.ask = func(string, persistent.ConnectorAuthorization) bool { return false }

	said, err := s.call(d, "crm__list_tools", `{}`)

	s.Require().NoError(err)
	s.Contains(said, `"status":"unavailable"`)
	_, finished := d.loginFinished(s.ctx, s.manager, "attempt-alice", "unused")
	s.False(finished, "nothing was asked, so nothing carries on")
}

// attachWithConsents is attach on a router that begins consents: s.begun, which the session's
// conversation shows to whoever it is asked for.
func (s *ConnectorLoginSuite) attachWithConsents(spec Spec) (*dispatcher, []harness.Tool, []ConnectorUnavailable, error) {
	manager := s.managerWith(fixtureRequestTimeout)
	manager.options.Connectors.Consents = func(_ context.Context, request ConsentRequest) (Consent, error) {
		if s.begun.AuthorizationID == "" || request.UserID == "" {
			return Consent{}, errors.New("no consent to begin")
		}
		return s.begun, nil
	}
	d, tools, unavailable, err := manager.attachConnectors(s.ctx, &spec)
	if d != nil {
		s.T().Cleanup(d.Close)
		s.T().Cleanup(d.closeLogins)
		if d.logins != nil {
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
