//go:build integration

package session

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// AttachConnectorsSuite is what a session opens of its config's connector bindings: whose
// connection each may use, what a required and an optional one do when it cannot, and which
// tools are offered. The prototype's connector_tools_integration_test.go cases on
// codex/connector-support at cf62af0d are here and in DispatcherSuite.
type AttachConnectorsSuite struct {
	connectorFixture
}

func TestAttachConnectorsSuite(t *testing.T) {
	suite.Run(t, new(AttachConnectorsSuite))
}

// TestEachVerifiedCallerMayChooseTheirOwnConnection: the prototype's
// TestConnectorSessionSelectionIsScopedToVerifiedCaller, first half.
func (s *AttachConnectorsSuite) TestEachVerifiedCallerMayChooseTheirOwnConnection() {
	alice, bob := s.connection("alice", "primary"), s.connection("bob", "secondary")
	config := s.config(s.chosen("crm", "whoami"))

	for user, id := range map[string]string{"alice": alice, "bob": bob} {
		d, tools, unavailable, err := s.attach(s.spec(config, user, map[string]string{"crm": id}))

		s.Require().NoError(err, user)
		s.Empty(unavailable, user)
		s.Equal([]string{"crm__whoami"}, names(tools), user)
		said, err := s.call(d, "crm__whoami", "{}")
		s.Require().NoError(err, user)
		s.Equal(map[string]string{"alice": "primary", "bob": "secondary"}[user], said)
	}
}

// TestAliceCannotChooseBobsConnection: the prototype's last case. The answer is the one a
// connection that does not exist gets, so it says nothing of whose the id is.
func (s *AttachConnectorsSuite) TestAliceCannotChooseBobsConnection() {
	bob := s.connection("bob", "secondary")
	config := s.config(required(s.chosen("crm", "whoami")))

	_, _, _, err := s.attach(s.spec(config, "alice", map[string]string{"crm": bob}))
	_, _, _, missing := s.attach(s.spec(config, "alice", map[string]string{"crm": "not-a-connection"}))

	s.ErrorContains(err, `required connector "crm" cannot be used: connection_unavailable`)
	s.Equal(missing.Error(), err.Error())
	s.Zero(s.provider.calls("secondary"))
	s.NotContains(s.provider.sent("secondary"), "initialize", "nothing was sent with Bob's credential")
}

// TestAnAnonymousCallerOrAGuestChoosesNoConnection: the prototype's second case. Either may go
// by Alice's name, which nobody vouched for.
func (s *AttachConnectorsSuite) TestAnAnonymousCallerOrAGuestChoosesNoConnection() {
	alice := s.connection("alice", "primary")
	config := s.config(required(s.chosen("crm", "whoami")))

	for _, kind := range []auth.Kind{auth.KindAnonymous, auth.KindGuest} {
		spec := s.spec(config, "alice", map[string]string{"crm": alice})
		spec.CallerKind = kind

		_, _, _, err := s.attach(spec)

		s.ErrorContains(err, "caller_unverified", string(kind))
	}
}

func (s *AttachConnectorsSuite) TestTheAppsBackendActingForAliceMayChooseHerConnection() {
	alice := s.connection("alice", "primary")
	config := s.config(required(s.chosen("crm", "whoami")))
	spec := s.spec(config, "alice", map[string]string{"crm": alice})
	spec.CallerKind = auth.KindServer

	_, tools, _, err := s.attach(spec)

	s.Require().NoError(err)
	s.Equal([]string{"crm__whoami"}, names(tools))
}

func (s *AttachConnectorsSuite) TestAFixedBindingNeverUsesAUsersConnection() {
	alice := s.connection("alice", "primary")
	config := s.config(s.chosen("other"))
	// A config save refuses it (api.unboundConnectors); a binding stored before that check
	// existed is refused here too.
	spec := s.spec(config, "alice", nil)
	spec.ConnectorBindings = []store.ConnectorBinding{required(s.fixed("crm", alice, "whoami"))}

	_, _, _, err := s.attach(spec)

	s.ErrorContains(err, "connection_unavailable")
}

// TestARequiredConnectorThatCannotBeUsedFailsTheSessionAndAnOptionalOneIsLeftOut: the
// prototype's TestRequiredConnectorNeedsAReadyAccountWhileOptionalConnectorIsOmitted.
func (s *AttachConnectorsSuite) TestARequiredConnectorThatCannotBeUsedFailsTheSessionAndAnOptionalOneIsLeftOut() {
	app := s.connection("", "primary")
	s.setState(app, func(state *core.CredentialState) { state.Status = store.ConnectionNeedsReauthorization })

	_, _, _, err := s.attach(s.spec(s.config(required(s.fixed("crm", app, "whoami"))), "", nil))
	d, tools, unavailable, optional := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))

	s.ErrorContains(err, `required connector "crm" cannot be used: needs_reauthorization: the provider no longer takes the connection's credential; reconnect it`)
	s.Require().NoError(optional)
	s.Empty(tools)
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableReauthorize}}, unavailable)
	_, err = s.call(d, "crm__whoami", "{}")
	s.Error(err, "nothing is offered under the alias")
	s.Zero(s.provider.calls("primary"))
}

func (s *AttachConnectorsSuite) TestARequiredSessionBindingWithNoSelectionFailsTheSession() {
	_, _, _, err := s.attach(s.spec(s.config(required(s.chosen("crm", "whoami"))), "alice", nil))

	s.ErrorContains(err, `required connector "crm" cannot be used: no_selection`)
}

func (s *AttachConnectorsSuite) TestAToolNotInTheGrantIsNotOffered() {
	app := s.connection("", "primary")

	_, tools, unavailable, err := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))

	s.Require().NoError(err)
	s.Empty(unavailable)
	s.Equal([]string{"crm__whoami"}, names(tools), "secret and slow are the provider's, and not granted")
}

// TestAToolWhoseSchemaChangedIsNotOffered: a grant pins a digest the provider's tool no
// longer has. The required binding fails; the optional one keeps the tools that still match.
func (s *AttachConnectorsSuite) TestAToolWhoseSchemaChangedIsNotOffered() {
	app := s.connection("", "primary")
	stale := s.fixed("crm", app, "whoami", "secret")
	stale.Tools[1].SchemaDigest = "0000000000000000000000000000000000000000000000000000000000000000"

	_, _, _, err := s.attach(s.spec(s.config(required(stale)), "", nil))
	_, tools, unavailable, optional := s.attach(s.spec(s.config(stale), "", nil))

	s.ErrorContains(err, "tool_unavailable")
	s.Require().NoError(optional)
	s.Equal([]string{"crm__whoami"}, names(tools))
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, unavailable)
}

func (s *AttachConnectorsSuite) TestAFixedAliasCannotBeGivenAConnection() {
	app, alice := s.connection("", "primary"), s.connection("alice", "secondary")
	config := s.config(s.fixed("crm", app, "whoami"))

	_, _, _, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))

	s.ErrorContains(err, `connector "crm" has the agent config's fixed connection, which a session cannot replace`)
}

func (s *AttachConnectorsSuite) TestAnAliasTheConfigDoesNotDeclareIsRefused() {
	alice := s.connection("alice", "primary")

	_, _, _, err := s.attach(s.spec(s.config(s.chosen("crm", "whoami")), "alice", map[string]string{"notes": alice}))

	s.ErrorContains(err, `the agent config has no connector binding "notes"`)
}

// TestAForkDropsASelectionItsConfigNoLongerDeclares: the fork re-resolves its parent's
// selections against the config as it is now, keeps what still holds and says what it
// dropped.
func (s *AttachConnectorsSuite) TestAForkDropsASelectionItsConfigNoLongerDeclares() {
	alice := s.connection("alice", "primary")
	spec := s.spec(s.config(s.chosen("crm", "whoami")), "alice", map[string]string{"crm": alice, "notes": alice})
	spec.ForkedFrom = "the parent"

	d, tools, unavailable, err := s.manager.attachConnectors(s.ctx, &spec)

	s.Require().NoError(err)
	defer d.Close()
	s.Equal([]string{"crm__whoami"}, names(tools))
	s.Equal([]ConnectorUnavailable{{Name: "notes", Reason: unavailableDropped}}, unavailable)
	s.Equal([]ConnectorSelection{{Name: "crm", ConnectionID: alice}}, spec.ConnectorSelections,
		"the fork keeps only what it re-resolved")
}

// TestASharedConversationUsesTheAppsConnectionsOnly: the multi-person rule. In a thread
// channel, where everyone in the external thread writes, Alice's own connection is not
// offered even though she chose it; the app's is.
func (s *AttachConnectorsSuite) TestASharedConversationUsesTheAppsConnectionsOnly() {
	app, alice := s.connection("", "primary"), s.connection("alice", "secondary")
	spec := s.spec(s.config(s.fixed("team", app, "whoami"), s.chosen("mine", "whoami")), "alice", map[string]string{"mine": alice})
	spec.ConversationID = "agent:" + persistent.ThreadChannelPrefix + "0199"

	_, tools, unavailable, err := s.attach(spec)

	s.Require().NoError(err)
	s.Equal([]string{"team__whoami"}, names(tools))
	s.Equal([]ConnectorUnavailable{{Name: "mine", ConnectorID: s.connectorID, Reason: unavailableShared}}, unavailable)
}

func (s *AttachConnectorsSuite) TestABindingToAnotherConnectorIsRefused() {
	alice := s.connection("alice", "primary")
	binding := required(s.chosen("crm", "whoami"))
	binding.ConnectorID = "custom_other"
	spec := s.spec(s.config(), "alice", map[string]string{"crm": alice})
	spec.ConnectorBindings = []store.ConnectorBinding{binding}

	_, _, _, err := s.attach(spec)

	s.ErrorContains(err, "provider_mismatch")
}

func (s *AttachConnectorsSuite) TestNoCallerIsNotAVerifiedCaller() {
	alice := s.connection("alice", "primary")
	spec := s.spec(s.config(required(s.chosen("crm", "whoami"))), "", map[string]string{"crm": alice})
	spec.Caller, spec.CallerKind = routing.Caller{}, auth.KindServer

	_, _, _, err := s.attach(spec)

	s.ErrorContains(err, "caller_unverified")
}
