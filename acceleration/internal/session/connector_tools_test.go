//go:build integration

package session

import (
	"bytes"
	"fmt"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
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

// TestASessionBindingWithNoSelectionUsesTheCallersOnlyConnectedConnection is AI-994 (F41 of
// the plugin migration run): a session created without connector_bindings, as an app created
// one for user_plugins, opens on the caller's one connected connection to the connector. Her
// newer connections that are pending, need reauthorization or were disconnected are not
// counted. It is not kept as the session's selection, and the log says which it used.
func (s *AttachConnectorsSuite) TestASessionBindingWithNoSelectionUsesTheCallersOnlyConnectedConnection() {
	alice := s.connection("alice", "primary")
	for _, status := range []string{store.ConnectionPending, store.ConnectionNeedsReauthorization, store.ConnectionDisconnected} {
		other := s.connection("alice", "secondary")
		s.setState(other, func(state *core.CredentialState) { state.Status = status })
	}
	spec := s.spec(s.config(required(s.chosen("crm", "whoami"))), "alice", nil)
	manager := s.managerWith(fixtureRequestTimeout)
	var logged bytes.Buffer
	manager.logger = slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelDebug}))

	d, tools, unavailable, err := manager.attachConnectors(s.ctx, &spec)

	s.Require().NoError(err)
	defer d.Close()
	s.Empty(unavailable)
	s.Equal([]string{"crm__whoami"}, names(tools))
	said, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Empty(spec.ConnectorSelections, "implied, not kept as the session's choice")
	s.Contains(logged.String(), "connection="+alice)
	s.Zero(s.provider.calls("secondary"))
}

// TestNoSelectionImpliesNothingWithNoneOrTwoConnected: with none connected, a connection that
// needs reauthorization only, or two connected, the session is opened as before: without it.
func (s *AttachConnectorsSuite) TestNoSelectionImpliesNothingWithNoneOrTwoConnected() {
	config := s.config(required(s.chosen("crm", "whoami")))
	_, _, _, none := s.attach(s.spec(config, "alice", nil))
	stale := s.connection("alice", "primary")
	s.setState(stale, func(state *core.CredentialState) { state.Status = store.ConnectionNeedsReauthorization })
	_, _, _, onlyStale := s.attach(s.spec(config, "alice", nil))
	s.connection("alice", "primary")
	s.connection("alice", "secondary")
	_, _, _, two := s.attach(s.spec(config, "alice", nil))

	for name, err := range map[string]error{"none": none, "only one needing reauthorization": onlyStale, "two": two} {
		s.ErrorContains(err, `required connector "crm" cannot be used: no_selection`, name)
	}
	s.Empty(s.provider.sent("primary"))
	s.Empty(s.provider.sent("secondary"))
}

// TestNoSelectionNeverImpliesAnotherUsersOrAnotherCustomersConnection: Bob's connection, and
// the one connection of a user of the same id at another customer, are not hers here. Her id
// is the test's own, so that one is the only connection under it anywhere.
func (s *AttachConnectorsSuite) TestNoSelectionNeverImpliesAnotherUsersOrAnotherCustomersConnection() {
	alice := "alice-" + uuid.NewString()
	s.connection("bob", "secondary")
	ours, revision := s.customerID, s.revision
	s.SetupTest()
	s.connection(alice, "moved")
	s.customerID, s.revision = ours, revision

	_, _, _, err := s.attach(s.spec(s.config(required(s.chosen("crm", "whoami"))), alice, nil))

	s.ErrorContains(err, `required connector "crm" cannot be used: no_selection`)
	s.Empty(s.provider.sent("secondary"))
	s.Empty(s.provider.sent("moved"))
}

// TestNoSelectionImpliesNothingForAnUnverifiedCallerOrASharedConversation: Alice's one
// connected connection is not used by a guest or an anonymous caller going by her name, nor in
// a conversation more than one person writes in.
func (s *AttachConnectorsSuite) TestNoSelectionImpliesNothingForAnUnverifiedCallerOrASharedConversation() {
	s.connection("alice", "primary")
	config := s.config(s.chosen("crm", "whoami"))
	shared := s.spec(config, "alice", nil)
	shared.ConversationID = "agent:" + persistent.ThreadChannelPrefix + "0199"
	shared.Thread = true
	guest, anonymous := s.spec(config, "alice", nil), s.spec(config, "alice", nil)
	guest.CallerKind, anonymous.CallerKind = auth.KindGuest, auth.KindAnonymous

	for name, c := range map[string]struct {
		spec   Spec
		reason string
	}{"shared": {shared, unavailableShared}, "guest": {guest, unavailableUnverified}, "anonymous": {anonymous, unavailableUnverified}} {
		_, tools, unavailable, err := s.attach(c.spec)

		s.Require().NoError(err, name)
		s.Empty(tools, name)
		s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: c.reason}}, unavailable, name)
	}
	s.Empty(s.provider.sent("primary"))
}

// TestAConnectionNamedAtSessionCreateWinsOverTheImpliedOne: one named is checked as before,
// and the caller's one connected connection is not used in its place.
func (s *AttachConnectorsSuite) TestAConnectionNamedAtSessionCreateWinsOverTheImpliedOne() {
	s.connection("alice", "primary")
	stale, bob := s.connection("alice", "secondary"), s.connection("bob", "moved")
	s.setState(stale, func(state *core.CredentialState) { state.Status = store.ConnectionNeedsReauthorization })
	config := s.config(required(s.chosen("crm", "whoami")))

	_, _, _, staleErr := s.attach(s.spec(config, "alice", map[string]string{"crm": stale}))
	_, _, _, bobErr := s.attach(s.spec(config, "alice", map[string]string{"crm": bob}))

	s.ErrorContains(staleErr, `required connector "crm" cannot be used: needs_reauthorization`)
	s.ErrorContains(bobErr, `required connector "crm" cannot be used: connection_unavailable`)
	s.Empty(s.provider.sent("primary"), "the implied connection is not used in its place")
}

// TestWithConnectorsOffNoSelectionImpliesNothing: a deployment with no connection clients
// leaves an optional binding out with no_selection, as before, not open_failed.
func (s *AttachConnectorsSuite) TestWithConnectorsOffNoSelectionImpliesNothing() {
	s.connection("alice", "primary")
	spec := s.spec(s.config(s.chosen("crm", "whoami")), "alice", nil)
	logger := slog.New(slog.DiscardHandler)
	manager := &Manager{logger: logger, options: ManagerOptions{Store: s.store, Logger: logger}}

	_, tools, unavailable, err := manager.attachConnectors(s.ctx, &spec)

	s.Require().NoError(err)
	s.Empty(tools)
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableNoSelection}}, unavailable)
}

func (s *AttachConnectorsSuite) TestAToolNotInTheGrantIsNotOffered() {
	app := s.connection("", "primary")

	d, tools, unavailable, err := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
	s.Require().NoError(err)
	_, ungranted := s.call(d, "crm__secret", "{}")

	s.Empty(unavailable)
	s.Equal([]string{"crm__whoami"}, names(tools), "secret and slow are the provider's, and not granted")
	s.ErrorContains(ungranted, "is not a tool this session can run", "a tool never granted is not one left out")
}

// TestAToolWhoseSchemaChangedIsNotOffered: a grant names a digest the provider's tool no
// longer has. The binding keeps the tools that still match, required or not, and says so in
// unavailable; a call of the one left out says to save the config at the new schema, since a
// schema the config names is no pin to approve (Athena's case).
func (s *AttachConnectorsSuite) TestAToolWhoseSchemaChangedIsNotOffered() {
	app := s.connection("", "primary")
	stale := s.fixed("crm", app, "secret", "whoami")
	stale.Tools[1].SchemaDigest = staleDigest

	for _, binding := range []store.ConnectorBinding{stale, required(stale)} {
		d, tools, unavailable, err := s.attach(s.spec(s.config(binding), "", nil))
		s.Require().NoError(err, "required: %v", binding.Required)
		drifted, err := s.call(d, "crm__whoami", "{}")
		s.Require().NoError(err)
		kept, err := s.call(d, "crm__secret", "{}")
		s.Require().NoError(err)

		s.Equal([]string{"crm__secret"}, names(tools))
		s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, unavailable)
		s.JSONEq(configWhoami, drifted)
		s.Equal("the secret", kept)
	}
}

// TestABindingWhoseEveryToolChangedOpensOnlyWhenOptional: with no tool left to offer a required
// binding fails with tool_unavailable, as on base; an optional one opens empty and says so.
func (s *AttachConnectorsSuite) TestABindingWhoseEveryToolChangedOpensOnlyWhenOptional() {
	app := s.connection("", "primary")
	all := s.fixed("crm", app, "whoami", "secret")
	all.Tools[0].SchemaDigest = staleDigest
	all.Tools[1].SchemaDigest = staleDigest

	_, _, _, failed := s.attach(s.spec(s.config(required(all)), "", nil))
	_, tools, unavailable, err := s.attach(s.spec(s.config(all), "", nil))

	s.ErrorContains(failed, `required connector "crm" cannot be used: tool_unavailable`)
	s.Require().NoError(err)
	s.Empty(tools)
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, unavailable)
}

// TestAGrantedToolTheProviderNoLongerListsIsNotLeftOut: gone is granted and the provider lists
// no such tool, so no schema changed: a call of it is a tool this session cannot run, as on
// base, and a required binding granting it still fails.
func (s *AttachConnectorsSuite) TestAGrantedToolTheProviderNoLongerListsIsNotLeftOut() {
	app := s.connection("", "primary")
	binding := s.fixed("crm", app, "secret")
	binding.Tools = append(binding.Tools, store.ToolGrant{Name: "gone", SchemaDigest: staleDigest})

	d, tools, unavailable, err := s.attach(s.spec(s.config(binding), "", nil))
	s.Require().NoError(err)
	_, gone := s.call(d, "crm__gone", "{}")
	_, _, _, failed := s.attach(s.spec(s.config(required(binding)), "", nil))

	s.Equal([]string{"crm__secret"}, names(tools))
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, unavailable)
	s.ErrorContains(gone, "is not a tool this session can run")
	s.ErrorContains(failed, `required connector "crm" cannot be used: tool_unavailable`)
}

// TestAGrantedToolTheConnectorLeavesOutSaysSo (AI-1394): whoami needs a scope the manifest's
// scopes.list never asks for, so the connector hides it and a required binding that grants it
// fails with a reason that names that, not a provider that stopped offering it.
func (s *AttachConnectorsSuite) TestAGrantedToolTheConnectorLeavesOutSaysSo() {
	s.connectorID = "custom_crm_scoped"
	scoped := strings.Replace(s.crmManifest(s.connectorID, 1), "sources:",
		"scopes:\n  list: [read]\nsources:", 1) + "    tools:\n      - name: whoami\n        needs_scopes: [write]\n"
	manifest, err := core.ParseManifest([]byte(scoped))
	s.Require().NoError(err)
	definition, err := s.store.CreateConnectorDefinition(s.ctx, s.customerID, manifest)
	s.Require().NoError(err)
	s.revision = definition.Revision
	app := s.connection("", "primary")

	_, _, _, err = s.attach(s.spec(s.config(required(s.fixed("crm", app, "whoami"))), "", nil))
	d, _, _, optional := s.attach(s.spec(s.config(s.fixed("crm", app, "whoami", "secret")), "", nil))
	s.Require().NoError(optional)
	_, hidden := s.call(d, "crm__whoami", "{}")

	s.ErrorContains(err, "tool_unavailable")
	s.ErrorContains(err, "needs a scope the connector never asks for")
	s.ErrorContains(hidden, "is not a tool this session can run", "no schema changed, so nothing says an admin must approve one")
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
	spec.Thread = true

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

// TestAToolGrantedByNameIsPinnedOnFirstUse: a session binding's grant of whoami by name. The
// first session on Alice's connection pins the digest it finds; the next one offers whoami on
// that pin and writes nothing.
func (s *AttachConnectorsSuite) TestAToolGrantedByNameIsPinnedOnFirstUse() {
	alice := s.connection("alice", "primary")
	config := s.config(byName(s.chosen("crm", "whoami")))

	d, first, unavailable, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)
	pinned := s.pinOf(alice, "whoami")
	_, again, _, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)

	s.Empty(unavailable)
	s.Equal([]string{"crm__whoami"}, names(first))
	s.Equal([]string{"crm__whoami"}, names(again))
	s.Equal(grants("whoami")[0].SchemaDigest, pinned.SchemaDigest)
	s.True(pinned.PinnedAt.Equal(s.pinOf(alice, "whoami").PinnedAt), "pinned once")
	said, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)
	s.Equal("primary", said)
}

// TestAToolGrantedByNameWhoseSchemaChangedSinceItsPinIsNotOffered: the provider changes
// whoami after the first session pinned it. The next session leaves it out as it leaves out a
// tool whose granted digest no longer matches, and the pin stays.
func (s *AttachConnectorsSuite) TestAToolGrantedByNameWhoseSchemaChangedSinceItsPinIsNotOffered() {
	alice := s.connection("alice", "primary")
	config := s.config(byName(s.chosen("crm", "whoami", "secret")))
	_, _, _, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)
	s.provider.redescribe("primary", "Says which account this is, and now something else too.")

	_, tools, unavailable, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))

	s.Require().NoError(err)
	s.Equal([]string{"crm__secret"}, names(tools))
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, unavailable)
	s.Equal(grants("whoami")[0].SchemaDigest, s.pinOf(alice, "whoami").SchemaDigest)
}

// TestADriftedPinSaysAnAdminMustApproveIt: whoami's schema changed after its pin, so a session
// leaves it out and a call of it says an admin must approve it. That the approve
// (POST .../tools/approve) gives it back to the next session is
// api.SessionConnectorsSuite.TestAToolApprovedAgainIsOfferedToTheNextSession.
func (s *AttachConnectorsSuite) TestADriftedPinSaysAnAdminMustApproveIt() {
	alice := s.connection("alice", "primary")
	config := s.config(byName(s.chosen("crm", "whoami", "secret")))
	spec := s.spec(config, "alice", map[string]string{"crm": alice})
	_, _, _, err := s.attach(spec)
	s.Require().NoError(err)
	s.provider.redescribe("primary", "Says which account this is, and now something else too.")

	d, tools, _, err := s.attach(spec)
	s.Require().NoError(err)
	refused, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.JSONEq(driftedWhoami, refused)
	s.Equal([]string{"crm__secret"}, names(tools))
}

// TestAReconnectPinsAToolGrantedByNameAgain: a consent connects Alice's connection anew, a
// new trust event, so the next session pins whoami as the provider lists it now.
func (s *AttachConnectorsSuite) TestAReconnectPinsAToolGrantedByNameAgain() {
	alice := s.connection("alice", "primary")
	config := s.config(byName(s.chosen("crm", "whoami")))
	_, _, _, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)
	s.provider.redescribe("primary", "Says which account this is, as changed.")
	s.setState(alice, func(state *core.CredentialState) { state.ConnectedAt = time.Now().UTC().Add(time.Second) })

	_, tools, unavailable, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))

	s.Require().NoError(err)
	s.Empty(unavailable)
	s.Equal([]string{"crm__whoami"}, names(tools))
	s.Equal("Says which account this is, as changed.", tools[0].Description)
	s.NotEqual(grants("whoami")[0].SchemaDigest, s.pinOf(alice, "whoami").SchemaDigest)
}

// TestAReconnectPinsAPatternMatchedToolAgain: a consent connects Alice's connection anew, a
// new trust event, so the next session pins the pattern's whoami as the provider lists it now.
func (s *AttachConnectorsSuite) TestAReconnectPinsAPatternMatchedToolAgain() {
	alice := s.connection("alice", "primary")
	config := s.config(withPatterns(s.chosen("crm"), "whoami"))
	_, _, _, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)
	s.provider.redescribe("primary", "Says which account this is, as changed.")
	s.setState(alice, func(state *core.CredentialState) { state.ConnectedAt = time.Now().UTC().Add(time.Second) })

	_, tools, unavailable, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))

	s.Require().NoError(err)
	s.Empty(unavailable)
	s.Equal([]string{"crm__whoami"}, names(tools))
	s.Equal("Says which account this is, as changed.", tools[0].Description)
	s.NotEqual(grants("whoami")[0].SchemaDigest, s.pinOf(alice, "whoami").SchemaDigest)
}

// TestEachConnectionIsPinnedOnItsOwn: Alice's pin is not Bob's. Bob's provider lists whoami
// with another description, which his first session pins for his connection.
func (s *AttachConnectorsSuite) TestEachConnectionIsPinnedOnItsOwn() {
	alice, bob := s.connection("alice", "primary"), s.connection("bob", "secondary")
	config := s.config(byName(s.chosen("crm", "whoami")))
	s.provider.redescribe("secondary", "Says which account this is, for Bob.")

	_, alices, _, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)
	_, bobs, unavailable, err := s.attach(s.spec(config, "bob", map[string]string{"crm": bob}))
	s.Require().NoError(err)

	s.Empty(unavailable)
	s.Equal([]string{"crm__whoami"}, names(alices))
	s.Equal([]string{"crm__whoami"}, names(bobs))
	s.NotEqual(s.pinOf(alice, "whoami").SchemaDigest, s.pinOf(bob, "whoami").SchemaDigest)
}

// TestAFixedBindingNeverPinsAToolGrantedByName: the API refuses such a grant
// (api.connectorBindingsComplaint); one stored anyway offers nothing and pins nothing.
func (s *AttachConnectorsSuite) TestAFixedBindingNeverPinsAToolGrantedByName() {
	app := s.connection("", "primary")

	_, tools, unavailable, err := s.attach(s.spec(s.config(byName(s.fixed("crm", app, "whoami"))), "", nil))

	s.Require().NoError(err)
	s.Empty(tools)
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, unavailable)
	var pins int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM connector_tool_pins WHERE connection_id = ?", app).Scan(&pins))
	s.Zero(pins)
}

// TestAGrantWithADigestOpensAsBeforeAndPinsNothing is the control: every expected value is
// what the same calls gave on accelerate before tools could be granted by name (89966e26):
// whoami offered, nothing unavailable, server/discover, tools/list and tools/call sent, and a
// config with no bindings opening no dispatcher.
func (s *AttachConnectorsSuite) TestAGrantWithADigestOpensAsBeforeAndPinsNothing() {
	alice := s.connection("alice", "primary")

	d, tools, unavailable, err := s.attach(s.spec(s.config(s.chosen("crm", "whoami")), "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)
	said, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)
	none, noTools, noneUnavailable, err := s.attach(s.spec(s.config(), "alice", nil))
	s.Require().NoError(err)

	s.Equal([]string{"crm__whoami"}, names(tools))
	s.Empty(unavailable)
	s.Equal("primary", said)
	s.Equal([]string{"server/discover", "tools/list", "tools/call"}, s.provider.sent("primary"))
	s.Nil(none)
	s.Empty(noTools)
	s.Empty(noneUnavailable)
	var pins int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM connector_tool_pins WHERE connection_id = ?", alice).Scan(&pins))
	s.Zero(pins)
}

// TestAStarPatternOffersEveryListedTool: tool_patterns ["*"] offers every tool the connection
// lists, the plugins rule that an allowlist of * offers each one (plugins.Offered).
func (s *AttachConnectorsSuite) TestAStarPatternOffersEveryListedTool() {
	alice := s.connection("alice", "primary")
	config := s.config(withPatterns(s.chosen("crm"), "*"))

	_, tools, unavailable, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))

	s.Require().NoError(err)
	s.Empty(unavailable)
	var want []string
	for name := range providerTools {
		want = append(want, "crm__"+name)
		s.Equal(grants(name)[0].SchemaDigest, s.pinOf(alice, name).SchemaDigest, name)
	}
	s.ElementsMatch(want, names(tools))
}

// TestAToolAddedLaterIsPinnedAndOfferedOnTheNextSession: a tool the server gains after the
// first session is pinned at the digest listed then, and the next session offers it.
func (s *AttachConnectorsSuite) TestAToolAddedLaterIsPinnedAndOfferedOnTheNextSession() {
	alice := s.connection("alice", "primary")
	config := s.config(withPatterns(s.chosen("crm"), "*"))
	_, first, _, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)
	s.NotContains(names(first), "crm__later")
	s.provider.addTool("primary", "later", "Added after the first session.")

	_, second, unavailable, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))

	s.Require().NoError(err)
	s.Empty(unavailable)
	s.Contains(names(second), "crm__later")
	s.NotEmpty(s.pinOf(alice, "later").SchemaDigest)
	s.Equal(grants("whoami")[0].SchemaDigest, s.pinOf(alice, "whoami").SchemaDigest)
}

// TestAPatternMatchedToolWhoseSchemaChangedIsNotOffered: whoami's description changes after
// its pin. The next session leaves it out and warns once; the pin stays, and the tools whose
// schemas did not move are still offered.
func (s *AttachConnectorsSuite) TestAPatternMatchedToolWhoseSchemaChangedIsNotOffered() {
	alice := s.connection("alice", "primary")
	config := s.config(withPatterns(s.chosen("crm"), "*"))
	_, _, _, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)
	s.provider.redescribe("primary", "Says which account this is, and now something else too.")
	manager := s.managerWith(fixtureRequestTimeout)
	var logged bytes.Buffer
	manager.logger = slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelWarn}))
	spec := s.spec(config, "alice", map[string]string{"crm": alice})

	d, tools, unavailable, err := manager.attachConnectors(s.ctx, &spec)

	s.Require().NoError(err)
	s.T().Cleanup(d.Close)
	s.NotContains(names(tools), "crm__whoami")
	s.Contains(names(tools), "crm__secret")
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, unavailable)
	s.Equal(grants("whoami")[0].SchemaDigest, s.pinOf(alice, "whoami").SchemaDigest)
	s.Equal(1, strings.Count(logged.String(), "schema it was pinned at"))
	drifted, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)
	s.JSONEq(driftedWhoami, drifted)
}

// TestARequiredPatternBindingKeepsItsOtherToolsWhenOneChanged: a binding granting a pattern
// lists no tool by name, so a required one with a single drifted match opens with the rest and
// says so in unavailable, as an optional one does.
func (s *AttachConnectorsSuite) TestARequiredPatternBindingKeepsItsOtherToolsWhenOneChanged() {
	alice := s.connection("alice", "primary")
	app := s.connection("", "primary")
	chosen, fixed := withPatterns(s.chosen("crm"), "*"), withPatterns(s.fixed("crm", app), "*")
	selections := func(binding store.ConnectorBinding) map[string]string {
		if binding.Connection.Type == selectionFixed {
			return nil
		}
		return map[string]string{"crm": alice}
	}
	for _, binding := range []store.ConnectorBinding{chosen, fixed} {
		_, _, _, err := s.attach(s.spec(s.config(binding), "alice", selections(binding)))
		s.Require().NoError(err, binding.Connection.Type)
	}
	s.provider.redescribe("primary", "Says which account this is, and now something else too.")

	for _, binding := range []store.ConnectorBinding{chosen, fixed} {
		selection := selections(binding)
		_, tools, unavailable, err := s.attach(s.spec(s.config(required(binding)), "alice", selection))

		s.Require().NoError(err, binding.Connection.Type)
		s.NotContains(names(tools), "crm__whoami", binding.Connection.Type)
		s.Contains(names(tools), "crm__secret", binding.Connection.Type)
		s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, unavailable, binding.Connection.Type)
	}
}

// TestAPatternBindingWhoseEveryMatchChangedOpensOnlyWhenOptional: with no match left to offer
// a required pattern binding fails with tool_unavailable; an optional one opens empty.
func (s *AttachConnectorsSuite) TestAPatternBindingWhoseEveryMatchChangedOpensOnlyWhenOptional() {
	alice := s.connection("alice", "primary")
	config := s.config(withPatterns(s.chosen("crm"), "whoami"))
	_, _, _, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)
	s.provider.redescribe("primary", "Says which account this is, and now something else too.")
	chosen := map[string]string{"crm": alice}

	// Another binding's tools are no tools of this one: the required binding still fails.
	chosen["other"] = alice
	_, _, _, failed := s.attach(s.spec(s.config(withPatterns(s.chosen("other"), "secret"),
		required(withPatterns(s.chosen("crm"), "whoami"))), "alice", chosen))
	_, tools, unavailable, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))

	s.ErrorContains(failed, `required connector "crm" cannot be used: tool_unavailable`)
	s.Require().NoError(err)
	s.Empty(tools)
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, unavailable)
}

// TestARequiredPinnedBindingFailsOnlyWhenEveryToolChanged: a tool granted by name and pinned in
// the connection opens a required binding with the rest when one drifted, and fails it when
// all did.
func (s *AttachConnectorsSuite) TestARequiredPinnedBindingFailsOnlyWhenEveryToolChanged() {
	alice := s.connection("alice", "primary")
	chosen := map[string]string{"crm": alice}
	_, _, _, err := s.attach(s.spec(s.config(byName(s.chosen("crm", "whoami", "secret"))), "alice", chosen))
	s.Require().NoError(err)
	s.provider.redescribe("primary", "Says which account this is, and now something else too.")

	_, tools, unavailable, err := s.attach(s.spec(s.config(required(byName(s.chosen("crm", "whoami", "secret")))), "alice", chosen))
	s.Require().NoError(err)
	s.Equal([]string{"crm__secret"}, names(tools))
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, unavailable)

	_, _, _, failed := s.attach(s.spec(s.config(required(byName(s.chosen("crm", "whoami")))), "alice", chosen))
	_, noTools, noneUnavailable, err := s.attach(s.spec(s.config(byName(s.chosen("crm", "whoami"))), "alice", chosen))
	s.ErrorContains(failed, `required connector "crm" cannot be used: tool_unavailable`)
	s.Require().NoError(err)
	s.Empty(noTools)
	s.Equal([]ConnectorUnavailable{{Name: "crm", ConnectorID: s.connectorID, Reason: unavailableTool}}, noneUnavailable)
}

// TestPatternPinsArePerConnection: Alice's pin is not Bob's. His provider lists whoami with
// another description, which his first session pins for his connection.
func (s *AttachConnectorsSuite) TestPatternPinsArePerConnection() {
	alice, bob := s.connection("alice", "primary"), s.connection("bob", "secondary")
	config := s.config(withPatterns(s.chosen("crm"), "whoami"))
	s.provider.redescribe("secondary", "Says which account this is, for Bob.")

	_, alices, _, err := s.attach(s.spec(config, "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)
	_, bobs, unavailable, err := s.attach(s.spec(config, "bob", map[string]string{"crm": bob}))
	s.Require().NoError(err)

	s.Empty(unavailable)
	s.Equal([]string{"crm__whoami"}, names(alices))
	s.Equal([]string{"crm__whoami"}, names(bobs))
	s.NotEqual(s.pinOf(alice, "whoami").SchemaDigest, s.pinOf(bob, "whoami").SchemaDigest)
}

// TestAFixedPatternBindingPinsOnFirstUse: a fixed binding with patterns and no digests pins
// the app connection's tools on first use and offers them.
func (s *AttachConnectorsSuite) TestAFixedPatternBindingPinsOnFirstUse() {
	app := s.connection("", "primary")

	_, tools, unavailable, err := s.attach(s.spec(s.config(withPatterns(s.fixed("crm", app), "*")), "", nil))

	s.Require().NoError(err)
	s.Empty(unavailable)
	s.Contains(names(tools), "crm__whoami")
	s.Equal(grants("whoami")[0].SchemaDigest, s.pinOf(app, "whoami").SchemaDigest)
}

// TestMoreThan128PatternMatchesOffersTheFirst128ByName: 129 matches, so the session offers
// the first 128 by name and warns once. 128 is patternToolCap, Tools' maxItems.
func (s *AttachConnectorsSuite) TestMoreThan128PatternMatchesOffersTheFirst128ByName() {
	alice := s.connection("alice", "primary")
	for index := range 129 {
		s.provider.addTool("primary", fmt.Sprintf("n%03d", index), "One of too many.")
	}
	manager := s.managerWith(fixtureRequestTimeout)
	var logged bytes.Buffer
	manager.logger = slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelWarn}))
	spec := s.spec(s.config(withPatterns(s.chosen("crm"), "n*")), "alice", map[string]string{"crm": alice})

	d, tools, unavailable, err := manager.attachConnectors(s.ctx, &spec)

	s.Require().NoError(err)
	s.T().Cleanup(d.Close)
	s.Empty(unavailable)
	offered := names(tools)
	s.Len(offered, 128)
	for index := range 128 {
		s.Contains(offered, "crm__"+fmt.Sprintf("n%03d", index))
	}
	s.NotContains(offered, "crm__n128")
	s.Equal(1, strings.Count(logged.String(), "first 128 by name"))
	s.Equal(1, strings.Count(logged.String(), "level=WARN"))
}

// TestABindingWithOnlyToolsOpensAsOnBase is the control. On e70351d4, before tool_patterns,
// TestAGrantWithADigestOpensAsBeforeAndPinsNothing passed against model_router_test_w9_pat_test
// (2026-10-12): whoami offered, nothing unavailable, the call answered "primary", the server
// was sent server/discover then tools/list then tools/call, a config with no bindings opened
// no dispatcher, and the connection had no pins.
func (s *AttachConnectorsSuite) TestABindingWithOnlyToolsOpensAsOnBase() {
	alice := s.connection("alice", "primary")

	d, tools, unavailable, err := s.attach(s.spec(s.config(s.chosen("crm", "whoami")), "alice", map[string]string{"crm": alice}))
	s.Require().NoError(err)
	said, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)
	none, noTools, noneUnavailable, err := s.attach(s.spec(s.config(), "alice", nil))
	s.Require().NoError(err)

	s.Equal([]string{"crm__whoami"}, names(tools))
	s.Empty(unavailable)
	s.Equal("primary", said)
	s.Equal([]string{"server/discover", "tools/list", "tools/call"}, s.provider.sent("primary"))
	s.Nil(none)
	s.Empty(noTools)
	s.Empty(noneUnavailable)
	var pins int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM connector_tool_pins WHERE connection_id = ?", alice).Scan(&pins))
	s.Zero(pins)
}

// TestABindingWhoseToolsAllMatchWarnsNothing: every granted tool is listed at its digest, so
// the binding opens whole, leaves nothing out and warns about nothing.
func (s *AttachConnectorsSuite) TestABindingWhoseToolsAllMatchWarnsNothing() {
	alice := s.connection("alice", "primary")
	manager := s.managerWith(fixtureRequestTimeout)
	var logged bytes.Buffer
	manager.logger = slog.New(slog.NewTextHandler(&logged, &slog.HandlerOptions{Level: slog.LevelWarn}))
	spec := s.spec(s.config(s.chosen("crm", "whoami", "secret")), "alice", map[string]string{"crm": alice})

	d, tools, unavailable, err := manager.attachConnectors(s.ctx, &spec)

	s.Require().NoError(err)
	s.T().Cleanup(d.Close)
	s.ElementsMatch([]string{"crm__whoami", "crm__secret"}, names(tools))
	s.Empty(unavailable)
	s.Empty(logged.String())
}

// pinOf is connection's pin of tool, as stored.
func (s *AttachConnectorsSuite) pinOf(connection, tool string) store.ConnectorToolPin {
	var pin store.ConnectorToolPin
	s.Require().NoError(s.store.DB().NewSelect().Model(&pin).
		Where("connection_id = ?", connection).Where("tool_name = ?", tool).Scan(s.ctx))
	return pin
}
