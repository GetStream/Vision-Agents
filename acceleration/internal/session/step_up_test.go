//go:build integration

package session

import (
	"context"
	"errors"
	"log/slog"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// StepUpSuite is what a session's connector call does when the provider refuses it for want
// of a scope (AI-854): which connections a step-up is asked for, how often, and that the
// grant stays as it was. The consent itself is the router's (api.ConnectorConsents), run end
// to end in the api package's ChatLoginsSuite; here a session is handed the consent its
// router would have begun, on a real attempt row.
type StepUpSuite struct {
	connectorFixture
	// asked are the consents the router was asked to begin, and events what the session's
	// watchers were sent.
	asked  []ConsentRequest
	events []Event
	// refuse makes the router fail to begin a consent.
	refuse bool
}

func TestStepUpSuite(t *testing.T) {
	suite.Run(t, new(StepUpSuite))
}

func (s *StepUpSuite) SetupTest() {
	s.connectorFixture.SetupTest()
	s.asked, s.events, s.refuse = nil, nil, false
}

// TestAScopeChallengeOnTheCallersOwnConnectionAsksOnceForAStepUp: the provider refuses every
// call for want of crm:write. The first call begins one step-up and sends one event; the
// second in the same reply is answered the same with no second attempt and no second event.
// The model reads neither the launch URL nor the handoff token, and the connection stays
// connected.
func (s *StepUpSuite) TestAScopeChallengeOnTheCallersOwnConnectionAsksOnceForAStepUp() {
	mine := s.connection("alice", "primary")
	d := s.attachWithStepUps(s.spec(s.config(s.chosen("crm", "guarded")), "alice", map[string]string{"crm": mine}))

	first, err := s.guarded(d, "turn-1", `{"scope":"crm:write"}`)
	s.Require().NoError(err)
	second, err := s.guarded(d, "turn-1", `{"scope":"crm:write"}`)
	s.Require().NoError(err)

	s.Contains(first, `"status":"scope_required"`)
	s.Equal(first, second)
	s.Require().Len(s.asked, 1, "one step-up")
	s.Equal(ConsentRequest{CustomerID: s.customerID, ConnectorID: s.connectorID, UserID: "alice", ConnectionID: mine,
		StepUp: &core.Outcome{Kind: core.OutcomeScopeRequired, Scopes: []string{"crm:write"}}}, s.asked[0])
	s.Require().Len(s.events, 1, "one event")
	event, ok := s.events[0].(ConnectorScopeRequired)
	s.Require().True(ok)
	s.Equal("crm", event.Name)
	s.Equal(mine, event.ConnectionID)
	s.Equal([]string{"crm:write"}, event.Scopes)
	s.NotContains(first, event.LaunchURL)
	s.NotContains(first, event.HandoffToken)
	s.Equal(2, s.provider.calls("primary"), "each call reached the provider once, none was sent again")
	s.Equal(store.ConnectionConnected, s.status(mine), "the old grant keeps working")
}

// TestAStepUpThatWasUsedIsAskedForAgain: once the person finished (or failed) the step-up its
// attempt is used, so the next refusal begins a new one rather than show a dead button.
func (s *StepUpSuite) TestAStepUpThatWasUsedIsAskedForAgain() {
	mine := s.connection("alice", "primary")
	d := s.attachWithStepUps(s.spec(s.config(s.chosen("crm", "guarded")), "alice", map[string]string{"crm": mine}))
	_, err := s.guarded(d, "turn-1", `{"scope":"crm:write"}`)
	s.Require().NoError(err)
	s.Require().Len(s.events, 1)
	_, err = s.store.ConsumeConnectorAuthorizationAttempt(s.ctx, s.events[0].(ConnectorScopeRequired).AuthorizationID)
	s.Require().NoError(err)

	_, err = s.guarded(d, "turn-1", `{"scope":"crm:write"}`)

	s.Require().NoError(err)
	s.Len(s.asked, 2)
	s.Len(s.events, 2)
}

// TestAChallengeForMoreThanTheOpenStepUpAsksForIsAskedForAgain: a step-up for crm:write does
// not grant crm:admin, so a call that needs crm:admin begins its own.
func (s *StepUpSuite) TestAChallengeForMoreThanTheOpenStepUpAsksForIsAskedForAgain() {
	mine := s.connection("alice", "primary")
	d := s.attachWithStepUps(s.spec(s.config(s.chosen("crm", "guarded")), "alice", map[string]string{"crm": mine}))
	_, err := s.guarded(d, "turn-1", `{"scope":"crm:write"}`)
	s.Require().NoError(err)

	_, err = s.guarded(d, "turn-1", `{"scope":"crm:admin"}`)

	s.Require().NoError(err)
	s.Require().Len(s.asked, 2)
	s.Equal([]string{"crm:admin"}, s.asked[1].StepUp.Scopes)
	s.Len(s.events, 2)
}

// TestAClaimsChallengeIsNotCoveredByAnOpenScopeStepUp: a claims challenge asks for no scope,
// yet a step-up for crm:write does not satisfy it, so it begins its own with its claims.
func (s *StepUpSuite) TestAClaimsChallengeIsNotCoveredByAnOpenScopeStepUp() {
	mine := s.connection("alice", "primary")
	d := s.attachWithStepUps(s.spec(s.config(s.chosen("crm", "guarded")), "alice", map[string]string{"crm": mine}))
	_, err := s.guarded(d, "turn-1", `{"scope":"crm:write"}`)
	s.Require().NoError(err)

	// eyJhY3IiOiJtZmEifQ== is base64 of {"acr":"mfa"}.
	read, err := s.guarded(d, "turn-1", `{"claims":"eyJhY3IiOiJtZmEifQ=="}`)

	s.Require().NoError(err)
	s.Contains(read, `"status":"scope_required"`)
	s.Require().Len(s.asked, 2)
	s.Equal(&core.Outcome{Kind: core.OutcomeScopeRequired, Claims: `{"acr":"mfa"}`}, s.asked[1].StepUp)
	s.Len(s.events, 2)
}

// TestAStepUpShownOnAnEarlierReplyIsAskedForAgain: the person pressed the button of the first
// step-up and closed the popup. Its attempt stays open, but its launch page cannot be handed
// off again, so the next reply's refused call begins a new step-up the person can open.
func (s *StepUpSuite) TestAStepUpShownOnAnEarlierReplyIsAskedForAgain() {
	mine := s.connection("alice", "primary")
	d := s.attachWithStepUps(s.spec(s.config(s.chosen("crm", "guarded")), "alice", map[string]string{"crm": mine}))
	_, err := s.guarded(d, "turn-1", `{"scope":"crm:write"}`)
	s.Require().NoError(err)
	s.Require().Len(s.events, 1)
	first := s.events[0].(ConnectorScopeRequired).AuthorizationID
	// What handOffConnectorLaunch does: the sealed blob is replaced, and the row stays open.
	s.Require().NoError(s.store.HandOffConnectorAuthorizationAttempt(s.ctx, first, []byte("sealed"), []byte("handed-off"), 1))

	read, err := s.guarded(d, "turn-2", `{"scope":"crm:write"}`)

	s.Require().NoError(err)
	s.Contains(read, `"status":"scope_required"`)
	s.Len(s.asked, 2)
	s.Require().Len(s.events, 2, "a new step-up the person can open")
	s.NotEqual(first, s.events[1].(ConnectorScopeRequired).AuthorizationID)
	s.Equal(store.ConnectionConnected, s.status(mine), "the old grant keeps working")
}

// TestACallInNoReplyReusesNoStepUp: a call that names no reply cannot be told to be in the
// same one, so each refused call begins its own step-up.
func (s *StepUpSuite) TestACallInNoReplyReusesNoStepUp() {
	mine := s.connection("alice", "primary")
	d := s.attachWithStepUps(s.spec(s.config(s.chosen("crm", "guarded")), "alice", map[string]string{"crm": mine}))

	_, err := s.guarded(d, "", `{"scope":"crm:write"}`)
	s.Require().NoError(err)
	_, err = s.guarded(d, "", `{"scope":"crm:write"}`)

	s.Require().NoError(err)
	s.Len(s.asked, 2)
	s.Len(s.events, 2)
}

// TestTheAppsConnectionIsNotSteppedUpByTheSessionsCaller: the app's connection is its
// backend's to reconnect, so the call fails as it did before and nobody is asked.
func (s *StepUpSuite) TestTheAppsConnectionIsNotSteppedUpByTheSessionsCaller() {
	app := s.connection("", "primary")
	d := s.attachWithStepUps(s.spec(s.config(s.fixed("crm", app, "guarded")), "alice", nil))

	parts, err := d.Run(s.ctx, s.toolCall("crm__guarded", `{"scope":"crm:write"}`))

	s.Error(err)
	s.Nil(parts)
	s.Empty(s.asked)
	s.Empty(s.events)
	s.Equal(store.ConnectionConnected, s.status(app))
}

// TestWithoutConsentsAScopeChallengeFailsAsBefore is the control: a deployment that cannot
// begin a consent (connectors off, or no ROUTER_PUBLIC_URL) has no step-ups, and the call
// fails as it did before this change.
func (s *StepUpSuite) TestWithoutConsentsAScopeChallengeFailsAsBefore() {
	mine := s.connection("alice", "primary")
	d, _, _, err := s.attach(s.spec(s.config(s.chosen("crm", "guarded")), "alice", map[string]string{"crm": mine}))
	s.Require().NoError(err)
	d.stepUps = newStepUps(nil, s.notify, slog.New(slog.DiscardHandler))

	parts, err := d.Run(s.ctx, s.toolCall("crm__guarded", `{"scope":"crm:write"}`))

	s.Nil(d.stepUps)
	s.Error(err)
	s.Nil(parts)
	s.Empty(s.events)
	s.Equal(store.ConnectionConnected, s.status(mine))
}

// TestAStepUpThatCannotBeBegunFailsAsBefore: the router could not begin the consent, so the
// call fails as it did before and nothing is sent to the watchers.
func (s *StepUpSuite) TestAStepUpThatCannotBeBegunFailsAsBefore() {
	mine := s.connection("alice", "primary")
	d := s.attachWithStepUps(s.spec(s.config(s.chosen("crm", "guarded")), "alice", map[string]string{"crm": mine}))
	s.refuse = true

	parts, err := d.Run(s.ctx, s.toolCall("crm__guarded", `{"scope":"crm:write"}`))

	s.Error(err)
	s.Nil(parts)
	s.Empty(s.events)
}

// attachWithStepUps opens spec's connectors with the suite's router behind its step-ups.
func (s *StepUpSuite) attachWithStepUps(spec Spec) *dispatcher {
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)
	s.Require().NotNil(d)
	d.stepUps = newStepUps(s.consents, s.notify, slog.New(slog.DiscardHandler))
	return d
}

// consents is the router beginning a step-up: a real attempt row, open for ten minutes, as
// api.ConnectorConsents stores one.
func (s *StepUpSuite) consents(ctx context.Context, request ConsentRequest) (Consent, error) {
	s.asked = append(s.asked, request)
	if s.refuse {
		return Consent{}, errors.New("the provider's consent could not be started")
	}
	id := store.NewID()
	expires := time.Now().Add(10 * time.Minute)
	err := s.store.CreateConnectorAuthorizationAttempt(ctx, &store.ConnectorAuthorizationAttempt{
		ID: id, CustomerID: request.CustomerID, ConnectionID: request.ConnectionID, Kind: store.AttemptStepUp,
		StateHash: store.AuthorizationStateHash(id), AttemptSealed: []byte("sealed"), KEKVersion: 1, ExpiresAt: expires,
	})
	s.Require().NoError(err)
	return Consent{ConnectionID: request.ConnectionID, AuthorizationID: id, LaunchURL: "https://router.example/launch/" + id,
		HandoffToken: "handoff-" + id, ExpiresAt: expires, Name: "CRM"}, nil
}

// toolCall is the model asking for name with arguments.
func (s *StepUpSuite) toolCall(name, arguments string) llm.ToolCall {
	return llm.ToolCall{ID: uuid.NewString(), Name: name, Arguments: arguments}
}

// guarded is the model calling crm's guarded tool with arguments in the reply turnID, and what
// it read.
func (s *StepUpSuite) guarded(d *dispatcher, turnID, arguments string) (string, error) {
	call := s.toolCall("crm__guarded", arguments)
	call.TurnID = turnID
	parts, err := d.Run(s.ctx, call)
	return llm.TextOf(parts), err
}

func (s *StepUpSuite) notify(event Event) {
	s.events = append(s.events, event)
}

// status is the connection's status as stored.
func (s *StepUpSuite) status(id string) string {
	connection, err := s.store.ConnectorConnection(s.ctx, s.customerID, id)
	s.Require().NoError(err)
	return connection.Status
}
