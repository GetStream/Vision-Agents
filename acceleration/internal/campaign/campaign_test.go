package campaign

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type CampaignSuite struct {
	suite.Suite
}

func TestCampaignSuite(t *testing.T) {
	suite.Run(t, new(CampaignSuite))
}

func (s *CampaignSuite) TestWhatOnePersonWasRungAboutIsAddedToWhatTheAgentAlwaysSays() {
	told := say("You are calling on behalf of Acme.", "Ask whether they still want the trial.")

	s.Equal("You are calling on behalf of Acme.\n\nAsk whether they still want the trial.", told)
}

func (s *CampaignSuite) TestAContactWithNothingOfItsOwnIsToldWhatEveryCallIs() {
	s.Equal("You are calling on behalf of Acme.", say("You are calling on behalf of Acme.", ""))
	s.Equal("Ask about the trial.", say("", "Ask about the trial."))
}

func (s *CampaignSuite) TestACampaignsLabelsAreBilledAlongsideTheAgentsOwn() {
	// What an outbound push cost has to be tellable from what the same agent cost
	// answering the phone, so both sets of labels reach the request rows.
	merged := merge(
		routing.Tags{"agent": "support", "project": "core"},
		routing.Tags{"project": "winback", "campaign": "may"},
	)

	s.Equal(routing.Tags{"agent": "support", "project": "winback", "campaign": "may"}, merged)
}

func (s *CampaignSuite) TestACampaignWithoutLabelsLeavesTheAgentsAlone() {
	configured := routing.Tags{"agent": "support"}

	s.Equal(configured, merge(configured, nil))
}

func (s *CampaignSuite) TestARunnerNeedsEverythingACallIsMadeOf() {
	// A campaign is a phone call, a conversation and a row. Half of one would be a
	// campaign that starts and then fails once per contact.
	_, err := New(Options{})

	s.Require().Error(err)
}

func (s *CampaignSuite) TestACampaignSessionJoinsTheCallItRang() {
	// The route sends whoever answers into the call of the session the call was placed
	// for, so the session is opened under that id, with voice.
	request := placing(store.Campaign{CustomerID: "customer-1", FromNumber: "+15550000001"},
		store.Contact{ID: "contact-7", ToNumber: "+15550000002"})
	s.Empty(request.SessionID)

	spec := session.Spec{Text: true, CustomerID: "customer-1"}
	meet(&spec, phone.Placed{SessionID: "0b9f2a52-5c1e-4d7a-9a3e-0f6b8f1c2d3e"})
	s.Require().NoError(spec.Normalize())

	s.Equal("0b9f2a52-5c1e-4d7a-9a3e-0f6b8f1c2d3e", spec.CallID)
	s.Equal("agent", spec.CallType)
	s.False(spec.Text)
}

func (s *CampaignSuite) TestACampaignsCallsAreOwnedByWhoeverScheduledIt() {
	campaign := store.Campaign{CustomerID: "customer-1", FromNumber: "+15550000001",
		APIKey: "key-a", UserID: "u1", UsageCategories: []string{"project:outreach"}}

	request := placing(campaign, store.Contact{ID: "contact-7", ToNumber: "+15550000002"})

	s.Equal(routing.Attribution{APIKey: "key-a", UserID: "u1", UsageCategories: []string{"project:outreach"}},
		request.Owner.Attribution)
}

func (s *CampaignSuite) TestACampaignThatNamesNoKeyLeavesItForTheWriterToReport() {
	request := placing(store.Campaign{CustomerID: "customer-1", UserID: "u1"}, store.Contact{ID: "contact-7"})

	s.Empty(request.Owner.APIKey, "a gap the writer counts and logs, not a job of the router's")
}

func (s *CampaignSuite) TestAContactWhoseCallWentUnansweredBeforeItsSessionOpenedIsDone() {
	refused := stack.Wrap(&session.CallUnansweredError{Outcome: outcome.Busy})

	finished := unopened(store.Contact{ID: "contact-7", State: store.Done}, "session-1", refused)

	s.Equal(store.Done, finished.State)
	s.Equal("session-1", finished.CallID, "the outcome is read by the call id")
	s.Empty(finished.Error)
}

func (s *CampaignSuite) TestAContactWhoseSessionFailedToOpenIsFailed() {
	finished := unopened(store.Contact{ID: "contact-7", State: store.Done}, "session-1", errors.New("no agent call type"))

	s.Equal(store.Failed, finished.State)
	s.Equal("no agent call type", finished.Error)
	s.Empty(finished.CallID)
}
