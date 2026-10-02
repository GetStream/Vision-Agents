package campaign

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
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
	// The route sends whoever answers into the call the request names, so the agent has
	// to join that call and not one of its own naming.
	request := placing(store.Campaign{CustomerID: "customer-1", FromNumber: "+15550000001"},
		store.Contact{ID: "contact-7", ToNumber: "+15550000002"}, "agent")
	s.Equal("campaign-contact-7", request.CallID)
	s.Equal("agent", request.CallType)

	var spec session.Spec
	meet(&spec, phone.Placed{CallID: request.CallID, CallType: request.CallType})

	s.Equal(request.CallID, spec.CallID)
	s.Equal(request.CallType, spec.CallType)
}
