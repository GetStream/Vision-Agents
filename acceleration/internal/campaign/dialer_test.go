package campaign

import (
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

func (s *CampaignSuite) TestACampaignCallsInstructionsCarryTheContactsVariables() {
	config := store.AgentConfig{ID: "config-1", CustomerID: "customer-1", Name: "renewals",
		Instructions: "Remind {{first_name}} about the renewal.", Greeting: "Hi {{ first_name }}!",
		Tags: routing.Tags{"agent": "renewals"}}
	campaign := store.Campaign{ID: "campaign-1", CustomerID: "customer-1", FromNumber: "+15550000001",
		Tags: routing.Tags{"campaign": "march"}, APIKey: "key-a"}
	contact := store.Contact{ID: "contact-1", ToNumber: "+15550000002", Instructions: "They asked for a call after five.",
		Variables: map[string]string{"first_name": "Anna", "plan": "Gold"}}

	spec := campaignSpec(config, campaign, DefaultSequence("config-1").First(), contact, phone.Placed{SessionID: "0b9f2a52-5c1e-4d7a-9a3e-0f6b8f1c2d3e", VendorCallID: "v-1"})

	s.Contains(spec.Instructions, "Remind Anna about the renewal.")
	s.Contains(spec.Instructions, "- plan: Gold")
	s.NotContains(spec.Instructions, "- first_name", "a variable the config names is not briefed again")
	s.Contains(spec.Instructions, "They asked for a call after five.")
	s.Equal("Hi Anna!", spec.Greeting)
	s.Equal("0b9f2a52-5c1e-4d7a-9a3e-0f6b8f1c2d3e", spec.ID)
	s.Equal("campaign-1", spec.CampaignID)
	s.Equal("contact-1", spec.ContactID)
	s.True(spec.Navigating)
	s.Equal(routing.Tags{"agent": "renewals", "campaign": "march"}, spec.Tags)
	s.Equal("key-a", spec.Attribution.APIKey)
	s.Require().NotNil(spec.Phone)
	s.Equal("+15550000002", spec.Phone.To)
	s.Equal("v-1", spec.Phone.VendorCallID)
}

func (s *CampaignSuite) TestTheContactsOwnInstructionsComeLast() {
	config := store.AgentConfig{Instructions: "Remind {{first_name}}."}
	contact := store.Contact{Instructions: "Be brief.", Variables: map[string]string{"first_name": "Anna", "plan": "Gold"}}

	spec := campaignSpec(config, store.Campaign{}, DefaultSequence("config-1").First(), contact, phone.Placed{})

	s.True(strings.HasPrefix(spec.Instructions, "Remind Anna.\n\nAbout the person you are calling:\n- plan: Gold\n\nBe brief.\n\n"), spec.Instructions)
}

func (s *CampaignSuite) TestACampaignCallIsToldItsGoalThenTheBriefThenTheContactThenHowToEnd() {
	config := store.AgentConfig{Instructions: "You are Jean."}
	step := DefaultSequence("config-1").First()
	step.Goal = "Confirm {{name}}'s visit on {{appointment}}."
	contact := store.Contact{Instructions: "Speak slowly.",
		Variables: map[string]string{"name": "Ali", "appointment": "Tuesday at 10 AM", "clinic": "Northwind"}}

	spec := campaignSpec(config, store.Campaign{}, step, contact, phone.Placed{})

	wantInOrder := []string{"You are Jean.", "This call: Confirm Ali's visit on Tuesday at 10 AM.\nWhen a person answers, tell them why you are calling in your first sentence.", "- clinic: Northwind", "Speak slowly.", "How the call ends:"}
	last := -1
	for _, want := range wantInOrder {
		at := strings.Index(spec.Instructions, want)
		s.Require().GreaterOrEqual(at, 0, "%q is missing from %q", want, spec.Instructions)
		s.Greater(at, last, "%q is out of order in %q", want, spec.Instructions)
		last = at
	}
	s.NotContains(spec.Instructions, "- name:")
	s.NotContains(spec.Instructions, "- appointment:")
}

func (s *CampaignSuite) TestTheClosingRuleNamesEveryOutcomeAndTellsTheAgentToEndTheCall() {
	step := DefaultSequence("config-1").First()
	step.Goal = "Confirm the visit."

	spec := campaignSpec(store.AgentConfig{}, store.Campaign{}, step, store.Contact{}, phone.Placed{})

	s.Contains(spec.Instructions, "say a short goodbye and end the call")
	s.True(strings.HasSuffix(spec.Instructions, "Say only the words you would speak aloud: never put a name or a label in front of them."))
	s.Contains(spec.Instructions, "- Success: The person agreed to what the call was about")
	s.Contains(spec.Instructions, "- Not interested: The person declined")
	s.Contains(spec.Instructions, "- Do not call: The person asked not to be called or contacted again.")
}

func (s *CampaignSuite) TestACampaignWithNoGoalAddsNoGoalLineButStillTellsTheAgentHowToEnd() {
	spec := campaignSpec(store.AgentConfig{Instructions: "You are Jean."}, store.Campaign{}, DefaultSequence("config-1").First(), store.Contact{}, phone.Placed{})

	s.NotContains(spec.Instructions, "This call:")
	s.NotContains(spec.Instructions, "tell them why you are calling")
	s.Contains(spec.Instructions, "How the call ends:")
}
