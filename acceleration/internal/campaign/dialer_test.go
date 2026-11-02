package campaign

import (
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

	spec := campaignSpec(config, campaign, contact, phone.Placed{SessionID: "0b9f2a52-5c1e-4d7a-9a3e-0f6b8f1c2d3e", VendorCallID: "v-1"})

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

	spec := campaignSpec(config, store.Campaign{}, contact, phone.Placed{})

	s.Equal("Remind Anna.\n\nAbout the person you are calling:\n- plan: Gold\n\nBe brief.", spec.Instructions)
}
