//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type CampaignsSuite struct {
	RouterSuite
}

func TestCampaignsSuite(t *testing.T) {
	runSuite(t, new(CampaignsSuite))
}

func (s *CampaignsSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *CampaignsSuite) TestACampaignIsCreatedWithNobodyToRing() {
	created := s.campaign(3)

	s.Equal(CampaignStateDraft, created.State,
		"a campaign that started itself would ring people nobody added yet")
	s.Equal(3, created.Concurrency)
	s.Empty(s.contacts(created.Id))
}

func (s *CampaignsSuite) TestACampaignKeepsWhoScheduledIt() {
	// Its calls are placed later, by no request, and are owned by whoever scheduled them.
	owner := s.data.createUser()
	scheduler := s.serverClient.actingFor(owner)
	scheduler.header.Set(auth.UsageCategoriesHeader, "project:outreach")
	var created Campaign
	s.Require().Equal(http.StatusCreated, scheduler.do(http.MethodPost, "/v1/agents/campaigns", map[string]any{
		"name": "may", "config_id": s.data.createAgentConfig().Id, "from_number": s.utils.number(),
	}, &created))

	stored, err := s.store.Campaign(context.Background(), s.customerID(), created.Id)

	s.Require().NoError(err)
	s.Equal(s.app.key, stored.APIKey)
	s.Equal(owner.userID, stored.UserID)
	s.Equal([]string{"project:outreach"}, stored.UsageCategories)
}

func (s *CampaignsSuite) TestACampaignNamingAConfigNobodyHasIsRefused() {
	// A campaign that named a config it could not use would fail one call at a time, at
	// whatever hour somebody started it.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/campaigns",
		map[string]any{"name": "may", "config_id": "nope", "from_number": s.utils.number()})

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, errUnknownConfig.Message)
}

func (s *CampaignsSuite) TestASuspendedAppCannotStartACampaign() {
	created := s.campaign(1)
	s.add(created.Id, map[string]any{"to_number": s.utils.number()})
	s.Require().NoError(s.store.SuspendTelephony(context.Background(), &store.TelephonySuspension{
		CustomerID: s.customerID(), Reason: "fraud", SuspendedBy: "ops@getstream.io",
	}))
	// The fixture's app is shared with other suites, which must not find it suspended.
	s.T().Cleanup(func() {
		s.Require().NoError(s.store.LiftTelephonySuspension(context.Background(), s.customerID()))
	})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/start", nil)

	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "suspended")
	s.Equal(ContactStatePending, s.contacts(created.Id)[0].State, "nobody was rung")
}

func (s *CampaignsSuite) TestAContactNotYetRungHasNoOutcome() {
	created := s.campaign(1)
	s.add(created.Id, map[string]any{"to_number": s.utils.number()})

	contacts := s.contacts(created.Id)

	s.Require().Len(contacts, 1)
	s.Equal(ContactStatePending, contacts[0].State)
	s.Nil(contacts[0].Outcome)
}

func (s *CampaignsSuite) TestAnotherAppsCampaignIsNotFound() {
	created := s.campaign(1)

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/campaigns/"+created.Id, nil, nil)
	})
}

func (s *CampaignsSuite) TestOnlyTheAppsOwnBackendMayCreateACampaign() {
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "config-" + s.utils.uuid(), "llm": "llm-flow"}, &config))

	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/campaigns", map[string]any{
			"name": "campaign-" + s.utils.uuid(), "config_id": config.Id,
			"from_number": s.utils.number(),
		}, nil)
	})
}

// campaign creates a campaign over a stored config and returns it.
func (s *CampaignsSuite) campaign(concurrency int) Campaign {
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "config-" + s.utils.uuid(), "llm": "llm-flow"}, &config))

	var created Campaign
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns",
		map[string]any{
			"name": "campaign-" + s.utils.uuid(), "config_id": config.Id,
			"from_number": s.utils.number(), "concurrency": concurrency,
		}, &created))
	return created
}

// add puts people on a campaign's list.
func (s *CampaignsSuite) add(campaignID string, contacts ...map[string]any) {
	s.Require().Equal(http.StatusCreated,
		s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+campaignID+"/contacts",
			map[string]any{"contacts": contacts}, nil))
}

// contacts is who a campaign holds, and what became of them.
func (s *CampaignsSuite) contacts(campaignID string) []Contact {
	var listed []Contact
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/campaigns/"+campaignID+"/contacts", nil, &listed))
	return listed
}
