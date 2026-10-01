//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"

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

func (s *CampaignsSuite) TestACampaignNamingAConfigNobodyHasIsRefused() {
	// A campaign that named a config it could not use would fail one call at a time, at
	// whatever hour somebody started it.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/campaigns",
		map[string]any{"name": "may", "config_id": "nope", "from_number": s.utils.number()})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "config")
}

func (s *CampaignsSuite) TestContactsAreRungInTheOrderTheyWereAdded() {
	created := s.campaign(1)
	first, second := s.utils.number(), s.utils.number()
	s.add(created.Id, map[string]any{"to_number": first, "instructions": "ask about the trial"},
		map[string]any{"to_number": second})

	ctx := context.Background()
	claimed, found, err := s.store.ClaimContact(ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal(first, claimed.ToNumber)
	s.Equal("ask about the trial", claimed.Instructions)
	s.Equal(store.Calling, claimed.State, "a claimed contact is nobody else's to ring")
	s.Equal(1, claimed.Attempts)

	next, found, err := s.store.ClaimContact(ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal(second, next.ToNumber, "the same person was taken twice")

	_, found, err = s.store.ClaimContact(ctx, created.Id)
	s.Require().NoError(err)
	s.False(found, "there was nobody left to ring")
}

func (s *CampaignsSuite) TestACallThatNeverFinishedIsRungAgainRatherThanLost() {
	// A process that stopped mid-call leaves a contact claimed by nobody. Ringing them
	// again is better than a campaign that quietly skips them.
	created := s.campaign(1)
	s.add(created.Id, map[string]any{"to_number": s.utils.number()})

	ctx := context.Background()
	_, found, err := s.store.ClaimContact(ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)

	s.Require().NoError(s.store.ReleaseContacts(ctx, created.Id))

	again, found, err := s.store.ClaimContact(ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal(2, again.Attempts, "the second attempt is counted as one")
}

func (s *CampaignsSuite) TestWhatBecameOfAContactIsShownAgainstIt() {
	created := s.campaign(1)
	s.add(created.Id, map[string]any{"to_number": s.utils.number()},
		map[string]any{"to_number": s.utils.number()})

	ctx := context.Background()
	rung, _, err := s.store.ClaimContact(ctx, created.Id)
	s.Require().NoError(err)
	s.Require().NoError(s.store.FinishContact(ctx, store.Contact{
		ID: rung.ID, State: store.Done, CallID: "session-1", VendorCallID: "CA1",
	}))

	unreachable, _, err := s.store.ClaimContact(ctx, created.Id)
	s.Require().NoError(err)
	s.Require().NoError(s.store.FinishContact(ctx, store.Contact{
		ID: unreachable.ID, State: store.Failed, Error: "the number is not in service",
	}))

	contacts := s.contacts(created.Id)
	s.Require().Len(contacts, 2)
	s.Equal(ContactStateDone, contacts[0].State)
	s.Equal("session-1", value(contacts[0].CallId))
	s.Equal(ContactStateFailed, contacts[1].State)
	s.Contains(value(contacts[1].Error), "not in service")
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
