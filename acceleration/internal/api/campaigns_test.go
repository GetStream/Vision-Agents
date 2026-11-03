//go:build integration

package api

import (
	"context"
	"log/slog"
	"net/http"
	"testing"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
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

func (s *CampaignsSuite) TestContactsAreRungInTheOrderTheyWereAdded() {
	created := s.campaign(1)
	first, second := s.utils.number(), s.utils.number()
	s.add(created.Id, map[string]any{"to_number": first, "instructions": "ask about the trial"},
		map[string]any{"to_number": second})

	s.running(created.Id)
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

	s.running(created.Id)
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

	s.running(created.Id)
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

func (s *CampaignsSuite) TestAContactShowsHowItsCallEnded() {
	created := s.campaign(1)
	s.add(created.Id, map[string]any{"to_number": s.utils.number()})
	s.running(created.Id)
	ctx := context.Background()
	rung, _, err := s.store.ClaimContact(ctx, created.Id)
	s.Require().NoError(err)
	callID := s.utils.uuid()
	token, err := s.router.phone.Outcomes().Placed(ctx, outcome.Call{
		Key: outcome.Key{
			CustomerID: s.customerID(), CallType: "default", CallID: callID,
			Local: created.FromNumber, Remote: rung.ToNumber,
		},
		Vendor:  "twilio",
		Reports: true,
	})
	s.Require().NoError(err)
	_, err = s.router.phone.Outcomes().Reported(ctx, token, outcome.NoAnswer, "no-answer")
	s.Require().NoError(err)
	s.Require().NoError(s.store.FinishContact(ctx, store.Contact{ID: rung.ID, State: store.Done, CallID: callID}))

	contacts := s.contacts(created.Id)

	s.Require().Len(contacts, 1)
	s.Equal(CallOutcomeNoAnswer, value(contacts[0].Outcome))
	s.Equal(ContactStateDone, contacts[0].State, "nobody answering is still a contact that was rung")
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

func (s *CampaignsSuite) TestAPauseOnAnotherRouterStopsTheCallsOfTheOneRunningIt() {
	created := s.campaign(1)
	s.add(created.Id, map[string]any{"to_number": s.utils.number()}, map[string]any{"to_number": s.utils.number()})
	// The suite's router runs the campaign; the pause lands on a second router over the same
	// database, as it does behind a load balancer.
	other := s.campaigns(s.router.sessions, slog.New(slog.DiscardHandler))
	locked := s.holdContacts(created.Id)
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/start", nil, nil))

	s.Require().NoError(other.Pause(context.Background(), s.customerID(), created.Id))
	s.Require().NoError(locked.Rollback())

	s.Never(func() bool {
		return s.state(created.Id) != CampaignStatePaused || s.rung(created.Id) > 0
	}, time.Second, 20*time.Millisecond, "the router running it rang nobody after the pause, and left it paused")
	// A call that never finished, which only a start that finds no loop running gives back.
	_, err := s.store.DB().NewUpdate().Table("campaign_contacts").Set("state = ?", store.Calling).
		Where("id = ?", s.contacts(created.Id)[0].Id).Exec(context.Background())
	s.Require().NoError(err)
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/start", nil, nil))
	s.Require().Eventually(func() bool { return s.state(created.Id) == CampaignStateFinished },
		10*time.Second, 20*time.Millisecond, "started again where its loop ended at the pause, it rings again")
	for _, contact := range s.contacts(created.Id) {
		s.Equal(ContactStateFailed, contact.State, "everybody it held back is rung")
		s.Equal(1, contact.Attempts)
	}
}

func (s *CampaignsSuite) TestACampaignNobodyPausesRingsEverybodyAndFinishes() {
	created := s.campaign(2)
	s.add(created.Id, map[string]any{"to_number": s.utils.number()}, map[string]any{"to_number": s.utils.number()},
		map[string]any{"to_number": s.utils.number()})

	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/start", nil, nil))

	s.Require().Eventually(func() bool { return s.state(created.Id) == CampaignStateFinished },
		10*time.Second, 20*time.Millisecond)
	contacts := s.contacts(created.Id)
	s.Require().Len(contacts, 3)
	for _, contact := range contacts {
		// The suite's phone service has no credentials, so every call it places fails.
		s.Equal(ContactStateFailed, contact.State)
		s.Equal(1, contact.Attempts)
	}
}

func (s *CampaignsSuite) TestNobodyIsClaimedFromACampaignThatIsNotRunning() {
	for _, state := range []string{store.Draft, store.Paused, store.Finished} {
		s.Run(state, func() {
			created := s.campaign(1)
			s.add(created.Id, map[string]any{"to_number": s.utils.number()})
			s.running(created.Id)
			s.Require().NoError(s.store.SetCampaignState(context.Background(), s.customerID(), created.Id, store.Paused))
			if state != store.Paused {
				_, err := s.store.DB().NewUpdate().Table("campaigns").Set("state = ?", state).
					Where("id = ?", created.Id).Exec(context.Background())
				s.Require().NoError(err)
			}

			_, found, err := s.store.ClaimContact(context.Background(), created.Id)

			s.Require().NoError(err)
			s.False(found, "a router that had not heard of the pause would ring them")
			contacts := s.contacts(created.Id)
			s.Equal(ContactStatePending, contacts[0].State)
			s.Zero(contacts[0].Attempts)
		})
	}
}

func (s *CampaignsSuite) TestACampaignThatRanOutWhilePausedStaysPaused() {
	created := s.campaign(1)
	s.running(created.Id)
	ctx := context.Background()
	s.Require().NoError(s.store.SetCampaignState(ctx, s.customerID(), created.Id, store.Paused))

	// What a loop on another router writes when its last claim found nobody.
	s.Require().NoError(s.store.SetCampaignState(ctx, s.customerID(), created.Id, store.Finished))

	stored, err := s.store.Campaign(ctx, s.customerID(), created.Id)
	s.Require().NoError(err)
	s.Equal(store.Paused, stored.State)
	s.Nil(stored.FinishedAt)
}

func (s *CampaignsSuite) TestAStartOnTheRouterWhoseLoopOutlivedAPauseRunsTheCampaign() {
	created := s.campaign(1)
	s.add(created.Id, map[string]any{"to_number": s.utils.number()})
	ctx := context.Background()
	locked := s.holdContacts(created.Id)
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/start", nil, nil))
	// Another router's pause, held open: the loop here waits on it at its next claim.
	pausing, err := s.store.DB().BeginTx(ctx, nil)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = pausing.Rollback() })
	_, err = pausing.NewUpdate().Table("campaigns").Set("state = ?", store.Paused).Where("id = ?", created.Id).Exec(ctx)
	s.Require().NoError(err)
	s.Require().NoError(locked.Rollback())

	started := make(chan int, 1)
	go func() {
		started <- s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/start", nil, nil)
	}()
	// The loop's claim and the start both wait on the pause, so neither has read it yet.
	s.Require().Eventually(func() bool { return s.waitingOn(created.Id) == 2 },
		10*time.Second, 20*time.Millisecond, "the start writes the campaign running, behind the pause")
	s.Require().NoError(pausing.Commit())

	s.Equal(http.StatusAccepted, <-started)
	s.Require().Eventually(func() bool { return s.state(created.Id) == CampaignStateFinished },
		10*time.Second, 20*time.Millisecond, "the start that came after the pause is the last word, and rings")
	s.Equal(1, s.rung(created.Id))
}

// running marks a campaign running without starting a loop, for a test that claims its
// contacts itself.
func (s *CampaignsSuite) running(campaignID string) {
	s.Require().NoError(s.store.SetCampaignState(context.Background(), s.customerID(), campaignID, store.Running))
}

// waitingOn counts the statements naming a campaign that wait on a lock.
func (s *CampaignsSuite) waitingOn(campaignID string) int {
	var waiting int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' AND query LIKE ?",
		"%"+campaignID+"%").Scan(&waiting))
	return waiting
}

// holdContacts locks a campaign's contacts until the returned transaction ends, which holds
// a runner off them, as its claim skips locked rows.
func (s *CampaignsSuite) holdContacts(campaignID string) bun.Tx {
	ctx := context.Background()
	locked, err := s.store.DB().BeginTx(ctx, nil)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = locked.Rollback() })
	_, err = locked.ExecContext(ctx, "SELECT id FROM campaign_contacts WHERE campaign_id = ? FOR UPDATE", campaignID)
	s.Require().NoError(err)
	return locked
}

// state is what the campaign's row says it is doing.
func (s *CampaignsSuite) state(campaignID string) CampaignState {
	var campaign Campaign
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/campaigns/"+campaignID, nil, &campaign))
	return campaign.State
}

// rung counts the times anybody on the campaign was claimed to be rung.
func (s *CampaignsSuite) rung(campaignID string) int {
	rung := 0
	for _, contact := range s.contacts(campaignID) {
		rung += contact.Attempts
	}
	return rung
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
