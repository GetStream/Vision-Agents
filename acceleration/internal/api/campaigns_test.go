//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"

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

func (s *CampaignsSuite) TestANewCampaignGetsTheDefaultSequenceAndHours() {
	config := s.data.createAgentConfig().Id

	created := s.create(map[string]any{"name": "may", "config_id": config})

	s.Equal(CampaignStateDraft, created.State)
	s.Equal(CampaignTypeStandard, created.Type)
	s.Require().Len(created.Sequence.Steps, 1)
	step := created.Sequence.Steps[0]
	s.Equal(config, step.ConfigId, "the step rings with the campaign's config")
	s.Equal([]CampaignAttempt{{0}, {4320}, {10080}, {14400}}, step.Attempts)
	s.True(created.Sequence.DoubleDial)
	s.Equal([]CallingWindow{{Start: "09:00", End: "18:00"}}, created.CallingHours.Windows["monday"])
	s.Empty(created.CallingHours.Windows["saturday"], "nobody is rung at the weekend unless the campaign says so")
	s.Equal("contact", created.CallingHours.TimeZone.Mode)
	s.Zero(created.Contacts.Total)
}

func (s *CampaignsSuite) TestAnAutoEnrollCampaignIsRefusedForNow() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/campaigns",
		map[string]any{"name": "may", "type": "auto_enroll"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "auto-enroll campaigns are not available yet")
}

func (s *CampaignsSuite) TestACampaignsTypeNeverChanges() {
	created := s.create(map[string]any{"name": "may"})

	status, _ := s.serverClient.failure(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"type": "auto_enroll"})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(CampaignTypeStandard, s.get(created.Id).Type)
}

func (s *CampaignsSuite) TestACampaignNamingAConfigNobodyHasIsRefused() {
	// A campaign that named a config it could not use would fail one call at a time, at
	// whatever hour somebody started it.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/campaigns",
		map[string]any{"name": "may", "config_id": "nope"})

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, errUnknownConfig.Message)
}

func (s *CampaignsSuite) TestChangingTheConfigChangesTheStepThatRingsWithIt() {
	created := s.create(map[string]any{"name": "may", "config_id": s.data.createAgentConfig().Id})
	other := s.data.createAgentConfig().Id

	var changed Campaign
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"config_id": other}, &changed))

	s.Equal(other, changed.ConfigId)
	s.Equal(other, changed.Sequence.Steps[0].ConfigId)
}

func (s *CampaignsSuite) TestChangingTheStepsConfigChangesTheCampaigns() {
	created := s.create(map[string]any{"name": "may", "config_id": s.data.createAgentConfig().Id})
	sequence := created.Sequence
	sequence.Steps[0].ConfigId = s.data.createAgentConfig().Id

	var changed Campaign
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence}, &changed))

	s.Equal(sequence.Steps[0].ConfigId, changed.ConfigId)
}

func (s *CampaignsSuite) TestAConfigAndASequenceThatDisagreeAreRefused() {
	created := s.create(map[string]any{"name": "may", "config_id": s.data.createAgentConfig().Id})
	sequence := created.Sequence
	sequence.Steps[0].ConfigId = s.data.createAgentConfig().Id

	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"config_id": s.data.createAgentConfig().Id, "sequence": sequence})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "config_id")
}

func (s *CampaignsSuite) TestAnInvalidSequenceIsRefused() {
	created := s.create(map[string]any{"name": "may"})
	sequence := created.Sequence
	sequence.Steps[0].Attempts = []CampaignAttempt{{5}}

	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "first attempt")
}

func (s *CampaignsSuite) TestOnlyTheAppsOwnBackendMayManageCampaigns() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/campaigns", map[string]any{"name": "campaign-" + s.utils.uuid()}, nil)
	})
}

func (s *CampaignsSuite) TestCampaignsPageNewestFirst() {
	s.useApp(s.data.createApp())
	first := s.create(map[string]any{"name": "first"})
	second := s.create(map[string]any{"name": "second"})
	third := s.create(map[string]any{"name": "third"})

	var page CampaignPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/campaigns?limit=2", nil, &page))
	s.Require().Len(page.Items, 2)
	s.Equal([]string{third.Id, second.Id}, []string{page.Items[0].Id, page.Items[1].Id})
	s.True(page.HasMore)
	s.Require().NotNil(page.NextCursor)

	var next CampaignPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/campaigns?limit=2&cursor="+*page.NextCursor, nil, &next))
	s.Require().Len(next.Items, 1)
	s.Equal(first.Id, next.Items[0].Id)
	s.False(next.HasMore)
}

func (s *CampaignsSuite) TestTheListCountsEachCampaignsContacts() {
	s.useApp(s.data.createApp())
	created := s.create(map[string]any{"name": "may"})
	s.add(created.Id, map[string]any{"to_number": s.utils.number()}, map[string]any{"to_number": s.utils.number()})

	var page CampaignPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/campaigns", nil, &page))

	s.Require().Len(page.Items, 1)
	s.Equal(2, page.Items[0].Contacts.Total)
	s.Equal(2, page.Items[0].Contacts.Enrolled)
}

func (s *CampaignsSuite) TestTheListFiltersByState() {
	s.useApp(s.data.createApp())
	s.create(map[string]any{"name": "draft"})
	launched := s.launch(s.ready())

	var page CampaignPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/campaigns?state=running", nil, &page))

	s.Require().Len(page.Items, 1)
	s.Equal(launched.Id, page.Items[0].Id)
}

func (s *CampaignsSuite) TestLaunchingWithoutConsentIsRefused() {
	created := s.ready()

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": false})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), codeConsentRequired)
	s.Equal(CampaignStateDraft, s.get(created.Id).State)
}

func (s *CampaignsSuite) TestLaunchingWithoutContactsSaysSo() {
	created := s.create(map[string]any{
		"name": "may", "config_id": s.data.createAgentConfig().Id, "from_number": s.heldNumber(),
	})

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), codeCampaignNotReady)
	s.Contains(string(payload), "contacts")
}

func (s *CampaignsSuite) TestLaunchingFromANumberTheAppDoesNotHaveIsRefused() {
	created := s.ready()
	// The number was the app's when it was set, and is not now.
	notOurs := s.utils.number()
	_, err := s.store.UpdateCampaign(context.Background(), s.customerID(), created.Id, store.CampaignChanges{FromNumber: &notOurs})
	s.Require().NoError(err)

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), codeCampaignNotReady)
	s.Contains(string(payload), notOurs)
}

func (s *CampaignsSuite) TestANumberTheAppDoesNotHoldIsRefused() {
	notOurs := s.utils.number()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/campaigns",
		map[string]any{"name": "may", "from_number": notOurs})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, notOurs)
}

func (s *CampaignsSuite) TestALaunchedCampaignCannotMoveToANumberTheAppDoesNotHold() {
	// Every call it placed from there would be refused, and each contact would run out of
	// attempts without anybody being rung.
	created := s.launch(s.ready())

	status, _ := s.serverClient.failure(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"from_number": s.utils.number()})
	s.Equal(http.StatusBadRequest, status)
	status, _ = s.serverClient.failure(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"from_number": ""})
	s.Equal(http.StatusBadRequest, status, "nor to no number at all")

	s.Equal(created.FromNumber, s.get(created.Id).FromNumber)
	moved := s.heldNumber()
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"from_number": moved}, nil))
	s.Equal(moved, s.get(created.Id).FromNumber)
}

func (s *CampaignsSuite) TestADraftMayLeaveItsNumberEmpty() {
	created := s.ready()

	var changed Campaign
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"from_number": ""}, &changed))

	s.Empty(changed.FromNumber)
}

func (s *CampaignsSuite) TestLaunchingWithoutAConfigIsRefused() {
	created := s.create(map[string]any{"name": "may", "from_number": s.heldNumber()})
	s.add(created.Id, map[string]any{"to_number": s.utils.number()})

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), codeCampaignNotReady)
	s.Contains(string(payload), "agent config")
}

func (s *CampaignsSuite) TestLaunchingWithoutAGoalIsRefused() {
	created := s.ready()
	s.setGoal(created, "  ")

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), codeCampaignNotReady)
	s.Contains(string(payload), "goal")
}

func (s *CampaignsSuite) TestAGoalIsStoredAndReturned() {
	created := s.create(map[string]any{"name": "may"})
	s.setGoal(created, "Confirm the visit for {{name}}")

	s.Equal("Confirm the visit for {{name}}", s.get(created.Id).Sequence.Steps[0].Goal)
}

func (s *CampaignsSuite) TestAFullSequenceWithoutAGoalIsStillAccepted() {
	created := s.create(map[string]any{"name": "may"})
	var sequence map[string]any
	raw, err := json.Marshal(created.Sequence)
	s.Require().NoError(err)
	s.Require().NoError(json.Unmarshal(raw, &sequence))
	delete(sequence["steps"].([]any)[0].(map[string]any), "goal")

	s.Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence}, nil))
}

func (s *CampaignsSuite) TestAGoalOverTheLimitIsRefused() {
	created := s.create(map[string]any{"name": "may"})
	sequence := created.Sequence
	sequence.Steps[0].Goal = strings.Repeat("a", 2001)

	status, _ := s.serverClient.failure(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence})

	s.Equal(http.StatusBadRequest, status)
	s.Empty(s.get(created.Id).Sequence.Steps[0].Goal)
}

func (s *CampaignsSuite) TestLaunchingAStartInThePastIsRefused() {
	created := s.ready()

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true, "start_at": time.Now().Add(-time.Hour)})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), codeCampaignNotReady)
}

func (s *CampaignsSuite) TestLaunchingLaterSchedulesIt() {
	created := s.ready()
	at := time.Now().Add(time.Hour).UTC().Truncate(time.Second)

	var launched Campaign
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true, "start_at": at}, &launched))

	s.Equal(CampaignStateScheduled, launched.State)
	s.Require().NotNil(launched.StartAt)
	s.True(at.Equal(*launched.StartAt))
	s.Nil(launched.StartedAt, "nobody has been rung yet")
}

func (s *CampaignsSuite) TestResumingBeforeTheStartTimeReturnsToScheduled() {
	created := s.ready()
	at := time.Now().Add(time.Hour).UTC().Truncate(time.Second)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true, "start_at": at}, nil))
	_, err := s.store.DB().ExecContext(context.Background(), "UPDATE campaigns SET state = 'paused' WHERE id = ?", created.Id)
	s.Require().NoError(err)

	resumed := s.move(created.Id, "resume")

	s.Equal(CampaignStateScheduled, resumed.State)
	s.Nil(resumed.StartedAt)
}

func (s *CampaignsSuite) TestALaunchedCampaignRefusesAChangeToItsSequence() {
	created := s.launch(s.ready())
	sequence := created.Sequence
	sequence.Steps[0].Attempts = []CampaignAttempt{{0}}

	status, payload := s.serverClient.call(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence})

	s.Equal(http.StatusConflict, status)
	s.Contains(string(payload), codeCampaignLocked)
	s.Len(s.get(created.Id).Sequence.Steps[0].Attempts, 4)
}

func (s *CampaignsSuite) TestALaunchedCampaignCanBeRenamed() {
	created := s.launch(s.ready())
	// The dashboard's settings form sends every field back, the locked ones unchanged.
	whole := map[string]any{
		"name": "renamed", "type": created.Type, "config_id": created.ConfigId, "from_number": created.FromNumber,
		"default_country": created.DefaultCountry, "concurrency": 2, "sequence": created.Sequence,
		"calling_hours": created.CallingHours, "tags": created.Tags,
	}

	var renamed Campaign
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id, whole, &renamed))

	s.Equal("renamed", renamed.Name)
	s.Equal(2, renamed.Concurrency)
	s.Equal(CampaignStateRunning, renamed.State)
}

func (s *CampaignsSuite) TestLaunchCapturesWhoLaunched() {
	// Its calls are placed later, by no request, and are owned by whoever launched it.
	created := s.ready()
	launcher, key := s.backendWithAnotherKey()
	owner := s.data.createUser()
	launcher = launcher.actingFor(owner)
	launcher.header.Set(auth.UsageCategoriesHeader, "project:outreach")

	s.Require().Equal(http.StatusOK, launcher.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true}, nil))

	stored, err := s.store.Campaign(context.Background(), s.customerID(), created.Id)
	s.Require().NoError(err)
	s.Equal(key, stored.APIKey, "the key that launched it, not the one that made it")
	s.NotEqual(s.app.key, stored.APIKey)
	s.Equal(owner.userID, stored.UserID)
	s.Equal([]string{"project:outreach"}, stored.UsageCategories)
}

func (s *CampaignsSuite) TestPauseResumeAndStop() {
	created := s.launch(s.ready())
	s.Equal(CampaignStateRunning, created.State)

	s.Equal(CampaignStatePaused, s.move(created.Id, "pause").State)
	s.Equal(CampaignStateRunning, s.move(created.Id, "resume").State)
	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/resume", nil)
	s.Equal(http.StatusConflict, status, "a running campaign is not resumed")
	s.Contains(string(payload), codeCampaignLocked)
	stopped := s.move(created.Id, "stop")
	s.Equal(CampaignStateStopped, stopped.State)
	s.NotNil(stopped.StoppedAt)
}

func (s *CampaignsSuite) TestDeletingARunningCampaignIsRefused() {
	created := s.launch(s.ready())

	status, payload := s.serverClient.call(http.MethodDelete, "/v1/agents/campaigns/"+created.Id, nil)

	s.Equal(http.StatusConflict, status)
	s.Contains(string(payload), codeCampaignLocked)
	s.move(created.Id, "stop")
	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/campaigns/"+created.Id, nil, nil))
	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodGet, "/v1/agents/campaigns/"+created.Id, nil, nil))
}

func (s *CampaignsSuite) TestDuplicatingCopiesSettings() {
	created := s.create(map[string]any{
		"name": "may", "config_id": s.data.createAgentConfig().Id, "concurrency": 3,
		"tags": map[string]string{"team": "sales"}, "default_country": "GB",
	})
	sequence := created.Sequence
	sequence.DoubleDial = false
	hours := created.CallingHours
	hours.Windows["saturday"] = []CallingWindow{{Start: "10:00", End: "12:00"}}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence, "calling_hours": hours}, nil))
	s.add(created.Id, map[string]any{"to_number": s.utils.number()}, map[string]any{"to_number": s.utils.number()})

	var copied Campaign
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/duplicate",
		map[string]any{"contacts": "all"}, &copied))

	s.NotEqual(created.Id, copied.Id)
	s.Equal("Copy of may", copied.Name)
	s.Equal(CampaignStateDraft, copied.State)
	s.Equal(3, copied.Concurrency)
	s.Equal("GB", copied.DefaultCountry)
	s.Equal(map[string]string{"team": "sales"}, copied.Tags)
	s.False(copied.Sequence.DoubleDial)
	s.Equal([]CallingWindow{{Start: "10:00", End: "12:00"}}, copied.CallingHours.Windows["saturday"])
	s.Equal(2, copied.Contacts.Total)
}

func (s *CampaignsSuite) TestAnalyticsOfANewCampaignAreZero() {
	created := s.ready()

	var analytics CampaignAnalytics
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/campaigns/"+created.Id+"/analytics", nil, &analytics))

	s.Zero(analytics.Dialed)
	s.Zero(analytics.Connected)
	s.Zero(analytics.Converted)
	s.Zero(analytics.ConversionRate, "nothing dialed converts at no rate, not at NaN")
	s.Empty(analytics.Outcomes)
	s.NotNil(analytics.DialsPerDay)
	s.Empty(analytics.DialsPerDay)
}

func (s *CampaignsSuite) TestASuspendedAppCannotLaunch() {
	s.useApp(s.data.createApp())
	created := s.ready()
	s.Require().NoError(s.store.SuspendTelephony(context.Background(), &store.TelephonySuspension{
		CustomerID: s.customerID(), Reason: "fraud", SuspendedBy: "ops@getstream.io",
	}))

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true})

	s.Equal(http.StatusForbidden, status)
	s.Contains(failure, "suspended")
	s.Equal(CampaignStateDraft, s.get(created.Id).State, "nobody is rung")
}

func (s *CampaignsSuite) TestAContactNotYetRungHasNoOutcome() {
	created := s.create(map[string]any{"name": "may"})
	s.add(created.Id, map[string]any{"to_number": s.utils.number()})

	contacts := s.contacts(created.Id)

	s.Require().Len(contacts, 1)
	s.Equal(ContactStageEnrolled, contacts[0].Stage)
	s.Nil(contacts[0].Outcome)
}

func (s *CampaignsSuite) TestAnotherAppsCampaignIsNotFound() {
	created := s.launch(s.ready())
	path := "/v1/agents/campaigns/" + created.Id

	for _, call := range []struct {
		method, path string
		body         any
	}{
		{http.MethodGet, path, nil},
		{http.MethodPatch, path, map[string]any{"name": "theirs"}},
		{http.MethodDelete, path, nil},
		{http.MethodPost, path + "/launch", map[string]any{"consent": true}},
		{http.MethodPost, path + "/pause", nil},
		{http.MethodPost, path + "/resume", nil},
		{http.MethodPost, path + "/stop", nil},
		{http.MethodPost, path + "/duplicate", map[string]any{"contacts": "all"}},
		{http.MethodGet, path + "/analytics", nil},
	} {
		s.assertHiddenFromOtherApps(func(as *testClient) int {
			return as.do(call.method, call.path, call.body, nil)
		})
	}
	s.Equal(CampaignStateRunning, s.get(created.Id).State, "and none of it reached the campaign")
	s.Equal(created.Name, s.get(created.Id).Name)
}

func (s *CampaignsSuite) TestTheListLeavesOutAnotherAppsCampaigns() {
	s.useApp(s.data.createApp())
	ours := s.create(map[string]any{"name": "ours"})

	var page CampaignPage
	s.Require().Equal(http.StatusOK, s.data.backendOfAnotherApp().do(http.MethodGet, "/v1/agents/campaigns", nil, &page))

	for _, item := range page.Items {
		s.NotEqual(ours.Id, item.Id)
	}
}

func (s *CampaignsSuite) TestABadCursorIsRefused() {
	status, _ := s.serverClient.failure(http.MethodGet, "/v1/agents/campaigns?cursor=not-a-cursor", nil)

	s.Equal(http.StatusBadRequest, status)
}

func (s *CampaignsSuite) TestCopyingTheContactsNeverRungOfARunningCampaignIsRefused() {
	// Who was never rung is not known while the campaign is still ringing people.
	created := s.launch(s.ready())

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/duplicate",
		map[string]any{"contacts": "not_contacted"})

	s.Equal(http.StatusConflict, status)
	s.Contains(string(payload), codeCampaignLocked)
}

func (s *CampaignsSuite) TestResumingACampaignWhoseConfigWasDeletedIsRefused() {
	created := s.launch(s.ready())
	s.move(created.Id, "pause")
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+created.ConfigId, nil, nil))

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/resume", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), codeCampaignNotReady)
	s.Equal(CampaignStatePaused, s.get(created.Id).State)
}

func (s *CampaignsSuite) TestResumingARunningCampaignOfAnAppThatMayNotCallIsAConflict() {
	s.useApp(s.data.createApp())
	created := s.launch(s.ready())
	s.Require().NoError(s.store.SuspendTelephony(context.Background(), &store.TelephonySuspension{
		CustomerID: s.customerID(), Reason: "fraud", SuspendedBy: "ops@getstream.io",
	}))

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/resume", nil)

	s.Equal(http.StatusConflict, status, "what is wrong with the request comes before what is wrong with the account")
	s.Contains(string(payload), codeCampaignLocked)
}

// create makes a draft from body.
func (s *CampaignsSuite) create(body map[string]any) Campaign {
	var created Campaign
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns", body, &created))
	return created
}

// ready is a draft that may launch: a config, one of the app's numbers and somebody to ring.
func (s *CampaignsSuite) ready() Campaign {
	created := s.create(map[string]any{
		"name": "campaign-" + s.utils.uuid(), "config_id": s.data.createAgentConfig().Id, "from_number": s.heldNumber(),
	})
	s.setGoal(created, "Confirm the visit")
	s.add(created.Id, map[string]any{"to_number": s.utils.number()})
	return created
}

func (s *CampaignsSuite) setGoal(created Campaign, goal string) {
	sequence := created.Sequence
	sequence.Steps[0].Goal = goal
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence}, nil))
}

// launch starts a draft now.
func (s *CampaignsSuite) launch(created Campaign) Campaign {
	var launched Campaign
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true}, &launched))
	return launched
}

// move pauses, resumes or stops a campaign.
func (s *CampaignsSuite) move(id, action string) Campaign {
	var moved Campaign
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+id+"/"+action, nil, &moved))
	return moved
}

func (s *CampaignsSuite) get(id string) Campaign {
	var campaign Campaign
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/campaigns/"+id, nil, &campaign))
	return campaign
}

// heldNumber is a number the app holds, which a campaign may call from.
func (s *CampaignsSuite) heldNumber() string {
	e164 := s.utils.number()
	s.Require().NoError(s.store.RecordNumber(context.Background(), &store.PhoneNumber{
		E164: e164, Vendor: "twilio", Country: "US", CustomerID: s.customerID(), PurchasedAt: time.Now().UTC(),
	}))
	return e164
}

// backendWithAnotherKey is the app's backend calling with a second key of the app's, and
// that key.
func (s *CampaignsSuite) backendWithAnotherKey() (*testClient, string) {
	other := s.data.keyed(s.app.organization, s.app.app)
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"server": true, "exp": time.Now().Add(time.Hour).Unix(),
	}).SignedString([]byte(other.secret))
	s.Require().NoError(err)
	header := http.Header{}
	header.Set(auth.APIKeyHeader, other.key)
	header.Set("Authorization", "Bearer "+token)
	header.Set(auth.AuthTypeHeader, auth.AuthTypeServer)
	return &testClient{suite: &s.RouterSuite, header: header, kind: server, token: token}, other.key
}

// add puts people on a campaign's list.
func (s *CampaignsSuite) add(campaignID string, contacts ...map[string]any) {
	s.Require().Equal(http.StatusCreated,
		s.serverClient.do(http.MethodPost, "/v1/agents/campaigns/"+campaignID+"/contacts",
			map[string]any{"contacts": contacts}, nil))
}

// contacts is who a campaign holds, and what became of them.
func (s *CampaignsSuite) contacts(campaignID string) []Contact {
	var listed ContactPage
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/campaigns/"+campaignID+"/contacts", nil, &listed))
	return listed.Items
}

func (s *CampaignsSuite) TestAVoicemailSettingIsStoredAndReturned() {
	created := s.create(map[string]any{"name": "may"})
	s.Nil(created.Sequence.Steps[0].Voicemail, "a campaign that never set it shows none")
	sequence := created.Sequence
	sequence.Steps[0].Voicemail = &CampaignVoicemail{Action: VoicemailActionLeaveMessage, Message: "Hi {{name}}, call us back."}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence}, nil))

	got := s.get(created.Id).Sequence.Steps[0].Voicemail
	s.Require().NotNil(got)
	s.Equal(VoicemailActionLeaveMessage, got.Action)
	s.Equal("Hi {{name}}, call us back.", got.Message)
}

func (s *CampaignsSuite) TestAVoicemailMessageOverTheLimitIsRefused() {
	created := s.create(map[string]any{"name": "may"})
	sequence := created.Sequence
	sequence.Steps[0].Voicemail = &CampaignVoicemail{Action: VoicemailActionLeaveMessage, Message: strings.Repeat("a", 1001)}

	status, _ := s.serverClient.failure(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence})

	s.Equal(http.StatusBadRequest, status)
	s.Nil(s.get(created.Id).Sequence.Steps[0].Voicemail)
}

func (s *CampaignsSuite) TestLeavingAMessageWithoutOneIsRefusedAtLaunch() {
	created := s.ready()
	sequence := s.get(created.Id).Sequence
	sequence.Steps[0].Voicemail = &CampaignVoicemail{Action: VoicemailActionLeaveMessage}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence}, nil))

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), codeCampaignNotReady)
	s.Contains(string(payload), "voicemail")
}

func (s *CampaignsSuite) TestLeavingAMessageWithASpeechToSpeechAgentIsRefusedAtLaunch() {
	var native AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Sts: pointerTo("stub/stub-sts"), Instructions: pointerTo("be brief"),
	}, &native))
	created := s.create(map[string]any{"name": "may", "config_id": native.Id, "from_number": s.heldNumber()})
	s.setGoal(created, "Confirm the visit")
	s.add(created.Id, map[string]any{"to_number": s.utils.number()})
	sequence := s.get(created.Id).Sequence
	sequence.Steps[0].Voicemail = &CampaignVoicemail{Action: VoicemailActionLeaveMessage, Message: "Call us back."}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence}, nil))

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/launch",
		map[string]any{"consent": true})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), codeCampaignNotReady)
	s.Contains(string(payload), "voicemail")
	s.Contains(string(payload), "speech-to-speech")

	sequence.Steps[0].Voicemail = &CampaignVoicemail{Action: VoicemailActionHangUp}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/campaigns/"+created.Id,
		map[string]any{"sequence": sequence}, nil))
	s.launch(created)
}
