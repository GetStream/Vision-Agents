//go:build integration

package api

import (
	"context"
	"encoding/json"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// OrganizationActivitySuite is Stream staff reading how far an organization got with Agents.
// Every test has an organization of its own.
type OrganizationActivitySuite struct {
	RouterSuite
	// staff is Stream's own tool, holding the ops key and no app's credentials.
	staff *testClient
}

func TestOrganizationActivitySuite(t *testing.T) {
	runSuite(t, new(OrganizationActivitySuite))
}

func (s *OrganizationActivitySuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.staff = &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {suiteOpsKey}}, kind: noCredential}
}

func (s *OrganizationActivitySuite) path(organizationID string) string {
	return "/v1/ops/orgs/" + organizationID + "/activity"
}

// read is what staff are told of the organization, read from since when it is not zero.
func (s *OrganizationActivitySuite) read(organizationID string, since time.Time) OrganizationActivity {
	path := s.path(organizationID)
	if !since.IsZero() {
		path += "?since=" + url.QueryEscape(since.Format(time.RFC3339))
	}
	var answer OrganizationActivity
	s.Require().Equal(http.StatusOK, s.staff.do(http.MethodGet, path, nil, &answer))
	return answer
}

// spent records a model call of app's that cost micros.
func (s *OrganizationActivitySuite) spent(appID, organizationID string, micros int64) {
	s.Require().NoError(s.store.RecordRequest(context.Background(), &store.Request{
		Modality: "llm", CustomerID: appID, OrganizationID: organizationID, Provider: "stub", Model: "stub-llm",
		StartedAt: time.Now(), InputTokens: 10, OutputTokens: 5, CostMicros: micros, Billable: true, Success: true,
	}))
}

// appOf is a second app of the suite's organization.
func (s *OrganizationActivitySuite) appOf() string {
	app := store.App{OrganizationID: s.app.organization.ID, Name: "app-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateApp(context.Background(), &app))
	return app.ID
}

func areaNames(answer OrganizationActivity) []string {
	names := make([]string, 0, len(answer.Areas))
	for _, area := range answer.Areas {
		names = append(names, area.Area)
	}
	return names
}

// Defect: the endpoint reads neither table the router writes as an app works, or reads the
// app rather than its organization.
func (s *OrganizationActivitySuite) TestAnAgentSetUpAndTalkedToIsAConversationWithItsModelsCost() {
	status, _ := s.serverClient.with(clientHeader, store.AuditSourceCLI).
		call(http.MethodPost, "/v1/agents/configs", map[string]any{"name": "agent-" + s.utils.uuid()})
	s.Require().Less(status, http.StatusBadRequest)
	s.serverClient.with(clientHeader, store.AuditSourceSDK).
		createSession(CreateSessionRequest{Incognito: pointerTo(true), Llm: pointerTo("llm-flow")})
	s.spent(s.customerID(), s.app.organization.ID, 7)
	took := 800.0
	s.Require().NoError(s.store.RecordRequest(context.Background(), &store.Request{
		Modality: "tts", CustomerID: s.customerID(), OrganizationID: s.app.organization.ID, Provider: "stub",
		Model: "stub-tts", StartedAt: time.Now(), AudioMs: 1500, Characters: 42, DurationMs: &took, CostMicros: 3,
		Billable: true, Success: true,
	}))

	var answer OrganizationActivity
	s.Require().Eventually(func() bool {
		answer = s.read(s.app.organization.ID, time.Time{})
		return slices.Contains(areaNames(answer), activity.AreaAgentSetup) &&
			slices.Contains(areaNames(answer), activity.AreaConversations)
	}, settleFor, 10*time.Millisecond, "the agent's setup and conversation were not read")

	s.Equal(s.app.organization.ID, answer.OrganizationId)
	s.Equal(activityStepConversations, answer.FurthestStep)
	for _, area := range answer.Areas {
		if area.Area == activity.AreaAgentSetup {
			s.Equal(store.AuditSourceCLI, area.Client)
		}
		if area.Area == activity.AreaConversations {
			s.Equal(store.AuditSourceSDK, area.Client)
		}
	}
	s.Require().NotNil(answer.FirstSeen)
	s.Require().NotNil(answer.LastSeen)
	s.False(answer.LastSeen.Before(*answer.FirstSeen))
	s.Equal([]ActivityMeter{
		{Meter: "llm", Requests: 1, InputTokens: 10, OutputTokens: 5, CostMicros: 7},
		{Meter: "tts", Requests: 1, AudioMs: 1500, Characters: 42, DurationMs: 800, CostMicros: 3},
	}, answer.Meters)
}

// Defect: the organization's other apps left out, or another organization's apps let in.
func (s *OrganizationActivitySuite) TestEveryAppOfTheOrganizationIsReadAndNoOtherOrganizations() {
	second := s.appOf()
	stranger := s.data.createApp()
	s.spent(s.customerID(), s.app.organization.ID, 1)
	s.spent(second, s.app.organization.ID, 2)
	s.spent(stranger.app.ID, stranger.organization.ID, 100)
	now := time.Now().UTC()
	s.Require().NoError(s.store.RecordActivity(context.Background(), []store.ActivityDelta{
		{OrganizationID: s.app.organization.ID, AppID: second, Area: activity.AreaWorkers, Client: store.AuditSourceSDK,
			Day: now, Requests: 3, FirstAt: now, LastAt: now},
		{OrganizationID: stranger.organization.ID, AppID: stranger.app.ID, Area: activity.AreaEvals,
			Client: store.AuditSourceSDK, Day: now, Requests: 9, FirstAt: now, LastAt: now},
	}))

	answer := s.read(s.app.organization.ID, time.Time{})

	s.Require().Len(answer.Meters, 1, "%+v", answer.Meters)
	s.Equal(int64(2), answer.Meters[0].Requests)
	s.Equal(int64(3), answer.Meters[0].CostMicros)
	s.Equal([]string{activity.AreaWorkers}, areaNames(answer))
}

// Defect: an organization nothing was recorded for answered 404, or with null lists, so a
// digest cannot say it has not started.
func (s *OrganizationActivitySuite) TestAnOrganizationWithNothingRecordedHasEmptyListsAndNoStep() {
	status, payload := s.staff.call(http.MethodGet, s.path("organization-"+s.utils.uuid()), nil)
	s.Require().Equal(http.StatusOK, status, string(payload))

	var raw map[string]json.RawMessage
	s.Require().NoError(json.Unmarshal(payload, &raw))
	s.JSONEq(`[]`, string(raw["areas"]))
	s.JSONEq(`[]`, string(raw["meters"]))
	s.JSONEq(`"none"`, string(raw["furthest_step"]))
	s.NotContains(raw, "first_seen")
	s.NotContains(raw, "last_seen")
}

// Defect: the operation reached with an app's credentials, even its backend's, or with a key
// that is not the ops key.
func (s *OrganizationActivitySuite) TestOnlyTheOpsKeyReadsAnOrganizationsActivity() {
	path := s.path(s.app.organization.ID)
	wrong := &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {"not-" + suiteOpsKey}}, kind: noCredential}

	s.Equal(http.StatusUnauthorized, s.serverClient.do(http.MethodGet, path, nil, nil))
	// An end user is turned away sooner, as from every operation that is not client-side.
	s.Equal(http.StatusForbidden, s.client.do(http.MethodGet, path, nil, nil))
	s.Equal(http.StatusUnauthorized, s.unauthenticatedClient.do(http.MethodGet, path, nil, nil))
	s.Equal(http.StatusUnauthorized, wrong.do(http.MethodGet, path, nil, nil))
	s.Equal(http.StatusOK, s.staff.do(http.MethodGet, path, nil, nil))
}

// Defect: a read reaching back further than every shard's requests should be scanned for.
func (s *OrganizationActivitySuite) TestSinceIsAtMostNinetyDaysAgo() {
	path := s.path(s.app.organization.ID) + "?since="

	status, payload := s.staff.call(http.MethodGet, path+url.QueryEscape(time.Now().Add(-91*24*time.Hour).Format(time.RFC3339)), nil)
	s.Equal(http.StatusBadRequest, status, string(payload))
	s.Contains(string(payload), "90 days")
	since := time.Now().Add(-89 * 24 * time.Hour).Truncate(time.Second)
	s.True(since.Equal(s.read(s.app.organization.ID, since).Since))
}

// Defect: a read that names no start reading another window than the last thirty days.
func (s *OrganizationActivitySuite) TestAReadNamingNoStartReadsTheLastThirtyDays() {
	asked := time.Now()

	since := s.read(s.app.organization.ID, time.Time{}).Since

	s.WithinDuration(asked.Add(-30*24*time.Hour), since, time.Minute)
}

// Defect: a request before since counted in the meters.
func (s *OrganizationActivitySuite) TestARequestBeforeSinceIsNotCounted() {
	s.spent(s.customerID(), s.app.organization.ID, 1)

	answer := s.read(s.app.organization.ID, time.Now().Add(time.Hour))

	s.Empty(answer.Meters)
}

// counted is a day of requests of the suite's app in area, its first and last at first and last.
func (s *OrganizationActivitySuite) counted(area string, first, last time.Time) store.ActivityDelta {
	return store.ActivityDelta{
		OrganizationID: s.app.organization.ID, AppID: s.customerID(), Area: area, Client: store.AuditSourceSDK,
		Day: first, Requests: 1, FirstAt: first, LastAt: last,
	}
}

// Defect: first_seen and last_seen read off the first or last area in the list, which is by
// name, rather than the earliest and latest of every area.
func (s *OrganizationActivitySuite) TestFirstAndLastSeenAreTheEarliestAndLatestOfAnyArea() {
	day := time.Now().UTC().Truncate(24 * time.Hour)
	s.Require().NoError(s.store.RecordActivity(context.Background(), []store.ActivityDelta{
		s.counted(activity.AreaAgentSetup, day.Add(2*time.Hour), day.Add(9*time.Hour)),
		s.counted(activity.AreaConversations, day.Add(time.Hour), day.Add(3*time.Hour)),
		s.counted(activity.AreaWorkers, day.Add(4*time.Hour), day.Add(5*time.Hour)),
	}))

	answer := s.read(s.app.organization.ID, time.Time{})

	s.Require().NotNil(answer.FirstSeen)
	s.Require().NotNil(answer.LastSeen)
	s.True(day.Add(time.Hour).Equal(*answer.FirstSeen), "first seen %s", answer.FirstSeen)
	s.True(day.Add(9*time.Hour).Equal(*answer.LastSeen), "last seen %s", answer.LastSeen)
}

// Defect: steady use read from conversations alone, so an organization calling the models on
// three days reads as one that never got anywhere.
func (s *OrganizationActivitySuite) TestModelCallsOnThreeDaysAreSteadyUse() {
	day := time.Now().UTC().Truncate(24 * time.Hour)
	var deltas []store.ActivityDelta
	for n := range 3 {
		at := day.Add(-time.Duration(n) * 24 * time.Hour)
		deltas = append(deltas, s.counted(activity.AreaModelAPI, at, at))
	}
	s.Require().NoError(s.store.RecordActivity(context.Background(), deltas))

	s.Equal(activityStepSteadyUse, s.read(s.app.organization.ID, time.Time{}).FurthestStep)
}

// Defect: an app of another organization whose id equals this organization's id is summed into
// this organization's meters, because the organization's id is always read as an app.
func (s *OrganizationActivitySuite) TestAnAppOfAnotherOrganizationWithThisOrganizationsIdIsNotRead() {
	stranger := s.data.createApp()
	app := store.App{ID: s.app.organization.ID, OrganizationID: stranger.organization.ID, Name: "app-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateApp(context.Background(), &app))
	s.spent(app.ID, stranger.organization.ID, 100)

	s.Empty(s.read(s.app.organization.ID, time.Time{}).Meters)
	s.Len(s.read(stranger.organization.ID, time.Time{}).Meters, 1)
}

// Defect: an app registered under another organization but last seen under this
// one, which is this one's, left out because its registration says otherwise.
func (s *OrganizationActivitySuite) TestAnAppLastSeenUnderThisOrganizationIsReadWhateverItsRegistrationSays() {
	stranger := s.data.createApp()
	app := store.App{ID: s.app.organization.ID, OrganizationID: stranger.organization.ID, Name: "app-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateApp(context.Background(), &app))
	s.Require().NoError(s.store.JoinOrganization(context.Background(), app.ID, s.app.organization.ID))
	s.spent(app.ID, s.app.organization.ID, 100)

	s.Require().Len(s.read(s.app.organization.ID, time.Time{}).Meters, 1)
}

// Defect: an app registered under this organization with this organization's id, whose latest
// request named another organization, is summed into this one, because its registration alone
// decides.
func (s *OrganizationActivitySuite) TestAnAppRegisteredHereAndLastSeenElsewhereIsNotRead() {
	stranger := s.data.createApp()
	app := store.App{ID: s.app.organization.ID, OrganizationID: s.app.organization.ID, Name: "app-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateApp(context.Background(), &app))
	s.Require().NoError(s.store.JoinOrganization(context.Background(), app.ID, stranger.organization.ID))
	s.spent(app.ID, stranger.organization.ID, 100)

	s.Empty(s.read(s.app.organization.ID, time.Time{}).Meters)
	s.Len(s.read(stranger.organization.ID, time.Time{}).Meters, 1)
}

// Defect: an app with this organization's id and no apps row, whose latest request named
// another organization, is summed into this one, because no registration contradicts it.
func (s *OrganizationActivitySuite) TestAnUnregisteredAppLastSeenElsewhereIsNotRead() {
	stranger := s.data.createApp()
	s.Require().NoError(s.store.JoinOrganization(context.Background(), s.app.organization.ID, stranger.organization.ID))
	s.spent(s.app.organization.ID, stranger.organization.ID, 100)

	s.Empty(s.read(s.app.organization.ID, time.Time{}).Meters)
}

// Defect: the app whose id is the organization's, with no apps row and no latest-request row,
// is dropped, so an organization's own first calls are not told.
func (s *OrganizationActivitySuite) TestAnUnregisteredOwnAppWithNoLatestRequestIsRead() {
	s.spent(s.app.organization.ID, s.app.organization.ID, 100)

	s.Require().Len(s.read(s.app.organization.ID, time.Time{}).Meters, 1)
}

// readAt is the suite organization's activity read from since, in the offset since carries.
func (s *OrganizationActivitySuite) readAt(since time.Time) OrganizationActivity {
	return s.read(s.app.organization.ID, since)
}

// Defect: a named since ignored when areas are read, so an area before it is still told.
func (s *OrganizationActivitySuite) TestANamedSinceLeavesOutAreasBeforeIt() {
	old := time.Now().UTC().Add(-10 * 24 * time.Hour)
	s.Require().NoError(s.store.RecordActivity(context.Background(), []store.ActivityDelta{
		s.counted(activity.AreaAgentSetup, old, old),
	}))
	s.Equal(activityStepAgentSetup, s.readAt(time.Time{}).FurthestStep)

	answer := s.readAt(time.Now().Add(-24 * time.Hour))
	s.Empty(answer.Areas)
	s.Equal(activityStepNone, answer.FurthestStep)
}

// Defect: a named since ignored when the days of use are counted, so days before it make an
// organization a steady user.
func (s *OrganizationActivitySuite) TestANamedSinceLeavesOutDaysBeforeIt() {
	day := time.Now().UTC().Truncate(24 * time.Hour)
	var deltas []store.ActivityDelta
	for n := range 3 {
		at := day.Add(-time.Duration(n+5) * 24 * time.Hour)
		deltas = append(deltas, s.counted(activity.AreaModelAPI, at, at))
	}
	s.Require().NoError(s.store.RecordActivity(context.Background(), deltas))
	s.Equal(activityStepSteadyUse, s.readAt(time.Time{}).FurthestStep)

	s.Equal(activityStepNone, s.readAt(time.Now().Add(-2*24*time.Hour)).FurthestStep)
}

// Defect: a since written with a non-UTC offset read from the day of its own offset, not its
// UTC day, so areas of the UTC day before it are told.
func (s *OrganizationActivitySuite) TestASinceInAnotherOffsetReadsFromItsUTCDay() {
	day := time.Now().UTC().Truncate(24 * time.Hour).Add(-3 * 24 * time.Hour)
	before := day.Add(-2 * time.Hour)
	s.Require().NoError(s.store.RecordActivity(context.Background(), []store.ActivityDelta{
		s.counted(activity.AreaAgentSetup, before, before),
	}))
	// 01:00 UTC on day, which at -05:00 is still the evening of the day before.
	since := day.Add(time.Hour).In(time.FixedZone("west", -5*3600))
	s.Empty(s.readAt(since).Areas)
}

// Defect: the same offset mistake when the days of use are counted.
func (s *OrganizationActivitySuite) TestASinceInAnotherOffsetCountsDaysFromItsUTCDay() {
	day := time.Now().UTC().Truncate(24 * time.Hour).Add(-2 * 24 * time.Hour)
	var deltas []store.ActivityDelta
	for n := -1; n <= 1; n++ {
		at := day.Add(time.Duration(n)*24*time.Hour + 2*time.Hour)
		deltas = append(deltas, s.counted(activity.AreaModelAPI, at, at))
	}
	s.Require().NoError(s.store.RecordActivity(context.Background(), deltas))
	since := day.Add(time.Hour).In(time.FixedZone("west", -5*3600))
	s.Equal(activityStepNone, s.readAt(since).FurthestStep)
}

// Defect: the answer silent on which shards its meters were read from, so a deployment that
// reads only its own shard looks complete. Without postgres.shards the one shard is named by a
// digest of its DSN, never by the DSN.
func (s *OrganizationActivitySuite) TestTheAnswerNamesTheShardsItsMetersWereReadFrom() {
	answer := s.read(s.app.organization.ID, time.Time{})

	s.Require().Len(answer.Shards, 1)
	s.Regexp(`^dsn:[0-9a-f]{16}$`, answer.Shards[0])
}

// Defect: adding sessions and latency changed a field the answer already had, its name, its
// value or its shape. The keys and values here were read off the answer before either was added.
func (s *OrganizationActivitySuite) TestTheFieldsTheAnswerHadBeforeSessionsAndLatencyAreUnchanged() {
	day := time.Now().UTC().Truncate(24 * time.Hour)
	first, last, refused := day.Add(time.Hour), day.Add(3*time.Hour), day.Add(2*time.Hour)
	s.Require().NoError(s.store.RecordActivity(context.Background(), []store.ActivityDelta{{
		OrganizationID: s.app.organization.ID, AppID: s.customerID(), Area: activity.AreaConversations,
		Client: store.AuditSourceSDK, Day: first, Requests: 4, Refused: 1, LastRefusalCode: "no_credit",
		LastRefusalAt: refused, FirstAt: first, LastAt: last,
	}}))
	s.spent(s.customerID(), s.app.organization.ID, 7)
	since := day.Add(-24 * time.Hour)

	status, payload := s.staff.call(http.MethodGet,
		s.path(s.app.organization.ID)+"?since="+url.QueryEscape(since.Format(time.RFC3339)), nil)
	s.Require().Equal(http.StatusOK, status, string(payload))

	var raw map[string]json.RawMessage
	s.Require().NoError(json.Unmarshal(payload, &raw))
	before := map[string]json.RawMessage{}
	for key, value := range raw {
		if key != "sessions" && key != "latency" {
			before[key] = value
		}
	}
	s.ElementsMatch([]string{"duration", "organization_id", "since", "furthest_step", "first_seen", "last_seen",
		"areas", "meters", "shards"}, slices.Collect(maps.Keys(before)))
	stamp := func(at time.Time) string { return `"` + at.Format(time.RFC3339) + `"` }
	s.JSONEq(`"`+s.app.organization.ID+`"`, string(before["organization_id"]))
	s.JSONEq(stamp(since), string(before["since"]))
	s.JSONEq(`"conversations"`, string(before["furthest_step"]))
	s.JSONEq(stamp(first), string(before["first_seen"]))
	s.JSONEq(stamp(last), string(before["last_seen"]))
	s.JSONEq(`[{"area":"conversations","client":"sdk","first_at":`+stamp(first)+`,"last_at":`+stamp(last)+
		`,"requests":4,"refused":1,"last_refusal_code":"no_credit"}]`, string(before["areas"]))
	s.JSONEq(`[{"meter":"llm","requests":1,"input_tokens":10,"output_tokens":5,"audio_ms":0,"characters":0,`+
		`"duration_ms":0,"cost_micros":7}]`, string(before["meters"]))
	s.Regexp(`^\["dsn:[0-9a-f]{16}"\]$`, string(before["shards"]))
}
