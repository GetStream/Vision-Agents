//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type ServerSuite struct {
	RouterSuite

	// base is the hour every statistic in this suite falls in. A fixed hour keeps the
	// windows a test asks for readable, and the app is new in every test, so nothing else
	// is counted in it.
	base time.Time
}

func TestServerSuite(t *testing.T) {
	runSuite(t, new(ServerSuite))
}

// SetupSuite records a request for the model this deployment routes to, so that whenever
// this suite first counts popularity, there is something for the model to have a share of.
func (s *ServerSuite) SetupSuite() {
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.RecordRequest(context.Background(), &store.Request{
		Modality: "stt", CustomerID: s.utils.uuid(),
		Provider: "stub", Model: "stub-model",
		StartedAt: time.Now(), Success: true,
	}))
}

// SetupTest gives every test an app of its own, because a statistic is everything one
// customer spent.
func (s *ServerSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.base = time.Date(2026, 4, 1, 9, 0, 0, 0, time.UTC)
}

func (s *ServerSuite) TestHealthReportsEveryDependencyAsOk() {
	var status HealthStatus
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/health", nil, &status))

	s.Equal(Ok, status.Status)
	s.Equal("ok", status.Dependencies["postgres"])
	s.Equal("ok", status.Dependencies["redis"])
	s.Equal("ok", status.Dependencies["stt"])
	s.Equal("ok", status.Dependencies["tts"])
}

func (s *ServerSuite) TestAnAnswerSaysHowLongTheServerTook() {
	round, response, body := s.timed(http.MethodGet, "/health")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	// The point of reporting it: what the server spent is a part of what the caller
	// waited, and the difference is the network.
	spent := s.durationOf(body)
	s.Positive(spent)
	s.Less(spent, round.Seconds()*1000)
	s.InDelta(spent, s.stampOf(response), 1, "the header and the body disagree")
}

func (s *ServerSuite) TestARefusalSaysHowLongItTookToRefuse() {
	_, response, body := s.timed(http.MethodGet, "/v1/agents/sessions/"+s.utils.uuid())
	s.Require().Equal(http.StatusNotFound, response.StatusCode)

	s.Positive(s.durationOf(body))
	s.Positive(s.stampOf(response))
}

func (s *ServerSuite) TestAnAnswerThatIsNotOneDocumentIsOnlyStamped() {
	_, response, body := s.timed(http.MethodGet, "/v1/data/export")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	// A record per line has nowhere to name a duration, so it is left as it was and the
	// header carries the whole of the answer.
	s.Positive(s.stampOf(response))
	s.NotContains(string(body), `"duration"`)
}

// timed makes one request and reports how long the caller waited for it, alongside the
// answer.
func (s *ServerSuite) timed(method, path string) (time.Duration, *http.Response, []byte) {
	request, err := http.NewRequest(method, s.server.URL+path, nil)
	s.Require().NoError(err)
	request.Header = s.serverClient.header.Clone()

	started := time.Now()
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return time.Since(started), response, body
}

// durationOf is the milliseconds a JSON answer says the server spent.
func (s *ServerSuite) durationOf(body []byte) float64 {
	var answered struct {
		Duration string `json:"duration"`
	}
	s.Require().NoError(json.Unmarshal(body, &answered), string(body))
	s.Require().NotEmpty(answered.Duration, "no duration in %s", string(body))

	spent, err := time.ParseDuration(answered.Duration)
	s.Require().NoError(err)
	return float64(spent) / float64(time.Millisecond)
}

// stampOf is the milliseconds the Server-Timing header reports.
func (s *ServerSuite) stampOf(response *http.Response) float64 {
	stamp := response.Header.Get("Server-Timing")
	s.Require().NotEmpty(stamp)

	var spent float64
	_, err := fmt.Sscanf(stamp, "app;dur=%f", &spent)
	s.Require().NoError(err, stamp)
	return spent
}

func (s *ServerSuite) TestRollupThenStatsReportsTheCustomersUsage() {
	s.recordTurn(s.base.Add(5*time.Minute), 3000, 120, true)
	s.recordTurn(s.base.Add(6*time.Minute), 2000, 240, true)
	s.recordTurn(s.base.Add(7*time.Minute), 1000, 360, false)

	var rollup RollupResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/stats/rollup",
		map[string]any{"granularity": "hourly", "from": s.from(), "to": s.to(time.Hour)}, &rollup))
	s.Equal(GranularityHourly, rollup.Granularity)
	s.Positive(rollup.BucketsWritten)

	buckets := s.stats("stt", "granularity=hourly", time.Hour)
	s.Require().Len(buckets, 1)

	bucket := buckets[0]
	s.Equal("stub", bucket.Provider)
	s.Equal("stub-model", bucket.Model)
	s.EqualValues(6000, bucket.AudioMsTotal, "audio duration is what providers bill for")
	s.EqualValues(3, bucket.RequestCount)
	s.EqualValues(1, bucket.ErrorCount)
	s.Require().NotNil(bucket.LatencyP50Ms)
	s.InDelta(240.0, *bucket.LatencyP50Ms, 0.001)
	s.Require().NotNil(bucket.Uptime)
	s.InDelta(2.0/3.0, *bucket.Uptime, 0.001)
}

func (s *ServerSuite) TestStatsAreReportedPerModality() {
	s.recordTurn(s.base.Add(5*time.Minute), 3000, 100, true)
	s.recordSynthesis(s.base.Add(6*time.Minute), 128, 6400)
	s.rollup(time.Hour)

	transcription := s.stats("stt", "", time.Hour)
	s.Require().Len(transcription, 1, "the synthesis belongs to the other modality")
	s.EqualValues(3000, transcription[0].AudioMsTotal)

	synthesis := s.stats("tts", "", time.Hour)
	s.Require().Len(synthesis, 1)
	s.Equal("stub", synthesis[0].Provider)
	s.EqualValues(128, synthesis[0].CharactersTotal)
	s.EqualValues(6400, synthesis[0].CostMicrosTotal, "cost is aggregated alongside usage")
}

func (s *ServerSuite) TestStatsAreScopedToTheCallingCustomer() {
	s.recordTurn(s.base.Add(5*time.Minute), 3000, 100, true)

	// Another customer's traffic in the same bucket must not show up.
	s.Require().NoError(s.store.RecordRequest(context.Background(), &store.Request{
		Modality:   "stt",
		CustomerID: s.utils.uuid(),
		Provider:   "stub",
		Model:      "stub-model",
		StartedAt:  s.base.Add(5 * time.Minute),
		AudioMs:    99000,
		Success:    true,
	}))
	s.rollup(time.Hour)

	buckets := s.stats("stt", "", time.Hour)
	s.Require().Len(buckets, 1)
	s.EqualValues(3000, buckets[0].AudioMsTotal)
}

func (s *ServerSuite) TestDailyGranularityCollapsesTheHours() {
	s.recordTurn(s.base.Add(1*time.Hour), 1000, 100, true)
	s.recordTurn(s.base.Add(6*time.Hour), 2000, 100, true)

	var rollup RollupResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/stats/rollup",
		map[string]any{"granularity": "daily", "from": s.from(), "to": s.to(24 * time.Hour)}, &rollup))
	s.Equal(GranularityDaily, rollup.Granularity)

	path := fmt.Sprintf("/v1/stt/stats?granularity=daily&from=%s&to=%s",
		s.base.Add(-24*time.Hour).Format(time.RFC3339), s.to(24*time.Hour))
	var buckets []StatsBucket
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, path, nil, &buckets))
	s.Require().Len(buckets, 1, "both hours belong to the same day")
	s.EqualValues(3000, buckets[0].AudioMsTotal)
}

func (s *ServerSuite) TestSpendCoversEveryModalityWithoutARollup() {
	s.recordTurn(s.base.Add(5*time.Minute), 3000, 120, true)
	s.recordSynthesis(s.base.Add(6*time.Minute), 128, 6400)

	buckets := s.spend("granularity=daily")
	s.Require().Len(buckets, 2, "one group per modality, with no rollup having run")
	s.Equal("tts", buckets[0].Value, "the synthesis is what cost money")
	s.EqualValues(6400, buckets[0].CostMicrosTotal)
	s.Equal("stt", buckets[1].Value)
	s.EqualValues(1, buckets[1].RequestCount)
}

func (s *ServerSuite) TestSpendCanBeGroupedByACostLabel() {
	s.recordLabelled(s.base.Add(5*time.Minute), 6000, map[string]string{"product": "support"})
	s.recordLabelled(s.base.Add(6*time.Minute), 2000, map[string]string{"product": "sales"})
	s.recordSynthesis(s.base.Add(7*time.Minute), 64, 100)

	buckets := s.spend("group_by=product&granularity=daily")
	s.Require().Len(buckets, 3)
	s.Equal("support", buckets[0].Value)
	s.EqualValues(6000, buckets[0].CostMicrosTotal)
	s.Equal("sales", buckets[1].Value)
	s.Equal("", buckets[2].Value, "the synthesis carries no product, and is still part of the bill")
	s.EqualValues(100, buckets[2].CostMicrosTotal)
}

func (s *ServerSuite) TestTagKeysReportWhichLabelIsWorthABreakdown() {
	s.recordLabelled(s.base.Add(5*time.Minute), 6000,
		map[string]string{"product": "support", "environment": "production"})
	s.recordLabelled(s.base.Add(6*time.Minute), 2000,
		map[string]string{"product": "sales", "environment": "production"})

	var keys []TagKeySummary
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/stats/tags/keys?from="+s.from()+"&to="+s.to(24*time.Hour), nil, &keys))
	s.Require().Len(keys, 2)

	byKey := map[string]TagKeySummary{}
	for _, key := range keys {
		byKey[key.Key] = key
	}
	s.EqualValues(2, byKey["product"].ValueCount)
	s.InDelta(1.0, byKey["product"].Coverage, 0.001, "every request carries it")
	s.Require().Len(byKey["product"].TopValues, 2)
	s.Equal("support", byKey["product"].TopValues[0].Value, "biggest spend first")
	s.InDelta(0.75, byKey["product"].TopValues[0].Share, 0.001)
	s.EqualValues(1, byKey["environment"].ValueCount, "one value is context, not a breakdown")
}

func (s *ServerSuite) TestActivityReportsWhoUsedTheAgentsAndHowMuch() {
	ctx := context.Background()
	session := &store.AgentSession{
		ID:         s.utils.uuid(),
		CustomerID: s.customerID(),
		AgentName:  "docs",
		UserID:     "randy",
		CallerKind: "authenticated",
		State:      store.SessionRunning,
		CreatedAt:  s.base.Add(time.Minute),
	}
	s.Require().NoError(s.store.SaveSession(ctx, session))
	s.Require().NoError(s.store.StartResponse(ctx, &store.AgentResponse{
		ID:         s.utils.uuid(),
		SessionID:  session.ID,
		CustomerID: s.customerID(),
		Said:       "how much does it cost",
		CreatedAt:  s.base.Add(2 * time.Minute),
	}))

	var buckets []ActivityBucket
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/stats/activity?from="+s.from()+"&to="+s.to(24*time.Hour), nil, &buckets))
	s.Require().Len(buckets, 1)
	s.EqualValues(1, buckets[0].Sessions)
	s.EqualValues(1, buckets[0].Messages)
	s.EqualValues(1, buckets[0].ActiveUsers, "one person asked one thing")
	s.Zero(buckets[0].Calls, "nobody rang anybody")
}

func (s *ServerSuite) TestAgentStatsReportThePeakAndTheToolsCalled() {
	ctx := context.Background()
	session := &store.AgentSession{
		ID:         s.utils.uuid(),
		CustomerID: s.customerID(),
		AgentName:  "docs",
		State:      store.SessionRunning,
		CreatedAt:  s.base.Add(time.Minute),
	}
	s.Require().NoError(s.store.SaveSession(ctx, session))
	responseID := s.utils.uuid()
	s.Require().NoError(s.store.StartResponse(ctx, &store.AgentResponse{
		ID:         responseID,
		SessionID:  session.ID,
		CustomerID: s.customerID(),
		CreatedAt:  s.base.Add(2 * time.Minute),
	}))
	s.Require().NoError(s.store.AppendResponseItems(ctx, []store.AgentResponseItem{
		{ResponseID: responseID, Ordinal: 0, SessionID: session.ID, Kind: store.ItemToolCall,
			ToolName: "lookup_order", Payload: map[string]any{"call_id": "c1"}, At: s.base.Add(2 * time.Minute)},
		{ResponseID: responseID, Ordinal: 1, SessionID: session.ID, Kind: store.ItemToolResult,
			ToolName: "lookup_order", Text: "shipped", Payload: map[string]any{"call_id": "c1"},
			At: s.base.Add(2*time.Minute + 250*time.Millisecond)},
	}))

	var stats AgentStats
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/stats/agents?from="+s.from()+"&to="+s.to(24*time.Hour), nil, &stats))
	s.EqualValues(1, stats.PeakConcurrentSessions)
	s.EqualValues(1, stats.ToolCalls.Calls)
	s.Require().Len(stats.Tools, 1)
	s.Equal("lookup_order", stats.Tools[0].Tool)
	s.Require().NotNil(stats.Tools[0].AvgMs)
	s.InDelta(250, *stats.Tools[0].AvgMs, 0.01)
}

func (s *ServerSuite) TestProvidersReportLiveHealth() {
	// Health is kept per model in the Redis every suite shares, and VoicesSuite records a
	// failure for this one (key: live.go, healthKey). Availability is read from all of it, so
	// the test starts from none. The key is the one this test reads and VoicesSuite writes,
	// so only deleting it starts clean; no other key is touched.
	redis := s.live.ShardRedis()
	s.Require().NoError(redis.Do(s.T().Context(),
		redis.B().Del().Key("health:stt:stub:stub-model").Build()).Error())
	s.Require().NoError(s.live.RecordRequest(context.Background(), live.Usage{
		Modality: "stt", CustomerID: s.customerID(),
		Provider: "stub", Model: "stub-model",
		LatencyMs: 150, AudioMs: 1000, Success: true,
	}))

	var providers []Provider
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/stt/providers", nil, &providers))

	var routed *Provider
	for i := range providers {
		if providers[i].Model == "stub-model" {
			routed = &providers[i]
		}
	}
	s.Require().NotNil(routed)
	s.Positive(routed.Health.Requests, "health should come from the live counters")
	s.True(routed.Health.Available)
}

func (s *ServerSuite) TestAProviderShowsTheTokenRatesItIsBilledAt() {
	var providers []Provider
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/llm/providers", nil, &providers))

	prices := map[string]*ProviderPrice{}
	for _, provider := range providers {
		prices[provider.Provider+"/"+provider.Model] = provider.Price
	}
	s.Require().Contains(prices, "counted/counted-model")
	s.Require().NotNil(prices["counted/counted-model"])
	s.Equal(pointerTo(1.0), prices["counted/counted-model"].PerMillionInputTokens)
	s.Equal(pointerTo(10.0), prices["counted/counted-model"].PerMillionOutputTokens)
	s.Nil(prices["stub/stub-model"], "a model billed by no token has no token rates")
}

func (s *ServerSuite) TestAModelThatServedRequestsHasAShareOfThem() {
	var providers []Provider
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/stt/providers", nil, &providers))

	s.Require().NotEmpty(providers)
	for _, provider := range providers {
		s.Require().NotNil(provider.UsageShare)
		s.GreaterOrEqual(*provider.UsageShare, 0.0)
		s.LessOrEqual(*provider.UsageShare, 1.0)
		if provider.Model == "stub-model" {
			s.Positive(*provider.UsageShare, "another customer's requests count towards popularity")
		}
	}
}

func (s *ServerSuite) TestAModelListsTheOptionsItCanBeAskedFor() {
	var providers []Provider
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/stt/providers", nil, &providers))

	var supports []string
	for _, provider := range providers {
		if provider.Model == "stub-model" {
			supports = provider.Supports
		}
	}
	s.Contains(supports, "keyterms", "the stub declares every term")
}

func (s *ServerSuite) TestNobodyIsToldWhatAnAppSpentWithoutCredentials() {
	s.Equal(http.StatusUnauthorized, s.unauthenticatedClient.do(http.MethodGet,
		"/v1/stats/spend?granularity=daily&from="+s.from()+"&to="+s.to(24*time.Hour), nil, nil))
}

// recordWho stores billable work of app, billed to organization, for user under categories.
func (s *ServerSuite) recordWho(app, organization string, costMicros int64, user string, categories ...string) {
	s.Require().NoError(s.store.RecordRequest(context.Background(), &store.Request{
		Modality: "llm", CustomerID: app, OrganizationID: organization, APIKey: "key-" + app, UserID: user,
		UsageCategories: categories, Provider: "stub", Model: "stub-llm", StartedAt: s.base.Add(5 * time.Minute),
		CostMicros: costMicros, Billable: true, Success: true,
	}))
}

// sibling is another app of the suite's app's organization.
func (s *ServerSuite) sibling() string {
	app := store.App{OrganizationID: s.app.organization.ID, Name: "app-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateApp(context.Background(), &app))
	return app.ID
}

// spentBy is each group's spend, by value.
func spentBy(buckets []SpendBucket) map[string]int64 {
	spent := map[string]int64{}
	for _, bucket := range buckets {
		spent[bucket.Value] += bucket.CostMicrosTotal
	}
	return spent
}

func (s *ServerSuite) TestSpendByUserAcrossTheOrganization() {
	organization, sibling := s.app.organization.ID, s.sibling()
	s.recordWho(s.customerID(), organization, 1_000, "u1")
	s.recordWho(sibling, organization, 2_000, "u1")
	s.recordWho(sibling, organization, 4_000, "")
	s.recordWho(s.data.createApp().app.ID, "another-"+s.utils.uuid(), 8_000, "u1")

	s.Equal(map[string]int64{s.customerID() + "/u1": 1_000, sibling + "/u1": 2_000, "": 4_000},
		spentBy(s.spend("group_by=who%3Auser&granularity=daily&scope=organization")),
		"one user id in two apps is two users, and another organization's work is not counted")
	s.Equal(map[string]int64{"u1": 1_000}, spentBy(s.spend("group_by=who%3Auser&granularity=daily")), "the app alone, by default")
	s.Equal(map[string]int64{s.customerID(): 1_000, sibling: 6_000}, spentBy(s.spend("group_by=who%3Aapp&granularity=daily&scope=organization")))
	s.Equal(map[string]int64{"key-" + s.customerID(): 1_000, "key-" + sibling: 6_000},
		spentBy(s.spend("group_by=who%3Aapi_key&granularity=daily&scope=organization")))
}

func (s *ServerSuite) TestSpendByCategoryCountsARowInEach() {
	organization := s.app.organization.ID
	s.recordWho(s.customerID(), organization, 1_000, "", "support", "billing", "team:finance")
	s.recordWho(s.customerID(), organization, 2_000, "")

	s.Equal(map[string]int64{"support": 1_000, "billing": 1_000, "": 2_000},
		spentBy(s.spend("group_by=who%3Acategory&granularity=daily")), "a row counts in each of its categories, and its team in none")
}

func (s *ServerSuite) TestSpendByTeam() {
	organization := s.app.organization.ID
	s.recordWho(s.customerID(), organization, 1_000, "", "team:finance", "support")
	s.recordWho(s.customerID(), organization, 2_000, "", "team:sales")
	s.recordWho(s.customerID(), organization, 4_000, "")

	s.Equal(map[string]int64{"finance": 1_000, "sales": 2_000, "": 4_000}, spentBy(s.spend("group_by=who%3Ateam&granularity=daily")))
	for value := range spentBy(s.spend("group_by=who%3Acategory&granularity=daily")) {
		s.NotContains(value, "team:")
	}
}

func (s *ServerSuite) TestALabelNamedLikeWhoAskedIsStillALabel() {
	s.recordLabelled(s.base.Add(5*time.Minute), 6000, map[string]string{"app": "1234", "team": "moderation"})

	s.Equal(map[string]int64{"1234": 6000}, spentBy(s.spend("group_by=app&granularity=daily")))
	s.Equal(map[string]int64{"moderation": 6000}, spentBy(s.spend("group_by=team&granularity=daily")))
	s.Equal(map[string]int64{s.customerID(): 6000}, spentBy(s.spend("group_by=who%3Aapp&granularity=daily")))
}

func (s *ServerSuite) TestSpendForTheOrganizationIsServerSideOnly() {
	path := "/v1/stats/spend?scope=organization&from=" + s.from() + "&to=" + s.to(24*time.Hour)

	status, failure := s.client.failure(http.MethodGet, path, nil)

	s.Equal(http.StatusForbidden, status, failure)
	s.Contains(failure, "server-side only")
}

// from and to are the window a statistic is asked for, as the API spells it.
func (s *ServerSuite) from() string { return s.base.Format(time.RFC3339) }

func (s *ServerSuite) to(after time.Duration) string {
	return s.base.Add(after).Format(time.RFC3339)
}

// rollup writes the buckets for the window, which is what the stats endpoints read.
func (s *ServerSuite) rollup(window time.Duration) {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/stats/rollup",
		map[string]any{"from": s.from(), "to": s.to(window)}, nil))
}

// stats reads one modality's buckets over the window.
func (s *ServerSuite) stats(modality, options string, window time.Duration) []StatsBucket {
	path := fmt.Sprintf("/v1/%s/stats?from=%s&to=%s", modality, s.from(), s.to(window))
	if options != "" {
		path += "&" + options
	}
	var buckets []StatsBucket
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, path, nil, &buckets))
	return buckets
}

// spend reads what the app spent over a day.
func (s *ServerSuite) spend(options string) []SpendBucket {
	path := fmt.Sprintf("/v1/stats/spend?%s&from=%s&to=%s", options, s.from(), s.to(24*time.Hour))
	status, payload := s.serverClient.call(http.MethodGet, path, nil)
	s.Require().Equal(http.StatusOK, status, string(payload))
	var buckets []SpendBucket
	s.Require().NoError(json.Unmarshal(payload, &buckets))
	return buckets
}

// recordTurn stores a completed speech-to-text turn for the suite's app.
func (s *ServerSuite) recordTurn(at time.Time, audioMs int64, latencyMs float64, success bool) {
	request := &store.Request{
		Modality:   "stt",
		CustomerID: s.customerID(),
		Provider:   "stub",
		Model:      "stub-model",
		StartedAt:  at,
		AudioMs:    audioMs,
		LatencyMs:  &latencyMs,
		Success:    success,
	}
	if !success {
		request.ErrorCode = "provider_fatal"
	}
	s.Require().NoError(s.store.RecordRequest(context.Background(), request))
}

// recordSynthesis stores a completed text-to-speech synthesis for the suite's app.
func (s *ServerSuite) recordSynthesis(at time.Time, characters, costMicros int64) {
	latencyMs := 180.0
	s.Require().NoError(s.store.RecordRequest(context.Background(), &store.Request{
		Modality:   "tts",
		CustomerID: s.customerID(),
		Provider:   "stub",
		Model:      "stub-voice",
		StartedAt:  at,
		AudioMs:    2500,
		Characters: characters,
		CostMicros: costMicros,
		LatencyMs:  &latencyMs,
		Success:    true,
	}))
}

// recordLabelled stores a completed LLM completion carrying the app's own cost labels.
func (s *ServerSuite) recordLabelled(at time.Time, costMicros int64, tags map[string]string) {
	latencyMs := 320.0
	s.Require().NoError(s.store.RecordRequest(context.Background(), &store.Request{
		Modality:     "llm",
		CustomerID:   s.customerID(),
		Provider:     "stub",
		Model:        "stub-llm",
		Tags:         tags,
		StartedAt:    at,
		InputTokens:  400,
		OutputTokens: 120,
		CostMicros:   costMicros,
		LatencyMs:    &latencyMs,
		Success:      true,
	}))
}
