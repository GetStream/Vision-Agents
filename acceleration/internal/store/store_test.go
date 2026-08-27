//go:build integration

package store

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// DSNEnvVar is where the tests look for a Postgres to run against.
const DSNEnvVar = "ROUTER_POSTGRES_DSN"

type StoreSuite struct {
	suite.Suite
	store *Store
	ctx   context.Context
	// base is the start of a fixed hour, so bucket boundaries are predictable.
	base time.Time
}

func TestStoreSuite(t *testing.T) {
	suite.Run(t, new(StoreSuite))
}

func (s *StoreSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}

	store, err := Open(dsn)
	s.Require().NoError(err)
	s.store = store
	s.ctx = context.Background()
	s.Require().NoError(store.Ping(s.ctx))

	// Start from an empty schema so the embedded migrations are what create the tables.
	_, err = store.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	s.Require().NoError(store.Migrate(s.ctx))

	s.base = time.Date(2026, 3, 1, 10, 0, 0, 0, time.UTC)
}

func (s *StoreSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *StoreSuite) SetupTest() {
	_, err := s.store.DB().ExecContext(
		s.ctx,
		"TRUNCATE requests, stats_hourly, stats_daily, stats_tags_hourly, stats_tags_daily,"+
			" turns, turn_stats_hourly, turn_stats_daily, call_events, phone_numbers, voices CASCADE",
	)
	s.Require().NoError(err)
}

// record stores a speech-to-text request, defaulting the fields a test does not care about.
func (s *StoreSuite) record(customerID string, at time.Time, audioMs int64, latencyMs float64, success bool) {
	request := &Request{
		Modality:   "stt",
		CustomerID: customerID,
		Provider:   "deepgram",
		Model:      "flux-general-en",
		StartedAt:  at,
		AudioMs:    audioMs,
		LatencyMs:  &latencyMs,
		Success:    success,
	}
	if !success {
		request.ErrorCode = "upstream_error"
	}
	s.Require().NoError(s.store.RecordRequest(s.ctx, request))
}

func (s *StoreSuite) TestMigrateIsIdempotent() {
	s.Require().NoError(s.store.Migrate(s.ctx))
}

func (s *StoreSuite) TestRecordRequestRequiresACustomer() {
	err := s.store.RecordRequest(s.ctx, &Request{Modality: "stt", Provider: "deepgram", Model: "flux-general-en"})
	s.ErrorContains(err, "customer id is required")
}

func (s *StoreSuite) TestRecordRequestRequiresAModality() {
	err := s.store.RecordRequest(s.ctx, &Request{CustomerID: "acme", Provider: "deepgram", Model: "flux-general-en"})
	s.ErrorContains(err, "modality is required")
}

func (s *StoreSuite) TestRecordRequestDefaultsStartedAt() {
	request := &Request{
		Modality: "stt", CustomerID: "acme",
		Provider: "deepgram", Model: "flux-general-en", Success: true,
	}

	s.Require().NoError(s.store.RecordRequest(s.ctx, request))

	s.False(request.StartedAt.IsZero(), "a request with no timestamp should be stamped now")
	s.Positive(request.ID, "the row should come back with its generated id")
}

func (s *StoreSuite) TestRollupAggregatesRequestsIntoOneBucket() {
	s.record("acme", s.base.Add(1*time.Minute), 1000, 100, true)
	s.record("acme", s.base.Add(2*time.Minute), 2000, 200, true)
	s.record("acme", s.base.Add(3*time.Minute), 500, 300, false)

	affected, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)
	s.EqualValues(1, affected, "three requests in one hour make one bucket")

	buckets, err := s.store.CustomerStats(s.ctx, "stt", "acme", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)

	bucket := buckets[0]
	s.Equal(s.base, bucket.Bucket.UTC(), "the bucket is the truncated hour")
	s.EqualValues(3500, bucket.AudioMsTotal)
	s.EqualValues(3, bucket.RequestCount)
	s.EqualValues(1, bucket.ErrorCount)
	s.Require().NotNil(bucket.LatencyP50Ms)
	s.InDelta(200.0, *bucket.LatencyP50Ms, 0.001)
	s.Require().NotNil(bucket.Uptime)
	s.InDelta(2.0/3.0, *bucket.Uptime, 0.001, "uptime is successes over total")
}

func (s *StoreSuite) TestRollupSplitsHoursAndKeepsProvidersApart() {
	s.record("acme", s.base.Add(10*time.Minute), 1000, 100, true)
	s.record("acme", s.base.Add(70*time.Minute), 1000, 100, true)

	other := &Request{
		Modality:   "stt",
		CustomerID: "acme",
		Provider:   "parakeet",
		Model:      "parakeet-tdt-0.6b-v3",
		StartedAt:  s.base.Add(20 * time.Minute),
		AudioMs:    4000,
		Success:    true,
	}
	s.Require().NoError(s.store.RecordRequest(s.ctx, other))

	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(3*time.Hour))
	s.Require().NoError(err)

	buckets, err := s.store.CustomerStats(s.ctx, "stt", "acme", Hourly, s.base, s.base.Add(3*time.Hour), nil)
	s.Require().NoError(err)
	s.Len(buckets, 3, "two hours for deepgram plus one for parakeet")

	byProvider := map[string]int64{}
	for _, bucket := range buckets {
		byProvider[bucket.Provider] += bucket.AudioMsTotal
	}
	s.EqualValues(2000, byProvider["deepgram"])
	s.EqualValues(4000, byProvider["parakeet"])
}

func (s *StoreSuite) TestModalitiesAreAggregatedSeparately() {
	s.record("acme", s.base.Add(1*time.Minute), 3000, 100, true)

	spoken := &Request{
		Modality:   "tts",
		CustomerID: "acme",
		Provider:   "elevenlabs",
		Model:      "eleven_flash_v2_5",
		StartedAt:  s.base.Add(2 * time.Minute),
		AudioMs:    1500,
		Characters: 84,
		CostMicros: 4200,
		Success:    true,
	}
	s.Require().NoError(s.store.RecordRequest(s.ctx, spoken))

	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	transcription, err := s.store.CustomerStats(s.ctx, "stt", "acme", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(transcription, 1, "the synthesis row belongs to the other modality")
	s.EqualValues(3000, transcription[0].AudioMsTotal)
	s.Zero(transcription[0].CharactersTotal)

	synthesis, err := s.store.CustomerStats(s.ctx, "tts", "acme", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(synthesis, 1)
	s.EqualValues(84, synthesis[0].CharactersTotal)
	s.EqualValues(4200, synthesis[0].CostMicrosTotal)
	s.EqualValues(1500, synthesis[0].AudioMsTotal)
}

func (s *StoreSuite) TestRollupIsIdempotent() {
	s.record("acme", s.base.Add(1*time.Minute), 1000, 100, true)

	for range 3 {
		_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
		s.Require().NoError(err)
	}

	buckets, err := s.store.CustomerStats(s.ctx, "stt", "acme", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(1, buckets[0].RequestCount, "re-running a rollup recomputes rather than accumulates")
	s.EqualValues(1000, buckets[0].AudioMsTotal)
}

func (s *StoreSuite) TestRollupPicksUpLateArrivingRequests() {
	s.record("acme", s.base.Add(1*time.Minute), 1000, 100, true)
	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	s.record("acme", s.base.Add(2*time.Minute), 1000, 100, true)
	_, err = s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	buckets, err := s.store.CustomerStats(s.ctx, "stt", "acme", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(2, buckets[0].RequestCount)
	s.EqualValues(2000, buckets[0].AudioMsTotal)
}

func (s *StoreSuite) TestDailyRollupCollapsesEveryHourOfTheDay() {
	s.record("acme", s.base, 1000, 100, true)
	s.record("acme", s.base.Add(5*time.Hour), 1000, 100, true)
	s.record("acme", s.base.Add(30*time.Hour), 1000, 100, true)

	_, err := s.store.Rollup(s.ctx, Daily, s.base, s.base.Add(48*time.Hour))
	s.Require().NoError(err)

	buckets, err := s.store.CustomerStats(
		s.ctx, "stt", "acme", Daily, s.base.Add(-24*time.Hour), s.base.Add(72*time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(buckets, 2, "two calendar days")
	s.EqualValues(2000, buckets[0].AudioMsTotal, "the first day holds two requests")
	s.EqualValues(1000, buckets[1].AudioMsTotal)
}

func (s *StoreSuite) TestRollupIgnoresRequestsOutsideTheWindow() {
	s.record("acme", s.base.Add(-2*time.Hour), 9999, 100, true)
	s.record("acme", s.base.Add(10*time.Minute), 1000, 100, true)

	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	buckets, err := s.store.CustomerStats(
		s.ctx, "stt", "acme", Hourly, s.base.Add(-24*time.Hour), s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(1000, buckets[0].AudioMsTotal)
}

func (s *StoreSuite) TestCustomerStatsAreIsolatedPerCustomer() {
	s.record("acme", s.base.Add(1*time.Minute), 1000, 100, true)
	s.record("globex", s.base.Add(1*time.Minute), 5000, 100, true)

	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	acme, err := s.store.CustomerStats(s.ctx, "stt", "acme", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(acme, 1)
	s.EqualValues(1000, acme[0].AudioMsTotal)

	globex, err := s.store.CustomerStats(s.ctx, "stt", "globex", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(globex, 1)
	s.EqualValues(5000, globex[0].AudioMsTotal)
}

func (s *StoreSuite) TestCustomerStatsIsEmptyForAnUnknownCustomer() {
	buckets, err := s.store.CustomerStats(s.ctx, "stt", "nobody", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Empty(buckets)
}

func (s *StoreSuite) TestRollupRejectsBadArguments() {
	_, err := s.store.Rollup(s.ctx, Granularity("weekly"), s.base, s.base.Add(time.Hour))
	s.ErrorContains(err, `unknown granularity "weekly"`)

	_, err = s.store.Rollup(s.ctx, Hourly, s.base, s.base)
	s.ErrorContains(err, "window must be non-empty")
}

func (s *StoreSuite) TestCustomerStatsRejectsBadArguments() {
	_, err := s.store.CustomerStats(s.ctx, "stt", "", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.ErrorContains(err, "customer id is required")

	_, err = s.store.CustomerStats(s.ctx, "", "acme", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.ErrorContains(err, "modality is required")

	_, err = s.store.CustomerStats(s.ctx, "stt", "acme", Granularity("weekly"), s.base, s.base.Add(time.Hour), nil)
	s.ErrorContains(err, "unknown granularity")
}

func (s *StoreSuite) TestOpenRequiresADSN() {
	_, err := Open("")
	s.ErrorContains(err, "dsn is required")
}

func (s *StoreSuite) TestSpendIsBrokenDownByWhateverTheCustomerLabelledIt() {
	s.tagged("acme", s.base.Add(time.Minute), 1000, map[string]string{"project": "support"})
	s.tagged("acme", s.base.Add(2*time.Minute), 3000, map[string]string{"project": "support"})
	s.tagged("acme", s.base.Add(3*time.Minute), 500, map[string]string{"project": "sales"})

	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	buckets, err := s.store.CustomerTagStats(
		s.ctx, "stt", "acme", "project", Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	s.Require().Len(buckets, 2, "one row per project")
	s.Equal("sales", buckets[0].TagValue)
	s.EqualValues(500, buckets[0].AudioMsTotal)
	s.Equal("support", buckets[1].TagValue)
	s.EqualValues(4000, buckets[1].AudioMsTotal)
}

func (s *StoreSuite) TestARequestCarryingTwoLabelsIsCountedUnderBoth() {
	s.tagged("acme", s.base.Add(time.Minute), 1000,
		map[string]string{"project": "support", "environment": "dev"})

	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	byProject, err := s.store.CustomerTagStats(
		s.ctx, "stt", "acme", "project", Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)
	s.Require().Len(byProject, 1)
	s.EqualValues(1000, byProject[0].AudioMsTotal)

	byEnvironment, err := s.store.CustomerTagStats(
		s.ctx, "stt", "acme", "environment", Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)
	s.Require().Len(byEnvironment, 1)
	s.EqualValues(1000, byEnvironment[0].AudioMsTotal, "the same request, seen through the other label")
}

func (s *StoreSuite) TestFilteringByLabelReadsTheRequestsTheRollupHasForgotten() {
	s.tagged("acme", s.base.Add(time.Minute), 1000, map[string]string{"project": "support"})
	s.tagged("acme", s.base.Add(2*time.Minute), 9000, map[string]string{"project": "sales"})

	buckets, err := s.store.CustomerStats(s.ctx, "stt", "acme", Hourly,
		s.base, s.base.Add(time.Hour), map[string]string{"project": "support"})
	s.Require().NoError(err)

	s.Require().Len(buckets, 1)
	s.EqualValues(1000, buckets[0].AudioMsTotal, "only the support request counts")
}

func (s *StoreSuite) TestATurnIsRecordedOnceEvenIfTheWriteIsRetried() {
	turn := s.turn("turn-1", s.base.Add(time.Minute), 120)

	s.Require().NoError(s.store.RecordTurn(s.ctx, turn))
	s.Require().NoError(s.store.RecordTurn(s.ctx, s.turn("turn-1", s.base.Add(time.Minute), 999)))

	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	buckets, err := s.store.CustomerTurnStats(s.ctx, "acme", "", Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(1, buckets[0].TurnCount)
}

func (s *StoreSuite) TestTurnStatsReportThePercentilesOfWhatCallersWaited() {
	for i, roundtrip := range []float64{100, 200, 300, 400} {
		turn := s.turn("turn-"+string(rune('a'+i)), s.base.Add(time.Duration(i)*time.Minute), roundtrip)
		s.Require().NoError(s.store.RecordTurn(s.ctx, turn))
	}

	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	buckets, err := s.store.CustomerTurnStats(s.ctx, "acme", "agent-1", Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(4, buckets[0].TurnCount)
	s.Require().NotNil(buckets[0].RoundtripP50Ms)
	s.InDelta(250, *buckets[0].RoundtripP50Ms, 0.001, "the median of 100,200,300,400")
}

func (s *StoreSuite) TestALegThatNeverHappenedIsNotCountedAsInstant() {
	// A realtime model that hears and speaks for itself has no transcription leg, and
	// counting that as zero would flatter the percentiles.
	turn := s.turn("turn-1", s.base.Add(time.Minute), 300)
	turn.STTLatencyMs = nil
	s.Require().NoError(s.store.RecordTurn(s.ctx, turn))

	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	buckets, err := s.store.CustomerTurnStats(s.ctx, "acme", "", Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.Nil(buckets[0].STTLatencyP50Ms, "no transcription happened, so there is nothing to report")
	s.NotNil(buckets[0].RoundtripP50Ms)
}

func (s *StoreSuite) TestWhatTheConversationDecidedIsReadBackInTheOrderItHappened() {
	// The trail only explains a call if it is in order: a wait followed by an answer is
	// an agent giving somebody time to finish, and the same two the other way round is
	// an agent talking over them.
	decisions := []CallEvent{
		s.decision("wait", "the caller has not finished the thought", s.base.Add(2*time.Second)),
		s.decision("answer", "a complete thought addressed to the agent", s.base.Add(3*time.Second)),
	}
	s.Require().NoError(s.store.RecordCallEvents(s.ctx, decisions))

	read, err := s.store.CallEvents(s.ctx, "acme", "call-1", 0)
	s.Require().NoError(err)
	s.Require().Len(read, 2)
	s.Equal("wait", read[0].Kind)
	s.Equal("the caller has not finished the thought", read[0].Reason)
	s.Equal("answer", read[1].Kind)
	s.Equal("caller", read[1].Participant)
	s.Require().NotNil(read[1].LatencyMs)
	s.InDelta(30, *read[1].LatencyMs, 0.001)
}

func (s *StoreSuite) TestOneCallsReasoningIsNotAnothers() {
	mine := s.decision("answer", "mine", s.base)
	theirs := s.decision("answer", "theirs", s.base)
	theirs.CallID = "call-2"
	s.Require().NoError(s.store.RecordCallEvents(s.ctx, []CallEvent{mine, theirs}))

	read, err := s.store.CallEvents(s.ctx, "acme", "call-1", 0)
	s.Require().NoError(err)
	s.Require().Len(read, 1)
	s.Equal("mine", read[0].Reason)
}

func (s *StoreSuite) TestRecordingNoDecisionsIsNotAnError() {
	s.Require().NoError(s.store.RecordCallEvents(s.ctx, nil))
}

// decision builds one judgement, defaulting what a test does not care about.
func (s *StoreSuite) decision(kind, reason string, at time.Time) CallEvent {
	latency := 30.0
	return CallEvent{
		CustomerID:  "acme",
		CallID:      "call-1",
		AgentID:     "agent-1",
		At:          at,
		Kind:        kind,
		Reason:      reason,
		TurnID:      "turn-1",
		Participant: "caller",
		Said:        "book a table for four",
		LatencyMs:   &latency,
	}
}

func (s *StoreSuite) TestANumberIsHeldUntilItIsReleased() {
	number := &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		Capabilities: []string{"voice"}, MonthlyCostMicros: 1_150_000,
		CustomerID: "acme", PurchasedAt: s.base,
	}
	s.Require().NoError(s.store.RecordNumber(s.ctx, number))

	held, err := s.store.CustomerNumbers(s.ctx, "acme", false)
	s.Require().NoError(err)
	s.Require().Len(held, 1)
	s.Equal("+15125551234", held[0].E164)
	s.Equal([]string{"voice"}, held[0].Capabilities)

	s.Require().NoError(s.store.ReleaseNumber(s.ctx, "acme", "+15125551234", s.base.Add(time.Hour)))

	held, err = s.store.CustomerNumbers(s.ctx, "acme", false)
	s.Require().NoError(err)
	s.Empty(held, "a released number is not one you can be called on")

	// The row stays, because what it cost while it was held is still part of the bill.
	all, err := s.store.CustomerNumbers(s.ctx, "acme", true)
	s.Require().NoError(err)
	s.Require().Len(all, 1)
	s.Require().NotNil(all[0].ReleasedAt)
}

func (s *StoreSuite) TestANumberSomebodyElseHoldsCannotBeReleased() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		CustomerID: "acme", PurchasedAt: s.base,
	}))

	err := s.store.ReleaseNumber(s.ctx, "globex", "+15125551234", s.base)

	s.ErrorContains(err, "is not a number globex holds")
}

func (s *StoreSuite) TestANumberRemembersWhichTrunkItsCallsArriveOn() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		CustomerID: "acme", PurchasedAt: s.base,
	}))

	s.Require().NoError(s.store.AttachNumber(
		s.ctx, "acme", "+15125551234", "trunk-7", "default", "phone-+15125551234"))

	number, err := s.store.Number(s.ctx, "acme", "+15125551234")
	s.Require().NoError(err)
	s.Equal("trunk-7", number.StreamTrunkID)
	s.Equal("phone-+15125551234", number.StreamCallID)
	s.Equal("default", number.StreamCallType)
}

func (s *StoreSuite) TestAnArrivingCallNamesTheCustomerWhoseNumberWasRung() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		CustomerID: "acme", PurchasedAt: s.base,
	}))
	s.Require().NoError(s.store.AttachNumber(
		s.ctx, "acme", "+15125551234", "trunk-7", "support", "the-support-line"))

	number, err := s.store.NumberByCall(s.ctx, "support", "the-support-line")

	s.Require().NoError(err)
	s.Equal("acme", number.CustomerID)
	s.Equal("+15125551234", number.E164)
}

func (s *StoreSuite) TestANumberAttachedBeforeItsCallWasRecordedIsStillFound() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		CustomerID: "acme", PurchasedAt: s.base,
	}))
	// What an attach looked like before the call was recorded: a trunk and nothing else.
	_, err := s.store.db.NewUpdate().Model((*PhoneNumber)(nil)).
		Set("stream_trunk_id = ?", "trunk-7").
		Where("e164 = ?", "+15125551234").
		Exec(s.ctx)
	s.Require().NoError(err)

	number, err := s.store.NumberByCall(s.ctx, "default", "phone-+15125551234")

	s.Require().NoError(err)
	s.Equal("acme", number.CustomerID)
}

func (s *StoreSuite) TestACallNoNumberReachesIsNotAttributedToAnybody() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		CustomerID: "acme", PurchasedAt: s.base,
	}))

	_, err := s.store.NumberByCall(s.ctx, "default", "some-video-call")

	s.ErrorContains(err, "no number reaches call default:some-video-call")
}

func (s *StoreSuite) TestAReleasedNumbersCallIsNotAttributedToItsFormerHolder() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		CustomerID: "acme", PurchasedAt: s.base,
	}))
	s.Require().NoError(s.store.AttachNumber(
		s.ctx, "acme", "+15125551234", "trunk-7", "default", "phone-+15125551234"))
	s.Require().NoError(s.store.ReleaseNumber(s.ctx, "acme", "+15125551234", s.base.Add(time.Hour)))

	_, err := s.store.NumberByCall(s.ctx, "default", "phone-+15125551234")

	s.ErrorContains(err, "no number reaches call")
}

func (s *StoreSuite) TestTheSameNumberCanBeBoughtAgainAfterBeingReleased() {
	first := &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		CustomerID: "acme", PurchasedAt: s.base,
	}
	s.Require().NoError(s.store.RecordNumber(s.ctx, first))
	s.Require().NoError(s.store.ReleaseNumber(s.ctx, "acme", "+15125551234", s.base.Add(time.Hour)))

	again := &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		CustomerID: "acme", PurchasedAt: s.base.Add(2 * time.Hour),
	}

	s.Require().NoError(s.store.RecordNumber(s.ctx, again))
}

// tagged stores a labelled request, which is what the tag rollups are built from.
func (s *StoreSuite) tagged(customerID string, at time.Time, audioMs int64, tags map[string]string) {
	latencyMs := 100.0
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality:   "stt",
		CustomerID: customerID,
		Provider:   "deepgram",
		Model:      "flux-general-en",
		Tags:       tags,
		StartedAt:  at,
		AudioMs:    audioMs,
		LatencyMs:  &latencyMs,
		Success:    true,
	}))
}

// turn builds one conversational turn with every leg measured.
func (s *StoreSuite) turn(turnID string, at time.Time, roundtripMs float64) *Turn {
	sttMs, ttftMs, ttfbMs, audioMs := 30.0, 120.0, 90.0, 2000.0
	return &Turn{
		CustomerID:         "acme",
		AgentID:            "agent-1",
		TurnID:             turnID,
		StartedAt:          at,
		STTLatencyMs:       &sttMs,
		LLMTTFTMs:          &ttftMs,
		TTSTTFBMs:          &ttfbMs,
		RoundtripMs:        &roundtripMs,
		SpeechEndToAudioMs: &roundtripMs,
		AudioOutMs:         &audioMs,
	}
}
