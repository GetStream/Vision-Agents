//go:build integration

package store

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// DSNEnvVar is where the tests look for a Postgres to run against.
const DSNEnvVar = "ROUTER_POSTGRES_DSN"

type StoreSuite struct {
	suite.Suite
	store *Store
	// dsn is the suite's own database, for a test that needs another pool on it, as a second
	// router would have.
	dsn string
	ctx context.Context
	// base is the start of a fixed hour, so bucket boundaries are predictable.
	base time.Time
}

func TestStoreSuite(t *testing.T) {
	suite.Run(t, new(StoreSuite))
}

// ShardedStoreSuite runs every store test again with the shared tables in one database, this
// shard's in another, and a second shard beside it, as a hosted deployment keeps them.
type ShardedStoreSuite struct {
	StoreSuite
	// other is a router of the second shard: the same shared database, the other shard's own.
	other *Store
}

func TestShardedStoreSuite(t *testing.T) {
	suite.Run(t, new(ShardedStoreSuite))
}

func (s *StoreSuite) SetupSuite() {
	s.ctx = context.Background()
	s.dsn = s.emptyDatabase("store")
	store, err := Open(s.dsn)
	s.Require().NoError(err)
	s.store = store
	s.Require().NoError(store.Ping(s.ctx))
	s.Require().NoError(store.Migrate(s.ctx))
	s.base = time.Date(2026, 3, 1, 10, 0, 0, 0, time.UTC)
}

func (s *ShardedStoreSuite) SetupSuite() {
	s.ctx = context.Background()
	s.dsn = s.emptyDatabase("store_global")
	shard, peer := s.emptyDatabase("store_shard"), s.emptyDatabase("store_peer")

	other, err := Open(peer)
	s.Require().NoError(err)
	s.Require().NoError(other.Migrate(s.ctx))
	s.Require().NoError(other.Close())

	shards := map[string]string{"here": shard, "there": peer}
	store, err := OpenShards(Options{DSN: s.dsn, ShardDSN: shard, Shards: shards})
	s.Require().NoError(err)
	s.store = store
	s.Require().NoError(store.Ping(s.ctx))
	s.Require().NoError(store.Migrate(s.ctx))
	s.other, err = OpenShards(Options{DSN: s.dsn, ShardDSN: peer, Shards: shards})
	s.Require().NoError(err)
	s.base = time.Date(2026, 3, 1, 10, 0, 0, 0, time.UTC)
}

func (s *ShardedStoreSuite) TearDownSuite() {
	if s.other != nil {
		s.Require().NoError(s.other.Close())
	}
	s.StoreSuite.TearDownSuite()
}

// emptyDatabase is a database of this suite's own with an empty schema, so the embedded
// migrations are what create the tables. It is the suite's own because it drops the schema and
// empties tables between tests, which is not something to do to a database another package is
// reading.
func (s *StoreSuite) emptyDatabase(name string) string {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	own := testenv.Database(dsn, name)
	db, err := Open(own)
	s.Require().NoError(err)
	defer db.Close()

	var database string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = db.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	return own
}

func (s *StoreSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *StoreSuite) SetupTest() {
	for _, db := range append([]bun.IDB{s.store.db}, s.store.shards()...) {
		_, err := db.ExecContext(
			s.ctx,
			"TRUNCATE requests, stats_hourly, stats_daily, stats_tags_hourly, stats_tags_daily,"+
				" turn_stats_hourly, turn_latency_hourly, calls, phone_numbers, sip_trunks,"+
				" voices, agent_sessions, agent_responses, agent_response_items, users,"+
				" policies, app_organizations, call_resources, stream_trunks, organizations, agent_configs,"+
				" agent_plugin_connections, agent_plugin_clients, data_changes, data_change_capture, connector_definitions, connector_broken_revisions,"+
				" connector_connections, connector_authorization_attempts, connector_oauth_clients,"+
				" connector_config_tokens, channel_threads, channel_thread_messages, channel_thread_waiting,"+
				" connector_event_destinations, connector_event_deliveries, contact_map, episodes,"+
				" connection_event_subscriptions, connection_event_deliveries,"+
				" connector_invocations, connector_audit, simulations, simulation_runs, simulation_cases,"+
				" scheduled_tasks, scheduled_task_runs, price_list_entries, price_changes,"+
				" limits, usage_resets, limit_grants, limit_grant_reconciles, phone_calls CASCADE",
		)
		s.Require().NoError(err)
	}
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

	var buckets []Bucket
	err = s.store.shard.NewSelect().Table("stats_hourly").Where("customer_id = ?", "acme").Scan(s.ctx, &buckets)
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

func (s *StoreSuite) TestModelRequestsCountEveryCustomerSinceTheWindowOpened() {
	s.record("acme", s.base.Add(1*time.Minute), 1000, 100, true)
	s.record("globex", s.base.Add(2*time.Minute), 1000, 100, false)
	s.record("acme", s.base.Add(-time.Hour), 1000, 100, true)
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "llm", CustomerID: "acme", Provider: "openai", Model: "gpt-5.6-sol", StartedAt: s.base, Success: true,
	}))

	counts, err := s.store.ModelRequests(s.ctx, "stt", s.base)
	s.Require().NoError(err)
	s.Equal(map[string]int64{"deepgram/flux-general-en": 2}, counts)
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

func (s *StoreSuite) TestTurnStatsReportThePercentilesOfWhatCallersWaited() {
	for i, roundtrip := range []float64{100, 200, 300, 400} {
		s.Require().NoError(s.store.RecordTurnStats(s.ctx, s.turn(s.base.Add(time.Duration(i)*time.Minute), roundtrip)))
	}

	buckets, err := s.store.CustomerTurnStats(s.ctx, "acme", "agent-1", Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(4, buckets[0].TurnCount)
	s.InDelta(8000, buckets[0].AudioOutMsTotal, 0.001)
	s.Require().NotNil(buckets[0].RoundtripP50Ms)
	s.InEpsilon(200, *buckets[0].RoundtripP50Ms, 0.05, "the median of 100,200,300,400, to the nearest rank")
	s.InEpsilon(400, *buckets[0].RoundtripP95Ms, 0.05)
	s.InEpsilon(120, *buckets[0].LLMTTFTP50Ms, 0.05)
}

func (s *StoreSuite) TestADayIsItsHoursAddedUp() {
	s.Require().NoError(s.store.RecordTurnStats(s.ctx, s.turn(s.base.Add(time.Minute), 100)))
	s.Require().NoError(s.store.RecordTurnStats(s.ctx, s.turn(s.base.Add(5*time.Hour), 100)))
	interrupted := s.turn(s.base.Add(9*time.Hour), 900)
	interrupted.Interrupted = true
	s.Require().NoError(s.store.RecordTurnStats(s.ctx, interrupted))

	hours, err := s.store.CustomerTurnStats(s.ctx, "acme", "", Hourly, s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)
	days, err := s.store.CustomerTurnStats(s.ctx, "acme", "", Daily, s.base, s.base.Add(24*time.Hour))
	s.Require().NoError(err)

	s.Len(hours, 3)
	s.Require().Len(days, 1)
	s.EqualValues(3, days[0].TurnCount)
	s.EqualValues(1, days[0].InterruptedCount)
	s.InEpsilon(100, *days[0].RoundtripP50Ms, 0.05)
	s.InEpsilon(900, *days[0].RoundtripP99Ms, 0.05, "the slow turn is the day's tail")
}

func (s *StoreSuite) TestALegThatNeverHappenedIsNotCountedAsInstant() {
	// A realtime model that hears and speaks for itself has no transcription leg, and
	// counting that as zero would flatter the percentiles.
	turn := s.turn(s.base.Add(time.Minute), 300)
	turn.STTLatencyMs = nil
	s.Require().NoError(s.store.RecordTurnStats(s.ctx, turn))

	buckets, err := s.store.CustomerTurnStats(s.ctx, "acme", "", Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.Nil(buckets[0].STTLatencyP50Ms, "no transcription happened, so there is nothing to report")
	s.NotNil(buckets[0].RoundtripP50Ms)
}

func (s *StoreSuite) TestOneAgentsTurnsAreNotAnothers() {
	s.Require().NoError(s.store.RecordTurnStats(s.ctx, s.turn(s.base.Add(time.Minute), 100)))
	theirs := s.turn(s.base.Add(time.Minute), 900)
	theirs.AgentID = "agent-2"
	s.Require().NoError(s.store.RecordTurnStats(s.ctx, theirs))

	buckets, err := s.store.CustomerTurnStats(s.ctx, "acme", "agent-1", Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(1, buckets[0].TurnCount)
	s.InEpsilon(100, *buckets[0].RoundtripP99Ms, 0.05)
}

func (s *StoreSuite) TestACallReportsTheModelsThatServedIt() {
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "stt", CustomerID: "acme", AgentID: "agent-1",
		Provider: "deepgram", Model: "flux-general-en",
		StartedAt: s.base.Add(time.Second), Success: true,
	}))
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "llm", CustomerID: "acme", AgentID: "agent-1",
		Provider: "gemini", Model: "gemini-3.5-flash-lite",
		StartedAt: s.base.Add(2 * time.Second), Success: true,
	}))
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "llm", CustomerID: "acme", AgentID: "agent-1",
		Provider: "openai", Model: "gpt-5.6-sol",
		StartedAt: s.base.Add(3 * time.Second), Success: true,
	}))
	// A failed start is not what served the call.
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "tts", CustomerID: "acme", AgentID: "agent-1",
		Provider: "cartesia", Model: "sonic-preview",
		StartedAt: s.base.Add(4 * time.Second), Success: false, ErrorCode: "start_failed",
	}))
	// Another agent is not this call.
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "stt", CustomerID: "acme", AgentID: "agent-2",
		Provider: "grok", Model: "grok-stt",
		StartedAt: s.base.Add(2 * time.Second), Success: true,
	}))

	used, err := s.store.CallUsedModels(s.ctx, "acme", "agent-1", s.base, nil)
	s.Require().NoError(err)
	s.Equal([]UsedModel{
		{Modality: "llm", Provider: "openai", Model: "gpt-5.6-sol"},
		{Modality: "llm", Provider: "gemini", Model: "gemini-3.5-flash-lite"},
		{Modality: "stt", Provider: "deepgram", Model: "flux-general-en"},
	}, used)
}

func (s *StoreSuite) TestACallAddsUpWhatEveryOneOfItsRequestsSpent() {
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "llm", CustomerID: "acme", AgentID: "agent-1",
		Provider: "openai", Model: "gpt-5.6-sol", StartedAt: s.base.Add(time.Second),
		InputTokens: 800, CachedInputTokens: 500, OutputTokens: 120,
		CostMicros: 1400, Success: true,
	}))
	// A turn the model read and then failed on is still billed for the reading.
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "llm", CustomerID: "acme", AgentID: "agent-1",
		Provider: "openai", Model: "gpt-5.6-sol", StartedAt: s.base.Add(2 * time.Second),
		InputTokens: 200, OutputTokens: 0, CostMicros: 100,
		Success: false, ErrorCode: "overloaded",
	}))
	// Another agent's spend is not this call's.
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "llm", CustomerID: "acme", AgentID: "agent-2",
		Provider: "openai", Model: "gpt-5.6-sol", StartedAt: s.base.Add(time.Second),
		InputTokens: 9000, OutputTokens: 9000, CostMicros: 90000, Success: true,
	}))

	spent, err := s.store.CallUsage(s.ctx, "acme", "agent-1", "session-1", s.base, nil)

	s.Require().NoError(err)
	s.Equal(CallUsage{
		InputTokens: 1000, CachedInputTokens: 500, OutputTokens: 120,
		CostMicros: 1500, Requests: 2,
	}, spent)
}

func (s *StoreSuite) TestACallsRuntimeIsItsOwnSessionsAndNoRequest() {
	// Two sessions of one agent at once: the call's models are keyed by agent and window, but
	// its time is its own session's.
	for _, row := range []Request{
		{Modality: "llm", SessionID: "session-1", Provider: "openai", Model: "gpt-5.6-sol", InputTokens: 100, CostMicros: 50},
		{Modality: RuntimeModality, SessionID: "session-1", Provider: "stream", Model: "voice", CostMicros: 600},
		{Modality: RuntimeModality, SessionID: "session-2", Provider: "stream", Model: "voice", CostMicros: 900},
	} {
		row.CustomerID, row.AgentID, row.StartedAt, row.Success = "acme", "agent-1", s.base.Add(time.Second), true
		s.Require().NoError(s.store.RecordRequest(s.ctx, &row))
	}

	spent, err := s.store.CallUsage(s.ctx, "acme", "agent-1", "session-1", s.base, nil)
	s.Require().NoError(err)
	s.EqualValues(650, spent.CostMicros, "the model's and this session's time, not the other session's")
	s.EqualValues(1, spent.Requests, "runtime is no request")

	used, err := s.store.CallTokens(s.ctx, "acme", "agent-1", "session-1", s.base, nil)
	s.Require().NoError(err)
	var cost, requests int64
	for _, model := range used {
		cost += model.CostMicros
		requests += model.Requests
	}
	s.EqualValues(650, cost)
	s.EqualValues(1, requests)
}

func (s *StoreSuite) TestACallThatMadeNoRequestsSpentNothing() {
	spent, err := s.store.CallUsage(s.ctx, "acme", "agent-with-no-work", "session-1", s.base, nil)

	s.Require().NoError(err)
	s.Equal(CallUsage{}, spent)
}

func (s *StoreSuite) TestWhatWasSpentAfterACallEndedIsNotTheCallsToPayFor() {
	ended := s.base.Add(10 * time.Second)
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "llm", CustomerID: "acme", AgentID: "agent-1",
		Provider: "openai", Model: "gpt-5.6-sol", StartedAt: s.base.Add(time.Second),
		InputTokens: 300, CostMicros: 400, Success: true,
	}))
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "llm", CustomerID: "acme", AgentID: "agent-1",
		Provider: "openai", Model: "gpt-5.6-sol", StartedAt: ended.Add(time.Minute),
		InputTokens: 7000, CostMicros: 8000, Success: true,
	}))

	spent, err := s.store.CallUsage(s.ctx, "acme", "agent-1", "session-1", s.base, &ended)

	s.Require().NoError(err)
	s.Equal(int64(300), spent.InputTokens)
	s.Equal(int64(1), spent.Requests)
}

func (s *StoreSuite) TestACallRemembersWhoTheAgentSpokeTo() {
	s.Require().NoError(s.store.StartCall(s.ctx, &Call{
		ID: "call-with-a-user", CustomerID: "acme", CallID: "stream-1",
		AgentID: "agent-1", UserID: "ada", StartedAt: s.base,
	}))

	found, err := s.store.Call(s.ctx, "acme", "call-with-a-user")

	s.Require().NoError(err)
	s.Equal("ada", found.UserID)
}

func (s *StoreSuite) TestARequestItsCallerCancelledIsLoggedAsNoError() {
	s.Require().NoError(s.store.StartCall(s.ctx, &Call{ID: "call", CustomerID: "cancel-log", AgentID: "agent", ConfigID: "cfg", StartedAt: s.base}))
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{Modality: "llm", CustomerID: "cancel-log", AgentID: "agent", Provider: "openai", Model: "gpt",
		StartedAt: s.base.Add(time.Second), OutputTokens: 4, Success: false, ErrorCode: ErrorCancelled}))
	logs, err := s.store.AgentLogs(s.ctx, LogFilter{CustomerID: "cancel-log", Limit: 10})
	s.Require().NoError(err)
	s.Require().Len(logs, 1)

	s.Equal("llm request cancelled", logs[0].Message)
	s.Equal("info", logs[0].Severity)
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

func (s *StoreSuite) TestANumberIsRecordedOnlyInE164Form() {
	for _, e164 := range []string{"", "15125551234", "5125551234", "+1 512 555 1234", "+05125551234"} {
		err := s.store.RecordNumber(s.ctx, &PhoneNumber{
			E164: e164, Vendor: "plivo", Country: "US", CustomerID: "acme",
		})
		s.ErrorIs(err, ErrNotE164, "%q", e164)
	}

	held, err := s.store.CustomerNumbers(s.ctx, "acme", true)
	s.Require().NoError(err)
	s.Empty(held)
}

func (s *StoreSuite) TestANumberWrittenInAnotherFormIsNotHeld() {
	// Written straight to the table, the way a row could be before every write checked
	// the form: the vendor reads it as +15125551234.
	_, err := s.store.db.ExecContext(s.ctx,
		"INSERT INTO phone_numbers (e164, vendor, country, customer_id) VALUES ('15125551234', 'plivo', 'US', 'globex')")
	s.Require().NoError(err)

	_, err = s.store.Number(s.ctx, "globex", "15125551234")

	s.ErrorIs(err, ErrNoNumber)
	s.ErrorContains(err, "is not a number globex holds")
}

func (s *StoreSuite) TestANumberRemembersWhichTrunkItsCallsArriveOn() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		CustomerID: "acme", PurchasedAt: s.base,
	}))

	s.Require().NoError(s.store.AttachNumber(
		s.ctx, "acme", "+15125551234", NumberAttachment{TrunkID: "trunk-7", CallType: "default", CallID: "phone-+15125551234"}))

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
		s.ctx, "acme", "+15125551234", NumberAttachment{TrunkID: "trunk-7", CallType: "support", CallID: "the-support-line"}))

	number, err := s.store.NumberByCallInApp(s.ctx, AppScope{Unpinned: true}, "support", "the-support-line")

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

	number, err := s.store.NumberByCallInApp(s.ctx, AppScope{Unpinned: true}, "default", "phone-+15125551234")

	s.Require().NoError(err)
	s.Equal("acme", number.CustomerID)
}

func (s *StoreSuite) TestACallNoNumberReachesIsNotAttributedToAnybody() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		CustomerID: "acme", PurchasedAt: s.base,
	}))

	_, err := s.store.NumberByCallInApp(s.ctx, AppScope{Unpinned: true}, "default", "some-video-call")

	s.ErrorContains(err, "no number reaches call default:some-video-call")
}

func (s *StoreSuite) TestAReleasedNumbersCallIsNotAttributedToItsFormerHolder() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "twilio", Country: "US",
		CustomerID: "acme", PurchasedAt: s.base,
	}))
	s.Require().NoError(s.store.AttachNumber(
		s.ctx, "acme", "+15125551234", NumberAttachment{TrunkID: "trunk-7", CallType: "default", CallID: "phone-+15125551234"}))
	s.Require().NoError(s.store.ReleaseNumber(s.ctx, "acme", "+15125551234", s.base.Add(time.Hour)))

	_, err := s.store.NumberByCallInApp(s.ctx, AppScope{Unpinned: true}, "default", "phone-+15125551234")

	s.ErrorContains(err, "no number reaches call")
}

// attachWithBridge holds a number of the vendor for acme and attaches it with trunk
// credentials.
func (s *StoreSuite) attachWithBridge(vendor, e164 string) {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: e164, Vendor: vendor, Country: "US", CustomerID: "acme", PurchasedAt: s.base,
	}))
	s.Require().NoError(s.store.AttachNumber(s.ctx, "acme", e164, NumberAttachment{
		TrunkID: "trunk-7", CallType: "agent", CallID: "{{uuid}}",
		BridgeURI: "sip:" + e164 + "@sip.example.test", BridgeUsername: "trunk-user",
		BridgePasswordSealed: []byte("sealed"), BridgePasswordKEKVersion: 3,
	}))
}

func (s *StoreSuite) TestAnAttachedNumberKeepsItsTrunkCredentials() {
	s.attachWithBridge("telnyx", "+15125551234")

	bridge, err := s.store.AttachedBridge(s.ctx, "telnyx", "+15125551234")

	s.Require().NoError(err)
	s.Equal(AttachedBridge{
		CustomerID: "acme", URI: "sip:+15125551234@sip.example.test", Username: "trunk-user",
		PasswordSealed: []byte("sealed"), KEKVersion: 3,
	}, bridge)
}

func (s *StoreSuite) TestAReleasedNumbersTrunkCredentialsAreGone() {
	s.attachWithBridge("telnyx", "+15125551234")
	s.Require().NoError(s.store.ReleaseNumber(s.ctx, "acme", "+15125551234", s.base.Add(time.Hour)))

	_, err := s.store.AttachedBridge(s.ctx, "telnyx", "+15125551234")
	s.ErrorIs(err, ErrNoNumber)

	var held PhoneNumber
	s.Require().NoError(s.store.db.NewSelect().Model(&held).Where("e164 = ?", "+15125551234").Scan(s.ctx))
	s.Empty(held.BridgeURI)
	s.Empty(held.BridgeUsername)
	s.Empty(held.BridgePasswordSealed, "a released number does not keep a password")
	s.Zero(held.BridgePasswordKEKVersion)
}

func (s *StoreSuite) TestAnotherVendorsNumberHasNoTrunkCredentials() {
	s.attachWithBridge("twilio", "+15125551234")

	_, err := s.store.AttachedBridge(s.ctx, "telnyx", "+15125551234")

	s.ErrorIs(err, ErrNoNumber)
}

func (s *StoreSuite) TestANumberAttachedWithoutCredentialsHasNoTrunkCredentials() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "telnyx", Country: "US", CustomerID: "acme", PurchasedAt: s.base,
	}))
	s.Require().NoError(s.store.AttachNumber(
		s.ctx, "acme", "+15125551234", NumberAttachment{TrunkID: "trunk-7", CallType: "agent", CallID: "{{uuid}}"}))

	_, err := s.store.AttachedBridge(s.ctx, "telnyx", "+15125551234")

	s.ErrorIs(err, ErrNoStoredBridge, "attaching it again keeps them")
	s.NotErrorIs(err, ErrNoNumber, "it is held and attached, so it is not an unknown number")
}

func (s *StoreSuite) TestANumberThatIsHeldButNotAttachedHasNoTrunkCredentials() {
	s.Require().NoError(s.store.RecordNumber(s.ctx, &PhoneNumber{
		E164: "+15125551234", Vendor: "telnyx", Country: "US", CustomerID: "acme", PurchasedAt: s.base,
	}))

	_, err := s.store.AttachedBridge(s.ctx, "telnyx", "+15125551234")

	s.ErrorIs(err, ErrNoNumber)
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
func (s *StoreSuite) turn(at time.Time, roundtripMs float64) TurnSample {
	sttMs, ttftMs, ttfbMs := 30.0, 120.0, 90.0
	return TurnSample{
		CustomerID:   "acme",
		AgentID:      "agent-1",
		StartedAt:    at,
		AudioOutMs:   2000,
		STTLatencyMs: &sttMs,
		LLMTTFTMs:    &ttftMs,
		TTSTTFBMs:    &ttfbMs,
		RoundtripMs:  &roundtripMs,
	}
}

func (s *StoreSuite) TestStatsIncludeNewRequestsWithoutRollups() {
	s.record("acme", s.base.Add(time.Minute), 1000, 100, true)
	s.record("other", s.base.Add(time.Minute), 9000, 100, true)
	buckets, err := s.store.CustomerStats(s.ctx, "stt", "acme", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(1, buckets[0].RequestCount)
	s.EqualValues(1000, buckets[0].AudioMsTotal)
	s.record("acme", s.base.Add(2*time.Minute), 2000, 100, true)
	buckets, err = s.store.CustomerStats(s.ctx, "stt", "acme", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(buckets, 1)
	s.EqualValues(2, buckets[0].RequestCount)
	s.EqualValues(3000, buckets[0].AudioMsTotal)
}

func (s *StoreSuite) TestAgentLogsPageFilterAndResume() {
	for i := 0; i < 520; i++ {
		entry := AgentLog{CustomerID: "log-test", ConfigID: "agent-a", SessionID: "session-a", UserID: "user-a", Source: "user", Severity: "info", EventType: "test", Message: "same timestamp", OccurredAt: s.base}
		if i%2 == 0 {
			entry.Severity = "error"
		}
		s.Require().NoError(s.store.RecordAgentLog(s.ctx, &entry))
	}
	other := AgentLog{CustomerID: "other", Source: "system", Severity: "error", EventType: "test", Message: "not yours", OccurredAt: s.base}
	s.Require().NoError(s.store.RecordAgentLog(s.ctx, &other))
	high, err := s.store.LogHighWater(s.ctx, "log-test")
	s.Require().NoError(err)
	f := LogFilter{CustomerID: "log-test", Limit: 250, Before: high + 1}
	first, err := s.store.AgentLogs(s.ctx, f)
	s.Require().NoError(err)
	s.Require().Len(first, 250)
	f.Before = first[len(first)-1].ID
	second, err := s.store.AgentLogs(s.ctx, f)
	s.Require().NoError(err)
	s.Require().Len(second, 250)
	s.Less(second[0].ID, first[len(first)-1].ID)
	f.Before = second[len(second)-1].ID
	third, err := s.store.AgentLogs(s.ctx, f)
	s.Require().NoError(err)
	s.Len(third, 20)
	fresh := AgentLog{CustomerID: "log-test", Source: "tool", Severity: "error", EventType: "tool_failed", Message: "Fresh", OccurredAt: s.base}
	s.Require().NoError(s.store.RecordAgentLog(s.ctx, &fresh))
	rows, err := s.store.AgentLogs(s.ctx, LogFilter{CustomerID: "log-test", After: high, Forward: true, Limit: 250})
	s.Require().NoError(err)
	s.Require().Len(rows, 1)
	s.Equal(fresh.ID, rows[0].ID)
	_, err = s.store.AgentLog(s.ctx, "other", fresh.ID)
	s.Error(err)
	rows, err = s.store.AgentLogs(s.ctx, LogFilter{CustomerID: "log-test", Severity: "error", Sources: []string{"user"}, UserID: "user-a", ConfigID: "agent-a", SessionID: "session-a", Limit: 250})
	s.Require().NoError(err)
	s.Len(rows, 250)
	for _, row := range rows {
		s.Equal("error", row.Severity)
		s.Equal("user", row.Source)
		s.Nil(row.Details)
	}
}

func (s *StoreSuite) TestAgentLogRedactsBeforeStorage() {
	entry := AgentLog{CustomerID: "redaction-test", Source: "agent", Severity: "info", EventType: "test", Message: "Bearer private-token api_key=secret-value", Details: map[string]any{"authorization": "secret", "tool": "lookup", "arguments": map[string]string{"password": "private"}}}
	s.Require().NoError(s.store.RecordAgentLog(s.ctx, &entry))
	row, err := s.store.AgentLog(s.ctx, "redaction-test", entry.ID)
	s.Require().NoError(err)
	s.NotContains(row.Message, "private-token")
	s.NotContains(row.Message, "secret-value")
	s.Equal(map[string]any{"tool": "lookup"}, row.Details)
}

func (s *StoreSuite) TestAgentLogProviderFailureDetails() {
	request := &Request{Modality: "search", CustomerID: "failure-details", AgentID: "test-agent", Provider: "perplexity", Model: "search", StartedAt: time.Now(), Success: false, ErrorCode: "build_failed", ErrorMessage: "perplexity: PERPLEXITY_API_KEY is required; api_key=private-key-value"}
	s.Require().NoError(s.store.RecordRequest(s.ctx, request))
	rows, err := s.store.AgentLogs(s.ctx, LogFilter{CustomerID: request.CustomerID, Limit: 1})
	s.Require().NoError(err)
	s.Require().Len(rows, 1)
	detail, err := s.store.AgentLog(s.ctx, request.CustomerID, rows[0].ID)
	s.Require().NoError(err)
	s.Equal("build_failed", detail.Details["error_code"])
	s.Contains(detail.Details["error_message"], "PERPLEXITY_API_KEY is required")
	s.NotContains(detail.Details["error_message"], "private-key-value")
	_, err = s.store.AgentLog(s.ctx, "another-customer", rows[0].ID)
	s.Require().Error(err)
}

func (s *StoreSuite) TestAMovedLogRowNeverShowsTheErrorOfTheRequestHoldingItsNumberHere() {
	// The destination's own failed request, which a moved log row's number happens to name.
	own := &Request{Modality: "llm", CustomerID: "moved-logs", AgentID: "agent", Provider: "openai", Model: "gpt",
		StartedAt: s.base, Success: false, ErrorCode: "upstream_error", ErrorMessage: "the destination's own failure"}
	s.Require().NoError(s.store.RecordRequest(s.ctx, own))
	// A row written as an import writes it, details as they came, naming own's number.
	logged := func(at time.Time, details map[string]any) *AgentLog {
		details["request_id"] = strconv.FormatInt(own.ID, 10)
		entry := &AgentLog{CustomerID: "moved-logs", AgentID: "agent", Source: "system", Severity: "info",
			EventType: "provider_request", Message: "llm request completed", OccurredAt: at, IngestedAt: at, Details: details}
		_, err := s.store.DB().NewInsert().Model(entry).Exec(s.ctx)
		s.Require().NoError(err)
		read, err := s.store.AgentLog(s.ctx, "moved-logs", entry.ID)
		s.Require().NoError(err)
		return read
	}

	carried := logged(s.base.Add(time.Hour), map[string]any{"error_code": nil, "error_message": nil})
	older := logged(s.base.Add(time.Hour), map[string]any{})
	ownOlder := logged(s.base, map[string]any{})

	s.Contains(carried.Details, "error_code")
	s.Nil(carried.Details["error_code"], "a row carrying its error is its own answer")
	s.NotContains(older.Details, "error_code", "an older row's number names a request that started at another time")
	if !s.store.Sharded() {
		s.Equal("upstream_error", ownOlder.Details["error_code"], "an older row of this deployment still finds its request")
	}
}

// recordCancelled stores a request its caller gave up on after it had generated something.
func (s *StoreSuite) recordCancelled(at time.Time, latencyMs *float64) {
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "stt", CustomerID: "acme", Provider: "deepgram", Model: "flux-general-en",
		Tags: map[string]string{"project": "support"}, StartedAt: at,
		AudioMs: 700, CostMicros: 50, LatencyMs: latencyMs,
		Success: false, ErrorCode: ErrorCancelled,
	}))
}

func (s *StoreSuite) TestACancelledRequestIsSpendButNeitherAnErrorNorALatency() {
	for index, latency := range []float64{100, 200} {
		s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
			Modality: "stt", CustomerID: "acme", Provider: "deepgram", Model: "flux-general-en",
			Tags: map[string]string{"project": "support"}, StartedAt: s.base.Add(time.Duration(index+1) * time.Minute),
			AudioMs: 1000, CostMicros: 100, LatencyMs: &latency, Success: true,
		}))
	}
	failedLatency := 900.0
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{
		Modality: "stt", CustomerID: "acme", Provider: "deepgram", Model: "flux-general-en",
		Tags: map[string]string{"project": "support"}, StartedAt: s.base.Add(3 * time.Minute),
		AudioMs: 500, CostMicros: 60, LatencyMs: &failedLatency, Success: false, ErrorCode: "upstream_error",
	}))
	// Closed after a first token, and so with a latency, and closed before one, with none.
	cancelledLatency := 5.0
	s.recordCancelled(s.base.Add(4*time.Minute), &cancelledLatency)
	s.recordCancelled(s.base.Add(5*time.Minute), nil)

	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	check := func(name string, requests, errors, audio, cost int64, p50 *float64, uptime *float64) {
		s.EqualValues(5, requests, name+": a cancelled request is still a request")
		s.EqualValues(1, errors, name+": and is no error")
		s.EqualValues(3900, audio, name+": what it generated is still counted")
		s.EqualValues(360, cost, name+": and still billed")
		s.Require().NotNil(p50, name)
		s.InDelta(200.0, *p50, 0.001, name+": its latency is not one the provider had")
		s.Require().NotNil(uptime, name)
		s.InDelta(4.0/5.0, *uptime, 0.001, name+": nor is it downtime")
	}

	byProvider, err := s.store.CustomerStats(s.ctx, "stt", "acme", Hourly, s.base, s.base.Add(time.Hour), nil)
	s.Require().NoError(err)
	s.Require().Len(byProvider, 1)
	bucket := byProvider[0]
	check("rollup", bucket.RequestCount, bucket.ErrorCount, bucket.AudioMsTotal, bucket.CostMicrosTotal,
		bucket.LatencyP50Ms, bucket.Uptime)

	byTag, err := s.store.CustomerTagStats(s.ctx, "stt", "acme", "project", Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)
	s.Require().Len(byTag, 1)
	tag := byTag[0]
	s.EqualValues(5, tag.RequestCount, "tag rollup: a cancelled request is still a request")
	s.EqualValues(1, tag.ErrorCount, "tag rollup: and is no error")
	s.EqualValues(360, tag.CostMicrosTotal, "tag rollup: and still billed")
	s.Require().NotNil(tag.LatencyP50Ms)
	s.InDelta(200.0, *tag.LatencyP50Ms, 0.001, "tag rollup: its latency is not one the provider had")

	filtered, err := s.store.CustomerStats(s.ctx, "stt", "acme", Hourly, s.base, s.base.Add(time.Hour),
		map[string]string{"project": "support"})
	s.Require().NoError(err)
	s.Require().Len(filtered, 1)
	bucket = filtered[0]
	check("raw rows", bucket.RequestCount, bucket.ErrorCount, bucket.AudioMsTotal, bucket.CostMicrosTotal,
		bucket.LatencyP50Ms, bucket.Uptime)
}

// usageRow is a request with a usage id of its own, defaulting what a test does not care about.
func (s *StoreSuite) usageRow(customerID string, change func(*Request)) Request {
	request := Request{
		UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: customerID,
		Provider: "openai", Model: "gpt", StartedAt: s.base, OutputTokens: 10, CostMicros: 25, Success: true,
	}
	if change != nil {
		change(&request)
	}
	return request
}

// tooBigToIndex is an id longer than a btree index row may be, even compressed.
func tooBigToIndex() string {
	raw := make([]byte, 10_000)
	_, _ = rand.Read(raw)
	return hex.EncodeToString(raw)
}

// requestsOf is every row a customer has in this shard, by usage id.
func (s *StoreSuite) requestsOf(customerID string) map[uuid.UUID]Request {
	var rows []Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).Where("customer_id = ?", customerID).Scan(s.ctx))
	byUsage := map[uuid.UUID]Request{}
	for _, row := range rows {
		byUsage[row.UsageID] = row
	}
	return byUsage
}

func (s *StoreSuite) TestRecordRequestsSkipsWrittenUsageIDs() {
	first, second := s.usageRow("batch", nil), s.usageRow("batch", nil)
	written, err := s.store.RecordRequests(s.ctx, []Request{first})
	s.Require().NoError(err)
	s.Equal([]uuid.UUID{first.UsageID}, written)

	written, err = s.store.RecordRequests(s.ctx, []Request{first, second})

	s.Require().NoError(err)
	s.Equal([]uuid.UUID{second.UsageID}, written, "a row already written is not written again")
	s.Len(s.requestsOf("batch"), 2)
}

func (s *StoreSuite) TestTwoRowsWithOneUsageIDInOneBatchWriteOne() {
	first := s.usageRow("twice", nil)
	second := s.usageRow("twice", func(request *Request) { request.UsageID = first.UsageID })

	written, err := s.store.RecordRequests(s.ctx, []Request{first, second})

	s.Require().NoError(err)
	s.Equal([]uuid.UUID{first.UsageID}, written)
	s.Len(s.requestsOf("twice"), 1)
}

func (s *StoreSuite) TestARowWithoutAUsageIDIsRefused() {
	written, err := s.store.RecordRequests(s.ctx, []Request{
		s.usageRow("refused", nil),
		s.usageRow("refused", func(request *Request) { request.UsageID = uuid.Nil }),
	})

	s.ErrorIs(err, ErrInvalidRequest)
	s.True(Unwritable(err), "a row with no usage id never becomes writable")
	s.Empty(written)
	s.Empty(s.requestsOf("refused"), "the batch is one transaction")
}

func (s *StoreSuite) TestTheTableRefusesARowWhoseUsageIDIsNull() {
	_, err := s.store.ShardDB().ExecContext(s.ctx, `INSERT INTO requests (usage_id, modality, customer_id, provider, model, started_at, success)
VALUES (NULL, 'llm', 'bare', 'openai', 'gpt', now(), true)`)

	s.Require().Error(err, "a writer that skips RecordRequests still cannot write a row with no usage id")
	s.True(Unwritable(err))
}

func (s *StoreSuite) TestARowWrittenWithoutAUsageIDGetsOneFromTheTable() {
	request := s.usageRow("bare", func(request *Request) { request.UsageID = uuid.Nil })

	_, err := s.store.ShardDB().NewInsert().Model(&request).Exec(s.ctx)

	s.Require().NoError(err, "as the router from before usage ids writes")
	s.Require().Len(s.requestsOf("bare"), 1)
	for usageID := range s.requestsOf("bare") {
		s.NotEqual(uuid.Nil, usageID)
	}
}

func (s *StoreSuite) TestARowPostgresRefusesIsUnwritable() {
	_, err := s.store.RecordRequests(s.ctx, []Request{
		s.usageRow("too-big", func(request *Request) { request.AgentID = tooBigToIndex() }),
	})

	s.Require().Error(err)
	s.True(Unwritable(err), "an id too big for its index is refused however often it is sent: %v", err)
}

func (s *StoreSuite) TestAShardedBatchWritesOneLogRowPerNewRow() {
	s.Require().NoError(s.store.StartCall(s.ctx, &Call{ID: "call", CustomerID: "batch-log", AgentID: "agent", ConfigID: "cfg", StartedAt: s.base}))
	batch := []Request{
		s.usageRow("batch-log", func(request *Request) { request.AgentID = "agent" }),
		s.usageRow("batch-log", func(request *Request) { request.AgentID = "agent" }),
		s.usageRow("batch-log", nil),
	}

	_, err := s.store.RecordRequests(s.ctx, batch)
	s.Require().NoError(err)
	_, err = s.store.RecordRequests(s.ctx, batch)
	s.Require().NoError(err)

	logs, err := s.store.AgentLogs(s.ctx, LogFilter{CustomerID: "batch-log", Limit: 10})
	s.Require().NoError(err)
	s.Len(logs, 2, "one log row per agent row, and none for a row written again")
	for _, log := range logs {
		s.Equal("call", log.SessionID)
	}
}

func (s *StoreSuite) TestABatchLogsEachRowInItsOwnCall() {
	for _, call := range []*Call{
		{ID: "first", CallID: "stream-1", CustomerID: "calls-a", AgentID: "agent", ConfigID: "cfg-1", StartedAt: s.base},
		{ID: "second", CallID: "stream-2", CustomerID: "calls-a", AgentID: "agent", ConfigID: "cfg-2", StartedAt: s.base.Add(time.Minute)},
		{ID: "other", CallID: "stream-3", CustomerID: "calls-b", AgentID: "agent", ConfigID: "cfg-3", StartedAt: s.base},
	} {
		s.Require().NoError(s.store.StartCall(s.ctx, call))
	}
	at := func(customer, callID string, after time.Duration) Request {
		return s.usageRow(customer, func(request *Request) {
			request.AgentID, request.CallID, request.StartedAt = "agent", callID, s.base.Add(after)
		})
	}
	batch := []Request{
		at("calls-a", "stream-1", 2*time.Minute),
		at("calls-b", "", time.Second),
		at("calls-a", "", 2*time.Minute),
		at("calls-a", "", 30*time.Second),
	}

	_, err := s.store.RecordRequests(s.ctx, batch)
	s.Require().NoError(err)

	sessions := func(customer string) map[string]string {
		logs, err := s.store.AgentLogs(s.ctx, LogFilter{CustomerID: customer, Limit: 10})
		s.Require().NoError(err)
		byRequest := map[string]string{}
		for _, log := range logs {
			logged, err := s.store.AgentLog(s.ctx, customer, log.ID)
			s.Require().NoError(err)
			byRequest[logged.Details["request_id"].(string)] = logged.SessionID + "/" + logged.ConfigID
		}
		return byRequest
	}
	id := func(request Request) string { return strconv.FormatInt(request.ID, 10) }
	s.Equal(map[string]string{
		id(batch[0]): "first/cfg-1",
		id(batch[2]): "second/cfg-2",
		id(batch[3]): "first/cfg-1",
	}, sessions("calls-a"), "the call a row names, else the agent's latest call started by it")
	s.Equal(map[string]string{id(batch[1]): "other/cfg-3"}, sessions("calls-b"))
}

// onShard runs a statement on this shard's database, failing the test if it fails.
func (s *StoreSuite) onShard(query string, args ...any) {
	_, err := s.store.ShardDB().ExecContext(s.ctx, query, args...)
	s.Require().NoError(err)
}

// usageIndex reports whether requests_usage_id_idx is in this shard's database, and if it
// is, whether it is valid.
func (s *StoreSuite) usageIndex() (valid, found bool) {
	var state sql.NullBool
	s.Require().NoError(s.store.ShardDB().QueryRowContext(s.ctx,
		"SELECT (SELECT indisvalid FROM pg_index WHERE indexrelid = to_regclass('requests_usage_id_idx'))").Scan(&state))
	return state.Bool, state.Valid
}

// restoreUsageIndex puts requests_usage_id_idx back as the migration leaves it once the test
// ends, whatever the test got to.
func (s *StoreSuite) restoreUsageIndex() {
	s.T().Cleanup(func() {
		s.onShard("TRUNCATE requests")
		s.onShard("DROP INDEX CONCURRENTLY IF EXISTS requests_usage_id_idx")
		s.onShard("CREATE UNIQUE INDEX CONCURRENTLY requests_usage_id_idx ON requests (usage_id)")
	})
}

func (s *StoreSuite) TestMigrateBuildsAUsageIndexSomebodyDropped() {
	s.restoreUsageIndex()
	s.onShard("DROP INDEX CONCURRENTLY requests_usage_id_idx")
	row := s.usageRow("dropped-index", nil)
	_, err := s.store.RecordRequests(s.ctx, []Request{row})
	s.Require().ErrorContains(err, "ON CONFLICT", "nothing to infer the conflict from without the index")

	s.Require().NoError(s.store.Migrate(s.ctx), "the migration is applied already, and Migrate builds the index itself")

	valid, found := s.usageIndex()
	s.True(found)
	s.True(valid)
	written, err := s.store.RecordRequests(s.ctx, []Request{row, row})
	s.Require().NoError(err)
	s.Equal([]uuid.UUID{row.UsageID}, written)
}

func (s *StoreSuite) TestMigrateRebuildsAUsageIndexThatDidNotBuild() {
	s.restoreUsageIndex()
	s.onShard("DROP INDEX CONCURRENTLY requests_usage_id_idx")
	twice := s.usageRow("invalid-index", nil)
	_, err := s.store.ShardDB().NewInsert().Model(&[]Request{twice, twice}).Exec(s.ctx)
	s.Require().NoError(err)
	_, err = s.store.ShardDB().ExecContext(s.ctx, "CREATE UNIQUE INDEX CONCURRENTLY requests_usage_id_idx ON requests (usage_id)")
	s.Require().Error(err, "two rows with one usage id fail the build")
	valid, found := s.usageIndex()
	s.Require().True(found, "a failed concurrent build leaves its index behind")
	s.Require().False(valid)

	err = s.store.Migrate(s.ctx)
	s.Require().Error(err, "the router does not start while the index cannot be built")
	s.Contains(err.Error(), "requests_usage_id_idx")

	s.onShard("DELETE FROM requests WHERE id = (SELECT max(id) FROM requests WHERE usage_id = ?)", twice.UsageID)
	s.Require().NoError(s.store.Migrate(s.ctx))

	valid, found = s.usageIndex()
	s.True(found)
	s.True(valid, "the invalid index is dropped and built again")
}

func (s *StoreSuite) TestMigrateRebuildsAGrantsIndexLeftInvalid() {
	// As a CONCURRENTLY build that failed leaves it.
	s.onShard("UPDATE pg_index SET indisvalid = false WHERE indexrelid = to_regclass('requests_limit_grants_started_idx')")
	var valid bool
	read := func() {
		s.Require().NoError(s.store.ShardDB().QueryRowContext(s.ctx,
			"SELECT indisvalid FROM pg_index WHERE indexrelid = to_regclass('requests_limit_grants_started_idx')").Scan(&valid))
	}
	read()
	s.Require().False(valid)

	s.Require().NoError(s.store.Migrate(s.ctx))

	read()
	s.True(valid, "dropped and built again")
}

// The reconcile reads every shard every few minutes, so it reads only the rows naming a grant,
// by the partial index, rather than every request.
func (s *StoreSuite) TestTheGrantReconcileReadsByTheGrantsIndex() {
	var plan []string
	err := s.store.ShardDB().RunInTx(s.ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		// A table this small costs any index read and a sort about the same, so which one the
		// planner takes moved with whatever rows other tests had left. The grants index is the
		// one that reads in the order the reconcile pages by, so with no sort to fall back on
		// the plan says whether the query can be served by it at all.
		if _, err := tx.ExecContext(ctx, "SET LOCAL enable_seqscan = off; SET LOCAL enable_sort = off; SET LOCAL enable_incremental_sort = off"); err != nil {
			return err
		}
		return tx.NewRaw("EXPLAIN "+reconcileGrantsQuery, time.Now(), 0, grantReconcilePage).Scan(ctx, &plan)
	})

	s.Require().NoError(err)
	s.Contains(strings.Join(plan, "\n"), "requests_limit_grants_started_idx")
}

func (s *StoreSuite) TestARouterWaitingToMigrateDoesNotHoldUpTheOneMigrating() {
	// The router migrating holds the lock on a connection of its own, as Migrate does.
	holder, err := s.store.DB().DB.Conn(s.ctx)
	s.Require().NoError(err)
	defer holder.Close()
	_, err = holder.ExecContext(s.ctx, "SELECT pg_advisory_lock(hashtextextended('router.migrate', 0))")
	s.Require().NoError(err)
	unlock := sync.OnceFunc(func() {
		_, err := holder.ExecContext(s.ctx, "SELECT pg_advisory_unlock(hashtextextended('router.migrate', 0))")
		s.Require().NoError(err)
	})
	defer unlock()

	waiter, err := Open(s.dsn)
	s.Require().NoError(err)
	defer waiter.Close()
	migrated := make(chan error, 1)
	go func() { migrated <- waiter.Migrate(s.ctx) }()
	select {
	case err := <-migrated:
		s.FailNow("migrated while another router held the lock", "%v", err)
	case <-time.After(300 * time.Millisecond):
	}

	// A concurrent build waits for every snapshot older than its own to end. goose runs the
	// usage migration's on another connection of the pool than the one holding the lock, so
	// Postgres sees no deadlock to break.
	s.T().Cleanup(func() {
		_, err := s.store.DB().ExecContext(s.ctx, "DROP INDEX CONCURRENTLY IF EXISTS requests_lock_wait_idx")
		s.Require().NoError(err)
	})
	ctx, cancel := context.WithTimeout(s.ctx, 10*time.Second)
	defer cancel()
	_, err = s.store.DB().ExecContext(ctx, "CREATE INDEX CONCURRENTLY requests_lock_wait_idx ON requests (customer_id)")
	s.Require().NoError(err, "a router waiting for the lock holds no snapshot the build waits on")
	unlock()

	select {
	case err := <-migrated:
		s.NoError(err)
	case <-time.After(30 * time.Second):
		s.FailNow("the waiting router never migrated once the lock was free")
	}
}

func (s *StoreSuite) TestBatchesNamingTheSameCustomersInAnotherOrderDoNotDeadlock() {
	// The agent log trigger locks each customer it logs a row for until the batch commits.
	batch := func(first, second string) []Request {
		var rows []Request
		for range 50 {
			rows = append(rows, s.usageRow(first, func(request *Request) { request.AgentID = "agent" }))
		}
		return append(rows, s.usageRow(second, func(request *Request) { request.AgentID = "agent" }))
	}

	for range 5 {
		one, other := batch("order-a", "order-b"), batch("order-b", "order-a")
		failed := make(chan error, 2)
		go func() { _, err := s.store.RecordRequests(s.ctx, one); failed <- err }()
		go func() { _, err := s.store.RecordRequests(s.ctx, other); failed <- err }()
		s.Require().NoError(<-failed)
		s.Require().NoError(<-failed)
	}

	s.Len(s.requestsOf("order-a"), 5*51)
	s.Len(s.requestsOf("order-b"), 5*51)
}

func (s *StoreSuite) TestARequestIDADataMoveTookIsWrittenOnTheNextTry() {
	var next int64
	s.Require().NoError(s.store.ShardDB().QueryRowContext(s.ctx,
		"SELECT nextval(pg_get_serial_sequence('requests', 'id'))").Scan(&next))
	// What a data move does: a row carried in with its own id, ahead of the sequence.
	moved := s.usageRow("moved-id", func(request *Request) { request.ID = next + 1 })
	_, err := s.store.ShardDB().NewInsert().Model(&moved).Exec(s.ctx)
	s.Require().NoError(err)
	row := s.usageRow("moved-id", nil)

	_, err = s.store.RecordRequests(s.ctx, []Request{row})
	s.Require().Error(err, "the sequence's next id is the moved row's")
	s.True(IDTaken(err), "%v", err)
	s.False(Unwritable(err), "the next try draws another id: %v", err)

	written, err := s.store.RecordRequests(s.ctx, []Request{row})
	s.Require().NoError(err)
	s.Equal([]uuid.UUID{row.UsageID}, written)
}

func (s *StoreSuite) TestManyMovedRequestIDsCostOneTry() {
	var next int64
	s.Require().NoError(s.store.ShardDB().QueryRowContext(s.ctx,
		"SELECT nextval(pg_get_serial_sequence('requests', 'id'))").Scan(&next))
	moved := make([]Request, 500)
	for i := range moved {
		moved[i] = s.usageRow("many-moved-ids", func(request *Request) { request.ID = next + 1 + int64(i) })
	}
	_, err := s.store.ShardDB().NewInsert().Model(&moved).Exec(s.ctx)
	s.Require().NoError(err)
	rows := []Request{s.usageRow("many-moved-ids", nil), s.usageRow("many-moved-ids", nil)}

	_, err = s.store.RecordRequests(s.ctx, rows)
	s.Require().True(IDTaken(err), "%v", err)
	written, err := s.store.RecordRequests(s.ctx, rows)

	s.Require().NoError(err, "the failed try moved the sequence past every moved id")
	s.Len(written, 2)
	s.Greater(rows[0].ID, next+500)
}

func (s *StoreSuite) TestTheRequestIDSequenceIsNeverMovedBack() {
	var ahead int64
	s.Require().NoError(s.store.ShardDB().QueryRowContext(s.ctx,
		"SELECT setval(pg_get_serial_sequence('requests', 'id'), nextval(pg_get_serial_sequence('requests', 'id')) + 1000)").Scan(&ahead))

	s.Require().NoError(s.store.advanceRequestIDs(s.ctx))

	rows := []Request{s.usageRow("sequence-ahead", nil)}
	_, err := s.store.RecordRequests(s.ctx, rows)
	s.Require().NoError(err)
	s.Equal(ahead+1, rows[0].ID, "a sequence ahead of every row is left where it is")
}

func (s *StoreSuite) TestABatchLeavesItsRowsInTheOrderSentWithTheirIDs() {
	sent := []Request{s.usageRow("order-z", nil), s.usageRow("order-y", nil), s.usageRow("order-z", nil)}

	_, err := s.store.RecordRequests(s.ctx, sent)

	s.Require().NoError(err)
	s.Equal([]string{"order-z", "order-y", "order-z"}, []string{sent[0].CustomerID, sent[1].CustomerID, sent[2].CustomerID})
	stored := s.requestsOf("order-z")
	s.Equal(sent[0].ID, stored[sent[0].UsageID].ID, "each row sent has the id it was written under")
	s.Equal(sent[2].ID, stored[sent[2].UsageID].ID)
	s.Less(sent[0].ID, sent[2].ID, "one customer's rows are written in the order they were sent")
}

func (s *StoreSuite) TestTheIdentityAndShippingColumnsAreStored() {
	shipped := s.base.Add(time.Minute)
	request := s.usageRow("identity", func(request *Request) {
		request.OrganizationID, request.OrganizationTier, request.APIKey = "org", "tier-a", "key"
		request.UserID, request.UsageCategories, request.SessionID = "ada", []string{"support", "team:core"}, "session"
		request.ToShip, request.ShippedAt, request.ShipError = true, &shipped, "refused"
	})
	_, err := s.store.RecordRequests(s.ctx, []Request{request})
	s.Require().NoError(err)

	stored := s.requestsOf("identity")[request.UsageID]

	s.Equal("org", stored.OrganizationID)
	s.Equal("tier-a", stored.OrganizationTier)
	s.Equal("key", stored.APIKey)
	s.Equal("ada", stored.UserID)
	s.Equal([]string{"support", "team:core"}, stored.UsageCategories)
	s.Equal("session", stored.SessionID)
	s.True(stored.ToShip)
	s.Require().NotNil(stored.ShippedAt)
	s.True(shipped.Equal(*stored.ShippedAt))
	s.Equal("refused", stored.ShipError)
}
