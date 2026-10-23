//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	usageservice "github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// UsageCommandsSuite runs `router usage` on a database of its own that it empties first.
type UsageCommandsSuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
	store    *store.Store
}

func TestUsageCommandsSuite(t *testing.T) {
	suite.Run(t, new(UsageCommandsSuite))
}

func (s *UsageCommandsSuite) SetupTest() {
	s.ctx = context.Background()
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	own := testenv.Database(dsn, "usagecli")
	empty, err := store.Open(own)
	s.Require().NoError(err)
	var database string
	s.Require().NoError(empty.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = empty.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	s.Require().NoError(empty.Migrate(s.ctx))
	s.store = empty
	s.T().Cleanup(func() { empty.Close() })

	s.settings = config.Defaults()
	s.settings.Postgres.DSN = own
	// Nothing is sent from here: a usage service only makes the rows wait to be shipped.
	s.settings.Usage = config.Usage{URL: "http://usage.invalid:8070", Token: "token"}
}

// lostLog is a router's log in which a writer that has closed lost one row, among other lines.
func (s *UsageCommandsSuite) lostLog() (path string, lost uuid.UUID) {
	var log bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&log, nil))
	logger.Info("listening", "address", ":8080")
	writer := routing.NewWriter(routing.WriterOptions{Store: s.store, Logger: logger})
	ctx, cancel := context.WithTimeout(s.ctx, time.Second)
	defer cancel()
	writer.Close(ctx)
	lost = uuid.Must(uuid.NewV7())
	routing.NewRecorderOn(routing.LLM, writer, nil, logger).Record(routing.ProviderConfig{Provider: "openai", Model: "gpt"}, routing.Stat{
		Owner:   routing.Owner{CustomerID: "app-1", Attribution: routing.Attribution{SessionID: "session-1"}, Tags: routing.Tags{"project": `a "quoted" one`}},
		UsageID: lost, StartedAt: time.Now().UTC(), Usage: routing.Usage{OutputTokens: 10}, ErrorMessage: "cut off\nmid line",
	})
	s.Require().Contains(log.String(), `msg="usage row lost" reason=shutdown`)
	logger.Info("shutting down")
	path = filepath.Join(s.T().TempDir(), "router.log")
	s.Require().NoError(os.WriteFile(path, log.Bytes(), 0o600))
	return path, lost
}

func (s *UsageCommandsSuite) requests() []store.Request {
	var rows []store.Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).Order("started_at").Scan(s.ctx))
	return rows
}

func (s *UsageCommandsSuite) TestReplayWritesLostRowsOnce() {
	path, lost := s.lostLog()
	var first, second bytes.Buffer

	s.Require().NoError(runUsage([]string{"replay", "--from", path}, s.settings, &first))
	s.Require().NoError(runUsage([]string{"replay", "--from", path}, s.settings, &second))

	s.Contains(first.String(), "1 lost rows: 1 written, 0 already there, 0 refused")
	s.Contains(second.String(), "1 lost rows: 0 written, 1 already there, 0 refused", "replaying the file again writes nothing")
	rows := s.requests()
	s.Require().Len(rows, 1)
	s.Equal(lost, rows[0].UsageID)
	s.Equal("session-1", rows[0].SessionID)
	s.Equal(map[string]string{"project": `a "quoted" one`}, rows[0].Tags)
	s.True(rows[0].ToShip, "queued for the sweeper, with a usage service")
	s.Equal(routing.Unattributed, rows[0].APIKey, "the writer's defaults, which the row was lost before it got")
	s.Equal(usageservice.DefaultTier, rows[0].OrganizationTier)
}

func (s *UsageCommandsSuite) TestAReplayedRowWithAUserTheUsageServiceWouldRefuseIsWrittenWithoutIt() {
	lost := store.Request{
		UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: "app-1", APIKey: "key-a", UserID: "auth0|x",
		Provider: "openai", Model: "gpt", StartedAt: time.Now().UTC(), Success: true,
	}
	row, err := json.Marshal(lost)
	s.Require().NoError(err)
	// As a router from before the rule would have logged it.
	var log bytes.Buffer
	slog.New(slog.NewTextHandler(&log, nil)).Error("usage row lost", "reason", "shutdown", "usage_id", lost.UsageID, "row", string(row))
	path := filepath.Join(s.T().TempDir(), "router.log")
	s.Require().NoError(os.WriteFile(path, log.Bytes(), 0o600))

	s.Require().NoError(runUsage([]string{"replay", "--from", path}, s.settings, &bytes.Buffer{}))

	rows := s.requests()
	s.Require().Len(rows, 1)
	s.Equal(lost.UsageID, rows[0].UsageID)
	s.Empty(rows[0].UserID)
	s.Equal("key-a", rows[0].APIKey)
}

func (s *UsageCommandsSuite) TestWithNoUsageServiceAReplayedRowIsNotQueued() {
	path, _ := s.lostLog()
	s.settings.Usage = config.Usage{}

	s.Require().NoError(runUsage([]string{"replay", "--from", path}, s.settings, &bytes.Buffer{}))

	s.False(s.requests()[0].ToShip)
}

// recorded writes a row of this shard that started ago.
func (s *UsageCommandsSuite) recorded(ago time.Duration, change func(*store.Request)) uuid.UUID {
	request := store.Request{
		UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: "app-1", Provider: "openai", Model: "gpt",
		StartedAt: time.Now().UTC().Add(-ago), Success: true,
	}
	change(&request)
	_, err := s.store.RecordRequests(s.ctx, []store.Request{request})
	s.Require().NoError(err)
	return request.UsageID
}

// shipState is whether a row waits to ship, and why the usage service refused it.
type shipState struct {
	toShip    bool
	shipError string
}

func (s *UsageCommandsSuite) states() map[uuid.UUID]shipState {
	states := map[uuid.UUID]shipState{}
	for _, row := range s.requests() {
		states[row.UsageID] = shipState{toShip: row.ToShip, shipError: row.ShipError}
	}
	return states
}

// rowsOfEveryKind writes, started ten minutes ago, a row the usage service has, one it refused,
// one written before the router shipped, and one a data move carried in, which arrives as the
// import writes it (see TestAnImportedRequestKeepsItsUsageIDAndIsNeverShipped).
func (s *UsageCommandsSuite) rowsOfEveryKind() (shipped, rejected, beforeShipping, imported uuid.UUID) {
	shippedAt := time.Now().UTC().Add(-5 * time.Minute)
	shipped = s.recorded(10*time.Minute, func(request *store.Request) { request.ShippedAt = &shippedAt })
	rejected = s.recorded(10*time.Minute, func(request *store.Request) { request.ShipError = usageservice.ReasonAPIKey })
	beforeShipping = s.recorded(10*time.Minute, func(*store.Request) {})
	imported = s.recorded(10*time.Minute, func(request *store.Request) { request.CustomerID = "moved-here" })
	return shipped, rejected, beforeShipping, imported
}

func (s *UsageCommandsSuite) TestReshipQueuesTheRowsTheServiceHas() {
	shipped, rejected, beforeShipping, imported := s.rowsOfEveryKind()
	since := time.Now().UTC().Add(-time.Hour).Format(time.RFC3339)
	var out bytes.Buffer

	s.Require().NoError(runUsage([]string{"reship", "--since", since}, s.settings, &out))

	s.Contains(out.String(), "queued 1 rows")
	s.Equal(map[uuid.UUID]shipState{
		shipped:        {toShip: true},
		rejected:       {shipError: usageservice.ReasonAPIKey},
		beforeShipping: {},
		imported:       {},
	}, s.states(), "a row the service never had is not billed now")
}

func (s *UsageCommandsSuite) TestReshipQueuesRejectedRows() {
	shipped, rejected, beforeShipping, imported := s.rowsOfEveryKind()
	older := s.recorded(3*time.Hour, func(request *store.Request) { request.ShipError = usageservice.ReasonAPIKey })
	var out bytes.Buffer

	s.Require().NoError(runUsage([]string{"reship", "--since", "1h", "--rejected"}, s.settings, &out))

	s.Contains(out.String(), "queued 1 rows")
	s.Equal(map[uuid.UUID]shipState{
		shipped:        {},
		rejected:       {toShip: true},
		beforeShipping: {},
		imported:       {},
		older:          {shipError: usageservice.ReasonAPIKey},
	}, s.states(), "only the refused row since then is queued, and why it was refused is cleared")
}

func (s *UsageCommandsSuite) TestReshipWalksPastAFullBatch() {
	_, err := s.store.ShardDB().ExecContext(s.ctx, `
INSERT INTO requests (usage_id, modality, customer_id, provider, model, started_at, shipped_at, success)
SELECT gen_random_uuid(), 'llm', 'app-1', 'openai', 'gpt', now() - interval '10 minutes', now() - interval '5 minutes', true
FROM generate_series(1, 10050)`)
	s.Require().NoError(err)
	var out bytes.Buffer

	s.Require().NoError(runUsage([]string{"reship", "--since", "1h"}, s.settings, &out))

	s.Contains(out.String(), "queued 10050 rows")
}

func (s *UsageCommandsSuite) TestReshipNeedsAUsageService() {
	s.settings.Usage = config.Usage{}

	err := runUsage([]string{"reship", "--since", "1h"}, s.settings, &bytes.Buffer{})

	s.ErrorContains(err, "ROUTER_USAGE_URL")
}
