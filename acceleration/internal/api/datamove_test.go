//go:build integration

package api

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// DataMoveSuite covers the three ways a customer's data goes in or out: the snapshot, the
// changes since it, and writing either of them into a deployment.
type DataMoveSuite struct {
	RouterSuite
}

// TestDataMoveSuite runs on its own rather than beside the other suites: the change feed waits
// for every transaction open on the Postgres server, and other suites hold one for longer than
// a change is waited for (a row locked past auditWriteTimeout).
func TestDataMoveSuite(t *testing.T) {
	suite.Run(t, new(DataMoveSuite))
}

func (s *DataMoveSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *DataMoveSuite) TestAnExportEndsWithTheCursorItsChangesCarryOnFrom() {
	// The cursor is last rather than first because it is also what says the export
	// finished: a stream that died halfway has none.
	lines := s.export()

	last := lines[len(lines)-1]
	s.Require().NotNil(last.Cursor, "the export did not finish")
	s.Equal(s.customerID(), last.Customer)
	s.NotNil(last.At)
}

func (s *DataMoveSuite) TestAnExportCarriesTheAppsOwnRows() {
	agent := s.data.createAgentConfig()

	exported := s.export()

	var configs []string
	for _, line := range exported {
		if line.Table == "agent_configs" {
			configs = append(configs, string(line.Row))
		}
	}
	s.Require().Len(configs, 1, "the app has one agent")
	s.Contains(configs[0], agent.Id)
}

func (s *DataMoveSuite) TestAnotherAppsRowsAreNotInTheExport() {
	s.skipOnARouterWithShards()
	stranger := s.data.backendOfAnotherApp()
	var theirs AgentConfig
	s.Require().Equal(http.StatusCreated, stranger.do(http.MethodPost, "/v1/agents/configs",
		AgentConfigRequest{Name: "agent-" + s.utils.uuid(), Llm: pointerTo("llm-flow")}, &theirs))

	status, exported := s.serverClient.call(http.MethodGet, "/v1/data/export", nil)

	s.Require().Equal(http.StatusOK, status)
	s.NotContains(string(exported), theirs.Id)
}

func (s *DataMoveSuite) TestWhatHappensAfterAnExportIsInTheChangesSinceIt() {
	s.export()
	agent := s.data.createAgentConfig()

	changed := s.changesSince(0)

	s.True(changed.CaughtUp, "fewer changes than were asked for is the end of them")
	var tables []string
	for _, change := range changed.Changes {
		tables = append(tables, change.Table)
	}
	s.Contains(tables, "agent_configs")
	s.Contains(s.rendered(changed.Changes), agent.Id)
}

func (s *DataMoveSuite) TestFollowingTheChangesFromWhereTheLastPageEndedRepeatsNothing() {
	s.export()
	s.data.createAgentConfig()

	first := s.changesSince(0)
	s.data.createAgentConfig()
	second := s.changesSince(first.Cursor)

	s.NotContains(s.rendered(second.Changes), s.rendered(first.Changes))
}

// The numbers are shared by every app, so the ones between the export's cursor and the app's
// first change after it can be another moving app's.
func (s *DataMoveSuite) TestTheFirstChangeAfterAnExportIsAnsweredWhenAnotherAppChangedInBetween() {
	s.skipOnARouterWithShards()
	stranger := s.data.backendOfAnotherApp()
	s.Require().Equal(http.StatusOK, stranger.do(http.MethodGet, "/v1/data/export", nil, nil))
	theirs := func() {
		s.Require().Equal(http.StatusCreated, stranger.do(http.MethodPost, "/v1/agents/configs",
			AgentConfigRequest{Name: "agent-" + s.utils.uuid(), Llm: pointerTo("llm-flow")}, nil))
	}
	theirs()
	exported := s.export()
	cursor := exported[len(exported)-1].Cursor
	s.Require().NotNil(cursor, "the export did not finish")
	theirs()
	agent := s.data.createAgentConfig()

	changed := s.changesSince(*cursor)

	s.Contains(s.rendered(changed.Changes), agent.Id)
}

func (s *DataMoveSuite) TestACursorThatIsNotOneIsRefused() {
	s.skipOnARouterWithShards()
	status, failure := s.serverClient.failure(http.MethodGet, "/v1/data/changes?after=yesterday", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "cursor")
}

func (s *DataMoveSuite) TestMoreChangesThanEitherEndCanHoldAreRefused() {
	s.skipOnARouterWithShards()
	status, failure := s.serverClient.failure(http.MethodGet, "/v1/data/changes?limit=5000", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "between 1 and")
}

func (s *DataMoveSuite) TestImportingAnExportWritesTheRowsItCarried() {
	s.data.createAgentConfig()
	exported := s.export()

	var written struct {
		Rows   int64            `json:"rows"`
		Tables map[string]int64 `json:"tables"`
		Cursor *int64           `json:"cursor"`
	}
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodPost, "/v1/data/import", ndjson(exported), &written))

	s.Equal(int64(1), written.Tables["agent_configs"])
	s.Equal(int64(len(exported)-1), written.Rows, "every line but the cursor was a row")
	s.Require().NotNil(written.Cursor)
}

func (s *DataMoveSuite) TestAnImportCannotClaimAnotherAppsNumberInAnotherForm() {
	ctx := context.Background()
	other, e164 := s.utils.uuid(), s.utils.number()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164: e164, Vendor: "plivo", Country: "US", CustomerID: other,
	}))

	// Plivo reads the number without its plus as the other app's.
	variant := strings.TrimPrefix(e164, "+")
	row, err := json.Marshal(map[string]any{
		"id": rand.Int64N(1 << 50), "e164": variant, "vendor": "plivo", "country": "US",
		"capabilities": []string{"voice"}, "monthly_cost_micros": 0, "customer_id": other,
		"tags": map[string]string{}, "purchased_at": time.Now().UTC(),
	})
	s.Require().NoError(err)
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/data/import",
		ndjson([]dataLine{{Table: "phone_numbers", Row: row}}))
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "E.164")

	status, _ = s.serverClient.call(http.MethodDelete, "/v1/phone/numbers/"+variant, nil)
	s.Equal(http.StatusNotFound, status)
	held, err := s.store.CustomerNumbers(ctx, other, false)
	s.Require().NoError(err)
	s.Len(held, 1, "the number stays with the app that holds it")
}

func (s *DataMoveSuite) TestARefusedNumberDoesNotSayWhoHoldsIt() {
	other, e164 := s.utils.uuid(), s.utils.number()
	s.Require().NoError(s.store.RecordNumber(context.Background(), &store.PhoneNumber{
		E164: e164, Vendor: "plivo", Country: "US", CustomerID: other,
	}))

	row, err := json.Marshal(map[string]any{
		"id": rand.Int64N(1 << 50), "e164": e164, "vendor": "bandwidth", "country": "US",
		"capabilities": []string{"voice"}, "monthly_cost_micros": 0, "customer_id": other,
		"tags": map[string]string{}, "purchased_at": time.Now().UTC(),
	})
	s.Require().NoError(err)
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/data/import",
		ndjson([]dataLine{{Table: "phone_numbers", Row: row}}))
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "this number cannot be imported")
	s.NotContains(failure, "app")
}

func (s *DataMoveSuite) TestSomethingThatIsNotAnExportIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/data/import", "not an export at all")

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "not an export")
}

func (s *DataMoveSuite) TestOnlyTheAppsOwnBackendMayTakeTheDataOut() {
	s.skipOnARouterWithShards()
	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodGet, "/v1/data/export", nil)
		return status
	})
}

func (s *DataMoveSuite) TestOnlyTheAppsOwnBackendMayWriteDataIn() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodPost, "/v1/data/import", nil)
		return status
	})
}

func (s *DataMoveSuite) TestOnlyTheAppsOwnBackendMayFollowTheChanges() {
	s.skipOnARouterWithShards()
	// An export somebody may not have is no safer if the change feed hands the same rows
	// over one at a time.
	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodGet, "/v1/data/changes", nil)
		return status
	})
}

// A write whose trigger looked for the recording before the export started it recorded
// nothing, so an export whose snapshot is read while it is open cannot have it either.
func (s *DataMoveSuite) TestAWriteOpenWhenAnExportStartsIsInTheExportOrItsChanges() {
	s.skipOnARouterWithShards()
	open := s.openWrite(s.customerID())
	s.waitAtMost(100 * time.Millisecond)

	first, body := s.serverClient.call(http.MethodGet, "/v1/data/export", nil)
	s.Require().NoError(open.tx.Commit())
	exported := string(body)
	if first != http.StatusOK {
		exported = ndjson(s.export())
	}
	var changed changedRows
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/data/changes?after="+strconv.FormatInt(s.cursorOf(exported), 10), nil, &changed))

	s.True(strings.Contains(exported, open.id) || strings.Contains(s.rendered(changed.Changes), open.id),
		"the write is in neither the export nor the changes after it")
	s.Equal(http.StatusServiceUnavailable, first, "an export that cannot wait it out says so")
}

func (s *DataMoveSuite) TestAnExportWaitsForAWriteOpenWhenItStartedToCommit() {
	s.skipOnARouterWithShards()
	open := s.openWrite(s.customerID())
	type answer struct {
		status int
		body   []byte
		err    error
	}
	request, err := http.NewRequest(http.MethodGet, s.serverClient.base()+"/v1/data/export", nil)
	s.Require().NoError(err)
	request.Header = s.serverClient.header.Clone()
	answered := make(chan answer, 1)
	go func() {
		response, err := s.server.Client().Do(request)
		if err != nil {
			answered <- answer{err: err}
			return
		}
		defer response.Body.Close()
		body, err := io.ReadAll(response.Body)
		answered <- answer{status: response.StatusCode, body: body, err: err}
	}()
	s.Require().EventuallyWithT(func(c *assert.CollectT) {
		var capturing bool
		assert.NoError(c, s.store.DB().QueryRowContext(context.Background(),
			"SELECT EXISTS (SELECT 1 FROM data_change_capture WHERE customer_id = ?)", s.customerID()).Scan(&capturing))
		assert.True(c, capturing, "the export has not started recording")
	}, settleFor, 10*time.Millisecond)
	s.Require().NoError(open.tx.Commit())

	select {
	case got := <-answered:
		s.Require().NoError(got.err)
		s.Equal(http.StatusOK, got.status)
		s.Contains(string(got.body), open.id)
	case <-time.After(exportWritesWait):
		s.Fail("the export did not answer once the write committed")
	}
}

// Postgres cannot tell another app's open write from this one's, so the export is bounded and
// answers rather than hangs.
func (s *DataMoveSuite) TestAnExportGivesUpOnAWriteThatStaysOpen() {
	s.skipOnARouterWithShards()
	stranger := s.openWrite("another-app-" + s.utils.uuid())
	s.waitAtMost(100 * time.Millisecond)

	status, body := s.serverClient.call(http.MethodGet, "/v1/data/export", nil)
	s.Equal(http.StatusServiceUnavailable, status)
	s.Contains(string(body), "still open")

	s.Require().NoError(stranger.tx.Rollback())
	s.export()
}

// A shard's tables record no changes, so a write to one after the export would never reach the
// changes: a router with a shard of its own refuses to be the source, before it records anything.
func (s *DataMoveSuite) TestARouterWithAShardOfItsOwnRefusesToBeTheSource() {
	s.refusesToBeTheSource(store.Options{DSN: s.sharedDSN(), ShardDSN: s.shardDSN()})
}

// A router writing its calls in the shared database still reads the other shards' calls, whose
// writes no change records either.
func (s *DataMoveSuite) TestARouterReadingOtherShardsRefusesToBeTheSource() {
	shared := s.sharedDSN()
	s.refusesToBeTheSource(store.Options{DSN: shared, Shards: map[string]string{"here": shared, "there": s.shardDSN()}})
}

// refusesToBeTheSource asserts a router on the databases options names refuses the export and the
// changes, and starts no recording for the caller.
func (s *DataMoveSuite) refusesToBeTheSource(options store.Options) {
	served, err := store.OpenShards(options)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(served.Close()) })
	other, _, _ := s.otherNodeServingFrom(served, nil, s.sealer, slog.New(slog.DiscardHandler))
	source := s.serverClient.on(other)

	for _, path := range []string{"/v1/data/export", "/v1/data/changes?after=0"} {
		status, body := source.call(http.MethodGet, path, nil)
		var refused ErrorResponse
		s.Require().NoError(json.Unmarshal(body, &refused), path)

		s.Equal(http.StatusBadRequest, status, path)
		s.Equal(codeNotConfigured, refused.Error.Code, path)
		s.Contains(refused.Error.Message, "with shards is not supported yet", path)
	}
	var capturing bool
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT EXISTS (SELECT 1 FROM data_change_capture WHERE customer_id = ?)", s.customerID()).Scan(&capturing))
	s.False(capturing, "a refused move started recording the app's changes")
}

// sharedDSN is the suite's database, which a router of shards shares.
func (s *DataMoveSuite) sharedDSN() string {
	return testDatabase(s.T(), os.Getenv("ROUTER_POSTGRES_DSN"))
}

// shardDSN is a migrated database of this suite's own, for one shard's tables.
func (s *DataMoveSuite) shardDSN() string {
	dsn := testenv.Database(s.sharedDSN(), "moveshard")
	shard, err := store.Open(dsn)
	s.Require().NoError(err)
	defer shard.Close()
	s.Require().NoError(migrate(context.Background(), shard, dsn))
	return dsn
}

// openWrite writes an agent config of a customer in a transaction the test ends.
func (s *DataMoveSuite) openWrite(customerID string) openWrite {
	tx, err := s.store.DB().BeginTx(context.Background(), nil)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = tx.Rollback() })
	id := "agent-" + s.utils.uuid()
	_, err = tx.ExecContext(context.Background(),
		"INSERT INTO agent_configs (id, customer_id, name, instructions) VALUES (?, ?, ?, ?)",
		id, customerID, id, "be brief")
	s.Require().NoError(err)
	return openWrite{tx: tx, id: id}
}

type openWrite struct {
	tx bun.Tx
	id string
}

// waitAtMost bounds how long this test's exports wait for the writes open when they start.
func (s *DataMoveSuite) waitAtMost(wait time.Duration) {
	s.Equal(exportWritesWait, s.router.exportWritesWait, "the default bound")
	s.router.exportWritesWait = wait
	s.T().Cleanup(func() { s.router.exportWritesWait = exportWritesWait })
}

// cursorOf is the cursor on the last line of an export.
func (s *DataMoveSuite) cursorOf(exported string) int64 {
	var cursor *int64
	decoder := json.NewDecoder(strings.NewReader(exported))
	for decoder.More() {
		var line dataLine
		s.Require().NoError(decoder.Decode(&line))
		cursor = line.Cursor
	}
	s.Require().NotNil(cursor, "the export did not finish")
	return *cursor
}

// skipOnARouterWithShards skips a test that takes the suite's own router as a move's source,
// which refuses to be one when ROUTER_POSTGRES_SHARD_DSN gives it shards (see the refusal tests
// below).
func (s *DataMoveSuite) skipOnARouterWithShards() {
	if s.store.HasShards() {
		s.T().Skip("a router with shards refuses to be a move's source")
	}
}

// export reads the whole snapshot, one JSON object per line.
func (s *DataMoveSuite) export() []dataLine {
	s.skipOnARouterWithShards()
	status, body := s.serverClient.call(http.MethodGet, "/v1/data/export", nil)
	s.Require().Equal(http.StatusOK, status)

	var lines []dataLine
	decoder := json.NewDecoder(strings.NewReader(string(body)))
	for decoder.More() {
		var line dataLine
		s.Require().NoError(decoder.Decode(&line))
		lines = append(lines, line)
	}
	s.Require().NotEmpty(lines)
	return lines
}

// changesSince waits for a change to be readable, asking from the beginning of what is
// recorded. The feed deliberately stops below the oldest transaction still open on the
// deployment, so a change written a moment ago is readable once what was in flight beside
// it has committed.
func (s *DataMoveSuite) changesSince(after int64) changedRows {
	s.skipOnARouterWithShards()
	var page changedRows
	s.Require().EventuallyWithT(func(c *assert.CollectT) {
		page = changedRows{}
		answered := s.serverClient.do(http.MethodGet,
			"/v1/data/changes?after="+strconv.FormatInt(after, 10), nil, &page)
		assert.Equal(c, http.StatusOK, answered, "the change feed's answer")
		assert.NotEmpty(c, page.Changes, "no change became readable")
	}, settleFor, 50*time.Millisecond)
	return page
}

// changedRows is what the change feed answers with.
type changedRows struct {
	Changes  []store.DataChange `json:"changes"`
	Cursor   int64              `json:"cursor"`
	CaughtUp bool               `json:"caught_up"`
}

// rendered is the changes as the JSON they carry, for asking whether a row is among them.
func (s *DataMoveSuite) rendered(changes []store.DataChange) string {
	encoded, err := json.Marshal(changes)
	s.Require().NoError(err)
	return string(encoded)
}

// ndjson is an export as the body an import takes.
func ndjson(lines []dataLine) string {
	var written strings.Builder
	for _, line := range lines {
		encoded, err := json.Marshal(line)
		if err != nil {
			continue
		}
		written.Write(encoded)
		written.WriteByte('\n')
	}
	return written.String()
}
