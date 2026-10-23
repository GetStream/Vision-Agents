//go:build integration

package routing

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/getsentry/sentry-go"
	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// WriterSuite runs the writer against a database of its own, since it empties requests and
// locks the table, and the shared Redis, under a customer of each test's own.
type WriterSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	live     *live.Client
	logs     *lockedBuffer
	logger   *slog.Logger
	customer string
}

func TestWriterSuite(t *testing.T) {
	suite.Run(t, new(WriterSuite))
}

func (s *WriterSuite) SetupSuite() {
	s.ctx = context.Background()
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR is not set")
	}

	own, err := store.Open(testenv.Database(dsn, "writer"))
	s.Require().NoError(err)
	var database string
	s.Require().NoError(own.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = own.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	s.Require().NoError(own.Migrate(s.ctx))
	s.store = own

	s.live, err = live.New(live.Options{Address: address})
	s.Require().NoError(err)
}

func (s *WriterSuite) TearDownSuite() {
	if s.live != nil {
		s.live.Close()
	}
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *WriterSuite) SetupTest() {
	_, err := s.store.ShardDB().ExecContext(s.ctx, "TRUNCATE requests")
	s.Require().NoError(err)
	s.logs = &lockedBuffer{}
	s.logger = slog.New(slog.NewTextHandler(s.logs, nil))
	s.customer = "writer-" + uuid.NewString()
}

// lockedBuffer is a log the writer writes while the test reads it.
type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// newWriter is a writer over the suite's store and Redis, closed when the test ends.
func (s *WriterSuite) newWriter(change func(*WriterOptions)) *Writer {
	options := WriterOptions{Store: s.store, Live: s.live, Logger: s.logger}
	if change != nil {
		change(&options)
	}
	writer := NewWriter(options)
	s.T().Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		writer.Close(ctx)
	})
	return writer
}

// downWriter is a writer whose Postgres refuses every connection, as one that is down does,
// over the suite's Redis. It is closed when the test ends, losing what it holds.
func (s *WriterSuite) downWriter(change func(*WriterOptions)) *Writer {
	down, err := store.Open("postgres://postgres:postgres@127.0.0.1:1/down_test?sslmode=disable")
	s.Require().NoError(err)
	options := WriterOptions{Store: down, Live: s.live, Logger: s.logger}
	if change != nil {
		change(&options)
	}
	writer := NewWriter(options)
	s.T().Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()
		writer.Close(ctx)
		s.Require().NoError(down.Close())
	})
	return writer
}

// counted waits until the test's customer's live llm counters hold n requests, and returns them.
func (s *WriterSuite) counted(n int64) live.CustomerUsage {
	var used live.CustomerUsage
	s.Require().Eventually(func() bool {
		var err error
		used, err = s.live.Usage(s.ctx, "llm", s.customer)
		return err == nil && used.Requests == n
	}, 5*time.Second, 10*time.Millisecond, "want %d requests counted", n)
	return used
}

// row is a row of the test's customer with a usage id of its own.
func (s *WriterSuite) row(change func(*store.Request)) queued {
	request := store.Request{
		UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: s.customer,
		Provider: "openai", Model: "gpt", StartedAt: time.Now().UTC(), OutputTokens: 10, CostMicros: 25, Success: true,
	}
	if change != nil {
		change(&request)
	}
	return queued{request: request}
}

// rows is what the test's customer has in requests.
func (s *WriterSuite) rows() []store.Request {
	var rows []store.Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).Where("customer_id = ?", s.customer).Order("id").Scan(s.ctx))
	return rows
}

// written waits until the test's customer has n rows, and returns them.
func (s *WriterSuite) written(n int) []store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = s.rows()
		return len(rows) == n
	}, 10*time.Second, 20*time.Millisecond, "want %d rows", n)
	return rows
}

// lock holds every write to requests until the returned function is called, as a Postgres
// that stopped answering would.
func (s *WriterSuite) lock() (release func()) {
	tx, err := s.store.ShardDB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	_, err = tx.ExecContext(s.ctx, "LOCK TABLE requests IN ACCESS EXCLUSIVE MODE")
	s.Require().NoError(err)
	var once sync.Once
	release = func() { once.Do(func() { _ = tx.Rollback() }) }
	s.T().Cleanup(release)
	return release
}

// risen waits until counter has gone up by at least by since it read before.
func (s *WriterSuite) risen(counter prometheus.Collector, before, by float64) {
	s.Require().Eventually(func() bool {
		return testutil.ToFloat64(counter)-before >= by
	}, 10*time.Second, 10*time.Millisecond)
}

// holding is a gate that keeps the writer in its first row's admission until released, so
// whatever is queued meanwhile leaves the queue as one batch.
type holding struct {
	release chan struct{}
}

func (h holding) Admit(ctx context.Context, _ string) (Admission, error) {
	select {
	case <-h.release:
	case <-ctx.Done():
	}
	return Admission{}, nil
}

// heldProvider names the row hold queues.
const heldProvider = "held"

// hold queues a row whose admission keeps the writer busy until the returned function is
// called.
func (s *WriterSuite) hold(writer *Writer) (release func()) {
	gate := holding{release: make(chan struct{})}
	first := s.row(func(request *store.Request) { request.Provider = heldProvider })
	first.gate = gate
	s.Require().True(writer.enqueue(first))
	s.Require().Eventually(func() bool { return writer.inFlight.Load() == 1 }, 5*time.Second, time.Millisecond)
	return func() { close(gate.release) }
}

// lostRow is the row a "usage row lost" line carries, decoded as replaying it would.
var lostRow = regexp.MustCompile(`msg="usage row lost".* row=("(?:[^"\\]|\\.)*")`)

// recording captures what a writer reports to Sentry.
type recording struct {
	mu     sync.Mutex
	events []*sentry.Event
}

func (r *recording) hub(s *WriterSuite) *sentry.Hub {
	client, err := sentry.NewClient(sentry.ClientOptions{
		BeforeSend: func(event *sentry.Event, _ *sentry.EventHint) *sentry.Event {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.events = append(r.events, event)
			return nil
		},
	})
	s.Require().NoError(err)
	return sentry.NewHub(client, sentry.NewScope())
}

// lostRows adds up the rows every lost-row event reports.
func (r *recording) lostRows() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	rows := 0
	for _, event := range r.events {
		if event.Tags["usage_alert"] == "lost" {
			rows += event.Contexts["usage"]["rows"].(int)
		}
	}
	return rows
}

func (r *recording) tagged(key, value string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	count := 0
	for _, event := range r.events {
		if event.Tags[key] == value {
			count++
		}
	}
	return count
}

func (s *WriterSuite) TestARowIsWrittenOnce() {
	before := testutil.ToFloat64(usageWritten)
	writer := s.newWriter(nil)
	row := s.row(nil)

	s.Require().True(writer.enqueue(row))

	rows := s.written(1)
	s.Equal(row.request.UsageID, rows[0].UsageID)
	s.EqualValues(25, rows[0].CostMicros)
	s.risen(usageWritten, before, 1)
}

func (s *WriterSuite) TestTheSameUsageIDIsWrittenOnce() {
	before := testutil.ToFloat64(usageDuplicates)
	writer := s.newWriter(nil)
	first := s.row(nil)
	again := s.row(func(request *store.Request) { request.UsageID = first.request.UsageID })

	writer.enqueue(first)
	writer.enqueue(again)

	s.risen(usageDuplicates, before, 1)
	s.Len(s.written(1), 1)
	s.Equal(1.0, testutil.ToFloat64(usageDuplicates)-before)
}

func (s *WriterSuite) TestAFailedBatchIsRetriedUntilWritten() {
	errorsBefore := testutil.ToFloat64(usageWriteErrors)
	writer := s.newWriter(func(options *WriterOptions) { options.WriteTimeout = 200 * time.Millisecond })
	release := s.lock()
	sent := []queued{s.row(nil), s.row(nil), s.row(nil)}
	for _, row := range sent {
		s.Require().True(writer.enqueue(row))
	}

	s.risen(usageWriteErrors, errorsBefore, 1)
	release()

	rows := s.written(3)
	for i, row := range sent {
		s.Equal(row.request.UsageID, rows[i].UsageID, "each row written once, in the order recorded")
	}
	s.Contains(s.logs.String(), "could not write usage rows, trying again")
}

func (s *WriterSuite) TestRecordNeverWaitsOnTheDatabase() {
	writer := s.newWriter(nil)
	recorder := NewRecorderOn(STT, writer, nil, s.logger)
	config := ProviderConfig{Provider: "deepgram", Model: "flux", Price: Price{PerAudioHour: 0.6}}
	release := s.lock()

	started := time.Now()
	for range 1000 {
		recorder.Record(config, Stat{Owner: Owner{CustomerID: s.customer}, StartedAt: started, Usage: Usage{AudioMs: 1000}, Success: true})
	}
	elapsed := time.Since(started)

	s.Less(elapsed, 10*time.Millisecond, "a thousand stats queued while Postgres is stuck")
	release()
	s.written(1000)
}

func (s *WriterSuite) TestAFullQueueLosesTheRowLoudly() {
	lostBefore := testutil.ToFloat64(usageLost.WithLabelValues(lostQueueFull))
	microsBefore := testutil.ToFloat64(usageLostMicros)
	writer := s.newWriter(func(options *WriterOptions) { options.QueueSize = 1 })
	release := s.lock()
	sent := []queued{s.row(nil), s.row(nil), s.row(nil)}

	var lost []uuid.UUID
	for _, row := range sent {
		if !writer.enqueue(row) {
			lost = append(lost, row.request.UsageID)
		}
	}

	s.Require().NotEmpty(lost, "a queue of one cannot take three rows while the first is stuck")
	s.risen(usageLost.WithLabelValues(lostQueueFull), lostBefore, float64(len(lost)))
	s.Equal(float64(len(lost)), testutil.ToFloat64(usageLost.WithLabelValues(lostQueueFull))-lostBefore)
	s.Equal(float64(25*len(lost)), testutil.ToFloat64(usageLostMicros)-microsBefore)
	for _, id := range lost {
		s.Contains(s.logs.String(), "usage_id="+id.String())
	}
	s.Contains(s.logs.String(), `msg="usage row lost" reason=queue_full`)
	release()
	s.written(len(sent) - len(lost))
}

func (s *WriterSuite) TestARowLostToAFullQueueStillReachesTheLiveCounters() {
	lostBefore := testutil.ToFloat64(usageLost.WithLabelValues(lostQueueFull))
	writer := s.downWriter(func(options *WriterOptions) { options.QueueSize = 1 })

	for i := range 3 {
		writer.enqueue(s.row(nil))
		s.counted(int64(i + 1))
	}

	s.risen(usageLost.WithLabelValues(lostQueueFull), lostBefore, 2)
	used := s.counted(3)
	s.EqualValues(75, used.CostMicros, "quotas and spend see every row, written or not")
}

func (s *WriterSuite) TestWithPostgresDownAFatalStatStillTakesItsProviderOutAtOnce() {
	errorsBefore := testutil.ToFloat64(usageWriteErrors)
	writer := s.downWriter(nil)
	recorder := NewRecorderOn(LLM, writer, nil, s.logger)
	config := ProviderConfig{Provider: "fatal-" + uuid.NewString(), Model: "m"}
	recorder.Record(config, Stat{Owner: Owner{CustomerID: s.customer}, StartedAt: time.Now().UTC(), Success: true})
	s.risen(usageWriteErrors, errorsBefore, 1)

	recorder.Record(config, Stat{
		Owner: Owner{CustomerID: s.customer}, StartedAt: time.Now().UTC(), ErrorCode: "account_refused", Fatal: true,
	})

	s.Require().Eventually(func() bool {
		health, err := s.live.Health(s.ctx, string(LLM), config.Provider, "m")
		return err == nil && !health.Available
	}, 2*time.Second, 10*time.Millisecond, "the batch Postgres keeps refusing does not hold back provider health")
	used := s.counted(2)
	s.EqualValues(1, used.Errors)
}

func (s *WriterSuite) TestALostRowIsLoggedSoItCanBeReplayed() {
	writer := s.newWriter(func(options *WriterOptions) { options.QueueSize = 1 })
	s.lock()
	sent := []queued{s.row(nil), s.row(nil), s.row(func(request *store.Request) { request.Tags = Tags{"project": `a "quoted" one`} })}
	var lost queued
	for _, row := range sent {
		if !writer.enqueue(row) {
			lost = row
		}
	}
	s.Require().NotEqual(uuid.Nil, lost.request.UsageID)

	var match [][]string
	s.Require().Eventually(func() bool {
		match = lostRow.FindAllStringSubmatch(s.logs.String(), -1)
		return len(match) > 0 && strings.Contains(match[len(match)-1][0], lost.request.UsageID.String())
	}, 5*time.Second, 10*time.Millisecond)
	raw, err := strconv.Unquote(match[len(match)-1][1])
	s.Require().NoError(err)
	var replayed store.Request
	s.Require().NoError(json.Unmarshal([]byte(raw), &replayed))

	s.Equal(lost.request.UsageID, replayed.UsageID)
	s.Equal(lost.request.CostMicros, replayed.CostMicros)
	s.Equal(lost.request.Tags, replayed.Tags)
}

func (s *WriterSuite) TestCloseWritesWhatIsQueued() {
	writer := NewWriter(WriterOptions{Store: s.store, Live: s.live, Logger: s.logger})
	for range 50 {
		s.Require().True(writer.enqueue(s.row(nil)))
	}

	ctx, cancel := context.WithTimeout(s.ctx, 10*time.Second)
	defer cancel()
	writer.Close(ctx)

	s.Len(s.rows(), 50, "every row queued before Close is written by the time it returns")
}

func (s *WriterSuite) TestCloseLosesWhatItCannotWrite() {
	before := testutil.ToFloat64(usageLost.WithLabelValues(lostShutdown))
	writer := NewWriter(WriterOptions{Store: s.store, Live: s.live, Logger: s.logger})
	release := s.lock()
	sent := []queued{s.row(nil), s.row(nil), s.row(nil)}
	for _, row := range sent {
		s.Require().True(writer.enqueue(row))
	}

	ctx, cancel := context.WithTimeout(s.ctx, 300*time.Millisecond)
	defer cancel()
	started := time.Now()
	writer.Close(ctx)

	s.Less(time.Since(started), time.Second, "Close stops at its deadline, not at the write timeout")
	s.Equal(3.0, testutil.ToFloat64(usageLost.WithLabelValues(lostShutdown))-before)
	for _, row := range sent {
		s.Contains(s.logs.String(), "usage_id="+row.request.UsageID.String())
	}
	s.False(writer.enqueue(s.row(nil)), "a closed writer takes nothing")
	release()
	s.Empty(s.rows())
}

func (s *WriterSuite) TestLiveCountersAreUpdatedOncePerRow() {
	errorsBefore := testutil.ToFloat64(usageWriteErrors)
	writer := s.newWriter(func(options *WriterOptions) { options.WriteTimeout = 100 * time.Millisecond })
	release := s.lock()

	writer.enqueue(s.row(nil))
	s.risen(usageWriteErrors, errorsBefore, 2)
	release()
	s.written(1)

	used, err := s.live.Usage(s.ctx, "llm", s.customer)
	s.Require().NoError(err)
	s.EqualValues(1, used.Requests, "a batch tried three times is one request")
	s.EqualValues(25, used.CostMicros)
}

func (s *WriterSuite) TestAFatalStatStillReachesProviderHealth() {
	writer := s.newWriter(nil)
	recorder := NewRecorderOn(STT, writer, nil, s.logger)
	provider := "fatal-" + uuid.NewString()

	recorder.Record(ProviderConfig{Provider: provider, Model: "m"}, Stat{
		Owner: Owner{CustomerID: s.customer}, StartedAt: time.Now().UTC(), ErrorCode: "account_refused", Fatal: true,
	})

	s.Require().Eventually(func() bool {
		health, err := s.live.Health(s.ctx, string(STT), provider, "m")
		return err == nil && !health.Available
	}, 10*time.Second, 20*time.Millisecond)
}

func (s *WriterSuite) TestTheOrganizationIsWrittenOnTheRow() {
	writer := s.newWriter(nil)
	row := s.row(nil)
	row.gate = admitting(Admission{OrganizationID: "o1", Tags: Tags{"team": "core"}})

	writer.enqueue(row)

	rows := s.written(1)
	s.Equal("o1", rows[0].OrganizationID)
	s.Equal(Tags{"team": "core"}, Tags(rows[0].Tags), "the gate's tags are still laid over the row")
}

func (s *WriterSuite) TestAGateThatKnowsNoOrganizationKeepsTheRowsOwn() {
	writer := s.newWriter(nil)
	row := s.row(func(request *store.Request) { request.OrganizationID = "from-the-owner" })
	row.gate = admitting(Admission{})

	writer.enqueue(row)

	s.Equal("from-the-owner", s.written(1)[0].OrganizationID)
}

func (s *WriterSuite) TestWithNoUsageServiceRowsAreNotQueued() {
	writer := s.newWriter(nil)

	writer.enqueue(s.row(func(request *store.Request) { request.ToShip = true }))

	rows := s.written(1)
	s.False(rows[0].ToShip, "nothing would ever ship it")
	s.Nil(rows[0].ShippedAt)
}

func (s *WriterSuite) TestTheWriterMarksAMissingKeyUnattributed() {
	before := testutil.ToFloat64(usageUnattributed)
	writer := s.newWriter(nil)
	keyless := s.row(func(request *store.Request) { request.AgentID = "agent-without-a-key" })
	keyed := s.row(func(request *store.Request) { request.APIKey = "key-1" })

	writer.enqueue(keyless)
	writer.enqueue(keyed)

	rows := s.written(2)
	s.Equal(unattributed, rows[0].APIKey)
	s.Equal("key-1", rows[1].APIKey)
	s.Equal(1.0, testutil.ToFloat64(usageUnattributed)-before)
	s.Contains(s.logs.String(), "agent=agent-without-a-key", "the log names where the owner site that missed it is")
	s.Contains(s.logs.String(), `level=WARN msg="usage row recorded with no api key or origin"`)
}

func (s *WriterSuite) TestManyRequestIDsADataMoveTookDoNotHoldTheWriterUp() {
	var next int64
	s.Require().NoError(s.store.ShardDB().QueryRowContext(s.ctx,
		"SELECT nextval(pg_get_serial_sequence('requests', 'id'))").Scan(&next))
	moved := make([]store.Request, 500)
	for i := range moved {
		moved[i] = s.row(func(request *store.Request) {
			request.ID, request.CustomerID = next+1+int64(i), "moved-"+s.customer
		}).request
	}
	_, err := s.store.ShardDB().NewInsert().Model(&moved).Exec(s.ctx)
	s.Require().NoError(err)
	errorsBefore := testutil.ToFloat64(usageWriteErrors)
	writer := s.newWriter(nil)

	started := time.Now()
	for range 3 {
		s.Require().True(writer.enqueue(s.row(nil)))
	}
	s.written(3)

	s.Less(time.Since(started), 2*time.Second, "500 ids taken ahead of the sequence")
	s.Equal(0.0, testutil.ToFloat64(usageWriteErrors)-errorsBefore, "an id taken is tried again at once, not after a wait")
}

// attributing is a gate that says whose work is after the fact, failing while failing is set,
// and keeping count of what it was asked.
type attributing struct {
	failing  atomic.Bool
	asked    atomic.Int64
	admitted atomic.Int64
	// stuck lookups wait for their context to end, but for the cached customers', which are
	// answered at once whatever the context, as policy.Enforcer answers from its cache.
	stuck  bool
	cached map[string]bool
}

func (a *attributing) Admit(context.Context, string) (Admission, error) {
	a.admitted.Add(1)
	return Admission{}, nil
}

func (a *attributing) Attribute(ctx context.Context, customerID string) (Admission, error) {
	a.asked.Add(1)
	if a.stuck && !a.cached[customerID] {
		<-ctx.Done()
		return Admission{}, ctx.Err()
	}
	if a.failing.Load() {
		return Admission{}, errors.New("the policies could not be read")
	}
	return Admission{OrganizationID: "o1", Tags: Tags{"team": "core"}}, nil
}

func (s *WriterSuite) TestARowWhoseLookupFailedIsLookedUpAgainOnTheNextTry() {
	errorsBefore := testutil.ToFloat64(usageWriteErrors)
	writer := s.newWriter(func(options *WriterOptions) { options.WriteTimeout = 200 * time.Millisecond })
	gate := &attributing{}
	gate.failing.Store(true)
	release := s.lock()
	row := s.row(nil)
	row.gate = gate

	writer.enqueue(row)
	s.risen(usageWriteErrors, errorsBefore, 1)
	gate.failing.Store(false)
	release()

	written := s.written(1)[0]
	s.Equal("o1", written.OrganizationID, "the Postgres blip that failed the lookup failed the write too")
	s.Equal(Tags{"team": "core"}, Tags(written.Tags))
	s.GreaterOrEqual(gate.asked.Load(), int64(2))
	s.Zero(gate.admitted.Load(), "a gate that can attribute is not asked to admit, which keeps what a failure made of it")
}

func (s *WriterSuite) TestAHungLookupHoldsABatchUpForOneLookupTimeout() {
	unresolvedBefore := testutil.ToFloat64(usageUnresolved)
	writer := s.newWriter(nil)
	cached := s.customer + "-cached"
	gate := &attributing{stuck: true, cached: map[string]bool{cached: true}}
	release := s.hold(writer)
	customers := []string{s.customer, s.customer + "-b", cached, s.customer + "-c"}
	for _, customer := range customers {
		row := s.row(func(request *store.Request) { request.CustomerID = customer })
		row.gate = gate
		writer.enqueue(row)
	}
	s.Require().Eventually(func() bool { return len(writer.prepared) == len(customers) }, 5*time.Second, time.Millisecond,
		"the rows wait for the held one as one batch")

	started := time.Now()
	release()
	s.risen(usageUnresolved, unresolvedBefore, 3)

	s.Less(time.Since(started), lookupTimeout+time.Second, "each lookup is bounded, and the rest do not wait out their own")
	s.Equal(int64(len(customers)), gate.asked.Load(), "the rest are still asked, with the context ended")
	for _, row := range s.written(2) {
		s.Empty(row.OrganizationID, "the held row and the one written without it")
	}
	var organizations map[string]string
	s.Require().Eventually(func() bool {
		var rows []store.Request
		s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).Where("customer_id IN (?)", bun.In(customers[1:])).Scan(s.ctx))
		organizations = map[string]string{}
		for _, row := range rows {
			organizations[row.CustomerID] = row.OrganizationID
		}
		return len(organizations) == 3
	}, 5*time.Second, 10*time.Millisecond)
	s.Equal(map[string]string{customers[1]: "", cached: "o1", customers[3]: ""}, organizations,
		"a cached answer still applies after another lookup ran out of time")
	s.Contains(s.logs.String(), "usage rows written without their organization")
}

func (s *WriterSuite) TestTheWriterDefaultsAMissingTier() {
	writer := s.newWriter(nil)

	writer.enqueue(s.row(nil))
	writer.enqueue(s.row(func(request *store.Request) { request.OrganizationTier = "tier-a" }))

	rows := s.written(2)
	s.Equal(defaultTier, rows[0].OrganizationTier)
	s.Equal("tier-a", rows[1].OrganizationTier)
}

func (s *WriterSuite) TestTwoRecordersShareOneWriter() {
	writer := s.newWriter(nil)
	speech, chat := NewRecorderOn(STT, writer, nil, s.logger), NewRecorderOn(LLM, writer, nil, s.logger)
	release := s.hold(writer)
	stat := Stat{Owner: Owner{CustomerID: s.customer}, StartedAt: time.Now().UTC(), Success: true}

	speech.Record(ProviderConfig{Provider: "deepgram", Model: "flux"}, stat)
	chat.Record(ProviderConfig{Provider: "openai", Model: "gpt"}, stat)
	release()
	s.written(3)

	var transactions []string
	s.Require().NoError(s.store.ShardDB().NewRaw(
		"SELECT DISTINCT xmin::text FROM requests WHERE customer_id = ? AND provider <> ?", s.customer, heldProvider,
	).Scan(s.ctx, &transactions))
	s.Len(transactions, 1, "both recorders' rows are written in one batch")
}

func (s *WriterSuite) TestARecordedStatKeepsItsUsageIDAndSession() {
	writer := s.newWriter(nil)
	recorder := NewRecorderOn(LLM, writer, nil, s.logger)
	usageID := UsageID("call_leg", s.customer)
	stat := Stat{Owner: Owner{CustomerID: s.customer, SessionID: "session-1"}, UsageID: usageID, StartedAt: time.Now().UTC(), Success: true}

	recorder.Record(ProviderConfig{Provider: "openai", Model: "gpt"}, stat)
	recorder.Record(ProviderConfig{Provider: "openai", Model: "gpt"}, stat)

	s.Require().Eventually(func() bool { return writer.pending() == 0 }, 10*time.Second, 10*time.Millisecond)
	rows := s.written(1)
	s.Equal(usageID, rows[0].UsageID, "recorded twice, written once")
	s.Equal("session-1", rows[0].SessionID)
}

func (s *WriterSuite) TestARecorderOfItsOwnWritesBeforeItCloses() {
	recorder := NewRecorder(LLM, s.store, s.live, nil, s.logger)

	recorder.Record(ProviderConfig{Provider: "openai", Model: "gpt"}, Stat{Owner: Owner{CustomerID: s.customer}, StartedAt: time.Now().UTC(), Success: true})
	recorder.Close()

	s.Len(s.rows(), 1)
	s.NotEqual(uuid.Nil, s.rows()[0].UsageID, "every row has a usage id, whoever made the recorder")
}

func (s *WriterSuite) TestARowPostgresRefusesIsLostAloneAndTheRestAreWritten() {
	before := testutil.ToFloat64(usageLost.WithLabelValues(lostInvalid))
	writer := s.newWriter(nil)
	release := s.hold(writer)
	refused := s.row(func(request *store.Request) { request.AgentID = tooBigToIndex() })

	writer.enqueue(s.row(nil))
	writer.enqueue(refused)
	writer.enqueue(s.row(nil))
	release()

	s.written(3)
	s.Equal(1.0, testutil.ToFloat64(usageLost.WithLabelValues(lostInvalid))-before)
	s.Contains(s.logs.String(), "usage_id="+refused.request.UsageID.String())
}

func (s *WriterSuite) TestLostRowsAreReportedToSentryAtMostOnceAMinute() {
	var reported recording
	writer := NewWriter(WriterOptions{
		Store: s.store, Live: s.live, Logger: s.logger, QueueSize: 1, WriteTimeout: 200 * time.Millisecond, Hub: reported.hub(s),
	})
	s.lock()

	s.Require().True(writer.enqueue(s.row(nil)), "the first row is kept, and stuck")
	lost := 0
	for range 4 {
		s.Require().False(writer.enqueue(s.row(nil)))
		lost++
		// The first loss is reported at once, and the others wait for the minute to pass.
		s.Require().Eventually(func() bool { return reported.lostRows() > 0 }, 5*time.Second, time.Millisecond)
	}

	s.Never(func() bool { return reported.tagged("usage_alert", "lost") > 1 }, 200*time.Millisecond, 10*time.Millisecond)
	s.Less(reported.lostRows(), lost, "the rest wait for the minute to pass")

	ctx, cancel := context.WithTimeout(s.ctx, 300*time.Millisecond)
	defer cancel()
	writer.Close(ctx)

	s.Equal(2, reported.tagged("usage_alert", "lost"), "what was waiting is reported as the writer closes")
	s.Equal(lost+1, reported.lostRows(), "every row lost, the one Close gave up on too, is in a report")
}

func (s *WriterSuite) TestAFailingWriteIsReportedToSentryAtMostOnceAMinute() {
	var reported recording
	errorsBefore := testutil.ToFloat64(usageWriteErrors)
	writer := s.newWriter(func(options *WriterOptions) {
		options.WriteTimeout = 50 * time.Millisecond
		options.Hub = reported.hub(s)
	})
	release := s.lock()

	writer.enqueue(s.row(nil))
	s.risen(usageWriteErrors, errorsBefore, 3)
	release()
	s.written(1)

	s.Equal(1, reported.tagged("component", "usage_writer"))
}

func (s *WriterSuite) TestTheQueueGaugesCountWhatIsNotWritten() {
	writer := s.newWriter(nil)
	release := s.lock()
	for range 3 {
		writer.enqueue(s.row(nil))
	}

	s.Require().Eventually(func() bool { return writer.pending() == 3 }, 5*time.Second, time.Millisecond)
	time.Sleep(50 * time.Millisecond)
	s.GreaterOrEqual(writer.waited(), 50*time.Millisecond, "the oldest row has waited since it was queued")
	release()
	s.written(3)
	s.Require().Eventually(func() bool { return writer.pending() == 0 && writer.waited() == 0 }, 5*time.Second, time.Millisecond)
}

// tooBigToIndex is an id longer than a btree index row may be, even compressed.
func tooBigToIndex() string {
	raw := make([]byte, 10_000)
	_, _ = rand.Read(raw)
	return hex.EncodeToString(raw)
}
