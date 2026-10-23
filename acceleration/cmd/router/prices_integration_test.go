//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// PriceSyncSuite starts the price list the way a router does, on databases of its own that it
// empties first.
type PriceSyncSuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
	logger   *slog.Logger
}

func TestPriceSyncSuite(t *testing.T) {
	suite.Run(t, new(PriceSyncSuite))
}

func (s *PriceSyncSuite) SetupTest() {
	s.ctx = context.Background()
	s.settings = config.Defaults()
	s.settings.Postgres.DSN = s.emptyDatabase("pricesync")
	s.logger = slog.New(slog.DiscardHandler)
}

// emptyDatabase is a database of this suite's own with an empty schema.
func (s *PriceSyncSuite) emptyDatabase(name string) string {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	own := testenv.Database(dsn, name)
	empty, err := store.Open(own)
	s.Require().NoError(err)
	defer empty.Close()
	var database string
	s.Require().NoError(empty.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = empty.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	return own
}

// started is the store a router opens at start, migrated.
func (s *PriceSyncSuite) started() *store.Store {
	pgStore, err := openStore(s.ctx, s.settings)
	s.Require().NoError(err)
	s.T().Cleanup(func() { pgStore.Close() })
	return pgStore
}

// ownFile writes a deployment's own price file and points the settings at it.
func (s *PriceSyncSuite) ownFile(body string) {
	path := filepath.Join(s.T().TempDir(), "own-prices.json")
	s.Require().NoError(os.WriteFile(path, []byte(body), 0o600))
	s.settings.PricesFile = path
}

// releasedAs runs the rest of the test as a binary built for that release.
func (s *PriceSyncSuite) releasedAs(name string) {
	previous := release
	release = name
	s.T().Cleanup(func() { release = previous })
}

// startRouter loads the price files as a router's start does.
func (s *PriceSyncSuite) startRouter(pgStore *store.Store) {
	load, err := loadPriceFiles(s.settings)
	s.Require().NoError(err)
	startPriceSync(s.ctx, pgStore, load, time.Hour, s.logger)
}

func (s *PriceSyncSuite) changes(pgStore *store.Store) []store.PriceChange {
	changes, err := pgStore.PriceChanges(s.ctx, store.PriceChangeFilter{})
	s.Require().NoError(err)
	return changes
}

// ownEntry is an item vendor.json does not have.
const ownEntry = `{"modality": "llm", "provider": "own", "model": "own-model", "rates": {"input_tokens": {"usd": 1, "per": 1000000}}}`

func (s *PriceSyncSuite) TestStartupSyncsAndAudits() {
	s.releasedAs("v1.3.0")
	vendor := prices.Vendor().Entries[0]
	override, err := json.Marshal(pricing.Entry{Modality: vendor.Modality, Provider: vendor.Provider, Model: vendor.Model, When: vendor.When,
		Rates: map[pricing.Meter]pricing.Rate{pricing.InputTokens: {USD: 99, Per: 1_000_000}}})
	s.Require().NoError(err)
	s.ownFile(`{"markup_bps": 1500, "entries": [` + ownEntry + `, ` + string(override) + `]}`)
	pgStore := s.started()

	s.startRouter(pgStore)

	changes := s.changes(pgStore)
	s.Require().Len(changes, 1)
	seed := changes[0]
	s.Equal(store.PriceChangeSeed, seed.Kind)
	s.Equal(s.settings.PricesFile+", vendor.json, router v1.3.0", seed.Source)
	s.Equal("router v1.3.0", seed.CreatedBy)
	markup, err := pgStore.ActiveMarkup(s.ctx, time.Now())
	s.Require().NoError(err)
	s.EqualValues(1_500, markup)
	active, err := pgStore.ActivePriceEntries(s.ctx, time.Now())
	s.Require().NoError(err)
	s.Len(active, len(prices.Vendor().Entries)+1)
	for _, row := range active {
		if row.Modality == vendor.Modality && row.Provider == vendor.Provider && row.Model == vendor.Model && len(row.Attributes) == 0 {
			s.InDelta(99.0, row.Rates[pricing.InputTokens].USD, 1e-12, "the deployment's own file wins over vendor.json")
		}
	}
	var audits []store.AuditEntry
	s.Require().NoError(pgStore.DB().NewSelect().Model(&audits).Where("resource_type = ?", store.AuditResourcePriceList).Scan(s.ctx))
	s.Require().Len(audits, 1)
	s.Equal(seed.Name, audits[0].ResourceID)
	s.Equal(store.AuditSourceCLI, audits[0].Source)
	s.Equal(store.AuditCreated, audits[0].Action)
	s.Empty(audits[0].CustomerID)
}

func (s *PriceSyncSuite) TestTheNextReleaseAddsOnlyItsNewItems() {
	s.releasedAs("v1.3.0")
	pgStore := s.started()
	s.startRouter(pgStore)
	s.releasedAs("v1.4.0")
	s.ownFile(`{"markup_bps": 1500, "entries": [` + ownEntry + `]}`)

	s.startRouter(pgStore)

	changes := s.changes(pgStore)
	s.Require().Len(changes, 2)
	s.Regexp(`^\d{14}_release_v1_4_0$`, changes[0].Name)
	s.Equal("Items new in v1.4.0", changes[0].Reason)
	_, entries, err := pgStore.PriceChange(s.ctx, changes[0].Name)
	s.Require().NoError(err)
	s.Require().Len(entries, 1)
	s.Equal("own-model", entries[0].Model)
	markup, err := pgStore.ActiveMarkup(s.ctx, time.Now())
	s.Require().NoError(err)
	s.Zero(markup, "only the first load reads a markup")
}

// TestRoutersStartingAtOnceLoadOnce: routers that start together on an empty database each
// migrate and load the price files, and between them write the seed once.
func (s *PriceSyncSuite) TestRoutersStartingAtOnceLoadOnce() {
	load, err := loadPriceFiles(s.settings)
	s.Require().NoError(err)
	ctx, cancel := context.WithTimeout(s.ctx, time.Minute)
	defer cancel()

	var wg sync.WaitGroup
	failed := make([]error, 6)
	for i := range failed {
		wg.Go(func() {
			pgStore, err := openStore(ctx, s.settings)
			if err != nil {
				failed[i] = err
				return
			}
			defer pgStore.Close()
			_, _, _, failed[i] = pgStore.SyncPrices(ctx, load)
		})
	}
	wg.Wait()

	for _, err := range failed {
		s.Require().NoError(err)
	}
	s.Len(s.changes(s.started()), 1)
}

func (s *PriceSyncSuite) TestSyncDryRunWritesNothing() {
	pgStore := s.started()
	var out bytes.Buffer

	s.Require().NoError(runPricesSync(s.ctx, []string{"--dry-run"}, s.settings, &out))

	s.Contains(out.String(), "the next start seeds the price list from vendor.json")
	s.Empty(s.changes(pgStore))
	s.startRouter(pgStore)
	out.Reset()
	s.Require().NoError(runPricesSync(s.ctx, []string{"--dry-run"}, s.settings, &out))
	s.Equal("nothing to add: the price list has every item in vendor.json\n", out.String())
}

func (s *PriceSyncSuite) TestSyncWithoutDryRunWritesNothing() {
	pgStore := s.started()

	err := runPricesSync(s.ctx, nil, s.settings, &bytes.Buffer{})

	s.ErrorContains(err, "only takes --dry-run")
	s.Empty(s.changes(pgStore))
}

func (s *PriceSyncSuite) TestShowSaysTheListInForce() {
	s.ownFile(`{"markup_bps": 1500, "entries": []}`)
	pgStore := s.started()
	s.startRouter(pgStore)
	var out bytes.Buffer

	s.Require().NoError(runPricesShow(s.ctx, []string{"--json"}, s.settings, &out))

	var shown shownList
	s.Require().NoError(json.Unmarshal(out.Bytes(), &shown))
	s.EqualValues(1_500, shown.MarkupBps)
	s.Len(shown.Entries, len(prices.Vendor().Entries))
	s.Empty(shown.Unpriced, "every model router.yaml routes is priced or local")
	priced := shown.Entries[0]
	for meter, rate := range priced.Rates {
		s.InDelta(rate.USD*1.15, priced.Charged[meter].USD, 1e-12)
	}
	out.Reset()
	s.Require().NoError(runPricesShow(s.ctx, []string{"--unpriced"}, s.settings, &out))
	s.Empty(out.String())
}

func (s *PriceSyncSuite) TestShowListsWhatIsUnpriced() {
	pgStore := s.started()
	s.startRouter(pgStore)
	routingConfig := filepath.Join(s.T().TempDir(), "routing.yaml")
	s.Require().NoError(os.WriteFile(routingConfig, []byte(`llm:
  providers:
    - provider: nobody
      model: unpriced
      languages: [en]
`), 0o600))
	s.settings.RoutingConfig = routingConfig
	var out bytes.Buffer

	s.Require().NoError(runPricesShow(s.ctx, []string{"--unpriced"}, s.settings, &out))

	s.Equal("llm/nobody/unpriced\n", out.String())
}

// TestAHeldPricesLockDoesNotKeepTheRouterFromServing: another session holds the lock a load
// takes. The start's load gives up within its lock timeout, the router goes on, and the
// retry loads the list once the lock is let go.
func (s *PriceSyncSuite) TestAHeldPricesLockDoesNotKeepTheRouterFromServing() {
	pgStore := s.started()
	holder, err := store.Open(s.settings.Postgres.DSN)
	s.Require().NoError(err)
	defer holder.Close()
	held, err := holder.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	_, err = held.ExecContext(s.ctx, "SELECT pg_advisory_xact_lock(hashtextextended('router.prices', 0))")
	s.Require().NoError(err)
	load, err := loadPriceFiles(s.settings)
	s.Require().NoError(err)
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	began := time.Now()
	startPriceSync(ctx, pgStore, load, 200*time.Millisecond, s.logger)

	s.Less(time.Since(began), 10*time.Second, "the start's load gave up on the lock")
	s.Empty(s.changes(pgStore))
	s.Require().NoError(held.Rollback())
	s.Eventually(func() bool { return len(s.changes(pgStore)) == 1 }, 15*time.Second, 100*time.Millisecond,
		"the retry loads the list once the lock is let go")
}

func (s *PriceSyncSuite) TestAFailedSyncIsRetried() {
	upstream, err := url.Parse(s.settings.Postgres.DSN)
	s.Require().NoError(err)
	forwarder := newForwarder(s.T(), upstream.Host)
	through := *upstream
	through.Host = forwarder.address()
	s.settings.Postgres.DSN = through.String()
	pgStore := s.started()
	forwarder.close()
	load, err := loadPriceFiles(s.settings)
	s.Require().NoError(err)
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	startPriceSync(ctx, pgStore, load, 100*time.Millisecond, s.logger)
	s.Empty(s.directChanges(upstream.String()), "the database could not be reached")
	forwarder.open()

	s.Eventually(func() bool { return len(s.directChanges(upstream.String())) == 1 }, 5*time.Second, 50*time.Millisecond,
		"the list is seeded within a retry of the database coming back")
}

// directChanges reads the price changes without the forwarder in between.
func (s *PriceSyncSuite) directChanges(dsn string) []store.PriceChange {
	direct, err := store.Open(dsn)
	s.Require().NoError(err)
	defer direct.Close()
	return s.changes(direct)
}

func (s *PriceSyncSuite) TestDriftReadsEveryShard() {
	shared := s.settings.Postgres.DSN
	here, there := s.emptyDatabase("pricedrift_here"), s.emptyDatabase("pricedrift_there")
	shards := map[string]string{"here": here, "there": there}
	for _, shard := range []string{here, there} {
		router, err := store.OpenShards(store.Options{DSN: shared, ShardDSN: shard, Shards: shards})
		s.Require().NoError(err)
		s.Require().NoError(router.Migrate(s.ctx))
		s.T().Cleanup(func() { router.Close() })
		// One model 5% under what xAI said it charged on every shard, the other within a
		// rounding difference.
		s.recordWithVendorCost(router, "grok-4.7", 950, 1_000)
		s.recordWithVendorCost(router, "grok-4.7", 1_900, 2_000)
		s.recordWithVendorCost(router, "grok-4.7-mini", 1_001, 1_000)
	}
	s.settings.Postgres = config.Postgres{DSN: shared, ShardDSN: here, Shards: shards}
	var out bytes.Buffer

	s.Require().NoError(runPricesDrift(s.ctx, []string{"--json"}, s.settings, &out))

	var models []driftedModel
	s.Require().NoError(json.Unmarshal(out.Bytes(), &models))
	s.Require().Len(models, 2)
	s.Equal("grok-4.7", models[0].Model)
	s.Equal("drift", models[0].Status)
	s.Equal(4, models[0].Rows, "two rows on each shard")
	s.InDelta(-0.05, models[0].MedianGap, 1e-9)
	s.Len(models[0].Examples, 4)
	s.EqualValues(1_000, models[0].Examples[0].VendorCostMicros)
	s.Equal("grok-4.7-mini", models[1].Model)
	s.Equal("ok", models[1].Status)
	s.Equal(2, models[1].Rows)
}

// TestDriftNamesARowByItsUsageIDWhereItHasOne: a database whose requests carry a usage id names
// each example row by it as well as by its id.
func (s *PriceSyncSuite) TestDriftNamesARowByItsUsageIDWhereItHasOne() {
	pgStore := s.started()
	_, err := pgStore.DB().ExecContext(s.ctx, "ALTER TABLE requests ADD COLUMN usage_id TEXT")
	s.Require().NoError(err)
	s.recordWithVendorCost(pgStore, "grok-4.7", 900, 1_000)
	_, err = pgStore.DB().ExecContext(s.ctx, "UPDATE requests SET usage_id = 'usage-1'")
	s.Require().NoError(err)
	var out bytes.Buffer

	s.Require().NoError(runPricesDrift(s.ctx, nil, s.settings, &out))

	s.Contains(out.String(), "drift  llm/grok/grok-4.7  1 rows, median gap -10.00%")
	s.Contains(out.String(), "(usage usage-1)")
}

// recordWithVendorCost records an LLM row the list priced at cost, of work the vendor said
// cost vendorCost.
func (s *PriceSyncSuite) recordWithVendorCost(router *store.Store, model string, cost, vendorCost int64) {
	s.Require().NoError(router.RecordRequest(s.ctx, &store.Request{
		Modality: "llm", CustomerID: "app", Provider: "grok", Model: model, StartedAt: time.Now(),
		CostMicros: cost, Billable: true, Success: true,
		Attributes: map[string]string{store.VendorCostAttribute: strconv.FormatInt(vendorCost, 10)},
	}))
}

// forwarder is a TCP proxy to a database that can be shut, which is a database a router
// cannot reach, and opened again.
type forwarder struct {
	t        *testing.T
	listener net.Listener
	upstream string
	mu       sync.Mutex
	shut     bool
	conns    []net.Conn
}

func newForwarder(t *testing.T, upstream string) *forwarder {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	f := &forwarder{t: t, listener: listener, upstream: upstream}
	t.Cleanup(func() {
		listener.Close()
		f.close()
	})
	go f.serve()
	return f
}

func (f *forwarder) address() string { return f.listener.Addr().String() }

func (f *forwarder) serve() {
	for {
		conn, err := f.listener.Accept()
		if err != nil {
			return
		}
		f.mu.Lock()
		if f.shut {
			f.mu.Unlock()
			conn.Close()
			continue
		}
		upstream, err := net.Dial("tcp", f.upstream)
		if err != nil {
			f.mu.Unlock()
			conn.Close()
			continue
		}
		f.conns = append(f.conns, conn, upstream)
		f.mu.Unlock()
		go func() { io.Copy(upstream, conn); upstream.Close() }() //nolint:errcheck // a closed side ends the copy
		go func() { io.Copy(conn, upstream); conn.Close() }()     //nolint:errcheck // a closed side ends the copy
	}
}

// close refuses new connections and cuts the ones open.
func (f *forwarder) close() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.shut = true
	for _, conn := range f.conns {
		conn.Close()
	}
	f.conns = nil
}

func (f *forwarder) open() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.shut = false
}
