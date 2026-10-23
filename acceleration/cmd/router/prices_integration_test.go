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
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
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
	s.startPrices(s.ctx, pgStore, time.Hour)
}

// startPrices starts the price list as a router's start does, retrying a failed load every
// retry, and reading the list again every hour, so only a retry's load reads it sooner.
func (s *PriceSyncSuite) startPrices(ctx context.Context, pgStore *store.Store, retry time.Duration) *pricing.Active {
	load, err := loadPriceFiles(s.settings)
	s.Require().NoError(err)
	fallback, err := newPrices(routing.Config{}, load, pricesVersion(s.settings), s.logger)
	s.Require().NoError(err)
	return startPrices(ctx, pgStore, routing.Config{}, fallback, load, true, retry, time.Hour, s.logger)
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

// changePrice writes a change through the management API's store call: the rate a meter of
// entry charges, taking effect at effectiveAt, or now when that is nil.
func (s *PriceSyncSuite) changePrice(pgStore *store.Store, name string, entry pricing.Entry, usd float64, markup *int64, effectiveAt *time.Time) store.PriceChange {
	entry.Rates = map[pricing.Meter]pricing.Rate{pricing.InputTokens: {USD: usd, Per: 1_000_000}}
	entry.Tiers, entry.Version = nil, ""
	change, _, _, err := pgStore.CreatePriceChange(s.ctx, store.PriceChangeWrite{
		Name: name, Reason: "A test's change", Source: "the test", EffectiveAt: effectiveAt, MarkupBps: markup,
		Entries: []store.PriceEntryWrite{{Entry: entry}}, ActorID: "tester", ActorName: "tester",
	}, func(store.PriceListAt) error { return nil })
	s.Require().NoError(err)
	return change
}

// tokenPriced is the first entry of vendor.json with an input token rate.
func (s *PriceSyncSuite) tokenPriced() pricing.Entry {
	for _, entry := range prices.Vendor().Entries {
		if entry.Rates[pricing.InputTokens].USD > 0 {
			return entry
		}
	}
	s.FailNow("vendor.json prices no input tokens")
	return pricing.Entry{}
}

// rateOf is the input token rate the list in force charges for an entry's item and When.
func (s *PriceSyncSuite) rateOf(pgStore *store.Store, entry pricing.Entry) float64 {
	active, err := pgStore.ActivePriceEntries(s.ctx, time.Now())
	s.Require().NoError(err)
	for _, row := range active {
		if prices.Name(row.Entry()) == prices.Name(entry) {
			return row.Rates[pricing.InputTokens].USD
		}
	}
	s.FailNow("the list does not price " + prices.Name(entry))
	return 0
}

// TestReloadPutsTheFilesPricesBack: rates changed in the database, a markup and a change
// scheduled for later all give way to what the files say, as one seed, and the rows priced
// before keep their price_version.
func (s *PriceSyncSuite) TestReloadPutsTheFilesPricesBack() {
	s.ownFile(`{"entries": [` + ownEntry + `]}`)
	pgStore := s.started()
	s.startRouter(pgStore)
	vendor := s.tokenPriced()
	own := pricing.Entry{Modality: "llm", Provider: "own", Model: "own-model"}
	fileRate := s.rateOf(pgStore, vendor)
	markup := int64(1_500)
	fixed := s.changePrice(pgStore, "vendor_rate_fix", vendor, fileRate*3, &markup, nil)
	s.changePrice(pgStore, "own_rate_fix", own, 5, nil, nil)
	later := time.Now().Add(time.Hour)
	s.changePrice(pgStore, "own_later", own, 7, nil, &later)
	s.Require().InDelta(fileRate*3, s.rateOf(pgStore, vendor), 1e-12)
	s.Require().InDelta(5.0, s.rateOf(pgStore, own), 1e-12)
	s.Require().NoError(pgStore.RecordRequest(s.ctx, &store.Request{
		Modality: "llm", CustomerID: "app", Provider: vendor.Provider, Model: vendor.Model, StartedAt: time.Now(),
		CostMicros: 10, Billable: true, Success: true, PriceVersion: fixed.Name,
	}))
	var out bytes.Buffer

	s.Require().NoError(runPricesReload(s.ctx, []string{"--development"}, s.settings, &out))

	s.InDelta(fileRate, s.rateOf(pgStore, vendor), 1e-12, "vendor.json's rate")
	s.InDelta(1.0, s.rateOf(pgStore, own), 1e-12, "the price file's rate")
	changes := s.changes(pgStore)
	s.Require().Len(changes, 1)
	seed := changes[0]
	s.Equal(store.PriceChangeSeed, seed.Kind)
	s.Equal(s.settings.PricesFile+", vendor.json, router dev", seed.Source)
	upcoming, err := pgStore.UpcomingPriceEntries(s.ctx, time.Now())
	s.Require().NoError(err)
	s.Empty(upcoming, "the scheduled change went with the rest")
	markupNow, err := pgStore.ActiveMarkup(s.ctx, time.Now())
	s.Require().NoError(err)
	s.Zero(markupNow, "the price file sets no markup")
	active, err := pgStore.ActivePriceEntries(s.ctx, time.Now())
	s.Require().NoError(err)
	s.Len(active, len(prices.Vendor().Entries)+1)

	var audits []store.AuditEntry
	s.Require().NoError(pgStore.DB().NewSelect().Model(&audits).Where("resource_type = ?", store.AuditResourcePriceList).
		Order("created_at", "id").Scan(s.ctx))
	s.Require().Len(audits, 6, "the first seed, three changes, the reload's removal and its seed")
	removal, seedAudit := audits[4], audits[5]
	s.Equal(store.AuditDeleted, removal.Action, "the removal is recorded first")
	s.Regexp(`^\d{14}_reload$`, removal.ResourceID)
	s.NotEqual(seed.Name, removal.ResourceID)
	s.Len(removal.Changes[0].Before, 4)
	s.Equal(store.AuditCreated, seedAudit.Action)
	s.Equal(seed.Name, seedAudit.ResourceID)
	s.Equal(store.AuditSourceCLI, seedAudit.Source)

	var priceVersion string
	s.Require().NoError(pgStore.ShardDB().QueryRowContext(s.ctx, "SELECT price_version FROM requests").Scan(&priceVersion))
	s.Equal(fixed.Name, priceVersion, "a row keeps what priced it")

	text := out.String()
	s.Regexp(`^database "\w+_pricesync_\w*test" at \S+:\d+ \(tcp\), which answers as [0-9a-f.:]+:\d+\n`, text,
		"where it is pointed comes first")
	s.NotContains(text, "postgres:postgres")
	s.Contains(text, "removed 4 price changes (1 not yet in force, 0 cancelled)")
	s.Contains(text, "the markup in force was 1500 basis points")
	s.Contains(text, "seeded the price list from "+s.settings.PricesFile+", vendor.json: change "+seed.Name)
	s.Contains(text, "2 items priced differently by the files:\n")
	s.Contains(text, "  "+prices.Name(vendor)+"\n")
	s.Contains(text, "  "+prices.Name(own)+"\n")
	s.Contains(text, "within 30s")
}

func (s *PriceSyncSuite) TestReloadDryRunWritesNothing() {
	pgStore := s.started()
	s.startRouter(pgStore)
	vendor := s.tokenPriced()
	s.changePrice(pgStore, "vendor_rate_fix", vendor, 42, nil, nil)
	var out bytes.Buffer

	s.Require().NoError(runPricesReload(s.ctx, []string{"--development", "--dry-run"}, s.settings, &out))

	s.Len(s.changes(pgStore), 2)
	s.InDelta(42.0, s.rateOf(pgStore, vendor), 1e-12)
	s.Regexp(`^database "\w+_pricesync_\w*test" at \S+:\d+ \(tcp\)`, out.String())
	s.Contains(out.String(), "would remove 2 price changes")
	s.Contains(out.String(), "would seed the price list from vendor.json")
	s.Contains(out.String(), "1 items priced differently by the files:\n  "+prices.Name(vendor)+"\n")
}

// TestReloadOnAFreshDatabaseSeedsIt: nothing to remove, the same seed a start writes.
func (s *PriceSyncSuite) TestReloadOnAFreshDatabaseSeedsIt() {
	pgStore := s.started()
	var out bytes.Buffer

	s.Require().NoError(runPricesReload(s.ctx, []string{"--development"}, s.settings, &out))

	changes := s.changes(pgStore)
	s.Require().Len(changes, 1)
	s.Equal(store.PriceChangeSeed, changes[0].Kind)
	s.Contains(out.String(), "removed 0 price changes")
	var audits []store.AuditEntry
	s.Require().NoError(pgStore.DB().NewSelect().Model(&audits).Where("resource_type = ?", store.AuditResourcePriceList).Scan(s.ctx))
	s.Len(audits, 1, "only the seed's: nothing was removed")
}

// TestReloadRefusesANonDevelopmentDatabase: without --development, under staging's ROUTER_ENV
// or against a database elsewhere, the list stays as it is.
func (s *PriceSyncSuite) TestReloadRefusesANonDevelopmentDatabase() {
	pgStore := s.started()
	s.startRouter(pgStore)
	seeded := s.changes(pgStore)

	s.ErrorContains(runPricesReload(s.ctx, nil, s.settings, &bytes.Buffer{}), "pass --development")
	remote := s.settings
	remote.Postgres.DSN = "postgres://postgres:postgres@db.example.com:5432/model_router?sslmode=require"
	s.ErrorContains(runPricesReload(s.ctx, []string{"--development"}, remote, &bytes.Buffer{}), `names the host "db.example.com"`)
	s.T().Setenv(config.EnvVar, config.Staging)
	s.ErrorContains(runPricesReload(s.ctx, []string{"--development"}, s.settings, &bytes.Buffer{}), `ROUTER_ENV is "staging"`)

	s.Equal(seeded, s.changes(pgStore))
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
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	began := time.Now()
	active := s.startPrices(ctx, pgStore, 200*time.Millisecond)

	s.Less(time.Since(began), 10*time.Second, "the start's load gave up on the lock")
	s.Empty(s.changes(pgStore))
	s.Error(active.Ready(), "the list is empty until the load")
	s.Require().NoError(held.Rollback())
	s.Eventually(func() bool { return len(s.changes(pgStore)) == 1 }, 15*time.Second, 100*time.Millisecond,
		"the retry loads the list once the lock is let go")
}

// TestARouterWhoseFirstLoadLostTheLockGoesReadyRightAfterTheRetry: the start's load gives up on
// a held lock and the list reads empty. Once the retry seeds it, the list is read again at once,
// not an interval (here an hour) later.
func (s *PriceSyncSuite) TestARouterWhoseFirstLoadLostTheLockGoesReadyRightAfterTheRetry() {
	pgStore := s.started()
	holder, err := store.Open(s.settings.Postgres.DSN)
	s.Require().NoError(err)
	defer holder.Close()
	held, err := holder.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	_, err = held.ExecContext(s.ctx, "SELECT pg_advisory_xact_lock(hashtextextended('router.prices', 0))")
	s.Require().NoError(err)
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	active := s.startPrices(ctx, pgStore, 200*time.Millisecond)
	s.Require().Error(active.Ready())
	s.Require().NoError(held.Rollback())

	s.Require().Eventually(func() bool { return len(s.changes(pgStore)) == 1 }, 15*time.Second, 20*time.Millisecond)
	seeded := time.Now()
	s.Eventually(func() bool { return active.Ready() == nil }, 2*time.Second, 10*time.Millisecond,
		"ready right after the seed")
	s.Less(time.Since(seeded), 2*time.Second)
	s.NotEqual(prices.EmbeddedVersion("dev"), active.Quote(pricing.Item{Modality: prices.Vendor().Entries[0].Modality,
		Provider: prices.Vendor().Entries[0].Provider, Model: prices.Vendor().Entries[0].Model,
		Attributes: prices.Vendor().Entries[0].When}, pricing.Quantities{pricing.InputTokens: 1}).Version,
		"priced from the database, not the files")
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
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	s.startPrices(ctx, pgStore, 100*time.Millisecond)
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

// TestDriftNamesARowByItsUsageIDWhereItHasOne: requests carry a usage id, so each example row
// is named by it as well as by its id.
func (s *PriceSyncSuite) TestDriftNamesARowByItsUsageIDWhereItHasOne() {
	pgStore := s.started()
	s.recordWithVendorCost(pgStore, "grok-4.7", 900, 1_000)
	var usageID string
	s.Require().NoError(pgStore.ShardDB().QueryRowContext(s.ctx, "SELECT usage_id::text FROM requests").Scan(&usageID))
	var out bytes.Buffer

	s.Require().NoError(runPricesDrift(s.ctx, nil, s.settings, &out))

	s.Contains(out.String(), "drift  llm/grok/grok-4.7  1 rows, median gap -10.00%")
	s.Contains(out.String(), "(usage "+usageID+")")
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
