//go:build integration

package store

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/pressly/goose/v3"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/migrations"
)

// Around 20261118120000_price_entry_prices.sql.
const (
	beforeStoredPrices int64 = 20261117110000
	storedPrices       int64 = 20261118120000
)

// StoredPricesMigrationSuite stops a database of its own where every deployed one is, with the
// list a deployed router wrote under a list markup, and migrates it as a router starting does.
type StoredPricesMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestStoredPricesMigrationSuite(t *testing.T) {
	suite.Run(t, new(StoredPricesMigrationSuite))
}

func (s *StoredPricesMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "pricemig")
	s.ctx = context.Background()
}

func (s *StoredPricesMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "pricemig")
	s.exec(db, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", beforeStoredPrices))
	s.db = db
}

func (s *StoredPricesMigrationSuite) exec(db *Store, statements ...string) {
	for _, statement := range statements {
		_, err := db.DB().ExecContext(s.ctx, statement)
		s.Require().NoError(err, statement)
	}
}

// oldList is a list as routers before stored prices wrote it: a seed at a list markup of 20%,
// with an entry of its own markup, a tiered one, a free one and a service tier; a later change
// of one item's rates; an item withdrawn; two models whose SKUs would be one; and a change
// scheduled for tomorrow that sets a list markup of 30%, beside a cancelled one of 90%.
func (s *StoredPricesMigrationSuite) oldList() {
	s.exec(s.db,
		`INSERT INTO price_changes (name, kind, reason, source, effective_at, markup_bps, created_by) VALUES
		 ('20261101000000_seed', 'seed', 'The first price list', 'files', now() - interval '10 days', 2000, 'router v1'),
		 ('20261102000000_cheaper_a', 'api', 'a costs less', 'vendor', now() - interval '5 days', NULL, 'staff'),
		 ('20261103000000_withdraw_gone', 'api', 'gone is gone', 'vendor', now() - interval '4 days', NULL, 'staff'),
		 ('20261120000000_dearer', 'api', 'dearer tomorrow', 'operator', now() + interval '1 day', 3000, 'staff'),
		 ('20261120000001_cancelled', 'api', 'never', 'operator', now() + interval '2 days', 9000, 'staff')`,
		`INSERT INTO price_entries (change, modality, provider, model, attributes, free, rates, tiers, markup_bps, withdrawn, effective_at)
		 SELECT change, modality, provider, model, attributes::jsonb, free, rates::jsonb, tiers::jsonb, markup_bps::int,
		        change = '20261103000000_withdraw_gone', effective_at
		 FROM (VALUES
		 ('20261101000000_seed', 'llm', 'openai', 'a', '{}', false,
		  '{"input_tokens": {"usd": 1, "per": 1000000}, "output_tokens": {"usd": 4, "per": 1000000}}', '[]', NULL, now()),
		 ('20261101000000_seed', 'llm', 'openai', 'a', '{"service_tier": "flex"}', false,
		  '{"input_tokens": {"usd": 0.5, "per": 1000000}}', '[]', NULL, now()),
		 ('20261101000000_seed', 'llm', 'openai', 'own', '{}', false,
		  '{"input_tokens": {"usd": 2, "per": 1000000}}', '[]', 500, now()),
		 ('20261101000000_seed', 'llm', 'openai', 'long', '{}', false,
		  '{"input_tokens": {"usd": 1.1, "per": 1000000}}',
		  '[{"above_input_tokens": 272000, "rates": {"input_tokens": {"usd": 2.2, "per": 1000000}}}]', NULL, now()),
		 ('20261101000000_seed', 'phone', 'telnyx', 'call', '{}', true, '{}', '[]', NULL, now()),
		 ('20261101000000_seed', 'llm', 'openai', 'gone', '{}', false, '{"input_tokens": {"usd": 1, "per": 1000000}}', '[]', NULL, now()),
		 ('20261101000000_seed', 'llm', 'openai', 'gpt-4.1', '{}', false, '{"input_tokens": {"usd": 2, "per": 1000000}}', '[]', NULL, now()),
		 ('20261101000000_seed', 'llm', 'openai', 'gpt-4-1', '{}', false, '{"input_tokens": {"usd": 2, "per": 1000000}}', '[]', NULL, now()),
		 ('20261102000000_cheaper_a', 'llm', 'openai', 'a', '{}', false,
		  '{"input_tokens": {"usd": 0.07, "per": 1000000}, "output_tokens": {"usd": 4, "per": 1000000}}', '[]', NULL, now()),
		 ('20261103000000_withdraw_gone', 'llm', 'openai', 'gone', '{}', false, '{}', '[]', NULL, now()),
		 ('20261120000000_dearer', 'llm', 'openai', 'b', '{}', false, '{"input_tokens": {"usd": 1, "per": 1000000}}', '[]', NULL, now()),
		 ('20261120000001_cancelled', 'llm', 'openai', 'c', '{}', false, '{"input_tokens": {"usd": 1, "per": 1000000}}', '[]', NULL, now())
		 ) AS e (change, modality, provider, model, attributes, free, rates, tiers, markup_bps, effective_at)`,
		`UPDATE price_changes SET cancelled_at = now(), cancelled_by = 'staff', cancel_reason = 'no' WHERE name = '20261120000001_cancelled'`,
	)
}

// migratedRow is what the migration left on one entry.
type migratedRow struct {
	SKU        string
	Prices     map[pricing.Meter]pricing.Rate
	PriceTiers []pricing.Tier
}

// row is what the migration left on the entry of change for model and attributes.
func (s *StoredPricesMigrationSuite) row(change, model, attributes string) migratedRow {
	var sku string
	var prices, tiers []byte
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		`SELECT sku, prices, price_tiers FROM price_entries WHERE change = ? AND model = ? AND attributes = ?::jsonb`,
		change, model, attributes).Scan(&sku, &prices, &tiers))
	row := migratedRow{SKU: sku}
	s.Require().NoError(json.Unmarshal(prices, &row.Prices))
	s.Require().NoError(json.Unmarshal(tiers, &row.PriceTiers))
	return row
}

func (s *StoredPricesMigrationSuite) TestEveryEntryIsPricedAtWhatItWasCharged() {
	s.oldList()

	s.Require().NoError(s.db.Migrate(s.ctx))

	a := s.row("20261102000000_cheaper_a", "a", "{}")
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(0.084), pricing.OutputTokens: perMillionTokens(4.8)}, a.Prices,
		"the list's 20%, to twelve digits")
	s.Equal(a.SKU, s.row("20261101000000_seed", "a", "{}").SKU, "every row of an item shares its SKU")
	s.Equal("llm-openai-a", a.SKU)
	flex := s.row("20261101000000_seed", "a", `{"service_tier": "flex"}`)
	s.Equal("llm-openai-a-service-tier-flex", flex.SKU)
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(0.6)}, flex.Prices)
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(2.1)}, s.row("20261101000000_seed", "own", "{}").Prices,
		"an entry's own 5% replaces the list's")
	long := s.row("20261101000000_seed", "long", "{}")
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(1.32)}, long.Prices)
	s.Equal([]pricing.Tier{{AboveInputTokens: 272_000, Rates: map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(2.64)}}},
		long.PriceTiers, "a tier keeps its threshold")
	free := s.row("20261101000000_seed", "call", "{}")
	s.Empty(free.Prices, "a free entry has no price")
	s.Empty(free.PriceTiers)
	s.Equal("phone-telnyx-call", free.SKU)
	s.Equal(s.row("20261101000000_seed", "gone", "{}").SKU, s.row("20261103000000_withdraw_gone", "gone", "{}").SKU)
	s.Equal("llm-openai-gpt-4-1", s.row("20261101000000_seed", "gpt-4.1", "{}").SKU, "the item written first keeps the base")
	s.Equal("llm-openai-gpt-4-1-2", s.row("20261101000000_seed", "gpt-4-1", "{}").SKU)
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(1.3)}, s.row("20261120000000_dearer", "b", "{}").Prices,
		"a scheduled entry is priced at the markup in force when it takes effect")
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(1.3)}, s.row("20261120000001_cancelled", "c", "{}").Prices,
		"a cancelled change's own 90% never applied: the 30% before it would have")
	var empty int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM price_entries WHERE sku = ''").Scan(&empty))
	s.Zero(empty)
}

// TestTheListChargesWhatItDidBefore: quoted from the stored prices, every item in force costs
// what the markup on its rates came to, and its base cost is still its rates.
func (s *StoredPricesMigrationSuite) TestTheListChargesWhatItDidBefore() {
	s.oldList()
	s.Require().NoError(s.db.Migrate(s.ctx))

	rows, err := s.db.ActivePriceEntries(s.ctx, time.Now())
	s.Require().NoError(err)
	entries := make([]pricing.Entry, 0, len(rows))
	for _, row := range rows {
		entries = append(entries, row.Entry())
	}
	list, err := pricing.NewList("database", entries, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	quote := func(model string, used pricing.Quantities) pricing.Quote {
		return list.Quote(pricing.Item{Modality: "llm", Provider: "openai", Model: model}, used)
	}

	a := quote("a", pricing.Quantities{pricing.InputTokens: 1_000_000, pricing.OutputTokens: 1_000_000})
	s.EqualValues(4_884_000, a.CostMicros, "(0.07 + 4) × 1.2")
	s.EqualValues(4_070_000, a.BaseCostMicros)
	s.Equal("llm-openai-a", a.SKU)
	s.EqualValues(2_100_000, quote("own", pricing.Quantities{pricing.InputTokens: 1_000_000}).CostMicros)
	long := quote("long", pricing.Quantities{pricing.InputTokens: 300_000})
	s.EqualValues(792_000, long.CostMicros, "300,000 at the tier's 2.2 × 1.2")
	s.EqualValues(660_000, long.BaseCostMicros)
	s.False(list.Billable(pricing.Item{Modality: "phone", Provider: "telnyx", Model: "call"}))
	_, gone := list.Entry(pricing.Item{Modality: "llm", Provider: "openai", Model: "gone"})
	s.False(gone)
}

func (s *StoredPricesMigrationSuite) TestTheGuardStillRefusesEveryUpdate() {
	s.oldList()
	s.Require().NoError(s.db.Migrate(s.ctx))

	for _, statement := range []string{
		`UPDATE price_entries SET prices = '{}' WHERE model = 'a'`,
		`UPDATE price_entries SET price_tiers = '[{"above_input_tokens": 1, "rates": {}}]' WHERE model = 'long'`,
		`UPDATE price_entries SET sku = 'renamed' WHERE model = 'a'`,
		`UPDATE price_entries SET rates = '{}' WHERE model = 'a'`,
		`DELETE FROM price_entries WHERE model = 'a'`,
	} {
		_, err := s.db.DB().ExecContext(s.ctx, statement)
		s.ErrorContains(err, "price_entries rows are never changed or deleted", statement)
	}
	var functions int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM pg_proc WHERE proname = 'price_entries_marked'").Scan(&functions))
	s.Zero(functions, "the backfill's helper is gone")
}

// TestAnEmptyListMigratesAndSeedsAtTheFactor: a database no router has seeded yet, as every
// shard's is, migrates with nothing to backfill, and its first load prices at the factor.
func (s *StoredPricesMigrationSuite) TestAnEmptyListMigratesAndSeedsAtTheFactor() {
	s.Require().NoError(s.db.Migrate(s.ctx))

	_, _, _, err := s.db.SyncPrices(s.ctx, PriceLoad{Entries: []pricing.Entry{{Modality: "llm", Provider: "openai", Model: "a",
		Rates: map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(1)}}}, Source: "files", Release: "v2"})
	s.Require().NoError(err)

	rows, err := s.db.ActivePriceEntries(s.ctx, time.Now())
	s.Require().NoError(err)
	s.Require().Len(rows, 1)
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(1.2)}, rows[0].Prices)
	s.Equal("llm-openai-a", rows[0].SKU)
	var sku string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		"SELECT column_default FROM information_schema.columns WHERE table_name = 'requests' AND column_name = 'sku'").Scan(&sku))
	s.Equal("''::text", sku, "a router built before it writes requests without the column")
}

func (s *StoredPricesMigrationSuite) TestTheMigrationComesOffAndAppliesAgain() {
	s.oldList()
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", storedPrices))
	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", beforeStoredPrices))

	var columns int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `SELECT count(*) FROM information_schema.columns
WHERE table_name IN ('price_entries', 'requests') AND column_name IN ('sku', 'prices', 'price_tiers')`).Scan(&columns))
	s.Zero(columns)
	var markup int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		"SELECT markup_bps FROM price_changes WHERE name = '20261101000000_seed'").Scan(&markup))
	s.Equal(2000, markup, "the markup a router built before it reads is still there")

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", storedPrices))
	s.Equal("llm-openai-a", s.row("20261101000000_seed", "a", "{}").SKU)
}
