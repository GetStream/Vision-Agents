//go:build integration

package store

import (
	"context"
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/pressly/goose/v3"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/migrations"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// 20261202120000_price_entry_region.sql.
const priceRegions int64 = 20261202120000

// PriceRegionMigrationSuite stops a database of its own where every deployed one is, the price
// list a router before regions wrote in it, and migrates it as a router starting does.
type PriceRegionMigrationSuite struct {
	suite.Suite
	dsn string
	ctx context.Context
	db  *Store
}

func TestPriceRegionMigrationSuite(t *testing.T) {
	suite.Run(t, new(PriceRegionMigrationSuite))
}

func (s *PriceRegionMigrationSuite) SetupSuite() {
	dsn := os.Getenv(DSNEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", DSNEnvVar)
	}
	s.dsn = testenv.Database(dsn, "priceregion")
	s.ctx = context.Background()
}

func (s *PriceRegionMigrationSuite) SetupTest() {
	db, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.NoError(db.Close()) })
	var name string
	s.Require().NoError(db.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&name))
	s.Require().True(strings.HasSuffix(name, "_test"), "refusing to drop the schema of %s, which is not a test database", name)
	s.Require().Contains(name, "priceregion")
	s.db = db
	s.exec("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	goose.SetBaseFS(migrations.FS)
	s.Require().NoError(goose.SetDialect("postgres"))
	s.Require().NoError(goose.UpToContext(s.ctx, db.DB().DB, ".", dropPriceMarkup))
}

func (s *PriceRegionMigrationSuite) exec(statements ...string) {
	for _, statement := range statements {
		_, err := s.db.DB().ExecContext(s.ctx, statement)
		s.Require().NoError(err, statement)
	}
}

func (s *PriceRegionMigrationSuite) count(query string) int {
	var n int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, query).Scan(&n), query)
	return n
}

// oldList is a list as a router before regions wrote it: a seed with a plain entry, a tiered
// one, a free one and two whose SKUs are numbered, at prices other than the factor would give,
// and a request it billed under the plain entry's SKU.
func (s *PriceRegionMigrationSuite) oldList() {
	s.exec(
		`INSERT INTO price_changes (name, kind, reason, source, effective_at, created_by) VALUES
		 ('20261101000000_seed', 'seed', 'The first price list', 'files', now() - interval '10 days', 'router v1')`,
		`INSERT INTO price_entries (change, sku, modality, provider, model, free, rates, tiers, prices, price_tiers, effective_at)
		 SELECT change, sku, modality, provider, model, free, rates::jsonb, tiers::jsonb, prices::jsonb, price_tiers::jsonb, now()
		 FROM (VALUES
		 ('20261101000000_seed', 'llm-openai-a', 'llm', 'openai', 'a', false,
		  '{"input_tokens": {"usd": 1, "per": 1000000}, "output_tokens": {"usd": 4, "per": 1000000}}', '[]',
		  '{"input_tokens": {"usd": 1.3, "per": 1000000}, "output_tokens": {"usd": 5.2, "per": 1000000}}', '[]'),
		 ('20261101000000_seed', 'llm-openai-long', 'llm', 'openai', 'long', false,
		  '{"input_tokens": {"usd": 1, "per": 1000000}}',
		  '[{"above_input_tokens": 272000, "rates": {"input_tokens": {"usd": 2, "per": 1000000}}}]',
		  '{"input_tokens": {"usd": 1.25, "per": 1000000}}',
		  '[{"above_input_tokens": 272000, "rates": {"input_tokens": {"usd": 2.5, "per": 1000000}}}]'),
		 ('20261101000000_seed', 'phone-telnyx-call', 'phone', 'telnyx', 'call', true, '{}', '[]', '{}', '[]'),
		 ('20261101000000_seed', 'llm-openai-gpt-4-1', 'llm', 'openai', 'gpt-4.1', false,
		  '{"input_tokens": {"usd": 2, "per": 1000000}}', '[]', '{"input_tokens": {"usd": 2.4, "per": 1000000}}', '[]'),
		 ('20261101000000_seed', 'llm-openai-gpt-4-1-2', 'llm', 'openai', 'gpt-4-1', false,
		  '{"input_tokens": {"usd": 2, "per": 1000000}}', '[]', '{"input_tokens": {"usd": 2.4, "per": 1000000}}', '[]')
		 ) AS e (change, sku, modality, provider, model, free, rates, tiers, prices, price_tiers)`,
		`INSERT INTO requests (modality, customer_id, provider, model, started_at, cost_micros, success, sku)
		 VALUES ('llm', 'app', 'openai', 'a', now(), 1300, true, 'llm-openai-a')`,
	)
}

// oldFiles is what the files of the router that wrote oldList say, read as a router after
// regions reads them: each entry in the us, and every llm one with its BYOK twin.
func oldFiles() []pricing.Entry {
	rate := func(usd float64) map[pricing.Meter]pricing.Rate {
		return map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(usd)}
	}
	us := func(entry pricing.Entry) pricing.Entry {
		entry.Region = pricing.RegionUS
		return entry
	}
	return pricing.PricedAll(pricing.WithBYOK([]pricing.Entry{
		us(pricing.Entry{Modality: "llm", Provider: "openai", Model: "a", Rates: map[pricing.Meter]pricing.Rate{
			pricing.InputTokens: perMillionTokens(1), pricing.OutputTokens: perMillionTokens(4)}}),
		us(pricing.Entry{Modality: "llm", Provider: "openai", Model: "long", Rates: rate(1),
			Tiers: []pricing.Tier{{AboveInputTokens: 272_000, Rates: rate(2)}}}),
		us(pricing.Entry{Modality: "phone", Provider: "telnyx", Model: "call", Free: true}),
		us(pricing.Entry{Modality: "llm", Provider: "openai", Model: "gpt-4.1", Rates: rate(2)}),
		us(pricing.Entry{Modality: "llm", Provider: "openai", Model: "gpt-4-1", Rates: rate(2)}),
	}))
}

// migrated is each entry's model, region, byok and SKU after the migration, by model.
type migrated struct {
	Region string
	BYOK   bool
	SKU    string
	Prices string
}

func (s *PriceRegionMigrationSuite) rows() map[string]migrated {
	rows, err := s.db.DB().QueryContext(s.ctx, "SELECT model, region, byok, sku, prices::text FROM price_list_entries WHERE NOT byok")
	s.Require().NoError(err)
	defer rows.Close()
	found := map[string]migrated{}
	for rows.Next() {
		var model string
		var row migrated
		s.Require().NoError(rows.Scan(&model, &row.Region, &row.BYOK, &row.SKU, &row.Prices))
		found[model] = row
	}
	s.Require().NoError(rows.Err())
	return found
}

// TestTheTableIsRenamedSoARouterBeforeRegionsCannotReadIt: a router built before this one reads
// price_entries by name, and finds nothing there to read.
func (s *PriceRegionMigrationSuite) TestTheTableIsRenamedSoARouterBeforeRegionsCannotReadIt() {
	s.oldList()

	s.Require().NoError(s.db.Migrate(s.ctx))

	_, err := s.db.DB().ExecContext(s.ctx, "SELECT modality, provider, model, sku FROM price_entries")
	s.ErrorContains(err, `relation "price_entries" does not exist`)
	s.Equal(5, s.count("SELECT count(*) FROM price_list_entries"))
}

// TestEveryEntryIsAUSEntryWhoseSKUEndsInItsRegion: the numbered SKU too, and no price moves.
// A request keeps the SKU it was billed under.
func (s *PriceRegionMigrationSuite) TestEveryEntryIsAUSEntryWhoseSKUEndsInItsRegion() {
	s.oldList()
	before := map[string]string{}
	rows, err := s.db.DB().QueryContext(s.ctx, "SELECT model, prices::text FROM price_entries")
	s.Require().NoError(err)
	for rows.Next() {
		var model, price string
		s.Require().NoError(rows.Scan(&model, &price))
		before[model] = price
	}
	s.Require().NoError(rows.Close())

	s.Require().NoError(s.db.Migrate(s.ctx))

	after := s.rows()
	s.Len(after, 5)
	for model, row := range after {
		s.Equal(pricing.RegionUS, row.Region, model)
		s.False(row.BYOK, model)
		s.Equal(before[model], row.Prices, model)
	}
	s.Equal("llm-openai-a-us", after["a"].SKU)
	s.Equal("llm-openai-long-us", after["long"].SKU)
	s.Equal("phone-telnyx-call-us", after["call"].SKU)
	s.Equal("llm-openai-gpt-4-1-us", after["gpt-4.1"].SKU)
	s.Equal("llm-openai-gpt-4-1-2-us", after["gpt-4-1"].SKU)
	var billed string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, "SELECT sku FROM requests").Scan(&billed))
	s.Equal("llm-openai-a", billed, "a request keeps the SKU it was billed under")
}

func (s *PriceRegionMigrationSuite) TestTheGuardStillRefusesEveryUpdateAndDelete() {
	s.oldList()
	s.Require().NoError(s.db.Migrate(s.ctx))

	for _, statement := range []string{
		"UPDATE price_list_entries SET sku = 'renamed' WHERE model = 'a'",
		"UPDATE price_list_entries SET region = 'eu' WHERE model = 'a'",
		"UPDATE price_list_entries SET byok = true WHERE model = 'a'",
		`UPDATE price_list_entries SET prices = '{}' WHERE model = 'a'`,
		"DELETE FROM price_list_entries WHERE model = 'a'",
	} {
		_, err := s.db.DB().ExecContext(s.ctx, statement)
		s.ErrorContains(err, "price_entries rows are never changed or deleted", statement)
	}
}

// TestAnInsertNamingNoRegionIsRefused: the default that filled the rows already written is
// gone, so a router built before this one could not write an entry here even by name.
func (s *PriceRegionMigrationSuite) TestAnInsertNamingNoRegionIsRefused() {
	s.oldList()
	s.Require().NoError(s.db.Migrate(s.ctx))

	_, missing := s.db.DB().ExecContext(s.ctx, `INSERT INTO price_list_entries (change, sku, modality, provider, model, free, effective_at)
VALUES ('20261101000000_seed', 'llm-openai-b-us', 'llm', 'openai', 'b', true, now())`)
	_, empty := s.db.DB().ExecContext(s.ctx, `INSERT INTO price_list_entries (change, sku, modality, provider, model, region, free, effective_at)
VALUES ('20261101000000_seed', 'llm-openai-b-us', 'llm', 'openai', 'b', '', true, now())`)

	s.ErrorContains(missing, `null value in column "region"`)
	s.ErrorContains(empty, "price_list_entries_region")
}

func (s *PriceRegionMigrationSuite) TestTheItemIndexLeadsWithTheRegionAndBYOKBeforeTheTime() {
	s.Require().NoError(s.db.Migrate(s.ctx))

	var definition string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		"SELECT indexdef FROM pg_indexes WHERE indexname = 'price_list_entries_item_idx'").Scan(&definition))
	s.Contains(definition, "(modality, provider, model, region, byok, effective_at DESC)")
	s.Zero(s.count("SELECT count(*) FROM pg_indexes WHERE indexname = 'price_entries_item_idx'"))
}

// TestTheNextStartAddsOnlyTheBYOKTwins: the files a router after regions loads name every item
// the list has, in the us, and the BYOK twin of each llm one, which the list lacks. The start
// writes one release change of those twins at a fifth of their rates, each under its own SKU,
// and every us item keeps its price and its SKU.
func (s *PriceRegionMigrationSuite) TestTheNextStartAddsOnlyTheBYOKTwins() {
	s.oldList()
	s.Require().NoError(s.db.Migrate(s.ctx))
	before := s.rows()

	change, added, wrote, err := s.db.SyncPrices(s.ctx, PriceLoad{Entries: oldFiles(), Source: "vendor.json", Release: "v2"})

	s.Require().NoError(err)
	s.True(wrote)
	s.Equal(PriceChangeRelease, change.Kind)
	s.Equal([]string{
		"llm/openai/a (us, byok) llm-openai-a-us-byok",
		"llm/openai/long (us, byok) llm-openai-long-us-byok",
		"llm/openai/gpt-4.1 (us, byok) llm-openai-gpt-4-1-us-byok",
		"llm/openai/gpt-4-1 (us, byok) llm-openai-gpt-4-1-us-byok-2",
	}, names(added))
	a := added[0]
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(0.2), pricing.OutputTokens: perMillionTokens(0.8)}, a.Prices)
	s.Equal([]pricing.Tier{{AboveInputTokens: 272_000, Rates: map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillionTokens(0.4)}}},
		added[1].PriceTiers)
	s.Equal(before, s.rows(), "every us item keeps its price and its SKU")

	rows, err := s.db.ActivePriceEntries(s.ctx, time.Now())
	s.Require().NoError(err)
	entries := make([]pricing.Entry, 0, len(rows))
	for _, row := range rows {
		entries = append(entries, row.Entry())
	}
	list, err := pricing.NewList("database", entries, nil)
	s.Require().NoError(err)
	used := pricing.Quantities{pricing.InputTokens: 1_000_000}
	us := list.Quote(llmItem("a", nil), used)
	byok := list.Quote(pricing.Item{Modality: "llm", Provider: "openai", Model: "a", Region: pricing.RegionUS, BYOK: true}, used)
	s.EqualValues(1_300_000, us.CostMicros, "the price the old router wrote, not the factor's")
	s.Equal("llm-openai-a-us", us.SKU)
	s.EqualValues(200_000, byok.CostMicros)
	s.Zero(byok.BaseCostMicros)

	_, again, wroteAgain, err := s.db.SyncPrices(s.ctx, PriceLoad{Entries: oldFiles(), Source: "vendor.json", Release: "v2"})
	s.Require().NoError(err)
	s.False(wroteAgain)
	s.Empty(again)
}

// TestADatabaseSeededFromVendorJSONBeforeRegionsGetsOnlyItsBYOKTwins is every deployed database:
// seeded from vendor.json by a router before regions, each item under the SKU that router gave
// it. The first start after the migration adds every BYOK twin vendor.json now has and nothing
// else, and no us item's price or SKU moves.
func (s *PriceRegionMigrationSuite) TestADatabaseSeededFromVendorJSONBeforeRegionsGetsOnlyItsBYOKTwins() {
	s.exec(`INSERT INTO price_changes (name, kind, reason, source, effective_at, created_by) VALUES
		 ('20261101000000_seed', 'seed', 'The first price list', 'vendor.json', now() - interval '10 days', 'router v1')`)
	taken := map[string]bool{}
	oldSKU := map[string]string{}
	for _, entry := range prices.Vendor().Entries {
		if entry.BYOK {
			continue
		}
		base := pricing.BaseSKU(entry.Modality, entry.Provider, entry.Model, "", false, entry.When)
		sku := base
		for n := 2; taken[sku]; n++ {
			sku = pricing.NumberedSKU(base, n)
		}
		taken[sku] = true
		oldSKU[prices.Name(entry)] = sku
		encode := func(value any) string {
			encoded, err := json.Marshal(value)
			s.Require().NoError(err)
			return string(encoded)
		}
		when := entry.When
		if when == nil {
			when = map[string]string{}
		}
		_, err := s.db.DB().ExecContext(s.ctx, `INSERT INTO price_entries
(change, sku, modality, provider, model, attributes, free, rates, tiers, prices, price_tiers, effective_at)
VALUES ('20261101000000_seed', ?, ?, ?, ?, ?::jsonb, ?, ?::jsonb, ?::jsonb, ?::jsonb, ?::jsonb, now())`,
			sku, entry.Modality, entry.Provider, entry.Model, encode(when), entry.Free, encode(orEmpty(entry.Rates)),
			encode(orNone(entry.Tiers)), encode(orEmpty(entry.Prices)), encode(orNone(entry.PriceTiers)))
		s.Require().NoError(err, prices.Name(entry))
	}
	s.Require().NoError(s.db.Migrate(s.ctx))
	inForce := func() []PriceEntry {
		rows, err := s.db.ActivePriceEntries(s.ctx, time.Now())
		s.Require().NoError(err)
		return rows
	}
	before := inForce()

	change, added, _, err := s.db.SyncPrices(s.ctx, PriceLoad{Entries: prices.Vendor().Entries, Source: prices.VendorFile, Release: "v2"})

	s.Require().NoError(err)
	s.Equal(PriceChangeRelease, change.Kind)
	var twins []string
	for _, entry := range prices.Vendor().Entries {
		if entry.BYOK {
			twins = append(twins, prices.Name(entry))
		}
	}
	s.Len(added, len(twins))
	skus := map[string]bool{}
	for _, entry := range added {
		s.True(entry.BYOK, prices.Name(entry))
		s.Contains(twins, prices.Name(entry))
		s.Equal(pricing.Priced(pricing.Entry{Modality: entry.Modality, Provider: entry.Provider, Model: entry.Model,
			Region: entry.Region, BYOK: true, When: entry.When, Free: entry.Free, Rates: entry.Rates, Tiers: entry.Tiers}).Prices,
			entry.Prices, "%s is a fifth of its rates", prices.Name(entry))
		base := pricing.BaseSKU(entry.Modality, entry.Provider, entry.Model, pricing.RegionUS, true, entry.When)
		s.True(entry.SKU == base || strings.HasPrefix(entry.SKU, base+"-"), "%s is %s", prices.Name(entry), entry.SKU)
		s.False(skus[entry.SKU], "%s is given once", entry.SKU)
		skus[entry.SKU] = true
	}
	after := inForce()
	s.Len(after, len(before)+len(added))
	for _, row := range before {
		s.Equal(oldSKU[prices.Name(row.Entry())]+"-us", row.SKU, "%s/%s", row.Provider, row.Model)
		i := slices.IndexFunc(after, func(other PriceEntry) bool { return other.ID == row.ID })
		s.Require().NotEqual(-1, i, "%s/%s is still in force", row.Provider, row.Model)
		s.Equal(row.Prices, after[i].Prices)
		s.Equal(row.SKU, after[i].SKU)
	}
}

func orEmpty(rates map[pricing.Meter]pricing.Rate) map[pricing.Meter]pricing.Rate {
	if rates == nil {
		return map[pricing.Meter]pricing.Rate{}
	}
	return rates
}

func orNone(tiers []pricing.Tier) []pricing.Tier {
	if tiers == nil {
		return []pricing.Tier{}
	}
	return tiers
}

func (s *PriceRegionMigrationSuite) TestTheMigrationComesOffAndAppliesAgain() {
	s.oldList()
	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", priceRegions))
	s.Require().NoError(goose.DownToContext(s.ctx, s.db.DB().DB, ".", dropPriceMarkup))

	var sku string
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, "SELECT sku FROM price_entries WHERE model = 'gpt-4-1'").Scan(&sku))
	s.Equal("llm-openai-gpt-4-1-2", sku, "the region comes off the SKU")
	s.Zero(s.count(`SELECT count(*) FROM information_schema.columns
WHERE table_name = 'price_entries' AND column_name IN ('region', 'byok')`))
	s.Equal(1, s.count("SELECT count(*) FROM pg_indexes WHERE indexname = 'price_entries_item_idx'"))

	s.Require().NoError(goose.UpToContext(s.ctx, s.db.DB().DB, ".", priceRegions))
	s.Equal("llm-openai-gpt-4-1-2-us", s.rows()["gpt-4-1"].SKU)
}

// TestDownRefusesAListWithABYOKEntry: a router before regions would take the BYOK entry for
// its item's newest price and bill every row of it at the fee.
func (s *PriceRegionMigrationSuite) TestDownRefusesAListWithABYOKEntry() {
	s.oldList()
	s.Require().NoError(s.db.Migrate(s.ctx))
	_, _, _, err := s.db.SyncPrices(s.ctx, PriceLoad{Entries: oldFiles(), Source: "vendor.json", Release: "v2"})
	s.Require().NoError(err)

	err = goose.DownToContext(s.ctx, s.db.DB().DB, ".", dropPriceMarkup)

	s.ErrorContains(err, "the price list holds byok or non-US entries")
	s.Equal(9, s.count("SELECT count(*) FROM price_list_entries"), "nothing came off")
}
