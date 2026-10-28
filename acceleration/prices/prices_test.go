package prices

import (
	"bytes"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

type PricesSuite struct {
	suite.Suite
}

func TestPricesSuite(t *testing.T) {
	suite.Run(t, new(PricesSuite))
}

func perMillion(usd float64) pricing.Rate { return pricing.Rate{USD: usd, Per: 1_000_000} }

func llm(model string, input float64) pricing.Entry {
	return pricing.Entry{Modality: "llm", Provider: "openai", Model: model, Region: pricing.RegionUS,
		Rates: map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillion(input)}}
}

// named is the file's entry of that Name.
func (s *PricesSuite) named(file File, name string) pricing.Entry {
	for _, entry := range file.Entries {
		if Name(entry) == name {
			return entry
		}
	}
	s.FailNow("no entry named " + name)
	return pricing.Entry{}
}

func (s *PricesSuite) TestTheVendorFileIsAValidList() {
	_, err := pricing.NewList("vendor", Vendor().Entries, slog.New(slog.DiscardHandler))

	s.Require().NoError(err)
	s.NotEmpty(Vendor().Entries)
}

// TestAnEntryWithoutAPriceIsPricedAtTheFactor: a file sets the vendor's rates, and an entry
// it gives no price is priced at them times pricing.PriceFactorBps; one it gives a price
// keeps it, and a free one has none.
func (s *PricesSuite) TestAnEntryWithoutAPriceIsPricedAtTheFactor() {
	file, err := Parse("own.json", []byte(`{"entries": [
		{"modality": "llm", "provider": "openai", "model": "plain", "region": "us", "rates": {"input_tokens": {"usd": 2, "per": 1000000}},
			"tiers": [{"above_input_tokens": 1000, "rates": {"input_tokens": {"usd": 4, "per": 1000000}}}]},
		{"modality": "llm", "provider": "openai", "model": "own", "region": "us", "sku": "own-model",
			"rates": {"input_tokens": {"usd": 2, "per": 1000000}}, "prices": {"input_tokens": {"usd": 2.1, "per": 1000000}}},
		{"modality": "llm", "provider": "openai", "model": "free", "region": "us", "free": true}]}`))
	s.Require().NoError(err)

	plain := s.named(file, "llm/openai/plain (us)")
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillion(2.4)}, plain.Prices)
	s.Equal([]pricing.Tier{{AboveInputTokens: 1_000, Rates: map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillion(4.8)}}},
		plain.PriceTiers)
	own := s.named(file, "llm/openai/own (us)")
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillion(2.1)}, own.Prices)
	s.Equal("own-model", own.SKU)
	s.Empty(s.named(file, "llm/openai/free (us)").Prices)
}

// TestEveryEntryOfABYOKModalityGetsATwinAtTheFee: reading a file adds a BYOK twin of each llm,
// knowledge and sandbox entry, priced at a fifth of its rates whatever price the entry it
// copies has, with no SKU of its own.
func (s *PricesSuite) TestEveryEntryOfABYOKModalityGetsATwinAtTheFee() {
	file, err := Parse("own.json", []byte(`{"entries": [
		{"modality": "llm", "provider": "openai", "model": "own", "region": "us", "sku": "own-model",
			"rates": {"input_tokens": {"usd": 1, "per": 1000000}}, "prices": {"input_tokens": {"usd": 2.1, "per": 1000000}}},
		{"modality": "stt", "provider": "deepgram", "model": "flux", "region": "us", "rates": {"audio_ms": {"usd": 0.3, "per": 3600000}}}]}`))
	s.Require().NoError(err)

	s.Require().Len(file.Entries, 3)
	twin := s.named(file, "llm/openai/own (us, byok)")
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillion(0.2)}, twin.Prices)
	s.Empty(twin.SKU)
}

func (s *PricesSuite) TestAFileEntryWithoutARegionIsRefused() {
	_, err := Parse("own.json", []byte(`{"entries": [{"modality": "llm", "provider": "openai", "model": "m",
		"rates": {"input_tokens": {"usd": 1, "per": 1000000}}}]}`))

	s.ErrorContains(err, "own.json: llm/openai/m names no region: every entry names its region, such as us")
}

// TestADeploymentFileEntryWithoutARegionIsReadInTheUS: only a deployment's own file may leave
// it out, and each entry that does is logged.
func (s *PricesSuite) TestADeploymentFileEntryWithoutARegionIsReadInTheUS() {
	path := filepath.Join(s.T().TempDir(), "own.json")
	s.Require().NoError(os.WriteFile(path, []byte(`{"entries": [
		{"modality": "llm", "provider": "openai", "model": "m", "rates": {"input_tokens": {"usd": 1, "per": 1000000}}},
		{"modality": "stt", "provider": "own", "model": "m", "region": "eu", "rates": {"audio_ms": {"usd": 1, "per": 3600000}}}]}`), 0o600))
	var logs bytes.Buffer

	file, err := LoadDeployment(path, slog.New(slog.NewTextHandler(&logs, nil)))

	s.Require().NoError(err)
	s.Equal(pricing.RegionUS, s.named(file, "llm/openai/m (us)").Region)
	s.Equal(pricing.RegionUS, s.named(file, "llm/openai/m (us, byok)").Region)
	s.Equal(pricing.RegionEU, s.named(file, "stt/own/m (eu)").Region)
	s.Contains(logs.String(), "llm/openai/m")
	s.NotContains(logs.String(), "stt/own/m")
}

func (s *PricesSuite) TestADeploymentFileEntryOfAnotherFormOfRegionIsRefused() {
	path := filepath.Join(s.T().TempDir(), "own.json")
	s.Require().NoError(os.WriteFile(path, []byte(`{"entries": [{"modality": "llm", "provider": "openai", "model": "m",
		"region": "EU", "rates": {"input_tokens": {"usd": 1, "per": 1000000}}}]}`), 0o600))

	_, err := LoadDeployment(path, slog.New(slog.DiscardHandler))

	s.ErrorIs(err, pricing.ErrRegion)
}

func (s *PricesSuite) TestAFileEntryOfAnotherFormOfRegionIsRefused() {
	_, err := Parse("own.json", []byte(`{"entries": [{"modality": "llm", "provider": "openai", "model": "m", "region": "EU",
		"rates": {"input_tokens": {"usd": 1, "per": 1000000}}}]}`))

	s.ErrorIs(err, pricing.ErrRegion)
	s.ErrorContains(err, "own.json")
}

// TestAFilesOwnBYOKEntryWins: a BYOK entry a file writes is kept as written, and laid over
// vendor.json it wins over the twin vendor.json's entry has.
func (s *PricesSuite) TestAFilesOwnBYOKEntryWins() {
	i := slices.IndexFunc(Vendor().Entries, func(entry pricing.Entry) bool {
		return entry.Modality == "llm" && len(entry.When) == 0 && len(entry.Tiers) == 0
	})
	s.Require().NotEqual(-1, i)
	vendor := Vendor().Entries[i]
	own := pricing.Entry{Modality: vendor.Modality, Provider: vendor.Provider, Model: vendor.Model, Region: pricing.RegionUS, BYOK: true,
		Rates: vendor.Rates, Prices: map[pricing.Meter]pricing.Rate{}}
	for meter, rate := range vendor.Rates {
		rate.USD = 0.000001
		own.Prices[meter] = rate
	}
	encoded, err := json.Marshal(File{Entries: []pricing.Entry{own}})
	s.Require().NoError(err)
	file, err := Parse("own.json", encoded)
	s.Require().NoError(err)

	s.Equal([]pricing.Entry{own}, file.Entries, "no twin is made for an entry that is BYOK itself")
	var merged []pricing.Entry
	for _, entry := range Merge(file, Vendor()) {
		if Name(entry) == Name(own) {
			merged = append(merged, entry)
		}
	}
	s.Equal([]pricing.Entry{own}, merged)
}

func (s *PricesSuite) TestAPriceWithOtherMetersThanItsRatesIsRefused() {
	_, err := Parse("own.json", []byte(`{"entries": [{"modality": "llm", "provider": "openai", "model": "m", "region": "us",
		"rates": {"input_tokens": {"usd": 1, "per": 1000000}, "output_tokens": {"usd": 2, "per": 1000000}},
		"prices": {"input_tokens": {"usd": 1.2, "per": 1000000}}}]}`))

	s.ErrorIs(err, pricing.ErrPriceMeters)
	s.ErrorContains(err, "own.json")
}

func (s *PricesSuite) TestTwoEntriesWithOneSKUAreRefused() {
	_, err := Parse("own.json", []byte(`{"entries": [
		{"modality": "llm", "provider": "openai", "model": "a", "region": "us", "sku": "one", "free": true},
		{"modality": "llm", "provider": "openai", "model": "b", "region": "us", "sku": "one", "free": true}]}`))

	s.ErrorContains(err, "both have the SKU one")
}

// TestOnlyALinuxContainerSandboxIsPriced: Daytona prices other classes, such as a Windows
// vCPU, higher, so a sandbox of any other class is unpriced rather than billed at the
// container's rates.
func (s *PricesSuite) TestOnlyALinuxContainerSandboxIsPriced() {
	list, err := pricing.NewList("vendor", Vendor().Entries, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	used := pricing.Quantities{pricing.VCPUSeconds: 3600, pricing.MemoryGBSeconds: 3600, pricing.DiskGBSeconds: 3 * 3600}

	for _, model := range []string{"running", "stopped"} {
		container := list.Quote(pricing.Item{Modality: "sandbox", Provider: "daytona", Model: model, Region: pricing.RegionUS,
			Attributes: map[string]string{"class": "container"}}, used)
		s.False(container.Unpriced, model)
		s.Positive(container.CostMicros, model)
		for _, class := range []string{"linux-vm", "windows", "android", ""} {
			other := list.Quote(pricing.Item{Modality: "sandbox", Provider: "daytona", Model: model, Region: pricing.RegionUS,
				Attributes: map[string]string{"class": class}}, used)
			s.True(other.Unpriced, "%s %s", model, class)
		}
	}
}

// TestTheVendorFileHoldsOnlyTheVendorsRates: vendor.json is public, so no entry in it sets a
// price, which is the operator's: each is priced at the factor. Nor does it write a BYOK entry:
// those are made when it is read.
func (s *PricesSuite) TestTheVendorFileHoldsOnlyTheVendorsRates() {
	var raw struct {
		MarkupBps *int64 `json:"markup_bps"`
		Entries   []struct {
			Model      string          `json:"model"`
			BYOK       json.RawMessage `json:"byok"`
			Prices     json.RawMessage `json:"prices"`
			PriceTiers json.RawMessage `json:"price_tiers"`
		} `json:"entries"`
	}
	s.Require().NoError(json.Unmarshal(vendorJSON, &raw))

	s.Nil(raw.MarkupBps)
	for _, entry := range raw.Entries {
		s.Nil(entry.Prices, entry.Model)
		s.Nil(entry.PriceTiers, entry.Model)
		s.Nil(entry.BYOK, entry.Model)
	}
}

// TestEveryVendorEntryIsInTheUS: vendor.json is the list prices of the us, and no entry
// leaves its region out.
func (s *PricesSuite) TestEveryVendorEntryIsInTheUS() {
	var raw struct {
		Entries []struct {
			Model  string  `json:"model"`
			Region *string `json:"region"`
		} `json:"entries"`
	}
	s.Require().NoError(json.Unmarshal(vendorJSON, &raw))

	s.NotEmpty(raw.Entries)
	for _, entry := range raw.Entries {
		s.Require().NotNil(entry.Region, entry.Model)
		s.Equal(pricing.RegionUS, *entry.Region, entry.Model)
	}
	for _, entry := range Vendor().Entries {
		s.Equal(pricing.RegionUS, entry.Region, Name(entry))
	}
}

// TestEveryLLMKnowledgeAndSandboxVendorEntryHasABYOKTwin: each, read, is followed by its BYOK
// twin at a fifth of its rates; no other modality has one.
func (s *PricesSuite) TestEveryLLMKnowledgeAndSandboxVendorEntryHasABYOKTwin() {
	byName := map[string]pricing.Entry{}
	for _, entry := range Vendor().Entries {
		byName[Name(entry)] = entry
	}
	var twins int
	for _, entry := range Vendor().Entries {
		if entry.BYOK {
			twins++
			continue
		}
		twin := entry
		twin.BYOK = true
		found, ok := byName[Name(twin)]
		if !slices.Contains(pricing.BYOKModalities, entry.Modality) {
			s.False(ok, Name(twin))
			continue
		}
		s.Require().True(ok, Name(twin))
		s.Equal(entry.Rates, found.Rates, Name(twin))
		s.Equal(entry.Tiers, found.Tiers, Name(twin))
		s.Equal(entry.Free, found.Free, Name(twin))
		s.Equal(pricing.Priced(pricing.Entry{Modality: entry.Modality, Provider: entry.Provider, Model: entry.Model,
			Region: entry.Region, BYOK: true, When: entry.When, Free: entry.Free, Rates: entry.Rates, Tiers: entry.Tiers}), found, Name(twin))
	}
	s.Equal(78, twins, "73 llm, 3 knowledge and 2 sandbox entries")
}

func (s *PricesSuite) TestTheFirstFileWinsForAnItem() {
	own := llm("m", 3)
	flex := llm("m", 1)
	flex.When = map[string]string{"service_tier": "flex"}

	merged := Merge(File{Entries: []pricing.Entry{own}}, File{Entries: []pricing.Entry{llm("m", 2), flex}})

	s.Equal([]pricing.Entry{own, flex}, merged, "the first file's default entry, and the second's flex entry beside it")
}

// TestMergeKeepsUSEUAndBYOKEntriesApart: an item's entries in another region, or BYOK, are other
// items: a file's US entry replaces neither.
func (s *PricesSuite) TestMergeKeepsUSEUAndBYOKEntriesApart() {
	us := llm("m", 3)
	eu := llm("m", 4)
	eu.Region = pricing.RegionEU
	byok := llm("m", 2)
	byok.BYOK = true
	vendorUS := llm("m", 1)

	merged := Merge(File{Entries: []pricing.Entry{us}}, File{Entries: []pricing.Entry{vendorUS, eu, byok}})

	s.Equal([]pricing.Entry{us, eu, byok}, merged)
}

func (s *PricesSuite) TestANameCarriesTheRegionAndBYOK() {
	byok := pricing.Entry{Modality: "llm", Provider: "anthropic", Model: "claude-haiku-5-5", Region: pricing.RegionUS, BYOK: true,
		When: map[string]string{"service_tier": "flex"}}

	s.Equal("llm/anthropic/claude-haiku-5-5 (us)", Name(pricing.Entry{Modality: "llm", Provider: "anthropic",
		Model: "claude-haiku-5-5", Region: pricing.RegionUS}))
	s.Equal("llm/anthropic/claude-haiku-5-5 (us, byok) when service_tier=flex", Name(byok))
}

func (s *PricesSuite) TestAFileWithAMisspeltRateIsRefused() {
	_, err := Parse("own.json", []byte(`{"entries": [{"modality": "llm", "provider": "openai", "model": "m",
		"rates": {"input_tokens": {"usd": 1, "per": 1000000}}, "rate": {}}]}`))

	s.ErrorContains(err, `own.json: json: unknown field "rate"`)
}

func (s *PricesSuite) TestAFileThatDoesNotBuildAListNamesTheEntry() {
	_, err := Parse("own.json", []byte(`{"entries": [{"modality": "llm", "provider": "openai", "model": "m", "region": "us",
		"rates": {"input_tokens": {"usd": 1, "per": 0}}}]}`))

	s.ErrorContains(err, "own.json: pricing: llm/openai/m (us) rates input_tokens per 0 units")
}

func (s *PricesSuite) TestAFileMayNotMarkAModelLocal() {
	_, err := Parse("own.json", []byte(`{"entries": [{"modality": "llm", "provider": "gemma", "model": "m", "region": "us", "local": true}]}`))

	s.ErrorContains(err, "own.json: llm/gemma/m (us) sets local")
}

func (s *PricesSuite) TestAFileMayNotNameAVersion() {
	_, err := Parse("own.json", []byte(`{"entries": [{"modality": "llm", "provider": "openai", "model": "m", "region": "us",
		"free": true, "version": "x"}]}`))

	s.ErrorContains(err, "own.json: llm/openai/m (us) sets a version")
}

// TestAFilesMarkupIsIgnoredWithAWarning: a deployment's file from before prices were stored
// still says markup_bps. It loads, its entries priced at the factor as any without a price,
// and the warning names it; a markup out of any range is no error either.
func (s *PricesSuite) TestAFilesMarkupIsIgnoredWithAWarning() {
	file, err := Parse("own.json", []byte(`{"markup_bps": 2000, "entries": [
		{"modality": "llm", "provider": "openai", "model": "m", "region": "us", "rates": {"input_tokens": {"usd": 1, "per": 1000000}}}]}`))
	s.Require().NoError(err)
	s.Equal(map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillion(1.2)}, file.Entries[0].Prices)

	var logs bytes.Buffer
	IgnoredMarkup(slog.New(slog.NewTextHandler(&logs, nil)), "own.json", file)
	s.Contains(logs.String(), "level=WARN")
	s.Contains(logs.String(), "markup_bps in a price file is ignored")
	s.Contains(logs.String(), "file=own.json markup_bps=2000 factor_bps=12000")

	_, err = Parse("own.json", []byte(`{"markup_bps": 100001, "entries": []}`))
	s.NoError(err)
	logs.Reset()
	IgnoredMarkup(slog.New(slog.NewTextHandler(&logs, nil)), "vendor.json", Vendor())
	s.Empty(logs.String())
}

func (s *PricesSuite) TestALocalItemIsMarkedOnEveryEntryOrGivenOne() {
	priced := llm("own-gpu", 1)
	flex := llm("own-gpu", 0.5)
	flex.When = map[string]string{"service_tier": "flex"}

	marked := WithLocal([]pricing.Entry{priced, flex}, []pricing.Item{
		{Modality: "llm", Provider: "openai", Model: "own-gpu", Region: pricing.RegionUS},
		{Modality: "stt", Provider: "self", Model: "unpriced", Region: pricing.RegionUS},
	})

	s.Require().Len(marked, 3)
	s.True(marked[0].Local)
	s.True(marked[1].Local)
	s.Equal(pricing.Entry{Modality: "stt", Provider: "self", Model: "unpriced", Region: pricing.RegionUS, Local: true}, marked[2])
	s.False(priced.Local, "the entries passed in are left as they were")
}

func (s *PricesSuite) TestTheEmbeddedListNamesItsRelease() {
	list, err := Embedded(nil, "v1.4.0", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	entry := Vendor().Entries[0]

	quote := list.Quote(pricing.Item{Modality: entry.Modality, Provider: entry.Provider, Model: entry.Model, Region: entry.Region,
		Attributes: entry.When}, pricing.Quantities{pricing.InputTokens: 1_000})

	s.Equal("embedded:v1.4.0", quote.Version)
}

// TestTheEmbeddedListPricesBYOKWorkAtTheFee: the binary's own list has the BYOK twins too, so a
// router that has not read its database yet prices BYOK work as the database would.
func (s *PricesSuite) TestTheEmbeddedListPricesBYOKWorkAtTheFee() {
	list, err := Embedded(nil, "v1.4.0", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	haiku := pricing.Item{Modality: "llm", Provider: "anthropic", Model: "claude-haiku-5-5", Region: pricing.RegionUS}
	byok := haiku
	byok.BYOK = true
	used := pricing.Quantities{pricing.InputTokens: 1_000_000}

	plain, own := list.Quote(haiku, used), list.Quote(byok, used)

	s.Equal(plain.BaseCostMicros*12/10, plain.CostMicros)
	s.Equal(plain.BaseCostMicros*2/10, own.CostMicros)
	s.Zero(own.BaseCostMicros)
	s.Equal("llm-anthropic-claude-haiku-5-5-us-byok", own.SKU)
}

// TestNoPriceTableIsTruncated: rows of the price list are never changed or deleted, which a
// trigger enforces, but a trigger on rows does not see TRUNCATE. So nothing the router runs
// may say it, but for one statement in store.ReloadPrices, which empties a development
// database's list (router prices reload, whose command refuses any other database). Test code
// resets its own databases and is not read.
func (s *PricesSuite) TestNoPriceTableIsTruncated() {
	for _, statement := range []string{"TRUNCATE price_entries", "truncate table only price_changes",
		"TRUNCATE requests, stats_hourly, price_entries CASCADE", "TRUNCATE public.price_entries",
		`TRUNCATE "public"."price_changes"`, `TRUNCATE "price_entries"`, "TRUNCATE audit_log, public.price_changes",
		"TRUNCATE /* x */ price_entries", "TRUNCATE/* x */price_entries", "TRUNCATE -- x\n  price_changes",
		"TRUNCATE TABLE ONLY public . price_entries *", "TRUNCATE requests *, price_changes", "TRUNCATE price_list_entries",
		"TRUNCATE public.price_list_entries, price_changes", `truncate "price_list_entries"`} {
		s.True(truncatesPriceTable.MatchString(statement), statement)
	}
	for _, text := range []string{"at := now.Truncate(time.Hour) // price_entries", "TRUNCATE requests",
		"TRUNCATE price_entries_archive", "-- price_entries are never truncated", "TRUNCATE price_list_entries_archive"} {
		s.False(truncatesPriceTable.MatchString(text), text)
	}
	placed := []byte(`package store
func (s *Store) ReloadPrices() { _ = func() string { return "TRUNCATE price_entries, price_changes" } }
const wipe = "TRUNCATE price_changes"
func other(db *bun.DB) { db.NewTruncateTable() }
`)
	allowed, refused := priceTableTruncates("store/prices.go", placed, "store")
	s.Equal(1, allowed, "the statement inside ReloadPrices")
	s.Len(refused, 2, "a constant beside ReloadPrices, and bun's truncate")
	allowed, refused = priceTableTruncates("elsewhere/prices.go", placed, "store")
	s.Zero(allowed)
	s.Len(refused, 3, "ReloadPrices outside internal/store is no exception")

	root, err := filepath.Abs("..")
	s.Require().NoError(err)
	storeDir := filepath.Join(root, "internal", "store")
	inReload := 0

	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if name := entry.Name(); name == "node_modules" || strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(path, "_test.go") || !(strings.HasSuffix(path, ".go") || strings.HasSuffix(path, ".sql")) {
			return nil
		}
		text, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		allowed, refused := priceTableTruncates(path, text, storeDir)
		inReload += allowed
		s.Emptyf(refused, "%s truncates a price table outside store.ReloadPrices", path)
		return nil
	})
	s.Require().NoError(err)
	s.Equal(1, inReload, "store.ReloadPrices truncates the price tables in one statement")
}

// truncatesPriceTable is a TRUNCATE of either price table, the entries under the name they have
// had since regions or the one before: in any case, with TABLE and ONLY or without, in a list,
// under a schema, quoted or not, and with comments between the words.
var truncatesPriceTable = func() *regexp.Regexp {
	gap := `(?:\s|/\*.*?\*/|--[^\n]*(?:\n|$))`
	name := `(?:"[^"]+"|\w+)`
	schema := `(?:` + name + gap + `*\.` + gap + `*)?`
	listed := schema + name + gap + `*\*?` + gap + `*,` + gap + `*`
	return regexp.MustCompile(`(?is)\bTRUNCATE` + gap + `+(?:TABLE` + gap + `+)?(?:ONLY` + gap + `+)?(?:` + listed + `)*` +
		schema + `"?price_(?:list_entries|entries|changes)\b`)
}()

// priceTableTruncates reads a .go or .sql file for truncates of a price table: how many sit in
// store.ReloadPrices, which may hold them when its file is in storeDir, and what every other
// one says. In Go only string literals are SQL, and any bun NewTruncateTable is refused, since
// which table it truncates is not in the source. A Go file that does not parse is read as SQL.
func priceTableTruncates(path string, text []byte, storeDir string) (allowed int, refused []string) {
	file, err := parser.ParseFile(token.NewFileSet(), path, text, parser.SkipObjectResolution)
	if !strings.HasSuffix(path, ".go") || err != nil {
		for _, found := range truncatesPriceTable.FindAll(text, -1) {
			refused = append(refused, string(found))
		}
		return 0, refused
	}
	for _, decl := range file.Decls {
		exempt := filepath.Dir(path) == storeDir && isReloadPrices(decl)
		ast.Inspect(decl, func(node ast.Node) bool {
			switch node := node.(type) {
			case *ast.BasicLit:
				value, err := strconv.Unquote(node.Value)
				if node.Kind != token.STRING || err != nil || !truncatesPriceTable.MatchString(value) {
					return true
				}
				if exempt {
					allowed++
				} else {
					refused = append(refused, value)
				}
			case *ast.SelectorExpr:
				if node.Sel.Name == "NewTruncateTable" {
					refused = append(refused, "NewTruncateTable")
				}
			}
			return true
		})
	}
	return allowed, refused
}

// isReloadPrices is whether a declaration is func (s *Store) ReloadPrices.
func isReloadPrices(decl ast.Decl) bool {
	fn, ok := decl.(*ast.FuncDecl)
	if !ok || fn.Name.Name != "ReloadPrices" || fn.Recv == nil || len(fn.Recv.List) != 1 {
		return false
	}
	star, ok := fn.Recv.List[0].Type.(*ast.StarExpr)
	if !ok {
		return false
	}
	receiver, ok := star.X.(*ast.Ident)
	return ok && receiver.Name == "Store"
}
