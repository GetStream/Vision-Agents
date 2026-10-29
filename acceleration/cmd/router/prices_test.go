package main

import (
	"bytes"
	"flag"
	"fmt"
	"log/slog"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// pricedConfig is a routing config whose price: blocks vendor.json is checked against. Empty
// checks the embedded router.yaml, for as long as it has them.
var pricedConfig = flag.String("priced-config", "", "a routing config with price: blocks to check vendor.json against")

// PriceFileSuite is the price commands that need no database.
type PriceFileSuite struct {
	suite.Suite
}

func TestPriceFileSuite(t *testing.T) {
	suite.Run(t, new(PriceFileSuite))
}

// everyMeter is each meter an entry may rate, so a usage drawn over them leaves none unpriced.
var everyMeter = []pricing.Meter{
	pricing.InputTokens, pricing.CachedInputTokens, pricing.CacheWriteInputTokens, pricing.OutputTokens,
	pricing.InputTextTokens, pricing.InputAudioTokens, pricing.InputImageTokens,
	pricing.CachedInputTextTokens, pricing.CachedInputAudioTokens, pricing.CachedInputImageTokens,
	pricing.OutputTextTokens, pricing.OutputAudioTokens,
	pricing.Characters, pricing.AudioMs, pricing.Images, pricing.Pixels, pricing.Requests, pricing.Messages,
	pricing.MessageSegments, pricing.CallSeconds, pricing.Seconds, pricing.SessionSeconds, pricing.VCPUSeconds,
	pricing.MemoryGBSeconds, pricing.DiskGBSeconds, pricing.BytesWritten, pricing.BytesQueried,
	pricing.BytesReturned, pricing.Results, pricing.ByteHours, pricing.Clones,
}

// goldenUsages are the usages each item is quoted on: the shapes rows take (a transcription,
// a cached prompt, a cache write, a speech-to-speech split, a long prompt past a tier,
// pictures, a search), then usages drawn over every meter from a fixed seed so a failure
// repeats.
func goldenUsages() []pricing.Quantities {
	usages := []pricing.Quantities{
		{},
		{pricing.AudioMs: 1_234},
		{pricing.InputTokens: 6_000, pricing.CachedInputTokens: 4_000, pricing.OutputTokens: 2_500},
		{pricing.InputTokens: 3, pricing.CacheWriteInputTokens: 2_386, pricing.OutputTokens: 22},
		{pricing.InputTokens: 3_162, pricing.OutputTokens: 114, pricing.InputTextTokens: 2_829, pricing.InputAudioTokens: 222,
			pricing.OutputAudioTokens: 54, pricing.OutputTextTokens: 60},
		{pricing.InputTokens: 300_000, pricing.CachedInputTokens: 20_000, pricing.OutputTokens: 4_000},
		{pricing.Characters: 777},
		{pricing.Images: 2, pricing.Pixels: 6_000_000},
		{pricing.Requests: 1},
		{pricing.CallSeconds: 61, pricing.Messages: 1, pricing.MessageSegments: 2},
	}
	random := rand.New(rand.NewPCG(1, 2))
	for range 200 {
		used := pricing.Quantities{}
		for _, meter := range everyMeter {
			used[meter] = random.Int64N(400_000)
		}
		usages = append(usages, used)
	}
	return usages
}

// TestTheVendorFilePricesWhatRouterYamlDid: the list built from vendor.json quotes every item
// router.yaml's price: blocks priced, at each of its service tiers, exactly as the list built
// from those blocks did, and holds no item they did not. Once router.yaml carries no prices,
// pass -priced-config with a copy that does (the PR that took them out says how).
func (s *PriceFileSuite) TestTheVendorFilePricesWhatRouterYamlDid() {
	routingConfig, err := routing.LoadConfig(*pricedConfig)
	s.Require().NoError(err)
	phoneConfig, err := phone.DefaultConfig()
	s.Require().NoError(err)
	declared := append(routing.PriceEntries(routingConfig), phone.PriceEntries(phoneConfig)...)
	if len(exportPrices(routingConfig, nil).Entries) == 0 {
		s.T().Skip("router.yaml carries no price: blocks: pass -priced-config=<a routing config with them> to check vendor.json against one")
	}
	discard := slog.New(slog.DiscardHandler)
	want, err := pricing.NewList("config", pricing.PricedAll(declared), discard)
	s.Require().NoError(err)
	got, err := prices.Embedded(routing.LocalItems(routingConfig), "test", discard)
	s.Require().NoError(err)
	usages := goldenUsages()

	var rounded []string
	for _, entry := range declared {
		item := pricing.Item{Modality: entry.Modality, Provider: entry.Provider, Model: entry.Model, Region: entry.Region, Attributes: entry.When}
		name := prices.Name(entry)
		s.Equal(want.Billable(item), got.Billable(item), name)
		wanted, _ := want.Entry(item)
		found, ok := got.Entry(item)
		identical := true
		if entry.Free || len(entry.Rates) > 0 || entry.Local {
			s.Require().True(ok, "%s is not in vendor.json", name)
			identical = s.requireSameEntry(wanted, found, name)
		}
		// A rate router.yaml multiplied into float noise (1.2 × 1.5 × 0.5 is 0.8999999999999999)
		// is written as the vendor publishes it, so a usage whose cost lands on half a micro
		// may round the other way. Every other entry quotes exactly.
		var slack int64
		if !identical {
			slack = 1
			rounded = append(rounded, name)
		}
		for _, used := range usages {
			a, b := want.Quote(item, used), got.Quote(item, used)
			if abs(a.CostMicros-b.CostMicros) > slack || abs(a.OutputCostMicros-b.OutputCostMicros) > slack ||
				a.Billable != b.Billable || a.Unpriced != b.Unpriced {
				s.Failf("a quote moved", "%s on %v: router.yaml %+v, vendor.json %+v", name, used, a, b)
				return
			}
		}
	}
	s.T().Logf("%d entries quote exactly; %d have a rate router.yaml left in float noise, within a micro: %v",
		len(declared)-len(rounded), len(rounded), rounded)
	for _, entry := range prices.Vendor().Entries {
		if entry.BYOK {
			continue
		}
		item := pricing.Item{Modality: entry.Modality, Provider: entry.Provider, Model: entry.Model, Region: entry.Region, Attributes: entry.When}
		wanted, ok := want.Entry(item)
		s.True(ok && prices.When(wanted.When) == prices.When(entry.When), "%s is in vendor.json and not in router.yaml", prices.Name(entry))
	}
}

// requireSameEntry fails unless two entries price alike: the same flags, and the same rates
// to twelve significant digits, which is what vendor.json keeps. It reports whether every
// rate is the same float.
func (s *PriceFileSuite) requireSameEntry(want, got pricing.Entry, name string) bool {
	s.Equal(want.Free, got.Free, name)
	s.Equal(want.Local, got.Local, name)
	identical := s.requireSameRates(want.Rates, got.Rates, name)
	s.Require().Len(got.Tiers, len(want.Tiers), name)
	for i := range want.Tiers {
		s.Equal(want.Tiers[i].AboveInputTokens, got.Tiers[i].AboveInputTokens, name)
		if !s.requireSameRates(want.Tiers[i].Rates, got.Tiers[i].Rates, fmt.Sprintf("%s tier %d", name, i)) {
			identical = false
		}
	}
	return identical
}

func (s *PriceFileSuite) requireSameRates(want, got map[pricing.Meter]pricing.Rate, name string) bool {
	s.Require().Len(got, len(want), name)
	identical := true
	for meter, rate := range want {
		other, ok := got[meter]
		s.Require().True(ok, "%s has no %s rate", name, meter)
		s.Equal(rate.Per, other.Per, name)
		s.Equal(rate.Increment, other.Increment, name)
		s.Equal(rate.Minimum, other.Minimum, name)
		s.LessOrEqual(math.Abs(rate.USD-other.USD), 1e-11*math.Abs(rate.USD), "%s %s: %v and %v", name, meter, rate.USD, other.USD)
		identical = identical && rate.USD == other.USD
	}
	return identical
}

func abs(micros int64) int64 { return max(micros, -micros) }

// TestTheVendorFileChecks: every entry is in the us, and the BYOK ones are the twins reading the
// file makes.
func (s *PriceFileSuite) TestTheVendorFileChecks() {
	var out bytes.Buffer
	var byok int
	for _, entry := range prices.Vendor().Entries {
		if entry.BYOK {
			byok++
		}
	}

	s.Require().NoError(runPricesCheck(nil, &out))

	s.Equal(fmt.Sprintf("vendor.json: %d entries, us %d (%d byok)\n", len(prices.Vendor().Entries), len(prices.Vendor().Entries), byok),
		out.String())
}

func (s *PriceFileSuite) TestCheckingABadFileNamesTheFileAndTheEntry() {
	path := filepath.Join(s.T().TempDir(), "own-prices.json")
	s.Require().NoError(os.WriteFile(path, []byte(`{"markup_bps": 1500, "entries": [
		{"modality": "llm", "provider": "openai", "model": "m", "region": "us", "rates": {"input_tokens": {"usd": -1, "per": 1000000}}}]}`), 0o600))

	err := runPricesCheck([]string{"--file", path}, &bytes.Buffer{})

	s.ErrorContains(err, path+": pricing: llm/openai/m (us) rates input_tokens below zero")
}

// TestCheckingAFileWithAnEntryInNoRegionReadsItInTheUS: chat's file names no region while a
// router built before regions may still read it, and the check reads it as a start does.
func (s *PriceFileSuite) TestCheckingAFileWithAnEntryInNoRegionReadsItInTheUS() {
	path := filepath.Join(s.T().TempDir(), "own-prices.json")
	s.Require().NoError(os.WriteFile(path, []byte(`{"entries": [
		{"modality": "runtime", "provider": "stream", "model": "voice", "free": true}]}`), 0o600))
	var out bytes.Buffer

	s.Require().NoError(runPricesCheck([]string{"--file", path}, &out))

	s.Equal(fmt.Sprintf("%s: 1 entries, us 1; %d entries laid over vendor.json\n", path, len(prices.Vendor().Entries)+1),
		out.String())
}

// TestADeploymentsOwnEntryInNoRegionIsLoadedInTheUS: the router starts, says so, and loads the
// entry as a US one; an item no region names would otherwise be unpriced.
func (s *PriceFileSuite) TestADeploymentsOwnEntryInNoRegionIsLoadedInTheUS() {
	settings := config.Defaults()
	settings.PricesFile = filepath.Join(s.T().TempDir(), "own-prices.json")
	s.Require().NoError(os.WriteFile(settings.PricesFile, []byte(`{"entries": [
		{"modality": "llm", "provider": "own", "model": "m", "rates": {"input_tokens": {"usd": 1, "per": 1000000}}}]}`), 0o600))
	var logs bytes.Buffer

	load, err := loadPriceFiles(settings, slog.New(slog.NewTextHandler(&logs, nil)))

	s.Require().NoError(err)
	s.Contains(logs.String(), "price file entries name no region and are read as us")
	s.Contains(logs.String(), "llm/own/m")
	var own []string
	for _, entry := range load.Entries {
		if entry.Provider == "own" {
			own = append(own, prices.Name(entry))
		}
	}
	s.ElementsMatch([]string{"llm/own/m (us)", "llm/own/m (us, byok)"}, own)
}

// TestCheckingAFileCountsItsEntriesByRegion: an eu entry and its BYOK twin beside a us one.
func (s *PriceFileSuite) TestCheckingAFileCountsItsEntriesByRegion() {
	path := filepath.Join(s.T().TempDir(), "own-prices.json")
	s.Require().NoError(os.WriteFile(path, []byte(`{"entries": [
		{"modality": "llm", "provider": "own", "model": "m", "region": "eu", "rates": {"input_tokens": {"usd": 1, "per": 1000000}}},
		{"modality": "stt", "provider": "own", "model": "m", "region": "us", "rates": {"audio_ms": {"usd": 1, "per": 3600000}}}]}`), 0o600))
	var out bytes.Buffer

	s.Require().NoError(runPricesCheck([]string{"--file", path}, &out))

	s.Equal(fmt.Sprintf("%s: 3 entries, eu 2 (1 byok), us 1; %d entries laid over vendor.json\n", path, len(prices.Vendor().Entries)+3),
		out.String())
}

// TestABadPricesFileStopsStartup: run returns what loadPriceFiles says, before it opens
// anything. Called here rather than through run, which tunes the process's HTTP transport
// that other tests are using.
func (s *PriceFileSuite) TestABadPricesFileStopsStartup() {
	settings := config.Defaults()
	settings.PricesFile = filepath.Join(s.T().TempDir(), "own-prices.json")
	s.Require().NoError(os.WriteFile(settings.PricesFile, []byte(`{"entries": [
		{"modality": "llm", "provider": "openai", "model": "m", "region": "us", "rates": {"input_tokens": {"usd": 1}}}]}`), 0o600))

	_, err := loadPriceFiles(settings, slog.New(slog.DiscardHandler))

	s.ErrorContains(err, settings.PricesFile+": pricing: llm/openai/m (us) rates input_tokens per 0 units")
}

// TestTheRouterPricesFromTheDeploymentsOwnFile: a model only the deployment's own file prices
// is priced, and the file's entry wins over vendor.json's for a model both have. An entry the
// file gives a price is charged it; one it gives none is charged its rates times the factor.
func (s *PriceFileSuite) TestTheRouterPricesFromTheDeploymentsOwnFile() {
	vendor := prices.Vendor().Entries[0]
	settings := config.Defaults()
	settings.PricesFile = filepath.Join(s.T().TempDir(), "own-prices.json")
	s.Require().NoError(os.WriteFile(settings.PricesFile, []byte(fmt.Sprintf(`{"markup_bps": 2000, "entries": [
		{"modality": "llm", "provider": "own", "model": "own-model", "region": "us", "rates": {"input_tokens": {"usd": 3, "per": 1000000}}},
		{"modality": %q, "provider": %q, "model": %q, "region": "us", "sku": "own-sku", "rates": {"input_tokens": {"usd": 7, "per": 1000000}},
		 "prices": {"input_tokens": {"usd": 7.7, "per": 1000000}}}]}`,
		vendor.Modality, vendor.Provider, vendor.Model)), 0o600))
	var logs bytes.Buffer
	files, err := loadPriceFiles(settings, slog.New(slog.NewTextHandler(&logs, nil)))
	s.Require().NoError(err)
	s.Contains(logs.String(), "markup_bps in a price file is ignored", "the file's markup is no error, and is said")
	routingConfig, err := routing.DefaultConfig()
	s.Require().NoError(err)

	list, err := newPrices(routingConfig, files, pricesVersion(settings), slog.New(slog.DiscardHandler))
	s.Require().NoError(err)

	used := pricing.Quantities{pricing.InputTokens: 1_000_000}
	own := list.Quote(pricing.Item{Modality: "llm", Provider: "own", Model: "own-model", Region: pricing.RegionUS}, used)
	s.EqualValues(3_600_000, own.CostMicros)
	s.EqualValues(3_000_000, own.BaseCostMicros)
	s.Equal("llm-own-own-model-us", own.SKU)
	s.Equal("files:dev", own.Version)
	ownBYOK := list.Quote(pricing.Item{Modality: "llm", Provider: "own", Model: "own-model", Region: pricing.RegionUS, BYOK: true}, used)
	s.EqualValues(600_000, ownBYOK.CostMicros, "the file's model has its BYOK twin, at a fifth of its rates")
	s.Zero(ownBYOK.BaseCostMicros)
	s.Equal("llm-own-own-model-us-byok", ownBYOK.SKU)
	overridden := list.Quote(pricing.Item{Modality: vendor.Modality, Provider: vendor.Provider, Model: vendor.Model, Region: pricing.RegionUS}, used)
	s.EqualValues(7_700_000, overridden.CostMicros)
	s.EqualValues(7_000_000, overridden.BaseCostMicros)
	s.Equal("own-sku", overridden.SKU)
}

func (s *PriceFileSuite) TestWithNoFileOfItsOwnTheRouterPricesFromTheVendorFile() {
	settings := config.Defaults()
	files, err := loadPriceFiles(settings, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	routingConfig, err := routing.DefaultConfig()
	s.Require().NoError(err)

	list, err := newPrices(routingConfig, files, pricesVersion(settings), slog.New(slog.DiscardHandler))
	s.Require().NoError(err)

	vendor := prices.Vendor().Entries[0]
	quote := list.Quote(pricing.Item{Modality: vendor.Modality, Provider: vendor.Provider, Model: vendor.Model, Region: vendor.Region},
		pricing.Quantities{pricing.InputTokens: 1_000_000})
	s.Equal("embedded:dev", quote.Version)
	s.Empty(list.Unpriced(routing.Items(routingConfig)))
}

func (s *PriceFileSuite) TestAMissingPricesFileStopsStartup() {
	settings := config.Defaults()
	settings.PricesFile = filepath.Join(s.T().TempDir(), "missing.json")

	_, err := loadPriceFiles(settings, slog.New(slog.DiscardHandler))

	s.ErrorContains(err, "missing.json")
}

func (s *PriceFileSuite) TestCheckingAGoodFileSaysItsMarkupIsIgnored() {
	path := filepath.Join(s.T().TempDir(), "own-prices.json")
	s.Require().NoError(os.WriteFile(path, []byte(`{"markup_bps": 1500, "entries": []}`), 0o600))
	var out bytes.Buffer

	s.Require().NoError(runPricesCheck([]string{"--file", path}, &out))

	s.Equal(fmt.Sprintf("%s: 0 entries; %d entries laid over vendor.json; markup_bps 1500 is ignored: each entry carries its price\n",
		path, len(prices.Vendor().Entries)), out.String())
}

// TestCheckingAFileRefusesWhatWouldStopTheStart: an entry the file alone takes, but that an
// item would match as well as one of vendor.json's, is refused as a router's start refuses it.
func (s *PriceFileSuite) TestCheckingAFileRefusesWhatWouldStopTheStart() {
	path := filepath.Join(s.T().TempDir(), "own-prices.json")
	s.Require().NoError(os.WriteFile(path, []byte(`{"entries": [
		{"modality": "llm", "provider": "openai", "model": "gpt-5.6-luna", "region": "us", "when": {"country": "DE"},
		 "rates": {"input_tokens": {"usd": 1, "per": 1000000}}}]}`), 0o600))

	err := runPricesCheck([]string{"--file", path}, &bytes.Buffer{})

	s.ErrorContains(err, "would match equally")
	_, err = loadPriceFiles(config.Config{PricesFile: path}, slog.New(slog.DiscardHandler))
	s.ErrorContains(err, "would match equally", "the start refuses it too")
}

func (s *PriceFileSuite) TestCheckingAGoodFileCountsItLaidOverTheVendorFile() {
	vendor := prices.Vendor().Entries[0]
	path := filepath.Join(s.T().TempDir(), "own-prices.json")
	s.Require().NoError(os.WriteFile(path, []byte(fmt.Sprintf(`{"entries": [
		{"modality": "llm", "provider": "own", "model": "own-model", "region": "us", "rates": {"input_tokens": {"usd": 3, "per": 1000000}}},
		{"modality": %q, "provider": %q, "model": %q, "region": "us", "rates": {"input_tokens": {"usd": 7, "per": 1000000}}}]}`,
		vendor.Modality, vendor.Provider, vendor.Model)), 0o600))
	var out bytes.Buffer

	s.Require().NoError(runPricesCheck([]string{"--file", path}, &out))

	s.Equal(fmt.Sprintf("%s: 3 entries, us 3 (1 byok); %d entries laid over vendor.json\n", path, len(prices.Vendor().Entries)+2),
		out.String(), "one new item and its BYOK twin, and one that replaces vendor.json's")
}

// ownConfig is a routing config of one priced and one local model, as a deployment of its
// own writes one.
const ownConfig = `llm:
  providers:
    # Billed per token.
    - provider: deepseek
      model: DeepSeek-V4-Flash-0731
      languages: [en]
      price:
        per_million_input_tokens: 0.13
        # Output is dearer.
        per_million_output_tokens: 0.26
        service_tiers:
          flex: 0.5

      description: "kept"
    - provider: gemma
      model: gemma-4-26B-A4B-it
      local: true
      languages: [en]
      price: { per_million_input_tokens: 0.24 }
`

func (s *PriceFileSuite) TestExportWritesAConfigsPricesAsAFile() {
	from := filepath.Join(s.T().TempDir(), "routing.yaml")
	s.Require().NoError(os.WriteFile(from, []byte(ownConfig), 0o600))
	var out bytes.Buffer

	s.Require().NoError(runPricesExport([]string{"--from", from}, config.Defaults(), &out))

	s.Equal(`{
  "entries": [
    {"modality":"llm","provider":"deepseek","model":"DeepSeek-V4-Flash-0731","region":"us","rates":{"input_tokens":{"usd":0.13,"per":1000000},"output_tokens":{"usd":0.26,"per":1000000}}},
    {"modality":"llm","provider":"deepseek","model":"DeepSeek-V4-Flash-0731","region":"us","when":{"service_tier":"flex"},"rates":{"input_tokens":{"usd":0.065,"per":1000000},"output_tokens":{"usd":0.13,"per":1000000}}},
    {"modality":"llm","provider":"gemma","model":"gemma-4-26B-A4B-it","region":"us","rates":{"input_tokens":{"usd":0.24,"per":1000000}}}
  ]
}
`, out.String())
}

func (s *PriceFileSuite) TestExportAddsThePhoneVendorsWhenAsked() {
	from := filepath.Join(s.T().TempDir(), "routing.yaml")
	s.Require().NoError(os.WriteFile(from, []byte(ownConfig), 0o600))
	var out bytes.Buffer

	s.Require().NoError(runPricesExport([]string{"--from", from, "--phone"}, config.Defaults(), &out))

	file, err := prices.Parse("exported", out.Bytes())
	s.Require().NoError(err)
	s.Contains(file.Entries, pricing.Entry{Modality: "phone", Provider: phone.SIPTrunkVendor, Model: phone.ModelDial,
		Region: pricing.DefaultRegion, Free: true})
}

func (s *PriceFileSuite) TestStripTakesOutEveryPriceBlockAndNothingElse() {
	from := filepath.Join(s.T().TempDir(), "routing.yaml")
	s.Require().NoError(os.WriteFile(from, []byte(ownConfig), 0o600))

	s.Require().NoError(runPricesExport([]string{"--from", from, "--out", filepath.Join(s.T().TempDir(), "own.json"), "--strip"},
		config.Defaults(), &bytes.Buffer{}))

	stripped, err := os.ReadFile(from)
	s.Require().NoError(err)
	s.Equal(`llm:
  providers:
    # Billed per token.
    - provider: deepseek
      model: DeepSeek-V4-Flash-0731
      languages: [en]

      description: "kept"
    - provider: gemma
      model: gemma-4-26B-A4B-it
      local: true
      languages: [en]
`, string(stripped))
}

func (s *PriceFileSuite) TestExportRefusesAConfigWithNoPrices() {
	from := filepath.Join(s.T().TempDir(), "routing.yaml")
	s.Require().NoError(os.WriteFile(from, []byte(strings.ReplaceAll(ownConfig, "price:", "unpriced:")), 0o600))

	err := runPricesExport([]string{"--from", from}, config.Defaults(), &bytes.Buffer{})

	s.Error(err)
}

// TestReloadRunsOnlyAgainstADevelopmentDatabase: the guard, before anything opens a database.
func (s *PriceFileSuite) TestReloadRunsOnlyAgainstADevelopmentDatabase() {
	withDSN := func(dsn string) config.Config {
		settings := config.Defaults()
		settings.Postgres.DSN = dsn
		return settings
	}
	for _, dsn := range []string{
		"postgres://postgres:postgres@localhost:55432/model_router?sslmode=disable",
		"postgres://postgres:postgres@127.0.0.1:55432/model_router?sslmode=disable",
		"postgres://postgres:postgres@[::1]:55432/model_router?sslmode=disable",
		"postgres://postgres:postgres@postgres:5432/model_router?sslmode=disable",
		"postgres://postgres:postgres@localhost/model_router?host=/var/run/postgresql/.s.PGSQL.5432",
		"postgres://postgres:postgres@localhost/model_router?host=127.0.0.1:55432",
	} {
		for _, environment := range []string{"", "local", "development", "testing"} {
			s.T().Setenv(config.EnvVar, environment)
			s.NoError(developmentDatabase(true, withDSN(dsn)), "%s with %s=%q", dsn, config.EnvVar, environment)
		}
	}

	s.T().Setenv(config.EnvVar, "")
	local := withDSN("postgres://postgres:secret@localhost:55432/model_router?sslmode=disable")
	s.ErrorContains(developmentDatabase(false, local), "pass --development")
	remote := developmentDatabase(true, withDSN("postgres://postgres:secret@db.example.com:5432/model_router?sslmode=require"))
	s.ErrorContains(remote, `names the host "db.example.com"`)
	s.NotContains(remote.Error(), "secret")
	// The driver connects to a host query parameter in place of the URL's host.
	for _, dsn := range []string{
		"postgres://postgres:secret@localhost/model_router?host=db.example.com:5432",
		"postgres://postgres:secret@localhost:55432/model_router?sslmode=disable&host=10.0.0.7",
	} {
		refused := developmentDatabase(true, withDSN(dsn))
		s.ErrorContains(refused, "connects to", dsn)
		s.NotContains(refused.Error(), "secret")
	}
	for _, dsn := range []string{
		"host=localhost user=postgres password=secret dbname=model_router",
		"postgres://postgres:secret@/model_router",
		"mysql://postgres:secret@localhost/model_router",
		"postgres://:secret@localhost/model_router",
		"postgres://postgres:secret@db.example.com/model_router?host=/var/run/postgresql/.s.PGSQL.5432",
	} {
		refused := developmentDatabase(true, withDSN(dsn))
		s.Error(refused, dsn)
		s.NotContains(refused.Error(), "secret", dsn)
	}
	s.ErrorContains(developmentDatabase(true, withDSN("")), "set postgres.dsn")
	s.T().Setenv(config.EnvVar, "staging")
	refused := developmentDatabase(true, local)
	s.ErrorContains(refused, `ROUTER_ENV is "staging"`)
	s.ErrorContains(refused, "unset, local, development or testing")
}
