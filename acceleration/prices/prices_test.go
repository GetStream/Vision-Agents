package prices

import (
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
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
	return pricing.Entry{Modality: "llm", Provider: "openai", Model: model, Rates: map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillion(input)}}
}

func (s *PricesSuite) TestTheVendorFileIsAValidList() {
	_, err := pricing.NewList("vendor", Vendor().Entries, slog.New(slog.DiscardHandler))

	s.Require().NoError(err)
	s.NotEmpty(Vendor().Entries)
}

func (s *PricesSuite) TestTheVendorFileHasNoMarkup() {
	vendor := Vendor()

	s.Nil(vendor.MarkupBps, "vendor.json holds the vendors' list prices: a markup is the operator's")
	for _, entry := range vendor.Entries {
		s.Nil(entry.MarkupBps, Name(entry))
	}
}

func (s *PricesSuite) TestTheFirstFileWinsForAnItem() {
	own := llm("m", 3)
	flex := llm("m", 1)
	flex.When = map[string]string{"service_tier": "flex"}

	merged := Merge(File{Entries: []pricing.Entry{own}}, File{Entries: []pricing.Entry{llm("m", 2), flex}})

	s.Equal([]pricing.Entry{own, flex}, merged, "the first file's default entry, and the second's flex entry beside it")
}

func (s *PricesSuite) TestAFileWithAMisspeltRateIsRefused() {
	_, err := Parse("own.json", []byte(`{"entries": [{"modality": "llm", "provider": "openai", "model": "m",
		"rates": {"input_tokens": {"usd": 1, "per": 1000000}}, "rate": {}}]}`))

	s.ErrorContains(err, `own.json: json: unknown field "rate"`)
}

func (s *PricesSuite) TestAFileThatDoesNotBuildAListNamesTheEntry() {
	_, err := Parse("own.json", []byte(`{"entries": [{"modality": "llm", "provider": "openai", "model": "m",
		"rates": {"input_tokens": {"usd": 1, "per": 0}}}]}`))

	s.ErrorContains(err, "own.json: pricing: llm/openai/m rates input_tokens per 0 units")
}

func (s *PricesSuite) TestAFileMayNotMarkAModelLocal() {
	_, err := Parse("own.json", []byte(`{"entries": [{"modality": "llm", "provider": "gemma", "model": "m", "local": true}]}`))

	s.ErrorContains(err, "own.json: llm/gemma/m sets local")
}

func (s *PricesSuite) TestAFileMayNotNameAVersion() {
	_, err := Parse("own.json", []byte(`{"entries": [{"modality": "llm", "provider": "openai", "model": "m", "free": true, "version": "x"}]}`))

	s.ErrorContains(err, "own.json: llm/openai/m sets a version")
}

func (s *PricesSuite) TestAFileMarkupIsBounded() {
	_, err := Parse("own.json", []byte(`{"markup_bps": 100001, "entries": []}`))

	s.ErrorContains(err, "outside 0 to 100000")
}

func (s *PricesSuite) TestAFileReadsItsMarkup() {
	file, err := Parse("own.json", []byte(`{"markup_bps": 1500, "entries": []}`))

	s.Require().NoError(err)
	s.Require().NotNil(file.MarkupBps)
	s.EqualValues(1_500, *file.MarkupBps)
}

func (s *PricesSuite) TestALocalItemIsMarkedOnEveryEntryOrGivenOne() {
	priced := llm("own-gpu", 1)
	flex := llm("own-gpu", 0.5)
	flex.When = map[string]string{"service_tier": "flex"}

	marked := WithLocal([]pricing.Entry{priced, flex}, []pricing.Item{
		{Modality: "llm", Provider: "openai", Model: "own-gpu"},
		{Modality: "stt", Provider: "self", Model: "unpriced"},
	})

	s.Require().Len(marked, 3)
	s.True(marked[0].Local)
	s.True(marked[1].Local)
	s.Equal(pricing.Entry{Modality: "stt", Provider: "self", Model: "unpriced", Local: true}, marked[2])
	s.False(priced.Local, "the entries passed in are left as they were")
}

func (s *PricesSuite) TestTheEmbeddedListNamesItsRelease() {
	list, err := Embedded(nil, "v1.4.0", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	entry := Vendor().Entries[0]

	quote := list.Quote(pricing.Item{Modality: entry.Modality, Provider: entry.Provider, Model: entry.Model, Attributes: entry.When},
		pricing.Quantities{pricing.InputTokens: 1_000})

	s.Equal("embedded:v1.4.0", quote.Version)
}

// TestNoPriceTableIsTruncated: rows of the price list are never changed or deleted, which a
// trigger enforces, but a trigger on rows does not see TRUNCATE. So nothing the router runs
// may say it. Test code resets its own databases and is not read.
func (s *PricesSuite) TestNoPriceTableIsTruncated() {
	truncates := regexp.MustCompile(`(?i)\bTRUNCATE\s+(TABLE\s+)?(ONLY\s+)?([\w."]+\s*,\s*)*"?price_(entries|changes)\b`)
	for _, statement := range []string{"TRUNCATE price_entries", "truncate table only price_changes",
		"TRUNCATE requests, stats_hourly, price_entries CASCADE"} {
		s.True(truncates.MatchString(statement), statement)
	}
	s.False(truncates.MatchString("at := now.Truncate(time.Hour) // price_entries"), "Go's Truncate is not SQL")
	root, err := filepath.Abs("..")
	s.Require().NoError(err)

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
		s.Falsef(truncates.Match(text), "%s truncates a price table", path)
		return nil
	})
	s.Require().NoError(err)
}
