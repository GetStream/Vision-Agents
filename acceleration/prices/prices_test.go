package prices

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
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
	return pricing.Entry{Modality: "llm", Provider: "openai", Model: model, Rates: map[pricing.Meter]pricing.Rate{pricing.InputTokens: perMillion(input)}}
}

func (s *PricesSuite) TestTheVendorFileIsAValidList() {
	_, err := pricing.NewList("vendor", Vendor().Entries, 0, slog.New(slog.DiscardHandler))

	s.Require().NoError(err)
	s.NotEmpty(Vendor().Entries)
}

func (s *PricesSuite) TestALookupIsPricedByTheBytesTurbopufferBillsItAsSearching() {
	lookup := pricing.Item{Modality: "knowledge", Provider: "turbopuffer", Model: "lookup"}
	searching := func(queried int64) pricing.Quantities {
		return pricing.Quantities{pricing.Requests: 1, pricing.BytesQueried: queried, pricing.BytesReturned: 1_000, pricing.Results: 5}
	}
	atList, err := pricing.NewList("vendor", Vendor().Entries, 0, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	marked, err := pricing.NewList("vendor", Vendor().Entries, 2_500, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)

	minimum := atList.Quote(lookup, searching(1_280_000_000))
	s.EqualValues(1, minimum.CostMicros, "1.28 GB at $1 a PB and a KB returned at $0.05 a GB is 1.33 µ$, rounded half up")
	s.True(minimum.Billable)
	s.EqualValues(2, marked.Quote(lookup, searching(1_280_000_000)).CostMicros, "the same lookup with a markup rounds up")
	s.EqualValues(10, atList.Quote(lookup, searching(10_000_000_000)).CostMicros,
		"a 10 GB namespace is billed what turbopuffer bills for searching it")
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
// may say it, but for one statement in store.ReloadPrices, which empties a development
// database's list (router prices reload, whose command refuses any other database). Test code
// resets its own databases and is not read.
func (s *PricesSuite) TestNoPriceTableIsTruncated() {
	for _, statement := range []string{"TRUNCATE price_entries", "truncate table only price_changes",
		"TRUNCATE requests, stats_hourly, price_entries CASCADE", "TRUNCATE public.price_entries",
		`TRUNCATE "public"."price_changes"`, `TRUNCATE "price_entries"`, "TRUNCATE audit_log, public.price_changes",
		"TRUNCATE /* x */ price_entries", "TRUNCATE/* x */price_entries", "TRUNCATE -- x\n  price_changes",
		"TRUNCATE TABLE ONLY public . price_entries *", "TRUNCATE requests *, price_changes"} {
		s.True(truncatesPriceTable.MatchString(statement), statement)
	}
	for _, text := range []string{"at := now.Truncate(time.Hour) // price_entries", "TRUNCATE requests",
		"TRUNCATE price_entries_archive", "-- price_entries are never truncated"} {
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

// truncatesPriceTable is a TRUNCATE of either price table: in any case, with TABLE and ONLY or
// without, in a list, under a schema, quoted or not, and with comments between the words.
var truncatesPriceTable = func() *regexp.Regexp {
	gap := `(?:\s|/\*.*?\*/|--[^\n]*(?:\n|$))`
	name := `(?:"[^"]+"|\w+)`
	schema := `(?:` + name + gap + `*\.` + gap + `*)?`
	listed := schema + name + gap + `*\*?` + gap + `*,` + gap + `*`
	return regexp.MustCompile(`(?is)\bTRUNCATE` + gap + `+(?:TABLE` + gap + `+)?(?:ONLY` + gap + `+)?(?:` + listed + `)*` +
		schema + `"?price_(?:entries|changes)\b`)
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
