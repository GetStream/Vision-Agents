package pricing

import (
	"strings"
)

// inRegion is the entry in region, BYOK or not.
func inRegion(entry Entry, region string, byok bool) Entry {
	entry.Region, entry.BYOK = region, byok
	return entry
}

// itemIn is the item in region, BYOK or not.
func itemIn(item Item, region string, byok bool) Item {
	item.Region, item.BYOK = region, byok
	return item
}

func (s *PricingSuite) TestAnEntryNeedsARegion() {
	_, err := NewList("test", atCost(inRegion(llmEntry("m", nil), "", false)), nil)

	s.ErrorIs(err, ErrRegion)
	s.ErrorContains(err, "every entry names its region, such as us: llm/openai/m names none")
}

func (s *PricingSuite) TestARegionIsALetterAndUpTo31More() {
	for _, region := range []string{"US", "1eu", "-us", "us_east", strings.Repeat("a", 33)} {
		_, err := NewList("test", atCost(inRegion(llmEntry("m", nil), region, false)), nil)
		s.ErrorIs(err, ErrRegion, region)
	}
	for _, region := range []string{"us", "eu", "us-east-1", strings.Repeat("a", 32)} {
		_, err := NewList("test", atCost(inRegion(llmEntry("m", nil), region, false)), nil)
		s.NoError(err, region)
	}
}

func (s *PricingSuite) TestAnEUItemIsPricedByItsEUEntry() {
	us := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)})
	list := s.list(us, inRegion(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1.1)}), RegionEU, false))
	used := Quantities{InputTokens: 1_000_000}

	s.Equal(int64(1_100_000), list.Quote(itemIn(llmItem("m", nil), RegionEU, false), used).CostMicros)
	s.Equal(int64(1_000_000), list.Quote(llmItem("m", nil), used).CostMicros)
}

// TestAnEUItemWithOnlyAUSEntryIsUnpriced: there is no fallback to another region's price, so
// EU work no EU entry prices is unpriced and logged, never billed at the US price.
func (s *PricingSuite) TestAnEUItemWithOnlyAUSEntryIsUnpriced() {
	list := s.list(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)}))
	eu := itemIn(llmItem("m", nil), RegionEU, false)

	quote := list.Quote(eu, Quantities{InputTokens: 1_000_000})

	s.True(quote.Unpriced)
	s.Zero(quote.CostMicros)
	s.Empty(quote.SKU)
	_, found := list.Entry(eu)
	s.False(found)
	s.Equal([]Item{eu}, list.Unpriced([]Item{eu, llmItem("m", nil)}))
	s.Contains(s.logs.String(), "provider=openai model=m region=eu byok=false")
}

// TestAnItemThatNamesNoRegionIsUnpriced: DefaultRegion is where a provider that names no region
// runs, which routing fills in; it never stands in for a region an item lacks.
func (s *PricingSuite) TestAnItemThatNamesNoRegionIsUnpriced() {
	list := s.list(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)}))

	s.True(list.Quote(itemIn(llmItem("m", nil), "", false), Quantities{InputTokens: 1}).Unpriced)
}

func (s *PricingSuite) TestABYOKItemIsNeverPricedByTheEntryThatIsNot() {
	list := s.list(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)}))

	quote := list.Quote(itemIn(llmItem("m", nil), RegionUS, true), Quantities{InputTokens: 1_000_000})

	s.True(quote.Unpriced)
	s.Zero(quote.CostMicros)
	s.Contains(s.logs.String(), "region=us byok=true")
}

func (s *PricingSuite) TestAnItemThatIsNotBYOKIsNeverPricedByTheBYOKEntry() {
	list := s.list(inRegion(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(0.2)}), RegionUS, true))

	quote := list.Quote(llmItem("m", nil), Quantities{InputTokens: 1_000_000})

	s.True(quote.Unpriced)
	s.Zero(quote.CostMicros)
}

// TestABYOKEntryWithoutAPriceIsPricedAtAFifthOfItsRates: if a million tokens cost the vendor $1,
// the price is $1.20, and with the customer's own key $0.20, its tiers included.
func (s *PricingSuite) TestABYOKEntryWithoutAPriceIsPricedAtAFifthOfItsRates() {
	entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1), OutputTokens: {USD: 1.1, Per: 1_000_000, Increment: 10}})
	entry.Tiers = []Tier{{AboveInputTokens: 1_000, Rates: map[Meter]Rate{InputTokens: perMillion(4), OutputTokens: perMillion(1.375)}}}

	byok := Priced(inRegion(entry, RegionUS, true))
	plain := Priced(entry)

	s.Equal(map[Meter]Rate{InputTokens: perMillion(0.2), OutputTokens: {USD: 0.22, Per: 1_000_000, Increment: 10}}, byok.Prices,
		"1.1 × 0.2 reads 0.22, not 0.22000000000000003")
	s.Equal([]Tier{{AboveInputTokens: 1_000, Rates: map[Meter]Rate{InputTokens: perMillion(0.8), OutputTokens: perMillion(0.275)}}},
		byok.PriceTiers)
	s.Equal(perMillion(1.2), plain.Prices[InputTokens])
	s.Equal(entry.Rates, byok.Rates, "the vendor's rates stay as they were")
	s.Equal(int64(2_000), int64(BYOKFeeBps))
}

// TestABYOKQuoteCostsTheOperatorNothing: the vendor bills the customer for the work, so a BYOK
// quote's base cost is 0 and what it charges is its price, the fee.
func (s *PricingSuite) TestABYOKQuoteCostsTheOperatorNothing() {
	entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1), OutputTokens: perMillion(4)})
	entry.Tiers = []Tier{{AboveInputTokens: 1_000_000, Rates: map[Meter]Rate{InputTokens: perMillion(2), OutputTokens: perMillion(8)}}}
	list, err := NewList("test", PricedAll([]Entry{inRegion(entry, RegionUS, true)}), nil)
	s.Require().NoError(err)
	byok := itemIn(llmItem("m", nil), RegionUS, true)

	short := list.Quote(byok, Quantities{InputTokens: 1_000_000, OutputTokens: 1_000_000})
	long := list.Quote(byok, Quantities{InputTokens: 2_000_000, OutputTokens: 1_000})

	s.Equal(int64(1_000_000), short.CostMicros, "(1 + 4) × 0.2")
	s.Equal(int64(800_000), short.OutputCostMicros)
	s.Zero(short.BaseCostMicros)
	s.Equal(int64(801_600), long.CostMicros, "the tier's 2 × 2 + 8 × 0.001, times 0.2")
	s.Zero(long.BaseCostMicros)
	s.Equal("llm-openai-m-us-byok", short.SKU)
	s.True(short.Billable)
}

// TestEveryLLMKnowledgeAndSandboxEntryGetsABYOKTwin: the twin is the same item, region and When
// with the same rates and free flag, and no price or SKU, so it is priced and named as a BYOK
// entry when it is written. Other modalities get none.
func (s *PricingSuite) TestEveryLLMKnowledgeAndSandboxEntryGetsABYOKTwin() {
	flex := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(0.5)})
	flex.When = map[string]string{"service_tier": "flex"}
	flex.SKU = "own-flex"
	flex.Prices = map[Meter]Rate{InputTokens: perMillion(0.7)}
	eu := inRegion(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1.1)}), RegionEU, false)
	knowledge := Entry{Modality: "knowledge", Provider: "turbopuffer", Model: "write", Region: RegionUS,
		Rates: map[Meter]Rate{BytesWritten: {USD: 2, Per: 1_000_000_000, Minimum: 10_000}}}
	sandbox := Entry{Modality: "sandbox", Provider: "daytona", Model: "running", Region: RegionUS,
		When: map[string]string{"class": "container"}, Rates: map[Meter]Rate{VCPUSeconds: {USD: 0.0504, Per: 3600}}}
	free := Entry{Modality: "llm", Provider: "gemma", Model: "shared", Region: RegionUS, Free: true}
	stt := Entry{Modality: "stt", Provider: "deepgram", Model: "flux", Region: RegionUS, Rates: map[Meter]Rate{AudioMs: {USD: 0.3, Per: 3_600_000}}}

	all := WithBYOK([]Entry{flex, eu, knowledge, sandbox, free, stt})

	twin := func(entry Entry) Entry {
		return Entry{Modality: entry.Modality, Provider: entry.Provider, Model: entry.Model, Region: entry.Region, BYOK: true,
			When: entry.When, Free: entry.Free, Rates: entry.Rates, Tiers: entry.Tiers}
	}
	s.Equal([]Entry{flex, twin(flex), eu, twin(eu), knowledge, twin(knowledge), sandbox, twin(sandbox), free, twin(free), stt}, all)
	s.Equal(perMillion(0.22), Priced(all[3]).Prices[InputTokens], "the EU twin is a fifth of the EU rates")
}

func (s *PricingSuite) TestABYOKEntryTheEntriesHaveIsKept() {
	own := inRegion(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)}), RegionUS, true)
	own.Prices = map[Meter]Rate{InputTokens: perMillion(0.05)}
	own.SKU = "own-byok"

	all := WithBYOK([]Entry{llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)}), own})

	s.Equal([]Entry{llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)}), own}, all)
}

// TestEntriesListEachItemsRegionsWithItsBYOKTwinAfterIt: by modality, provider, model and
// region, the entry that is not BYOK first, then the fewest conditions.
func (s *PricingSuite) TestEntriesListEachItemsRegionsWithItsBYOKTwinAfterIt() {
	flex := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(0.5)})
	flex.When = map[string]string{"service_tier": "flex"}
	list := s.list(PricedAll(WithBYOK([]Entry{
		inRegion(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1.1)}), RegionEU, false),
		flex,
		llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)}),
	}))...)

	var names []string
	for _, entry := range list.Entries() {
		names = append(names, nameWithWhen(entry))
	}

	s.Equal([]string{"llm/openai/m (eu)", "llm/openai/m (eu, byok)", "llm/openai/m (us)", "llm/openai/m (us) when service_tier=flex",
		"llm/openai/m (us, byok)", "llm/openai/m (us, byok) when service_tier=flex"}, names)
}

// TestALocalItemIsMarkedInEveryRegionAndBYOKState: a model the deployment runs itself is local
// wherever it is priced, and gets an entry of its own in its region when it has none there.
func (s *PricingSuite) TestALocalItemIsMarkedInEveryRegionAndBYOKState() {
	entries := WithBYOK([]Entry{
		llmEntry("own-gpu", map[Meter]Rate{InputTokens: perMillion(1)}),
		inRegion(llmEntry("own-gpu", map[Meter]Rate{InputTokens: perMillion(1.1)}), RegionEU, false),
	})

	marked := WithLocal(entries, []Item{llmItem("own-gpu", nil), itemIn(llmItem("eu-only", nil), RegionEU, false)})

	s.Require().Len(marked, 5)
	for _, entry := range marked[:4] {
		s.True(entry.Local, nameWithWhen(entry))
	}
	s.Equal(Entry{Modality: "llm", Provider: "openai", Model: "eu-only", Region: RegionEU, Local: true}, marked[4])
}
