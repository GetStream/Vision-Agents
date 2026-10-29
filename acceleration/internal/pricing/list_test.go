package pricing

import (
	"bytes"
	"log/slog"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type PricingSuite struct {
	suite.Suite
	// logs is what the list under test logged.
	logs *bytes.Buffer
}

func TestPricingSuite(t *testing.T) {
	suite.Run(t, new(PricingSuite))
}

func (s *PricingSuite) SetupTest() {
	s.logs = &bytes.Buffer{}
}

// list is a price list of the entries, each with no price of its own priced at its vendor
// rates, logging to s.logs.
func (s *PricingSuite) list(entries ...Entry) *List {
	list, err := NewList("test", atCost(entries...), slog.New(slog.NewTextHandler(s.logs, nil)))
	s.Require().NoError(err)
	return list
}

// atCost is the entries, each with no price of its own priced at its vendor rates, as most
// of these tests price them: what a rate charges is then what it costs.
func atCost(entries ...Entry) []Entry {
	priced := make([]Entry, 0, len(entries))
	for _, entry := range entries {
		if !entry.Free && entry.Prices == nil && entry.PriceTiers == nil {
			entry.Prices, entry.PriceTiers = entry.Rates, entry.Tiers
		}
		priced = append(priced, entry)
	}
	return priced
}

// perMillion is a token rate as vendors publish it.
func perMillion(usd float64) Rate { return Rate{USD: usd, Per: 1_000_000} }

func llmEntry(model string, rates map[Meter]Rate) Entry {
	return Entry{Modality: "llm", Provider: "openai", Model: model, Region: RegionUS, Rates: rates}
}

func llmItem(model string, attributes map[string]string) Item {
	return Item{Modality: "llm", Provider: "openai", Model: model, Region: RegionUS, Attributes: attributes}
}

// terra is a text model with a rate for cache writes, at what OpenAI charges for it.
var terra = llmEntry("gpt-5.6-terra", map[Meter]Rate{
	InputTokens:           perMillion(2),
	CachedInputTokens:     perMillion(0.20),
	CacheWriteInputTokens: perMillion(2.50),
	OutputTokens:          perMillion(12),
})

func (s *PricingSuite) TestARateIsChargedPerItsUnit() {
	list := s.list(llmEntry("m", map[Meter]Rate{OutputTokens: perMillion(4)}))

	quote := list.Quote(llmItem("m", nil), Quantities{OutputTokens: 250_000})

	s.Equal(int64(1_000_000), quote.CostMicros)
	s.Equal("test", quote.Version)
	s.True(quote.Billable)
}

func (s *PricingSuite) TestAnIncrementRoundsTheQuantityUp() {
	list := s.list(Entry{Modality: "phone", Provider: "telnyx", Model: "minutes", Region: RegionUS, Rates: map[Meter]Rate{
		CallSeconds: {USD: 0.007, Per: 60, Increment: 60},
	}})

	quote := list.Quote(Item{Modality: "phone", Provider: "telnyx", Model: "minutes", Region: RegionUS}, Quantities{CallSeconds: 61})

	s.Equal(int64(14_000), quote.CostMicros, "61 seconds are two started minutes")
}

func (s *PricingSuite) TestAMinimumIsBilledOnceAnythingIsUsed() {
	list := s.list(Entry{Modality: "phone", Provider: "telnyx", Model: "minutes", Region: RegionUS, Rates: map[Meter]Rate{
		CallSeconds: {USD: 0.06, Per: 60, Minimum: 60},
	}})
	item := Item{Modality: "phone", Provider: "telnyx", Model: "minutes", Region: RegionUS}

	s.Equal(int64(60_000), list.Quote(item, Quantities{CallSeconds: 1}).CostMicros)
	s.Equal(int64(0), list.Quote(item, Quantities{CallSeconds: 0}).CostMicros)
}

func (s *PricingSuite) TestTheEntryMatchingMostAttributesWins() {
	sms := func(when map[string]string, usd float64) Entry {
		return Entry{Modality: "sms", Provider: "telnyx", Model: "sms", Region: RegionUS, When: when,
			Rates: map[Meter]Rate{MessageSegments: {USD: usd, Per: 1}}}
	}
	list := s.list(sms(nil, 0.004), sms(map[string]string{"country": "CA"}, 0.0075))
	in := func(country string) Item {
		return Item{Modality: "sms", Provider: "telnyx", Model: "sms", Region: RegionUS, Attributes: map[string]string{"country": country}}
	}

	s.Equal(int64(7_500), list.Quote(in("CA"), Quantities{MessageSegments: 1}).CostMicros)
	s.Equal(int64(4_000), list.Quote(in("GB"), Quantities{MessageSegments: 1}).CostMicros)
}

func (s *PricingSuite) TestAServiceTierPicksItsEntry() {
	flex := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)})
	flex.When = map[string]string{"service_tier": "flex"}
	list := s.list(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(2)}), flex)
	used := Quantities{InputTokens: 1_000_000}

	s.Equal(int64(1_000_000), list.Quote(llmItem("m", map[string]string{"service_tier": "flex"}), used).CostMicros)
	s.Equal(int64(2_000_000), list.Quote(llmItem("m", map[string]string{"service_tier": "default"}), used).CostMicros)
	s.Equal(int64(2_000_000), list.Quote(llmItem("m", nil), used).CostMicros)
	s.Equal(int64(2_000_000), list.Quote(llmItem("m", map[string]string{"service_tier": "warp"}), used).CostMicros)
}

func (s *PricingSuite) TestACachedTokenWithNoRateCostsAnInputToken() {
	list := s.list(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(0.15), OutputTokens: perMillion(0.47)}))

	quote := list.Quote(llmItem("m", nil), Quantities{InputTokens: 6_000, CachedInputTokens: 4_000})

	s.Equal(int64(1_500), quote.CostMicros, "10,000 prompt tokens at $0.15, not 6,000 and the rest free")
}

func (s *PricingSuite) TestACacheWriteWithNoRateCostsAnInputToken() {
	list := s.list(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(2), CachedInputTokens: perMillion(0.2)}))

	quote := list.Quote(llmItem("m", nil), Quantities{InputTokens: 1_000, CacheWriteInputTokens: 1_000})

	s.Equal(int64(4_000), quote.CostMicros)
}

func (s *PricingSuite) TestAZeroRateWrittenOutStopsTheFallback() {
	list := s.list(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(2), CachedInputTokens: {USD: 0, Per: 1}}))

	quote := list.Quote(llmItem("m", nil), Quantities{InputTokens: 1_000, CachedInputTokens: 1_000})

	s.Equal(int64(2_000), quote.CostMicros)
}

func (s *PricingSuite) TestAQuoteSaysWhetherTheWorkUsedAnythingItIsPricedBy() {
	list := s.list(Entry{Modality: "sts", Provider: "xai", Model: "voice", Region: RegionUS, Rates: map[Meter]Rate{
		AudioMs:           {USD: 4.8, Per: 3_600_000},
		OutputAudioTokens: {USD: 0, Per: 1_000_000},
	}})
	item := Item{Modality: "sts", Provider: "xai", Model: "voice", Region: RegionUS}

	s.True(list.Quote(item, Quantities{AudioMs: 2_000}).Rated)
	s.True(list.Quote(item, Quantities{OutputAudioTokens: 40}).Rated, "a zero rate written out is still what the work is priced by")
	s.False(list.Quote(item, Quantities{InputTextTokens: 40}).Rated, "a meter the entry has no rate for")
	s.False(list.Quote(item, Quantities{AudioMs: 0}).Rated, "a meter that used nothing")
}

func (s *PricingSuite) TestQuantitiesOfNothingAreZero() {
	s.True(Quantities{}.Zero())
	s.True(Quantities{InputTokens: 0, OutputTokens: 0}.Zero())
	s.False(Quantities{InputTokens: 0, Requests: 1}.Zero())
}

// TestACacheWriteIsPricedAtItsOwnRate is a first turn OpenAI cached whole: 3 tokens read
// fresh, 2,386 written to the cache and 22 generated, which OpenAI billed 6,235 µ$. At the
// input rate the writes would come to 5,042.
func (s *PricingSuite) TestACacheWriteIsPricedAtItsOwnRate() {
	quote := s.list(terra).Quote(llmItem("gpt-5.6-terra", nil),
		Quantities{InputTokens: 3, CacheWriteInputTokens: 2_386, OutputTokens: 22})

	s.Equal(int64(6_235), quote.CostMicros)
}

func (s *PricingSuite) TestALongPromptPricesTheWholeRequestAtItsTier() {
	entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1), OutputTokens: perMillion(4)})
	entry.Tiers = []Tier{{AboveInputTokens: 272_000, Rates: map[Meter]Rate{InputTokens: perMillion(2), OutputTokens: perMillion(6)}}}
	list := s.list(entry)

	long := list.Quote(llmItem("m", nil), Quantities{InputTokens: 272_001, OutputTokens: 1_000_000})
	limit := list.Quote(llmItem("m", nil), Quantities{InputTokens: 272_000, OutputTokens: 1_000_000})

	s.Equal(int64(544_002+6_000_000), long.CostMicros)
	s.Equal(int64(272_000+4_000_000), limit.CostMicros)
}

func (s *PricingSuite) TestATierIsMeasuredOnTheWholePrompt() {
	entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1), CachedInputTokens: perMillion(0.1)})
	entry.Tiers = []Tier{{AboveInputTokens: 100, Rates: map[Meter]Rate{InputTokens: perMillion(10_000), CachedInputTokens: perMillion(1_000)}}}

	quote := s.list(entry).Quote(llmItem("m", nil), Quantities{InputTokens: 60, CachedInputTokens: 60})

	s.Equal(int64(600_000+60_000), quote.CostMicros, "60 fresh and 60 cached is a prompt of 120")
}

// realtime is a speech-to-speech model priced by modality, with its audio rates as the
// generic ones.
var realtime = Entry{Modality: "sts", Provider: "openai", Model: "gpt-realtime-2.1", Region: RegionUS, Rates: map[Meter]Rate{
	InputTokens:       perMillion(32),
	CachedInputTokens: perMillion(0.40),
	OutputTokens:      perMillion(64),
	InputTextTokens:   perMillion(4),
	OutputTextTokens:  perMillion(24),
	OutputAudioTokens: perMillion(64),
	InputImageTokens:  perMillion(5),
}}

// TestAModalitySplitReplacesTheTotals is a typed Realtime turn: 760 text tokens in, 61 text
// and 111 audio out, which OpenAI billed 11,608 µ$. The row carries the totals as well,
// and they are not billed a second time.
func (s *PricingSuite) TestAModalitySplitReplacesTheTotals() {
	list := s.list(realtime)
	item := Item{Modality: "sts", Provider: "openai", Model: "gpt-realtime-2.1", Region: RegionUS}

	quote := list.Quote(item, Quantities{
		InputTokens: 760, OutputTokens: 172,
		InputTextTokens: 760, OutputTextTokens: 61, OutputAudioTokens: 111,
	})

	s.Equal(int64(11_608), quote.CostMicros)
	s.Equal(int64(11_608-3_040), quote.OutputCostMicros)
}

func (s *PricingSuite) TestAModalityMeterWithNoRateFallsBackToTheGenericOne() {
	list := s.list(realtime)
	item := Item{Modality: "sts", Provider: "openai", Model: "gpt-realtime-2.1", Region: RegionUS}

	quote := list.Quote(item, Quantities{InputAudioTokens: 1_000, CachedInputAudioTokens: 1_000})

	s.Equal(int64(32_000+400), quote.CostMicros, "audio in at the generic input rate, cached audio at the cached one")
}

// TestCostsRoundHalfUp includes a turn that wrote 2,373 tokens to the cache, read 17 fresh
// and generated 24: 6,254.5 µ$.
func (s *PricingSuite) TestCostsRoundHalfUp() {
	list := s.list(
		llmEntry("half", map[Meter]Rate{InputTokens: {USD: 7.5, Per: 1_000_000}}),
		llmEntry("under", map[Meter]Rate{InputTokens: {USD: 7.49, Per: 1_000_000}}),
		terra,
	)

	s.Equal(int64(8), list.Quote(llmItem("half", nil), Quantities{InputTokens: 1}).CostMicros)
	s.Equal(int64(7), list.Quote(llmItem("under", nil), Quantities{InputTokens: 1}).CostMicros)
	s.Equal(int64(6_255), list.Quote(llmItem("gpt-5.6-terra", nil),
		Quantities{InputTokens: 17, CacheWriteInputTokens: 2_373, OutputTokens: 24}).CostMicros)
}

func (s *PricingSuite) TestOutputCostCoversEveryOutputMeter() {
	list := s.list(realtime)
	item := Item{Modality: "sts", Provider: "openai", Model: "gpt-realtime-2.1", Region: RegionUS}

	quote := list.Quote(item, Quantities{InputTextTokens: 1_000, OutputTextTokens: 1_000, OutputAudioTokens: 1_000})

	s.Equal(int64(24_000+64_000), quote.OutputCostMicros)
	s.Equal(int64(4_000+24_000+64_000), quote.CostMicros)
}

func (s *PricingSuite) TestAFreeOrLocalModelWithoutAPriceIsNotBillable() {
	free := llmEntry("free", nil)
	free.Free = true
	local := llmEntry("local", nil)
	local.Local = true
	list := s.list(free, local)

	for _, model := range []string{"free", "local"} {
		quote := list.Quote(llmItem(model, nil), Quantities{InputTokens: 1_000})
		s.False(quote.Billable, model)
		s.False(quote.Unpriced, model)
		s.Zero(quote.CostMicros, model)
		s.False(list.Billable(llmItem(model, nil)), model)
	}
	s.Empty(s.logs.String())
}

func (s *PricingSuite) TestALocalModelWithAPriceIsBillable() {
	local := llmEntry("local", map[Meter]Rate{InputTokens: perMillion(1)})
	local.Local = true
	list := s.list(local)

	s.True(list.Billable(llmItem("local", nil)))
	s.Equal(int64(1), list.Quote(llmItem("local", nil), Quantities{InputTokens: 1}).CostMicros)
}

func (s *PricingSuite) TestAModelWithoutAPriceIsBillableAndLoggedOnce() {
	list := s.list()
	item := Item{Modality: "memory", Provider: "mem0", Model: "search", Region: RegionUS}

	first := list.Quote(item, Quantities{})
	list.Quote(item, Quantities{})

	s.True(first.Billable)
	s.True(first.Unpriced)
	s.Zero(first.CostMicros)
	s.Equal(1, strings.Count(s.logs.String(), "billable work has no price"))
	s.Contains(s.logs.String(), "modality=memory provider=mem0 model=search")
}

func (s *PricingSuite) TestAModelWithoutAPriceIsLoggedAgainOnceTheIntervalHasPassed() {
	list := s.list()
	clock := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	list.unpriced.now = func() time.Time { return clock }
	list.unpriced.every = time.Minute
	item := Item{Modality: "memory", Provider: "mem0", Model: "search", Region: RegionUS}

	list.Quote(item, Quantities{})
	clock = clock.Add(time.Minute - time.Nanosecond)
	list.Quote(item, Quantities{})
	s.Equal(1, strings.Count(s.logs.String(), "billable work has no price"), "a repeat within the interval")

	clock = clock.Add(time.Nanosecond)
	list.Quote(item, Quantities{})
	s.Equal(2, strings.Count(s.logs.String(), "billable work has no price"), "a repeat once the interval has passed")

	clock = clock.Add(time.Second)
	list.Quote(item, Quantities{})
	s.Equal(2, strings.Count(s.logs.String(), "billable work has no price"), "the interval runs from the last log")
}

func (s *PricingSuite) TestAnEntryHandedOutIsACopy() {
	entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)})
	entry.When = map[string]string{"service_tier": "flex"}
	entry.Tiers = []Tier{{AboveInputTokens: 1_000, Rates: map[Meter]Rate{InputTokens: perMillion(2)}}}
	entry.SKU = "m-flex"
	entry.Prices = map[Meter]Rate{InputTokens: perMillion(1)}
	entry.PriceTiers = []Tier{{AboveInputTokens: 1_000, Rates: map[Meter]Rate{InputTokens: perMillion(2)}}}
	list := s.list(entry)
	item := llmItem("m", map[string]string{"service_tier": "flex"})

	handed, ok := list.Entry(item)
	s.Require().True(ok)
	handed.Rates[InputTokens] = perMillion(100)
	handed.Tiers[0].Rates[InputTokens] = perMillion(100)
	handed.Prices[InputTokens] = perMillion(100)
	handed.PriceTiers[0].Rates[InputTokens] = perMillion(100)
	handed.PriceTiers[0].AboveInputTokens = 1
	handed.When["service_tier"] = "warp"

	s.Equal(int64(1_000), list.Quote(item, Quantities{InputTokens: 1_000}).CostMicros)
	s.Equal(int64(2_002), list.Quote(item, Quantities{InputTokens: 1_001}).CostMicros)
	again, _ := list.Entry(item)
	s.Equal(entry, again)
}

// TestAnEntryCanBeChangedWhileTheListQuotes is for the race detector: rows are priced from
// many goroutines while another caller holds an entry.
func (s *PricingSuite) TestAnEntryCanBeChangedWhileTheListQuotes() {
	list := s.list(llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)}))
	done := make(chan int64)
	go func() {
		var cost int64
		for range 1_000 {
			cost = list.Quote(llmItem("m", nil), Quantities{InputTokens: 1_000}).CostMicros
		}
		done <- cost
	}()

	for range 1_000 {
		handed, _ := list.Entry(llmItem("m", nil))
		handed.Rates[InputTokens] = perMillion(100)
	}

	s.Equal(int64(1_000), <-done)
}

func (s *PricingSuite) TestUnpricedListsTheBillableItemsWithNoRate() {
	free := llmEntry("free", nil)
	free.Free = true
	list := s.list(free, llmEntry("priced", map[Meter]Rate{InputTokens: perMillion(1)}), llmEntry("forgotten", nil))

	unpriced := list.Unpriced([]Item{llmItem("free", nil), llmItem("priced", nil), llmItem("forgotten", nil), llmItem("unknown", nil)})

	s.Equal([]Item{llmItem("forgotten", nil), llmItem("unknown", nil)}, unpriced)
}

func (s *PricingSuite) TestNewListRefusesTwoEntriesForTheSameItem() {
	_, err := NewList("test", atCost(terra, terra), nil)

	s.ErrorContains(err, "two entries")
}

func (s *PricingSuite) TestNewListRefusesTwoEntriesOneItemWouldMatchEqually() {
	ca := llmEntry("m", nil)
	ca.When = map[string]string{"country": "CA"}
	flex := llmEntry("m", nil)
	flex.When = map[string]string{"service_tier": "flex"}

	_, err := NewList("test", []Entry{ca, flex}, nil)

	s.ErrorContains(err, "two entries")
}

func (s *PricingSuite) TestNewListRefusesARateWithNoUnit() {
	_, err := NewList("test", []Entry{llmEntry("m", map[Meter]Rate{InputTokens: {USD: 1}})}, nil)

	s.ErrorContains(err, "a rate needs a unit")
}

func (s *PricingSuite) TestNewListRefusesAnUnknownMeter() {
	_, err := NewList("test", []Entry{llmEntry("m", map[Meter]Rate{"tokens": perMillion(1)})}, nil)

	s.ErrorContains(err, "not a meter")
}

func (s *PricingSuite) TestNewListRefusesANegativeRate() {
	_, err := NewList("test", []Entry{llmEntry("m", map[Meter]Rate{InputTokens: perMillion(-1)})}, nil)

	s.ErrorContains(err, "below zero")
}

func (s *PricingSuite) TestNewListRefusesATierMissingARate() {
	entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1), OutputTokens: perMillion(4)})
	entry.Tiers = []Tier{{AboveInputTokens: 1_000, Rates: map[Meter]Rate{InputTokens: perMillion(2)}}}

	_, err := NewList("test", []Entry{entry}, nil)

	s.ErrorContains(err, "no output_tokens rate")
}

func (s *PricingSuite) TestNewListRefusesTiersOutOfOrder() {
	rates := map[Meter]Rate{InputTokens: perMillion(1)}
	entry := llmEntry("m", rates)
	entry.Tiers = []Tier{{AboveInputTokens: 2_000, Rates: rates}, {AboveInputTokens: 1_000, Rates: rates}}

	_, err := NewList("test", []Entry{entry}, nil)

	s.ErrorContains(err, "not above the one before it")
}

func (s *PricingSuite) TestNewListRefusesAnEntryWithNoModel() {
	_, err := NewList("test", []Entry{{Modality: "llm", Provider: "openai"}}, nil)

	s.ErrorContains(err, "needs a modality, a provider and a model")
}

func (s *PricingSuite) TestAnEntryPricesWithoutAList() {
	quote := atCost(terra)[0].Quote(Quantities{InputTokens: 3, CacheWriteInputTokens: 2_386, OutputTokens: 22})

	s.Equal(int64(6_235), quote.CostMicros)
	s.True(quote.Billable)
	s.Empty(quote.Version)
}

func (s *PricingSuite) TestAVersionFollowsOneRule() {
	written := llmEntry("written", map[Meter]Rate{InputTokens: perMillion(1)})
	written.Version = "20261110130000_seed"
	listed := llmEntry("listed", map[Meter]Rate{InputTokens: perMillion(1)})
	local := llmEntry("local", nil)
	local.Local = true
	localRated := llmEntry("local-rated", map[Meter]Rate{InputTokens: perMillion(0.1)})
	localRated.Local, localRated.Version = true, "20261110140000_local_rate"
	list := s.list(written, listed, local, localRated)
	used := Quantities{InputTokens: 1_000}

	s.Equal("20261110130000_seed", list.Quote(llmItem("written", nil), used).Version, "the change that wrote the entry")
	s.Equal("test", list.Quote(llmItem("listed", nil), used).Version, "the list's own version")
	s.Empty(list.Quote(llmItem("local", nil), used).Version, "a local entry with no rate is priced by no change")
	s.Equal("20261110140000_local_rate", list.Quote(llmItem("local-rated", nil), used).Version,
		"a local entry with a rate is priced by the change that wrote it")
	s.Empty(list.Quote(llmItem("unknown", nil), used).Version, "unpriced work is priced by nothing")
}

func (s *PricingSuite) TestARateDearerThanAnyUnitIsRefused() {
	_, err := NewList("test", []Entry{llmEntry("m", map[Meter]Rate{OutputTokens: {USD: 1_000_000_001, Per: 1_000_000}})}, nil)
	s.ErrorContains(err, "more than $1000 a unit")

	_, err = NewList("test", []Entry{llmEntry("m", map[Meter]Rate{OutputTokens: {USD: math.Inf(1), Per: 1}})}, nil)
	s.ErrorContains(err, "more than $1000 a unit")

	_, err = NewList("test", atCost(llmEntry("m", map[Meter]Rate{OutputTokens: {USD: 1_000, Per: 1}})), nil)
	s.NoError(err, "a thousand dollars a unit is the most, and allowed")

	priced := llmEntry("m", map[Meter]Rate{OutputTokens: perMillion(1)})
	priced.Prices = map[Meter]Rate{OutputTokens: {USD: 1_001, Per: 1}}
	_, err = NewList("test", []Entry{priced}, nil)
	s.ErrorContains(err, "more than $1000 a unit", "a price is bounded as a rate is")
}

// TestAQuoteChargesThePriceAndCostsTheRates: the price is what the work is charged, and the
// vendor's rates only say what it cost, as BaseCostMicros.
func (s *PricingSuite) TestAQuoteChargesThePriceAndCostsTheRates() {
	entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1), OutputTokens: perMillion(4)})
	entry.Prices = map[Meter]Rate{InputTokens: perMillion(1.5), OutputTokens: perMillion(5)}
	entry.SKU = "llm-openai-m"
	list := s.list(entry)

	quote := list.Quote(llmItem("m", nil), Quantities{InputTokens: 1_000_000, OutputTokens: 1_000_000})

	s.Equal(int64(6_500_000), quote.CostMicros)
	s.Equal(int64(5_000_000), quote.OutputCostMicros)
	s.Equal(int64(5_000_000), quote.BaseCostMicros)
	s.Equal("llm-openai-m", quote.SKU)
	s.True(quote.Billable)
}

// TestALongPromptPicksTheSameTierOfPricesAndOfRates: a prompt past a threshold is charged at
// that tier's prices and costed at that tier's rates.
func (s *PricingSuite) TestALongPromptPicksTheSameTierOfPricesAndOfRates() {
	entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)})
	entry.Tiers = []Tier{{AboveInputTokens: 1_000, Rates: map[Meter]Rate{InputTokens: perMillion(2)}}}
	entry.Prices = map[Meter]Rate{InputTokens: perMillion(3)}
	entry.PriceTiers = []Tier{{AboveInputTokens: 1_000, Rates: map[Meter]Rate{InputTokens: perMillion(7)}}}
	list := s.list(entry)

	short := list.Quote(llmItem("m", nil), Quantities{InputTokens: 1_000})
	long := list.Quote(llmItem("m", nil), Quantities{InputTokens: 2_000})

	s.Equal(int64(3_000), short.CostMicros)
	s.Equal(int64(1_000), short.BaseCostMicros)
	s.Equal(int64(14_000), long.CostMicros)
	s.Equal(int64(4_000), long.BaseCostMicros)
}

func (s *PricingSuite) TestAnEntryWithRatesAndNoPriceIsRefused() {
	_, err := NewList("test", []Entry{llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)})}, nil)

	s.ErrorIs(err, ErrPriceMeters)
}

func (s *PricingSuite) TestAPriceWithOtherMetersThanTheRatesIsRefused() {
	entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1), OutputTokens: perMillion(4)})
	entry.Prices = map[Meter]Rate{InputTokens: perMillion(1.2)}

	_, err := NewList("test", []Entry{entry}, nil)

	s.ErrorIs(err, ErrPriceMeters)
	s.ErrorContains(err, "rates input_tokens, output_tokens and prices input_tokens")
}

func (s *PricingSuite) TestPriceTiersOtherThanTheRatesTiersAreRefused() {
	tiered := func(above ...int64) Entry {
		entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1)})
		entry.Prices = map[Meter]Rate{InputTokens: perMillion(1.2)}
		entry.Tiers = []Tier{{AboveInputTokens: 1_000, Rates: map[Meter]Rate{InputTokens: perMillion(2)}}}
		for _, threshold := range above {
			entry.PriceTiers = append(entry.PriceTiers, Tier{AboveInputTokens: threshold, Rates: map[Meter]Rate{InputTokens: perMillion(2.4)}})
		}
		return entry
	}

	_, err := NewList("test", []Entry{tiered()}, nil)
	s.ErrorIs(err, ErrPriceTiers, "no price tier for a tier of rates")
	_, err = NewList("test", []Entry{tiered(2_000)}, nil)
	s.ErrorIs(err, ErrPriceTiers, "a price tier at another threshold")
	_, err = NewList("test", []Entry{tiered(1_000, 2_000)}, nil)
	s.ErrorIs(err, ErrPriceTiers, "a price tier more")
	_, err = NewList("test", []Entry{tiered(1_000)}, nil)
	s.NoError(err)

	extra := tiered(1_000)
	extra.Tiers[0].Rates[OutputTokens] = perMillion(5)
	_, err = NewList("test", []Entry{extra}, nil)
	s.ErrorIs(err, ErrPriceTiers, "a tier of rates with a meter its price tier lacks")
}

func (s *PricingSuite) TestAFreeEntryWithAPriceIsRefused() {
	free := llmEntry("m", nil)
	free.Free = true
	free.Prices = map[Meter]Rate{InputTokens: perMillion(1)}

	_, err := NewList("test", []Entry{free}, nil)

	s.ErrorContains(err, "is free and has a price")
}

// TestPricedIsTheCostTimesTheFactor: an entry written without a price is priced at its
// vendor's rates times PriceFactorBps, to twelve significant digits.
func (s *PricingSuite) TestPricedIsTheCostTimesTheFactor() {
	entry := llmEntry("m", map[Meter]Rate{InputTokens: perMillion(1.375), OutputTokens: {USD: 1.1, Per: 1_000_000, Increment: 10}})
	entry.Tiers = []Tier{{AboveInputTokens: 1_000, Rates: map[Meter]Rate{InputTokens: perMillion(4), OutputTokens: perMillion(1)}}}

	priced := Priced(entry)

	s.Equal(map[Meter]Rate{InputTokens: perMillion(1.65), OutputTokens: {USD: 1.32, Per: 1_000_000, Increment: 10}}, priced.Prices,
		"1.1 × 1.2 reads 1.32, not 1.3200000000000003")
	s.Equal([]Tier{{AboveInputTokens: 1_000, Rates: map[Meter]Rate{InputTokens: perMillion(4.8), OutputTokens: perMillion(1.2)}}}, priced.PriceTiers)
	s.Equal(entry.Rates, priced.Rates, "the vendor's rates stay as they were")
	s.Nil(entry.Prices, "the entry handed in is not changed")
	s.Equal(int64(12_000), int64(PriceFactorBps))

	own := entry
	own.Prices = map[Meter]Rate{InputTokens: perMillion(9), OutputTokens: perMillion(9)}
	s.Equal(own, Priced(own), "a price of its own is kept")

	free := llmEntry("free", nil)
	free.Free = true
	s.Equal(free, Priced(free), "a free entry carries no price")
}

func (s *PricingSuite) TestAQuotedAmountIsResoldAtTheFactor() {
	number := Item{Modality: "phone", Provider: "telnyx", Model: "number", Region: RegionUS}
	list := s.list(Entry{Modality: "phone", Provider: "telnyx", Model: "number", Region: RegionUS, SKU: "phone-telnyx-number-us", Free: true})

	quote := list.Quote(number, Quantities{})
	s.False(quote.Billable)
	resold := list.Quoted(number, 1_000_003)

	s.Equal(int64(1_200_004), resold.CostMicros, "1,000,003 × 1.2 = 1,200,003.6, rounded once")
	s.Equal(int64(1_000_003), resold.BaseCostMicros)
	s.Equal("phone-telnyx-number-us", resold.SKU)
	s.Equal(QuotedVersion, resold.Version)
	s.True(resold.Billable)
	s.False(list.Quoted(number, 0).Billable)
	s.Empty(list.Quoted(Item{Modality: "phone", Provider: "telnyx", Model: "other", Region: RegionUS}, 10).SKU)
}

// TestAQuotedAmountForABYOKItemIsChargedTheFeeAndCostsNothing: the vendor bills the customer
// what it quoted, so the operator pays nothing and charges BYOKFeeBps of it.
func (s *PricingSuite) TestAQuotedAmountForABYOKItemIsChargedTheFeeAndCostsNothing() {
	list := s.list()

	resold := list.Quoted(Item{Modality: "llm", Provider: "openai", Model: "m", Region: RegionUS, BYOK: true}, 1_000_000)

	s.Equal(int64(200_000), resold.CostMicros)
	s.Zero(resold.BaseCostMicros)
}

// TestEveryEntryOfAListHasASKU: an entry without one is given its BaseSKU, numbered past one
// another entry has.
func (s *PricingSuite) TestEveryEntryOfAListHasASKU() {
	flex := llmEntry("GPT-4.1", map[Meter]Rate{InputTokens: perMillion(1)})
	flex.When = map[string]string{"service_tier": "flex"}
	taken := llmEntry("taken", map[Meter]Rate{InputTokens: perMillion(1)})
	taken.SKU = "llm-openai-gpt-4-1-us"
	list := s.list(llmEntry("GPT-4.1", map[Meter]Rate{InputTokens: perMillion(1)}), flex, taken)

	plain, _ := list.Entry(llmItem("GPT-4.1", nil))
	tiered, _ := list.Entry(llmItem("GPT-4.1", map[string]string{"service_tier": "flex"}))
	s.Equal("llm-openai-gpt-4-1-us-2", plain.SKU, "the base is another entry's")
	s.Equal("llm-openai-gpt-4-1-service-tier-flex-us", tiered.SKU)
	s.Equal("sku", BaseSKU("--", "", "", "", false, nil))
}

// TestASKUEndsInItsRegionThenBYOK: the attributes of when come first, then the region, then
// byok for a BYOK item.
func (s *PricingSuite) TestASKUEndsInItsRegionThenBYOK() {
	batch := map[string]string{"service_tier": "batch"}

	s.Equal("llm-anthropic-claude-haiku-5-5-us", BaseSKU("llm", "anthropic", "claude-haiku-5.5", RegionUS, false, nil))
	s.Equal("llm-anthropic-claude-haiku-5-5-us-byok", BaseSKU("llm", "anthropic", "claude-haiku-5.5", RegionUS, true, nil))
	s.Equal("llm-anthropic-claude-haiku-5-5-eu", BaseSKU("llm", "anthropic", "claude-haiku-5.5", RegionEU, false, nil))
	s.Equal("llm-openai-gpt-6-luna-service-tier-batch-us", BaseSKU("llm", "openai", "gpt-6-luna", RegionUS, false, batch))
	s.Equal("llm-openai-gpt-6-luna-service-tier-batch-eu-byok", BaseSKU("llm", "openai", "gpt-6-luna", RegionEU, true, batch))
	s.Equal("knowledge-turbopuffer-write-us-byok", BaseSKU("knowledge", "turbopuffer", "write", RegionUS, true, nil))
	s.Equal("knowledge-turbopuffer-write-us-byok-2", NumberedSKU(BaseSKU("knowledge", "turbopuffer", "write", RegionUS, true, nil), 2))
}

func (s *PricingSuite) TestTwoEntriesWithOneSKUAreRefused() {
	a, b := llmEntry("a", nil), llmEntry("b", nil)
	a.SKU, b.SKU = "one", "one"

	_, err := NewList("test", []Entry{a, b}, nil)

	s.ErrorContains(err, "both have the SKU one")
}

func (s *PricingSuite) TestASKUOfAnotherFormIsRefused() {
	for _, sku := range []string{"Upper", "two--hyphens", "-lead", "trail-", "under_score", strings.Repeat("a", MaxSKULength+1)} {
		entry := llmEntry("m", nil)
		entry.SKU = sku
		_, err := NewList("test", []Entry{entry}, nil)
		s.ErrorContains(err, "a SKU is 1 to 255 characters", sku)
	}
	entry := llmEntry("m", nil)
	entry.SKU = "gpt-4-1-2026"
	_, err := NewList("test", []Entry{entry}, nil)
	s.NoError(err)
}
