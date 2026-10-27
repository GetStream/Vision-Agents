package routing

import (
	"bytes"
	"io"
	"log/slog"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// quoted is what a row recording usage on the provider is priced at, as Record prices it.
func quoted(pricer pricing.Pricer, modality Modality, provider ProviderConfig, usage Usage) pricing.Quote {
	return quote(pricer, modality, provider, nil, row(modality, provider, Stat{Usage: usage}).Quantities())
}

// atCost is a config's price entries each priced at its vendor rates, so what a rate charges
// is what it costs.
func atCost(config Config) []pricing.Entry {
	return pricing.AllAtCost(PriceEntries(config))
}

func (s *RoutingSuite) TestPriceEntriesExpandServiceAndContextTiers() {
	provider := ProviderConfig{Provider: "openai", Model: "gpt", Price: Price{
		PerMillionInputTokens:  2,
		PerMillionOutputTokens: 8,
		ContextTiers:           []ContextTier{{AboveInputTokens: 272_000, Input: 2, Output: 1.5}},
		ServiceTiers:           map[string]float64{"flex": 0.5, "fast": 2},
	}}

	entries := PriceEntries(Config{LLM: {Providers: []ProviderConfig{provider}}})

	s.Require().Len(entries, 3)
	rates := func(input, output float64) map[pricing.Meter]pricing.Rate {
		return map[pricing.Meter]pricing.Rate{
			pricing.InputTokens:  {USD: input, Per: 1_000_000},
			pricing.OutputTokens: {USD: output, Per: 1_000_000},
		}
	}
	tiers := func(input, output float64) []pricing.Tier {
		return []pricing.Tier{{AboveInputTokens: 272_000, Rates: rates(input, output)}}
	}
	s.Equal(pricing.Entry{Modality: "llm", Provider: "openai", Model: "gpt", Rates: rates(2, 8), Tiers: tiers(4, 12)}, entries[0])
	s.Equal(pricing.Entry{Modality: "llm", Provider: "openai", Model: "gpt", When: map[string]string{"service_tier": "fast"},
		Rates: rates(4, 16), Tiers: tiers(8, 24)}, entries[1])
	s.Equal(pricing.Entry{Modality: "llm", Provider: "openai", Model: "gpt", When: map[string]string{"service_tier": "flex"},
		Rates: rates(1, 4), Tiers: tiers(2, 6)}, entries[2])
}

func (s *RoutingSuite) TestAServiceTierThePriceListDoesNotNameIsPricedAtTheDefaultRates() {
	provider := ProviderConfig{Provider: "openai", Model: "gpt", Price: Price{
		PerMillionInputTokens: 2,
		ServiceTiers:          map[string]float64{"flex": 0.5},
	}}
	list, err := pricing.NewList("config", atCost(Config{LLM: {Providers: []ProviderConfig{provider}}}), nil)
	s.Require().NoError(err)
	used := row(LLM, provider, Stat{Usage: Usage{InputTokens: 1_000_000}}).Quantities()

	s.EqualValues(1_000_000, quote(list, LLM, provider, map[string]string{"service_tier": "flex"}, used).CostMicros)
	s.EqualValues(2_000_000, quote(list, LLM, provider, map[string]string{"service_tier": "default"}, used).CostMicros)
}

func (s *RoutingSuite) TestATierThePriceListDoesNotNameIsLoggedOnce() {
	provider := ProviderConfig{Provider: "openai", Model: "gpt", Price: Price{
		PerMillionInputTokens: 2,
		ServiceTiers:          map[string]float64{"flex": 0.5},
	}}
	list, err := pricing.NewList("config", atCost(Config{LLM: {Providers: []ProviderConfig{provider}}}), nil)
	s.Require().NoError(err)
	logs := &bytes.Buffer{}
	// Its writer runs no stage, so every stat stays in the queue rather than being written.
	writer := &Writer{live: &live.Client{}, logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		queue: make(chan queued, 16), queueSize: 16}
	recorder := &Recorder{modality: LLM, writer: writer, pricer: list, logger: slog.New(slog.NewTextHandler(logs, nil))}

	for _, tier := range []string{"turbo", "turbo", "flex", "default", ""} {
		recorder.Record(provider, Stat{Usage: Usage{InputTokens: 1_000}, Attributes: map[string]string{"service_tier": tier}})
	}

	s.Equal(1, strings.Count(logs.String(), "does not name"), "one line per model and tier, and none for a tier it names")
	s.Contains(logs.String(), "service_tier=turbo")
	s.Contains(logs.String(), "model=gpt")
}

// TestABuiltInModelsTiersAreNamedByTheVendorFile: the built-in config sets no price, so the
// tiers a model is priced at are the vendor file's entries, and only a tier with none is said.
func (s *RoutingSuite) TestABuiltInModelsTiersAreNamedByTheVendorFile() {
	config, list := s.defaultPrices()
	var luna ProviderConfig
	for _, provider := range config[LLM].Providers {
		if provider.Name() == "openai/gpt-6-luna" {
			luna = provider
		}
	}
	s.Require().Equal("gpt-6-luna", luna.Model)
	logs := &bytes.Buffer{}
	// Its writer runs no stage, so every stat stays in the queue rather than being written.
	writer := &Writer{live: &live.Client{}, logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		queue: make(chan queued, 16), queueSize: 16}
	recorder := &Recorder{modality: LLM, writer: writer, pricer: list, logger: slog.New(slog.NewTextHandler(logs, nil))}

	for _, tier := range []string{"flex", "batch", "fast", "priority", "default"} {
		recorder.Record(luna, Stat{Usage: Usage{InputTokens: 1_000}, Attributes: map[string]string{"service_tier": tier}})
	}
	s.Empty(logs.String(), "every tier vendor.json names is priced at its own rates, silently")

	recorder.Record(luna, Stat{Usage: Usage{InputTokens: 1_000}, Attributes: map[string]string{"service_tier": "ultrafast"}})

	s.Equal(1, strings.Count(logs.String(), "does not name"))
	s.Contains(logs.String(), "service_tier=ultrafast")
}

func (s *RoutingSuite) TestConfigRejectsAServiceTierAtNoRate() {
	config := s.config()
	config.Providers[0].Price = Price{PerMillionInputTokens: 1, ServiceTiers: map[string]float64{"flex": 0}}

	s.ErrorContains(config.Validate(), "quick/en prices service tier flex at 0 times its rates")
}

func (s *RoutingSuite) TestTheSelfHostedModelsAreMarkedLocal() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	local := LocalItems(config)

	for _, item := range []pricing.Item{
		{Modality: "stt", Provider: "parakeet", Model: "parakeet-tdt-0.6b-v3"},
		{Modality: "tts", Provider: "s2pro", Model: "s2-pro"},
		{Modality: "tts", Provider: "breeze", Model: "breeze-tts-2"},
		{Modality: "llm", Provider: "gemma", Model: "gemma-4-26B-A4B-it"},
	} {
		s.Contains(local, item)
	}
}

// defaultPrices is the built-in config and the price list it is priced by: the vendor list
// embedded in the binary, with the models the config runs itself marked local.
func (s *RoutingSuite) defaultPrices() (Config, *pricing.List) {
	config, err := DefaultConfig()
	s.Require().NoError(err)
	list, err := prices.Embedded(LocalItems(config), "test", slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	return config, list
}

// TestEveryRoutedModelIsPriced: every provider the built-in config routes to is marked local,
// or has a rate or free in vendor.json.
func (s *RoutingSuite) TestEveryRoutedModelIsPriced() {
	config, list := s.defaultPrices()

	s.Empty(list.Unpriced(Items(config)))
}

func (s *RoutingSuite) TestTheBuiltInConfigSetsNoPrice() {
	config, err := DefaultConfig()
	s.Require().NoError(err)

	for modality, section := range config {
		for _, provider := range section.Providers {
			s.Zerof(provider.Price, "%s/%s sets a price: it goes in prices/vendor.json", modality, provider.Name())
		}
	}
}

func (s *RoutingSuite) TestADefaultListTakesTheConfigsOwnRateThenTheVendorFile() {
	vendor := prices.Vendor().Entries[0]
	own := ProviderConfig{Provider: "own", Model: "model", Price: Price{PerMillionInputTokens: 3}}
	listed := ProviderConfig{Provider: vendor.Provider, Model: vendor.Model}

	list, err := Prices(Config{LLM: {Providers: []ProviderConfig{own}}, Modality(vendor.Modality): {Providers: []ProviderConfig{listed}}},
		slog.New(slog.DiscardHandler))
	s.Require().NoError(err)

	fromConfig := quoted(list, LLM, own, Usage{InputTokens: 1_000_000})
	s.EqualValues(3_000_000, fromConfig.CostMicros, "the config's own rate is charged as it is written")
	s.Equal("config", fromConfig.Version)
	found, ok := list.Entry(ItemOf(Modality(vendor.Modality), listed, nil))
	s.Require().True(ok)
	s.Equal(vendor.Rates, found.Rates)
	s.Equal(vendor.Prices, found.Prices, "the vendor file's entry is priced as the file is read, at the factor")
	s.Equal(pricing.Priced(pricing.Entry{Rates: vendor.Rates, Tiers: vendor.Tiers}).Prices, found.Prices)
	s.Equal("embedded", quoted(list, Modality(vendor.Modality), listed, Usage{InputTokens: 1_000_000}).Version)
}

func (s *RoutingSuite) TestACustomersOwnModelIsPricedFromItsOwnConfig() {
	own := ProviderConfig{Provider: CustomProvider, Model: "mine", Price: Price{PerMillionInputTokens: 1, PerMillionOutputTokens: 2}}
	list, err := pricing.NewList("config", nil, nil)
	s.Require().NoError(err)

	quote := quoted(list, LLM, own, Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000})

	s.EqualValues(3_000_000, quote.CostMicros, "at what the customer's host bills, with nothing added")
	s.EqualValues(3_000_000, quote.BaseCostMicros)
	s.EqualValues(2_000_000, quote.OutputCostMicros)
	s.False(quote.Billable, "the customer pays their own host")
	s.False(quote.Unpriced)
}

func (s *RoutingSuite) TestACustomersOwnModelWithNoRatesIsFree() {
	logs := &bytes.Buffer{}
	list, err := pricing.NewList("config", nil, slog.New(slog.NewTextHandler(logs, nil)))
	s.Require().NoError(err)

	quote := quoted(list, LLM, ProviderConfig{Provider: CustomProvider, Model: "mine"}, Usage{InputTokens: 1_000, OutputTokens: 1_000})

	s.Zero(quote.CostMicros)
	s.False(quote.Billable)
	s.False(quote.Unpriced)
	s.Empty(logs.String())
}

func (s *RoutingSuite) TestARowPricesTheMetersItStores() {
	provider := ProviderConfig{Provider: "openai", Model: "gpt", Price: Price{
		PerMillionInputTokens:           2,
		PerMillionCacheWriteInputTokens: 2.5,
		PerMegapixel:                    0.02,
	}}
	list, err := pricing.NewList("config", atCost(Config{LLM: {Providers: []ProviderConfig{provider}}}), nil)
	s.Require().NoError(err)

	request := row(LLM, provider, Stat{
		Usage:  Usage{InputTokens: 2_389, Pixels: 1_000_000},
		Meters: pricing.Quantities{pricing.CacheWriteInputTokens: 2_386},
	})

	s.Equal(pricing.Quantities{pricing.CacheWriteInputTokens: 2_386, pricing.Pixels: 1_000_000}, request.Meters)
	s.EqualValues(6+5_965+20_000, quote(list, LLM, provider, nil, request.Quantities()).CostMicros,
		"3 fresh tokens, 2,386 written to the cache and a megapixel")
}

func (s *RoutingSuite) TestCostIsZeroWithoutAPrice() {
	s.Zero(s.costOf(Price{}, Usage{Characters: 1000, AudioMs: 60_000, InputTokens: 500, OutputTokens: 500}))
}

func (s *RoutingSuite) TestCostPricesCharacters() {
	// A million characters at $50 per million is $50, which is 50 million micros.
	price := Price{PerMillionChars: 50}

	s.EqualValues(50_000_000, s.costOf(price, Usage{Characters: 1_000_000}))
	s.EqualValues(50_000, s.costOf(price, Usage{Characters: 1_000}), "a thousand characters is five cents")
}

func (s *RoutingSuite) TestCostPricesAudioByTheHour() {
	// An hour of audio at $0.36 per hour is 360000 micros; a minute is a sixtieth of that.
	price := Price{PerAudioHour: 0.36}

	s.EqualValues(360_000, s.costOf(price, Usage{AudioMs: 3_600_000}))
	s.EqualValues(6_000, s.costOf(price, Usage{AudioMs: 60_000}))
}

func (s *RoutingSuite) TestCostPricesTokensByDirection() {
	// DeepSeek Flash: $0.13 per million in, $0.26 per million out.
	price := Price{PerMillionInputTokens: 0.13, PerMillionOutputTokens: 0.26}

	s.EqualValues(130_000, s.costOf(price, Usage{InputTokens: 1_000_000}))
	s.EqualValues(260_000, s.costOf(price, Usage{OutputTokens: 1_000_000}))
	s.EqualValues(390_000, s.costOf(price, Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000}),
		"output is dearer than input, so the direction has to be tracked separately")
}

func (s *RoutingSuite) TestCostBillsCachedPromptTokensOnlyOnce() {
	// $1 per million fresh, a tenth of that cached. Half of a million-token prompt was
	// cached, so the bill is half a dollar plus five cents rather than the full dollar.
	price := Price{PerMillionInputTokens: 1, PerMillionCachedInputTokens: 0.1}

	s.EqualValues(550_000, s.costOf(price, Usage{InputTokens: 1_000_000, CachedInputTokens: 500_000}))
	s.EqualValues(100_000, s.costOf(price, Usage{InputTokens: 1_000_000, CachedInputTokens: 1_000_000}),
		"a prompt served entirely from cache costs only the cached rate")
}

func (s *RoutingSuite) TestCostIgnoresMoreCachedTokensThanPromptTokens() {
	// A provider that reported nonsense should not produce a negative bill.
	price := Price{PerMillionInputTokens: 1, PerMillionCachedInputTokens: 0.1}

	s.EqualValues(100_000, s.costOf(price, Usage{InputTokens: 100, CachedInputTokens: 1_000_000}))
}

func (s *RoutingSuite) TestCostAddsEveryUnitTheModelBillsFor() {
	price := Price{PerMillionChars: 50, PerAudioHour: 0.36}

	s.EqualValues(410_000, s.costOf(price, Usage{Characters: 1_000, AudioMs: 3_600_000}),
		"an hour of audio plus a thousand characters")
}

func (s *RoutingSuite) TestCostPricesImagesByThePicture() {
	// Qwen Image 3 on FAL: $0.04 a picture, whatever its size.
	price := Price{PerImage: 0.04}

	s.EqualValues(40_000, s.costOf(price, Usage{Images: 1, Pixels: 1024 * 1024}))
	s.EqualValues(120_000, s.costOf(price, Usage{Images: 3, Pixels: 3 * 2048 * 2048}),
		"three pictures cost three times one, however large they are")
}

func (s *RoutingSuite) TestCostPricesImagesByThePixelsThatCameBack() {
	// $0.02 a megapixel, a million pixels to the megapixel.
	price := Price{PerMegapixel: 0.02}

	s.EqualValues(20_000, s.costOf(price, Usage{Images: 1, Pixels: 1_000_000}))
	s.EqualValues(83_886, s.costOf(price, Usage{Images: 1, Pixels: 2048 * 2048}),
		"a larger picture costs more even though it is still one picture")
	s.Zero(s.costOf(price, Usage{Images: 0, Pixels: 0}), "nothing drawn is nothing billed")
}

func (s *RoutingSuite) TestCostPricesASearchByTheRequest() {
	price := Price{PerThousandRequests: 5}

	s.EqualValues(5_000, s.costOf(price, Usage{Requests: 1}))
}

// costOf is what a row with that usage costs on a provider with that price.
func (s *RoutingSuite) costOf(price Price, usage Usage) int64 {
	provider := ProviderConfig{Provider: "vendor", Model: "model", Price: price}
	list, err := pricing.NewList("config", atCost(Config{LLM: {Providers: []ProviderConfig{provider}}}),
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
	return quoted(list, LLM, provider, usage).CostMicros
}
