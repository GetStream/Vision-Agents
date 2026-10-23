package routing

import (
	"bytes"
	"io"
	"log/slog"
	"math/rand/v2"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// legacyCostMicros is how a row was priced before the price list: every rate summed in
// dollars and truncated to micros. The price list must agree with it within a micro.
func legacyCostMicros(p Price, usage Usage) int64 {
	freshInputTokens := usage.InputTokens - usage.CachedInputTokens
	if freshInputTokens < 0 {
		freshInputTokens = 0
	}

	dollars := p.PerMillionChars*float64(usage.Characters)/1_000_000 +
		p.PerAudioHour*float64(usage.AudioMs)/3_600_000 +
		p.PerMillionInputTokens*float64(freshInputTokens)/1_000_000 +
		p.PerMillionCachedInputTokens*float64(usage.CachedInputTokens)/1_000_000 +
		p.PerMillionOutputTokens*float64(usage.OutputTokens)/1_000_000 +
		p.PerImage*float64(usage.Images) +
		p.PerMegapixel*float64(usage.Pixels)/1_000_000
	return int64(dollars * 1_000_000)
}

// atItsContextTier is the price a prompt of that length is billed at: the legacy price knew
// no long-prompt tier, so a usage past one is compared with the rates the tier multiplies.
func atItsContextTier(p Price, usage Usage) Price {
	billed := p
	for _, tier := range p.ContextTiers {
		if usage.InputTokens <= tier.AboveInputTokens {
			continue
		}
		billed = p
		billed.PerMillionInputTokens *= tier.Input
		billed.PerMillionCachedInputTokens *= tier.Input
		billed.PerMillionCacheWriteInputTokens *= tier.Input
		billed.PerMillionOutputTokens *= tier.Output
	}
	return billed
}

// legacyRequestMicros is how a search was priced before: the rate per thousand set outright.
func legacyRequestMicros(p Price) int64 {
	return int64(p.PerThousandRequests * 1_000)
}

// legacyOutputCostMicros is the part of a legacy cost the generated tokens were priced at.
func legacyOutputCostMicros(p Price, usage Usage) int64 {
	return min(legacyCostMicros(p, Usage{OutputTokens: usage.OutputTokens}), legacyCostMicros(p, usage))
}

// cachedAtTheInputRate are the models with an input rate and no cached one. Their cache hits
// were priced at nothing, and are now priced as fresh input.
var cachedAtTheInputRate = []string{
	"decision_model/openrouter/cloudflare/clef",
	"decision_model/openrouter/cloudflare/clef-flash",
	"decision_model/openrouter/inception/mercury-decide",
	"decision_model/openrouter/jaredpalmer/kev-4b",
	"decision_model/openrouter/liquid/d1",
	"decision_model/openrouter/openai/gpt-6-luna-decisions",
	"decision_model/openrouter/typesafe/jev-1.13",
	"decision_model/perplexity/pplx-decider-v1-27b",
	"decision_model/perplexity/pplx-decider-v1.1-27b",
	"decision_model/typesafe/jev-latest",
	"llm/gemma/gemma-4-26B-A4B-it",
	"llm/moondream/moondream3-preview",
	"llm/together/Qwen/Qwen3.8-Flash",
	"sts/gemini/gemini-3.1-flash-live-preview",
	"sts/gemini/gemini-3.8-live",
	"sts/gemini/gemini-3.8-live-extended-thinking",
}

// defaultPrices is the price list the built-in config declares.
func (s *RoutingSuite) defaultPrices() (Config, *pricing.List) {
	config, err := DefaultConfig()
	s.Require().NoError(err)
	list, err := pricing.NewList("config", PriceEntries(config), slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
	return config, list
}

// quoted is what a row recording usage on the provider is priced at, as Record prices it.
func quoted(pricer pricing.Pricer, modality Modality, provider ProviderConfig, usage Usage) pricing.Quote {
	return quote(pricer, modality, provider, nil, row(modality, provider, Stat{Usage: usage}).Quantities())
}

// requireLegacyCost fails unless the quote is within a micro of the legacy price, cost and
// output cost both.
func (s *RoutingSuite) requireLegacyCost(name string, price Price, usage Usage, quote pricing.Quote) {
	s.InDeltaf(legacyCostMicros(price, usage), quote.CostMicros, 1, "%s on %+v", name, usage)
	s.InDeltaf(legacyOutputCostMicros(price, usage), quote.OutputCostMicros, 1, "%s output on %+v", name, usage)
}

func (s *RoutingSuite) TestPriceEntriesCostWhatPriceCostMicrosDid() {
	config, list := s.defaultPrices()
	cached := Usage{InputTokens: 10_000, CachedInputTokens: 4_000, OutputTokens: 2_500}
	usages := []Usage{
		{},
		{AudioMs: 1_234},
		cached,
		{Characters: 777},
		{Images: 2, Pixels: 2 * 3_000_000},
	}

	var atTheInputRate []string
	for modality, section := range config {
		for _, provider := range section.Providers {
			name := string(modality) + "/" + provider.Name()
			for _, usage := range usages {
				price := atItsContextTier(provider.Price, usage)
				if usage == cached && price.PerMillionInputTokens != 0 && price.PerMillionCachedInputTokens == 0 {
					atTheInputRate = append(atTheInputRate, name)
					price.PerMillionCachedInputTokens = price.PerMillionInputTokens
				}
				s.requireLegacyCost(name, price, usage, quoted(list, modality, provider, usage))
			}
			if provider.Price.PerThousandRequests != 0 {
				s.InDeltaf(legacyRequestMicros(provider.Price), quoted(list, modality, provider, Usage{Requests: 1}).CostMicros, 1,
					"%s per request", name)
			}
		}
	}
	s.ElementsMatch(cachedAtTheInputRate, atTheInputRate,
		"a model with an input rate and no cached one prices its cache hits as input; name a new one here")
}

// TestPriceEntriesCostWhatPriceCostMicrosDidForAnyUsage checks every provider against the
// legacy price on usages drawn at random, from a fixed seed so a failure repeats.
func (s *RoutingSuite) TestPriceEntriesCostWhatPriceCostMicrosDidForAnyUsage() {
	config, list := s.defaultPrices()
	random := rand.New(rand.NewPCG(1, 2))

	for modality, section := range config {
		for _, provider := range section.Providers {
			for range 200 {
				input := random.Int64N(400_000)
				usage := Usage{
					AudioMs:      random.Int64N(10_000_000),
					Characters:   random.Int64N(100_000),
					InputTokens:  input,
					OutputTokens: random.Int64N(50_000),
					Images:       random.Int64N(5),
					Pixels:       random.Int64N(20_000_000),
				}
				price := atItsContextTier(provider.Price, usage)
				if price.PerMillionCachedInputTokens != 0 {
					usage.CachedInputTokens = random.Int64N(input + 1)
				}
				s.requireLegacyCost(string(modality)+"/"+provider.Name(), price, usage, quoted(list, modality, provider, usage))
			}
		}
	}
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
	list, err := pricing.NewList("config", PriceEntries(Config{LLM: {Providers: []ProviderConfig{provider}}}), nil)
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
	list, err := pricing.NewList("config", PriceEntries(Config{LLM: {Providers: []ProviderConfig{provider}}}), nil)
	s.Require().NoError(err)
	logs := &bytes.Buffer{}
	done := make(chan struct{})
	close(done)
	// It keeps every stat in its queue rather than writing it anywhere.
	recorder := &Recorder{modality: LLM, live: &live.Client{}, pricer: list, logger: slog.New(slog.NewTextHandler(logs, nil)),
		queue: make(chan queued, 16), done: done}

	for _, tier := range []string{"turbo", "turbo", "flex", "default", ""} {
		recorder.Record(provider, Stat{Usage: Usage{InputTokens: 1_000}, Attributes: map[string]string{"service_tier": tier}})
	}

	s.Equal(1, strings.Count(logs.String(), "does not name"), "one line per model and tier, and none for a tier it names")
	s.Contains(logs.String(), "service_tier=turbo")
	s.Contains(logs.String(), "model=gpt")
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

func (s *RoutingSuite) TestTheBuiltInConfigHasNoBillableModelWithoutAPrice() {
	config, list := s.defaultPrices()

	s.Empty(list.Unpriced(Items(config)))
}

func (s *RoutingSuite) TestACustomersOwnModelIsPricedFromItsOwnConfig() {
	own := ProviderConfig{Provider: CustomProvider, Model: "mine", Price: Price{PerMillionInputTokens: 1, PerMillionOutputTokens: 2}}
	list, err := pricing.NewList("config", nil, nil)
	s.Require().NoError(err)

	quote := quoted(list, LLM, own, Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000})

	s.EqualValues(3_000_000, quote.CostMicros)
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
	list, err := pricing.NewList("config", PriceEntries(Config{LLM: {Providers: []ProviderConfig{provider}}}), nil)
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
	list, err := pricing.NewList("config", PriceEntries(Config{LLM: {Providers: []ProviderConfig{provider}}}),
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
	return quoted(list, LLM, provider, usage).CostMicros
}
