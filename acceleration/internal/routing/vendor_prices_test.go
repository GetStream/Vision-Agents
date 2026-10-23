package routing

import "github.com/GetStream/Vision-Agents/acceleration/internal/pricing"

// The usages below are what each vendor reported for real calls on 2026-10-20, and the
// costs what its published prices make of them: what the built-in price list must charge.
// Each test also names what the config charged before the vendor's report reached the
// price list, which is the move a deploy shows in the hourly cost per model; that old cost
// was truncated rather than rounded, so it is compared within a micro.

// builtIn is the built-in config's provider for that modality and model, and the price
// list the config declares.
func (s *RoutingSuite) builtIn(modality Modality, name string) (ProviderConfig, *pricing.List) {
	config, list := s.defaultPrices()
	for _, provider := range config[modality].Providers {
		if provider.Name() == name {
			return provider, list
		}
	}
	s.FailNow("the built-in config has no " + name)
	return ProviderConfig{}, nil
}

// charged is what a row recording the stat on the provider costs.
func charged(list *pricing.List, modality Modality, provider ProviderConfig, stat Stat) int64 {
	return quote(list, modality, provider, stat.Attributes, row(modality, provider, stat).Quantities()).CostMicros
}

// realtimeTurn is one reply as OpenAI Realtime reported it: input by modality with its
// cached part, which was all text, and output by modality.
type realtimeTurn struct {
	model                   string
	text, audio, cachedText int64
	outputText, outputAudio int64
	billed, before          int64
}

func (s *RoutingSuite) TestARealtimeTurnCostsWhatOpenAIBills() {
	for _, turn := range []realtimeTurn{
		{"gpt-realtime-2.1", 760, 0, 0, 61, 111, 11_608, 35_328},
		{"gpt-realtime-2.1", 795, 33, 768, 95, 176, 15_015, 19_571},
		{"gpt-realtime-2.1", 844, 33, 768, 61, 101, 9_595, 14_163},
		{"gpt-realtime-2.1", 890, 33, 768, 71, 67, 7_843, 14_099},
		{"gpt-realtime-2", 760, 0, 0, 30, 68, 8_112, 30_592},
		{"gpt-realtime-2", 799, 101, 768, 61, 120, 12_807, 16_115},
		{"gpt-realtime-2", 847, 221, 0, 27, 61, 15_012, 39_808},
		{"gpt-realtime-2", 894, 282, 0, 27, 72, 17_856, 43_968},
	} {
		provider, list := s.builtIn(STS, "openai/"+turn.model)
		totals := Usage{
			InputTokens:       turn.text + turn.audio,
			CachedInputTokens: turn.cachedText,
			OutputTokens:      turn.outputText + turn.outputAudio,
		}
		split := pricing.Quantities{
			pricing.InputTextTokens:       turn.text - turn.cachedText,
			pricing.InputAudioTokens:      turn.audio,
			pricing.CachedInputTextTokens: turn.cachedText,
			pricing.OutputTextTokens:      turn.outputText,
			pricing.OutputAudioTokens:     turn.outputAudio,
		}

		s.EqualValuesf(turn.billed, charged(list, STS, provider, Stat{Usage: totals, Meters: split}), "%+v", turn)
		s.InDeltaf(turn.before, legacyCostMicros(provider.Price, totals), 1, "%+v at the audio rate", turn)
	}
}

// liveTurn is one turn as Gemini Live reported it: the prompt's total and its details by
// modality, the response by modality, and the thinking, outside the response count.
type liveTurn struct {
	model                      string
	prompt, text, audio, image int64
	response, responseText     int64
	responseAudio, thoughts    int64
	billed, before             int64
}

func (s *RoutingSuite) TestAGeminiLiveTurnCostsWhatGoogleBills() {
	for _, turn := range []liveTurn{
		{"gemini-3.8-live-extended-thinking", 3162, 2829, 222, 0, 54, 0, 54, 60, 3_706, 10_134},
		{"gemini-3.8-live-extended-thinking", 3337, 2849, 361, 0, 51, 0, 51, 35, 3_989, 10_623},
		{"gemini-3.8-live-extended-thinking", 3753, 2920, 412, 256, 104, 0, 104, 118, 5_461, 12_507},
		{"gemini-3.8-live-extended-thinking", 3898, 2943, 516, 256, 47, 0, 47, 86, 4_962, 12_258},
		{"gemini-3.8-live-extended-thinking", 7319, 6199, 563, 352, 119, 27, 92, 696, 11_048, 23_385},
		{"gemini-3.8-live-extended-thinking", 4138, 3006, 655, 256, 63, 0, 63, 39, 5_407, 13_170},
		{"gemini-3.8-live", 625, 354, 222, 0, 53, 0, 53, 49, 1_788, 2_511},
		{"gemini-3.8-live", 799, 374, 360, 0, 47, 0, 47, 53, 2_163, 2_961},
		{"gemini-3.8-live", 1212, 445, 407, 256, 109, 0, 109, 112, 3_623, 4_944},
		{"gemini-3.8-live", 1362, 468, 516, 256, 12, 0, 0, 49, 2_430, 4_230},
		{"gemini-3.8-live", 1394, 482, 516, 256, 95, 0, 95, 86, 3_693, 5_322},
		{"gemini-3.1-flash-live-preview", 222, 175, 0, 0, 55, 0, 55, 0, 791, 1_326},
		{"gemini-3.1-flash-live-preview", 400, 195, 138, 0, 58, 0, 58, 0, 1_256, 1_896},
		{"gemini-3.1-flash-live-preview", 572, 214, 196, 64, 102, 0, 102, 0, 2_037, 2_940},
		{"gemini-3.1-flash-live-preview", 1462, 480, 596, 128, 79, 0, 67, 0, 3_134, 5_334},
		{"gemini-3.1-flash-live-preview", 850, 265, 365, 64, 61, 0, 61, 0, 2_090, 3_282},
	} {
		provider, list := s.builtIn(STS, "gemini/"+turn.model)
		totals := Usage{InputTokens: turn.prompt, OutputTokens: turn.response + turn.thoughts}
		// What the response details leave out, such as a function call, is text, and so is
		// the thinking.
		outputText := turn.response - turn.responseAudio + turn.thoughts
		split := pricing.Quantities{
			pricing.InputTextTokens:   turn.text,
			pricing.InputAudioTokens:  turn.audio,
			pricing.InputImageTokens:  turn.image,
			pricing.OutputTextTokens:  outputText,
			pricing.OutputAudioTokens: turn.responseAudio,
		}

		s.EqualValuesf(turn.billed, charged(list, STS, provider, Stat{Usage: totals, Meters: split}), "%+v", turn)
		s.InDeltaf(turn.before, legacyCostMicros(provider.Price, Usage{InputTokens: turn.prompt, OutputTokens: turn.response}), 1,
			"%+v on the totals, the thinking unread", turn)
	}
}

func (s *RoutingSuite) TestAnOpenAICacheWriteCostsWhatOpenAIBills() {
	terra, list := s.builtIn(LLM, "openai/gpt-5.6-terra")
	first := Stat{
		Usage:  Usage{InputTokens: 2_389, OutputTokens: 22},
		Meters: pricing.Quantities{pricing.CacheWriteInputTokens: 2_386},
	}

	s.EqualValues(6_235, charged(list, LLM, terra, first), "3 fresh, 2,386 written at 1.25 times and 22 generated")
	s.EqualValues(5_042, legacyCostMicros(terra.Price, first.Usage), "the writes were priced as fresh input")
}

func (s *RoutingSuite) TestALongOpenAIPromptIsPricedAtItsTier() {
	luna, list := s.builtIn(LLM, "openai/gpt-6-luna")
	long := Stat{
		Usage:  Usage{InputTokens: 279_024, OutputTokens: 5},
		Meters: pricing.Quantities{pricing.CacheWriteInputTokens: 279_021},
	}
	short := Stat{Usage: Usage{InputTokens: 272_000}}

	s.EqualValues(69_760, charged(list, LLM, luna, long), "twice the input and write rates, 1.5 times the output")
	s.EqualValues(27_904, legacyCostMicros(luna.Price, long.Usage))
	s.EqualValues(27_200, charged(list, LLM, luna, short), "272,000 tokens are not over the threshold")
}

func (s *RoutingSuite) TestTheTierOpenAIServedATurnAtPricesIt() {
	luna, list := s.builtIn(LLM, "openai/gpt-6-luna")
	astra, _ := s.builtIn(LLM, "openai/gpt-6-astra")
	at := func(provider ProviderConfig, tier string) int64 {
		return charged(list, LLM, provider, Stat{
			Usage:      Usage{InputTokens: 9, OutputTokens: 5},
			Attributes: map[string]string{"service_tier": tier},
		})
	}

	s.EqualValues(3, at(luna, "default"))
	s.EqualValues(2, at(luna, "flex"))
	s.EqualValues(2, at(luna, "batch"))
	s.EqualValues(7, at(luna, "fast"))
	s.EqualValues(3, at(luna, "ultrafast"), "GPT-6 Luna has no ultrafast tier, so the name prices nothing extra")
	for _, older := range []string{"openai/gpt-5.6-luna", "openai/gpt-5.6-terra", "openai/gpt-5.6-sol"} {
		provider, _ := s.builtIn(LLM, older)
		s.Equalf(2*at(provider, "default"), at(provider, "priority"),
			"%s answers priority for a fast request, which is billed as fast", older)
	}
	s.EqualValues(2_040, at(astra, "ultrafast"), "six times 90 + 250 µ$")
}

func (s *RoutingSuite) TestGPT56SolIsPricedAtItsPromotion() {
	sol, list := s.builtIn(LLM, "openai/gpt-5.6-sol")
	plain := Stat{Usage: Usage{InputTokens: 23, OutputTokens: 6}}

	s.EqualValues(212, charged(list, LLM, sol, plain))
}

func (s *RoutingSuite) TestGrokAndHaikuLongPromptsArePricedAtTheirTiers() {
	grok, list := s.builtIn(LLM, "xai/grok-4.7")
	haiku, _ := s.builtIn(LLM, "anthropic/claude-haiku-5-5")

	s.EqualValues(405_998, charged(list, LLM, grok, Stat{Usage: Usage{InputTokens: 199_999, OutputTokens: 1_000}}))
	s.EqualValues(812_000, charged(list, LLM, grok, Stat{Usage: Usage{InputTokens: 200_000, OutputTokens: 1_000}}),
		"200,000 tokens and more are billed at twice the rates")
	s.EqualValues(10_500, charged(list, LLM, haiku, Stat{Usage: Usage{InputTokens: 100_000, OutputTokens: 1_000}}))
	s.EqualValues(52_501, charged(list, LLM, haiku, Stat{Usage: Usage{InputTokens: 100_001, OutputTokens: 1_000}}),
		"over 100,000 tokens is billed at five times the rates")
}

// decisionAnswer is one answer from a decision model and the cost OpenRouter returned with
// it, in millionths of a dollar. Output tokens are reported and never billed.
type decisionAnswer struct {
	model        string
	inputTokens  int64
	cost, before float64
}

func (s *RoutingSuite) TestTheDecisionModelsCostWhatOpenRouterCharged() {
	for _, answer := range []decisionAnswer{
		{"cloudflare/clef-flash", 337, 12.806, 7.077},
		{"cloudflare/clef", 337, 14.154, 80.88},
		{"liquid/d1", 138, 2.76, 5.52},
	} {
		provider, list := s.builtIn(DecisionModel, "openrouter/"+answer.model)
		used := Stat{Usage: Usage{InputTokens: answer.inputTokens, OutputTokens: 1}}

		s.InDeltaf(answer.cost, float64(charged(list, DecisionModel, provider, used)), 0.5,
			"%s, which cost %v µ$ at the old rate", answer.model, answer.before)
	}
}
