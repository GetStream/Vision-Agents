//go:build integration

package routing

import (
	"log/slog"
	"sort"
	"strings"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// vendorOfItsOwn is a provider no other test records under, so the traffic metrics of a test
// start empty however often it runs.
func vendorOfItsOwn() ProviderConfig {
	return ProviderConfig{Provider: "traffic-" + uuid.NewString()[:8], Model: "m"}
}

// traffic is every series of the collector whose label is value: its other labels, joined,
// with a counter's value, or a histogram's count and sum.
func traffic(collector prometheus.Collector, label, value string) map[string]float64 {
	metrics := make(chan prometheus.Metric)
	go func() {
		collector.Collect(metrics)
		close(metrics)
	}()
	series := map[string]float64{}
	for metric := range metrics {
		var written dto.Metric
		if err := metric.Write(&written); err != nil {
			panic(err)
		}
		var names []string
		matched := false
		for _, pair := range written.GetLabel() {
			if pair.GetName() == label {
				matched = pair.GetValue() == value
				continue
			}
			names = append(names, pair.GetName()+"="+pair.GetValue())
		}
		if !matched {
			continue
		}
		sort.Strings(names)
		key := strings.Join(names, ",")
		if histogram := written.GetHistogram(); histogram != nil {
			series[key+" count"] = float64(histogram.GetSampleCount())
			series[key+" sum"] = histogram.GetSampleSum()
			continue
		}
		series[key] = written.GetCounter().GetValue()
	}
	return series
}

func (s *RecorderSuite) TestASuccessfulLLMRowCountsItsRequestAndEachKindOfToken() {
	vendor := vendorOfItsOwn()

	s.recordedAs(LLM, s.prices, vendor, Stat{Success: true,
		Usage: Usage{InputTokens: 1_000, CachedInputTokens: 200, OutputTokens: 100}})

	s.Equal(map[string]float64{"modality=llm,model=m,outcome=ok": 1}, traffic(trafficRequests, "provider", vendor.Provider))
	s.Equal(map[string]float64{
		"kind=input,modality=llm,model=m":        1_000,
		"kind=cached_input,modality=llm,model=m": 200,
		"kind=output,modality=llm,model=m":       100,
	}, traffic(trafficTokens, "provider", vendor.Provider), "the input counts the cached part too, as the row does")
	s.Empty(traffic(trafficCharacters, "provider", vendor.Provider), "an LLM row has no characters")
	s.Empty(traffic(trafficAudio, "provider", vendor.Provider), "nor audio")
	s.Empty(traffic(trafficCost, "provider", vendor.Provider), "and a row with no price costs nothing")
}

func (s *RecorderSuite) TestAFailureIsCountedUnderItsOutcomeNeverItsCode() {
	vendor := vendorOfItsOwn()

	s.recordedAs(LLM, s.prices, vendor,
		Stat{ErrorCode: ErrorCancelled},
		Stat{ErrorCode: ErrorTimeout},
		Stat{ErrorCode: "rate_limited"},
		Stat{})

	s.Equal(map[string]float64{
		"modality=llm,model=m,outcome=cancelled": 1,
		"modality=llm,model=m,outcome=timeout":   1,
		"modality=llm,model=m,outcome=error":     2,
	}, traffic(trafficRequests, "provider", vendor.Provider))
}

func (s *RecorderSuite) TestATTSRowCountsItsCharactersAndAudioAndNoTokens() {
	vendor := vendorOfItsOwn()

	s.recordedAs(TTS, s.prices, vendor, Stat{Success: true, Usage: Usage{Characters: 42, AudioMs: 1_500}})

	s.Equal(map[string]float64{"modality=tts,model=m": 42}, traffic(trafficCharacters, "provider", vendor.Provider))
	s.Equal(map[string]float64{"modality=tts,model=m": 1.5}, traffic(trafficAudio, "provider", vendor.Provider))
	s.Empty(traffic(trafficTokens, "provider", vendor.Provider), "no series for tokens a TTS row never has")
}

func (s *RecorderSuite) TestARowWithoutTimingsAddsNoObservation() {
	vendor := vendorOfItsOwn()

	s.recordedAs(STT, s.prices, vendor, Stat{Success: true, LatencyMs: 120, DurationMs: 900}, Stat{Success: true})

	s.Equal(map[string]float64{"modality=stt count": 1, "modality=stt sum": 0.12},
		traffic(trafficFirstOutput, "provider", vendor.Provider), "only the row that has a latency, in seconds")
	s.Equal(map[string]float64{"modality=stt count": 1, "modality=stt sum": 0.9},
		traffic(trafficDuration, "provider", vendor.Provider), "only the row that has a duration, in seconds")
}

func (s *RecorderSuite) TestTheTimingsAreBucketedFrom25MillisecondsTo60SecondsFinestWhereVoiceLands() {
	vendor := vendorOfItsOwn()

	s.recordedAs(STT, s.prices, vendor, Stat{Success: true, LatencyMs: 120, DurationMs: 900})

	for _, histogram := range []*prometheus.HistogramVec{trafficFirstOutput, trafficDuration} {
		var written dto.Metric
		s.Require().NoError(histogram.WithLabelValues(string(STT), vendor.Provider).(prometheus.Metric).Write(&written))
		var bounds []float64
		for _, bucket := range written.GetHistogram().GetBucket() {
			bounds = append(bounds, bucket.GetUpperBound())
		}
		s.Equal([]float64{0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 2, 4, 8, 15, 30, 60}, bounds)
	}
}

func (s *RecorderSuite) TestABillableRowCountsItsPriceNotTheVendorsCharge() {
	vendor := vendorOfItsOwn()
	vendor.Price = Price{PerMillionInputTokens: 2}
	prices, err := pricing.NewList(pricing.ActiveVersion, pricing.PricedAll(PriceEntries(Config{LLM: {Providers: []ProviderConfig{vendor}}})), nil)
	s.Require().NoError(err)

	row := s.recordedAs(LLM, prices, vendor, Stat{Success: true, Usage: Usage{InputTokens: 1_000}})[0]

	s.Require().True(row.Billable)
	s.Require().NotEqual(row.BaseCostMicros, row.CostMicros, "priced above its cost, so the two differ")
	s.Equal(map[string]float64{"modality=llm,model=m": float64(row.CostMicros)}, traffic(trafficCost, "provider", vendor.Provider))
}

func (s *RecorderSuite) TestANonBillableRowCountsNoCost() {
	// A customer's own model is priced at its own rates and bills nobody.
	own := ProviderConfig{Provider: CustomProvider, Model: "traffic-" + uuid.NewString()[:8], Price: Price{PerMillionInputTokens: 2}}
	before := traffic(trafficRequests, "provider", CustomProvider)["modality=llm,model=custom,outcome=ok"]
	costBefore := traffic(trafficCost, "provider", CustomProvider)["modality=llm,model=custom"]

	row := s.recordedAs(LLM, s.prices, own, Stat{Success: true, Usage: Usage{InputTokens: 1_000}})[0]

	s.Require().False(row.Billable)
	s.Require().Positive(row.CostMicros, "it has a cost, so leaving it out is the rule and not a zero")
	s.Equal(costBefore, traffic(trafficCost, "provider", CustomProvider)["modality=llm,model=custom"])
	s.Equal(before+1, traffic(trafficRequests, "provider", CustomProvider)["modality=llm,model=custom,outcome=ok"],
		"its request still counts, under the model custom")
}

func (s *RecorderSuite) TestACustomersOwnModelNameIsNeverALabel() {
	own := ProviderConfig{Provider: CustomProvider, Model: "traffic-" + uuid.NewString()[:8]}

	s.recordedAs(LLM, s.prices, own, Stat{Success: true, Usage: Usage{InputTokens: 1_000, OutputTokens: 10}})

	s.Empty(traffic(trafficRequests, "model", own.Model))
	s.Empty(traffic(trafficTokens, "model", own.Model))
}

func (s *RecorderSuite) TestARecorderThatWritesNowhereCountsNothing() {
	vendor := vendorOfItsOwn()
	recorder := NewRecorder(LLM, nil, nil, s.prices, slog.New(slog.DiscardHandler))
	defer recorder.Close()

	recorder.Record(vendor, Stat{Success: true, Usage: Usage{InputTokens: 1_000}})

	s.Empty(traffic(trafficRequests, "provider", vendor.Provider), "it recorded no row")
	s.Empty(traffic(trafficTokens, "provider", vendor.Provider))
}
