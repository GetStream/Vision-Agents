package routing

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The outcomes a request is counted under: a bounded set, never the row's own error code.
const (
	outcomeOK        = "ok"
	outcomeCancelled = "cancelled"
	outcomeTimeout   = "timeout"
	outcomeError     = "error"
)

// The kinds of token a row counts.
const (
	tokensInput       = "input"
	tokensCachedInput = "cached_input"
	tokensOutput      = "output"
)

// trafficBuckets span 25 ms to 60 s: fine steps where voice first output lands (100 to 800 ms),
// wide ones for the long LLM requests. Each is a series per modality and provider.
var trafficBuckets = []float64{0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 2, 4, 8, 15, 30, 60}

var (
	trafficRequests = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "requests_total",
		Help:      "Requests recorded as usage rows, vendor calls and also the router's own metered work (runtime and sandbox slices, phone legs, numbers, storage), so filter by modality; by outcome: ok when Success, cancelled and timeout by ErrorCode, otherwise error",
	}, []string{"modality", "provider", "model", "outcome"})
	trafficTokens = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "tokens_total",
		Help:      "Tokens of the usage rows, by kind: input is InputTokens, the cached part included; cached_input is CachedInputTokens; output is OutputTokens",
	}, []string{"modality", "provider", "model", "kind"})
	trafficCharacters = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "characters_total",
		Help:      "Characters of the usage rows: the billable text, from Characters",
	}, itemLabels)
	trafficAudio = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "audio_seconds_total",
		Help:      "Audio of the usage rows, transcribed or produced: AudioMs / 1000",
	}, itemLabels)
	trafficCost = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "cost_micros_total",
		Help:      "What the billable usage rows cost, in millionths of a dollar: CostMicros",
	}, itemLabels)
	trafficFirstOutput = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "first_output_seconds",
		Help:      "How long a request took to its first useful output: LatencyMs / 1000, for rows that have it",
		Buckets:   trafficBuckets,
	}, []string{"modality", "provider"})
	trafficDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "request_seconds",
		Help:      "How long a whole request took: DurationMs / 1000, for rows that have it",
		Buckets:   trafficBuckets,
	}, []string{"modality", "provider"})
)

// countTraffic counts a priced row in the traffic metrics. A quantity of zero adds nothing, so
// no series is made for what a modality never has.
func countTraffic(request store.Request) {
	modality, provider, model := request.Modality, request.Provider, request.Model
	if provider == CustomProvider {
		// The model is a name each customer chose: unbounded as a label.
		model = CustomProvider
	}
	trafficRequests.WithLabelValues(modality, provider, model, outcome(request)).Inc()
	tokens := func(kind string, count int64) {
		if count > 0 {
			trafficTokens.WithLabelValues(modality, provider, model, kind).Add(float64(count))
		}
	}
	tokens(tokensInput, request.InputTokens)
	tokens(tokensCachedInput, request.CachedInputTokens)
	tokens(tokensOutput, request.OutputTokens)
	if request.Characters > 0 {
		trafficCharacters.WithLabelValues(modality, provider, model).Add(float64(request.Characters))
	}
	if request.AudioMs > 0 {
		trafficAudio.WithLabelValues(modality, provider, model).Add(float64(request.AudioMs) / 1000)
	}
	if request.Billable && request.CostMicros > 0 {
		trafficCost.WithLabelValues(modality, provider, model).Add(float64(request.CostMicros))
	}
	if request.LatencyMs != nil {
		trafficFirstOutput.WithLabelValues(modality, provider).Observe(*request.LatencyMs / 1000)
	}
	if request.DurationMs != nil {
		trafficDuration.WithLabelValues(modality, provider).Observe(*request.DurationMs / 1000)
	}
}

// outcome is what the request is counted under.
func outcome(request store.Request) string {
	switch {
	case request.Success:
		return outcomeOK
	case request.ErrorCode == ErrorCancelled:
		return outcomeCancelled
	case request.ErrorCode == ErrorTimeout:
		return outcomeTimeout
	default:
		return outcomeError
	}
}
