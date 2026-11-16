package routing

import (
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The outcomes a request is counted under: a bounded set, never the row's own error code. Work
// on a customer's own key that failed is customer_key, whatever failed, so one customer's revoked
// key or spent account is not the vendor's error rate.
const (
	outcomeOK          = "ok"
	outcomeCancelled   = "cancelled"
	outcomeRefused     = "refused"
	outcomeTimeout     = "timeout"
	outcomeCustomerKey = "customer_key"
	outcomeError       = "error"
)

// The kinds of token a row counts.
const (
	tokensInput       = "input"
	tokensCachedInput = "cached_input"
	tokensOutput      = "output"
)

// The stages an LLM request's time to first token is split into, in the order it passes them.
const (
	StageRouter  = "router"
	StageConnect = "connect"
	StageWait    = "wait"
	StageStream  = "stream"
)

// trafficBuckets span 25 ms to 60 s: fine steps where voice first output lands (100 to 800 ms),
// wide ones for the long LLM requests. Each is a series per modality, provider and, for first
// output, model.
var trafficBuckets = []float64{0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 2, 4, 8, 15, 30, 60}

// stageBuckets are trafficBuckets with steps down to 1 ms in front: the router's own stage is a
// few Redis and Postgres round trips, which would all land in trafficBuckets' first, 25 ms.
var stageBuckets = append([]float64{0.001, 0.0025, 0.005, 0.01}, trafficBuckets...)

var (
	trafficRequests = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "requests_total",
		Help:      "Requests recorded as usage rows, vendor calls and also the router's own metered work (runtime and sandbox slices, phone legs, numbers, storage), so filter by modality; by outcome: ok when Success, cancelled by ErrorCode, customer_key for any other failure of work on the customer's own key, timeout by ErrorCode, otherwise error",
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
	}, itemLabels)
	trafficDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "request_seconds",
		Help:      "How long a whole request took: DurationMs / 1000, for rows that have it",
		Buckets:   trafficBuckets,
	}, []string{"modality", "provider"})
	llmStages = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "llm_stage_seconds",
		Help:      "Where an LLM request's time to first token went, by stage: router from Session.Create to the provider call, connect from asking the transport for a connection to getting a new one (dial and TLS; a reused one is only counted in router_llm_connections_total), wait from the request written to the response's first byte, stream from that byte to the first token. A stage a request never reached is not observed. The stages do not add up to the time to first token: writing the request, the SDK's own steps and the choice of a key before the connection is asked for, and the backoff before a retry fall in none of them",
		Buckets:   stageBuckets,
	}, []string{"provider", "model", "stage"})
	llmConnections = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "llm_connections_total",
		Help:      "Connections the LLM requests that got one were sent on, by whether the transport reused a pooled one",
	}, []string{"provider", "reused"})
)

// ObserveLLMStage records how long an LLM request spent in one stage before its first token.
func ObserveLLMStage(config ProviderConfig, stage string, took time.Duration) {
	model := config.Model
	if config.Provider == CustomProvider {
		// The model is a name each customer chose: unbounded as a label.
		model = CustomProvider
	}
	llmStages.WithLabelValues(config.Provider, model, stage).Observe(took.Seconds())
}

// CountLLMConnection counts the connection an LLM request was sent on.
func CountLLMConnection(provider string, reused bool) {
	llmConnections.WithLabelValues(provider, strconv.FormatBool(reused)).Inc()
}

// countTraffic counts a priced row in the traffic metrics, byok for work on the customer's own
// key. A quantity of zero adds nothing, so no series is made for what a modality never has.
func countTraffic(request store.Request, byok bool) {
	modality, provider, model := request.Modality, request.Provider, request.Model
	if provider == CustomProvider {
		// The model is a name each customer chose: unbounded as a label.
		model = CustomProvider
	}
	trafficRequests.WithLabelValues(modality, provider, model, outcome(request, byok)).Inc()
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
		trafficFirstOutput.WithLabelValues(modality, provider, model).Observe(*request.LatencyMs / 1000)
	}
	if request.DurationMs != nil {
		trafficDuration.WithLabelValues(modality, provider).Observe(*request.DurationMs / 1000)
	}
}

// outcome is what the request is counted under.
func outcome(request store.Request, byok bool) string {
	switch {
	case request.Success:
		return outcomeOK
	case request.ErrorCode == ErrorCancelled:
		return outcomeCancelled
	case request.ErrorCode == ErrorRefused:
		return outcomeRefused
	case byok:
		return outcomeCustomerKey
	case request.ErrorCode == ErrorTimeout:
		return outcomeTimeout
	default:
		return outcomeError
	}
}
