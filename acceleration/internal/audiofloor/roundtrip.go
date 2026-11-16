package audiofloor

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// What a request to the server is for, as roundTrips labels it: a decision while the caller
// talks over the agent, which has the deployment's wait (Client.HearDeadline), any other
// decision, and the words alone; and as backedOff labels one not sent, the words so far while
// the caller talks (interims.go), which are asked for alone.
const (
	kindOverAgent  = "over_agent"
	kindPause      = "pause"
	kindTranscribe = "transcribe"
	kindInterim    = "interim"
)

// The requests that set decisions up, as setupRoundTrips labels them: registering an agent and
// reading the policy.
const (
	routeAgents = "agents"
	routeInfo   = "info"
)

var (
	roundTrips = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "chorus_asr_round_trip_seconds",
		Help: "How long each request for a decision or for words took the chorus-asr server, from sending it to its answer " +
			"(a decision's, not the words that follow it), by kind: over_agent (a decision while the caller talks over the agent, " +
			"waited for ROUTER_CHORUS_ASR_OVERLAP_TIMEOUT), pause (any other decision) or transcribe (the words alone); " +
			"and by outcome: ok, deadline (the router stopped waiting), canceled (newer words, or the call ended) or error.",
		Buckets: []float64{.01, .025, .05, .075, .1, .15, .2, .25, .3, .35, .4, .45, .5, .75, 1, 2, 4, 8},
	}, []string{"kind", "outcome"})
	deadlineMisses = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "chorus_asr_deadline_misses_total",
		Help: "Requests to the chorus-asr server the router stopped waiting for at their deadline, by kind, as " +
			"chorus_asr_round_trip_seconds labels them: a decision the flow controller made instead, or words asked for again.",
	}, []string{"kind"})
	setupRoundTrips = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "chorus_asr_setup_round_trip_seconds",
		Help: "How long each request that sets decisions up took the chorus-asr server, by route: agents (POST /v1/agents, " +
			"an agent registered, on a decision's deadline) or info (GET /v1/info, the policy read); and by outcome, as " +
			"chorus_asr_round_trip_seconds says it.",
		Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2, 4, 8},
	}, []string{"route", "outcome"})
	requestErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "chorus_asr_request_errors_total",
		Help: "Requests to the chorus-asr server that failed, by kind (over_agent, pause, transcribe, agents or info) and code: " +
			"the HTTP status the server answered (404, 429, 503...), busy or no_words (it answered without the words), " +
			"timeout (the router stopped waiting) or connection (no answer). A decision the server answered 404 is counted " +
			"before it is asked again, and requests canceled for newer words are not counted.",
	}, []string{"kind", "code"})
	backedOff = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "chorus_asr_backed_off_total",
		Help: "Requests the router did not send the chorus-asr server while a Retry-After it answered a 503 or 429 with " +
			"lasted, by kind: over_agent (a check while the caller talks over the agent, which the flow controller rules on " +
			"instead) or interim (the words so far while the caller talks). Pauses are asked about all the same.",
	}, []string{"kind"})
)

// measure records a request of kind, sent at sent within ctx, that came to err: how long its
// round trip took and what became of it, and a deadline it missed, which is said in the debug
// log with how long the request had.
func measure(ctx context.Context, kind string, sent time.Time, err error) {
	took := time.Since(sent)
	result := outcomeOf(ctx, err)
	roundTrips.WithLabelValues(kind, result).Observe(took.Seconds())
	counted(kind, result, err)
	if result != "deadline" {
		return
	}
	deadlineMisses.WithLabelValues(kind).Inc()
	var had time.Duration
	if deadline, ok := ctx.Deadline(); ok {
		had = deadline.Sub(sent)
	}
	slog.Default().Debug("the chorus-asr server missed the router's deadline", "kind", kind,
		"deadline_ms", had.Round(time.Millisecond).Milliseconds(), "took_ms", took.Round(time.Millisecond).Milliseconds())
}

// measureSetup records a request that set decisions up, to route, sent at sent within ctx,
// that came to err.
func measureSetup(ctx context.Context, route string, sent time.Time, err error) {
	result := outcomeOf(ctx, err)
	setupRoundTrips.WithLabelValues(route, result).Observe(time.Since(sent).Seconds())
	counted(route, result, err)
}

// outcomeOf is what became of a request within ctx that came to err, as roundTrips says it.
func outcomeOf(ctx context.Context, err error) string {
	switch {
	case err == nil:
		return "ok"
	case errors.Is(err, context.DeadlineExceeded), errors.Is(ctx.Err(), context.DeadlineExceeded):
		return "deadline"
	case errors.Is(err, context.Canceled), errors.Is(ctx.Err(), context.Canceled):
		return "canceled"
	}
	return "error"
}

// counted counts a request of kind that failed, its outcome result, by why (failure): one
// canceled was not the server's to answer.
func counted(kind, result string, err error) {
	switch result {
	case "ok", "canceled":
		return
	case "deadline":
		requestErrors.WithLabelValues(kind, "timeout").Inc()
		return
	}
	requestErrors.WithLabelValues(kind, failure(err)).Inc()
}
