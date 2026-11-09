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
// decision, and the words alone.
const (
	kindOverAgent  = "over_agent"
	kindPause      = "pause"
	kindTranscribe = "transcribe"
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
)

// measure records a request of kind, sent at sent within ctx, that came to err: how long its
// round trip took and what became of it, and a deadline it missed, which is said in the debug
// log with how long the request had.
func measure(ctx context.Context, kind string, sent time.Time, err error) {
	took := time.Since(sent)
	result := "ok"
	switch {
	case err == nil:
	case errors.Is(err, context.DeadlineExceeded), errors.Is(ctx.Err(), context.DeadlineExceeded):
		result = "deadline"
	case errors.Is(err, context.Canceled), errors.Is(ctx.Err(), context.Canceled):
		result = "canceled"
	default:
		result = "error"
	}
	roundTrips.WithLabelValues(kind, result).Observe(took.Seconds())
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
