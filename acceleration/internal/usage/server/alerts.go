package server

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
)

const (
	// lagAlertSeconds is the event lag past which a router is not shipping.
	lagAlertSeconds = 600
	// failingFor is how long errors must keep coming before they page.
	failingFor = 2 * time.Minute
	// lagSpan is how far back the event lag's percentile is read.
	lagSpan = 10 * time.Minute
)

// Conditions are the service's alerts: a rejected event, a batch refused whole for what it
// holds (too big or not a batch), a batch refused for its token, ingest failing, a router's
// events arriving late, and Redis not taking the totals admission reads.
func Conditions() []alerts.Condition {
	unauthorized := func() float64 { return alerts.Total(BatchesRefused.WithLabelValues("401")) }
	return []alerts.Condition{
		{Name: "service_rejected", Level: alerts.Page, Check: alerts.Increased(func() float64 { return alerts.Total(EventsRejected) })},
		{Name: "batch_refused", Level: alerts.Page, Check: alerts.Increased(func() float64 { return alerts.Total(BatchesRefused) - unauthorized() })},
		// A producer without the token is misconfigured rather than shipping a broken batch,
		// and a caller that is no producer at all looks the same.
		{Name: "batch_unauthorized", Level: alerts.Warn, Check: alerts.Increased(unauthorized)},
		{Name: "ingest_failing", Level: alerts.Page, Check: alerts.IncreasingFor(func() float64 { return alerts.Total(IngestErrors) }, failingFor)},
		{Name: "event_lag", Level: alerts.Warn, Check: alerts.ShareAbove(EventLag, lagAlertSeconds, 0.01, lagSpan)},
		{Name: "mirror_failing", Level: alerts.Page, Check: alerts.IncreasingFor(func() float64 { return alerts.Total(ledger.MirrorFailures) }, failingFor)},
	}
}
