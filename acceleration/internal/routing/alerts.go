package routing

import (
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
)

const (
	// writeFailingFor is how long writes must keep failing, with nothing written, to page.
	writeFailingFor = 2 * time.Minute
	// shipStaleSeconds is how long ago the oldest unshipped row may have started before the
	// balances the usage service keeps are stale.
	shipStaleSeconds = 900
	// shipStaleFor is how long the oldest unshipped row must stay past shipStaleSeconds to page.
	// A row's age is counted from when its work started, so a long call written a moment ago
	// is already old; the shipper sends it within seconds.
	shipStaleFor = 2 * time.Minute
	// unownedFor is how long rows must keep waiting for an organization to warn: an app's
	// membership is normally written within seconds of its first request.
	unownedFor = 15 * time.Minute
	// pendingSlowSeconds is how long a row may wait to be written before Postgres is too slow.
	pendingSlowSeconds = 60
)

// Alerts are the router's usage conditions, for the one alerts loop of a process, which sends
// one event of a condition at most every limit. A row lost or refused while the limit holds the
// alert back is sent once the limit has passed, and each row is sent once. The names are the
// alerts' everywhere they are routed.
func Alerts(limit time.Duration) []alerts.Condition {
	return []alerts.Condition{
		// Every lost row is revenue, so there is no hold.
		{Name: "lost", Level: alerts.Page, Check: lostCheck(limit)},
		// A producer bug the service refuses.
		{Name: "rejected", Level: alerts.Page, Check: alerts.IncreasedSinceAlert(func() float64 { return alerts.Total(usageRejected) }, limit)},
		{Name: "write_failing", Level: alerts.Page, Check: writeFailing(writeFailingFor)},
		{Name: "ship_stale", Level: alerts.Page, Check: aboveFor(unshippedOldest, shipStaleSeconds, shipStaleFor, "oldest_unshipped_started_seconds_ago")},
		{Name: "unowned", Level: alerts.Warn, Check: alerts.IncreasingFor(func() float64 { return alerts.Total(usageUnowned) }, unownedFor)},
		{Name: "pending_slow", Level: alerts.Warn, Check: aboveFor(oldestPending, pendingSlowSeconds, 0, "oldest_pending_seconds")},
		// A site that owns work and names neither a key nor an origin: its rows are billed, and
		// counted against no key's limit.
		{Name: "unattributed", Level: alerts.Warn, Check: alerts.IncreasedSinceAlert(func() float64 { return alerts.Total(usageUnattributed) }, limit)},
	}
}

// lostCheck holds when rows were lost since it last held, saying what they cost and why.
func lostCheck(limit time.Duration) alerts.Check {
	increased := alerts.IncreasedSinceAlert(func() float64 { return alerts.Total(usageLost) }, limit)
	return func(now time.Time) (bool, map[string]any) {
		holds, extra := increased(now)
		byReason := map[string]any{}
		for _, reason := range []string{lostQueueFull, lostShutdown, lostInvalid} {
			byReason[reason] = alerts.Total(usageLost.WithLabelValues(reason))
		}
		extra["lost_by_reason"] = byReason
		extra["lost_micros"] = alerts.Total(usageLostMicros)
		return holds, extra
	}
}

// writeFailing holds when writes kept failing over span and none was written.
func writeFailing(span time.Duration) alerts.Check {
	failing := alerts.IncreasingFor(func() float64 { return alerts.Total(usageWriteErrors) }, span)
	flat := alerts.UnchangedFor(func() float64 { return alerts.Total(usageWritten) }, span)
	return func(now time.Time) (bool, map[string]any) {
		holds, extra := failing(now)
		unchanged, written := flat(now)
		extra["written"] = written["value"]
		return holds && unchanged, extra
	}
}

// aboveFor holds once the reading has been above threshold at every check for span, sending
// it as name.
func aboveFor(read func() float64, threshold float64, span time.Duration, name string) alerts.Check {
	var mu sync.Mutex
	var since time.Time
	return func(now time.Time) (bool, map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		value := read()
		extra := map[string]any{name: value, "threshold": threshold}
		if value <= threshold {
			since = time.Time{}
			return false, extra
		}
		if since.IsZero() {
			since = now
		}
		extra["above_for"] = now.Sub(since).String()
		return now.Sub(since) >= span, extra
	}
}
