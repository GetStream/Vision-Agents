package routing

import (
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Why a row's usage is the router's estimate.
const (
	// estimatedCancelled is a response cut short, which most vendors then never report: it is
	// expected, so it is counted and alarms nobody.
	estimatedCancelled = "cancelled"
	// estimatedTimeout is a response a deadline cut short, ours or its caller's: counted, and
	// alarming nobody, as a cut does.
	estimatedTimeout = "timeout"
	// estimatedMissing is work that finished and whose vendor never said what it used: an
	// adapter to fix.
	estimatedMissing = "missing"
)

// itemLabels name the item a metering counter counts, and the item an alert is about.
var itemLabels = []string{"modality", "provider", "model"}

// pricedItemLabels are itemLabels with the item's region and whether it is BYOK: an item is
// priced in each region and BYOK state apart, so it can be unpriced in one and priced in the
// others.
var pricedItemLabels = append(slices.Clone(itemLabels), "region", "byok")

var (
	usageMissing = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_missing_total",
		Help:      "Billable work that succeeded and recorded none of what its price rates: an adapter or session that lost the usage, written at nothing",
	}, itemLabels)
	usageEstimated = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_estimated_total",
		Help:      "Billable work the vendor never said the usage of, billed on the router's own count: reason cancelled for a response its caller cut short, timeout for one a deadline did, missing for one that finished",
	}, []string{"modality", "provider", "model", "reason"})
	usageUnpriced = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_unpriced_total",
		Help:      "Billable work that succeeded with no price, written at nothing until the item is priced in that region and BYOK state",
	}, pricedItemLabels)
	usageMismatch = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_mismatch_total",
		Help:      "Billable work whose input and output tokens do not add up to the vendor's own total: a part of the bill the adapter misses",
	}, itemLabels)
)

// MeteringAlerts are the conditions of an item whose rows the router cannot bill right, one
// alert per item at most every limit of the loop that runs them: billable work recorded with
// none of what it is priced by, finished on an estimate because the vendor never said, priced
// by nothing, or with tokens that do not add up to the vendor's total. A response cut short is
// estimated without an alert. None pages: the rows are written meanwhile, and each is an
// adapter or a price to fix.
func MeteringAlerts(limit time.Duration) []alerts.Condition {
	per := func(counter *prometheus.CounterVec, where map[string]string) alerts.ItemCheck {
		return alerts.IncreasedPer(counter, limit, itemLabels, where)
	}
	return []alerts.Condition{
		{Name: "missing", Level: alerts.Warn, Items: per(usageMissing, nil)},
		{Name: "estimated", Level: alerts.Warn, Items: per(usageEstimated, map[string]string{"reason": estimatedMissing})},
		{Name: "unpriced", Level: alerts.Warn, Items: alerts.IncreasedPer(usageUnpriced, limit, pricedItemLabels, nil)},
		{Name: "mismatch", Level: alerts.Warn, Items: per(usageMismatch, nil)},
	}
}

// watch counts a billable row the router cannot bill right. A row raises at most one of
// missing, estimated and unpriced; a total that does not add up is counted beside them. Work
// that is not billable costs nobody anything whatever it reported, so it is left alone.
func (r *Recorder) watch(config ProviderConfig, entry Stat, request store.Request, quote pricing.Quote) {
	if !request.Billable {
		return
	}
	item := [3]string{string(r.modality), config.Provider, config.Model}
	fields := []any{"modality", item[0], "provider", item[1], "model", item[2]}
	if entry.VendorTotalTokens != 0 && entry.VendorTotalTokens != request.InputTokens+request.OutputTokens {
		usageMismatch.WithLabelValues(item[:]...).Inc()
		if r.mismatchLogged.due(item, usageLogEvery) {
			r.logger.Error("billable work's tokens do not add up to the vendor's total", append(fields,
				"vendor_total_tokens", entry.VendorTotalTokens,
				"input_tokens", request.InputTokens, "output_tokens", request.OutputTokens)...)
		}
	}

	quoted := request.Success && entry.CostMicros == 0
	switch {
	case quoted && quote.Unpriced:
		// The list logs it as billable work with no price.
		usageUnpriced.WithLabelValues(item[0], item[1], item[2], regionOf(config), strconv.FormatBool(config.BYOK)).Inc()
	case entry.Attributes[AttributeUsageEstimated] == "true":
		reason := estimatedMissing
		switch request.ErrorCode {
		case ErrorCancelled:
			reason = estimatedCancelled
		case ErrorTimeout:
			reason = estimatedTimeout
		}
		usageEstimated.WithLabelValues(item[0], item[1], item[2], reason).Inc()
		if reason == estimatedMissing && r.estimatedLogged.due(item, usageLogEvery) {
			r.logger.Error("billable work recorded an estimate", fields...)
		}
	case quoted && !quote.Rated && request.Attributes[AttributeMeasuredBy] != MeasuredByRouter:
		// The vendor did the work and nothing says how much: a bug in an adapter or a
		// session, never a free request. What the router measured itself is the count, even
		// at zero.
		usageMissing.WithLabelValues(item[:]...).Inc()
		if r.missingLogged.due(item, usageLogEvery) {
			r.logger.Error("billable work recorded no usage", fields...)
		}
	}
}

// throttle is when each key was last let through, so what is logged on a busy path is logged
// at most once an interval. Rows are recorded from many goroutines, so the time is swapped in
// rather than read and then written.
type throttle struct {
	last sync.Map
}

// due reports whether key was not let through within every, and if so lets it through now.
func (t *throttle) due(key any, every time.Duration) bool {
	now := time.Now()
	for {
		last, seen := t.last.LoadOrStore(key, now)
		if !seen {
			return true
		}
		if now.Sub(last.(time.Time)) < every {
			return false
		}
		if t.last.CompareAndSwap(key, last, now) {
			return true
		}
	}
}
