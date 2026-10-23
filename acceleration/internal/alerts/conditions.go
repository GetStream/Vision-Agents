package alerts

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// Total is the sum of every series of a counter, or of a histogram's observation counts.
func Total(collector prometheus.Collector) float64 {
	var total float64
	for _, metric := range collect(collector) {
		switch {
		case metric.Counter != nil:
			total += metric.Counter.GetValue()
		case metric.Histogram != nil:
			total += float64(metric.Histogram.GetSampleCount())
		}
	}
	return total
}

// Above is how many of a histogram's observations were above bound, which must be one of
// its bucket bounds, and how many there were in all.
func Above(histogram prometheus.Collector, bound float64) (above, all float64) {
	for _, metric := range collect(histogram) {
		if metric.Histogram == nil {
			continue
		}
		count := float64(metric.Histogram.GetSampleCount())
		all += count
		for _, bucket := range metric.Histogram.GetBucket() {
			if bucket.GetUpperBound() == bound {
				above += count - float64(bucket.GetCumulativeCount())
			}
		}
	}
	return above, all
}

func collect(collector prometheus.Collector) []*dto.Metric {
	metrics := make(chan prometheus.Metric)
	go func() {
		collector.Collect(metrics)
		close(metrics)
	}()
	var written []*dto.Metric
	for metric := range metrics {
		out := &dto.Metric{}
		if err := metric.Write(out); err == nil {
			written = append(written, out)
		}
	}
	return written
}

// history keeps a reading's values over a span, to compare now with earlier. A process's
// counters start at zero, so before the first reading every value was zero.
type history struct {
	mu      sync.Mutex
	span    time.Duration
	samples []sample
}

type sample struct {
	at     time.Time
	values []float64
}

// add records values at now and drops what is older than the span needs.
func (h *history) add(now time.Time, values ...float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.samples = append(h.samples, sample{at: now, values: values})
	// Keep the newest sample at or before now-span, the oldest one a comparison needs.
	cut := 0
	for i, s := range h.samples {
		if !s.at.After(now.Add(-h.span)) {
			cut = i
		}
	}
	h.samples = h.samples[cut:]
}

// at is the values of the newest sample at or before t, or zeros when there is none.
func (h *history) at(t time.Time, width int) []float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	found := make([]float64, width)
	for _, s := range h.samples {
		if s.at.After(t) {
			break
		}
		found = s.values
	}
	return found
}

// Increased holds when the reading went up since the previous check.
func Increased(read func() float64) Check {
	var mu sync.Mutex
	var last float64
	return func(time.Time) (bool, map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		value := read()
		previous := last
		last = value
		return value > previous, map[string]any{"value": value, "increase": value - previous}
	}
}

// IncreasingFor holds when the reading went up in each of the last two halves of span, so a
// failure that keeps happening alerts and one that happened once does not.
func IncreasingFor(read func() float64, span time.Duration) Check {
	h := &history{span: span}
	return func(now time.Time) (bool, map[string]any) {
		value := read()
		h.add(now, value)
		half := h.at(now.Add(-span/2), 1)[0]
		start := h.at(now.Add(-span), 1)[0]
		return value > half && half > start, map[string]any{"value": value, "increase": value - start, "over": span.String()}
	}
}

// ShareAbove holds when more than share of a histogram's observations over the last span
// were above bound, which must be one of its bucket bounds: its percentile 1-share is above
// bound.
func ShareAbove(histogram prometheus.Collector, bound, share float64, span time.Duration) Check {
	h := &history{span: span}
	return func(now time.Time) (bool, map[string]any) {
		above, all := Above(histogram, bound)
		h.add(now, above, all)
		then := h.at(now.Add(-span), 2)
		over, count := above-then[0], all-then[1]
		holds := count > 0 && over/count > share
		return holds, map[string]any{"above": over, "observations": count, "bound": bound, "over": span.String()}
	}
}
