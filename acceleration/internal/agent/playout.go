package agent

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// playout reads the edge's outgoing queue across each utterance, so that the line saying it
// finished also says what went out, what was thrown away, and how much speech was still
// waiting behind it. It is used only from the goroutine consuming the voice.
//
// The counts are the queue's, not the utterance's: two utterances open at once share theirs.
type playout struct {
	// counter is nil when the edge has no queue to count.
	counter PlayoutCounter
	dropped prometheus.Counter
	peaks   prometheus.Observer
	// last is the previous reading, which the drops since are counted from.
	last PlayoutStats
	// open is each utterance started and not finished.
	open map[string]playoutSpan
}

// playoutSpan is an open utterance: the reading at its start, and the deepest the queue has
// been since. A reading starts the next peak from what is queued, so what the utterance found
// waiting at its start is in the first peak read after it.
type playoutSpan struct {
	start PlayoutStats
	peak  time.Duration
}

func newPlayout(edge Edge, provider string) *playout {
	counter, ok := edge.(PlayoutCounter)
	if !ok {
		return &playout{}
	}
	// A voice swapped mid-call gets a reader of its own, which must not count again the drops
	// the last one did.
	return &playout{
		counter: counter,
		dropped: playoutDroppedFrames.WithLabelValues(provider),
		peaks:   playoutQueuePeak.WithLabelValues(provider),
		last:    counter.PlayoutStats(),
		open:    map[string]playoutSpan{},
	}
}

// started takes the reading an utterance is measured from.
func (p *playout) started(synthesisID string) {
	if p.counter == nil {
		return
	}
	p.open[synthesisID] = playoutSpan{start: p.read()}
}

// finished returns what the queue did while the utterance was open, as log fields, or none
// when the edge has no queue or the utterance was never seen to start.
func (p *playout) finished(synthesisID string) []any {
	if _, ok := p.open[synthesisID]; !ok {
		return nil
	}
	stats := p.read()
	span := p.open[synthesisID]
	delete(p.open, synthesisID)
	p.peaks.Observe(span.peak.Seconds())
	return []any{
		"frames_queued", stats.Queued - span.start.Queued,
		"frames_sent", stats.Sent - span.start.Sent,
		"frames_dropped", stats.Dropped - span.start.Dropped,
		"peak_queued_ms", span.peak.Milliseconds(),
		"queued_ms", stats.Waiting.Milliseconds(),
	}
}

// read takes a reading. Its peak is the deepest since the last one, so it belongs to every
// utterance open now.
func (p *playout) read() PlayoutStats {
	stats := p.counter.PlayoutStats()
	p.dropped.Add(float64(stats.Dropped - p.last.Dropped))
	p.last = stats
	for id, span := range p.open {
		span.peak = max(span.peak, stats.Peak)
		p.open[id] = span
	}
	return stats
}
