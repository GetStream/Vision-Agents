package agent

import (
	"log/slog"

	"github.com/prometheus/client_golang/prometheus"
)

// overlap notices one voice publishing two utterances at once. The edge plays chunks in
// the order they are published, so interleaved chunks are heard spliced into one another,
// as the agent talking over itself. It is used only from the goroutine consuming the voice.
//
// An utterance counts from its first published chunk, not from its start: a voice that
// takes each sentence as its own request starts the next sentence while the last one is
// still heard, and only holds back its audio (tts.InOrder).
type overlap struct {
	provider  string
	abandoned func(turnID string) bool
	logger    *slog.Logger
	pairs     prometheus.Counter
	seconds   prometheus.Counter
	// playing is each utterance with audio published and no completion yet.
	playing map[string]struct{}
	// warned is each pair of playing utterances already reported.
	warned map[[2]string]struct{}
}

func newOverlap(provider string, abandoned func(turnID string) bool, logger *slog.Logger) *overlap {
	return &overlap{
		provider:  provider,
		abandoned: abandoned,
		logger:    logger,
		pairs:     speechOverlaps.WithLabelValues(provider),
		seconds:   speechOverlapSeconds.WithLabelValues(provider),
		playing:   map[string]struct{}{},
		warned:    map[[2]string]struct{}{},
	}
}

// published notes a chunk of a synthesis gone to the edge. It runs once per frame, so past
// the first sight of a pair it only counts.
func (o *overlap) published(synthesisID string, durationMs float64) {
	if _, ok := o.playing[synthesisID]; !ok {
		o.playing[synthesisID] = struct{}{}
	}
	if len(o.playing) == 1 {
		return
	}
	overlapping := false
	for other := range o.playing {
		if other == synthesisID {
			continue
		}
		// A turn left behind has its audio dropped rather than published, and may not
		// complete until its provider stops sending.
		if o.abandoned(turnOf(other)) {
			delete(o.playing, other)
			continue
		}
		overlapping = true
		pair := [2]string{min(synthesisID, other), max(synthesisID, other)}
		if _, seen := o.warned[pair]; seen {
			continue
		}
		o.warned[pair] = struct{}{}
		o.pairs.Inc()
		o.logger.Warn("two utterances are being published at once, so they are heard spliced together",
			"synthesis", synthesisID, "turn", turnOf(synthesisID),
			"other_synthesis", other, "other_turn", turnOf(other), "provider", o.provider)
	}
	if overlapping {
		o.seconds.Add(durationMs / 1000)
	}
}

// finished forgets a synthesis that has completed.
func (o *overlap) finished(synthesisID string) {
	delete(o.playing, synthesisID)
	for pair := range o.warned {
		if pair[0] == synthesisID || pair[1] == synthesisID {
			delete(o.warned, pair)
		}
	}
}
