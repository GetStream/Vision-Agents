package agent

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// turnQueueSize bounds how far the turn writer may fall behind before rows are dropped.
const turnQueueSize = 256

// turnWriteTimeout bounds a single write so a stuck database cannot wedge the writer.
const turnWriteTimeout = 5 * time.Second

// turnTracker assembles the timings of an exchange as it unfolds.
//
// A request row already measures each provider call on its own. What it cannot say is
// how long the participant waited between finishing a sentence and hearing the answer
// start, because that delay spans three providers and the agent's own handling. This
// gathers the pieces and reports them once the turn is over.
type turnTracker struct {
	mu   sync.Mutex
	open map[string]*openTurn
	// finished receives each turn once, when nothing more can be learned about it.
	finished func(Turn)
}

// openTurn is a turn that has not finished yet.
type openTurn struct {
	participant stt.Participant
	// heardAt is when the settled transcript arrived, which is when the wait the
	// participant feels begins.
	heardAt      time.Time
	sttLatencyMs float64
	llmTTFTMs    float64
	ttsTTFBMs    float64
	roundtripMs  float64
	audioOutMs   float64
	// audioDroppedMs is speech that was synthesised and paid for but never published,
	// because the turn had been abandoned by the time it arrived.
	audioDroppedMs float64
	// modelDone means the reply is fully generated, so how many syntheses the turn will
	// produce is known.
	modelDone bool
	// expected is how many syntheses the turn will produce in total.
	expected int
	// settled is how many of them have completed.
	settled     int
	interrupted bool
}

func newTurnTracker(finished func(Turn)) *turnTracker {
	return &turnTracker{open: map[string]*openTurn{}, finished: finished}
}

// begin opens a turn. The speech-to-text latency is the provider's own decode time for
// the transcript that settled it.
func (t *turnTracker) begin(turnID string, participant stt.Participant, heardAt time.Time, sttLatencyMs float64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.open[turnID] = &openTurn{
		participant:  participant,
		heardAt:      heardAt,
		sttLatencyMs: sttLatencyMs,
	}
}

// firstAudio records the moment the first audio of a reply reached the edge, which is
// what ends the participant's wait.
func (t *turnTracker) firstAudio(turnID string, at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()

	current, ok := t.open[turnID]
	if !ok || current.roundtripMs > 0 {
		return
	}
	current.roundtripMs = msBetween(current.heardAt, at)
}

// dropped records speech that was synthesised but never reached the participant. A turn
// closed by an interruption is already measured, so nothing is recorded against it: what
// this is for is the abandoned audio nobody asked for, which is a fault rather than a
// caller changing their mind.
func (t *turnTracker) dropped(turnID string, audioDurationMs float64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	current, ok := t.open[turnID]
	if !ok {
		return
	}
	current.audioDroppedMs += audioDurationMs
}

// spoke records a completed synthesis. A turn spoken sentence by sentence has several,
// so the wait is the first one's and the audio is all of them.
func (t *turnTracker) spoke(turnID string, timeToFirstByteMs, audioDurationMs float64) {
	t.mu.Lock()
	current, ok := t.open[turnID]
	if !ok {
		t.mu.Unlock()
		return
	}
	if current.ttsTTFBMs == 0 {
		current.ttsTTFBMs = timeToFirstByteMs
	}
	current.audioOutMs += audioDurationMs
	current.settled++
	finished := t.settleLocked(turnID, current)
	t.mu.Unlock()

	t.report(finished)
}

// completed records that the model finished, along with how many syntheses the turn will
// produce. Knowing the count is what lets the turn be closed exactly once the last of
// them has been spoken.
func (t *turnTracker) completed(turnID string, timeToFirstTokenMs float64, syntheses int) {
	t.mu.Lock()
	current, ok := t.open[turnID]
	if !ok {
		t.mu.Unlock()
		return
	}
	current.llmTTFTMs = timeToFirstTokenMs
	current.modelDone = true
	current.expected = syntheses
	finished := t.settleLocked(turnID, current)
	t.mu.Unlock()

	t.report(finished)
}

// interrupt closes a turn a participant talked over. Whatever was measured before the
// interruption still happened and is still worth reporting.
func (t *turnTracker) interrupt(turnID string) {
	t.mu.Lock()
	current, ok := t.open[turnID]
	if !ok {
		t.mu.Unlock()
		return
	}
	current.interrupted = true
	delete(t.open, turnID)
	finished := measure(turnID, current)
	t.mu.Unlock()

	t.report(&finished)
}

// settleLocked closes the turn if nothing more can be learned about it.
func (t *turnTracker) settleLocked(turnID string, current *openTurn) *Turn {
	if !current.modelDone || current.settled < current.expected {
		return nil
	}
	delete(t.open, turnID)
	finished := measure(turnID, current)
	return &finished
}

func (t *turnTracker) report(finished *Turn) {
	if finished != nil && t.finished != nil {
		t.finished(*finished)
	}
}

func measure(turnID string, current *openTurn) Turn {
	return Turn{
		TurnID:       turnID,
		Participant:  current.participant,
		StartedAt:    current.heardAt,
		STTLatencyMs: current.sttLatencyMs,
		LLMTTFTMs:    current.llmTTFTMs,
		TTSTTFBMs:    current.ttsTTFBMs,
		RoundtripMs:  current.roundtripMs,
		// Voice in to voice out is the wait the participant felt plus the time the
		// transcriber spent deciding the turn was over, since that ran first.
		SpeechEndToAudioMs: speechEndToAudio(current),
		AudioOutMs:         current.audioOutMs,
		AudioDroppedMs:     current.audioDroppedMs,
		Interrupted:        current.interrupted,
	}
}

func speechEndToAudio(current *openTurn) float64 {
	if current.roundtripMs == 0 {
		return 0
	}
	return current.roundtripMs + current.sttLatencyMs
}

func msBetween(from, to time.Time) float64 {
	return float64(to.Sub(from).Microseconds()) / 1000
}

// turnRecorder writes finished turns to Postgres off the conversation's path. A
// conversation must never wait on a database, so recording is asynchronous and rows are
// what gets dropped when the writer cannot keep up.
type turnRecorder struct {
	store  *store.Store
	owner  routing.Owner
	logger *slog.Logger

	queue chan store.Turn
	done  chan struct{}

	closeOnce sync.Once
	dropped   atomic.Int64
}

func newTurnRecorder(pgStore *store.Store, owner routing.Owner, logger *slog.Logger) *turnRecorder {
	r := &turnRecorder{
		store:  pgStore,
		owner:  owner,
		logger: logger,
		queue:  make(chan store.Turn, turnQueueSize),
		done:   make(chan struct{}),
	}
	go r.run()
	return r
}

// Record queues a finished turn, dropping it if the writer is too far behind.
func (r *turnRecorder) Record(turn Turn) {
	row := store.Turn{
		CustomerID:         r.owner.CustomerID,
		AgentID:            r.owner.AgentID,
		CallID:             r.owner.CallID,
		TurnID:             turn.TurnID,
		Tags:               r.owner.Tags,
		StartedAt:          turn.StartedAt.UTC(),
		STTLatencyMs:       measured(turn.STTLatencyMs),
		LLMTTFTMs:          measured(turn.LLMTTFTMs),
		TTSTTFBMs:          measured(turn.TTSTTFBMs),
		RoundtripMs:        measured(turn.RoundtripMs),
		SpeechEndToAudioMs: measured(turn.SpeechEndToAudioMs),
		AudioOutMs:         measured(turn.AudioOutMs),
		AudioDroppedMs:     measured(turn.AudioDroppedMs),
		Interrupted:        turn.Interrupted,
	}

	select {
	case r.queue <- row:
	default:
		r.dropped.Add(1)
	}
}

// Close drains the queue and stops the writer.
func (r *turnRecorder) Close() {
	r.closeOnce.Do(func() {
		close(r.queue)
		<-r.done
		if dropped := r.dropped.Load(); dropped > 0 {
			r.logger.Warn("dropped turns because the writer fell behind", "count", dropped)
		}
	})
}

func (r *turnRecorder) run() {
	defer close(r.done)

	for row := range r.queue {
		ctx, cancel := context.WithTimeout(context.Background(), turnWriteTimeout)
		if err := r.store.RecordTurn(ctx, &row); err != nil {
			r.logger.Error("could not record turn", "error", err)
		}
		cancel()
	}
}

// measured keeps a leg that never happened out of the percentiles rather than counting
// it as instant.
func measured(ms float64) *float64 {
	if ms <= 0 {
		return nil
	}
	return &ms
}
