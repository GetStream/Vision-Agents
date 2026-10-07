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

// playoutGrace is how long a turn whose speech is queued on an edge waits for the track to
// take it, once everything else about it is known. An edge takes queued speech within the
// depth of its queue, so this only runs out when the track stopped pulling or the speech was
// dropped, and the turn is then reported without the moment it never had.
const playoutGrace = time.Second

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
	// grace is how long a turn waits for the track to take its queued speech.
	grace time.Duration
}

// openTurn is a turn that has not finished yet.
type openTurn struct {
	participant  stt.Participant
	transcriptAt time.Time
	readyAt      time.Time
	decisionAt   time.Time
	modelAt      time.Time
	firstTextAt  time.Time
	ttsAt        time.Time
	firstAudioAt time.Time
	// holdMs is how long the reply's audio waited for the caller to have been quiet, before its
	// first sound and before the sentences that follow a pause, as they were let out. Zero when it
	// did not wait.
	holdMs float64
	// queuedAt is when the edge queued the first frame of the reply for its outgoing track, and
	// pulledAt when the track took the first frame that was not silence. Only an edge that
	// reports them sets them, and firstAudioAt is when publishing returned, which for a chunk
	// longer than the edge's queue is later than either.
	queuedAt     time.Time
	pulledAt     time.Time
	sttLatencyMs float64
	llmTTFTMs    float64
	ttsTTFBMs    float64
	roundtripMs  float64
	audioOutMs   float64
	// audioDroppedMs is speech that was synthesised and paid for but never published,
	// because the turn had been abandoned by the time it arrived.
	audioDroppedMs float64
	// audioHeldMs is speech that is waiting in a hold for the caller to have been quiet, which
	// is neither published nor dropped yet. A turn closed while it is there reports it dropped:
	// it was paid for, and it never reached the caller.
	audioHeldMs float64
	// modelDone means the reply is fully generated, so how many syntheses the turn will
	// produce is known.
	modelDone bool
	// expected is how many syntheses the turn will produce in total.
	expected int
	// settled is how many of them have completed.
	settled     int
	interrupted bool
	// marks is what the edge reports queuedAt and pulledAt to.
	marks turnMarks
	// waiting is the timer that closes the turn once the track has taken its speech, or
	// playoutGrace has passed. released is set when it has run.
	waiting  *time.Timer
	released bool
}

// turnMarks is what an edge reports one turn's playout milestones to. It lives on the open turn,
// so handing it out for each chunk of a reply costs nothing, and a report that outlives the
// turn finds it closed and is dropped.
type turnMarks struct {
	tracker *turnTracker
	turn    *openTurn
	id      string
}

func (m *turnMarks) FirstFrameQueued(at time.Time)   { m.tracker.queued(m.turn, m.id, at) }
func (m *turnMarks) FirstAudiblePulled(at time.Time) { m.tracker.pulled(m.turn, m.id, at) }

func newTurnTracker(finished func(Turn)) *turnTracker {
	return &turnTracker{open: map[string]*openTurn{}, finished: finished, grace: playoutGrace}
}

// begin opens a turn. The speech-to-text latency is the provider's own decode time for
// the transcript that settled it.
func (t *turnTracker) begin(turnID string, participant stt.Participant, readyAt, transcriptAt time.Time, sttLatencyMs float64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if transcriptAt.IsZero() {
		transcriptAt = readyAt
	}

	current := &openTurn{
		participant:  participant,
		transcriptAt: transcriptAt,
		readyAt:      readyAt,
		sttLatencyMs: sttLatencyMs,
	}
	current.marks = turnMarks{tracker: t, id: turnID, turn: current}
	t.open[turnID] = current
}

func (t *turnTracker) modelStarted(turnID string, at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil {
		current.modelAt = at
	}
}

func (t *turnTracker) decided(turnID string, at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil && current.decisionAt.IsZero() {
		current.decisionAt = at
	}
}

func (t *turnTracker) firstText(turnID string, at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil && current.firstTextAt.IsZero() {
		current.firstTextAt = at
	}
}

func (t *turnTracker) modelTiming(turnID string, ttftMs float64) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil {
		current.llmTTFTMs = ttftMs
		return true
	}
	return false
}

func (t *turnTracker) ttsStarted(turnID string, at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil && current.ttsAt.IsZero() {
		current.ttsAt = at
	}
}

// firstAudio records the moment the first audio of a reply reached the edge, which is
// what ends the participant's wait.
func (t *turnTracker) firstAudio(turnID string, at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()

	current, ok := t.open[turnID]
	if !ok || !current.firstAudioAt.IsZero() {
		return
	}
	current.firstAudioAt = at
	current.roundtripMs = msBetween(current.transcriptAt, at)
}

// held records how long audio of a reply waited for the caller to have been quiet, and adds it
// to what the turn has waited so far: a reply is held before its first audio and again before the
// sentences that follow a pause. The wait before the first audio is inside the turn's roundtrip
// and the legs from its text to its audio, which it explains; one before a later sentence comes
// after them and is inside neither.
func (t *turnTracker) held(turnID string, waited time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil {
		current.holdMs += float64(waited.Microseconds()) / 1000
	}
}

// participantOf is who the reply of an open turn is for, or the zero participant for a turn that
// is not open, which is any that nobody was answering.
func (t *turnTracker) participantOf(turnID string) stt.Participant {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil {
		return current.participant
	}
	return stt.Participant{}
}

// marksFor returns what an edge reports the first frames of a reply to, or nil when the turn
// is not open or has already been told when its speech was taken, which leaves nothing for
// the edge to report.
func (t *turnTracker) marksFor(turnID string) PlayoutMarks {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current, ok := t.open[turnID]; ok && current.pulledAt.IsZero() {
		return &current.marks
	}
	return nil
}

// queued records when the edge queued the first frame of the turn's reply.
func (t *turnTracker) queued(current *openTurn, turnID string, at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.open[turnID] == current && current.queuedAt.IsZero() {
		current.queuedAt = at
	}
}

// pulled records when the track took the first frame of the turn's reply that was not
// silence. It runs on the track's goroutine, so it only wakes the turn if it was waiting to
// close: reporting it is the timer's work, because the track must never wait on a consumer.
func (t *turnTracker) pulled(current *openTurn, turnID string, at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.open[turnID] != current || !current.pulledAt.IsZero() {
		return
	}
	current.pulledAt = at
	if current.waiting != nil {
		current.waiting.Reset(0)
	}
}

// release closes a turn that was waiting for the track, now that it has taken the speech or
// playoutGrace has passed.
func (t *turnTracker) release(current *openTurn, turnID string) {
	t.mu.Lock()
	if t.open[turnID] != current {
		t.mu.Unlock()
		return
	}
	current.released = true
	finished := t.settleLocked(turnID, current)
	t.mu.Unlock()

	t.report(finished)
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

// buffered records speech that has gone into a hold for the caller to have been quiet.
func (t *turnTracker) buffered(turnID string, audioDurationMs float64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil {
		current.audioHeldMs += audioDurationMs
	}
}

// unbuffered records that the speech held for the turn has left the hold to be published.
func (t *turnTracker) unbuffered(turnID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil {
		current.audioHeldMs = 0
	}
}

// droppedFromHold records speech that was held and has been given up with its reply. It moves
// from held to dropped in one step, so a turn closed in between does not count it twice or lose it.
func (t *turnTracker) droppedFromHold(turnID string, audioDurationMs float64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil {
		current.audioHeldMs = max(0, current.audioHeldMs-audioDurationMs)
		current.audioDroppedMs += audioDurationMs
	}
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
	if current.llmTTFTMs == 0 {
		current.llmTTFTMs = timeToFirstTokenMs
	}
	current.modelDone = true
	current.expected = syntheses
	finished := t.settleLocked(turnID, current)
	t.mu.Unlock()

	t.report(finished)
}

// interrupting records that a turn is being abandoned, ahead of interrupt closing it. What
// abandoning it sets going, such as a held reply being given up, can complete the turn before
// interrupt is reached, and a turn completed by then is reported as one nobody interrupted.
func (t *turnTracker) interrupting(turnID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil {
		current.interrupted = true
	}
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
	if current.waiting != nil {
		current.waiting.Stop()
	}
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
	// Speech queued on an edge that has not taken it yet is about to give the turn its last
	// moment, so closing now would report it without. The timer closes it as soon as the track
	// has taken the speech, or when the grace says it is not going to.
	if !current.queuedAt.IsZero() && current.pulledAt.IsZero() && !current.released {
		if current.waiting == nil {
			current.waiting = time.AfterFunc(t.grace, func() { t.release(current, turnID) })
		}
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
	decidedAt := current.decisionAt
	if decidedAt.IsZero() {
		decidedAt = current.modelAt
	}
	textWaitAt := current.modelAt
	if decidedAt.After(textWaitAt) {
		textWaitAt = decidedAt
	}
	return Turn{
		TurnID:              turnID,
		Participant:         current.participant,
		StartedAt:           current.transcriptAt,
		STTLatencyMs:        current.sttLatencyMs,
		CadenceMs:           leg(current.transcriptAt, current.readyAt),
		DecisionMs:          leg(current.readyAt, decidedAt),
		ModelToFirstTextMs:  leg(textWaitAt, current.firstTextAt),
		TextToTTSMs:         leg(current.firstTextAt, current.ttsAt),
		TTSToAudioMs:        leg(current.ttsAt, current.firstAudioAt),
		ReplyHoldMs:         current.holdMs,
		FirstFrameQueuedMs:  leg(current.transcriptAt, current.queuedAt),
		FirstAudibleFrameMs: leg(current.transcriptAt, current.pulledAt),
		LLMTTFTMs:           current.llmTTFTMs,
		TTSTTFBMs:           current.ttsTTFBMs,
		RoundtripMs:         current.roundtripMs,
		// Voice in to voice out is the wait the participant felt plus the time the
		// transcriber spent deciding the turn was over, since that ran first.
		SpeechEndToAudioMs:   speechEndToAudio(current),
		SpeechEndToAudibleMs: speechEndToAudible(current),
		AudioOutMs:           current.audioOutMs,
		AudioDroppedMs:       current.audioDroppedMs + current.audioHeldMs,
		Interrupted:          current.interrupted,
	}
}

func speechEndToAudio(current *openTurn) float64 {
	if current.roundtripMs == 0 {
		return 0
	}
	return current.roundtripMs + current.sttLatencyMs
}

// speechEndToAudible is voice in to the first frame of the reply the participants could hear,
// worked out as speechEndToAudio is, from the moment the track took it.
func speechEndToAudible(current *openTurn) float64 {
	if leg(current.transcriptAt, current.pulledAt) == 0 {
		return 0
	}
	return msBetween(current.transcriptAt, current.pulledAt) + current.sttLatencyMs
}

func msBetween(from, to time.Time) float64 {
	return float64(to.Sub(from).Microseconds()) / 1000
}

func leg(from, to time.Time) float64 {
	if from.IsZero() || to.IsZero() || to.Before(from) {
		return 0
	}
	return msBetween(from, to)
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

	// closing guards the queue against the one send that cannot be recovered from. A
	// send on a closed channel panics even from a select with a default, so closing the
	// queue and sending to it have to be ordered rather than merely non-blocking.
	closing   sync.RWMutex
	closed    bool
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
		CustomerID:           r.owner.CustomerID,
		AgentID:              r.owner.AgentID,
		CallID:               r.owner.CallID,
		TurnID:               turn.TurnID,
		Tags:                 r.owner.Tags,
		StartedAt:            turn.StartedAt.UTC(),
		CadenceMs:            measured(turn.CadenceMs),
		DecisionMs:           measured(turn.DecisionMs),
		ModelToFirstTextMs:   measured(turn.ModelToFirstTextMs),
		TextToTTSMs:          measured(turn.TextToTTSMs),
		TTSToAudioMs:         measured(turn.TTSToAudioMs),
		ReplyHoldMs:          measured(turn.ReplyHoldMs),
		STTLatencyMs:         measured(turn.STTLatencyMs),
		LLMTTFTMs:            measured(turn.LLMTTFTMs),
		TTSTTFBMs:            measured(turn.TTSTTFBMs),
		RoundtripMs:          measured(turn.RoundtripMs),
		SpeechEndToAudioMs:   measured(turn.SpeechEndToAudioMs),
		FirstFrameQueuedMs:   measured(turn.FirstFrameQueuedMs),
		FirstAudibleFrameMs:  measured(turn.FirstAudibleFrameMs),
		SpeechEndToAudibleMs: measured(turn.SpeechEndToAudibleMs),
		AudioOutMs:           measured(turn.AudioOutMs),
		AudioDroppedMs:       measured(turn.AudioDroppedMs),
		Interrupted:          turn.Interrupted,
	}

	// A turn can finish after the recorder has been closed: interrupting a session closes
	// it and reports the turn that was cut short, in that order. That is a turn with
	// nowhere left to go rather than a writer falling behind, so it is let go quietly
	// instead of taking the process down with the queue.
	r.closing.RLock()
	defer r.closing.RUnlock()
	if r.closed {
		return
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
		r.closing.Lock()
		r.closed = true
		close(r.queue)
		r.closing.Unlock()
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
