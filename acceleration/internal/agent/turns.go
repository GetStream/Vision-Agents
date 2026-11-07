package agent

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// turnQueueSize bounds how far the turn writer may fall behind before rows are dropped.
const turnQueueSize = 256

// turnWriteTimeout bounds a single write so a stuck database cannot wedge the writer.
const turnWriteTimeout = 5 * time.Second

// turnDrainTimeout bounds how long Close waits for the queue to be written, however many
// rows are in it. It is two writes: the one in flight may use its whole turnWriteTimeout,
// and the second is what is left for the rows behind it. That is a tradeoff, not a promise
// to write them all: a full queue takes the store about 25 to 40 ms a row (measured against
// a local Postgres and Redis under load), so a store that answers but is slow loses the
// tail of a full queue at the deadline. What is not written by then is dropped and counted,
// so a stuck database cannot hold a call's teardown for turnQueueSize writes of
// turnWriteTimeout each.
const turnDrainTimeout = 2 * turnWriteTimeout

// drainWithin waits for a writer to finish what is queued, for at most timeout. After that
// it cancels the writer's context, which makes the writer drop and count what is left, and
// waits for it to stop. It is how every recorder's Close is bound by one deadline.
func drainWithin(done <-chan struct{}, timeout time.Duration, cancel context.CancelFunc) {
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	select {
	case <-done:
	case <-deadline.C:
		cancel()
		<-done
	}
	cancel()
}

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
	// owner is who the turn belongs to, which is what its span is labelled with.
	owner routing.Owner
	// grace is how long a turn waits for the track to take its queued speech.
	grace time.Duration
	// origins is what started each of the call's latest originsKept turns. Like
	// Agent.abandoned it outlives the turn's measurement, because an utterance may finish
	// after it, but it is capped: a backchannel or a hold line is never measured, so nothing
	// would otherwise say when its entry may go.
	origins map[string]origin
	// out is when the audio of each turn in origins first went out to the call, for the
	// turns whose audio has.
	out map[string]time.Time
	// order is the turns in origins, oldest first.
	order []string
	// caller is the latest caller turn when a turn starts, which is the parent it names. A
	// tool reply after the caller spoke again names the newer caller turn, and a backchannel
	// the one before the words it murmurs over; the turn a tool call answers is on the
	// tool's own line.
	caller string
	// native says whether the agent is on a speech-to-speech model now; nil is a cascade.
	native func() bool
}

// originsKept is how many turns' origins are kept. An utterance is logged within seconds of
// its turn starting, so this is far more than any of them can reach back.
const originsKept = 256

// What starts a turn: the caller, or the agent on its own after a caller turn.
const (
	kindCaller = "caller"
	// kindTool says what a tool returned.
	kindTool = "tool"
	// kindFollowUp says what delegated work came back with.
	kindFollowUp    = "follow-up"
	kindBackchannel = "backchannel"
	// kindHold, kindUpdate and kindCheckIn are lines the model writes for a silence:
	// while a tool runs, while delegated work runs, and when nobody has spoken.
	kindHold    = "hold"
	kindUpdate  = "update"
	kindCheckIn = "checkin"
	// kindSay is a line said as written, such as the greeting.
	kindSay     = "say"
	kindHandoff = "handoff"
)

// origin is what started a turn, and the caller turn it follows when it was not the caller.
type origin struct {
	kind   string
	parent string
	// pipeline is the pipeline the agent ran when the turn started: a settings change may
	// swap it while the turn is still open, and the turn finishes on the one it began on.
	pipeline string
	// scheduled says a scheduled task fired the caller turn, not somebody in the call.
	scheduled bool
}

// openTurn is a turn that has not finished yet.
type openTurn struct {
	participant  stt.Participant
	transcriptAt time.Time
	readyAt      time.Time
	modelAt      time.Time
	firstTextAt  time.Time
	ttsAt        time.Time
	firstAudioAt time.Time
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
	// modelDone means the reply is fully generated, so how many syntheses the turn will
	// produce is known.
	modelDone bool
	// expected is how many syntheses the turn will produce in total.
	expected int
	// settled is how many of them have completed.
	settled int
	// owed means a caller turn's reply handed work on, to a tool or to the subagent, so when
	// none of its own audio went out the turn waits for the first line said for it.
	owed        bool
	interrupted bool
	// why is why the reply was abandoned, set with interrupted.
	why interruption
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

func newTurnTracker(owner routing.Owner, finished func(Turn)) *turnTracker {
	return &turnTracker{open: map[string]*openTurn{}, finished: finished, owner: owner, grace: playoutGrace, origins: map[string]origin{},
		out: map[string]time.Time{}}
}

// started records what started a turn, before anything of it is measured or said. A caller
// turn still waiting for a line said for it is reported without one when the caller speaks
// again: what follows answers the newer turn.
func (t *turnTracker) started(turnID, kind string) {
	t.mu.Lock()
	pipeline := pipelineCascade
	if t.native != nil && t.native() {
		pipeline = pipelineNative
	}
	if kind != kindCaller {
		t.remember(turnID, origin{kind: kind, parent: t.caller, pipeline: pipeline})
		t.mu.Unlock()
		return
	}
	t.caller = turnID
	t.remember(turnID, origin{kind: kind, pipeline: pipeline})
	given := t.giveUpLocked()
	t.mu.Unlock()

	t.reportAll(given)
}

// giveUp reports each caller turn still waiting for a line said for it, with no first audio.
func (t *turnTracker) giveUp() {
	t.mu.Lock()
	given := t.giveUpLocked()
	t.mu.Unlock()

	t.reportAll(given)
}

func (t *turnTracker) giveUpLocked() []settledTurn {
	var given []settledTurn
	for id, current := range t.open {
		if current.owed {
			current.owed = false
			given = append(given, settledTurn{id: id, current: current, finished: t.settleLocked(id, current)})
		}
	}
	return given
}

// settledTurn is a turn settled under the lock, to be reported once it is let go.
type settledTurn struct {
	id       string
	current  *openTurn
	finished *Turn
}

func (t *turnTracker) reportAll(settled []settledTurn) {
	for _, turn := range settled {
		t.report(turn.id, turn.current, turn.finished)
	}
}

// owes records that a caller turn's reply handed work on. It is told before the reply is
// completed, so a reply that said nothing is not closed without the line said for it. Nobody
// waits on a turn a scheduled task fired, so that one owes nothing.
func (t *turnTracker) owes(turnID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil && t.origins[turnID].kind == kindCaller && !t.origins[turnID].scheduled {
		current.owed = true
	}
}

// scheduledTurn records that a scheduled task fired the caller turn turnID, once started has
// recorded it.
func (t *turnTracker) scheduledTurn(turnID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if started, known := t.origins[turnID]; known {
		started.scheduled = true
		t.origins[turnID] = started
	}
}

// answering records that a native reply of kind nobody's silence opened answers the latest
// caller turn, when that turn still waits for a line said for it, and records nothing otherwise.
func (t *turnTracker) answering(turnID, kind string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if caller := t.open[t.caller]; caller != nil && caller.owed {
		t.remember(turnID, origin{kind: kind, parent: t.caller, pipeline: pipelineNative})
	}
}

// answersCaller says whether a turn of kind is a line said for the caller turn it follows: a hold
// or update line while the work runs, or what a tool or the subagent came back with.
func answersCaller(kind string) bool {
	switch kind {
	case kindHold, kindUpdate, kindTool, kindFollowUp:
		return true
	}
	return false
}

// remember keeps the origin of a turn, letting the oldest go past originsKept.
func (t *turnTracker) remember(turnID string, started origin) {
	if _, kept := t.origins[turnID]; !kept {
		t.order = append(t.order, turnID)
	}
	t.origins[turnID] = started
	if len(t.order) > originsKept {
		delete(t.origins, t.order[0])
		delete(t.out, t.order[0])
		t.order = t.order[1:]
	}
}

// originOf is what started a turn, zero for one never recorded.
func (t *turnTracker) originOf(turnID string) origin {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.origins[turnID]
}

// wentOut records when a turn's first audio was sent to the call, the earliest the call can
// have heard any of it. It outlives the turn's measurement, as its origin does, since the
// audio may play on after the turn has closed.
func (t *turnTracker) wentOut(turnID string, at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, known := t.origins[turnID]; !known {
		return
	}
	if _, already := t.out[turnID]; !already {
		t.out[turnID] = at
	}
}

// answerHeardBefore says whether any of the agent's answer to the caller turn turnID is part
// of went out before at: the reply to that turn, or a line the agent said after it on its own,
// such as what a tool came back with. ok is false for a turn that is part of no answer, a
// murmur or a line said before the caller spoke.
func (t *turnTracker) answerHeardBefore(turnID string, at time.Time) (heard, ok bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	started, known := t.origins[turnID]
	caller := turnID
	if started.kind != kindCaller {
		caller = started.parent
	}
	if !known || started.kind == kindBackchannel || caller == "" {
		return false, false
	}
	for id, out := range t.out {
		o := t.origins[id]
		answers := id == caller || o.parent == caller && o.kind != kindCaller && o.kind != kindBackchannel
		if answers && out.Before(at) {
			return true, true
		}
	}
	return false, true
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
	if current := t.open[turnID]; current != nil && current.modelAt.IsZero() {
		current.modelAt = at
	}
}

func (t *turnTracker) firstText(turnID string, at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil && current.firstTextAt.IsZero() {
		current.firstTextAt = at
	}
}

func (t *turnTracker) ttsStarted(turnID string, at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if current := t.open[turnID]; current != nil && current.ttsAt.IsZero() {
		current.ttsAt = at
	}
}

// sentToVoice reports whether any of a turn's reply has been sent to the voice.
func (t *turnTracker) sentToVoice(turnID string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	current, ok := t.open[turnID]
	return ok && !current.ttsAt.IsZero()
}

// firstAudio records the moment the first audio of a reply reached the edge, which is
// what ends the participant's wait. The first audio of a line said for a caller turn that
// had none of its own ends that turn's wait too.
func (t *turnTracker) firstAudio(turnID string, at time.Time) {
	t.mu.Lock()
	if current, ok := t.open[turnID]; ok && current.firstAudioAt.IsZero() {
		current.firstAudioAt = at
		current.roundtripMs = msBetween(current.transcriptAt, at)
	}
	var answered settledTurn
	if started := t.origins[turnID]; answersCaller(started.kind) && !t.origins[started.parent].scheduled {
		if caller := t.open[started.parent]; caller != nil && caller.firstAudioAt.IsZero() {
			caller.firstAudioAt = at
			caller.roundtripMs = msBetween(caller.transcriptAt, at)
			answered = settledTurn{id: started.parent, current: caller, finished: t.settleLocked(started.parent, caller)}
		}
	}
	t.mu.Unlock()

	t.report(answered.id, answered.current, answered.finished)
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

	t.report(turnID, current, finished)
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

	t.report(turnID, current, finished)
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

	t.report(turnID, current, finished)
}

// interrupt closes a turn a participant talked over. Whatever was measured before the
// interruption still happened and is still worth reporting. It returns how long the reply
// had been going out, from its first audio, zero when none went out.
func (t *turnTracker) interrupt(turnID string, why interruption) float64 {
	t.mu.Lock()
	current, ok := t.open[turnID]
	if !ok {
		t.mu.Unlock()
		return 0
	}
	current.interrupted = true
	current.why = why
	playedMs := leg(current.firstAudioAt, time.Now())
	if current.waiting != nil {
		current.waiting.Stop()
	}
	delete(t.open, turnID)
	finished := measure(turnID, current)
	t.mu.Unlock()

	t.report(turnID, current, &finished)
	return playedMs
}

// settleLocked closes the turn if nothing more can be learned about it.
func (t *turnTracker) settleLocked(turnID string, current *openTurn) *Turn {
	if !current.modelDone || current.settled < current.expected {
		return nil
	}
	if current.owed && current.firstAudioAt.IsZero() {
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

func (t *turnTracker) report(turnID string, current *openTurn, finished *Turn) {
	if finished == nil {
		return
	}
	traceTurn(t.owner, turnID, current)
	if t.finished != nil {
		t.finished(*finished)
	}
}

func measure(turnID string, current *openTurn) Turn {
	return Turn{
		TurnID:              turnID,
		Participant:         current.participant,
		StartedAt:           current.transcriptAt,
		STTLatencyMs:        current.sttLatencyMs,
		CadenceMs:           leg(current.transcriptAt, current.readyAt),
		DecisionMs:          leg(current.readyAt, current.modelAt),
		ModelToFirstTextMs:  leg(current.modelAt, current.firstTextAt),
		TextToTTSMs:         leg(current.firstTextAt, current.ttsAt),
		TTSToAudioMs:        leg(current.ttsAt, current.firstAudioAt),
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
		AudioDroppedMs:       current.audioDroppedMs,
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

// The pipelines a turn latency is labelled with, named as the session's mode is.
const (
	pipelineCascade = "cascade"
	pipelineNative  = "native"
)

// observeTurnLatency records a caller turn's waits. A leg that did not happen is zero, and is
// left out rather than counted as an instant reply.
func observeTurnLatency(turn Turn, pipeline string) {
	for _, observed := range []struct {
		leg string
		ms  float64
	}{
		{"roundtrip", turn.RoundtripMs},
		{"speech_end_to_audio", turn.SpeechEndToAudioMs},
		{"speech_end_to_audible", turn.SpeechEndToAudibleMs},
	} {
		if observed.ms > 0 {
			turnLatencySeconds.WithLabelValues(observed.leg, pipeline).Observe(observed.ms / 1000)
		}
	}
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

// turnRecorder writes finished turns off the conversation's path: the turn to Redis for
// the hour somebody may read the call back in, and its counts to Postgres for the stats.
// A conversation must never wait on a database, so recording is asynchronous and turns
// are what gets dropped when the writer cannot keep up. Either side may be nil.
type turnRecorder struct {
	store  *store.Store
	live   *live.Client
	owner  routing.Owner
	logger *slog.Logger

	queue chan live.Turn
	done  chan struct{}

	// write stores one row; it is the store and Redis unless a test says otherwise.
	write func(ctx context.Context, row live.Turn) error

	// ctx is the writer's, cancelled when Close gives up on what is still queued.
	ctx          context.Context
	cancel       context.CancelFunc
	drainTimeout time.Duration
	writeTimeout time.Duration

	// closing guards the queue against the one send that cannot be recovered from. A
	// send on a closed channel panics even from a select with a default, so closing the
	// queue and sending to it have to be ordered rather than merely non-blocking.
	closing   sync.RWMutex
	closed    bool
	closeOnce sync.Once
	dropped   atomic.Int64
}

func newTurnRecorder(pgStore *store.Store, liveClient *live.Client, owner routing.Owner, logger *slog.Logger) *turnRecorder {
	return buildTurnRecorder(pgStore, liveClient, owner, logger).start()
}

// buildTurnRecorder is a recorder that has not started writing, so a test can swap how.
func buildTurnRecorder(pgStore *store.Store, liveClient *live.Client, owner routing.Owner, logger *slog.Logger) *turnRecorder {
	r := &turnRecorder{
		store:        pgStore,
		live:         liveClient,
		owner:        owner,
		logger:       logger,
		queue:        make(chan live.Turn, turnQueueSize),
		done:         make(chan struct{}),
		drainTimeout: turnDrainTimeout,
		writeTimeout: turnWriteTimeout,
	}
	r.write = r.persist
	r.ctx, r.cancel = context.WithCancel(context.Background())
	return r
}

func (r *turnRecorder) start() *turnRecorder {
	go r.run()
	return r
}

// Record queues a finished turn, dropping it if the writer is too far behind.
func (r *turnRecorder) Record(turn Turn) {
	row := live.Turn{
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

// Close drains the queue and stops the writer, giving the writes drainTimeout between
// them. Rows still unwritten then are dropped and counted with the rest.
func (r *turnRecorder) Close() {
	r.closeOnce.Do(func() {
		r.closing.Lock()
		r.closed = true
		close(r.queue)
		r.closing.Unlock()

		drainWithin(r.done, r.drainTimeout, r.cancel)
		if dropped := r.dropped.Load(); dropped > 0 {
			r.logger.Warn("dropped turns because the writer fell behind", "count", dropped)
		}
	})
}

func (r *turnRecorder) run() {
	defer close(r.done)

	for row := range r.queue {
		if r.ctx.Err() != nil {
			r.dropped.Add(1)
			continue
		}
		ctx, cancel := context.WithTimeout(r.ctx, r.writeTimeout)
		err := r.write(ctx, row)
		// A write cut by its own timeout or by the drain deadline is a row lost, and is
		// counted either way; one the store refused is only logged by the writer.
		lost := err != nil && ctx.Err() != nil
		cancel()
		if lost {
			r.dropped.Add(1)
		}
	}
}

// persist writes a row to Redis and to Postgres, logging what fails.
func (r *turnRecorder) persist(ctx context.Context, row live.Turn) error {
	var failures []error
	if r.live != nil {
		if err := r.live.RecordTurn(ctx, row); err != nil {
			r.logger.Error("could not record turn", "error", err)
			failures = append(failures, err)
		}
	}
	if r.store != nil {
		if err := r.store.RecordTurnStats(ctx, sampleOf(row)); err != nil {
			r.logger.Error("could not count turn", "error", err)
			failures = append(failures, err)
		}
	}
	return errors.Join(failures...)
}

func sampleOf(turn live.Turn) store.TurnSample {
	sample := store.TurnSample{
		CustomerID:   turn.CustomerID,
		AgentID:      turn.AgentID,
		StartedAt:    turn.StartedAt,
		Interrupted:  turn.Interrupted,
		STTLatencyMs: turn.STTLatencyMs,
		LLMTTFTMs:    turn.LLMTTFTMs,
		TTSTTFBMs:    turn.TTSTTFBMs,
		RoundtripMs:  turn.RoundtripMs,
	}
	if turn.AudioOutMs != nil {
		sample.AudioOutMs = *turn.AudioOutMs
	}
	return sample
}

// measured keeps a leg that never happened out of the percentiles rather than counting
// it as instant.
func measured(ms float64) *float64 {
	if ms <= 0 {
		return nil
	}
	return &ms
}
