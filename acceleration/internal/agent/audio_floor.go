package agent

import (
	"context"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// audioFloorPending bounds the turns still waiting on their other side's answer.
const audioFloorPending = 64

var audioFloorCompared = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "agent_chorus_asr_compared_total",
	Help:      "Turns both the flow controller and, in shadow mode, chorus-asr ruled on, by each one's disposition and floor, and how the model's was read: end_of_turn, preset:NAME, max_wait_ms:N for the measured point fitted to N, or custom for the agent's own bars, and dictation_guard, whether that point passes the dictation guard (pass, fail, or unmeasured for the agent's own). The model's ruling is not acted on.",
}, []string{"controller_disposition", "model_disposition", "controller_floor", "model_floor", "end_of_turn", "dictation_guard"})

// audioFloor keeps each participant's last 16 s for the chorus-asr model, encoded once as
// it arrives. Acting, the model is the flow controller's Hearer. In shadow mode it is asked
// about each turn beside the flow controller and nothing it says is acted on: the two
// rulings are counted and logged side by side, so that real calls can say whether it should
// act. A shadow question costs a call nothing, as none is waited for and one still out
// drops the next.
//
// Nil is off, and every method of a nil one does nothing.
type audioFloor struct {
	client *audiofloor.Client
	hearer *audiofloor.Hearer
	shadow bool
	logger *slog.Logger
	ctx    context.Context
	cancel context.CancelFunc
	// busy is a shadow question still out, and running the goroutine asking it.
	busy    atomic.Bool
	running sync.WaitGroup

	mu sync.Mutex
	// streams holds each participant's, or nil for one that could not be kept.
	streams map[string]*audiofloor.Stream
	// turns holds what each side said about a turn until the other has.
	turns map[string]*audioFloorTurn
	// checked are the rulings the transcriber's pause checks made, acting, each until the
	// flow controller asks about its words, and settled the candidates they were found for.
	checked []ruledWords
	settled map[string]harness.Decided
}

// ruledWords is a ruling a pause check made, on a participant's words.
type ruledWords struct {
	participant string
	ruled       audiofloor.Ruled
	at          time.Time
}

// ruledFor is how long a pause check's ruling waits for the flow controller to ask about its
// words, which it does as soon as they settle: a final is settled in tens of milliseconds.
const ruledFor = 5 * time.Second

// audioFloorTurn is what each side said about a turn, and when the model was asked.
type audioFloorTurn struct {
	model                  *audiofloor.Check
	paused, took           time.Duration
	firstCheck, firstPause bool
	controller             *harness.Decided
}

// newAudioFloor returns nil unless the model is asked. A turn with the floor free ends by the
// bars endOfTurn chooses.
func newAudioFloor(client *audiofloor.Client, mode audiofloor.Mode, endOfTurn audiofloor.EndOfTurn, logger *slog.Logger) *audioFloor {
	if client == nil || (mode != audiofloor.Shadow && mode != audiofloor.Act) {
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &audioFloor{
		client:  client,
		hearer:  audiofloor.NewHearer(client, endOfTurn, logger),
		shadow:  mode == audiofloor.Shadow,
		logger:  logger,
		ctx:     ctx,
		cancel:  cancel,
		streams: map[string]*audiofloor.Stream{},
		turns:   map[string]*audioFloorTurn{},
		settled: map[string]harness.Decided{},
	}
}

// acting is the flow controller's Hearer, or nil when the model only shadows it. It rules on
// the words a pause check already ruled on as that check did, and asks about any others.
func (v *audioFloor) acting() harness.Hearer {
	if v == nil || v.shadow {
		return nil
	}
	return ruledFirst{v}
}

// ruledFirst is the Hearer of an agent that acts on the model.
type ruledFirst struct{ v *audioFloor }

func (r ruledFirst) Hear(ctx context.Context, turn harness.FlowTurn) (harness.Decided, error) {
	r.v.mu.Lock()
	ruled, ok := r.v.settled[turn.ID]
	delete(r.v.settled, turn.ID)
	r.v.mu.Unlock()
	if ok {
		return ruled, nil
	}
	return r.v.hearer.Hear(ctx, turn)
}

// asksOnPauses says the transcriber is to ask about each pause for the agent: acting, the
// model decides when the turn ends, from the audio, and writes its words in the same pass.
func (v *audioFloor) asksOnPauses() bool { return v != nil && !v.shadow }

// keep holds a pause check's ruling on participant's words for the flow controller.
func (v *audioFloor) keep(participant string, ruled audiofloor.Ruled) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.checked = slices.DeleteFunc(v.checked, func(r ruledWords) bool { return time.Since(r.at) > ruledFor })
	v.checked = append(v.checked, ruledWords{participant: participant, ruled: ruled, at: time.Now()})
}

// findRuled finds the pause check's ruling on turn's words, if one made one: the
// participant's, on words the turn's end with. A check over the agent ended the turn at its
// pause, and its ruling stands once the agent has gone quiet, as asking again with the floor
// free would wait for the turn's end a second time; a check with the floor free heard nothing
// of the agent's speech, so with the agent talking since there is none.
func (v *audioFloor) findRuled(said, participant string, speaking bool, turnID string) {
	v.mu.Lock()
	defer v.mu.Unlock()
	said = strings.ToLower(words(said))
	for i, r := range v.checked {
		if r.participant != participant || (speaking && !r.ruled.Speaking) || time.Since(r.at) > ruledFor {
			continue
		}
		if heard := strings.ToLower(words(r.ruled.Words)); heard == "" || !strings.HasSuffix(said, heard) {
			continue
		}
		v.checked = slices.Delete(v.checked, i, i+1)
		if len(v.settled) >= audioFloorPending {
			clear(v.settled)
		}
		v.settled[turnID] = r.ruled.Ruling
		return
	}
}

// heard keeps a frame of a participant's audio. It runs once per frame.
func (v *audioFloor) heard(inbound InboundAudio) {
	if v == nil || inbound.Audio.SampleRate != audiofloor.SampleRate || inbound.Audio.Channels != 1 {
		return
	}
	stream := v.stream(inbound.Participant.ID)
	if stream == nil {
		return
	}
	if err := stream.Write(inbound.Audio.Samples); err != nil {
		v.logger.Debug("the chorus-asr model missed a frame", "participant", inbound.Participant.ID, "error", err)
	}
}

func (v *audioFloor) stream(participant string) *audiofloor.Stream {
	v.mu.Lock()
	defer v.mu.Unlock()
	stream, ok := v.streams[participant]
	if !ok {
		var err error
		if stream, err = v.client.NewStream(audiofloor.Window); err != nil {
			v.logger.Warn("the chorus-asr model will not hear a participant", "participant", participant, "error", err)
		}
		v.streams[participant] = stream
	}
	return stream
}

// forget lets go of what was kept for a participant who has left: their audio and its
// encoder, and any ruling on their words still waiting for the flow controller. A call that
// people come and go from would otherwise keep every one of them until it ended.
func (v *audioFloor) forget(participant string) {
	if v == nil {
		return
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	delete(v.streams, participant)
	v.checked = slices.DeleteFunc(v.checked, func(r ruledWords) bool { return r.participant == participant })
}

// listen gives turn what the model hears: the participant's last 16 s, and the customer's
// tools, and finds the ruling a pause check made on its words, if one did: said, the words of
// the utterance whole, which the check heard, whatever of them the turn is answered on. The
// turn names the participant by ID, which the model's checks of them are counted by:
// Participant is how the flow controller's prompt calls them, and two callers may be called
// alike.
func (v *audioFloor) listen(turn *harness.FlowTurn, participant, said string, tools []harness.Tool) {
	if v == nil {
		return
	}
	turn.ParticipantID = participant
	turn.Tools = customerTools(tools)
	v.findRuled(said, participant, turn.Speaking, turn.ID)
	if stream := v.stream(participant); stream != nil {
		turn.Audio = stream.Last(audiofloor.Window)
	}
}

// pauseAsker is what an acting agent tells its chorus-asr transcriber about the call, for it
// to ask about each of the caller's pauses (audiofloor.Asker).
type pauseAsker struct{ a *Agent }

// Moment is the call as ask would put a participant's words to the flow controller now.
func (p pauseAsker) Moment(participant stt.Participant) harness.FlowTurn {
	a := p.a
	a.mu.Lock()
	history := llm.OmitImages(harness.WithoutNotes(a.history))
	instructions := a.prompted()
	if line := severalPeople(len(a.speakers), a.writers); line != "" {
		instructions += "\n\n" + line
	}
	speaking := a.generating || a.utterances > 0 || a.pendingTools > 0
	reply, unsaid := a.replyLocked(history)
	heard, measured := a.replyHeardLocked(reply, time.Now())
	tools := a.options.Tools.Tools
	a.mu.Unlock()
	return harness.FlowTurn{
		Instructions: instructions,
		History:      history,
		Speaking:     speaking || a.speechPending(),
		Reply:        reply,
		Heard:        heard,
		Measured:     measured,
		Unsaid:       unsaid,
		Tools:        customerTools(tools),
	}
}

// Hearer is the agent's, which reads each check by the agent's choice of how to end a turn.
func (p pauseAsker) Hearer() *audiofloor.Hearer { return p.a.audioFloor.hearer }

// Ruled keeps a pause check's ruling for the flow controller, which asks about its words
// once the transcriber has sent them: the words as the conversation will see them, the
// agent's own voice heard back dropped from their end as it is from the final (consumeSTT),
// which is what the turn's words are matched against (findRuled).
func (p pauseAsker) Ruled(participant stt.Participant, ruled audiofloor.Ruled) {
	ruled.Words, _ = p.a.withoutEcho(transcriptOf(participant, ruled))
	p.a.audioFloor.keep(participant.ID, ruled)
}

// transcriptOf is a check's words as a transcript of them: what the agent tells its own
// echo from the caller by.
func transcriptOf(participant stt.Participant, ruled audiofloor.Ruled) stt.Transcript {
	return stt.Transcript{Participant: participant, Mode: stt.ModeReplacement, Text: ruled.Words, Provider: audiofloor.ProviderName,
		ProcessingTimeMs: ruled.WordsMs, AudioDurationMs: ruled.AudioMs}
}

// Hearing has the agent take the caller as heard now, while they are in an utterance the
// model has not ended: a silence is the line's, which the agent may fill with a check-in, only
// once nobody is talking. Their words come at a pause or at the turn's end, and with interims
// off or the words so far unstable, a caller counting slowly was checked in on mid-count.
func (p pauseAsker) Hearing(stt.Participant) {
	p.a.mu.Lock()
	defer p.a.mu.Unlock()
	p.a.lastHeardAt = time.Now()
}

// Speaking says the agent is talking, as Moment tells the model.
func (p pauseAsker) Speaking() bool {
	a := p.a
	a.mu.Lock()
	speaking := a.generating || a.utterances > 0 || a.pendingTools > 0
	a.mu.Unlock()
	return speaking || a.speechPending()
}

// Overheard acts at once on a check's ruling to stop or shorten over the agent, asked while
// the caller talked on, as the agent does on a provisional ruling about a revision's words
// (overlapRuled): its own voice heard back on the caller's line is dropped from the words
// first, as from any chorus-asr transcript, and an echo alone, or a murmur, is no reason to
// stop.
func (p pauseAsker) Overheard(participant stt.Participant, ruled audiofloor.Ruled) {
	a := p.a
	left, ok := a.droppingEcho(transcriptOf(participant, ruled), "a check over the agent")
	if !ok {
		return
	}
	if !takesFloor(left) {
		a.logger.Debug("a check over the agent ruled on a murmur, which is no reason to stop",
			"participant", participant.ID, "text", left, "floor", ruled.Ruling.Floor)
		return
	}
	a.act(a.converse.Overheard(ruled.Ruling, participant, left, ruled.Utterance, a.floor()))
}

// customerTools are the tools a customer declared. The router's own stay out of the model's
// prompt: listing them flips its answers on natural requests.
func customerTools(tools []harness.Tool) []llm.Tool {
	var out []llm.Tool
	for _, tool := range tools {
		switch {
		case telephonyTool(tool.Name), tool.Name == toolTransfer, tool.Name == toolPress, tool.Name == toolEndCall,
			tool.Name == toolLookup, tool.Name == toolSearch, tool.Name == confirmTool, tool.Name == VideoFramesTool:
			continue
		}
		out = append(out, llm.Tool{Name: tool.Name, Description: tool.Description, Parameters: tool.Parameters})
	}
	return out
}

// ask puts turn to the model beside the flow controller, in shadow mode, without waiting.
func (v *audioFloor) ask(turn harness.FlowTurn) {
	if v == nil || !v.shadow {
		return
	}
	if !v.busy.CompareAndSwap(false, true) {
		return
	}
	v.running.Add(1)
	go func() {
		defer v.running.Done()
		defer v.busy.Store(false)
		ctx, cancel := context.WithTimeout(v.ctx, harness.HearDeadline(turn))
		defer cancel()
		start := time.Now()
		check, err := v.hearer.Check(ctx, turn)
		if err != nil {
			v.logger.Debug("the chorus-asr model did not rule", "candidate", turn.ID, "error", err)
			return
		}
		v.settle(turn.ID, func(t *audioFloorTurn) {
			t.model, t.took = &check, time.Since(start)
			t.paused, t.firstCheck, t.firstPause = turn.Paused, !turn.Waited, turn.FirstPause
		})
	}()
}

// ruled takes what the flow controller decided.
func (v *audioFloor) ruled(decided harness.Decided) {
	if v == nil || !v.shadow || !decided.Valid() {
		return
	}
	v.settle(decided.CandidateID, func(t *audioFloorTurn) { t.controller = &decided })
}

func (v *audioFloor) settle(candidate string, say func(*audioFloorTurn)) {
	v.mu.Lock()
	turn, ok := v.turns[candidate]
	if !ok {
		if len(v.turns) >= audioFloorPending {
			// A turn only one side answered waits for nothing else.
			clear(v.turns)
		}
		turn = &audioFloorTurn{}
		v.turns[candidate] = turn
	}
	say(turn)
	both := turn.model != nil && turn.controller != nil
	if both {
		delete(v.turns, candidate)
	}
	v.mu.Unlock()
	if !both {
		return
	}
	model, controller := turn.model.Ruling, turn.controller
	audioFloorCompared.WithLabelValues(string(controller.Disposition), string(model.Disposition), string(controller.Floor), string(model.Floor),
		turn.model.Ending.Name, turn.model.Ending.Guard()).Inc()
	answer := turn.model.Answer
	var pEnd any
	if answer.PEOT != nil {
		pEnd = *answer.PEOT
	}
	var tool string
	var cannot bool
	if answer.Ruling != nil {
		tool, cannot = answer.Ruling.Tool, answer.Ruling.Cannot
	}
	v.logger.Debug("the chorus-asr model beside the flow controller", "candidate", candidate,
		"controller_disposition", controller.Disposition, "controller_floor", controller.Floor,
		"model_disposition", model.Disposition, "model_floor", model.Floor,
		"model_answer", answer.Answer, "model_p", answer.AnswerProbability, "p_end", pEnd,
		"end_of_turn", turn.model.Ending.Name, "dictation_guard", turn.model.Ending.Guard(),
		"bar", turn.model.Bar, "in_a_row", turn.model.InARow, "checks_in_a_row", turn.model.Ending.ChecksInARow, "ended", turn.model.Ended, "paused_ms", turn.paused.Milliseconds(),
		"first_check", turn.firstCheck, "first_pause", turn.firstPause,
		"tool", tool, "cannot", cannot, "took_ms", turn.took.Milliseconds())
}

// close stops a question still out and waits for it.
func (v *audioFloor) close() {
	if v == nil {
		return
	}
	v.cancel()
	v.running.Wait()
}
