package agent

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// audioFloorPending bounds the turns still waiting on their other side's answer.
const audioFloorPending = 64

var audioFloorCompared = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "agent_chorus_asr_compared_total",
	Help:      "Turns both the flow controller and, in shadow mode, the chorus-asr model ruled on, by each one's disposition and floor, and the end-of-turn preset the model's was read by. The model's ruling is not acted on.",
}, []string{"controller_disposition", "model_disposition", "controller_floor", "model_floor", "preset"})

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
}

// audioFloorTurn is what each side said about a turn, and when the model was asked.
type audioFloorTurn struct {
	model                  *audiofloor.Check
	paused, took           time.Duration
	firstCheck, firstPause bool
	controller             *harness.Decided
}

// newAudioFloor returns nil unless the model is asked.
func newAudioFloor(client *audiofloor.Client, mode audiofloor.Mode, logger *slog.Logger) *audioFloor {
	if client == nil || (mode != audiofloor.Shadow && mode != audiofloor.Act) {
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &audioFloor{
		client:  client,
		hearer:  audiofloor.NewHearer(client),
		shadow:  mode == audiofloor.Shadow,
		logger:  logger,
		ctx:     ctx,
		cancel:  cancel,
		streams: map[string]*audiofloor.Stream{},
		turns:   map[string]*audioFloorTurn{},
	}
}

// acting is the flow controller's Hearer, or nil when the model only shadows it.
func (v *audioFloor) acting() harness.Hearer {
	if v == nil || v.shadow {
		return nil
	}
	return v.hearer
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

// listen gives turn what the model hears: the participant's last 16 s, and the customer's
// tools.
func (v *audioFloor) listen(turn *harness.FlowTurn, participant string, tools []harness.Tool) {
	if v == nil {
		return
	}
	if stream := v.stream(participant); stream != nil {
		turn.Audio = stream.Last(audiofloor.Window)
	}
	turn.Tools = customerTools(tools)
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
		turn.model.Preset).Inc()
	answer := turn.model.Answer
	var pEnd any
	if answer.PEOT != nil {
		pEnd = *answer.PEOT
	}
	v.logger.Debug("the chorus-asr model beside the flow controller", "candidate", candidate,
		"controller_disposition", controller.Disposition, "controller_floor", controller.Floor,
		"model_disposition", model.Disposition, "model_floor", model.Floor,
		"model_answer", answer.Answer, "model_p", answer.AnswerProbability, "p_end", pEnd,
		"preset", turn.model.Preset, "bar", turn.model.Bar, "ended", turn.model.Ended, "paused_ms", turn.paused.Milliseconds(),
		"first_check", turn.firstCheck, "first_pause", turn.firstPause,
		"tool", model.Tool, "cannot", model.Cannot, "took_ms", turn.took.Milliseconds())
}

// close stops a question still out and waits for it.
func (v *audioFloor) close() {
	if v == nil {
		return
	}
	v.cancel()
	v.running.Wait()
}
