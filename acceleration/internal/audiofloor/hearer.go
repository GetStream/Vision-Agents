package audiofloor

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Mode is what an agent does with the model's answers.
type Mode string

const (
	// Shadow asks it about each turn beside the flow controller and acts on nothing it says:
	// the two answers are counted and logged side by side.
	Shadow Mode = "shadow"
	// Act asks it first, and its ruling is the flow controller's. The flow controller
	// decides the turns it doesn't answer in time.
	Act Mode = "act"
)

var (
	checks = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "audio_floor_checks_total",
		Help:      "Turns put to the audio floor model, by what became of them. Acting on it, every outcome but answered and superseded is a turn the flow controller decided instead.",
	}, []string{"outcome"})
	checkSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "audio_floor_check_seconds",
		Help:      "How long the audio floor model took to answer a turn, the request included.",
		Buckets:   []float64{.01, .025, .05, .1, .25, .5, 1, 2},
	})
)

// Hearer rules on the flow controller's turns from the speaker's audio. It is the
// harness.Hearer of an agent that acts on the model, and what shadow mode asks beside the
// flow controller.
type Hearer struct {
	client *Client
}

// NewHearer returns a Hearer asking client.
func NewHearer(client *Client) *Hearer { return &Hearer{client: client} }

// Check is one question put to the model about a turn, and what came of it.
type Check struct {
	Ruling harness.Decided
	Answer Answer
	// Bar is what p_eot had to reach, by Preset, for a turn with the floor free to have
	// ended, and Ended whether it had. Over the agent there is no bar, and the words' end
	// is the transcriber's to say.
	Bar    float64
	Preset string
	Ended  bool
}

// Hear is the ruling of Check.
func (h *Hearer) Hear(ctx context.Context, turn harness.FlowTurn) (harness.Decided, error) {
	check, err := h.Check(ctx, turn)
	return check.Ruling, err
}

// Check asks the model about turn and reads its answer: with the floor free, the turn has
// ended when p_eot reaches the policy's bar or the pause its cap, and otherwise the ruling
// is to wait for the policy's interval.
func (h *Hearer) Check(ctx context.Context, turn harness.FlowTurn) (Check, error) {
	start := time.Now()
	check, err := h.check(ctx, turn)
	checks.WithLabelValues(outcome(err)).Inc()
	if err == nil {
		checkSeconds.Observe(time.Since(start).Seconds())
	}
	return check, err
}

func (h *Hearer) check(ctx context.Context, turn harness.FlowTurn) (Check, error) {
	policy, err := h.client.Policy(ctx)
	if err != nil {
		return Check{}, err
	}
	answer, err := h.client.Decide(ctx, Agent{Instructions: turn.Instructions, Tools: turn.Tools}, turnOf(turn), turn.Audio, false)
	if err != nil {
		return Check{}, err
	}
	r := answer.Ruling
	check := Check{Answer: answer, Ended: true, Ruling: harness.Decided{
		Disposition: harness.Disposition(r.Disposition), Floor: harness.Floor(r.Floor), Tool: r.Tool, Cannot: r.Cannot,
	}}
	if !check.Ruling.Disposition.Valid() || !check.Ruling.Floor.Valid() {
		return Check{}, stack.Wrap(fmt.Errorf("audiofloor: a ruling the flow controller has no word for: %+v", *r))
	}
	if turn.Speaking {
		return check, nil
	}
	if answer.PEOT == nil {
		return Check{}, stack.Wrap(errors.New("audiofloor: no p_eot with the floor free"))
	}
	check.Bar, check.Preset = policy.Bar(turn.Paused, turn.FirstPause), policy.Default
	check.Ended = *answer.PEOT >= check.Bar || turn.Paused >= policy.Cap()
	if !check.Ended {
		check.Ruling = harness.Decided{Disposition: harness.Wait, Floor: harness.Continue, Again: policy.Interval()}
	}
	return check, nil
}

func outcome(err error) string {
	switch {
	case err == nil:
		return "answered"
	case errors.Is(err, context.Canceled):
		return "superseded"
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case errors.Is(err, ErrRefused):
		return "refused"
	case errors.Is(err, ErrUnsupported):
		return "unsupported"
	}
	return "error"
}

// turnOf says a flow controller's turn in the model's terms. With the floor free the audio
// says whether the words are finished, and the agent is saying nothing.
func turnOf(turn harness.FlowTurn) Turn {
	out := Turn{Speaker: caller, Speaking: turn.Speaking}
	if turn.AnotherVoice {
		out.Speaker = anotherVoice
	}
	if turn.Speaking {
		out.Said, out.Unfinished = turn.Reply, turn.Unfinished
	}
	for _, message := range turn.History {
		if message.Role == llm.System || strings.TrimSpace(message.Content) == "" {
			continue
		}
		out.Conversation = append(out.Conversation, Line{Agent: message.Role == llm.Assistant, Text: message.Content})
	}
	return out
}
