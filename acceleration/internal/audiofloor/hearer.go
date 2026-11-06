package audiofloor

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Mode is what an agent does with the model's answers: its floor control, which each agent
// chooses for itself.
type Mode string

const (
	// Off asks it nothing about turns, as the empty mode does.
	Off Mode = "off"
	// Shadow asks it about each turn beside the flow controller and acts on nothing it says:
	// the two answers are counted and logged side by side.
	Shadow Mode = "shadow"
	// Act asks it first, and its ruling is the flow controller's. The flow controller
	// decides the turns it doesn't answer in time.
	Act Mode = "act"
)

// Asks says whether the model is asked about turns: in shadow, or acting.
func (m Mode) Asks() bool {
	return m == Shadow || m == Act
}

var (
	checks = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "chorus_asr_checks_total",
		Help:      "Turns put to the chorus-asr model, by what became of them. Acting on it, every outcome but answered and superseded is a turn the flow controller decided instead.",
	}, []string{"outcome"})
	checkSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "chorus_asr_check_seconds",
		Help:      "How long the chorus-asr model took to answer a turn, the request included.",
		Buckets:   []float64{.01, .025, .05, .1, .25, .5, 1, 2},
	})
)

// Hearer rules on the flow controller's turns from the speaker's audio. It is the
// harness.Hearer of an agent that acts on the model, and what shadow mode asks beside the
// flow controller. It ends a turn with the floor free by the agent's choice of a point, or
// bars of its own.
type Hearer struct {
	client    *Client
	endOfTurn EndOfTurn
	logger    *slog.Logger
	// unresolved is set once the agent's choice has been found to name nothing the server
	// publishes, so that it is said once per session rather than at each check, and said
	// is the name of the Ending last said to be the session's.
	unresolved atomic.Bool
	said       atomic.Value
	// inARow counts each participant's checks in a row that reached their bar.
	inARow runs
}

// NewHearer returns a Hearer asking client, ending a turn with the floor free by endOfTurn,
// or by the server's default preset when that names nothing the server publishes.
func NewHearer(client *Client, endOfTurn EndOfTurn, logger *slog.Logger) *Hearer {
	return &Hearer{client: client, endOfTurn: endOfTurn, logger: logger}
}

// Check is one question put to the model about a turn, and what came of it.
type Check struct {
	Ruling harness.Decided
	Answer Answer
	// Bar is what p_eot had to reach at this check, by Ending, InARow how many checks in a
	// row of this pause have reached their bar, this one included, and Ended whether the turn
	// with the floor free had ended: Ending's ChecksInARow of them, or the cap. Over the agent
	// there is no bar, and the words' end is the transcriber's to say.
	Bar    float64
	Ending Ending
	InARow int
	Ended  bool
}

// Hear is the ruling of Check.
func (h *Hearer) Hear(ctx context.Context, turn harness.FlowTurn) (harness.Decided, error) {
	check, err := h.Check(ctx, turn)
	return check.Ruling, err
}

// Check asks the model about turn and reads its answer: with the floor free, the turn has
// ended when p_eot has reached the bar at as many checks in a row as the agent's point asks,
// or the pause the cap, and otherwise the ruling is to wait for the policy's interval.
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
	answer, err := h.client.Decide(ctx, Agent{Instructions: turn.Instructions, Tools: turn.Tools}, turnOf(turn), turn.Audio)
	if err != nil {
		return Check{}, err
	}
	return h.rule(policy, turn, answer)
}

// rule reads the model's answer about turn by policy and the agent's choice of a point: the
// ruling of the Hearer's own checks and of a transcriber asking for the agent (Asker).
func (h *Hearer) rule(policy Policy, turn harness.FlowTurn, answer Answer) (Check, error) {
	r := answer.Ruling
	check := Check{Answer: answer, Ended: true, Ruling: harness.Decided{
		Disposition: harness.Disposition(r.Disposition), Floor: harness.Floor(r.Floor),
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
	check.Ending = h.ending(policy)
	check.Bar = policy.Bar(check.Ending.Bars, turn.Paused, turn.FirstPause)
	check.InARow = h.inARow.count(turn.ParticipantID, !turn.Waited, turn.Paused, *answer.PEOT >= check.Bar)
	check.Ended = check.InARow >= check.Ending.ChecksInARow || turn.Paused >= policy.Cap()
	if !check.Ended {
		check.Ruling = harness.Decided{Disposition: harness.Wait, Floor: harness.Continue, Again: policy.Interval()}
	}
	return check, nil
}

// ending is how the agent chose to end turns, under the server's policy. A choice the policy
// can't resolve, as when the server has moved to a model with other presets or waits since
// it was made, is said once and ends turns by the server's default. Which point that is,
// and what it was measured at, is said whenever it changes.
func (h *Hearer) ending(policy Policy) Ending {
	ending, err := policy.Resolve(h.endOfTurn)
	if err != nil {
		ending = policy.DefaultEnding()
		if h.unresolved.CompareAndSwap(false, true) {
			h.logger.Warn("the agent's end_of_turn names nothing the server publishes, so its turns end by the server's default",
				"error", err, "end_of_turn", ending.Name)
		}
	}
	if previous := h.said.Swap(ending.Name); previous != ending.Name {
		attributes := []any{"end_of_turn", ending.Name, "checks_in_a_row", ending.ChecksInARow, "dictation_guard", ending.Guard()}
		if point := ending.Point; point != nil {
			m := point.Measured
			attributes = append(attributes, "max_wait_ms", point.MaxWaitMs, "cut_ins", m.CutIns, "answer_ms_p90", m.AnswerMsP90,
				"dictation_pauses_cut", m.DictationPausesCut, "slow_counts_cut", m.SlowCountsCut, "slow_counts", m.SlowCounts)
		}
		h.logger.Info("chorus-asr ends this agent's turns by", attributes...)
	}
	return ending
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

// turnOf says a flow controller's turn in the model's terms, as its rows were written (floor's
// testdata): the conversation is what was said on the call, the caller's words and the agent's,
// not a turn typed or handed in by the app nor a tool's result; and over the agent, what it
// "has so far said" is the reply as far as the caller has heard it, which is no line of the
// conversation before it. With the floor free the audio says whether the words are finished,
// and the agent is saying nothing.
func turnOf(turn harness.FlowTurn) Turn {
	out := Turn{Speaker: caller, Speaking: turn.Speaking}
	if turn.AnotherVoice {
		out.Speaker = anotherVoice
	}
	// replying is where the reply being said is in the history, -1 when it is not: the last
	// line said on the call, whatever the router kept after it (a tool's result, a turn typed
	// in), when that line is the agent's and is the reply.
	replying := -1
	if turn.Speaking {
		out.Said, out.Unfinished = turn.Reply, turn.Unfinished
		if turn.Measured {
			out.Said = turn.Heard
		}
		for i := len(turn.History) - 1; i >= 0; i-- {
			if !saidOnTheCall(turn.History[i]) {
				continue
			}
			if turn.History[i].Role == llm.Assistant && sameText(turn.History[i].Content, turn.Reply) {
				replying = i
			}
			break
		}
	}
	for i, message := range turn.History {
		if i == replying || !saidOnTheCall(message) {
			continue
		}
		out.Conversation = append(out.Conversation, Line{Agent: message.Role == llm.Assistant, Text: message.Content})
	}
	return out
}

// saidOnTheCall reports whether a message is a line said on the call, the agent's or a
// caller's: not a turn typed or handed in, a tool's result or a note, and not empty.
func saidOnTheCall(message llm.Message) bool {
	said := message.Role == llm.Assistant || message.Role == llm.User && !message.Typed
	return said && strings.TrimSpace(message.Content) != ""
}

// sameText reports whether two texts are the same words, spaced alike or not.
func sameText(a, b string) bool {
	return strings.Join(strings.Fields(a), " ") == strings.Join(strings.Fields(b), " ")
}
