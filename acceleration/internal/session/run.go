package session

import (
	"cmp"
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

// Usage is what a run's model calls spent, summed over every request.
type Usage struct {
	CachedInputTokens int64
	CostMicros        int64
	InputTokens       int64
	OutputTokens      int64
	Requests          int64
}

func (u *Usage) add(other Usage) {
	u.CachedInputTokens += other.CachedInputTokens
	u.CostMicros += other.CostMicros
	u.InputTokens += other.InputTokens
	u.OutputTokens += other.OutputTokens
	u.Requests += other.Requests
}

// ChildRan is a child agent a session handed work to having finished, with what its run
// spent. It is sent on the parent's session.
type ChildRan struct {
	// Tool is the name the parent's model called the child by.
	Tool  string
	Usage Usage
}

// RunOutcome is what a run's events add up to while it settles.
type RunOutcome struct {
	SessionID string
	FinalText string
	// Failed is a model failure no later reply made up for.
	Failed error
	// Usage is what the run spent, its children's runs included.
	Usage Usage
	// Children is what the runs of the child agents it handed work to spent, by tool.
	Children map[string]Usage
	// AnsweredBy is the provider/model of the last reply that was served, which is not the
	// session's first choice when that one failed and the reply fell back.
	AnsweredBy string
}

// see adds event to the outcome, and says whether it ends the run: an answer the agent
// gave with nothing left to do after it.
func (r *RunOutcome) see(event Event) bool {
	switch typed := event.(type) {
	case agent.Responded:
		if typed.Text != "" {
			r.FinalText, r.Failed = typed.Text, nil
		}
		return typed.Text != "" && !typed.PendingWork && typed.Purpose == ""
	case agent.Error:
		if typed.Context == "llm" || typed.Context == "flow" {
			r.Failed = typed.Err
		}
	case agent.ModelCall:
		r.Usage.add(Usage{
			InputTokens: typed.InputTokens, CachedInputTokens: typed.CachedInputTokens,
			OutputTokens: typed.OutputTokens, CostMicros: typed.CostMicros, Requests: 1,
		})
		if typed.Success && typed.Purpose == "reply" {
			r.AnsweredBy = typed.Provider + "/" + typed.Model
		}
	case ChildRan:
		r.Usage.add(typed.Usage)
		if r.Children == nil {
			r.Children = map[string]Usage{}
		}
		spent := r.Children[typed.Tool]
		spent.add(typed.Usage)
		r.Children[typed.Tool] = spent
	}
	return false
}

// NotStarted is a run whose session could not be created from its spec.
type NotStarted struct{ Err error }

func (n NotStarted) Error() string { return n.Err.Error() }
func (n NotStarted) Unwrap() error { return n.Err }

// Run answers input on a headless, incognito text session made from spec, waits until the
// agent has nothing left to do or ctx ends, and ends the session. A session that could not be
// created is a NotStarted. ctx's deadline is the run's: nothing the session hands work to
// outlives it.
func (m *Manager) Run(ctx context.Context, spec Spec, input string, images []llm.ImagePart) (RunOutcome, error) {
	spec.Text = true
	spec.CallID = ""
	spec.STSTarget = ""
	spec.Greeting = ""
	// Nothing about a run is kept, so one run's input never reaches the next one's.
	spec.Incognito = true
	if deadline, ok := ctx.Deadline(); ok {
		spec.RunDeadline = deadline
	}
	created, err := m.Create(ctx, spec)
	if err != nil {
		return RunOutcome{}, NotStarted{Err: err}
	}
	run := RunOutcome{SessionID: created.ID()}
	// The images are counted here because nothing else a run logs says whether it had any,
	// and an answer that says it saw no image is otherwise impossible to place.
	m.logger.Info("a run started", "customer", spec.CustomerID, "session", created.ID(), "config", spec.ConfigID,
		"images", len(images))
	events, detach := created.Watch()
	defer detach()
	defer func() {
		// A run given up on still has a reply in flight, which closing would wait out.
		if ctx.Err() != nil {
			created.Interrupt()
		}
		if _, err := m.Close(created.ID(), OwnerOf(created.Spec())); err != nil {
			m.logger.Error("could not end a run's session", "session", created.ID(), "error", err)
		}
	}()

	if _, err := created.Respond(ctx, input, images, options.LLM{}); err != nil {
		return run, err
	}
	created.Settle(ctx, events, run.see)
	_, model, _, _ := created.Resolved()
	run.AnsweredBy = cmp.Or(run.AnsweredBy, model)
	return run, nil
}
