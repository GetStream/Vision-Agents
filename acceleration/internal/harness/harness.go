// Package harness sits between what a caller says and the model that answers them.
//
// The model on the live path is chosen for how quickly it starts talking, which is not
// the same as how well it thinks. Rather than accept that trade on every sentence, the
// harness gives that model somewhere to put the work it is bad at: it writes a request
// for help into its reply, the harness takes the request back out before the reply
// reaches the voice, and a slower model answers it while the conversation carries on.
//
// Nothing here waits. Delegating returns as soon as the request is on its way, and the
// answer is folded into a later turn whenever it arrives — or abandoned, if the caller
// has moved on by then.
package harness

import (
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
)

// eventBuffer is how many events may queue before a slow consumer applies backpressure.
const eventBuffer = 32

// defaultTasks caps delegated work in flight when the caller names no limit. A model
// that asks for help on every sentence would otherwise open a session's worth of them.
const defaultTasks = 3

// Options configures a Harness.
type Options struct {
	// Model is the fast voice model. The harness holds it because a turn goes to the
	// harness first and it is the harness that decides what to ask.
	Model *llmrouter.Session
	// Subagent is the slower, more capable model that runs delegated work. Without one
	// the harness offers no skills and the fast model answers everything itself. The
	// harness takes ownership of the session and closes it.
	Subagent *llmrouter.Session
	// Skills are what the fast model may hand over. They mean nothing without a subagent.
	Skills Skills
	// Tasks caps how much delegated work may run at once.
	Tasks int
	// MaxTokens caps each reply. Zero leaves the model's own default in place.
	MaxTokens int
	Logger    *slog.Logger
}

// Turn is what the harness is asked to answer.
type Turn struct {
	// ID correlates the reply with the turn the agent is measuring.
	ID string
	// Instructions is the agent's own system prompt, already carrying whatever it
	// recalled about the caller. The harness adds to it rather than replacing it.
	Instructions string
	// History is the conversation so far, oldest first, ending with what to reply to.
	History []llm.Message
	// Note is something true of this turn alone, such as the caller not having been
	// heard clearly. It is not remembered past the reply it shapes.
	Note string
}

// Harness decides what the fast model is asked and what becomes of what it answers.
type Harness struct {
	options Options
	logger  *slog.Logger
	emitter *Emitter
	// tasks is nil when there is no subagent, which is what makes delegation optional
	// rather than a second thing to configure before the agent works at all.
	tasks *manager

	mu sync.Mutex
	// notes are what has come back from the subagent since the fast model last spoke.
	// They are folded into the next prompt and then forgotten.
	notes []string
	// history is the conversation as of the last turn, so a request for help carries the
	// context it was asked in rather than only the sentence that prompted it.
	history []llm.Message

	// scan splits replies into speech and requests for help. Only the consumer of the
	// model's deltas touches it, which is one goroutine.
	scan scanner

	running   sync.WaitGroup
	closeOnce sync.Once
}

// New validates the options and returns a Harness. It opens nothing: the sessions it is
// given are already started.
func New(options Options) (*Harness, error) {
	if options.Model == nil {
		return nil, errors.New("harness: a model session is required")
	}
	if options.Tasks <= 0 {
		options.Tasks = defaultTasks
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	if err := options.Skills.Validate(); err != nil {
		return nil, err
	}

	h := &Harness{
		options: options,
		logger:  options.Logger,
		emitter: NewEmitter(eventBuffer),
	}

	if options.Subagent != nil {
		h.tasks = newManager(options.Subagent, options.Tasks, h.logger)
		h.running.Add(1)
		go h.consumeTasks()
	}
	return h, nil
}

// Respond asks the fast model to answer a turn. It returns once the request is on its
// way: the reply arrives on the model session's own events and is filtered by Filter.
func (h *Harness) Respond(turn Turn) error {
	h.mu.Lock()
	h.history = append([]llm.Message(nil), turn.History...)
	instructions := h.instructions(turn.Instructions, turn.Note)
	h.mu.Unlock()

	return h.options.Model.Respond(llm.Request{
		ID:           turn.ID,
		Instructions: instructions,
		Messages:     turn.History,
		MaxTokens:    h.options.MaxTokens,
	})
}

// Filter takes a delta of a reply and returns the part of it the caller should hear.
//
// A request for help is not speech, so it never reaches the voice. Everything else is
// released as soon as it is known not to be one, because the caller is listening to the
// gap.
func (h *Harness) Filter(turnID, delta string) string {
	speech, directives := h.scan.Add(delta)
	for _, found := range directives {
		h.act(turnID, found)
	}
	return speech
}

// Flush releases what the filter was still holding when a reply ended.
func (h *Harness) Flush() string { return h.scan.Flush() }

// Reset forgets a reply that was abandoned part-way through.
func (h *Harness) Reset() { h.scan.Reset() }

// Delegating reports whether the subagent is still working on something, which is what
// tells the agent the conversation is not finished even though nobody is talking.
func (h *Harness) Delegating() bool {
	if h.tasks == nil {
		return false
	}
	return h.tasks.Running() > 0
}

// Pending reports whether anything has come back that the caller has not been told
// about. The agent uses it to know it owes them a turn nobody asked for.
func (h *Harness) Pending() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.notes) > 0
}

// Events carries what the harness decided. It is closed by Close.
func (h *Harness) Events() <-chan Event { return h.emitter.Events() }

// Close abandons every task in flight and releases the subagent.
func (h *Harness) Close() error {
	var err error
	h.closeOnce.Do(func() {
		if h.tasks != nil {
			err = h.tasks.Close()
		}
		h.running.Wait()
		h.emitter.Close()
	})
	return err
}

// act carries out one request the model made of the harness.
func (h *Harness) act(turnID string, found directive) {
	if h.tasks == nil {
		return
	}

	if found.kind == kindDrop {
		h.tasks.CancelSkill(found.skill, ReasonDropped)
		return
	}

	skill, known := h.options.Skills.Lookup(found.skill)
	if !known {
		h.logger.Debug("the model asked for a skill that does not exist", "skill", found.skill)
		return
	}
	if strings.TrimSpace(found.body) == "" {
		return
	}

	h.mu.Lock()
	history := append([]llm.Message(nil), h.history...)
	h.mu.Unlock()

	taskID, err := h.tasks.Create(skill, found.body, history)
	if err != nil {
		h.logger.Error("could not delegate", "skill", skill.Name, "error", err)
		return
	}
	h.emitter.Send(Delegated{
		TaskID: taskID,
		Skill:  skill.Name,
		Prompt: found.body,
		TurnID: turnID,
	})
}

// consumeTasks folds finished work into the next prompt and reports it.
func (h *Harness) consumeTasks() {
	defer h.running.Done()

	for result := range h.tasks.Results() {
		if written := note(result); written != "" {
			h.mu.Lock()
			h.notes = append(h.notes, written)
			h.mu.Unlock()
		}
		h.emitter.Send(Settled{Result: result})
	}
}

// instructions is the system prompt for a turn: what the agent was told to be, what it
// may hand over, and whatever has come back since it last spoke. It must be called with
// the lock held, because taking the notes is what clears them.
func (h *Harness) instructions(agent, note string) string {
	parts := make([]string, 0, 4)
	if agent != "" {
		parts = append(parts, agent)
	}
	if h.tasks != nil {
		if index := h.options.Skills.Prompt(); index != "" {
			parts = append(parts, index)
		}
	}
	if len(h.notes) > 0 {
		parts = append(parts, strings.Join(h.notes, "\n"))
		h.notes = nil
	}
	if note != "" {
		parts = append(parts, note)
	}
	return strings.Join(parts, "\n\n")
}

// note is what the fast model needs to be told about a finished task, or empty when
// there is nothing worth saying. A cancelled task is not one: its premise is gone, so
// nobody is waiting to hear about it.
func note(result Result) string {
	switch {
	case result.Question != "":
		return fmt.Sprintf("Your colleague cannot finish the %s you asked for until the "+
			"caller answers this: %s. Ask them, in your own words.", result.Skill, result.Question)
	case result.Answered():
		return fmt.Sprintf("Your colleague has come back on the %s you asked for: %s. "+
			"Tell the caller, in your own words.", result.Skill, result.Text)
	case result.State == Failed:
		return fmt.Sprintf("The %s you asked for did not come back. Tell the caller you "+
			"could not find out, and carry on.", result.Skill)
	}
	return ""
}
