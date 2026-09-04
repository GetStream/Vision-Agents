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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
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
	// Controller is a second fast-model session that decides when evolving speech is
	// complete, relevant, or interrupting. The harness takes ownership of it.
	Controller *llmrouter.Session
	// Skills are what the fast model may hand over. They mean nothing without a subagent.
	Skills Skills
	// Tools are what the fast model may do rather than say. The harness offers them and
	// reports what was asked for; the agent is what runs them.
	Tools Tools
	// Sandbox is where the subagent runs code it writes. It is offered to the subagent
	// alone: running code takes seconds, and the model holding the conversation does not
	// have seconds. Nil means the subagent works everything out in its head.
	Sandbox sandbox.Sandbox
	// Tasks caps how much delegated work may run at once.
	Tasks int
	// MaxTokens caps each reply. Zero leaves the model's own default in place.
	MaxTokens int
	// CacheKey buckets this agent's requests in the provider's prompt cache, so the
	// instructions every one of them opens with are written once and read back after.
	// It is shared by every call the agent takes and means nothing to a provider whose
	// capabilities do not report PromptCacheKey.
	CacheKey string
	Logger   *slog.Logger
}

// noted is something for the fast model to be told, and the skill it came from. The skill
// is empty for a note about the harness itself rather than about work that ran. asking
// marks a note that is a question for the caller rather than something to pass on.
type noted struct {
	skill  string
	text   string
	asking bool
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
	flow  *flow

	mu sync.Mutex
	// notes are what has come back from the subagent since the fast model last spoke.
	// They are folded into the next prompt and then forgotten.
	notes []noted
	// reporting is the skills whose answers the reply being written was handed, so it can
	// be stopped from handing the same work back.
	reporting []string
	// asking is whether the reply being written carries a colleague's question for the
	// caller, which is the one turn that has nothing to look up.
	asking bool
	// asked is tool calls the model wrote as skill tags. They wait until the reply
	// finishes, because running one mid-sentence would start a second turn while the
	// first is still being spoken.
	asked []llm.ToolCall
	// askN numbers those calls so each has an id the provider can match a result to.
	askN int
	// history is the conversation as of the last turn, so a request for help carries the
	// context it was asked in rather than only the sentence that prompted it.
	history []llm.Message
	// stored is what the provider has already read of this conversation, when it keeps
	// what it generates.
	stored stored
	// compaction is private summary work in flight.
	compaction *compaction

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
	if err := options.Tools.Validate(); err != nil {
		return nil, err
	}

	h := &Harness{
		options: options,
		logger:  options.Logger,
		emitter: NewEmitter(eventBuffer),
	}

	if options.Subagent != nil {
		h.tasks = newManager(options.Subagent, options.Tasks, options.Sandbox, h.logger)
		h.running.Add(1)
		go h.consumeTasks()
	}
	if options.Controller != nil {
		h.flow = newFlow(options.Controller, h.emitter, h.logger)
	}
	return h, nil
}

// stored is what a provider that keeps what it generates has already read.
//
// The whole conversation is sent again by default, because routing may answer consecutive
// turns from different providers and a conversation held by the caller survives that. This
// is the shortcut taken when it is provably safe: the same model, the same instructions,
// and an input that only appends to what it last saw.
type stored struct {
	// responseID is what the provider called the last reply it kept.
	responseID string
	// identity is the provider and model that kept it. A failover changes it, and the
	// shortcut is dropped rather than offered to a provider that never saw the response.
	identity string
	// instructions is what that reply was told. Changed instructions mean the turn is no
	// longer a continuation of it.
	instructions string
	// sent is exactly the input the provider has read, so what to append can be worked
	// out rather than assumed. Compaction rewrites the conversation from the front, and
	// what it produces is not a continuation of anything.
	sent []llm.Message
}

// Remember records the reply a provider kept, so the next turn can continue from it.
func (h *Harness) Remember(response llm.Response) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if response.ProviderResponseID == "" {
		h.stored = stored{}
		return
	}
	h.stored.responseID = response.ProviderResponseID
	h.stored.identity = response.Provider + "/" + response.Model
}

// resume works out how much of the input still has to be sent, and what to continue from.
//
// It returns the whole input and no previous response whenever the shortcut cannot be
// proven safe, which costs bandwidth and nothing else.
func (h *Harness) resume(instructions string, input []llm.Message) ([]llm.Message, string) {
	model := h.options.Model.Capabilities()
	if !model.Store {
		return input, ""
	}

	previous := h.stored
	h.stored.instructions = instructions
	h.stored.sent = input

	if previous.responseID == "" ||
		previous.identity != h.options.Model.Provider()+"/"+h.options.Model.Model() ||
		previous.instructions != instructions ||
		!appendsTo(previous.sent, input) {
		h.stored.responseID = ""
		return input, ""
	}
	return input[len(previous.sent):], previous.responseID
}

// appendsTo reports whether the longer conversation starts with the shorter one unchanged.
func appendsTo(sent, input []llm.Message) bool {
	if len(sent) > len(input) {
		return false
	}
	for i, message := range sent {
		if !sameMessage(message, input[i]) {
			return false
		}
	}
	return true
}

func sameMessage(a, b llm.Message) bool {
	if a.Role != b.Role || a.Content != b.Content || a.ToolCallID != b.ToolCallID ||
		len(a.ToolCalls) != len(b.ToolCalls) {
		return false
	}
	for i, call := range a.ToolCalls {
		if call != b.ToolCalls[i] {
			return false
		}
	}
	return true
}

// Respond asks the fast model to answer a turn and returns the stream the reply arrives
// on. The caller drains it and passes each delta through Filter.
func (h *Harness) Respond(ctx context.Context, turn Turn) (*llm.Stream, error) {
	h.mu.Lock()
	h.history = append([]llm.Message(nil), turn.History...)
	instructions := h.instructions(turn.Instructions, turn.Note)
	// A colleague asks only for what the caller alone can say, so a reply carrying its
	// question has nothing to look up. Left holding a tool the model reaches for one and
	// narrates the reaching instead, and the question never reaches the caller.
	tools := h.options.Tools.Requests()
	if h.asking {
		tools = nil
	}
	input, previous := h.resume(instructions, answerable(turn.History))
	model := h.options.Model.Capabilities()
	h.mu.Unlock()

	return h.options.Model.Create(ctx, llm.ResponseParams{
		ID:                 turn.ID,
		Instructions:       instructions,
		Input:              input,
		MaxOutputTokens:    h.options.MaxTokens,
		Tools:              tools,
		Store:              model.Store,
		PreviousResponseID: previous,
		// Every turn of every call this agent takes opens with the same instructions, so
		// they are written to the provider's cache once under a key the agent owns and
		// read back from there on every turn after.
		PromptCacheKey: h.options.CacheKey,
	})
}

// resumption is what a turn nobody prompted is asked, when the conversation so far ends
// with the agent's own words. What came back is already in the instructions, so this only
// has to say that it has.
const resumption = "The work you handed over has come back, and is in your instructions. " +
	"Tell the caller what it found."

// answerable returns a history with something at the end of it to reply to.
//
// A completion answers whatever came last, so a history ending in the assistant's own turn
// asks the model to follow itself. Providers disagree on what that means and Gemini refuses
// it outright, which left the caller never hearing what came back. Only a turn nobody
// prompted ends that way: the agent builds one when delegated work returns and there is no
// new sentence from the caller to answer.
//
// The added turn belongs to the request rather than the conversation, so it is not kept.
func answerable(history []llm.Message) []llm.Message {
	if len(history) == 0 || history[len(history)-1].Role != llm.Assistant {
		return history
	}
	return append(append([]llm.Message(nil), history...),
		llm.Message{Role: llm.User, Content: resumption})
}

// Requested reports the tools the model asked to have run in one reply.
//
// A call for a tool that was never offered is dropped rather than passed on: models invent
// them, and the agent should not have to know which names are real. Running the rest is the
// agent's job, because a tool acts on the call and the harness does not know there is one.
func (h *Harness) Requested(turnID string, calls []llm.ToolCall) {
	for _, call := range calls {
		if _, known := h.options.Tools.Lookup(call.Name); !known {
			h.logger.Debug("the model asked for a tool that does not exist", "tool", call.Name)
			continue
		}
		h.emitter.Send(ToolRequested{TurnID: turnID, Call: call})
	}
}

// Decide asks the fast flow controller what to do with an evolving transcript.
func (h *Harness) Decide(turn FlowTurn) error {
	if h.flow == nil {
		return errors.New("harness: a flow controller is required")
	}
	return h.flow.Decide(turn)
}

// CancelDecision abandons a flow decision whose transcript changed.
func (h *Harness) CancelDecision(candidateID string) error {
	if h.flow == nil {
		return nil
	}
	return h.flow.Cancel(candidateID)
}

// CancelTurn abandons delegated work whose conversational premise changed.
func (h *Harness) CancelTurn(turnID, reason string) {
	if h.tasks != nil {
		h.tasks.CancelTurn(turnID, reason)
	}
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
func (h *Harness) Reset() {
	h.scan.Reset()
	h.mu.Lock()
	h.asked = nil
	h.mu.Unlock()
}

// TakeAsked returns the tool calls the model wrote as skill tags in the reply that just
// finished, and forgets them. The agent records them on the turn before running them, so a
// provider that matches results to calls is not handed an answer to nothing.
func (h *Harness) TakeAsked() []llm.ToolCall {
	h.mu.Lock()
	defer h.mu.Unlock()
	asked := h.asked
	h.asked = nil
	return asked
}

// Delegating reports whether the subagent is still working on something, which is what
// tells the agent the conversation is not finished even though nobody is talking.
func (h *Harness) Delegating() bool {
	if h.tasks == nil {
		return false
	}
	return h.tasks.RunningPublic() > 0
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
		var failures []error
		if h.flow != nil {
			if failure := h.flow.Close(); failure != nil {
				failures = append(failures, failure)
			}
		}
		if h.tasks != nil {
			if failure := h.tasks.Close(); failure != nil {
				failures = append(failures, failure)
			}
		}
		h.running.Wait()
		h.emitter.Close()
		err = errors.Join(failures...)
	})
	return err
}

// act carries out one request the model made of the harness.
func (h *Harness) act(turnID string, found directive) {
	if found.kind == kindDrop {
		if h.tasks != nil {
			h.tasks.CancelSkill(found.skill, ReasonDropped)
		}
		return
	}

	skill, known := h.options.Skills.Lookup(found.skill)
	if !known {
		h.actUnknown(found)
		return
	}
	if h.tasks == nil || strings.TrimSpace(found.body) == "" {
		return
	}

	h.mu.Lock()
	// This reply was written to tell the caller what the colleague said, and handing the
	// same work straight back is how the two of them talk to each other: the colleague
	// asks its question again, the answer earns another turn, and round it goes without
	// the caller ever being asked anything. Models do it with the question verbatim.
	if slices.Contains(h.reporting, skill.Name) {
		h.mu.Unlock()
		h.logger.Debug("the model handed back the work it was just told about",
			"skill", skill.Name, "prompt", found.body)
		return
	}
	history := append([]llm.Message(nil), h.history...)
	h.mu.Unlock()

	taskID, err := h.tasks.Create(skill, found.body, history, turnID, false)
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

// actUnknown handles a request for a skill that was never declared. Voice models mix up
// the two ways of asking for help: they write a skill tag for a tool they were offered,
// then go quiet waiting for a colleague who is not coming. A name that is a tool is run
// as one. Anything else is a note, so the next turn can tell the caller rather than hang.
func (h *Harness) actUnknown(found directive) {
	if tool, offered := h.options.Tools.Lookup(found.skill); offered {
		if strings.TrimSpace(found.body) == "" {
			return
		}
		h.logger.Debug("the model asked for a tool as if it were a skill", "tool", tool.Name)
		h.mu.Lock()
		h.askN++
		h.asked = append(h.asked, llm.ToolCall{
			ID:        fmt.Sprintf("ask-%d", h.askN),
			Name:      tool.Name,
			Arguments: argumentsFor(tool, found.body),
		})
		h.mu.Unlock()
		return
	}

	h.logger.Debug("the model asked for a skill that does not exist", "skill", found.skill)
	h.mu.Lock()
	h.notes = append(h.notes, noted{text: fmt.Sprintf(
		"There is no skill named %s. Carry on with the caller.", found.skill)})
	h.mu.Unlock()
}

// argumentsFor turns the body of a skill tag into the JSON object the tool expects. The
// tag is prose, so it is filed under query when the tool has one, otherwise under the
// first required argument. A body that is already an object is left alone.
func argumentsFor(tool Tool, body string) string {
	body = strings.TrimSpace(body)
	if json.Valid([]byte(body)) && strings.HasPrefix(body, "{") {
		return body
	}

	name := "query"
	if props, ok := tool.Parameters["properties"].(map[string]any); ok {
		if _, exists := props["query"]; !exists {
			name = firstRequired(tool.Parameters["required"])
			if name == "" {
				name = "query"
			}
		}
	}

	raw, err := json.Marshal(map[string]string{name: body})
	if err != nil {
		return "{}"
	}
	return string(raw)
}

func firstRequired(required any) string {
	switch names := required.(type) {
	case []any:
		if len(names) > 0 {
			if name, ok := names[0].(string); ok {
				return name
			}
		}
	case []string:
		if len(names) > 0 {
			return names[0]
		}
	}
	return ""
}

// consumeTasks folds finished work into the next prompt and reports it.
func (h *Harness) consumeTasks() {
	defer h.running.Done()

	for result := range h.tasks.Results() {
		if result.Skill == compactionSkillName {
			h.finishCompaction(result)
			continue
		}
		if written := note(result); written != "" {
			h.mu.Lock()
			h.notes = append(h.notes, noted{
				skill:  result.Skill,
				text:   written,
				asking: result.Question != "",
			})
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
	// Taking the notes is also what settles which skills this turn is reporting on, so a
	// reply written to deliver an answer cannot ask for that answer again.
	h.reporting = nil
	h.asking = false
	if len(h.notes) > 0 {
		lines := make([]string, 0, len(h.notes))
		for _, written := range h.notes {
			lines = append(lines, written.text)
			if written.skill != "" {
				h.reporting = append(h.reporting, written.skill)
			}
			h.asking = h.asking || written.asking
		}
		parts = append(parts, strings.Join(lines, "\n"))
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
	case result.State == Failed, result.Reason == ReasonDeadline:
		return fmt.Sprintf("The %s you asked for did not come back. Tell the caller you "+
			"could not find out, and carry on.", result.Skill)
	}
	return ""
}
