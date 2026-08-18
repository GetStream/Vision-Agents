package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// resultBuffer is how many finished tasks may queue before the manager waits on the
// harness to read them.
const resultBuffer = 16

// toolRounds caps how many times one task may run code before it has to answer. The
// deadline bounds it too, but a model that has written the same broken program four times
// is not about to write a fifth that works.
const toolRounds = 4

// codeDeadline bounds one round of running code, which is not on the live path but is
// still part of an answer somebody is waiting for.
const codeDeadline = 60 * time.Second

// manager runs delegated work on the subagent and reports what came of it.
//
// A task is one completion on the subagent's session, so the completion id is the task
// id and abandoning a task is one targeted interrupt. Nothing here waits: Create returns
// as soon as the request is on its way, and the answer arrives on Results whenever it
// arrives, which may be several turns of conversation later.
type manager struct {
	subagent *llmrouter.Session
	// limit caps how much work may be in flight at once, because a model that asks for
	// help on every sentence would otherwise open a session's worth of completions.
	limit int
	// box is where the subagent runs code, when it has anywhere to run it. Nil means the
	// tool is not offered, and the subagent works everything out in its head.
	box    sandbox.Sandbox
	logger *slog.Logger

	results *emit.Emitter[Result]

	mu sync.Mutex
	// running holds every task that has not settled, including ones already abandoned:
	// a cancelled task still produces a completion, and that completion still needs to
	// know which task it belonged to.
	running map[string]*task
	// bySkill is the live task for each skill, which is what makes a newer request for
	// the same skill supersede the older one.
	bySkill map[string]string
	closed  bool

	sequence  atomic.Int64
	forwarder sync.WaitGroup
	closeOnce sync.Once
}

// task is one piece of delegated work in flight.
type task struct {
	id        string
	skill     string
	turnID    string
	private   bool
	prompt    string
	startedAt time.Time
	// deadline abandons the task once the answer has stopped being worth having.
	deadline *time.Timer
	// reason is why the task was abandoned, set before the interrupt so the completion
	// that follows is reported as cancelled rather than answered.
	reason string
	// failure is what the provider reported for this task, which turns its completion
	// into a failure rather than an answer.
	failure error
	// instructions and messages are what the task was asked, kept so that running code
	// can be followed by asking again with what the code said.
	instructions string
	messages     []llm.Message
	// rounds is how many times this task has run code.
	rounds int
}

// live reports whether the task is still expected to produce an answer.
func (t *task) live() bool { return t.reason == "" }

func newManager(
	subagent *llmrouter.Session,
	limit int,
	box sandbox.Sandbox,
	logger *slog.Logger,
) *manager {
	m := &manager{
		subagent: subagent,
		limit:    limit,
		box:      box,
		logger:   logger,
		results:  emit.New[Result](resultBuffer),
		running:  map[string]*task{},
		bySkill:  map[string]string{},
	}
	m.forwarder.Add(1)
	go m.forward()
	return m
}

// Create delegates a piece of work and returns its task id.
//
// A newer request for a skill supersedes the one it already had: the caller has said
// something since, so the older question was asked about a conversation that no longer
// exists.
func (m *manager) Create(
	skill Skill,
	prompt string,
	history []llm.Message,
	turnID string,
	private bool,
) (string, error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return "", fmt.Errorf("harness: the conversation has ended")
	}
	var superseded string
	if existing, ok := m.bySkill[skill.Name]; ok && m.cancelLocked(existing, ReasonSuperseded) {
		superseded = existing
	}
	if m.liveLocked() >= m.limit {
		m.mu.Unlock()
		m.abandon(superseded)
		return "", fmt.Errorf("harness: %d tasks are already running", m.limit)
	}

	messages := append(append([]llm.Message(nil), history...), llm.Message{Role: llm.User, Content: prompt})
	created := &task{
		id:           fmt.Sprintf("task-%d-%d", time.Now().UnixNano(), m.sequence.Add(1)),
		skill:        skill.Name,
		turnID:       turnID,
		private:      private,
		prompt:       prompt,
		startedAt:    time.Now(),
		instructions: skill.Instructions,
		messages:     messages,
	}
	created.deadline = time.AfterFunc(skill.Deadline, func() {
		m.Cancel(created.id, ReasonDeadline)
	})
	m.running[created.id] = created
	m.bySkill[skill.Name] = created.id
	m.mu.Unlock()

	m.abandon(superseded)

	if err := m.subagent.Respond(llm.Request{
		ID:           created.id,
		Instructions: skill.Instructions,
		Messages:     messages,
		Tools:        m.tools(),
	}); err != nil {
		m.forget(created.id)
		return "", fmt.Errorf("harness: delegate %s: %w", skill.Name, err)
	}
	return created.id, nil
}

// CancelTurn abandons work whose conversational premise was superseded.
func (m *manager) CancelTurn(turnID, reason string) {
	m.mu.Lock()
	var abandoned []string
	for id, running := range m.running {
		if running.live() && running.turnID == turnID {
			m.cancelLocked(id, reason)
			abandoned = append(abandoned, id)
		}
	}
	m.mu.Unlock()

	for _, id := range abandoned {
		m.abandon(id)
	}
}

// Cancel abandons a task. The completion it was running still settles, and is reported
// as cancelled rather than answered.
func (m *manager) Cancel(taskID, reason string) {
	m.mu.Lock()
	abandoned := m.cancelLocked(taskID, reason)
	m.mu.Unlock()

	if abandoned {
		m.abandon(taskID)
	}
}

// abandon stops the provider generating for a task already marked cancelled. It is called
// without the lock, because a provider is entitled to take its time.
func (m *manager) abandon(taskID string) {
	if taskID == "" {
		return
	}
	if err := m.subagent.Interrupt(taskID); err != nil {
		m.logger.Error("could not abandon a task", "task", taskID, "error", err)
	}
}

// CancelSkill abandons whatever a skill is working on.
func (m *manager) CancelSkill(skill, reason string) {
	m.mu.Lock()
	taskID, ok := m.bySkill[skill]
	m.mu.Unlock()

	if ok {
		m.Cancel(taskID, reason)
	}
}

// CancelAll abandons every task in flight.
func (m *manager) CancelAll(reason string) {
	m.mu.Lock()
	abandoned := make([]string, 0, len(m.running))
	for id, running := range m.running {
		if running.live() {
			abandoned = append(abandoned, id)
		}
	}
	for _, id := range abandoned {
		m.cancelLocked(id, reason)
	}
	m.mu.Unlock()

	for _, id := range abandoned {
		m.abandon(id)
	}
}

// Running is how many tasks are still expected to answer.
func (m *manager) Running() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.liveLocked()
}

// RunningPublic is how much work the caller is waiting to hear about.
func (m *manager) RunningPublic() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	var live int
	for _, running := range m.running {
		if running.live() && !running.private {
			live++
		}
	}
	return live
}

// Results carries finished tasks. It is closed by Close.
func (m *manager) Results() <-chan Result { return m.results.Events() }

// Close abandons everything in flight and releases the subagent.
func (m *manager) Close() error {
	var err error
	m.closeOnce.Do(func() {
		m.CancelAll(ReasonClosed)

		m.mu.Lock()
		m.closed = true
		m.mu.Unlock()

		// Closing the session is what ends the forwarder: its events channel closes once
		// the provider has settled everything it was running.
		err = m.subagent.Close()
		m.forwarder.Wait()
	})
	return err
}

// cancelLocked marks a task abandoned, reporting whether it was still live. The task
// stays in running because the completion it was serving has yet to settle.
func (m *manager) cancelLocked(taskID, reason string) bool {
	running, ok := m.running[taskID]
	if !ok || !running.live() {
		return false
	}
	running.reason = reason
	running.deadline.Stop()
	// The skill is free again the moment its task is abandoned, so the next request for
	// it is not mistaken for a supersession of one nobody is waiting on.
	if m.bySkill[running.skill] == taskID {
		delete(m.bySkill, running.skill)
	}
	return true
}

// liveLocked counts the tasks still expected to answer.
func (m *manager) liveLocked() int {
	var live int
	for _, running := range m.running {
		if running.live() {
			live++
		}
	}
	return live
}

// forget drops a task that never reached the provider, so nothing waits on a completion
// that will not arrive.
func (m *manager) forget(taskID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if running, ok := m.running[taskID]; ok {
		running.deadline.Stop()
		if m.bySkill[running.skill] == taskID {
			delete(m.bySkill, running.skill)
		}
		delete(m.running, taskID)
	}
}

// forward turns the subagent's completions into results.
func (m *manager) forward() {
	defer m.forwarder.Done()
	defer m.results.Close()

	for event := range m.subagent.Events() {
		switch typed := event.(type) {
		case llm.CompletionComplete:
			m.settle(typed)
		case llm.Error:
			m.fail(typed)
		}
	}
}

// tools are what the subagent may do rather than say. Only running code is offered, and
// only when there is somewhere to run it.
func (m *manager) tools() []llm.Tool {
	if m.box == nil {
		return nil
	}
	return []llm.Tool{sandbox.Tool()}
}

// settle reports what became of a task.
func (m *manager) settle(complete llm.CompletionComplete) {
	m.mu.Lock()
	finished, ok := m.running[complete.CompletionID]
	if !ok {
		m.mu.Unlock()
		return
	}
	if m.box != nil && len(complete.ToolCalls) > 0 && finished.live() &&
		finished.rounds < toolRounds {
		finished.rounds++
		m.mu.Unlock()
		go m.resume(finished, complete)
		return
	}
	m.mu.Unlock()

	var result Result
	switch {
	case finished.reason != "":
		result.State = Cancelled
		result.Reason = finished.reason
	case complete.Interrupted:
		// Nothing named this task, so the whole session was stopped.
		result.State = Cancelled
		result.Reason = ReasonClosed
	case finished.failure != nil:
		result.State = Failed
		result.Err = finished.failure
	default:
		result.State = Done
		result.Text, result.Question = answer(complete.Text)
	}

	m.report(finished, result)
}

// report drops a task and says what became of it.
func (m *manager) report(finished *task, result Result) {
	m.mu.Lock()
	finished.deadline.Stop()
	delete(m.running, finished.id)
	if m.bySkill[finished.skill] == finished.id {
		delete(m.bySkill, finished.skill)
	}
	m.mu.Unlock()

	result.TaskID = finished.id
	result.Skill = finished.skill
	result.ElapsedMs = float64(time.Since(finished.startedAt).Microseconds()) / 1000
	m.results.Send(result)
}

// resume runs what a task asked for and puts the same question again with the answer.
//
// The deadline is not restarted: running code is part of the work the task was given,
// not licence to take longer over it.
func (m *manager) resume(running *task, complete llm.CompletionComplete) {
	ctx, cancel := context.WithTimeout(context.Background(), codeDeadline)
	defer cancel()

	messages := append(running.messages, llm.Message{
		Role:      llm.Assistant,
		Content:   complete.Text,
		ToolCalls: complete.ToolCalls,
	})
	for _, call := range complete.ToolCalls {
		messages = append(messages, llm.Message{
			Role:       llm.ToolResult,
			ToolCallID: call.ID,
			Content:    m.ran(ctx, call),
		})
	}

	m.mu.Lock()
	running.messages = messages
	instructions := running.instructions
	abandoned := running.reason
	closed := m.closed
	m.mu.Unlock()

	if abandoned != "" || closed {
		// Nothing is waiting for this any more, but something has to settle the task:
		// the completion it was running has already been and gone.
		reason := abandoned
		if reason == "" {
			reason = ReasonClosed
		}
		m.report(running, Result{State: Cancelled, Reason: reason})
		return
	}

	if err := m.subagent.Respond(llm.Request{
		ID:           running.id,
		Instructions: instructions,
		Messages:     messages,
		Tools:        m.tools(),
	}); err != nil {
		m.report(running, Result{State: Failed, Err: fmt.Errorf("harness: resume %s: %w", running.skill, err)})
	}
}

// ran executes one tool call and renders what happened in words the model can read.
//
// A failure is described rather than returned: the model asked for this mid-thought, and
// it can only do something sensible about code that did not run if it is told so.
func (m *manager) ran(ctx context.Context, call llm.ToolCall) string {
	if call.Name != sandbox.ToolName {
		return "There is no tool called " + call.Name + "."
	}

	var arguments struct {
		Code string `json:"code"`
	}
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		return "Those arguments are not valid JSON: " + err.Error()
	}

	result, err := m.box.Run(ctx, arguments.Code)
	if err != nil {
		m.logger.Error("could not run code", "error", err)
		return "The code could not be run: " + err.Error()
	}
	if result.ExitCode != 0 {
		return fmt.Sprintf("The code exited with %d and printed:\n%s", result.ExitCode, result.Output)
	}
	if strings.TrimSpace(result.Output) == "" {
		return "The code ran and printed nothing."
	}
	return result.Output
}

// fail records a provider failure against the task it names, so the completion that
// follows is reported as a failure rather than as an empty answer.
func (m *manager) fail(failure llm.Error) {
	if failure.CompletionID == "" {
		m.logger.Error("the subagent failed", "error", failure.Err, "context", failure.Context)
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if running, ok := m.running[failure.CompletionID]; ok && running.failure == nil {
		running.failure = failure.Err
	}
}
