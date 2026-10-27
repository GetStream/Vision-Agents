package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// resultBuffer is how many finished tasks may queue before the manager waits on the
// harness to read them.
const resultBuffer = 16

// toolRounds caps how many times one task may run code before it has to answer. The
// deadline bounds it too, but a model that has written the same broken program four times
// is not about to write a fifth that works.
const toolRounds = 4

// manager runs delegated work on the subagent and reports what came of it.
//
// A task is one response on the subagent's session, drained by a goroutine of its own that
// lives as long as the task does, code it runs included. Abandoning a task is closing its
// stream. Nothing here waits: Create returns as soon as the request is on its way, and the
// answer arrives on Results whenever it arrives, which may be several turns of conversation
// later.
type manager struct {
	subagent *opening
	// retired are subagents a swap replaced, kept open for the tasks still running on them.
	retired []*opening
	capture func(context.Context, CaptureRequest) ([]llm.ContentPart, error)
	// load reads a skill that was offered by name alone.
	load    func(context.Context, string) (Loaded, error)
	ctx     context.Context
	cancel  context.CancelFunc
	warming sync.WaitGroup
	// limit caps how much work may be in flight at once, because a model that asks for
	// help on every sentence would otherwise open a session's worth of completions.
	limit int
	// box is where the subagent runs code, when it has anywhere to run it. Nil means the
	// tool is not offered, and the subagent works everything out in its head.
	box sandbox.Sandbox
	// publish puts the files code hands back where the caller can see them. Nil means
	// there is nowhere to, and the subagent is told so.
	publish sandbox.Publisher
	// lookup and search answer the subagent out of the knowledge base and the web. Nil
	// means the tool is not offered.
	lookup func(ctx context.Context, turnID, query string) (string, error)
	search func(ctx context.Context, turnID, query string) (string, error)
	// overwrites is what the caller asked to change about how the model answers, written
	// over delegated work as well as over the conversation itself: somebody who asked for
	// more thinking meant the thinking, which is what runs here.
	overwrites  options.LLM
	onModelCall func(llm.CallTiming)
	logger      *slog.Logger

	results *emit.Emitter[Result]

	mu sync.Mutex
	// running holds every task that has not settled, including ones already abandoned:
	// a cancelled task still produces a completion, and that completion still needs to
	// know which task it belonged to.
	running map[string]*task
	// bySkill is the live task for each skill, which is what makes a newer request for
	// the same skill supersede the older one.
	bySkill map[string]string
	// written are the skill files already put into the sandbox, by skill, so a skill's
	// scripts are written once per version rather than before every run. They were written
	// into the sandbox's generation; a sandbox made again holds none of them.
	written    map[string]map[string]string
	generation int
	closed     bool

	sequence  atomic.Int64
	drainers  sync.WaitGroup
	closeOnce sync.Once
}

// task is one piece of delegated work in flight.
type task struct {
	ctx       context.Context
	cancel    context.CancelFunc
	capture   bool
	selection CaptureRequest
	evidence  []string
	// attached is a task given images the caller sent with the turn, rather than frames it
	// captures for itself.
	attached  bool
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
	// bundle is the rest of the skill's folder, by path, which the subagent reads one file
	// at a time and whose scripts it runs in the sandbox.
	bundle map[string]string
	// rounds is how many times this task has run code.
	rounds int
	// broken is what went wrong with the last code it ran, empty once code runs cleanly.
	broken string
	// files are what its code handed back and was published, the latest of each name.
	files []sandbox.Attachment
	// stream is what the subagent is answering on, and closing it is what abandons the
	// task. It is replaced each time the task runs code and asks again.
	stream *llm.Stream
}

// live reports whether the task is still expected to produce an answer.
func (t *task) live() bool { return t.reason == "" }

func newManager(
	subagent *llmrouter.Session,
	limit int,
	box sandbox.Sandbox,
	overwrites options.LLM,
	logger *slog.Logger,
) *manager {
	ctx, cancel := context.WithCancel(context.Background())
	m := &manager{
		ctx: ctx, cancel: cancel,
		limit:      limit,
		box:        box,
		overwrites: overwrites,
		logger:     logger,
		results:    emit.New[Result](resultBuffer),
		running:    map[string]*task{},
		bySkill:    map[string]string{},
		written:    map[string]map[string]string{},
	}
	if subagent != nil {
		ready := make(chan struct{})
		close(ready)
		m.subagent = &opening{ready: ready, session: subagent}
	}
	return m
}

// Create delegates a piece of work and returns its task id. Work a scheduled turn asked
// for, ctx marked so, is counted as that turn is (chargedAs); the work outlives it.
//
// A newer request for a skill supersedes the one it already had: the caller has said
// something since, so the older question was asked about a conversation that no longer
// exists.
func (m *manager) Create(
	ctx context.Context,
	skill Skill,
	prompt string,
	history []llm.Message,
	turnID string,
	private bool,
) (string, error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return "", stack.Wrap(fmt.Errorf("harness: the conversation has ended"))
	}
	var superseded string
	if existing, ok := m.bySkill[skill.Name]; ok && m.cancelLocked(existing, ReasonSuperseded) {
		superseded = existing
	}
	if m.liveLocked() >= m.limit {
		m.mu.Unlock()
		m.abandon(superseded)
		return "", stack.Wrap(fmt.Errorf("harness: %d tasks are already running", m.limit))
	}

	messages := append(append([]llm.Message(nil), history...), llm.Message{Role: llm.User, Content: prompt})
	ctx, cancel := context.WithCancel(chargedAs(m.ctx, ctx))
	created := &task{
		ctx: ctx, cancel: cancel, capture: skill.CaptureVideo,
		id:           fmt.Sprintf("task-%d-%d", time.Now().UnixNano(), m.sequence.Add(1)),
		skill:        skill.Name,
		turnID:       turnID,
		private:      private,
		prompt:       prompt,
		startedAt:    time.Now(),
		instructions: skill.Instructions,
		messages:     messages,
	}
	for _, message := range messages {
		if message.HasImage() {
			created.attached = true
			for _, part := range message.Parts {
				if part.Text != "" {
					created.evidence = append(created.evidence, part.Text)
				}
			}
		}
	}
	created.selection = CaptureRequest{TaskID: created.id, At: created.startedAt, Source: skill.VideoSource, Frames: skill.VideoFrames}
	created.deadline = time.AfterFunc(skill.Deadline, func() {
		m.Cancel(created.id, ReasonDeadline)
	})
	m.running[created.id] = created
	m.bySkill[skill.Name] = created.id
	m.drainers.Add(1)
	m.mu.Unlock()

	m.abandon(superseded)

	go m.start(created)
	return created.id, nil
}

// chargedAs returns ctx marked as scheduled when a scheduled task fired the turn asked
// (llmrouter Session.charged). Nothing else of asked is kept, its cancellation least of
// all: delegated work outlives the turn that asked for it.
func chargedAs(ctx, asked context.Context) context.Context {
	if routing.ScheduledTurn(asked) {
		ctx = routing.WithScheduledTurn(ctx)
	}
	return ctx
}

// Attached reports whether a skill is already looking at images the caller sent with this
// turn. Delegating the skill again would supersede the one task that has them.
func (m *manager) Attached(skill, turnID string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	running, ok := m.running[m.bySkill[skill]]
	return ok && running.attached && running.turnID == turnID
}

func (m *manager) start(created *task) {
	if created.instructions == "" && m.load != nil {
		loaded, err := m.load(created.ctx, created.skill)
		if err != nil {
			m.report(created, Result{State: Failed, Err: fmt.Errorf("harness: loading skill %s: %w", created.skill, err)})
			m.drainers.Done()
			return
		}
		created.instructions = loaded.Instructions
		if len(loaded.Files) > 0 {
			created.bundle = loaded.Files
			created.instructions += "\n\n" + Listing(created.skill, slices.Sorted(maps.Keys(loaded.Files)), m.box != nil)
		}
	}
	if m.lookup != nil && m.search != nil {
		created.instructions += "\n\nAlways ask the knowledge base (lookup) before the web (search)."
	}
	if created.capture && !llm.HasImage(created.messages) {
		if m.capture == nil {
			m.report(created, Result{State: Failed, Err: fmt.Errorf("harness: no video source")})
			m.drainers.Done()
			return
		}
		parts, err := m.capture(created.ctx, created.selection)
		if err != nil {
			m.report(created, Result{State: Failed, Err: err})
			m.drainers.Done()
			return
		}
		if !llm.HasImage([]llm.Message{{Parts: parts}}) {
			m.report(created, Result{State: Done, Question: llm.TextOf(parts)})
			m.drainers.Done()
			return
		}
		for _, part := range parts {
			if part.Text != "" {
				created.evidence = append(created.evidence, part.Text)
			}
		}
		created.messages = append(created.messages, llm.Message{Role: llm.User, Parts: parts})
	}
	stream, err := m.ask(created, created.messages)
	if err != nil {
		m.report(created, Result{State: Failed, Err: err})
		m.drainers.Done()
		return
	}
	m.hold(created, stream)
	m.drain(created, stream)
}

func (m *manager) ask(running *task, messages []llm.Message) (*llm.Stream, error) {
	model, err := m.model(running.ctx)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	overwrites := m.overwrites
	m.mu.Unlock()
	return model.Create(running.ctx, llm.ResponseParams{
		ID: running.id, Purpose: "subagent", TurnID: running.turnID,
		OnTiming:     m.onModelCall,
		Instructions: running.instructions, Input: messages, Tools: m.tools(running),
	}.Overwrite(overwrites))
}

func (m *manager) setOverwrites(overwrites options.LLM) {
	m.mu.Lock()
	m.overwrites = overwrites
	m.mu.Unlock()
}

// hold records the stream a task is answering on, closing it straight away when the task
// was abandoned while the request was still going out.
func (m *manager) hold(running *task, stream *llm.Stream) {
	m.mu.Lock()
	running.stream = stream
	abandoned := !running.live() || m.closed
	m.mu.Unlock()

	if abandoned {
		stream.Close()
	}
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

	m.mu.Lock()
	running, ok := m.running[taskID]
	var stream *llm.Stream
	if ok {
		stream = running.stream
	}
	m.mu.Unlock()

	// A task whose request has not gone out yet is closed by hold instead, once it has.
	if stream != nil {
		stream.Close()
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

// RunningSkills names the skills of the public tasks still expected to answer.
func (m *manager) RunningSkills() []string {
	m.mu.Lock()
	defer m.mu.Unlock()

	var skills []string
	for _, running := range m.running {
		if running.live() && !running.private && !slices.Contains(skills, running.skill) {
			skills = append(skills, running.skill)
		}
	}
	slices.Sort(skills)
	return skills
}

// Results carries finished tasks. It is closed by Close.
func (m *manager) Results() <-chan Result { return m.results.Events() }

// Close abandons everything in flight and releases the subagent.
func (m *manager) Close() error {
	var err error
	m.closeOnce.Do(func() {
		m.mu.Lock()
		m.closed = true
		m.mu.Unlock()
		m.CancelAll(ReasonClosed)
		m.cancel()
		m.warming.Wait()
		var failures []error
		for _, opened := range append(m.retired, m.subagent) {
			if opened == nil || opened.session == nil {
				continue
			}
			if failure := opened.session.Close(); failure != nil {
				failures = append(failures, failure)
			}
		}
		err = errors.Join(failures...)
		m.drainers.Wait()
		m.results.Close()
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
	running.cancel()
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

// drain follows one task for the whole of its life, code it runs included, and reports what
// became of it.
func (m *manager) drain(running *task, stream *llm.Stream) {
	defer m.drainers.Done()

	for {
		response := m.consume(running, stream)

		next, more := m.advance(running, response)
		if !more {
			return
		}
		stream = next
	}
}

// consume drains one response, remembering a failure against the task it belonged to so
// that what settles is reported as a failure rather than as an empty answer.
func (m *manager) consume(running *task, stream *llm.Stream) llm.Response {
	defer stream.Close()

	for stream.Next() {
		failed, ok := stream.Current().(llm.ResponseFailed)
		if !ok {
			continue
		}
		m.mu.Lock()
		if running.failure == nil {
			running.failure = failed.Err
		}
		m.mu.Unlock()
	}
	return stream.Response()
}

// tools are what the subagent may do rather than say: run code when there is somewhere to
// run it, read the skill's own files when it brought some, and look things up in the
// knowledge base and on the web when the agent can.
func (m *manager) tools(running *task) []llm.Tool {
	var tools []llm.Tool
	if m.box != nil {
		tools = append(tools, sandbox.Tool(m.box))
	}
	if len(running.bundle) > 0 {
		tools = append(tools, readFileTool(slices.Sorted(maps.Keys(running.bundle))))
	}
	if m.lookup != nil {
		tools = append(tools, queryTool(subagentLookupTool, "Look something up in the knowledge base, which is "+
			"what this business has written down: prices, policies, how a product works. What it says is the "+
			"answer the business wants given, even where the web says otherwise.", "What to look for, in the "+
			"terms the question used."))
	}
	if m.search != nil {
		tools = append(tools, queryTool(subagentSearchTool, "Search the web for what is true right now: news, "+
			"releases, prices, anything that depends on today rather than on what you know. What you remember "+
			"about any of these is out of date.", "What to find out, written as you would type it into a "+
			"search box."))
	}
	return tools
}

// subagentLookupTool and subagentSearchTool are the names the subagent knows the knowledge
// base and the web by, the same the model holding the conversation knows them by.
const (
	subagentLookupTool = "lookup"
	subagentSearchTool = "search"
)

func queryTool(name, description, query string) llm.Tool {
	return llm.Tool{
		Name:        name,
		Description: description,
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{"query": map[string]any{"type": "string", "description": query}},
			"required":   []string{"query"},
		},
	}
}

// advance decides what a settled response means for its task: either the task runs the code
// it asked for and puts the same question again, in which case the stream to follow next is
// returned, or it is finished with and reported.
func (m *manager) advance(running *task, response llm.Response) (*llm.Stream, bool) {
	m.mu.Lock()
	if _, known := m.running[running.id]; !known {
		m.mu.Unlock()
		return nil, false
	}
	if len(m.tools(running)) > 0 && len(response.ToolCalls) > 0 && running.live() &&
		running.rounds < toolRounds {
		running.rounds++
		m.mu.Unlock()
		return m.resume(running, response)
	}
	reason, failure, broken := running.reason, running.failure, running.broken
	m.mu.Unlock()

	var result Result
	switch {
	case reason != "":
		result.State = Cancelled
		result.Reason = reason
	case response.Status == llm.StatusCancelled:
		// Nothing named this task, so the whole session was stopped.
		result.State = Cancelled
		result.Reason = ReasonClosed
	case failure != nil:
		result.State = Failed
		result.Err = failure
	default:
		result.State = Done
		result.Text, result.Question = answer(response.OutputText)
	}
	// An empty answer, or one still asking to run code it has run out of rounds for, is
	// work that never finished, and the caller was promised it would.
	if result.State == Done && result.Question == "" && (result.Text == "" || len(response.ToolCalls) > 0) {
		result.State = Failed
		result.Err = fmt.Errorf("harness: %s did not finish", running.skill)
		if broken != "" {
			result.Err = fmt.Errorf("harness: %s did not finish: %s", running.skill, broken)
		}
	} else if result.State == Done && broken != "" {
		result.Err = errors.New(broken)
	}

	m.report(running, result)
	return nil, false
}

// report drops a task and says what became of it.
func (m *manager) report(finished *task, result Result) {
	m.mu.Lock()
	finished.deadline.Stop()
	finished.cancel()
	if finished.reason != "" {
		result.State = Cancelled
		result.Reason = finished.reason
		result.Err = nil
		result.Text = ""
		result.Question = ""
	}
	delete(m.running, finished.id)
	if m.bySkill[finished.skill] == finished.id {
		delete(m.bySkill, finished.skill)
	}
	m.mu.Unlock()

	if result.State == Done {
		result.Files = finished.files
	}
	result.Evidence = finished.evidence
	result.TaskID = finished.id
	result.TurnID = finished.turnID
	result.Skill = finished.skill
	result.ElapsedMs = float64(time.Since(finished.startedAt).Microseconds()) / 1000
	m.results.Send(result)
}

// resume runs what a task asked for and puts the same question again with the answer,
// returning the stream the new answer arrives on.
//
// The deadline is not restarted: running code is part of the work the task was given,
// not licence to take longer over it. It is all the code is bounded by besides the
// sandbox's own limit, so a skill that renders for minutes says so in its deadline.
func (m *manager) resume(running *task, response llm.Response) (*llm.Stream, bool) {
	messages := append(running.messages, llm.Message{
		Role:      llm.Assistant,
		Content:   response.OutputText,
		ToolCalls: response.ToolCalls,
	})
	for _, call := range response.ToolCalls {
		messages = append(messages, llm.Message{
			Role:       llm.ToolResult,
			ToolCallID: call.ID,
			Content:    m.ran(running, call),
		})
	}

	m.mu.Lock()
	running.messages = messages
	abandoned := running.reason
	closed := m.closed
	m.mu.Unlock()

	if abandoned != "" || closed {
		// Nothing is waiting for this any more, but something has to settle the task:
		// the response it was running has already been and gone.
		reason := abandoned
		if reason == "" {
			reason = ReasonClosed
		}
		m.report(running, Result{State: Cancelled, Reason: reason})
		return nil, false
	}

	stream, err := m.ask(running, messages)
	if err != nil {
		m.report(running, Result{State: Failed,
			Err: fmt.Errorf("harness: resume %s: %w", running.skill, err)})
		return nil, false
	}
	m.hold(running, stream)
	return stream, true
}

// ran executes one tool call and renders what happened in words the model can read.
//
// A failure is described rather than returned: the model asked for this mid-thought, and
// it can only do something sensible about code that did not run if it is told so.
func (m *manager) ran(running *task, call llm.ToolCall) string {
	if call.Name == ReadSkillFileTool && len(running.bundle) > 0 {
		var arguments struct {
			Path string `json:"path"`
		}
		if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
			return "Those arguments are not valid JSON: " + err.Error()
		}
		content, err := bundled(running.bundle, arguments.Path)
		if err != nil {
			return err.Error()
		}
		return content
	}
	if ask := m.asker(call.Name); ask != nil {
		var arguments struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
			return "Those arguments are not valid JSON: " + err.Error()
		}
		said, err := ask(running.ctx, running.turnID, arguments.Query)
		if err != nil {
			m.logger.Warn("the subagent could not look something up", "tool", call.Name, "error", err)
			return "That did not work: " + err.Error()
		}
		return said
	}
	if call.Name != sandbox.ToolName || m.box == nil {
		return "There is no tool called " + call.Name + "."
	}

	var arguments struct {
		Code  string   `json:"code"`
		Files []string `json:"files"`
	}
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		return "Those arguments are not valid JSON: " + err.Error()
	}

	if err := m.writeBundle(running); err != nil {
		m.logger.Error("could not put the skill's files in the sandbox", "skill", running.skill, "error", err)
		m.setBroken(running, "the skill's files could not be put in the sandbox: "+err.Error())
		return "The skill's files could not be put in the sandbox: " + err.Error()
	}
	result, err := m.box.Run(running.ctx, arguments.Code, arguments.Files)
	if err != nil {
		m.logger.Error("could not run code", "error", err)
		m.setBroken(running, "the code could not be run: "+err.Error())
		return "The code could not be run: " + err.Error()
	}
	m.setBroken(running, "")
	if result.ExitCode != 0 {
		m.setBroken(running, fmt.Sprintf("the code exited with %d", result.ExitCode))
	}
	said, files := HandBack(running.ctx, m.publish, result, m.logger)
	m.mu.Lock()
	for _, attachment := range files {
		running.files = slices.DeleteFunc(running.files, func(kept sandbox.Attachment) bool { return kept.Name == attachment.Name })
		running.files = append(running.files, attachment)
	}
	m.mu.Unlock()
	return said
}

// asker is what answers a call for lookup or search, nil for any other tool or one the
// agent cannot answer.
func (m *manager) asker(name string) func(ctx context.Context, turnID, query string) (string, error) {
	switch name {
	case subagentLookupTool:
		return m.lookup
	case subagentSearchTool:
		return m.search
	}
	return nil
}

// writeBundle puts a skill's files into the sandbox under skills/<name>/, the first time one
// of its tasks runs code and again whenever the skill, or the sandbox, has changed since.
func (m *manager) writeBundle(running *task) error {
	if len(Scripts(slices.Collect(maps.Keys(running.bundle)))) == 0 {
		return nil
	}
	generation, err := m.box.Ready(running.ctx)
	if err != nil {
		return err
	}
	m.mu.Lock()
	if generation != m.generation {
		m.written, m.generation = map[string]map[string]string{}, generation
	}
	current := maps.Equal(m.written[running.skill], running.bundle)
	m.mu.Unlock()
	if current {
		return nil
	}
	for _, name := range slices.Sorted(maps.Keys(running.bundle)) {
		if err := m.box.Write(running.ctx, BundlePath(running.skill, name), []byte(running.bundle[name])); err != nil {
			return err
		}
	}
	m.mu.Lock()
	m.written[running.skill] = running.bundle
	m.mu.Unlock()
	return nil
}

func (m *manager) setBroken(running *task, broken string) {
	m.mu.Lock()
	running.broken = broken
	m.mu.Unlock()
}

// HandBack says what a run of code did in words the model can read, with how to connect what
// the run had no token for, and publishes the files it returned where the person can see
// them, returning the ones that were. A nil publish
// means the conversation has nowhere to show a file.
func HandBack(ctx context.Context, publish sandbox.Publisher, result sandbox.Result, logger *slog.Logger) (string, []sandbox.Attachment) {
	var said string
	switch {
	case result.ExitCode != 0:
		said = fmt.Sprintf("The code exited with %d and printed:\n%s", result.ExitCode, result.Output)
	case strings.TrimSpace(result.Output) == "":
		said = "The code ran and printed nothing."
	default:
		said = result.Output
	}
	var names, shown, failed []string
	var published []sandbox.Attachment
	for _, file := range result.Files {
		names = append(names, file.Name)
		if publish == nil {
			continue
		}
		attachment, err := publish(ctx, file)
		if err != nil {
			logger.Error("could not publish a file", "file", file.Name, "error", err)
			failed = append(failed, file.Name)
			continue
		}
		published = append(published, attachment)
		shown = append(shown, file.Name)
	}
	var handed []string
	if publish == nil && len(names) > 0 {
		handed = append(handed, "These files were written, but this conversation has nowhere to show them: "+strings.Join(names, ", "))
	}
	if len(shown) > 0 {
		handed = append(handed, "These files will be attached to the reply the person sees: "+strings.Join(shown, ", "))
	}
	if len(failed) > 0 {
		handed = append(handed, "These files could not be attached: "+strings.Join(failed, ", "))
	}
	if len(handed) > 0 {
		said += "\n\n" + strings.Join(handed, "\n")
	}
	if len(result.Missing) > 0 {
		said += "\n\nThese files were not written, or were too large to return: " + strings.Join(result.Missing, ", ")
	}
	for _, grant := range result.Unconnected {
		said += fmt.Sprintf("\n\n"+unconnectedLine+"%s), the user has not connected it: call %s__list_tools "+
			"so they can, then run the code again; if you cannot call it, tell the user they must connect %s.",
			grant.Binding, strings.Join(grant.Hosts, ", "), grant.Binding, grant.Binding)
	}
	for _, binding := range result.Connected {
		said += "\n\n" + binding + justConnected
	}
	return said, published
}

// unconnectedLine and justConnected begin and end the lines HandBack adds for a late grant,
// each of which tells the model to run the same code again once the user has connected
// it (RunAgain).
const (
	unconnectedLine = "If this code needed %s (a 401 from "
	justConnected   = " was just connected: if this run got 401, wait 20 seconds and run the code again."
)

// RunAgain reports whether a run of code's result (HandBack) tells the model to run the same
// code again now: it said a binding was just connected, or named one as not connected that is
// among connected, the bindings the user has connected since (sandbox.Connecting). Run again
// while the user has not connected it, the same code would only get the same 401, as often as
// the model asks.
func RunAgain(result string, connected []string) bool {
	return strings.Contains(result, justConnected) || slices.ContainsFunc(connected, func(binding string) bool {
		return strings.Contains(result, fmt.Sprintf(unconnectedLine, binding))
	})
}
