// Package session runs voice agents for callers who are not in this process.
//
// The agent package already knows how to hold a conversation; what it has never had is a
// way to be started by somebody else. A session is that: a spec arrives over HTTP, an agent
// joins the call, and what it hears, says and asks for is fanned out to whoever is watching
// over a WebSocket.
//
// The one thing that travels in both directions is tools. A caller's own functions live
// wherever the caller is, so the model asks for one here and the answer comes back from
// there, which is what lets a Python SDK register a function against a conversation whose
// audio it never touches.
package session

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// watcherBuffer is how many events may queue for one watcher before it starts losing them.
// A watcher is a control channel rather than the conversation, so a slow one costs its
// reader some detail and nobody on the call anything.
const watcherBuffer = 256

// Defaults a caller can leave out. They are the same ones cmd/agent's flags carry, so a
// session started with an almost empty spec behaves like the demo.
const (
	defaultCallType     = "agent"
	defaultUserID       = "vision-agent"
	defaultUserName     = "Vision Agent"
	defaultLLMTarget    = "llm-fast"
	defaultSTTTarget    = "en-low-latency"
	defaultTTSTarget    = "en-low-latency"
	defaultSearchTarget = "search-fast"
	// The model the skills run on. A quality tier rather than a fast one, since the
	// conversation carries on without it: what is handed over is what the talking model
	// could not answer itself.
	defaultSubagentTarget = "llm-thinking"
	// The model the flow controller runs on. A non-thinking fast model, since deciding who
	// holds the floor is a small classification the caller waits through on every turn.
	defaultControllerTarget = "llm-flow"
)

// daytonaProvider is the one sandbox a caller may ask for by name.
const daytonaProvider = "daytona"

// finishWithin bounds how long a session ending gives the agent to be heard out. Leaving
// the call discards whatever audio has not gone out yet, and a voice streams a reply faster
// than it is spoken, so closing the moment the provider stops sending cuts the last of the
// reply off mid-word. It is a bound rather than a wait: a caller ending a session is not
// made to sit through a monologue.
const finishWithin = 5 * time.Second

// State is where a session is in its life.
type State string

const (
	// Live means the agent is in the call.
	Live State = "live"
	// Ended means it has left, whether it was asked to or the call finished.
	Ended State = "ended"
)

// Event is what a watcher sees. Everything the agent emits passes through unchanged; the
// types declared in this package are the session's own.
type Event any

// CommandStopped is how one named command ended after somebody asked for it to stop. It
// is separate from the receipt a submission returns, because a watcher has to tell a
// command it asked to stop from a command that was just accepted.
type CommandStopped struct {
	persistent.CommandReceipt
}

// ToolCall is the model asking for one of the caller's own tools. It is the only event a
// watcher is obliged to answer: everything else is a report.
type ToolCall struct {
	Cancel    bool
	CommandID string
	TurnID    string
	// ID is what a result must quote to answer this call.
	ID string
	// Name is which tool was asked for.
	Name string
	// Arguments is the JSON object the model filled in, left as text because it is the
	// caller that knows what shape to expect.
	Arguments string
}

// Session is one conversation this process is running on somebody's behalf.
type Session struct {
	logs    *logRecorder
	id      string
	spec    Spec
	created time.Time
	logger  *slog.Logger

	persisted  *persistent.Conversation
	voiceAgent *agent.Agent
	tools      *bridge
	transcript Transcript
	// skills are what the fast model may hand over, resolved once when the session was
	// created. Kept so the call's row can say what was on offer: the spec carries names
	// or nothing at all, and nothing at all means the built-in set.
	skills harness.Skills

	// Serializes persistent command acceptance/start with command-targeted interruption.
	commandMu sync.Mutex
	mu        sync.Mutex
	// watchers are the connections being fanned out to, keyed so one can detach without
	// disturbing the others.
	watchers    map[uint64]*watcher
	nextWatcher uint64
	state       State

	// said is the conversation as it happens, kept so a finished call can be reviewed
	// without reading back what was written to chat. It has a lock of its own so
	// recording a line does not contend with the fan-out a watcher is detaching from.
	saidMu sync.Mutex
	said   []spoken

	// records keeps the turns and what each one did, so a conversation can be read back
	// without the socket that heard it. Nil for an incognito session and for a deployment
	// with no store, which is what makes the flag safe: there is nowhere to write rather
	// than a check to remember.
	records recorder
	// turnsMu guards the bookkeeping that turns a stream of events into rows.
	turnsMu sync.Mutex
	// turns maps a turn id to the response row it is being recorded as, so the events of
	// one turn -- which arrive interleaved with another's -- land on the right one.
	turns map[string]*recordedTurn

	// closers undo what Create wired up, in reverse.
	closers   []func()
	closeOnce sync.Once
	running   sync.WaitGroup
}

// recordedTurn is one turn being written down as it happens.
type recordedTurn struct {
	// id is the response row, which is not the turn id: a turn id is the agent's own and
	// repeats across sessions, while a response is a row of its own.
	id string
	// ordinal is the next item's position, assigned here rather than by the database so
	// items keep the order they happened in rather than the order they were inserted in.
	ordinal int
	// blocked marks a turn the guardrail refused, so the answer that follows is recorded as
	// the refusal it is rather than as what the model wanted to say.
	blocked bool
}

// ID is the handle a caller holds the session by.
func (s *Session) ID() string { return s.id }

// Spec is what the session was asked for.
func (s *Session) Spec() Spec { return s.spec }

// CreatedAt is when the session joined.
func (s *Session) CreatedAt() time.Time { return s.created }

// State reports whether the agent is still in the call.
func (s *Session) State() State {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state
}

// Provider names the model answering and the voice speaking, which are only known once the
// routers have picked them.
func (s *Session) Provider() (llm string, tts string) {
	_, llm, tts, _ = s.Resolved()
	return llm, tts
}

// Resolved names the providers routing picked for this call. Transcription is empty until
// somebody has been heard, because a listener is opened per speaker rather than up front.
func (s *Session) Resolved() (stt, llm, tts, subagent string) {
	if s.voiceAgent == nil {
		return
	}
	if ears := s.voiceAgent.STT(); ears != nil {
		stt = ears.Provider() + "/" + ears.Model()
	}
	if model := s.voiceAgent.LLM(); model != nil {
		llm = model.Provider() + "/" + model.Model()
	}
	if voice := s.voiceAgent.TTS(); voice != nil {
		tts = voice.Provider() + "/" + voice.Model()
	}
	if think := s.voiceAgent.Subagent(); think != nil {
		subagent = think.Provider() + "/" + think.Model()
	}
	return stt, llm, tts, subagent
}

// Speech names the speech-to-speech model routing picked for a native call, and nothing
// for a call the cascade held.
func (s *Session) Speech() string {
	if s.voiceAgent == nil {
		return ""
	}
	model := s.voiceAgent.STS()
	if model == nil {
		return ""
	}
	return model.Provider() + "/" + model.Model()
}

// watcher is one attached consumer.
//
// The channel is closed exactly once, whether the consumer detached or the session ended
// underneath it. Both happen, and often together: a caller hanging up closes its socket at
// the same moment the agent leaves the call.
type watcher struct {
	events chan Event
	once   sync.Once
}

func (w *watcher) close() { w.once.Do(func() { close(w.events) }) }

// send offers an event without waiting, reporting whether it was taken. A watcher is a
// control channel rather than the conversation, so one that has fallen behind costs its
// reader some detail and nobody on the call anything.
func (w *watcher) send(event Event) bool {
	select {
	case w.events <- event:
		return true
	default:
		return false
	}
}

// Watch attaches a consumer and returns it along with the way to detach.
//
// Every watcher sees everything from the moment it attached. Nothing is replayed: a caller
// that connects late has missed the conversation, and a control channel that opened with a
// backlog would have it answering tool calls that timed out before it arrived.
func (s *Session) Watch() (<-chan Event, func()) {
	return s.watch(false)
}

// WatchPendingVoiceTools opts a durable tool host into replay of pending live
// voice requests. Text command recovery uses the persistent conversation path.
func (s *Session) WatchPendingVoiceTools() (<-chan Event, func()) {
	return s.watch(true)
}

func (s *Session) watch(replayVoiceTools bool) (<-chan Event, func()) {
	attached := &watcher{events: make(chan Event, watcherBuffer)}

	s.mu.Lock()
	if s.state == Ended {
		s.mu.Unlock()
		attached.close()
		return attached.events, func() {}
	}
	id := s.nextWatcher
	s.nextWatcher++
	s.watchers[id] = attached
	if replayVoiceTools && !s.spec.Text && s.persisted == nil && s.tools != nil {
		for _, pending := range s.tools.Pending() {
			attached.send(pending)
		}
	}
	s.mu.Unlock()

	return attached.events, func() {
		s.mu.Lock()
		delete(s.watchers, id)
		detached := len(s.watchers) == 0 && s.persisted != nil
		s.mu.Unlock()
		attached.close()
		// Persistent text clients own no call; a disconnected operator leaves no tool host.
		// End this session so the saved channel can be reopened after a terminal crash.
		if detached {
			go func() {
				// Interrupt belongs to the same one-time teardown as Close. A late
				// detach from an ended session must not cancel a reopened conversation.
				s.closeOnce.Do(func() { s.Interrupt(); _ = s.close() })
			}()
		}
	}
}

// Say speaks a piece of text without going through the model.
func (s *Session) Say(ctx context.Context, text string) error {
	return s.voiceAgent.Say(ctx, text)
}

// Respond answers a piece of text through the model, as though a participant had said it.
// Images attach to that turn as content parts.
// It returns the id the turn was recorded as, which is the handle a caller follows one
// particular answer by: its items are asked for under it. Empty when nothing is being
// recorded -- an incognito session, or a deployment with no store -- and empty for a native
// session, where the model decides for itself what counts as a turn.
func (s *Session) Respond(ctx context.Context, text string, images []llm.ImagePart) (string, error) {
	s.commandMu.Lock()
	defer s.commandMu.Unlock()
	if s.persisted != nil {
		if s.spec.Caller.UserID != "" {
			return "", errors.New("personal conversations require a command ID")
		}
		if err := s.persisted.Begin(text); err != nil {
			return "", err
		}
	}
	turnID, err := s.voiceAgent.RespondTo(ctx, text, images)
	if err != nil {
		if s.persisted != nil {
			s.persisted.Cancel()
		}
		return "", err
	}
	if turnID == "" {
		return "", nil
	}
	// Opening the turn here as well as on the event is what makes the id available to whoever
	// asked for the answer. It happens once whichever way round the two arrive.
	return s.openTurn(turnID, text), nil
}

// RespondCommand accepts one durable text submission. The receipt may be replayed,
// but only the first successful acceptance is allowed to invoke the model.
func (s *Session) RespondCommand(ctx context.Context, id, text string) (persistent.CommandReceipt, error) {
	s.commandMu.Lock()
	defer s.commandMu.Unlock()
	if s.persisted == nil {
		return persistent.CommandReceipt{}, errors.New("command IDs require a persistent text conversation")
	}
	if err := s.persisted.CheckCaller(ctx, s.spec.Caller.UserID); err != nil {
		return persistent.CommandReceipt{}, err
	}
	receipt, err := s.persisted.BeginCommand(id, text)
	if err != nil {
		return receipt, err
	}
	s.broadcast(receipt)
	if receipt.Duplicate {
		return receipt, nil
	}
	turnID, err := s.voiceAgent.RespondTo(ctx, text, nil)
	if err != nil {
		s.persisted.Cancel()
		return receipt, err
	}
	s.persisted.BindTurn(receipt.CommandID, turnID)
	return receipt, nil
}

// Report publishes a failure the watcher should see, without ending the session.
func (s *Session) Report(err error, context string) {
	if err == nil {
		return
	}
	s.broadcast(agent.Error{Err: err, Context: context})
}

// Ask answers a piece of text in writing, without speaking any of it, and writes the answer
// into the conversation.
//
// The answer is stored here rather than by the caller because the transcript is the
// session's: a caller holding a session has no channel to write to, and an answer that only
// went back over HTTP would be missing from the conversation it belongs to.
func (s *Session) Ask(ctx context.Context, text string) (string, error) {
	answer, err := s.voiceAgent.Ask(ctx, text)
	if err != nil {
		return "", err
	}
	if s.transcript != nil {
		s.transcript.Reply(answer)
	}
	return answer, nil
}

// Interrupt abandons the reply being spoken.
func (s *Session) Interrupt() {
	s.commandMu.Lock()
	defer s.commandMu.Unlock()
	s.voiceAgent.Interrupt()
	if s.persisted != nil {
		s.persisted.Cancel()
	}
}

// Command reads what is known about a durable command without accepting, running or
// stopping anything. The caller must already be authorized for this session.
func (s *Session) Command(id string) (persistent.CommandReceipt, error) {
	if s.persisted == nil {
		return persistent.CommandReceipt{}, persistent.ErrCommandNotFound
	}
	if err := s.persisted.CheckCaller(context.Background(), s.spec.Caller.UserID); err != nil {
		return persistent.CommandReceipt{}, err
	}
	return s.persisted.Command(id)
}

// InterruptCommand targets a durable text command, never whichever command starts
// later. Holding commandMu across lookup/interruption prevents a new submission
// from entering the acceptance-to-execution gap.
func (s *Session) InterruptCommand(id string) (persistent.CommandReceipt, error) {
	s.commandMu.Lock()
	defer s.commandMu.Unlock()
	if s.persisted == nil {
		return persistent.CommandReceipt{}, persistent.ErrCommandNotFound
	}
	if err := s.persisted.CheckCaller(context.Background(), s.spec.Caller.UserID); err != nil {
		return persistent.CommandReceipt{}, err
	}
	receipt, err := s.persisted.Command(id)
	if err != nil {
		return persistent.CommandReceipt{}, err
	}
	switch receipt.State {
	case "completed", "cancelled", "interrupted", "failed":
		return s.stopped(s.persisted.CancelCommand(id))
	}
	s.voiceAgent.Interrupt()
	return s.stopped(s.persisted.CancelCommand(id))
}

// stopped tells the watchers how the named command ended. An unknown outcome is not
// published: the caller holding the stop is the one that has to retry it.
func (s *Session) stopped(receipt persistent.CommandReceipt, err error) (persistent.CommandReceipt, error) {
	if err == nil {
		s.broadcast(CommandStopped{CommandReceipt: receipt})
	}
	return receipt, err
}

// Busy reports whether the agent still has something to finish, which is how anything
// driving a conversation knows a turn is over rather than merely answered once.
func (s *Session) Busy() bool { return s.voiceAgent.Busy() }

// SetInstructions changes what the agent is told to be from the next turn on.
func (s *Session) SetInstructions(text string) {
	s.spec.Instructions = text
	s.voiceAgent.SetInstructions(s.spec.prompt())
}

// ResolveTool hands a tool result back to the model waiting for it, reporting whether
// anything was.
func (s *Session) ResolveTool(id, output, failure string) bool {
	return s.ResolveToolParts(id, llm.TextParts(output), failure)
}

// ResolveToolParts is ResolveTool for a result that may carry images.
func (s *Session) ResolveToolParts(id string, parts []llm.ContentPart, failure string) bool {
	if s.persisted != nil && s.spec.Caller.UserID != "" {
		return false
	}
	return s.tools.Resolve(id, "", parts, failure)
}

func (s *Session) ResolveCommandTool(id, commandID, turnID string, parts []llm.ContentPart, failure string) bool {
	if s.persisted == nil || commandID == "" || turnID == "" {
		return false
	}
	expected, bound := s.persisted.CommandForTurn(turnID)
	if !bound || expected != commandID {
		return false
	}
	return s.tools.Resolve(id, turnID, parts, failure)
}

// Close leaves the call and releases everything the session opened. It is safe to call
// more than once.
func (s *Session) Close() error {
	var err error
	s.closeOnce.Do(func() { err = s.close() })
	return err
}

func (s *Session) close() error {
	// The tools go first so a model waiting on one is told the call ended rather than
	// waiting out a timeout against a conversation that is already over.
	s.tools.Close()

	// Hanging up mid-sentence is rude, and leaving is what throws away the audio that has
	// not been heard yet, so the last utterance is given a moment to land first.
	finishing, cancel := context.WithTimeout(context.Background(), finishWithin)
	if err := s.voiceAgent.Finish(finishing); err != nil {
		s.logger.Debug("the agent was still talking when the session ended", "error", err)
	}
	cancel()

	err := s.voiceAgent.Close()

	// The fan-out ends on its own once the agent's events channel closes, and waiting
	// for it here is what makes the last events, including what the harness abandoned on
	// the way out, reach the watchers before they are disconnected.
	s.running.Wait()

	// A turn still open when the session ends was abandoned rather than answered. It is
	// closed here, after the fan-out has drained, so a conversation read back later has no
	// turn that appears to still be thinking.
	s.abandonTurns()

	for i := len(s.closers) - 1; i >= 0; i-- {
		s.closers[i]()
	}

	s.mu.Lock()
	s.state = Ended
	watchers := s.watchers
	s.watchers = map[uint64]*watcher{}
	s.mu.Unlock()

	for _, attached := range watchers {
		attached.close()
	}
	return err
}

// consume fans the agent's events out to the watchers and records the transcript.
func (s *Session) consume() {
	defer s.running.Done()

	for event := range s.voiceAgent.Events() {
		if s.persisted != nil {
			s.persisted.Observe(event)
		}
		if s.transcript != nil {
			s.transcript.Record(event)
		}
		s.remember(event)
		s.record(event)
		s.broadcast(event)
	}
}

// record writes down what the turn did, which is what items.unwind() reads back.
//
// Deltas are deliberately not recorded. A hundred fragments of one sentence are the
// sentence, and keeping them would make the table mostly punctuation; a caller watching a
// turn happen reads the deltas off the socket, and a caller reading one back wants the
// shape of it -- the question, the tools, the answer.
//
// Nothing here checks whether the session is incognito, because an incognito session has no
// recorder to check: the manager never hands it one.
func (s *Session) record(event Event) {
	if s.records == nil {
		return
	}

	switch typed := event.(type) {
	case agent.Responding:
		s.openTurn(typed.TurnID, typed.Prompt)
	case agent.ToolStarted:
		s.item(typed.TurnID, store.ItemToolCall, "", typed.Tool, map[string]any{
			"call_id": typed.ID, "product": typed.Product, "sdk": typed.SDK,
		})
	case agent.ToolRan:
		payload := map[string]any{"call_id": typed.ID, "arguments": typed.Arguments}
		if typed.Err != nil {
			payload["error"] = typed.Err.Error()
		}
		s.item(typed.TurnID, store.ItemToolResult, typed.Result, typed.Tool, payload)
	case agent.LookedUp:
		s.item(typed.TurnID, store.ItemToolCall, typed.Query, "search", nil)
	case agent.Delegated:
		s.item(typed.TurnID, store.ItemThought, typed.Prompt, typed.Skill, map[string]any{
			"task_id": typed.TaskID,
		})
	case agent.Blocked:
		s.blockTurn(typed.TurnID)
		s.item(typed.TurnID, store.ItemBlocked, "", "", map[string]any{
			"reason": typed.Reason, "probability": typed.Probability,
		})
	case agent.Responded:
		// A reply followed by tools or delegated work is not the end of the turn, so the
		// row stays open: the agent will speak again once the work comes back, and both
		// halves belong to the same response.
		kind := store.ItemAnswer
		if s.turnBlocked(typed.TurnID) {
			kind = store.ItemBlocked
		}
		s.item(typed.TurnID, kind, typed.Text, "", nil)
		if !typed.PendingWork {
			s.endTurn(typed.TurnID, store.ResponseCompleted, "")
		}
	case agent.Interrupted:
		// What the agent had already said still counts, which is why an interrupted turn is
		// cancelled rather than failed: the caller stopped it, nothing went wrong.
		s.endTurn(typed.TurnID, store.ResponseCancelled, "")
	case agent.Error:
		// An error names which part failed rather than which turn, because a transcriber or
		// a voice falling over is not a property of one turn. So it fails whatever was in
		// flight: a turn left open would be read back forever as a question nobody answered.
		s.failTurns(typed.Context, typed.Err)
	}
}

// openTurn opens the row a turn is recorded as and returns its id, opening it only once.
//
// Idempotent on purpose, because two things race to open the same turn. Respond knows the
// turn id as soon as it has asked for the answer and wants the response id back to hand to
// its caller; the Responding event arrives on the fan-out goroutine and is what opens a turn
// nobody prompted. Whichever gets there first writes the row, and the other finds it open.
//
// A turn can also start responding more than once without being new: work handed to a
// subagent comes back and the agent speaks again under the same turn id. That is one response
// with more items in it, not two.
func (s *Session) openTurn(turnID, said string) string {
	if s.records == nil {
		return ""
	}

	s.turnsMu.Lock()
	if s.turns == nil {
		s.turns = map[string]*recordedTurn{}
	}
	if held, open := s.turns[turnID]; open {
		s.turnsMu.Unlock()
		return held.id
	}
	turn := &recordedTurn{id: newID()}
	s.turns[turnID] = turn
	s.turnsMu.Unlock()

	s.records.Responding(store.AgentResponse{
		ID: turn.id, SessionID: s.id, CustomerID: s.spec.CustomerID,
		Said: said, CreatedAt: time.Now().UTC(),
	})
	if said != "" {
		s.item(turnID, store.ItemSaid, said, "", nil)
	}
	return turn.id
}

// item queues one thing that happened, against whichever response the turn is being
// recorded as. An item for a turn that never announced itself is dropped rather than
// inventing a response for it: the row would have no question on it and read as a turn
// nobody asked for.
func (s *Session) item(turnID, kind, text, tool string, payload map[string]any) {
	s.turnsMu.Lock()
	turn, open := s.turns[turnID]
	if !open {
		s.turnsMu.Unlock()
		return
	}
	ordinal := turn.ordinal
	turn.ordinal++
	responseID := turn.id
	s.turnsMu.Unlock()

	s.records.Item(store.AgentResponseItem{
		ResponseID: responseID, Ordinal: ordinal, SessionID: s.id,
		Kind: kind, Text: text, ToolName: tool, Payload: payload, At: time.Now().UTC(),
	})
}

// blockTurn marks a turn the guardrail refused.
func (s *Session) blockTurn(turnID string) {
	s.turnsMu.Lock()
	defer s.turnsMu.Unlock()
	if turn, open := s.turns[turnID]; open {
		turn.blocked = true
	}
}

func (s *Session) turnBlocked(turnID string) bool {
	s.turnsMu.Lock()
	defer s.turnsMu.Unlock()
	turn, open := s.turns[turnID]
	return open && turn.blocked
}

// abandonTurns closes whatever was still in flight when the session ended. Cancelled rather
// than failed: nothing went wrong, the conversation just stopped.
func (s *Session) abandonTurns() {
	if s.records == nil {
		return
	}

	s.turnsMu.Lock()
	open := make([]string, 0, len(s.turns))
	for turnID := range s.turns {
		open = append(open, turnID)
	}
	s.turnsMu.Unlock()

	for _, turnID := range open {
		s.endTurn(turnID, store.ResponseCancelled, "")
	}
}

// failTurns records a failure against every turn still in flight and closes them.
func (s *Session) failTurns(where string, cause error) {
	reason := cause.Error()
	if where != "" {
		reason = where + ": " + reason
	}

	s.turnsMu.Lock()
	failing := make([]string, 0, len(s.turns))
	for turnID := range s.turns {
		failing = append(failing, turnID)
	}
	s.turnsMu.Unlock()

	for _, turnID := range failing {
		s.item(turnID, store.ItemError, reason, "", nil)
		s.endTurn(turnID, store.ResponseFailed, reason)
	}
}

// endTurn closes the row and forgets the turn, so a session that runs for hours does not
// accumulate a map entry per turn it took.
func (s *Session) endTurn(turnID, status, failure string) {
	s.turnsMu.Lock()
	turn, open := s.turns[turnID]
	if open {
		delete(s.turns, turnID)
	}
	s.turnsMu.Unlock()

	if !open {
		return
	}
	s.records.Responded(turn.id, status, failure, time.Now().UTC())
}

// remember keeps what was said, which is all a review needs of a call.
func (s *Session) remember(event Event) {
	var line spoken
	switch typed := event.(type) {
	case agent.Heard:
		line = spoken{text: typed.Text}
	case agent.Responding:
		// A conversation in writing is never Heard: nothing transcribed it, so the only
		// record of what was said to the agent is the prompt it was handed. Without this
		// a text call is reviewed on the agent's half of it. A call with a voice reports
		// both, and taking them both would record everything the caller said twice.
		if !s.spec.Text {
			return
		}
		line = spoken{text: typed.Prompt}
	case agent.Responded:
		line = spoken{agent: true, text: typed.Text}
	default:
		return
	}
	if line.text == "" {
		return
	}

	s.saidMu.Lock()
	defer s.saidMu.Unlock()
	if len(s.said) < reviewLimit {
		s.said = append(s.said, line)
	}
}

// conversation is what was said on the call, oldest first.
func (s *Session) conversation() []spoken {
	s.saidMu.Lock()
	defer s.saidMu.Unlock()
	return append([]spoken(nil), s.said...)
}

// broadcast sends one event to every watcher, dropping it for any that has fallen behind.
func (s *Session) broadcast(event Event) {
	s.recordLog(event)
	s.mu.Lock()
	defer s.mu.Unlock()

	for id, attached := range s.watchers {
		if !attached.send(event) {
			s.logger.Debug("dropped an event for a watcher that is behind",
				"session", s.id, "watcher", id)
		}
	}
}

// askTool publishes a tool request, failing when nobody is connected to answer it.
//
// The failure matters: without it the model would wait out the whole timeout on a call the
// caller disconnected from, and the caller would hear a pause it could not explain.
func (s *Session) askTool(call ToolCall) error {
	if call.TurnID != "" && s.persisted != nil {
		call.CommandID, _ = s.persisted.CommandForTurn(call.TurnID)
	}
	if s.persisted != nil && s.spec.Caller.UserID != "" && (call.CommandID == "" || call.TurnID == "") {
		return errors.New("session: tool call has no durable command binding")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.watchers) == 0 {
		return errors.New("session: nobody is connected to run it")
	}
	delivered := false
	for _, attached := range s.watchers {
		if attached.send(call) {
			delivered = true
		}
	}
	if !delivered {
		return fmt.Errorf("session: %s could not be asked for, the connection is behind", call.Name)
	}
	return nil
}

// think names the model the skills run on when the spec did not, so that an agent written
// down as instructions alone can hand the hard parts over rather than guess at them.
//
// It is the one target looked up before it is asked for. A target a caller named and this
// deployment cannot route is a refusal, but a deployment routing no thinking model should
// still take calls: that agent answers everything itself, the way it goes without search.
func (m *Manager) think(ctx context.Context, spec *Spec) {
	if target, exists := spec.Subagents["default"]; exists {
		spec.SubagentTarget = target
		return
	}
	if spec.Text || spec.SubagentTarget != "" {
		return
	}
	if _, err := m.options.LLM.Resolve(ctx, defaultSubagentTarget, spec.LanguageHints); err != nil {
		m.logger.Debug("this agent has nothing to hand the hard parts to",
			"target", defaultSubagentTarget, "error", err)
		return
	}
	spec.SubagentTarget = defaultSubagentTarget
}

// skills are what the voice model may hand over, which is nothing without a subagent to
// run them. Loading them is skipped rather than failed when the agent is answering
// everything itself, the same way cmd/agent does it.
//
// A spec may spell its skills out, name them, or say nothing and take the built-in set.
// Naming them is what an agent config does, so that editing what a skill means changes
// every agent that uses it rather than every request that mentions it.
func (m *Manager) skills(ctx context.Context, spec Spec) (harness.Skills, error) {
	if spec.SubagentTarget == "" && len(spec.Subagents) == 0 {
		return harness.Skills{}, nil
	}
	if spec.Skills != nil {
		if err := spec.Skills.Validate(); err != nil {
			return harness.Skills{}, err
		}
		declared := *spec.Skills
		declared.Normalize()
		return declared, nil
	}
	if len(spec.SkillNames) == 0 {
		builtins, err := harness.DefaultSkills()
		if err != nil {
			return harness.Skills{}, err
		}
		var available []harness.Skill
		for _, skill := range builtins.Skills {
			binding := skill.Subagent
			if binding == "" {
				binding = "default"
			}
			target, declared := spec.Subagents[binding]
			if !declared && binding == "default" {
				target = spec.SubagentTarget
			}
			if target != "" {
				available = append(available, skill)
			}
		}
		return harness.Skills{Skills: available}, nil
	}
	return m.namedSkills(ctx, spec.CustomerID, spec.ConfigID, spec.SkillNames)
}

// namedSkills looks up skills by name, against the config's own and the built-in set.
//
// The config's own wins over a built-in of the same name: redefining what "think" means is
// how an agent is given a different one, not a collision. A name nothing defines is an
// error rather than a skill quietly missing from the prompt, because the model would
// otherwise be told to hand work to a colleague who does not exist.
func (m *Manager) namedSkills(ctx context.Context, customerID, configID string, names []string) (harness.Skills, error) {
	builtin, err := harness.DefaultSkills()
	if err != nil {
		return harness.Skills{}, err
	}

	defined := map[string]harness.Skill{}
	if m.options.Store != nil {
		stored, err := m.options.Store.SkillsNamed(ctx, customerID, configID, names)
		if err != nil {
			return harness.Skills{}, err
		}
		for _, skill := range stored {
			defined[skill.Name] = harness.Skill{
				Name:         skill.Name,
				Description:  skill.Description,
				Instructions: skill.Instructions,
				Deadline:     time.Duration(skill.DeadlineMs) * time.Millisecond,
				Subagent:     skill.Subagent, CaptureVideo: skill.CaptureVideo,
			}
		}
	}

	resolved := harness.Skills{Skills: make([]harness.Skill, 0, len(names))}
	for _, name := range names {
		skill, known := defined[name]
		if !known {
			skill, known = builtin.Lookup(name)
		}
		if !known {
			return harness.Skills{}, fmt.Errorf("session: there is no skill called %q", name)
		}
		resolved.Skills = append(resolved.Skills, skill)
	}
	if err := resolved.Validate(); err != nil {
		return harness.Skills{}, err
	}
	resolved.Normalize()
	return resolved, nil
}

// CapturesVideo reports whether a configured skill needs the local video worker.
func (s *Session) CapturesVideo() bool {
	for _, skill := range s.skills.Skills {
		if skill.CaptureVideo {
			return true
		}
	}
	return false
}
