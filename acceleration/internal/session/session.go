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
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
)

// watcherBuffer is how many events may queue for one watcher before it starts losing them.
// A watcher is a control channel rather than the conversation, so a slow one costs its
// reader some detail and nobody on the call anything.
const watcherBuffer = 256

// Defaults a caller can leave out. They are the same ones cmd/agent's flags carry, so a
// session started with an almost empty spec behaves like the demo.
const (
	defaultCallType     = "default"
	defaultUserID       = "vision-agent"
	defaultUserName     = "Vision Agent"
	defaultLLMTarget    = "llm-fast"
	defaultSTTTarget    = "en-low-latency"
	defaultTTSTarget    = "en-low-latency"
	defaultSearchTarget = "search-fast"
	// The model the skills run on. A quality tier rather than a fast one, since the
	// conversation carries on without it: what is handed over is what the talking model
	// could not answer itself.
	defaultSubagentTarget = "multilingual-high-accuracy"
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

// ToolCall is the model asking for one of the caller's own tools. It is the only event a
// watcher is obliged to answer: everything else is a report.
type ToolCall struct {
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
	id      string
	spec    Spec
	created time.Time
	logger  *slog.Logger

	voiceAgent *agent.Agent
	tools      *bridge
	transcript Transcript
	// skills are what the fast model may hand over, resolved once when the session was
	// created. Kept so the call's row can say what was on offer: the spec carries names
	// or nothing at all, and nothing at all means the built-in set.
	skills harness.Skills

	mu sync.Mutex
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

	// closers undo what Create wired up, in reverse.
	closers   []func()
	closeOnce sync.Once
	running   sync.WaitGroup
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
	if model := s.voiceAgent.LLM(); model != nil {
		llm = model.Provider() + "/" + model.Model()
	}
	if voice := s.voiceAgent.TTS(); voice != nil {
		tts = voice.Provider() + "/" + voice.Model()
	}
	return llm, tts
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
	s.mu.Unlock()

	return attached.events, func() {
		s.mu.Lock()
		delete(s.watchers, id)
		s.mu.Unlock()
		attached.close()
	}
}

// Say speaks a piece of text without going through the model.
func (s *Session) Say(ctx context.Context, text string) error {
	return s.voiceAgent.Say(ctx, text)
}

// Respond answers a piece of text through the model, as though a participant had said it.
func (s *Session) Respond(ctx context.Context, text string) error {
	return s.voiceAgent.SimpleResponse(ctx, text)
}

// Interrupt abandons the reply being spoken.
func (s *Session) Interrupt() { s.voiceAgent.Interrupt() }

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
	return s.tools.Resolve(id, output, failure)
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
		if s.transcript != nil {
			s.transcript.Record(event)
		}
		s.remember(event)
		s.broadcast(event)
	}
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
	if spec.SubagentTarget != "" {
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
	if spec.SubagentTarget == "" {
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
		return harness.DefaultSkills()
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
