package session

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// EdgeFactory opens the transport a session's agent talks over.
//
// It is a function rather than a field so this package does not depend on any one of them.
// The production edge is Stream's WebRTC, whose Opus path is cgo; keeping that in the
// command that wires it means a session can be tested without a sound library or a Stream
// account.
type EdgeFactory func(spec Spec, logger *slog.Logger) (agent.Edge, error)

// Transcript stores what was said, so a call leaves something behind.
type Transcript interface {
	Start(ctx context.Context) error
	Record(event agent.Event)
	// Close reports nothing, because a transcript that failed to flush its last line is
	// not something the caller who ended the call can do anything about.
	Close()
}

// TranscriptFactory opens the transcript for a session. A nil factory, or one that
// declines, means the conversation is simply not kept.
type TranscriptFactory func(spec Spec, logger *slog.Logger) (Transcript, error)

// ManagerOptions is everything a session needs that is the same for all of them.
type ManagerOptions struct {
	LLM *llmrouter.Router
	STT *sttrouter.Router
	TTS *ttsrouter.Router

	// Edge is required: without it there is no call to join.
	Edge EdgeFactory
	// Transcript is optional.
	Transcript TranscriptFactory
	// Memory is optional. Without it every session starts knowing nothing.
	Memory memory.Store
	// Knowledge is optional, and is what a session with a namespace looks things up in.
	Knowledge knowledge.Store
	// Search is optional, and is what every session finds out what is true now with.
	Search *searchrouter.Router
	// Phone is optional, and is what a session with a number transfers through.
	Phone *phone.Service

	Store  *store.Store
	Live   *live.Client
	Logger *slog.Logger
}

// Manager owns the sessions this process is running.
type Manager struct {
	options ManagerOptions
	logger  *slog.Logger
	// calls records conversations so they can be found after this process is gone. Nil
	// without a store, in which case a call is only ever what is happening now.
	calls *callRecorder
	// reviews says what a finished call went like, onto the row calls wrote.
	reviews *reviewer

	mu       sync.Mutex
	sessions map[string]*Session
	closed   bool
}

// NewManager validates the options and returns a Manager. It starts nothing.
func NewManager(options ManagerOptions) (*Manager, error) {
	if options.LLM == nil {
		return nil, errors.New("session: an llm router is required")
	}
	if options.STT == nil {
		return nil, errors.New("session: an stt router is required")
	}
	if options.TTS == nil {
		return nil, errors.New("session: a tts router is required")
	}
	if options.Edge == nil {
		return nil, errors.New("session: an edge factory is required")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	manager := &Manager{
		options:  options,
		logger:   options.Logger,
		sessions: map[string]*Session{},
	}
	if options.Store != nil {
		manager.calls = newCallRecorder(options.Store, options.Logger)
		manager.reviews = newReviewer(options.LLM, options.Store, options.Logger)
	}
	return manager, nil
}

// Create joins a call and returns the session running it.
//
// The agent is joined before this returns, so a caller that gets a session back has one
// that is already listening. A failure anywhere unwinds what was opened: a half-joined
// session would hold a model session and a place in a call that nobody holds a handle to.
func (m *Manager) Create(ctx context.Context, spec Spec) (*Session, error) {
	if err := spec.Normalize(); err != nil {
		return nil, err
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil, errors.New("session: the manager is shut down")
	}
	m.mu.Unlock()

	skills, err := m.skills(ctx, spec)
	if err != nil {
		return nil, err
	}

	box, err := m.box(spec)
	if err != nil {
		return nil, err
	}

	// A text session joins nothing, so no edge is opened for it. Everything downstream
	// treats a missing edge as the conversation having no call rather than as a failure.
	var edge agent.Edge
	switch {
	case spec.Text:
	case spec.Edge != nil:
		edge = spec.Edge
	default:
		edge, err = m.options.Edge(spec, m.logger)
		if err != nil {
			return nil, err
		}
	}

	line, err := m.line(spec)
	if err != nil {
		return nil, err
	}

	created := &Session{
		id:       newID(),
		spec:     spec,
		created:  time.Now(),
		logger:   m.logger,
		watchers: map[uint64]*watcher{},
		state:    Live,
		skills:   skills,
	}
	created.tools = newBridge(
		time.Duration(spec.ToolTimeoutMs)*time.Millisecond,
		created.askTool,
	)

	// The built-in tools are this process's to run rather than the caller's. They go
	// after the caller's so a caller cannot quietly replace a transfer, and the agent
	// drops whichever of them nothing on this call can carry out.
	tools := append([]harness.Tool(nil), spec.Tools...)
	if line != nil || m.reading(spec) || m.searching(spec) {
		builtin, err := harness.DefaultTools()
		if err != nil {
			return nil, err
		}
		tools = append(tools, builtin.Tools...)
	}

	mcp, pluginTools := attachPlugins(ctx, spec, m.options.Store, m.logger)
	tools = append(tools, pluginTools...)
	runner := agent.ToolRunner(created.tools)
	if mcp != nil {
		runner = &pluginRunner{mcp: mcp, next: created.tools}
		created.closers = append(created.closers, mcp.Close)
	}

	created.voiceAgent, err = agent.New(agent.Options{
		Edge:               edge,
		Text:               spec.Text,
		Instructions:       spec.prompt(),
		CustomerID:         spec.CustomerID,
		AgentID:            spec.AgentID,
		CallID:             spec.CallID,
		Tags:               spec.Tags,
		LLM:                m.options.LLM,
		LLMTarget:          spec.LLMTarget,
		STT:                m.options.STT,
		STTTarget:          spec.STTTarget,
		TTS:                m.options.TTS,
		TTSTarget:          spec.TTSTarget,
		SubagentTarget:     spec.SubagentTarget,
		Skills:             skills,
		Telephony:          line,
		ToolRunner:         runner,
		Tools:              harness.Tools{Tools: tools},
		Sandbox:            box,
		Tasks:              spec.Tasks,
		Duplex:             spec.duplex(),
		Voice:              spec.Voice,
		LanguageHints:      spec.LanguageHints,
		Keyterms:           spec.Keyterms,
		MaxTokens:          spec.MaxTokens,
		Memory:             m.options.Memory,
		Knowledge:          m.options.Knowledge,
		KnowledgeNamespace: spec.KnowledgeNamespace,
		Search:             m.options.Search,
		SearchTarget:       spec.SearchTarget,
		AppID:              spec.Memory.AppID,
		MemoryUserID:       spec.Memory.UserID,
		MemoryFilter:       spec.Memory.Filter,
		Store:              m.options.Store,
		Live:               m.options.Live,
		Logger:             m.logger,
	})
	if err != nil {
		return nil, err
	}

	if box != nil {
		created.closers = append(created.closers, func() {
			if err := box.Close(); err != nil {
				m.logger.Error("could not release the sandbox",
					"session", created.id, "error", err)
			}
		})
	}

	if m.options.Transcript != nil {
		// A transcript that cannot be opened is not a reason to refuse the call. What was
		// said is worth keeping; it is not worth not having the conversation for.
		transcript, err := m.options.Transcript(spec, m.logger)
		if err != nil {
			m.logger.Warn("not storing the transcript", "call", spec.CallID, "error", err)
		} else if err := transcript.Start(ctx); err != nil {
			m.logger.Warn("not storing the transcript", "call", spec.CallID, "error", err)
			transcript.Close()
		} else {
			created.transcript = transcript
			created.closers = append(created.closers, transcript.Close)
		}
	}

	// The fan-out starts before joining so nothing said between joining and the first
	// watcher attaching is lost to a channel nobody is reading.
	created.running.Add(1)
	go created.consume()

	// Join takes the background rather than the request's context: the conversation
	// outlives the HTTP call that asked for it, and a session cancelled when the request
	// returned would hang up on the caller immediately.
	if err := created.voiceAgent.Join(context.WithoutCancel(ctx)); err != nil {
		created.Close()
		return nil, err
	}

	if spec.Greeting != "" {
		if err := created.voiceAgent.Say(ctx, spec.Greeting); err != nil {
			created.Close()
			return nil, fmt.Errorf("session: greet: %w", err)
		}
	}

	// The row is queued before the session is reachable, so the call cannot be recorded
	// as ending before it is recorded as starting.
	if m.calls != nil {
		created.closers = append(created.closers, func() {
			m.calls.Ended(created.id, time.Now().UTC())
			// The review runs on a model rather than in this closer, so it is started
			// here and lands on the row whenever it comes back.
			if !spec.NoReview {
				m.reviews.Review(row(created), spec.SubagentTarget, created.conversation())
			}
		})
		m.calls.Started(row(created))
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		created.Close()
		return nil, errors.New("session: the manager is shut down")
	}
	m.sessions[created.id] = created
	m.mu.Unlock()

	m.logger.Info("session joined",
		"session", created.id, "call", spec.CallID, "customer", spec.CustomerID)
	return created, nil
}

// Get returns a session belonging to a customer. Two customers cannot see each other's,
// so an id that exists but belongs to someone else is reported as not existing at all.
func (m *Manager) Get(id, customerID string) (*Session, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	found, ok := m.sessions[id]
	if !ok || found.spec.CustomerID != customerID {
		return nil, false
	}
	return found, true
}

// List returns a customer's sessions, newest first.
func (m *Manager) List(customerID string) []*Session {
	m.mu.Lock()
	defer m.mu.Unlock()

	var theirs []*Session
	for _, found := range m.sessions {
		if found.spec.CustomerID == customerID {
			theirs = append(theirs, found)
		}
	}
	sort.Slice(theirs, func(i, j int) bool {
		return theirs[i].created.After(theirs[j].created)
	})
	return theirs
}

// Close ends a customer's session, reporting whether they had one by that id.
func (m *Manager) Close(id, customerID string) (bool, error) {
	found, ok := m.Get(id, customerID)
	if !ok {
		return false, nil
	}

	m.mu.Lock()
	delete(m.sessions, id)
	m.mu.Unlock()

	return true, found.Close()
}

// Shutdown ends every session, which is what a router does on its way down rather than
// dropping calls by exiting.
func (m *Manager) Shutdown() error {
	m.mu.Lock()
	m.closed = true
	running := make([]*Session, 0, len(m.sessions))
	for _, found := range m.sessions {
		running = append(running, found)
	}
	m.sessions = map[string]*Session{}
	m.mu.Unlock()

	var failures []error
	for _, found := range running {
		if err := found.Close(); err != nil {
			failures = append(failures, err)
		}
	}

	// The recorder goes last so the endings those closes queued are written rather than
	// lost on the way out. The reviews go with it: a summary is worth having, but not
	// worth holding a shutdown open for a model to finish writing.
	if m.calls != nil {
		m.reviews.Close()
		m.calls.Close()
	}
	return errors.Join(failures...)
}

// box is where the session's subagent runs the code it writes, which is nowhere unless the
// caller asked for a sandbox. Nothing is created here: the provider opens one the first
// time code is actually run, so a session that never delegates never pays for it.
func (m *Manager) box(spec Spec) (sandbox.Sandbox, error) {
	if spec.Sandbox == "" {
		return nil, nil
	}
	if spec.Sandbox != daytonaProvider {
		return nil, fmt.Errorf("session: there is no sandbox provider called %q", spec.Sandbox)
	}
	return daytona.New(daytona.Options{Logger: m.logger})
}

// reading reports whether this session has anything to look things up in, which is a
// knowledge store and a namespace to read out of it.
func (m *Manager) reading(spec Spec) bool {
	return m.options.Knowledge != nil && spec.KnowledgeNamespace != ""
}

// searching reports whether this session can find out what is true now. Unlike a handbook
// today is not scoped to a customer, so all this asks is that something routes the search.
func (m *Manager) searching(spec Spec) bool {
	return m.options.Search != nil && spec.SearchTarget != ""
}

// line is what the session may do to the call it is on, which is nothing unless it was
// given a number to act from.
func (m *Manager) line(spec Spec) (agent.Telephony, error) {
	if spec.Phone == nil || spec.Phone.Number == "" {
		return nil, nil
	}
	if m.options.Phone == nil {
		return nil, errors.New("session: this deployment has no telephony, so a number cannot be used")
	}

	return m.options.Phone.Line(phone.LineOptions{
		Owner:        routing.Owner{CustomerID: spec.CustomerID, Tags: spec.Tags},
		From:         spec.Phone.Number,
		CallID:       spec.CallID,
		CallType:     spec.CallType,
		Vendor:       spec.Phone.Vendor,
		VendorCallID: spec.Phone.VendorCallID,
	}), nil
}

// newID is the handle a caller holds a session by. It is random rather than sequential
// because it is the only thing standing between two customers who both guessed at an id.
func newID() string {
	raw := make([]byte, 16)
	// rand.Read on crypto/rand never returns an error, which is why the result is not
	// checked: the alternative would be a session that could not be created.
	_, _ = rand.Read(raw)
	return hex.EncodeToString(raw)
}
