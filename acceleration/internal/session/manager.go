package session

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/guardrail"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// EdgeFactory opens the transport a session's agent talks over.
//
// It is a function rather than a field so this package does not depend on any one of them.
// The production edge is Stream's WebRTC, whose Opus path is cgo; keeping that in the
// command that wires it means a session can be tested without a sound library or a Stream
// account.
//
// It is handed the Stream app the session is pinned to, which is the app the call has to
// be joined in: the one the customer's clients created it in.
type EdgeFactory func(ctx context.Context, spec Spec, stream streamapp.Bound, logger *slog.Logger) (agent.Edge, error)

// Transcript stores what was said, so a call leaves something behind.
type Transcript interface {
	Start(ctx context.Context) error
	Record(event agent.Event)
	// Reply stores something the agent wrote rather than said. It is separate from Record
	// because a written answer is not an agent event: nothing was spoken, so no turn was
	// started and no reply streamed.
	Reply(text string)
	// Close reports nothing, because a transcript that failed to flush its last line is
	// not something the caller who ended the call can do anything about.
	Close()
}

// TranscriptFactory opens the transcript for a session, in the Stream app the session is
// pinned to. A nil factory, or one that declines, means the conversation is simply not kept.
type TranscriptFactory func(ctx context.Context, spec Spec, stream streamapp.Bound, logger *slog.Logger) (Transcript, error)

// ManagerOptions is everything a session needs that is the same for all of them.
type ManagerOptions struct {
	LLM *llmrouter.Router
	// Speech routers are optional for text sessions and native speech-to-speech.
	STT *sttrouter.Router
	TTS *ttsrouter.Router
	// STS is optional, and is what a session naming a speech-to-speech target holds its
	// conversation with instead of the three above. A deployment without one refuses
	// such a session rather than falling back to the cascade unasked.
	STS *stsrouter.Router

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
	// Classifier is optional, and is what a session with a guardrail asks whether a turn
	// may be answered. Without it a session declaring a guardrail that needs one is
	// refused rather than held unguarded.
	Classifier *lcmrouter.Router
	// Phone is optional, and is what a session with a number transfers through.
	Phone *phone.Service
	// SpeculativeReplies has every agent start its reply before the flow controller has
	// ruled on the words, and hold it until the ruling says to answer.
	SpeculativeReplies bool
	// Stream says which Stream app, and with which credential, each session acts in. It
	// is optional: without it a session has no app, and anything needing one fails where
	// it needs it, as it does on a deployment with no Stream credentials.
	Stream *streamapp.Clients
	// Conversations is optional, and is the persistent text store a caller already holds.
	// Without one the manager opens its own over the configured outbox directory.
	Conversations *persistent.Service

	Store *store.Store
	// Configs reads what a session is opened from -- the skills it names -- through
	// whatever cache is in front of Postgres. Built over Store when it is not given, in
	// which case every read is a query, which is what a deployment without Redis does.
	Configs *appconfig.Store
	Live    *live.Client
	// Directory is where this node says which sessions it is running, so the deployment's
	// other nodes can forward what only this one can answer. Nil keeps a session
	// reachable on this node alone.
	Directory *node.Directory
	// PluginAuth signs an end user into the plugins an agent names per user, sending the
	// provider back to this deployment's public URL. Nil sends it to localhost.
	PluginAuth *plugins.Auth
	// DetachedGrace is how long a persistent text session outlives its last watcher.
	// Zero is defaultDetachedGrace.
	DetachedGrace time.Duration
	Logger        *slog.Logger
}

// defaultDetachedGrace is long enough to finish a plugin login in another tab, or to leave
// the page a conversation is on and come back to it.
const defaultDetachedGrace = 5 * time.Minute

// Manager owns the sessions this process is running.
type Manager struct {
	// hookPins are the apps hooks came from, by the call or channel they named.
	hookPinsMu sync.Mutex
	hookPins   map[string]hookPinned

	logs          *logRecorder
	conversations *persistent.Service
	options       ManagerOptions
	logger        *slog.Logger
	// calls records conversations so they can be found after this process is gone. Nil
	// without a store, in which case a call is only ever what is happening now.
	calls *callRecorder
	// records keeps sessions, turns and what each turn did, which is what an old
	// conversation is read back from. Nil without a store, and never handed an incognito
	// session.
	records *sessionRecorder
	// reviews says what a finished call went like, onto the row calls wrote.
	reviews *reviewer
	// titles names persistent conversations nobody named, on the session row and the channel.
	titles *titler
	// hosts runs the tools workers host for an agent config. Nil offers none.
	hosts ToolHosts

	mu       sync.Mutex
	sessions map[string]*Session
	closed   bool
}

// NewManager validates the options and returns a Manager. It starts nothing.
func NewManager(options ManagerOptions) (*Manager, error) {
	if options.LLM == nil {
		return nil, errors.New("session: an llm router is required")
	}
	if options.Edge == nil {
		return nil, errors.New("session: an edge factory is required")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	if options.DetachedGrace == 0 {
		options.DetachedGrace = defaultDetachedGrace
	}

	manager := &Manager{
		hookPins:      map[string]hookPinned{},
		options:       options,
		logger:        options.Logger,
		sessions:      map[string]*Session{},
		conversations: options.Conversations,
	}
	if options.Store != nil {
		if options.Configs == nil {
			configs, err := appconfig.New(appconfig.Options{Store: options.Store, Logger: options.Logger})
			if err != nil {
				return nil, err
			}
			manager.options.Configs = configs
		}
		manager.logs = newLogRecorder(options.Store, options.Logger)
		manager.calls = newCallRecorder(options.Store, options.Logger)
		manager.records = newSessionRecorder(options.Store, options.Logger)
		manager.reviews = newReviewer(options.LLM, options.Store, options.Logger)
	}
	manager.titles = newTitler(options.LLM, manager.records, options.Logger)
	return manager, nil
}

// Create joins a call and returns the session running it.
//
// The agent is joined before this returns, so a caller that gets a session back has one
// that is already listening. A failure anywhere unwinds what was opened: a half-joined
// session would hold a model session and a place in a call that nobody holds a handle to.
func (m *Manager) Create(ctx context.Context, spec Spec) (*Session, error) {
	if err := spec.Normalize(); err != nil {
		return nil, stack.Wrap(err)
	}
	// Refuse unsupported voice modes before opening a call or persistent resource.
	if spec.Native() && m.options.STS == nil {
		return nil, stack.Wrap(errors.New("session: this deployment routes no speech-to-speech model"))
	}
	if !spec.Text && !spec.Native() {
		if m.options.STT == nil {
			return nil, stack.Wrap(errors.New("session: an stt router is required for voice sessions"))
		}
		if m.options.TTS == nil {
			return nil, stack.Wrap(errors.New("session: a tts router is required for voice sessions"))
		}
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil, stack.Wrap(errors.New("session: the manager is shut down"))
	}
	_, live := m.sessions[spec.ID]
	m.mu.Unlock()
	if live {
		return nil, stack.Wrap(ErrSessionExists)
	}
	// The id is the row's primary key whoever owns it, so one somebody already used would
	// write this session over theirs.
	if m.options.Store != nil {
		taken, err := m.options.Store.SessionExists(ctx, spec.ID)
		if err != nil {
			return nil, stack.Wrap(err)
		}
		if taken {
			return nil, stack.Wrap(ErrSessionExists)
		}
	}

	// The Stream app the session acts in is settled once, before anything is done there,
	// and the whole session keeps it: its call, its transcript and the rows it writes.
	stream, err := m.stream(ctx, spec)
	if err != nil {
		return nil, err
	}
	spec.StreamApp = stream.Identity.StreamApp
	if err := m.keepable(ctx, spec, stream); err != nil {
		return nil, err
	}
	// In app mode a session with no Stream app at all is pinned to none, rather than left
	// unpinned, which reads as the deployment's own app and would let that app's hooks and
	// backfill claim it.
	if stream.Client == nil && m.options.Stream != nil && m.options.Stream.PerApp() {
		spec.StreamApp = store.ForeignStreamApp
	}

	var remembering memory.Store
	if spec.Memory.UserID != "" {
		if m.options.Memory == nil {
			return nil, stack.Wrap(ErrNoMemory)
		}
		remembering = m.options.Memory
	}

	opened := false
	// A new text conversation is kept in Stream Chat by default, but a deployment without Chat
	// credentials still holds it: what was said is worth keeping, not worth refusing the
	// conversation for. Resuming or forking one needs the channel, so those still fail.
	if spec.PersistConversation && spec.Text && spec.ConversationID == "" && spec.Recall == nil {
		if _, err := m.Conversations(); err != nil {
			m.logger.Warn("not keeping the conversation in Stream Chat", "error", err)
			spec.PersistConversation = false
		}
	}
	var conv *persistent.Conversation
	var previous []llm.Message
	if spec.PersistConversation {
		if !spec.Text {
			return nil, stack.Wrap(errors.New("persistent conversations require text mode"))
		}
		service, err := m.Conversations()
		if err != nil {
			return nil, stack.Wrap(err)
		}
		if spec.ConversationID != "" {
			m.takeOver(spec.CustomerID, spec.ConversationID)
		}
		var truncated bool
		conv, previous, truncated, err = service.OpenInApp(ctx, spec.StreamApp, spec.CustomerID, spec.AgentID, spec.ConversationID, spec.Caller.UserID, spec.UserID, spec.Custom, memory.Scope{AppID: spec.Memory.AppID, UserID: spec.Memory.UserID, Extra: spec.Memory.Filter})
		if errors.Is(err, streamapp.ErrReadOnly) {
			return nil, ErrConversationReadOnly
		}
		if err != nil {
			return nil, stack.Wrap(err)
		}
		// A conversation is kept where it was first written, and a session resuming it
		// acts there too, whichever app its customer acts in now.
		if kept := conv.StreamApp(); kept != spec.StreamApp && m.options.Stream != nil {
			if stream, err = m.options.Stream.ForApp(ctx, spec.CustomerID, kept); err != nil {
				conv.Release()
				if errors.Is(err, streamapp.ErrReadOnly) {
					return nil, ErrConversationReadOnly
				}
				return nil, fmt.Errorf("session: the app conversation %s is kept in: %w", conv.CID(), err)
			}
			spec.StreamApp = kept
		}
		spec.ConversationID = conv.CID()
		conv.ShowTools(spec.VisibleTools)
		conv.AcceptLogins(Logins(spec))
		// A resume was deliberately not given an agent id, because only the conversation
		// knows the one its transcript was written under.
		spec.AgentID = conv.Agent()
		spec.ContextTruncated = truncated
		displays := map[string]persistent.ToolDisplay{}
		for _, tool := range spec.Tools {
			display := persistent.ToolDisplay{Title: tool.DisplayTitle, Client: tool.Client}
			if approval := tool.Approval; approval != nil {
				display.Approval = &persistent.ToolApproval{
					Title: approval.Title, Message: approval.Message, ReasonArgument: approval.ReasonArgument,
					AllowTitle: approval.AllowTitle, DeclineTitle: approval.DeclineTitle,
				}
			}
			displays[tool.Name] = display
		}
		conv.DescribeTools(displays)
		// A fork opens an empty channel of its own and then reads the parent's, so the model
		// carries on from what was said while the transcripts stay separate. The parent's
		// half goes first because it happened first.
		if spec.Recall != nil && spec.Recall.Messages == nil {
			recalled, cut, err := service.ContextForCaller(ctx, spec.CustomerID,
				spec.Recall.AgentID, spec.Recall.ConversationID, spec.Caller.UserID)
			if err != nil {
				conv.Release()
				return nil, stack.Wrap(fmt.Errorf("session: reading the conversation being forked: %w", err))
			}
			previous = append(recalled, previous...)
			spec.ContextTruncated = spec.ContextTruncated || cut
		}
		defer func() {
			if !opened {
				conv.Release()
			}
		}()
	} else if spec.ConversationID != "" {
		service, err := m.Conversations()
		if err != nil {
			return nil, stack.Wrap(err)
		}
		// A call on a conversation writes its words into the conversation's channel, which
		// is only there in the app the conversation is kept in.
		if !spec.Text {
			if err := m.sameApp(ctx, service, spec); err != nil {
				return nil, err
			}
		}
		var truncated bool
		previous, truncated, err = service.ContextForCaller(ctx, spec.CustomerID, spec.AgentID, spec.ConversationID, spec.Caller.UserID, spec.UserID)
		if err != nil {
			return nil, stack.Wrap(err)
		}
		spec.ContextTruncated = truncated
	}
	if spec.Recall != nil && spec.Recall.Messages != nil {
		previous = append(append([]llm.Message(nil), spec.Recall.Messages...), previous...)
	}
	m.supersede(spec)
	m.think(ctx, &spec)

	skills, err := m.skills(ctx, spec)
	if err != nil {
		return nil, stack.Wrap(err)
	}

	box, err := m.box(spec)
	if err != nil {
		return nil, stack.Wrap(err)
	}

	// A text session joins nothing, so no edge is opened for it. Everything downstream
	// treats a missing edge as the conversation having no call rather than as a failure.
	var edge agent.Edge
	switch {
	case spec.Text:
	case spec.Edge != nil:
		edge = spec.Edge
	default:
		edge, err = m.options.Edge(ctx, spec, stream, m.logger)
		if err != nil {
			return nil, stack.Wrap(err)
		}
	}

	line, err := m.line(spec)
	if err != nil {
		return nil, stack.Wrap(err)
	}

	created := &Session{
		logs:      m.logs,
		persisted: conv,
		id:        spec.ID,
		spec:      spec,
		// Postgres keeps microseconds. The live session sorts by the same instant as its
		// row, or a cursor taken from one would hand the other back on the next page.
		created:       time.Now().UTC().Truncate(time.Microsecond),
		logger:        m.logger,
		watchers:      map[uint64]*watcher{},
		detachedGrace: m.options.DetachedGrace,
		state:         Live,
		modality:      store.ModalityVoice,
		skills:        skills,
	}
	if spec.Text {
		created.modality = store.ModalityText
	}
	created.tools = newBridge(
		time.Duration(spec.ToolTimeoutMs)*time.Millisecond,
		created.askTool,
	)

	// The built-in tools are this process's to run rather than the caller's. They go
	// after the caller's so a caller cannot quietly replace a transfer, and the agent
	// drops whichever of them nothing on this call can carry out.
	tools := append([]harness.Tool(nil), spec.Tools...)
	var callers agent.ToolRunner = created.tools
	tools, callers = m.hostedTools(spec, created.id, tools, callers)
	if line != nil || m.reading(spec) || m.searching(spec) {
		builtin, err := harness.DefaultTools()
		if err != nil {
			return nil, stack.Wrap(err)
		}
		tools = append(tools, builtin.Tools...)
	}

	mcp, pluginTools, unconnected := attachPlugins(ctx, spec, m.options.Store, m.options.PluginAuth, m.logger)
	tools = append(tools, pluginTools...)
	tools = append(tools, unconnectedTools(unconnected)...)
	spec.ServerInstructions = serverInstructions(spec.MCPServers, mcp)
	created.spec.ServerInstructions = spec.ServerInstructions
	var runner agent.ToolRunner = &videoRunner{next: callers, session: created}
	if mcp != nil || len(unconnected) > 0 {
		runner = &pluginRunner{mcp: mcp, unconnected: unconnected, next: runner}
	}
	if mcp != nil {
		created.closers = append(created.closers, mcp.Close)
	}
	if own := m.userPlugins(spec, runner); own != nil {
		tools = append(tools, plugins.UserTools(own.offered)...)
		runner = own
		created.closers = append(created.closers, own.Close)
	}

	var toolStarted func(agent.ToolStarted)
	if conv != nil {
		toolStarted = func(event agent.ToolStarted) { conv.Observe(event) }
	}
	screening, err := m.guardrail(ctx, spec, stream.Identity)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	if screening != nil {
		created.closers = append(created.closers, func() {
			if err := screening.Close(); err != nil {
				m.logger.Error("could not close the guardrail",
					"session", created.id, "error", err)
			}
		})
	}
	created.voiceAgent, err = agent.New(agent.Options{
		OnToolStarted: toolStarted,
		Edge:          edge,
		Text:          spec.Text,
		Instructions:  spec.prompt(),
		CustomerID:    spec.CustomerID,
		Caller:        spec.Caller,
		AgentID:       spec.AgentID,
		ConfigID:      spec.ConfigID,
		CallID:        spec.CallID,
		Tags:          spec.Tags,
		LLM:           m.options.LLM,
		LLMTarget:     spec.LLMTarget,
		STT:           m.options.STT,
		STTTarget:     spec.STTTarget,
		TTS:           m.options.TTS,
		TTSTarget:     spec.TTSTarget,
		// Every router the deployment has is handed over, whichever pipeline the session
		// starts on, so it can be moved onto the other one mid-call.
		STS:                m.options.STS,
		STSTarget:          spec.STSTarget,
		SubagentTarget:     spec.SubagentTarget,
		ControllerTarget:   spec.ControllerTarget,
		Skills:             skills,
		Telephony:          line,
		ToolRunner:         runner,
		Tools:              harness.Tools{Tools: tools},
		Sandbox:            box,
		Publish:            publisher(conv),
		Tasks:              spec.Tasks,
		Duplex:             spec.duplex(),
		VideoSource:        spec.VideoSource,
		VideoMaxFrames:     spec.VideoMaxFrames,
		Voice:              spec.Voice,
		Speed:              spec.Speed,
		LanguageHints:      spec.LanguageHints,
		Keyterms:           spec.Keyterms,
		MaxTokens:          spec.MaxTokens,
		Overwrites:         spec.LLMOverwrites(),
		Memory:             remembering,
		Knowledge:          m.options.Knowledge,
		KnowledgeNamespace: spec.KnowledgeNamespace,
		Search:             m.options.Search,
		SearchTarget:       spec.SearchTarget,
		Guardrail:          screening,
		SpeculativeReplies: m.options.SpeculativeReplies,
		AppID:              spec.Memory.AppID,
		SessionID:          spec.ID,
		Incognito:          spec.Incognito,
		MemoryUserID:       spec.Memory.UserID,
		MemoryFilter:       spec.Memory.Filter,
		Store:              m.options.Store,
		Live:               m.options.Live,
		Logger:             m.logger,
	})
	if err != nil {
		return nil, stack.Wrap(err)
	}

	if box != nil {
		created.closers = append(created.closers, func() {
			if err := box.Close(); err != nil {
				m.logger.Error("could not release the sandbox",
					"session", created.id, "error", err)
			}
		})
	}

	if m.options.Transcript != nil && conv == nil && !spec.Incognito {
		// A transcript that cannot be opened is not a reason to refuse the call. What was
		// said is worth keeping; it is not worth not having the conversation for.
		transcript, err := m.options.Transcript(ctx, spec, stream, m.logger)
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

	if conv == nil && spec.ConversationID != "" {
		m.logger.Info("restored conversation history into the voice session",
			"call", spec.CallID, "conversation", spec.ConversationID,
			"turns", len(previous), "truncated", spec.ContextTruncated)
	}
	created.voiceAgent.RestoreHistory(previous)
	if conv != nil {
		conv.Attach(func(update persistent.Updated) { created.broadcast(update) })
		created.closers = append(created.closers, conv.Release)
		// A caller that named the conversation named it; only an unnamed one is named here.
		if service, err := m.Conversations(); err == nil && spec.Title == "" && spec.Description == "" {
			created.naming = &naming{titles: m.titles, service: service, earlier: spokenOf(previous)}
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
		return nil, stack.Wrap(err)
	}

	if spec.Greeting != "" {
		// A native model has no way to say exact words, so it is asked to open with the
		// greeting rather than handed it to read out: what the caller hears is the model's
		// own rendering of it, which is the only kind of speech such a model has.
		greet := created.voiceAgent.Say
		greeting := spec.Greeting
		if spec.Native() {
			greet = created.voiceAgent.Prompt
			greeting = "Open the call by greeting the caller. Say this, in these words or close to them: " + spec.Greeting
		}
		if err := greet(ctx, greeting); err != nil {
			created.Close()
			return nil, stack.Wrap(fmt.Errorf("session: greet: %w", err))
		}
	}

	// The row is queued before the session is reachable, so the call cannot be recorded
	// as ending before it is recorded as starting. An incognito session has none: the row
	// names the session, its caller and the instructions it ran with.
	if m.calls != nil && !spec.Incognito {
		created.calls = m.calls
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

	// An incognito session is never handed the recorder, so nothing downstream has to
	// remember to check the flag: there is simply nowhere for it to write.
	if m.records != nil && !spec.Incognito {
		created.records = m.records
		created.closers = append(created.closers, func() {
			m.records.Closed(created.id, time.Now().UTC())
		})
		m.records.Opened(sessionRow(created))
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		created.Close()
		return nil, stack.Wrap(errors.New("session: the manager is shut down"))
	}
	if _, raced := m.sessions[created.id]; raced {
		m.mu.Unlock()
		created.Close()
		return nil, stack.Wrap(ErrSessionExists)
	}
	m.sessions[created.id] = created
	m.mu.Unlock()

	// Said after the session is here to be found, so a peer that forwards a request on
	// the strength of it has somewhere to forward it to.
	if m.options.Directory != nil {
		m.options.Directory.Hold(ctx, created.id, spec.AgentID)
	}

	m.logger.Info("session joined",
		"session", created.id, "call", spec.CallID, "customer", spec.CustomerID)
	opened = true
	return created, nil
}

// supersede ends whatever this agent was already doing in this call, before the new one
// joins rather than after.
//
// A session outlives the connection that asked for it, so an agent that was restarted or
// killed is still in its call. Two instances of one agent hear each other: each transcribes
// the other as a caller and they answer each other until nobody else can be heard. There is
// never a reason to want both, so the newest wins.
func (m *Manager) supersede(spec Spec) {
	m.mu.Lock()
	var left []*Session
	for id, found := range m.sessions {
		if sameSeat(found.spec, spec) {
			left = append(left, found)
			delete(m.sessions, id)
		}
	}
	m.mu.Unlock()

	for _, found := range left {
		m.release(found.id)
	}

	for _, found := range left {
		m.logger.Info("ending the session this agent left behind in the call",
			"session", found.id, "call", spec.CallID, "user", spec.UserID)
		if err := found.Close(); err != nil {
			m.logger.Warn("the session left behind did not end cleanly",
				"session", found.id, "error", err)
		}
	}
}

// takeOver ends the session holding the persistent conversation cid when nobody is
// watching it, so reopening the conversation does not wait out that session's grace: the
// client reopening it is most likely the one that stopped watching, after a crash.
func (m *Manager) takeOver(customer, cid string) {
	m.mu.Lock()
	var left []*Session
	for id, found := range m.sessions {
		if found.spec.CustomerID == customer && found.unwatchedFor(cid) {
			left = append(left, found)
			delete(m.sessions, id)
		}
	}
	m.mu.Unlock()

	for _, found := range left {
		m.release(found.id)
		m.logger.Info("reopening a conversation nobody was watching",
			"session", found.id, "conversation", cid)
		found.abandon()
	}
}

// sameSeat reports whether two specs put the same agent in the same call. A text session
// joins nothing, so it shares a seat with nobody.
func sameSeat(existing, wanted Spec) bool {
	return !existing.Text && !wanted.Text &&
		existing.CustomerID == wanted.CustomerID &&
		existing.CallType == wanted.CallType &&
		existing.CallID == wanted.CallID &&
		existing.UserID == wanted.UserID
}

// Owner is who a session belongs to: the customer it is billed to, and the end user it is
// for. That user is whoever asked from their own device, or whoever the customer's backend
// said it was asking on behalf of.
type Owner struct {
	CustomerID string
	UserID     string
	Kind       auth.Kind
}

// OwnerOf is who the session a spec describes will belong to.
func OwnerOf(spec Spec) Owner {
	return Owner{CustomerID: spec.CustomerID, UserID: spec.Caller.UserID, Kind: spec.CallerKind}
}

// Reaches reports whether a caller may have a session owned by other.
//
// Exported because the same question has to be asked of a session that has ended, which
// this package no longer holds: the row is read by the API and checked against this, so a
// stored conversation is reached on exactly the terms a live one is.
//
// A backend reaches every session its customer has: it runs the application, so closing a
// session a device left behind is its job. An end user reaches only their own, and both
// halves of who they are have to match. The kind is half of it because an anonymous
// caller may go by any name it likes: without it, typing somebody else's user id into a
// query parameter would be enough to read their conversation.
//
// An anonymous caller that names nobody at all owns nothing anybody else can be told
// apart from, so for those the session id is the whole of the authority — it is random
// and it is never listed. Naming a user id is what makes an anonymous session private.
//
// The one place the kinds need not be equal is a session a backend opened in somebody's
// name, which that person then reaches from their own device: a backend opening the
// conversation and handing over the id is the ordinary shape of an integration, and
// requiring equality would refuse them the session that was made for them. Anonymous is
// left out of that, because an anonymous name is a claim nobody checked and allowing it
// would make guessing whose a session was enough to read it.
func (o Owner) Reaches(other Owner) bool {
	if o.CustomerID != other.CustomerID {
		return false
	}
	if o.Kind == auth.KindServer {
		return true
	}
	if o.UserID != other.UserID {
		return false
	}
	return o.Kind == other.Kind ||
		(other.Kind == auth.KindServer && o.Kind.Verified() && o.UserID != "")
}

// Get returns a session its owner may have. Two customers cannot see each other's and
// neither can two people, so an id that exists but belongs to somebody else is reported
// as not existing at all: a refusal would confirm it was real.
func (m *Manager) Get(id string, owner Owner) (*Session, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	found, ok := m.sessions[id]
	if !ok || !owner.Reaches(OwnerOf(found.spec)) {
		return nil, false
	}
	return found, true
}

// ByAgent returns the session writing to an agent id, whoever it belongs to.
//
// No customer is asked for, unlike Get, because the callers that need this have no customer
// to ask with: an arriving message names a channel and nothing else, and the session running
// on it is what says whose it is. An agent id names one session at a time, because a second
// session on the same one would be two agents writing into one conversation.
//
// The newest wins if that ever happens, which is the one a person writing there is watching.
func (m *Manager) ByAgent(agentID string) (*Session, bool) {
	return m.ByAgentWhere(agentID, nil)
}

// ByAgentWhere is ByAgent among the sessions a test admits, by their customer and the Stream
// app they act in. An agent id is the caller's to choose, so two customers' sessions may
// share one, each in its own app.
func (m *Manager) ByAgentWhere(agentID string, admits func(customer string, app int64) bool) (*Session, bool) {
	if agentID == "" {
		return nil, false
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	var newest *Session
	for _, found := range m.sessions {
		if found.spec.AgentID != agentID {
			continue
		}
		if admits != nil && !admits(found.spec.CustomerID, found.spec.StreamApp) {
			continue
		}
		if newest == nil || found.created.After(newest.created) {
			newest = found
		}
	}
	return newest, newest != nil
}

// List returns the sessions an owner may have, newest first. A backend gets its
// customer's; an end user gets their own, which is what stops a list being a way to find
// out who else is talking to the agent.
func (m *Manager) List(owner Owner) []*Session {
	m.mu.Lock()
	defer m.mu.Unlock()

	var theirs []*Session
	for _, found := range m.sessions {
		if !owner.Reaches(OwnerOf(found.spec)) {
			continue
		}
		// An anonymous caller that named nobody is told about nothing. It reaches its
		// own session by holding the id, and listing them would hand one stranger
		// another's.
		if owner.Kind != auth.KindServer && owner.UserID == "" {
			continue
		}
		theirs = append(theirs, found)
	}
	sort.Slice(theirs, func(i, j int) bool {
		return theirs[i].created.After(theirs[j].created)
	})
	return theirs
}

// Found is a session a query turned up, whether or not this process still holds it.
//
// Live is the session if it is still running here, nil for one that ended or one another
// instance is holding. Stored is the row, nil for a session running without a store to
// record it. At least one of the two is set, and a caller rendering these reads the live
// one first: a session in flight knows what it resolved its models to, which the row does
// not carry.
type Found struct {
	Live   *Session
	Stored *store.AgentSession
}

// ID is the session's id whichever half is holding it.
func (f Found) ID() string {
	switch {
	case f.Live != nil:
		return f.Live.ID()
	case f.Stored != nil:
		return f.Stored.ID
	}
	return ""
}

// Query returns the sessions an owner may have, most recently updated first, including ones
// that have already ended.
//
// The live sessions and the stored rows are one list rather than two, deduplicated by id,
// because a caller asking for their conversations does not care which of them this process
// happens to be holding. A live session wins where both exist: it is the same conversation
// and the live one knows more about it.
//
// Without a store this is List with filters, which is the honest answer for a deployment
// that keeps nothing: there is no history to offer.
func (m *Manager) Query(ctx context.Context, owner Owner, filter store.SessionFilter) ([]Found, error) {
	return m.find(ctx, owner, "", filter)
}

// Search is Query with words instead of a filter, and needs a store: what it searches is
// what the caller named their conversations, which only the rows carry.
func (m *Manager) Search(ctx context.Context, owner Owner, text string, filter store.SessionFilter) ([]Found, error) {
	return m.find(ctx, owner, text, filter)
}

func (m *Manager) find(ctx context.Context, owner Owner, text string, filter store.SessionFilter) ([]Found, error) {
	// An end user only ever reaches their own, so the filter is narrowed here rather than
	// trusted from the request. A user id a caller could widen is not a boundary at all.
	if owner.Kind != auth.KindServer {
		if owner.UserID == "" {
			// An anonymous caller that named nobody is told about nothing, the same as in
			// List: they reach their own session by holding its id, and listing them would
			// hand one stranger another's.
			return nil, nil
		}
		filter.UserID = owner.UserID
	}

	found := make([]Found, 0, filter.Limit)
	seen := map[string]int{}
	// Only a live session nothing records is listed from memory. A recorded one is listed
	// by its row, so it sorts and pages on the row's updated_at like every other.
	for _, live := range m.List(owner) {
		// A live session has no title to search, so a search skips them unless the row
		// behind them matches. Whatever the store turns up is merged in below.
		if text != "" || live.records != nil || !matchesLive(live, filter) {
			continue
		}
		seen[live.ID()] = len(found)
		found = append(found, Found{Live: live})
	}

	if m.options.Store != nil {
		// Rows are written behind the conversation, so a session opened a moment ago is
		// only listed once the writer has caught up with it.
		if m.records != nil {
			if err := m.records.Flush(ctx); err != nil {
				return nil, err
			}
		}
		var rows []store.AgentSession
		var err error
		if text != "" {
			rows, err = m.options.Store.SearchSessions(ctx, owner.CustomerID, text, filter)
		} else {
			rows, err = m.options.Store.QuerySessions(ctx, owner.CustomerID, filter)
		}
		if err != nil {
			return nil, err
		}
		for _, row := range rows {
			stored := row
			if at, already := seen[row.ID]; already {
				found[at].Stored = &stored
				continue
			}
			entry := Found{Stored: &stored}
			// A row this process is still holding is handed back with both halves even when
			// the live pass skipped it, which is what a search does.
			if live, ok := m.Get(row.ID, owner); ok {
				entry.Live = live
			}
			seen[row.ID] = len(found)
			found = append(found, entry)
		}
	}

	// A search is already ranked by the store, so only a plain query is sorted here, on
	// the same keys as the store so a cursor from either half holds for both.
	if text == "" {
		sort.Slice(found, func(i, j int) bool {
			a, b := found[i].updatedAt(), found[j].updatedAt()
			if !a.Equal(b) {
				return a.After(b)
			}
			return found[i].ID() > found[j].ID()
		})
	}
	// One more than the page, which is how the caller tells there is another.
	if limit := store.SessionLimit(filter.Limit) + 1; len(found) > limit {
		found = found[:limit]
	}
	return found, nil
}

// Position is where the session sits in the list it was found in, which is what a cursor
// holds.
func (f Found) Position() store.SessionPosition {
	position := store.SessionPosition{UpdatedAt: f.updatedAt(), ID: f.ID()}
	if f.Stored != nil {
		position.Rank = f.Stored.Rank
	}
	return position
}

// updatedAt is what the list sorts on: the row's updated_at, or when it began for a session
// nothing records, which has no later change written anywhere to sort by.
func (f Found) updatedAt() time.Time {
	if f.Stored != nil {
		return f.Stored.UpdatedAt
	}
	if f.Live != nil {
		return f.Live.CreatedAt()
	}
	return time.Time{}
}

// matchesLive applies the filter to a session that has not been written down, so a query
// answers the same way with a store and without one.
func matchesLive(live *Session, filter store.SessionFilter) bool {
	spec := live.Spec()
	switch {
	case filter.UserID != "" && spec.Caller.UserID != filter.UserID:
		return false
	case filter.ConfigID != "" && spec.ConfigID != filter.ConfigID:
		return false
	case filter.AgentName != "" && spec.AgentName != filter.AgentName:
		return false
	case filter.Project != "" && spec.Project != filter.Project:
		return false
	case filter.Modality != "" && live.Modality() != filter.Modality:
		return false
	case filter.AgentID != "" && spec.AgentID != filter.AgentID:
		return false
	case filter.State == store.SessionRunning && live.State() != Live,
		filter.State == store.SessionClosed && live.State() != Ended:
		return false
	case !filter.After.IsZero() && live.CreatedAt().Before(filter.After):
		return false
	case !filter.Before.IsZero() && !live.CreatedAt().Before(filter.Before):
		return false
	case !contains(spec.Custom, filter.Custom):
		return false
	case filter.Cursor != nil && !before(live.CreatedAt(), live.ID(), *filter.Cursor):
		return false
	}
	return true
}

// contains is the store's custom @> ?::jsonb, for a session that has no row to ask.
func contains(custom map[string]any, wanted map[string]string) bool {
	for key, value := range wanted {
		held, ok := custom[key]
		if !ok || fmt.Sprint(held) != value {
			return false
		}
	}
	return true
}

// before is the store's (updated_at, id) < (?, ?), for a session that has no row to ask.
// Ids are lowercase hex, which Postgres collates the same way Go compares bytes.
func before(updated time.Time, id string, cursor store.SessionPosition) bool {
	if !updated.Equal(cursor.UpdatedAt) {
		return updated.Before(cursor.UpdatedAt)
	}
	return id < cursor.ID
}

// Close ends a session its owner may have, reporting whether they had one by that id.
func (m *Manager) Close(id string, owner Owner) (bool, error) {
	found, ok := m.Get(id, owner)
	if !ok {
		return false, nil
	}

	m.mu.Lock()
	delete(m.sessions, id)
	m.mu.Unlock()
	m.release(id)

	return true, found.Close()
}

// EndPinned ends every session acting in one of a customer's Stream apps, which is what
// the router does once it stops acting there: the app disconnected, blocked, or the key it
// was opened with dropped. Each closes its call, transcript and guardrail with it. It
// reports how many it ended.
func (m *Manager) EndPinned(customer string, app int64) int {
	m.mu.Lock()
	var pinned []*Session
	for id, found := range m.sessions {
		if spec := found.Spec(); spec.CustomerID == customer && spec.StreamApp == app {
			pinned = append(pinned, found)
			delete(m.sessions, id)
		}
	}
	m.mu.Unlock()

	for _, found := range pinned {
		if err := found.Close(); err != nil {
			m.logger.Warn("a session pinned to a Stream app the router stopped acting in did not close cleanly",
				"session", found.ID(), "customer_id", customer, "error", err)
		}
	}
	return len(pinned)
}

// releaseTimeout bounds taking back this node's claim on a session. Short, because a
// claim that is not taken back expires on its own.
const releaseTimeout = 2 * time.Second

// release stops telling this deployment's other nodes that a session is here.
func (m *Manager) release(id string) {
	if m.options.Directory == nil {
		return
	}
	// Not the caller's context: a session is let go of on paths that have none, and one
	// cancelled the moment the answer is written would leave the claim behind.
	ctx, cancel := context.WithTimeout(context.Background(), releaseTimeout)
	defer cancel()
	m.options.Directory.Release(ctx, id)
}

// Running reports whether this process is running a session, whoever it belongs to.
//
// No owner is asked for, unlike Get, because which node holds a session is not a question
// about who may see it: the node that answers checks that for itself.
func (m *Manager) Running(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	_, running := m.sessions[id]
	return running
}

// Delete stops a session if it is running and deletes it: its row, its turns, and what it
// taught the memory store. Whoever asks has to have checked the session is the owner's, since
// one that ended is no longer here to check against.
func (m *Manager) Delete(ctx context.Context, id string, owner Owner) error {
	if _, err := m.Close(id, owner); err != nil {
		return err
	}
	// The row is written behind the conversation, so a session opened a moment ago may not
	// have one yet; deleting before it lands would leave it to be written afterwards.
	if m.records != nil {
		if err := m.records.Flush(ctx); err != nil {
			return err
		}
	}
	if m.options.Memory != nil {
		if err := m.options.Memory.ForgetRun(ctx, owner.CustomerID, id); err != nil {
			return err
		}
	}
	if m.options.Store == nil {
		return nil
	}
	return m.options.Store.DeleteSession(ctx, owner.CustomerID, id)
}

// TruncateMemories deletes everything remembered about one of the customer's users, from
// every session and every agent.
func (m *Manager) TruncateMemories(ctx context.Context, customerID, userID string) error {
	if m.options.Memory == nil {
		return stack.Wrap(ErrNoMemory)
	}
	return m.options.Memory.Truncate(ctx, customerID, userID)
}

// ForgetSession deletes what one session learned. Whoever asks has to have checked the
// session is the customer's: the id alone says nothing about whose it is.
func (m *Manager) ForgetSession(ctx context.Context, customerID, sessionID string) error {
	if m.options.Memory == nil {
		return stack.Wrap(ErrNoMemory)
	}
	return m.options.Memory.ForgetRun(ctx, customerID, sessionID)
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

	// The recorders go last so the endings those closes queued are written rather than
	// lost on the way out. The reviews go with it: a summary is worth having, but not
	// worth holding a shutdown open for a model to finish writing.
	m.titles.Close()
	if m.calls != nil {
		m.reviews.Close()
		m.calls.Close()
		m.records.Close()
		m.logs.close()
	}
	// A conversation store the caller passed in outlives this manager.
	if m.conversations != nil && m.options.Conversations == nil {
		m.conversations.Close()
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
		return nil, stack.Wrap(fmt.Errorf("session: there is no sandbox provider called %q", spec.Sandbox))
	}
	return daytona.New(daytona.Options{Config: spec.SandboxOptions, Logger: m.logger})
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

// guardrail builds what screens this session's turns, or nil where the agent declared no
// policy.
//
// A policy that cannot be honoured refuses the session rather than starting one that
// screens nothing. That is the one decision in this feature worth being strict about:
// every other failure here loses a reply, and this one would answer a question the
// customer wrote a file to prevent being answered.
func (m *Manager) guardrail(ctx context.Context, spec Spec, stream streamapp.Identity) (guardrail.Guardrail, error) {
	if strings.TrimSpace(spec.Guardrail) == "" {
		return nil, nil
	}

	// A native model hears the caller and speaks back on its own: nothing here sees the
	// words before they are answered, and there is no reply to hold while a verdict
	// arrives. Refusing is the honest answer - the alternative is a guardrail that is
	// configured, reported, and enforcing nothing.
	if spec.Native() {
		return nil, stack.Wrap(errors.New(
			"session: a speech-to-speech agent answers the caller directly, so a guardrail cannot screen its turns"))
	}

	policy, err := guardrail.Parse(spec.Guardrail)
	if err != nil {
		return nil, err
	}

	return guardrail.New(ctx, policy, guardrail.Deps{
		Owner: routing.Owner{
			CustomerID: spec.CustomerID,
			AgentID:    spec.AgentID,
			CallID:     spec.CallID,
			Tags:       spec.Tags,
		},
		Classifier: m.options.Classifier,
		LLM:        m.options.LLM,
		Secret:     stream.Secret.Reveal(),
		APIKey:     ownKey(spec.CustomerID, stream),
		Logger:     m.logger,
	})
}

// stream is the Stream app a new session acts in. A deployment with no Stream app still
// holds a text conversation, so having none is not an error here: what needs one fails
// where it needs it.
func (m *Manager) stream(ctx context.Context, spec Spec) (streamapp.Bound, error) {
	nowhere := streamapp.Bound{Identity: streamapp.Identity{CustomerID: spec.CustomerID}}
	if m.options.Stream == nil {
		return nowhere, nil
	}
	// A session a hook started acts in the app the hook came from.
	if pin, ok := m.hookPin(spec); ok {
		bound, err := m.options.Stream.ForApp(ctx, spec.CustomerID, pin)
		if err != nil {
			return streamapp.Bound{}, fmt.Errorf("session: the app the hook for this session came from: %w", err)
		}
		return bound, nil
	}
	// A call that already has lines, a number's or a placed call's, is in the app they were
	// made in, and the agent has to join it there.
	if m.options.Store != nil && spec.CallID != "" && !spec.Text {
		pin, found, err := m.options.Store.CallPin(ctx, spec.CustomerID, spec.CallType, spec.CallID)
		if err != nil {
			return streamapp.Bound{}, err
		}
		if found {
			bound, err := m.options.Stream.ForApp(ctx, spec.CustomerID, pin)
			if err != nil {
				return streamapp.Bound{}, fmt.Errorf("session: the app call %s is in: %w", spec.CallID, err)
			}
			return bound, nil
		}
	}
	bound, err := m.options.Stream.For(ctx, spec.CustomerID)
	if errors.Is(err, streamapp.ErrNoIdentity) {
		return nowhere, nil
	}
	if err != nil {
		return streamapp.Bound{}, fmt.Errorf("session: which Stream app to act in: %w", err)
	}
	return bound, nil
}

// hookPinFor is how long a hook's app is remembered for the session it starts.
const hookPinFor = 10 * time.Minute

type hookPinned struct {
	app int64
	at  time.Time
}

// PinHook remembers which app a hook for a customer's call or channel came from, for the
// session a worker opens to answer it, which acts there.
func (m *Manager) PinHook(customer, cid string, app int64) {
	m.hookPinsMu.Lock()
	defer m.hookPinsMu.Unlock()
	now := time.Now()
	for key, pinned := range m.hookPins {
		if now.Sub(pinned.at) > hookPinFor {
			delete(m.hookPins, key)
		}
	}
	m.hookPins[customer+"\x00"+cid] = hookPinned{app: app, at: now}
}

// hookPin is the app a hook for the session's call or channel came from, if one did lately.
func (m *Manager) hookPin(spec Spec) (int64, bool) {
	cid := streamapp.AgentChannelType + ":" + spec.AgentID
	if spec.CallID != "" && !spec.Text {
		cid = spec.CallType + ":" + spec.CallID
	}
	m.hookPinsMu.Lock()
	defer m.hookPinsMu.Unlock()
	pinned, ok := m.hookPins[spec.CustomerID+"\x00"+cid]
	if !ok || time.Since(pinned.at) > hookPinFor {
		return 0, false
	}
	return pinned.app, true
}

// ErrNoStreamApp is a conversation in writing asked to be kept by an app that has no Stream
// app to keep it in.
var ErrNoStreamApp = errors.New("session: this app has no Stream app to keep the conversation in: " +
	"register this app's Stream keys, or open an incognito session")

// ErrConversationReadOnly is a conversation kept in the router's shared Stream app, which its
// customer may read and no longer add to.
var ErrConversationReadOnly = errors.New("session: this conversation is kept in the router's shared " +
	"Stream app and can only be read there: fork it to carry on")

// ErrConversationElsewhere is a call bound to a conversation kept in another Stream app than
// the one the call is made in.
var ErrConversationElsewhere = errors.New("session: this conversation is kept in another Stream app " +
	"than this call is made in: fork it to carry on")

// keepable refuses a conversation in writing that has nowhere safe to be kept. In app mode
// an app that registered no Stream app has nowhere at all, which used to mean a
// conversation quietly not kept; and a registered app whose agent channel type lets a
// client make, change or join a conversation's channel would keep it where anybody could
// rewrite whose it is.
func (m *Manager) keepable(ctx context.Context, spec Spec, stream streamapp.Bound) error {
	if m.options.Stream == nil || !m.options.Stream.PerApp() || !spec.PersistConversation || !spec.Text {
		return nil
	}
	if stream.Client == nil {
		return ErrNoStreamApp
	}
	if !stream.Identity.Registered {
		return nil
	}
	readiness, err := m.options.Stream.Readiness(ctx, stream)
	if err != nil {
		// Stream being out of reach is found out by the conversation itself, which says so.
		m.logger.Warn("could not check the Stream app a conversation is kept in", "customer_id", spec.CustomerID, "error", err)
		return nil
	}
	switch readiness.ChannelType {
	case streamapp.TypeMissing:
		return fmt.Errorf("session: this app's Stream app has no %s channel type to keep the conversation in", streamapp.AgentChannelType)
	case streamapp.TypeUnsafe:
		return fmt.Errorf("session: this app's Stream app lets clients make, change or join %s channels, "+
			"so a conversation kept there could be rewritten by anybody: restrict the channel type's grants", streamapp.AgentChannelType)
	}
	return nil
}

// sameApp refuses a call on a conversation kept in another app than the call is made in:
// the agent would join the call in one app and look for the conversation's channel in it,
// where it is not.
func (m *Manager) sameApp(ctx context.Context, service *persistent.Service, spec Spec) error {
	if m.options.Stream == nil {
		return nil
	}
	kept, err := service.AppOf(ctx, spec.CustomerID, spec.ConversationID)
	if err != nil {
		return err
	}
	deployment := m.options.Stream.DeploymentApp()
	same := func(a, b int64) bool {
		if a == 0 {
			a = deployment
		}
		if b == 0 {
			b = deployment
		}
		return a == b
	}
	if !same(kept, spec.StreamApp) {
		return ErrConversationElsewhere
	}
	return nil
}

// ownKey is the key a guardrail webhook names when the session acts in its customer's own
// app, so the customer can tell which of their keys signed it. A session acting in the
// deployment's shared app names none, and is signed as it always was.
func ownKey(customer string, stream streamapp.Identity) string {
	if stream.StreamApp == 0 || streamapp.CustomerOf(stream.StreamApp) != customer {
		return ""
	}
	return stream.APIKey
}

// line is what the session may do to the call it is on, which is nothing unless it was
// given a number to act from.
func (m *Manager) line(spec Spec) (agent.Telephony, error) {
	if spec.Phone == nil || spec.Phone.Number == "" {
		return nil, nil
	}
	if m.options.Phone == nil {
		return nil, stack.Wrap(errors.New("session: this deployment has no telephony, so a number cannot be used"))
	}

	return m.options.Phone.Line(phone.LineOptions{
		Owner:        routing.Owner{CustomerID: spec.CustomerID, Tags: spec.Tags},
		From:         spec.Phone.Number,
		CallID:       spec.CallID,
		CallType:     spec.CallType,
		StreamApp:    spec.StreamApp,
		Vendor:       spec.Phone.Vendor,
		VendorCallID: spec.Phone.VendorCallID,
	}), nil
}

// ErrSessionExists is a session asked for by an id some session already has.
var ErrSessionExists = errors.New("session: a session with that id already exists")

// ErrNoMemory is memory asked of a deployment with no memory provider.
var ErrNoMemory = errors.New("session: memory is unavailable; configure the backend memory provider")

// newID is a handle for an agent or a turn. It is random rather than sequential because it
// is the only thing standing between two customers who both guessed at an id.
func newID() string {
	raw := make([]byte, 16)
	// rand.Read on crypto/rand never returns an error, which is why the result is not
	// checked: the alternative would be a session that could not be created.
	_, _ = rand.Read(raw)
	return hex.EncodeToString(raw)
}

func (m *Manager) Conversations() (*persistent.Service, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.conversations == nil {
		// A conversation is kept in the Stream app its session acts in, so without a way
		// to say which app that is there is nowhere to keep one.
		if m.options.Stream == nil {
			return nil, stack.Wrap(errors.New("Stream Chat credentials are required for persistent conversations"))
		}
		service := persistent.NewForChats(persistent.StreamApps(m.options.Stream))
		if m.options.Store != nil {
			service.SetPins(m.options.Store.ConversationPin)
		}
		m.conversations = service
	}
	return m.conversations, nil
}

// LoginFinished marks the end user's login with this OAuth state as connected on the
// conversation that asked for it, and has the session holding that conversation carry on
// with what the login was asked for, so nobody has to ask again.
func (m *Manager) LoginFinished(state string) {
	conversations, err := m.Conversations()
	if err != nil {
		return
	}
	conv, pluginID, ok := conversations.Connected(state)
	if !ok {
		return
	}
	name := pluginID
	if plugin, listed := plugins.Lookup(pluginID); listed {
		name = plugin.Name
	}
	m.mu.Lock()
	var held *Session
	for _, s := range m.sessions {
		if s.persisted == conv {
			held = s
		}
	}
	m.mu.Unlock()
	if held == nil {
		return
	}
	text := name + " is connected now. Carry on with what I asked for before you needed it."
	if err := held.FollowUp(context.Background(), text); err != nil {
		m.logger.Warn("could not carry on after a plugin login", "plugin", pluginID, "error", err)
	}
}

// publisher is where files the subagent's code hands back are shown: the conversation's
// channel when the session is kept in one, and nowhere when it is not.
func publisher(conv *persistent.Conversation) sandbox.Publisher {
	if conv == nil {
		return nil
	}
	return conv.Publish
}
