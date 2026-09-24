package stream

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// events is how many frames may be waiting to be read before the reader is holding up the
// socket. A conversation produces a handful per turn, so anybody keeping up never fills it.
const events = 64

// ErrNotOnCall is returned by an action on a pipeline that has not joined anything, or has
// already left.
var ErrNotOnCall = errors.New("stream: the agent is not on a call")

// Config is how a pipeline is set up before it joins anything.
//
// Every target is a provider/model name or a capability shortcut such as "llm-fast";
// leaving one empty takes the backend's default for that modality.
type Config struct {
	// Agent is the name a stored agent config was stored under, which is what a person
	// actually knows the agent as: "docs" rather than an id they never chose. Everything
	// else here overrides what that config says, so a configuration can be reused and one
	// call still changed.
	//
	// The name goes to the router as it is, which resolves it and refuses one that matches
	// nothing. That refusal is the point: a typo would otherwise start a working session
	// with default instructions, which is far harder to notice than an error.
	Agent string
	// ConfigID names a stored config by id instead, for a caller that was handed one rather
	// than choosing a name -- a dispatched message, say, which carries the id of the config
	// it was routed to. Naming both this and Agent is refused, since there is no sensible
	// answer when they disagree.
	ConfigID string
	// LLM is the model that answers.
	LLM string
	// STT is the model that transcribes.
	STT string
	// TTS is the model that speaks.
	TTS string
	// Subagent is the model that does the thinking a harness delegates. Overridden by the
	// harness when it names one.
	Subagent string
	Video    *acceleration.SessionVideo
	// Voice is a provider-specific voice id.
	Voice string
	// Language is a hint, which narrows the candidates in every modality.
	Language string
	// Greeting is said on joining without going through the model. Empty means the agent
	// waits to be spoken to.
	Greeting string
	// Backchannel murmurs while a caller is still talking, the way a person does.
	Backchannel bool
	// MaxTokens is a ceiling on a reply. Zero leaves the backend's default.
	MaxTokens int
	// ToolTimeout is how long the model waits for one of your functions before carrying on
	// without it. Zero leaves the backend's default.
	ToolTimeout time.Duration
	// Backend is where the router is and who is billed. Its zero value reads the
	// environment.
	Backend Backend
	// Functions are the caller's own, which the model is offered and this process runs.
	//
	// For a caller that keeps a registry of its own and hands the same one to several
	// conversations. Nil makes an empty registry, which Pipeline.Functions hands back to
	// register into, and which is how a single pipeline is usually set up.
	Functions *tools.Registry
	// Logger is where the pipeline reports what it could not do. Nil uses the default.
	Logger *slog.Logger
}

// Call is what one conversation is about, as opposed to the agent behind it.
//
// The backend joins a call that already exists, so the id here names one somebody has
// created. The harness, cost and memory fields are rendered from an agent's configuration
// before it joins.
type Call struct {
	PersistConversation bool
	ConversationID      string
	// ID is the call to join. Empty holds the conversation in writing instead.
	ID string
	// Type is the Stream call type. Empty leaves the backend's default.
	Type string
	// UserID is who the agent joins the call as.
	UserID string
	// UserName is the agent's display name in the call.
	UserName string
	// AgentID keys transcripts and statistics. Empty means the call id.
	AgentID string
	// Instructions is the system prompt.
	Instructions string

	// Title and Description are what a person finds this conversation by afterwards. Both
	// are searched.
	Title       string
	Description string
	// Project groups conversations, and is carried as a cost label too.
	Project string
	// Custom is the caller's own labels, handed back untouched and queryable.
	Custom map[string]any
	// Incognito holds the conversation and keeps nothing: no session row, no turns, no
	// transcript whatever PersistConversation says. It cannot be found afterwards, which is
	// the point of it.
	Incognito bool
	// ModelOverwrites changes the models for this conversation alone, over whatever the
	// agent config decided.
	ModelOverwrites *acceleration.ModelOverwrites

	// Tags are cost labels, carried onto every request the session makes.
	Tags map[string]string
	// Memory is who the session's memories are about and what narrows recall.
	Memory *acceleration.SessionMemory

	// Subagent is the model that runs delegated work, from the agent's harness.
	Subagent string
	Video    *acceleration.SessionVideo
	// Tasks is how much delegated work may run at once.
	Tasks int
	// Sandbox is where the subagent may run code it writes.
	Sandbox string
	// Skills replace the built-in set. Nil leaves them alone, and an empty non-nil slice
	// turns delegation off: the two mean different things.
	Skills *[]acceleration.SessionSkill

	// Phone is the number the session acts from, which is what turns transferring on.
	Phone *acceleration.SessionPhone
	// Navigating says the agent placed this call, so let recordings finish and answer their
	// menus.
	Navigating bool
}

// Participant is somebody on the call, as the backend reports them.
type Participant struct {
	ID     string
	UserID string
	Name   string
}

// Event is one thing the conversation did.
//
// Kind is the backend's own name for it: joined, heard, responding, response_delta,
// responded, spoke, turn, delegated, task_settled, task_cancelled, tool_ran, transferred,
// pressed, looked_up, backchannel, interrupted, overlap_decided, conversation_compacted,
// error and left. The fields below are filled from whichever of those carry them, and Frame
// is the whole thing for anything they do not cover.
type Event struct {
	Kind        string
	Text        string
	Participant Participant
	PendingWork bool
	Interrupted bool
	Error       string
	Frame       Frame
}

// Pipeline is a whole voice or text pipeline, running in the acceleration backend.
//
// It does no inference and touches no media. The backend joins the call, hears the caller,
// answers and speaks, and what arrives here are the events saying so. What stays here is
// function calling, since the functions are here, and configuration, since the decisions
// are yours.
type Pipeline struct {
	config    Config
	backend   Backend
	logger    *slog.Logger
	functions *tools.Registry

	mu      sync.Mutex
	session *acceleration.Session
	socket  *Socket
	events  chan Event
	stop    context.CancelFunc
	running sync.WaitGroup
	watcher sync.WaitGroup
}

// Accelerated configures a pipeline to run remotely.
//
//	llm := stream.Accelerated(stream.Config{LLM: "gemma4", STT: "parakeet", TTS: "sonic_36"})
func Accelerated(config Config) *Pipeline {
	logger := config.Logger
	if logger == nil {
		logger = slog.Default()
	}
	functions := config.Functions
	if functions == nil {
		functions = tools.NewRegistry()
	}
	return &Pipeline{
		config:    config,
		backend:   config.Backend,
		logger:    logger,
		functions: functions,
	}
}

// Functions are the caller's own, which the model is offered and this process runs.
func (p *Pipeline) Functions() *tools.Registry {
	return p.functions
}

// Backend is where the router is and who is billed, with the environment already read.
func (p *Pipeline) Backend() (Backend, error) {
	return p.backend.Resolve()
}

// Session is the session the pipeline is holding, or nil when it is not on a call.
func (p *Pipeline) Session() *acceleration.Session {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.session
}

// Join creates the session and starts watching it.
//
// It returns once the backend is in the call, so an agent that has joined is one that is
// already listening.
func (p *Pipeline) Join(ctx context.Context, call Call) (*acceleration.Session, error) {
	if _, err := p.ready(); err != nil {
		return nil, err
	}
	return p.JoinWith(ctx, p.request(call))
}

// JoinWith creates a session from a request spelled out in full and starts watching it.
//
// Where Join renders an agent's configuration into a request, this takes one already
// written. The resource surface in sdks/go/client builds its own, and building it there
// while joining through here means both end up on one socket, one tool runner and one way of
// being closed, rather than two that drift apart.
func (p *Pipeline) JoinWith(
	ctx context.Context,
	request acceleration.CreateSessionRequest,
) (*acceleration.Session, error) {
	client, err := p.ready()
	if err != nil {
		return nil, err
	}

	// The functions registered here are declared unless the caller already declared their
	// own, since the model can only be offered what this process can run.
	if declared := p.tools(); len(declared) > 0 && request.Tools == nil {
		request.Tools = &declared
	}

	created, err := client.CreateSessionWithResponse(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("stream: creating the session: %w", err)
	}
	session, err := sessionOf(created)
	if err != nil {
		return nil, err
	}
	if err := p.Watch(ctx, session); err != nil {
		return nil, err
	}

	if request.CallId == nil || *request.CallId == "" {
		p.logger.Info("opened a text session", "session", session.Id)
	} else {
		p.logger.Info("joined a call remotely", "call", *request.CallId, "session", session.Id)
	}
	return session, nil
}

// Watch starts watching a session the router has already created.
//
// Separate from JoinWith because a fork is created by a different request and is otherwise
// the same thing afterwards: the same socket, the same functions, the same Leave.
func (p *Pipeline) Watch(ctx context.Context, session *acceleration.Session) error {
	backend, err := p.backend.Resolve()
	if err != nil {
		return err
	}
	credentials, err := backend.Credentials()
	if err != nil {
		p.abandon(ctx, session.Id)
		return err
	}

	socket := NewSocket(
		backend.SocketURL("/v1/agents/sessions/"+session.Id+"/events"),
		credentials,
		backend.HTTPClient,
		p.logger,
	)
	if err := socket.Open(ctx); err != nil {
		// The session is live in the backend even though nothing here can watch it, so it
		// is closed rather than left holding a call nobody is listening to.
		p.abandon(ctx, session.Id)
		return err
	}

	watching, stop := context.WithCancel(context.WithoutCancel(ctx))
	watched := make(chan Event, events)

	p.mu.Lock()
	if p.session != nil {
		p.mu.Unlock()
		stop()
		_ = socket.Close()
		return errors.New("stream: the agent is already on a call")
	}
	p.session = session
	p.socket = socket
	p.events = watched
	p.stop = stop
	p.mu.Unlock()

	p.watcher.Add(1)
	go p.watch(watching, socket, watched)
	return nil
}

// ready refuses a pipeline that is already holding a conversation, and returns the client
// for the one it is about to join.
func (p *Pipeline) ready() (*acceleration.ClientWithResponses, error) {
	p.mu.Lock()
	held := p.session
	p.mu.Unlock()
	if held != nil {
		return nil, errors.New("stream: the agent is already on a call")
	}

	backend, err := p.backend.Resolve()
	if err != nil {
		return nil, err
	}
	return backend.Client()
}

// abandon closes a session nothing here can watch. Whatever went wrong on the way to
// watching it has already been reported, so a second failure here is nothing to add.
func (p *Pipeline) abandon(ctx context.Context, id string) {
	backend, err := p.backend.Resolve()
	if err != nil {
		return
	}
	client, err := backend.Client()
	if err != nil {
		return
	}
	_, _ = client.CloseSessionWithResponse(ctx, id)
}

// Events yields what the backend did until the call ends, when the channel closes.
//
// Nil before the pipeline has joined anything.
func (p *Pipeline) Events() <-chan Event {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.events
}

// Say speaks text on the call without going through the model.
func (p *Pipeline) Say(text string, interrupt bool) error {
	if interrupt {
		if err := p.command(Frame{"type": "interrupt"}); err != nil {
			return err
		}
	}
	return p.command(Frame{"type": "say", "text": text})
}

// Respond answers text through the model, as though it had been said on the call.
func (p *Pipeline) Respond(text string, interrupt bool) error {
	if interrupt {
		if err := p.command(Frame{"type": "interrupt"}); err != nil {
			return err
		}
	}
	return p.command(Frame{"type": "respond", "text": text})
}

// Interrupt abandons the reply being spoken.
func (p *Pipeline) Interrupt() error {
	return p.command(Frame{"type": "interrupt"})
}

// SetInstructions changes what the agent is told to be, from the next turn.
func (p *Pipeline) SetInstructions(instructions string) error {
	return p.command(Frame{"type": "instructions", "instructions": instructions})
}

// Leave ends the call. Safe to call after it has already ended.
func (p *Pipeline) Leave(ctx context.Context) error {
	p.mu.Lock()
	session := p.session
	socket := p.socket
	p.session = nil
	p.mu.Unlock()

	if session == nil {
		return nil
	}

	var failure error
	if socket != nil && socket.IsOpen() {
		failure = socket.Send(Frame{"type": "close"})
	} else if backend, err := p.backend.Resolve(); err == nil {
		if client, err := backend.Client(); err == nil {
			_, failure = client.CloseSessionWithResponse(ctx, session.Id)
		}
	}

	p.stopWatching()
	return failure
}

// request renders the agent's configuration as a session to create.
func (p *Pipeline) request(call Call) acceleration.CreateSessionRequest {
	request := acceleration.CreateSessionRequest{
		ConversationId: &call.ConversationID,
		Backchannel:    &p.config.Backchannel,
	}
	if call.ID == "" {
		text := true
		request.Text = &text
	} else {
		request.CallId = &call.ID
	}

	// An incognito conversation writes no transcript by definition, so asking for one is a
	// contradiction the router refuses rather than quietly honours. Dropped here so a caller
	// that set both gets the conversation they asked for rather than a 400.
	if call.Incognito {
		request.Incognito = &call.Incognito
	} else {
		request.PersistConversation = &call.PersistConversation
	}
	if len(call.Custom) > 0 {
		custom := call.Custom
		request.Custom = &custom
	}
	request.ModelOverwrites = call.ModelOverwrites

	setString(&request.CallType, call.Type)
	setString(&request.UserId, call.UserID)
	setString(&request.UserName, call.UserName)
	setString(&request.AgentId, call.AgentID)
	setString(&request.Instructions, call.Instructions)
	setString(&request.Title, call.Title)
	setString(&request.Description, call.Description)
	setString(&request.Project, call.Project)
	setString(&request.Agent, p.config.Agent)
	setString(&request.ConfigId, p.config.ConfigID)
	setString(&request.Llm, p.config.LLM)
	setString(&request.Stt, p.config.STT)
	setString(&request.Tts, p.config.TTS)
	setString(&request.Voice, p.config.Voice)
	setString(&request.Greeting, p.config.Greeting)

	if p.config.Language != "" {
		request.Languages = &[]string{p.config.Language}
	}
	if p.config.MaxTokens > 0 {
		request.MaxTokens = &p.config.MaxTokens
	}
	if p.config.ToolTimeout > 0 {
		milliseconds := int(p.config.ToolTimeout / time.Millisecond)
		request.ToolTimeoutMs = &milliseconds
	}

	// The harness names the subagent when it has one, and the pipeline's own is the
	// fallback for an agent configured without a harness.
	subagent := call.Subagent
	if subagent == "" {
		subagent = p.config.Subagent
	}
	setString(&request.Subagent, subagent)
	request.Video = p.config.Video
	if call.Video != nil {
		request.Video = call.Video
	}

	if call.Tasks > 0 {
		request.Tasks = &call.Tasks
	}
	if call.Sandbox != "" {
		sandbox := acceleration.Sandbox(call.Sandbox)
		request.Sandbox = &sandbox
	}
	if call.Skills != nil {
		request.Skills = call.Skills
	}
	if len(call.Tags) > 0 {
		request.Tags = &call.Tags
	}
	if call.Memory != nil {
		request.Memory = call.Memory
	}
	if call.Phone != nil {
		request.Phone = call.Phone
	}
	if call.Navigating {
		request.Navigating = &call.Navigating
	}

	if declared := p.tools(); len(declared) > 0 {
		request.Tools = &declared
	}
	return request
}

// tools are the functions registered here, as the model will be offered them.
func (p *Pipeline) tools() []acceleration.SessionTool {
	registered := p.functions.List()
	declared := make([]acceleration.SessionTool, 0, len(registered))
	for _, function := range registered {
		tool := acceleration.SessionTool{Name: function.Name, Description: function.Description}
		if function.Parameters != nil {
			parameters := function.Parameters
			tool.Parameters = &parameters
		}
		declared = append(declared, tool)
	}
	return declared
}

// command acts on the session over the socket it is being watched on.
func (p *Pipeline) command(frame Frame) error {
	p.mu.Lock()
	socket := p.socket
	p.mu.Unlock()

	if socket == nil || !socket.IsOpen() {
		return ErrNotOnCall
	}
	return socket.Send(frame)
}

// watch reads the session's socket until it ends, translating as it goes.
//
// The socket and the channel are handed in rather than read back off the pipeline, because
// leaving clears both and a call can be left before this has run at all.
func (p *Pipeline) watch(ctx context.Context, socket *Socket, out chan<- Event) {
	defer p.watcher.Done()
	defer close(out)

	for {
		frame, _, err := socket.Read()
		if err != nil {
			if ctx.Err() == nil && !errors.Is(err, ErrSocketClosed) {
				p.logger.Debug("the session socket ended", "error", err)
			}
			return
		}
		if frame == nil {
			continue
		}

		if frame.Type() == "tool_call" {
			p.running.Add(1)
			go func() {
				defer p.running.Done()
				p.runTool(ctx, frame)
			}()
			continue
		}

		select {
		case out <- eventOf(frame):
		case <-ctx.Done():
			return
		}
	}
}

// runTool runs one of the caller's functions and answers the model with what it said.
//
// A failure is reported rather than dropped: the model is mid-sentence waiting for this,
// and it can say something useful about a tool that did not work only if it is told that it
// did not work.
func (p *Pipeline) runTool(ctx context.Context, frame Frame) {
	name := frame.String("name")
	result := Frame{"type": "tool_result", "tool_call_id": frame.String("id")}
	// A durable command's result is only accepted back with the command and turn it names.
	for _, key := range []string{"command_id", "turn_id"} {
		if value := frame.String(key); value != "" {
			result[key] = value
		}
	}

	output, err := p.functions.Call(ctx, name, frame.String("arguments"))
	if err != nil {
		p.logger.Error("the tool failed", "tool", name, "error", err)
		result["error"] = err.Error()
	} else {
		result["output"] = output
	}

	p.mu.Lock()
	socket := p.socket
	p.mu.Unlock()
	if socket == nil || !socket.IsOpen() {
		return
	}
	if err := socket.Send(result); err != nil {
		p.logger.Error("the tool result never reached the model", "tool", name, "error", err)
	}
}

// stopWatching drops the socket and everything reading it.
func (p *Pipeline) stopWatching() {
	p.mu.Lock()
	stop := p.stop
	socket := p.socket
	p.socket = nil
	p.stop = nil
	p.mu.Unlock()

	if stop != nil {
		stop()
	}
	if socket != nil {
		_ = socket.Close()
	}
	p.watcher.Wait()
	p.running.Wait()
}

// eventOf fills in the fields the frames that carry them have in common.
func eventOf(frame Frame) Event {
	event := Event{
		Kind:        frame.Type(),
		Text:        frame.String("text"),
		Interrupted: frame.Bool("interrupted"),
		PendingWork: frame.Bool("pending_work"),
		Error:       frame.String("error"),
		Frame:       frame,
	}
	if participant := frame.Frame("participant"); participant != nil {
		event.Participant = Participant{
			ID:     participant.String("id"),
			UserID: participant.String("user_id"),
			Name:   participant.String("name"),
		}
	}
	return event
}

// sessionOf unwraps a create response, raising what the router said went wrong instead.
func sessionOf(response *acceleration.CreateSessionResponse) (*acceleration.Session, error) {
	if response.JSON201 != nil {
		return response.JSON201, nil
	}
	for _, failure := range []*acceleration.Error{response.JSON400, response.JSON401, response.JSON404} {
		if failure != nil {
			return nil, fmt.Errorf("stream: %s", failure.Error)
		}
	}
	return nil, fmt.Errorf("stream: the router answered %s rather than with a session", response.Status())
}

func setString(field **string, value string) {
	if value != "" {
		*field = &value
	}
}
