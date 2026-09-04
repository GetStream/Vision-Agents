package stream

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/agents-core-go/acceleration"
	"github.com/GetStream/Vision-Agents/agents-core-go/tools"
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
	// Agent is a stored agent config to start from, named either by its id or by the name
	// it was stored under. Everything else here overrides what it says, so a configuration
	// can be reused and one call still changed.
	Agent string
	// LLM is the model that answers.
	LLM string
	// STT is the model that transcribes.
	STT string
	// TTS is the model that speaks.
	TTS string
	// Subagent is the model that does the thinking a harness delegates. Overridden by the
	// harness when it names one.
	Subagent string
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
	// Logger is where the pipeline reports what it could not do. Nil uses the default.
	Logger *slog.Logger
}

// Call is what one conversation is about, as opposed to the agent behind it.
//
// The backend joins a call that already exists, so the id here names one somebody has
// created. The harness, cost and memory fields are rendered from an agent's configuration
// before it joins.
type Call struct {
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

	// Tags are cost labels, carried onto every request the session makes.
	Tags map[string]string
	// Memory is who the session's memories are about and what narrows recall.
	Memory *acceleration.SessionMemory

	// Subagent is the model that runs delegated work, from the agent's harness.
	Subagent string
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

	// configured is Config.Agent resolved to the id the backend wants, looked up once.
	configured string
}

// Accelerated configures a pipeline to run remotely.
//
//	llm := stream.Accelerated(stream.Config{LLM: "gemma4", STT: "parakeet", TTS: "sonic_36"})
func Accelerated(config Config) *Pipeline {
	logger := config.Logger
	if logger == nil {
		logger = slog.Default()
	}
	return &Pipeline{
		config:    config,
		backend:   config.Backend,
		logger:    logger,
		functions: tools.NewRegistry(),
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
	p.mu.Lock()
	if p.session != nil {
		p.mu.Unlock()
		return nil, errors.New("stream: the agent is already on a call")
	}
	p.mu.Unlock()

	backend, err := p.backend.Resolve()
	if err != nil {
		return nil, err
	}
	client, err := backend.Client()
	if err != nil {
		return nil, err
	}

	config, err := p.configID(ctx, client)
	if err != nil {
		return nil, err
	}

	created, err := client.CreateSessionWithResponse(ctx, p.request(call, config))
	if err != nil {
		return nil, fmt.Errorf("stream: creating the session: %w", err)
	}
	session, err := sessionOf(created)
	if err != nil {
		return nil, err
	}

	socket := NewSocket(
		backend.SocketURL("/v1/agents/sessions/"+session.Id+"/events"),
		backend.Headers(),
		backend.HTTPClient,
		p.logger,
	)
	if err := socket.Open(ctx); err != nil {
		// The session is live in the backend even though nothing here can watch it, so it
		// is closed rather than left holding a call nobody is listening to.
		_, _ = client.CloseSessionWithResponse(ctx, session.Id)
		return nil, err
	}

	watching, stop := context.WithCancel(context.WithoutCancel(ctx))
	watched := make(chan Event, events)

	p.mu.Lock()
	p.session = session
	p.socket = socket
	p.events = watched
	p.stop = stop
	p.mu.Unlock()

	p.watcher.Add(1)
	go p.watch(watching, socket, watched)

	p.logger.Info("joined a call remotely", "call", call.ID, "session", session.Id)
	return session, nil
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

// configID turns Config.Agent into the id the backend looks a config up by.
//
// The backend takes an id, but an agent is worth naming, so a name is resolved to one here
// and remembered. A name that matches nothing stored is passed through untouched: it is
// then either an id, or a mistake the backend is better placed to report than a guess here
// would be.
func (p *Pipeline) configID(ctx context.Context, client *acceleration.ClientWithResponses) (string, error) {
	if p.config.Agent == "" {
		return "", nil
	}

	p.mu.Lock()
	resolved := p.configured
	p.mu.Unlock()
	if resolved != "" {
		return resolved, nil
	}

	listed, err := client.ListAgentConfigsWithResponse(ctx)
	if err != nil {
		return "", fmt.Errorf("stream: looking up the agent config: %w", err)
	}

	resolved = p.config.Agent
	if listed.JSON200 != nil {
		for _, stored := range *listed.JSON200 {
			if stored.Name == p.config.Agent {
				resolved = stored.Id
				break
			}
		}
	}

	p.mu.Lock()
	p.configured = resolved
	p.mu.Unlock()
	return resolved, nil
}

// request renders the agent's configuration as a session to create.
func (p *Pipeline) request(call Call, config string) acceleration.CreateSessionRequest {
	request := acceleration.CreateSessionRequest{
		Backchannel: &p.config.Backchannel,
	}
	if call.ID == "" {
		text := true
		request.Text = &text
	} else {
		request.CallId = &call.ID
	}

	setString(&request.CallType, call.Type)
	setString(&request.UserId, call.UserID)
	setString(&request.UserName, call.UserName)
	setString(&request.AgentId, call.AgentID)
	setString(&request.Instructions, call.Instructions)
	setString(&request.ConfigId, config)
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

	if call.Tasks > 0 {
		request.Tasks = &call.Tasks
	}
	if call.Sandbox != "" {
		sandbox := acceleration.CreateSessionRequestSandbox(call.Sandbox)
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
