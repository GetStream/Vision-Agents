package stream

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// events is how many frames may be waiting to be read before the reader is holding up the
// socket. A conversation produces a handful per turn, so anybody keeping up never fills it.
const events = 64

// hangupWithin bounds how long Leave waits for the backend to end the session it was told to
// close. The backend closes the socket once the session has ended, well inside this.
const hangupWithin = 5 * time.Second

// resumeFor is how long a dropped text session's socket is reconnected for. It is how long
// the router keeps a text session nobody watches, after which a reconnect still reopens the
// conversation but has nothing left to send again.
const resumeFor = 5 * time.Minute

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
	TTS   string
	Video *acceleration.SessionVideo
	// Voice is a provider's own voice id, or one of your voices by id or by name. Prefix it
	// with custom: to mean only yours: a name that is not one of yours is then refused
	// instead of passed to the provider's library. A native (sts) agent takes the vendor's
	// own voice name instead.
	Voice string
	// Language is a hint, which narrows the candidates in every modality.
	Language string
	// Greeting is said on joining without going through the model. Empty means the agent
	// waits to be spoken to.
	Greeting string
	// GreetingMode is exact, the default, or variation, which has the model say its own
	// variation of Greeting on every call.
	GreetingMode acceleration.GreetingMode
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
// created. The cost and memory fields are rendered from an agent's configuration
// before it joins.
type Call struct {
	// SessionID is the UUID to hold the session by. Empty lets the router choose one.
	SessionID string
	// Voice has the agent join the session's call: CallType and CallID, else its own,
	// agent:<session id>. False holds the conversation in writing instead.
	Voice bool
	// CallType and CallID name an existing Stream call for the agent to join, such as the
	// default call the people are already on. Both empty is agent:<session id>.
	CallType string
	CallID   string
	// UserID is the name the agent joins the call as. It is not the owner of the session:
	// the owner is the caller, from its auth.
	UserID string
	// UserName is the agent's display name in the call.
	UserName string
	// AgentID keys transcripts and statistics. Empty means the session id.
	AgentID string

	// Title and Description are what a person finds this conversation by afterwards. Both
	// are searched.
	Title       string
	Description string
	// ProjectID groups conversations, and is carried as a cost label too.
	ProjectID string
	// Custom is the caller's own labels, handed back untouched and queryable.
	Custom map[string]any
	// Incognito holds the conversation and keeps nothing: no session row, no turns, no
	// transcript. It cannot be found afterwards, which is the point of it.
	Incognito bool
	// ModelOverwrites changes the models for this conversation alone, over whatever the
	// agent config decided.
	ModelOverwrites *acceleration.ModelOverwrites
	// Timezone is the end user's IANA timezone, such as Europe/Amsterdam, which sets
	// the zone the agent reads dates in and is remembered as the user's.
	Timezone string

	// Tags are cost labels, carried onto every request the session makes.
	Tags map[string]string
	// Memory is who the session's memories are about and what narrows recall.
	Memory *acceleration.SessionMemory

	Video *acceleration.SessionVideo

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
	// Files are what a task_settled's code handed back, uploaded where the person can see
	// them.
	Files []Attachment
	// Resumed is set on a resumed event, which a text session's socket is sent after it
	// dropped and reconnected.
	Resumed *Resumption
	Frame   Frame
}

// Resumption is what a socket that reconnected was told.
//
// Complete is false when some of what the session said while the socket was away is gone:
// the router restarted, or it was away too long. What was said is still in the
// conversation, to read back. Command is the receipt of the last question asked through
// Responses, when the router knows it: an interrupted one was stopped by the router going
// away and is never answered, so ask it again.
type Resumption struct {
	Complete bool
	Command  *acceleration.CommandReceipt
}

// Attachment is something the agent's code made, at a URL the person can reach.
type Attachment struct {
	Name     string `json:"name"`
	MIMEType string `json:"mime_type"`
	URL      string `json:"url"`
	Size     int    `json:"size"`
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
	holds   holds

	// awaiting is the request id of the last question asked, which a reconnect asks after.
	awaiting atomic.Value
	// firstRetry is the first wait before reconnecting a dropped text session.
	firstRetry time.Duration
}

// String names the pipeline by its backend, which leaves the credentials out. Without it fmt
// prints the unexported backend field by reflection, secret and all.
func (p *Pipeline) String() string { return "stream.Pipeline{" + p.backend.String() + "}" }

// GoString is String, so %#v leaves the credentials out too.
func (p *Pipeline) GoString() string { return p.String() }

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
		config:     config,
		backend:    config.Backend,
		logger:     logger,
		functions:  functions,
		firstRetry: firstRetry,
	}
}

// Awaiting names the question a text session is waiting on the answer to, by the request
// id it was asked with, so a socket that reconnects is told how that question ended.
func (p *Pipeline) Awaiting(requestID string) { p.awaiting.Store(requestID) }

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

// sessionID is the id of the session the pipeline is holding, or empty.
func (p *Pipeline) sessionID() string {
	if session := p.Session(); session != nil {
		return session.Id
	}
	return ""
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

	if session.CallId == "" {
		p.logger.Info("opened a text session", "session", session.Id)
	} else {
		p.logger.Info("joined a call remotely", "call", session.CallId, "session", session.Id)
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

	// A text session outlives its socket, so its socket is resumable and is reconnected
	// when it drops. A call's is not: the call is the router's, and ends with it.
	resumable := session.CallId == ""
	asked := p.runs()
	if resumable {
		asked.Set("resumable", "true")
	}
	socket := p.socketFor(backend, credentials, session.Id, asked)
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
	go p.watch(watching, socket, watched, resumable)
	return nil
}

func (p *Pipeline) socketFor(backend Backend, credentials http.Header, id string, asked url.Values) *Socket {
	address := backend.SocketURL("/v1/agents/sessions/" + id + "/events")
	if len(asked) > 0 {
		address += "?" + asked.Encode()
	}
	return NewSocket(address, credentials, backend.HTTPClient, p.logger)
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

// abandon stops a session nothing here can watch. Whatever went wrong on the way to
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
	_, _ = client.StopSessionWithResponse(ctx, id)
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

// Interrupt abandons the reply being spoken.
func (p *Pipeline) Interrupt() error {
	return p.command(Frame{"type": "interrupt"})
}

// AnswerApproval answers a tool call whose step waits for a person to allow it (an
// ai_tool_call step awaiting_approval): the step's id, the request_id of the command its reply
// answers, whether they allow it and, declined, what the step says. A connector call the router
// holds for approval (a binding's require_approval) is sent only once allowed.
func (p *Pipeline) AnswerApproval(toolCallID, requestID string, allowed bool, summary string) error {
	frame := Frame{"type": "tool_approval", "tool_call_id": toolCallID, "request_id": requestID, "allowed": allowed}
	if summary != "" {
		frame["summary"] = summary
	}
	return p.command(frame)
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
		if failure = socket.Send(Frame{"type": "close"}); failure == nil {
			p.awaitHangup(ctx)
		}
	} else if backend, err := p.backend.Resolve(); err == nil {
		if client, err := backend.Client(); err == nil {
			_, failure = client.StopSessionWithResponse(ctx, session.Id)
		}
	}

	p.stopWatching()
	return failure
}

// request renders the agent's configuration as a session to create.
func (p *Pipeline) request(call Call) acceleration.CreateSessionRequest {
	request := acceleration.CreateSessionRequest{
		Backchannel: &p.config.Backchannel,
	}
	if call.Voice {
		request.StartVoice = &call.Voice
	}
	if call.CallID != "" {
		request.Call = &acceleration.SessionCall{Type: call.CallType, Id: call.CallID}
	}

	if call.Incognito {
		request.Incognito = &call.Incognito
	}
	if len(call.Custom) > 0 {
		custom := call.Custom
		request.Custom = &custom
	}
	request.ModelOverwrites = call.ModelOverwrites

	setString(&request.Id, call.SessionID)
	setString(&request.UserId, call.UserID)
	setString(&request.UserName, call.UserName)
	setString(&request.AgentId, call.AgentID)
	setString(&request.Title, call.Title)
	setString(&request.Description, call.Description)
	setString(&request.Timezone, call.Timezone)
	setString(&request.ProjectId, call.ProjectID)
	setString(&request.Agent, p.config.Agent)
	setString(&request.ConfigId, p.config.ConfigID)
	setString(&request.Llm, p.config.LLM)
	setString(&request.Stt, p.config.STT)
	setString(&request.Tts, p.config.TTS)
	setString(&request.Voice, p.config.Voice)
	if p.config.Greeting != "" {
		request.Greeting = &acceleration.Greeting{Text: p.config.Greeting}
		if p.config.GreetingMode != "" {
			request.Greeting.Mode = &p.config.GreetingMode
		}
	}

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

	request.Video = p.config.Video
	if call.Video != nil {
		request.Video = call.Video
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

// runs is the socket's query saying which of the session's tools this process runs: the
// functions registered here, and none when there are none.
func (p *Pipeline) runs() url.Values {
	var names []string
	for _, function := range p.functions.List() {
		names = append(names, function.Name)
	}
	return url.Values{"tools": {strings.Join(names, ",")}}
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
		if function.Interruptible {
			interruptible := true
			tool.Interruptible = &interruptible
		}
		if function.Timeout > 0 {
			timeout := function.Timeout.Milliseconds()
			tool.TimeoutMs = &timeout
		}
		if function.Approval != "" {
			tool.Approval = &acceleration.SessionToolApproval{Title: function.Approval}
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
// leaving clears both and a call can be left before this has run at all. A resumable
// socket that drops without the session ending is opened again after the last frame read,
// and the channel stays open across it.
func (p *Pipeline) watch(ctx context.Context, socket *Socket, out chan<- Event, resumable bool) {
	defer p.watcher.Done()
	defer close(out)

	cursor := ""
	for reconnected := false; ; reconnected = true {
		err := p.read(ctx, socket, out, &cursor, reconnected)
		if !resumable || !dropped(err) || ctx.Err() != nil || p.Session() == nil {
			if err != nil && ctx.Err() == nil && !errors.Is(err, ErrSocketClosed) {
				p.logger.Debug("the session socket ended", "error", err)
			}
			return
		}
		p.logger.Info("lost the session socket, reconnecting", "error", err)
		_ = socket.Close()
		if socket = p.reconnect(ctx, cursor); socket == nil {
			return
		}
	}
}

// read reads one socket until it ends, keeping cursor at the last frame read. A reconnected
// socket's resumed frame is handed on, where a first one only says where the stream starts.
func (p *Pipeline) read(ctx context.Context, socket *Socket, out chan<- Event, cursor *string, reconnected bool) error {
	// A read blocks in the socket rather than on a channel, so a pipeline told to stop
	// reaches it by closing the connection underneath it.
	stopped := make(chan struct{})
	defer close(stopped)
	go func() {
		select {
		case <-ctx.Done():
			_ = socket.Close()
		case <-stopped:
		}
	}()

	for {
		frame, _, err := socket.Read()
		if err != nil {
			return err
		}
		if frame == nil {
			continue
		}
		if at := frame.String("cursor"); at != "" {
			*cursor = at
		}

		switch frame.Type() {
		case "tool_call":
			// Every watcher is sent every call, and the router says which are to run it.
			// One from before it said holds the same rule: answering a tool held elsewhere
			// would beat the process that runs it. A call sent again after a reconnect is
			// fenced by the registry, which runs it once.
			if frame["run"] == false || !p.functions.Holds(frame.String("name")) {
				p.logger.Debug("leaving a call for a tool held elsewhere", "tool", frame.String("name"))
				continue
			}
			p.running.Add(1)
			go func() {
				defer p.running.Done()
				p.runTool(ctx, frame)
			}()
			continue
		case "tool_cancel":
			p.holds.answer(frame.String("id"), false)
			p.functions.Cancel(p.sessionID(), frame.String("id"))
		case "tool_held":
			p.holds.answer(frame.String("id"), true)
		case "resumed":
			if !reconnected {
				continue
			}
		}

		select {
		case out <- eventOf(frame):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// dropped says whether a socket ended without the session ending: anything but the router
// closing it normally, which is how it says the session is over, or this side closing it.
func dropped(err error) bool {
	return err != nil && !errors.Is(err, ErrSocketClosed) && !websocket.IsCloseError(err, websocket.CloseNormalClosure)
}

// reconnect opens the session's socket again after cursor, waiting longer between each
// attempt, until it opens, the pipeline stops, the router refuses it for good, or resumeFor
// runs out. It returns nil when it gave up.
func (p *Pipeline) reconnect(ctx context.Context, cursor string) *Socket {
	session := p.Session()
	if session == nil {
		return nil
	}
	asked := p.runs()
	asked.Set("resumable", "true")
	if cursor != "" {
		asked.Set("cursor", cursor)
	}
	if id, _ := p.awaiting.Load().(string); id != "" {
		asked.Set("request_id", id)
	}

	giveUp := time.Now().Add(resumeFor)
	for retry := p.firstRetry; ; retry = min(retry*2, lastRetry) {
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(retry):
		}
		socket, err := p.open(ctx, session.Id, asked)
		if err == nil {
			p.mu.Lock()
			defer p.mu.Unlock()
			// Left while this was dialling: the new socket is nobody's.
			if ctx.Err() != nil || p.session == nil {
				_ = socket.Close()
				return nil
			}
			p.socket = socket
			return socket
		}
		var refused *RouterError
		// A refusal that says who the caller is or what the session is will say it again.
		if errors.As(err, &refused) && refused.Status >= 400 && refused.Status < 500 && refused.Status != 429 {
			p.logger.Warn("the router refused the session socket, not reconnecting", "error", err)
			return nil
		}
		if time.Now().After(giveUp) {
			p.logger.Warn("could not reconnect the session socket", "error", err, "for", resumeFor)
			return nil
		}
		p.logger.Debug("could not reconnect the session socket", "error", err, "retrying in", retry)
	}
}

// open dials the session's socket, with credentials minted for it: a token signed when the
// session opened may have expired by the time it reconnects.
func (p *Pipeline) open(ctx context.Context, id string, asked url.Values) (*Socket, error) {
	backend, err := p.backend.Resolve()
	if err != nil {
		return nil, err
	}
	credentials, err := backend.Credentials()
	if err != nil {
		return nil, err
	}
	socket := p.socketFor(backend, credentials, id, asked)
	if err := socket.Open(ctx); err != nil {
		return nil, err
	}
	return socket, nil
}

// runTool runs one of the caller's functions and answers the model with what it said.
//
// A failure is reported rather than dropped: the model is mid-sentence waiting for this,
// and it can say something useful about a tool that did not work only if it is told that it
// did not work.
func (p *Pipeline) runTool(ctx context.Context, frame Frame) {
	name := frame.String("name")
	result := Frame{"type": "tool_result", "tool_call_id": frame.String("id")}
	// A durable request's result is only accepted back with the request and turn it names.
	for _, key := range []string{"request_id", "turn_id"} {
		if value := frame.String(key); value != "" {
			result[key] = value
		}
	}

	line := tools.WithLine(ctx, toolLine{send: p.command, holds: &p.holds})
	output, err := p.functions.Run(line, toolCallContextOf(frame, p.sessionID()), name, frame.String("arguments"))
	if err != nil {
		p.logger.Error("the tool failed", "tool", name, "error", err)
		result["error"] = err.Error()
	} else {
		result["output"] = output.Wire()
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

// toolCallContextOf is the call a tool_call frame asks for, as tools.ToolCallContextFrom
// gives it to the function.
// sessionID stands in for a frame that does not name its session, as a router's from before
// session_id did not.
func toolCallContextOf(frame Frame, sessionID string) tools.Call {
	call := tools.Call{ID: frame.String("id"), SessionID: frame.String("session_id"), TurnID: frame.String("turn_id"), Approved: frame.Bool("approved")}
	if call.SessionID == "" {
		call.SessionID = sessionID
	}
	if user := frame.Frame("user"); user != nil {
		call.User = tools.User{ID: user.String("id"), Name: user.String("name")}
		call.User.Custom, _ = user["custom"].(map[string]any)
	}
	if conversation := frame.Frame("conversation"); conversation != nil {
		call.Conversation = tools.Conversation{CID: conversation.String("cid")}
		call.Conversation.Custom, _ = conversation["custom"].(map[string]any)
	}
	return call
}

// stopWatching drops the socket and everything reading it.
// awaitHangup waits for the backend to close the socket after the close frame, or for
// hangupWithin, whichever is first.
//
// Closing the connection straight after writing the frame can lose it: a proxy between the
// two ends may reset its side before forwarding what was still in flight, leaving the
// session running with nobody on it.
func (p *Pipeline) awaitHangup(ctx context.Context) {
	closed := make(chan struct{})
	go func() {
		p.watcher.Wait()
		close(closed)
	}()
	waiting := time.NewTimer(hangupWithin)
	defer waiting.Stop()
	select {
	case <-closed:
	case <-waiting.C:
	case <-ctx.Done():
	}
}

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
	files, _ := frame["files"].([]any)
	for _, listed := range files {
		file, ok := listed.(map[string]any)
		if !ok {
			continue
		}
		event.Files = append(event.Files, Attachment{
			Name: Frame(file).String("name"), MIMEType: Frame(file).String("mime_type"),
			URL: Frame(file).String("url"), Size: Frame(file).Int("size"),
		})
	}
	if event.Kind == "resumed" {
		event.Resumed = &Resumption{Complete: frame.Bool("complete")}
		if command := frame.Frame("command"); command != nil {
			event.Resumed.Command = &acceleration.CommandReceipt{
				RequestId: command.String("request_id"), State: command.String("state"),
				UserMessageId: command.String("user_message_id"), AssistantMessageId: command.String("assistant_message_id"),
				Duplicate: command.Bool("duplicate"),
			}
		}
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
	return nil, NewRouterError(response.HTTPResponse, response.Body, "stream", "",
		"the router answered "+response.Status()+" rather than with a session")
}

func setString(field **string, value string) {
	if value != "" {
		*field = &value
	}
}
