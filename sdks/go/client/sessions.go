package client

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// SessionOptions is what a conversation is opened with, beyond which agent holds it.
//
// Everything here is the caller's own decision about this one conversation: what to call it
// so they can find it again, which models to overrule, and whether to keep it at all.
type SessionOptions struct {
	// Title and Description are what a person finds the conversation by later. Both are
	// searched.
	Title       string
	Description string
	// Project groups conversations, and is carried as a cost label too.
	Project string
	// Custom is the caller's own labels, which a query can match on.
	Custom map[string]any
	// Incognito holds the conversation and keeps nothing: no row, no turns, no transcript.
	// It cannot be searched for, listed or forked afterwards, which is the point of it.
	Incognito bool
	// ModelOverwrites changes the models for this conversation alone.
	ModelOverwrites *acceleration.ModelOverwrites

	// CallID is the call to join. Empty holds the conversation in writing.
	CallID string
	// CallType is the Stream call type. Empty leaves the backend's default.
	CallType string
	// Persist keeps what was said in Stream Chat, so it outlives the session. What a
	// conversation somebody comes back to wants, which is most of them.
	Persist bool
	// ConversationID resumes the channel an earlier session was held in.
	ConversationID string
	// Instructions overrides the agent's own system prompt for this conversation.
	Instructions string
	// UserID is who the conversation belongs to, for a backend opening one on somebody's
	// behalf. A client acting for a user leaves it empty: the token already says who.
	UserID string

	// Functions are the caller's own, for this conversation only. Nil uses the agent's,
	// which is the usual arrangement.
	Functions *tools.Registry
	// Logger is where the session reports what it could not do. Nil uses the default.
	Logger *slog.Logger
}

// Query is which of an agent's conversations to list.
type Query struct {
	// Project narrows to one project's.
	Project string
	// UserID narrows to one user's, which only a server-side caller may ask for: anybody
	// else is narrowed to their own whatever they send.
	UserID string
	// State is "running" or "closed". Empty is both.
	State string
	// Custom are labels a conversation must carry, all of them.
	Custom map[string]any
	// After and Before bound when the conversation was opened. Zero is unbounded.
	After  time.Time
	Before time.Time
	// Limit is up to 200. Zero is 25.
	Limit  int
	Offset int
}

// Sessions is one agent's conversations: the ones being held and the ones that were.
type Sessions struct {
	client *Client
	agent  *Agent
}

// Create opens a conversation and starts watching it.
//
// It returns once the backend is holding the conversation, so a session that has opened is
// one that is already listening. Without a CallID it is held in writing, which is what a
// conversation somebody comes back to usually is.
func (s *Sessions) Create(ctx context.Context, options SessionOptions) (*Session, error) {
	functions := options.Functions
	if functions == nil {
		functions = s.agent.functions
	}

	// The pipeline is what holds the socket and runs the caller's functions, so this builds
	// one rather than opening a second socket of its own. What differs between the two
	// surfaces is only how the session was asked for.
	pipeline := stream.Accelerated(stream.Config{
		Backend:   s.client.backend,
		Functions: functions,
		Logger:    options.Logger,
	})

	created, err := pipeline.JoinWith(ctx, s.requestOf(options))
	if err != nil {
		return nil, err
	}
	return newSession(s.client, s.agent, pipeline, created), nil
}

// Query returns the agent's conversations, newest first, the ones that ended included.
//
// What comes back are the rows rather than live handles: reading a conversation back is not
// the same as holding one, and most of these are over.
func (s *Sessions) Query(ctx context.Context, query Query) ([]acceleration.Session, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}

	params := acceleration.ListSessionsParams{Agent: pointer(s.agent.name)}
	narrow(query, &params.UserId, &params.Project, &params.Custom,
		&params.CreatedAfter, &params.CreatedBefore, &params.Limit, &params.Offset)
	if query.State != "" {
		state := acceleration.ListSessionsParamsState(query.State)
		params.State = &state
	}

	listed, err := api.ListSessionsWithResponse(ctx, &params)
	if err != nil {
		return nil, fmt.Errorf("client: listing the sessions of %s: %w", s.agent.name, err)
	}
	if listed.JSON200 == nil {
		return nil, failure("listing the sessions of "+s.agent.name, listed.Status(),
			listed.JSON400, listed.JSON401)
	}
	return *listed.JSON200, nil
}

// Search finds a conversation by what it was called.
//
// It reads the title, the description and the opening question, which is what a person
// remembers a conversation by. An incognito conversation is never found: nothing about it was
// written down to search.
func (s *Sessions) Search(ctx context.Context, text string, query Query) ([]acceleration.Session, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}

	params := acceleration.SearchSessionsParams{Agent: pointer(s.agent.name), Q: pointer(text)}
	narrow(query, &params.UserId, &params.Project, &params.Custom,
		&params.CreatedAfter, &params.CreatedBefore, &params.Limit, &params.Offset)
	if query.State != "" {
		state := acceleration.SearchSessionsParamsState(query.State)
		params.State = &state
	}

	found, err := api.SearchSessionsWithResponse(ctx, &params)
	if err != nil {
		return nil, fmt.Errorf("client: searching the sessions of %s: %w", s.agent.name, err)
	}
	if found.JSON200 == nil {
		return nil, failure("searching the sessions of "+s.agent.name, found.Status(),
			found.JSON400, found.JSON401)
	}
	return *found.JSON200, nil
}

// Get is one conversation, whether or not it is still being held.
func (s *Sessions) Get(ctx context.Context, id string) (*acceleration.Session, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}

	got, err := api.GetSessionWithResponse(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("client: reading the session %s: %w", id, err)
	}
	if got.JSON200 == nil {
		return nil, failure("reading the session "+id, got.Status(),
			got.JSON401, got.JSON403, got.JSON404)
	}
	return got.JSON200, nil
}

// Responses is a session's turns, read back without holding the conversation.
//
// For a conversation that has ended, or one being held somewhere else: the rows are in the
// backend either way, so reading them needs no socket.
func (s *Sessions) Responses(id string) *Responses {
	return &Responses{client: s.client, sessionID: id, Items: newItems(s.client, id, "")}
}

// requestOf renders the options as the session request, with the agent named by name.
func (s *Sessions) requestOf(options SessionOptions) acceleration.CreateSessionRequest {
	request := acceleration.CreateSessionRequest{
		Agent:           pointer(s.agent.name),
		Title:           pointer(options.Title),
		Description:     pointer(options.Description),
		Project:         pointer(options.Project),
		Incognito:       pointer(options.Incognito),
		ModelOverwrites: options.ModelOverwrites,
		CallId:          pointer(options.CallID),
		CallType:        pointer(options.CallType),
		ConversationId:  pointer(options.ConversationID),
		Instructions:    pointer(options.Instructions),
		UserId:          pointer(options.UserID),
	}
	if len(options.Custom) > 0 {
		custom := options.Custom
		request.Custom = &custom
	}
	// Held in writing unless a call was named, which is what the resource surface is mostly
	// for: a conversation somebody comes back to.
	if options.CallID == "" {
		request.Text = pointer(true)
	}
	// An incognito conversation writes no transcript by definition, so asking for one is a
	// contradiction the router refuses rather than quietly honours. Dropped here so a caller
	// setting both gets the conversation they asked for instead of a 400.
	if options.Persist && !options.Incognito {
		request.PersistConversation = pointer(true)
	}
	return request
}

// narrow fills in the filters the listing and the search share.
//
// One function rather than two, so the two cannot drift apart in what they honour: a filter
// respected by one and forgotten by the other would be a surprise at best, and at worst a
// list somebody reads another user's conversations out of.
func narrow(
	query Query,
	userID, project, custom **string,
	after, before **time.Time,
	limit, offset **int,
) {
	*userID = pointer(query.UserID)
	*project = pointer(query.Project)
	if len(query.Custom) > 0 {
		// Whatever will not encode is left off rather than sent broken: a label that cannot
		// be written down was never going to match anything, and a failed request would tell
		// the caller less than a list without it.
		if encoded, err := json.Marshal(query.Custom); err == nil {
			*custom = pointer(string(encoded))
		}
	}
	if !query.After.IsZero() {
		at := query.After
		*after = &at
	}
	if !query.Before.IsZero() {
		at := query.Before
		*before = &at
	}
	*limit = pointer(query.Limit)
	*offset = pointer(query.Offset)
}

// Session is one conversation being held in the backend.
//
// Nothing here does inference or touches media. The backend hears the caller, answers and
// speaks, and what arrives here are the events saying so. What stays here is function
// calling, because the functions are here.
type Session struct {
	// Responses is this conversation's turns, and what each of them was made of.
	Responses *Responses

	client   *Client
	agent    *Agent
	pipeline *stream.Pipeline
	created  *acceleration.Session
}

func newSession(client *Client, agent *Agent, pipeline *stream.Pipeline, created *acceleration.Session) *Session {
	return &Session{
		Responses: &Responses{
			client:    client,
			sessionID: created.Id,
			Items:     newItems(client, created.Id, ""),
		},
		client:   client,
		agent:    agent,
		pipeline: pipeline,
		created:  created,
	}
}

// ID is the backend's id for the conversation.
func (s *Session) ID() string { return s.created.Id }

// Created is what the router said when it opened the conversation.
func (s *Session) Created() *acceleration.Session { return s.created }

// ConversationID is the channel replies are written into, empty for one that keeps none.
func (s *Session) ConversationID() string {
	if s.created.ConversationId == nil {
		return ""
	}
	return *s.created.ConversationId
}

// Functions are the ones this conversation offers the model, to register into.
func (s *Session) Functions() *tools.Registry { return s.pipeline.Functions() }

// Events yields what the backend did until the conversation ends, when the channel closes.
func (s *Session) Events() <-chan stream.Event { return s.pipeline.Events() }

// Say speaks text without going through the model, for when the words were never in question.
func (s *Session) Say(text string, interrupt bool) error { return s.pipeline.Say(text, interrupt) }

// Interrupt abandons the reply being spoken.
func (s *Session) Interrupt() error { return s.pipeline.Interrupt() }

// SetInstructions changes what the agent is told to be, from the next turn.
func (s *Session) SetInstructions(instructions string) error {
	return s.pipeline.SetInstructions(instructions)
}

// Close ends the conversation. Safe to call after it has already ended.
func (s *Session) Close(ctx context.Context) error { return s.pipeline.Leave(ctx) }

// ForkOptions is what to change about a conversation while continuing it.
type ForkOptions struct {
	// Agent continues with a different agent, which is one of the reasons to fork.
	Agent           string
	Title           string
	Description     string
	Project         string
	Custom          map[string]any
	Instructions    string
	Incognito       bool
	ModelOverwrites *acceleration.ModelOverwrites
	// CallID is the call the fork joins, required when the parent held one and refused when
	// it did not: a voice conversation cannot be forked into a written one.
	CallID string
	// WithoutMessages starts the same configuration over from nothing rather than carrying
	// the parent's history across. What comparing two answers to one opening question wants.
	WithoutMessages bool
	// ResponseID carries the history only up to the end of this response, so the fork
	// branches from that point rather than from where the parent is now.
	ResponseID string

	// Functions are the fork's own. Nil inherits this session's, since a conversation
	// continued without them would offer the model tools nothing can run.
	Functions *tools.Registry
	Logger    *slog.Logger
}

// Fork continues this conversation as a new one.
//
// What a fork is for is asking the same question differently: from here on with a harder
// model, or of a different agent, or down a branch to be kept apart from the one already
// there. The parent is untouched and keeps its own transcript; the fork writes its own, so
// continuing a conversation twice gives two readable transcripts rather than one with both
// halves interleaved. An incognito conversation cannot be forked, because there is nothing to
// fork from.
func (s *Session) Fork(ctx context.Context, options ForkOptions) (*Session, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}

	request := acceleration.ForkSessionRequest{
		Agent:           pointer(options.Agent),
		Title:           pointer(options.Title),
		Description:     pointer(options.Description),
		Project:         pointer(options.Project),
		Instructions:    pointer(options.Instructions),
		Incognito:       pointer(options.Incognito),
		ModelOverwrites: options.ModelOverwrites,
		CallId:          pointer(options.CallID),
		ResponseId:      pointer(options.ResponseID),
	}
	if len(options.Custom) > 0 {
		custom := options.Custom
		request.Custom = &custom
	}
	// The history comes across by default, so only the refusal is worth sending: an absent
	// field and a true one mean the same thing and false is the whole request.
	if options.WithoutMessages {
		no := false
		request.Messages = &no
	}

	forked, err := api.ForkSessionWithResponse(ctx, s.ID(), request)
	if err != nil {
		return nil, fmt.Errorf("client: forking the session %s: %w", s.ID(), err)
	}
	if forked.JSON201 == nil {
		return nil, failure("forking the session "+s.ID(), forked.Status(),
			forked.JSON400, forked.JSON401, forked.JSON403, forked.JSON404)
	}

	functions := options.Functions
	if functions == nil {
		functions = s.pipeline.Functions()
	}
	// The fork is a conversation of its own with its own socket, so it gets its own pipeline
	// rather than moving this one onto it: the parent is still being held.
	pipeline := stream.Accelerated(stream.Config{
		Backend:   s.client.backend,
		Functions: functions,
		Logger:    options.Logger,
	})
	if err := pipeline.Watch(ctx, forked.JSON201); err != nil {
		return nil, err
	}

	agent := s.agent
	if options.Agent != "" && options.Agent != agent.name {
		agent = s.client.Agent(options.Agent)
	}
	return newSession(s.client, agent, pipeline, forked.JSON201), nil
}

// Chat is Stream Chat, opened on this conversation's transcript.
type Chat struct {
	// Client is the connected Stream client.
	Client *getstream.Stream
	// Channel is where replies are written.
	Channel *getstream.Channels
}

// Video is the Stream call this conversation is on.
type Video struct {
	Client *getstream.Stream
	Call   *getstream.Call
}

// Chat is the Stream Chat channel this conversation is written into.
//
// It needs a credential of its own: the channel is Stream Chat rather than this router, so a
// client reached by customer id has nothing to connect with. A conversation that keeps no
// transcript has no channel, and an incognito one never does.
func (s *Session) Chat() (*Chat, error) {
	channel := s.ConversationID()
	if channel == "" {
		return nil, fmt.Errorf("client: the session %s keeps no transcript, so there is no "+
			"channel to read: open it with Persist, and note that an incognito session never "+
			"has one", s.ID())
	}

	connected, err := s.client.stream()
	if err != nil {
		return nil, err
	}
	// The wire writes a conversation as type:id, which is what a Stream Chat CID is.
	// Splitting it here keeps that spelling out of the caller's way.
	kind, id := split(channel, "agent")
	return &Chat{Client: connected, Channel: connected.Chat().Channel(kind, id)}, nil
}

// Video is the Stream call the agent is on, ready to be joined.
//
// A conversation held in writing joins no call, and asking for one says so rather than
// handing back a call nobody is in.
func (s *Session) Video() (*Video, error) {
	if s.created.CallId == "" {
		return nil, fmt.Errorf("client: the session %s is held in writing, so there is no "+
			"call to join", s.ID())
	}

	connected, err := s.client.stream()
	if err != nil {
		return nil, err
	}
	kind := s.created.CallType
	if kind == "" {
		kind = "agent"
	}
	return &Video{Client: connected, Call: connected.Video().Call(kind, s.created.CallId)}, nil
}

// split reads a type:id pair, falling back to a default type for a bare id.
func split(pair, fallback string) (string, string) {
	for at := range len(pair) {
		if pair[at] == ':' {
			return pair[:at], pair[at+1:]
		}
	}
	return fallback, pair
}
