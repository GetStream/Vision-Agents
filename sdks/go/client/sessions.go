package client

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
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
	// ID is the UUID to hold the session by, for a caller that wants to know it before the
	// session exists. Empty lets the router generate a UUIDv7.
	ID string
	// Title and Description are what a person finds the conversation by later. Both are
	// searched.
	Title       string
	Description string
	// ProjectID groups conversations, and is carried as a cost label too.
	ProjectID string
	// Custom is the caller's own labels.
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
	// ConversationID resumes the channel an earlier session was held in.
	ConversationID string
	// History is the conversation so far, oldest first, for a backend that keeps its own
	// thread: the model is handed it before the first response and the router records none
	// of it. Up to 100 messages; not with ConversationID. Server side only.
	History []acceleration.HistoryMessage
	// Instructions overrides the agent's own system prompt for this conversation.
	Instructions string
	// UserID is who the conversation belongs to, for a backend opening one on somebody's
	// behalf. A client acting for a user leaves it empty: the token already says who.
	UserID string

	// Tools are the caller's own, for this conversation only. Nil uses the agent's, which is
	// the usual arrangement.
	Tools *tools.Registry
	// Logger is where the session reports what it could not do. Nil uses the default.
	Logger *slog.Logger
}

// Query is which of an agent's conversations to list.
type Query struct {
	// ProjectID narrows to one project's. A search covers every project, so Search refuses it.
	ProjectID string
	// UserID narrows to one user's, which only a server-side caller may ask for.
	UserID string
	// Modality narrows to how the user took part: "text", "voice" or "video".
	Modality string
	// State narrows to the sessions still running, "live", or the ones over, "ended".
	State string
	// AgentID narrows to the sessions created with this agent id.
	AgentID string
	// ConfigID narrows to the sessions one agent config ran.
	ConfigID string
	// Custom narrows to the sessions whose custom object holds every one of these pairs,
	// which is how a caller finds again what it labelled.
	Custom map[string]string
	// CreatedAfter and CreatedBefore bound when the session started, after inclusive and
	// before not, so two windows that meet share no session.
	CreatedAfter  time.Time
	CreatedBefore time.Time
	// Limit is up to 200. Zero is 25.
	Limit int
	// Cursor is the NextCursor of the page before, with the same filters. Empty is the
	// first page.
	Cursor string
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
	functions := options.Tools
	if functions == nil {
		functions = s.agent.tools
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

// Query returns a page of the agent's conversations, most recently updated first, the ones
// that ended included. Pass the page's NextCursor as Query.Cursor for the next one.
//
// What comes back are the rows rather than live handles: reading a conversation back is not
// the same as holding one, and most of these are over.
func (s *Sessions) Query(ctx context.Context, query Query) (*acceleration.SessionPage, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}

	listed, err := api.QuerySessionsWithResponse(ctx, s.queryOf("", query))
	if err != nil {
		return nil, fmt.Errorf("client: listing the sessions of %s: %w", s.agent.name, err)
	}
	if listed.JSON200 == nil {
		return nil, failure("listing the sessions of "+s.agent.name, listed.Status(),
			listed.JSON400, listed.JSON401)
	}
	return listed.JSON200, nil
}

// Search finds a conversation by what it was called, best match first.
//
// It reads the title, the description and the opening question, which is what a person
// remembers a conversation by. An incognito conversation is never found: nothing about it was
// written down to search. It pages the same way Query does.
func (s *Sessions) Search(ctx context.Context, text string, query Query) (*acceleration.SessionPage, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}

	found, err := api.QuerySessionsWithResponse(ctx, s.queryOf(text, query))
	if err != nil {
		return nil, fmt.Errorf("client: searching the sessions of %s: %w", s.agent.name, err)
	}
	if found.JSON200 == nil {
		return nil, failure("searching the sessions of "+s.agent.name, found.Status(),
			found.JSON400, found.JSON401)
	}
	return found.JSON200, nil
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

// Update changes one conversation, whether or not it is still being held.
//
// Only this session changes: the agent config it started from is untouched. One that ended
// can still be renamed and relabelled; instructions, models and voice need it running, and
// take over from its next turn. A target that does not route is refused and the session
// carries on as it was. Only a backend may ask.
func (s *Sessions) Update(ctx context.Context, id string, update SessionUpdate) (*acceleration.Session, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}

	updated, err := api.UpdateSessionWithResponse(ctx, id, update)
	if err != nil {
		return nil, fmt.Errorf("client: updating the session %s: %w", id, err)
	}
	if updated.JSON200 == nil {
		return nil, failure("updating the session "+id, updated.Status(),
			updated.JSON400, updated.JSON401, updated.JSON403, updated.JSON404)
	}
	return updated.JSON200, nil
}

// Delete deletes a conversation, running or ended: it is stopped, and its turns and what it
// remembered are deleted with it. The user's other memories are kept.
func (s *Sessions) Delete(ctx context.Context, id string) error {
	api, err := s.client.api()
	if err != nil {
		return err
	}

	deleted, err := api.DeleteSessionWithResponse(ctx, id)
	if err != nil {
		return fmt.Errorf("client: deleting session %s: %w", id, err)
	}
	if deleted.StatusCode() != http.StatusNoContent {
		return failure("deleting session "+id, deleted.Status(),
			deleted.JSON400, deleted.JSON401, deleted.JSON404)
	}
	return nil
}

// DeleteMemories deletes what one conversation remembered, running or ended, and leaves the
// rest of the user's memories alone. Only a backend may ask.
func (s *Sessions) DeleteMemories(ctx context.Context, id string) error {
	api, err := s.client.api()
	if err != nil {
		return err
	}

	deleted, err := api.DeleteSessionMemoriesWithResponse(ctx, id)
	if err != nil {
		return fmt.Errorf("client: deleting the memories of %s: %w", id, err)
	}
	if deleted.StatusCode() != http.StatusNoContent {
		return failure("deleting the memories of "+id, deleted.Status(),
			deleted.JSON400, deleted.JSON401, deleted.JSON403, deleted.JSON404)
	}
	return nil
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
		Id:              pointer(options.ID),
		Agent:           pointer(s.agent.name),
		Title:           pointer(options.Title),
		Description:     pointer(options.Description),
		ProjectId:       pointer(options.ProjectID),
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
	if len(options.History) > 0 {
		history := options.History
		request.History = &history
	}
	// Held in writing unless a call was named, which is what the resource surface is mostly
	// for: a conversation somebody comes back to.
	if options.CallID == "" {
		request.Text = pointer(true)
	}
	return request
}

// queryOf renders the query the listing and the search share, narrowed to this agent. Text
// makes it a search.
//
// One function rather than two, so the two cannot drift apart in what they honour: a filter
// respected by one and forgotten by the other would be a surprise at best, and at worst a
// list somebody reads another user's conversations out of.
func (s *Sessions) queryOf(text string, query Query) acceleration.SessionQuery {
	filter := acceleration.SessionFilter{
		Agent:     equals(s.agent.name),
		ProjectId: equals(query.ProjectID),
		UserId:    equals(query.UserID),
		Modality:  equals(query.Modality),
		State:     equals(query.State),
		AgentId:   equals(query.AgentID),
		ConfigId:  equals(query.ConfigID),
	}
	if len(query.Custom) > 0 {
		custom := query.Custom
		filter.Custom = &custom
	}
	if window := timeRangeOf(query.CreatedAfter, query.CreatedBefore); window != nil {
		filter.CreatedAt = window
	}
	if text != "" {
		filter.Text = &acceleration.TextMatch{Q: text}
	}
	return acceleration.SessionQuery{
		Filter: &filter,
		Limit:  pointer(int64(query.Limit)),
		Cursor: pointer(query.Cursor),
	}
}

// timeRangeOf is the window the two bounds name, or nil when neither was given.
func timeRangeOf(after, before time.Time) *acceleration.TimeRange {
	if after.IsZero() && before.IsZero() {
		return nil
	}
	window := &acceleration.TimeRange{}
	if !after.IsZero() {
		window.Gte = &after
	}
	if !before.IsZero() {
		window.Lt = &before
	}
	return window
}

// equals is a filter field matching value exactly, or nil to leave the field out.
func equals(value string) *acceleration.Equals {
	if value == "" {
		return nil
	}
	var matched acceleration.Equals
	// A string always encodes.
	_ = matched.FromEquals0(value)
	return &matched
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

// Hold wraps the session a pipeline is already holding, for a caller that opened it its own
// way: the agents package renders a whole configuration into the request rather than naming
// a stored one, and hands what it opened over here so both end up as the same Session.
func (c *Client) Hold(agent string, pipeline *stream.Pipeline) (*Session, error) {
	created := pipeline.Session()
	if created == nil {
		return nil, errors.New("client: the pipeline is not holding a session")
	}
	return newSession(c, c.Agent(agent), pipeline, created), nil
}

func newSession(client *Client, agent *Agent, pipeline *stream.Pipeline, created *acceleration.Session) *Session {
	return &Session{
		Responses: &Responses{
			client:    client,
			sessionID: created.Id,
			kept:      created.ConversationId != nil && *created.ConversationId != "",
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

// ContextTruncated says whether older history was left out of what the model was given.
func (s *Session) ContextTruncated() bool {
	return s.created.ContextTruncated != nil && *s.created.ContextTruncated
}

// Tools are the ones this conversation offers the model, to add to.
func (s *Session) Tools() *tools.Registry { return s.pipeline.Functions() }

// Events yields what the backend did until the conversation ends, when the channel closes.
func (s *Session) Events() <-chan stream.Event { return s.pipeline.Events() }

// Say speaks text without going through the model, for when the words were never in question.
func (s *Session) Say(text string, interrupt bool) error { return s.pipeline.Say(text, interrupt) }

// Wait blocks until the conversation ends or the context does.
func (s *Session) Wait(ctx context.Context) error {
	events := s.Events()
	for {
		select {
		case _, open := <-events:
			if !open {
				return nil
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// Interrupt abandons the reply being spoken.
func (s *Session) Interrupt() error { return s.pipeline.Interrupt() }

// SetInstructions changes what the agent is told to be, from the next turn.
func (s *Session) SetInstructions(instructions string) error {
	return s.pipeline.SetInstructions(instructions)
}

// Close stops the conversation. Safe to call after it has already ended. What it recorded
// and remembered is kept; Delete takes it away.
func (s *Session) Close(ctx context.Context) error { return s.pipeline.Leave(ctx) }

// Delete deletes this conversation. See Sessions.Delete.
func (s *Session) Delete(ctx context.Context) error {
	return s.agent.Sessions.Delete(ctx, s.ID())
}

// DeleteMemories deletes what this conversation remembered. See Sessions.DeleteMemories.
func (s *Session) DeleteMemories(ctx context.Context) error {
	return s.agent.Sessions.DeleteMemories(ctx, s.ID())
}

// SessionUpdate is what to change about a session. A nil field is left as it is; an empty
// Sts makes the session a cascade again, and an empty Voice returns to the provider's
// default. The id, the call and incognito cannot change.
type SessionUpdate = acceleration.UpdateSessionRequest

// Update changes this session: its title, description, custom labels, instructions, models
// or voice. See Sessions.Update.
func (s *Session) Update(ctx context.Context, update SessionUpdate) (*acceleration.Session, error) {
	return s.agent.Sessions.Update(ctx, s.ID(), update)
}

// ForkOptions is what to change about a conversation while continuing it.
type ForkOptions struct {
	// Agent continues with a different agent, which is one of the reasons to fork.
	Agent           string
	Title           string
	Description     string
	ProjectID       string
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

	// Tools are the fork's own. Nil inherits this session's, since a conversation continued
	// without them would offer the model tools nothing can run.
	Tools  *tools.Registry
	Logger *slog.Logger
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
		ProjectId:       pointer(options.ProjectID),
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

	functions := options.Tools
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
// client reached by customer id has nothing to connect with. An incognito conversation keeps
// no transcript, so it never has a channel.
func (s *Session) Chat() (*Chat, error) {
	channel := s.ConversationID()
	if channel == "" {
		return nil, fmt.Errorf("client: the session %s keeps no transcript, so there is no "+
			"channel to read: an incognito session never has one", s.ID())
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
