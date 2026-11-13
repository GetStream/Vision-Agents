package client

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"slices"
	"sync/atomic"
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
	// Tags are cost labels for this conversation, merged with the agent's own. Where both
	// name a key, these win.
	Tags map[string]string
	// Incognito holds the conversation and keeps nothing: no row, no turns, no transcript.
	// It cannot be searched for, listed or forked afterwards, which is the point of it.
	Incognito bool
	// ModelOverwrites changes the models for this conversation alone.
	ModelOverwrites *acceleration.ModelOverwrites

	// StartVoice has the agent join the session's call as soon as it opens: Call, else its
	// own, agent:<session id>. False holds the conversation in writing, and Session.Voice
	// starts it later.
	StartVoice bool
	// Call is an existing Stream call for the agent to join, such as the default call the
	// people are already on. It needs StartVoice. Nil is agent:<session id>. Server side only.
	Call *acceleration.SessionCall
	// History is the conversation so far, oldest first, for a backend that keeps its own
	// thread: the model is handed it before the first response and the router records none
	// of it. Up to 100 messages. Server side only.
	History []acceleration.HistoryMessage
	// Instructions replace the agent's for this conversation alone, for a role given at
	// call time. Empty leaves the agent's. Server side only.
	Instructions string
	// UserID is who the conversation belongs to, for a backend opening one on somebody's
	// behalf. A client acting for a user leaves it empty: the token already says who.
	UserID string
	// Team is the team the conversation's work is counted under, its team spend limits
	// included, for a backend opening one for a team: 1 to 59 letters, digits, _ . and -.
	// Empty is the team the caller's token names, else the agent's tags: team. A token
	// signed for another team is refused. Server side only.
	Team string
	// ConnectorBindings are the connections to use for the agent's connector bindings chosen
	// per session, by alias. Each must be the caller's own; a binding with the agent's fixed
	// connection cannot be given one.
	ConnectorBindings map[string]string
	// ReturnURL is the app's own page a login or step-up the conversation asks for goes back
	// to, with connection_id and status added to its query; its origin is the page that opens
	// the consent's launch_url. Empty sends them back to the dashboard.
	ReturnURL string
	// Timezone is the end user's IANA timezone, such as Europe/Amsterdam. It sets the
	// zone the agent reads dates in for this conversation and, for a named user, is
	// remembered as theirs for the scheduled tasks and sessions that follow. Empty
	// leaves the user's stored zone, then the agent's default, to decide.
	Timezone string

	// Tools are the caller's own, for this conversation only. Nil uses the agent's, which is
	// the usual arrangement.
	Tools *tools.Registry
	// Logger is where the session reports what it could not do. Nil uses the default.
	Logger *slog.Logger
}

// Filter is which sessions to list, in the shape the router reads: a map of field keys and
// $and / $or, sent as it is written. The keys of one map are ANDed; $and and $or take 1 to 10
// filters and nest at most 3 deep. The fields, and nothing else:
//
//   - "project_id": a string, {"$eq": s} or {"$in": [up to 50 strings]}
//   - "text": {"$q": "words"}, once, at the top level or directly in the top-level $and, and
//     never inside $or. Sessions.Search writes it.
//   - "review_score": an integer 1 to 5, or {"$eq" | "$gt" | "$gte" | "$lt" | "$lte": n},
//     several of which all have to hold. Server side only.
//   - "agent": set by Sessions.Query to the agent asked, whatever the caller wrote there.
//   - "user_id": a string or {"$eq": s}, at the top level or directly in the top-level $and.
//     Server side only.
//   - "config_id": a string or {"$eq": s}, the agent config, which a rename leaves alone.
//   - "state": "live" or "ended", or {"$eq": s}.
//   - "created_at": {"$gte": t, "$lt": t}, either bound or both, as RFC 3339 times.
//   - "custom": a map of string labels the session's custom object holds, every one of them.
//
// Anything else is refused by the router with a 400 that names it.
//
//	client.Filter{"$or": []client.Filter{{"project_id": "docs"}, {"project_id": "blog"}}}
type Filter map[string]any

// The fields a Sort orders by.
const (
	// SortLastMessageAt is when the agent last answered, or when the session began if it has
	// not. Either direction. It is the order when no Sort is given.
	SortLastMessageAt = "last_message_at"
	// SortPinned lists the pinned sessions first. It sorts descending only, comes first, and
	// may be followed by SortLastMessageAt in either direction.
	SortPinned = "pinned"
	// SortReviewScore is the review's score, sessions without one last either way. Server
	// side only.
	SortReviewScore = "review_score"
	// SortRelevance is the best match first: the order of a text search, and only of one.
	SortRelevance = "relevance"
)

// Sort is one field to order by.
type Sort struct {
	// Field is one of SortLastMessageAt, SortPinned, SortReviewScore and SortRelevance.
	Field string `json:"field"`
	// Direction is -1, descending, or 1, ascending. Zero leaves it to the router, which is
	// descending.
	Direction int `json:"direction,omitempty"`
}

// Query is which of an agent's conversations to list, in what order, and which page of them.
type Query struct {
	// Filter narrows the list. Nil is every session of the agent the caller may see.
	Filter Filter
	// Sort is the order. Empty is the router's own: SortLastMessageAt descending. The orders
	// it takes are one SortLastMessageAt; SortPinned alone or followed by SortLastMessageAt;
	// and one SortReviewScore. Search takes none.
	Sort []Sort
	// Limit is up to 200. Zero is 25.
	Limit int
	// Cursor is the NextCursor of the page before, with the same filter and sort. Empty is
	// the first page.
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
// one that is already listening. Without StartVoice it is held in writing, which is what a
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

// Resume carries on a conversation held in writing, by the id of the session it was held
// in, and starts watching it. One that ended is reopened with what was said in it.
func (s *Sessions) Resume(ctx context.Context, id string) (*Session, error) {
	got, err := s.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	pipeline := stream.Accelerated(stream.Config{Backend: s.client.backend, Functions: s.agent.tools})
	if err := pipeline.Watch(ctx, got); err != nil {
		return nil, err
	}
	return newSession(s.client, s.agent, pipeline, got), nil
}

// Query returns a page of the agent's conversations, the ones that ended included: the one
// that last had an answer first, unless Query.Sort says otherwise. Pass the page's NextCursor
// as Query.Cursor for the next one.
//
// The filter is sent as written, with its top-level "agent" set to this agent.
//
// What comes back are the rows rather than live handles: reading a conversation back is not
// the same as holding one, and most of these are over.
func (s *Sessions) Query(ctx context.Context, query Query) (*acceleration.SessionPage, error) {
	return s.query(ctx, "listing", s.filterOf(query.Filter), query)
}

// Search finds a conversation by what it was called, best match first.
//
// It reads the title, the description and the opening question, which is what a person
// remembers a conversation by. An incognito conversation is never found: nothing about it was
// written down to search. It takes the same filter and pages the same way Query does, with
// the phrase as the filter's top-level "text": a filter that already has one is refused, and
// so is a Sort, because a search is ordered by relevance.
func (s *Sessions) Search(ctx context.Context, text string, query Query) (*acceleration.SessionPage, error) {
	if _, given := query.Filter["text"]; given {
		return nil, errors.New(`client: the filter already has "text": Search writes it from the phrase`)
	}
	if len(query.Sort) > 0 {
		return nil, errors.New("client: a search is ordered by relevance and takes no Sort")
	}
	filter := s.filterOf(query.Filter)
	filter["text"] = map[string]any{"$q": text}
	return s.query(ctx, "searching", filter, query)
}

// filterOf is the filter sent for this agent: the caller's top-level keys with "agent" set
// to the agent's name.
//
// The keys are copied rather than wrapped in a new $and, because the router takes "text" and
// "user_id" at the top level or directly in the top-level $and and nowhere deeper.
func (s *Sessions) filterOf(given Filter) Filter {
	filter := make(Filter, len(given)+2)
	maps.Copy(filter, given)
	filter["agent"] = s.agent.name
	return filter
}

// sessionQuery is the body of a session query. The filter is a map rather than the generated
// acceleration.SessionFilter so that it goes over exactly as the caller wrote it.
type sessionQuery struct {
	Filter Filter `json:"filter"`
	Sort   []Sort `json:"sort,omitempty"`
	Limit  int    `json:"limit,omitempty"`
	Cursor string `json:"cursor,omitempty"`
}

// query posts the one request the listing and the search share.
func (s *Sessions) query(ctx context.Context, doing string, filter Filter, query Query) (*acceleration.SessionPage, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(sessionQuery{Filter: filter, Sort: query.Sort, Limit: query.Limit, Cursor: query.Cursor})
	if err != nil {
		return nil, fmt.Errorf("client: %s the sessions of %s: the filter does not encode: %w", doing, s.agent.name, err)
	}
	listed, err := api.QuerySessionsWithBodyWithResponse(ctx, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("client: %s the sessions of %s: %w", doing, s.agent.name, err)
	}
	if listed.JSON200 == nil {
		return nil, failure(doing+" the sessions of "+s.agent.name, listed.HTTPResponse, listed.Body)
	}
	return listed.JSON200, nil
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
		return nil, failure("reading the session "+id, got.HTTPResponse, got.Body)
	}
	return got.JSON200, nil
}

// Update changes one conversation, whether or not it is still being held.
//
// Only this session changes: the agent config it started from is untouched. One that ended
// can still be renamed and relabelled; models and voice need it running, and
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
		return nil, failure("updating the session "+id, updated.HTTPResponse, updated.Body)
	}
	return updated.JSON200, nil
}

// Pin pins a conversation, so a Query sorted by SortPinned lists it first. Its PinnedAt says
// when; pinning one that is already pinned keeps the time it was first pinned, and a pin does
// not move it in the SortLastMessageAt order.
func (s *Sessions) Pin(ctx context.Context, id string) (*acceleration.Session, error) {
	pinned := true
	return s.Update(ctx, id, SessionUpdate{Pinned: &pinned})
}

// Unpin takes the pin off a conversation, and its PinnedAt with it.
func (s *Sessions) Unpin(ctx context.Context, id string) (*acceleration.Session, error) {
	pinned := false
	return s.Update(ctx, id, SessionUpdate{Pinned: &pinned})
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
		return failure("deleting session "+id, deleted.HTTPResponse, deleted.Body)
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
		return failure("deleting the memories of "+id, deleted.HTTPResponse, deleted.Body)
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
		StartVoice:      pointer(options.StartVoice),
		Call:            options.Call,
		Instructions:    pointer(options.Instructions),
		UserId:          pointer(options.UserID),
		Team:            pointer(options.Team),
		ReturnUrl:       pointer(options.ReturnURL),
		Timezone:        pointer(options.Timezone),
	}
	if len(options.Custom) > 0 {
		custom := options.Custom
		request.Custom = &custom
	}
	if len(options.Tags) > 0 {
		tags := options.Tags
		request.Tags = &tags
	}
	if len(options.History) > 0 {
		history := options.History
		request.History = &history
	}
	if len(options.ConnectorBindings) > 0 {
		chosen := make([]acceleration.SessionConnectorBinding, 0, len(options.ConnectorBindings))
		for _, name := range slices.Sorted(maps.Keys(options.ConnectorBindings)) {
			chosen = append(chosen, acceleration.SessionConnectorBinding{Name: name, ConnectionId: options.ConnectorBindings[name]})
		}
		request.ConnectorBindings = &chosen
	}
	return request
}

// Session is one conversation being held in the backend.
//
// Nothing here does inference or touches media. The backend hears the caller, answers and
// speaks, and what arrives here are the events saying so. What stays here is function
// calling, because the functions are here.
type Session struct {
	// Responses is this conversation's turns, and what each of them was made of.
	Responses *Responses
	// Voice starts and stops the agent talking on the session's call.
	Voice *Voice

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
	voice := &Voice{client: client, sessionID: created.Id}
	voice.on.Store(created.CallId != "")
	voice.call.Store(&acceleration.SessionCall{Type: created.CallType, Id: created.CallId})
	return &Session{
		Responses: &Responses{
			client:    client,
			sessionID: created.Id,
			Items:     newItems(client, created.Id, ""),
			pipeline:  pipeline,
		},
		Voice:    voice,
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

// AnswerApproval answers a tool call whose step waits for a person to allow it, as
// stream.Pipeline.AnswerApproval does.
func (s *Session) AnswerApproval(toolCallID, requestID string, allowed bool, summary string) error {
	return s.pipeline.AnswerApproval(toolCallID, requestID, allowed, summary)
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
// default. Pinned pins the session when true and unpins it when false. The id, the call and
// incognito cannot change.
type SessionUpdate = acceleration.UpdateSessionRequest

// Update changes this session: its title, description, custom labels, models,
// voice or pin. See Sessions.Update.
func (s *Session) Update(ctx context.Context, update SessionUpdate) (*acceleration.Session, error) {
	return s.agent.Sessions.Update(ctx, s.ID(), update)
}

// Pin pins this session. See Sessions.Pin.
func (s *Session) Pin(ctx context.Context) (*acceleration.Session, error) {
	return s.agent.Sessions.Pin(ctx, s.ID())
}

// Unpin takes the pin off this session. See Sessions.Unpin.
func (s *Session) Unpin(ctx context.Context) (*acceleration.Session, error) {
	return s.agent.Sessions.Unpin(ctx, s.ID())
}

// ForkOptions is what to change about a conversation while continuing it.
type ForkOptions struct {
	// Agent continues with a different agent, which is one of the reasons to fork.
	Agent           string
	Title           string
	Description     string
	ProjectID       string
	Custom          map[string]any
	Incognito       bool
	ModelOverwrites *acceleration.ModelOverwrites
	// WithoutMessages starts the same configuration over from nothing rather than carrying
	// the parent's history across. What comparing two answers to one opening question wants.
	WithoutMessages bool
	// ResponseID carries the history only up to the end of this response, so the fork
	// branches from that point rather than from where the parent is now.
	ResponseID string
	// ToolTimeout is how long the fork's model waits for one of your functions. Zero keeps a
	// running parent's, and a parent that ended has the router's default of 30 seconds.
	ToolTimeout time.Duration

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
		Incognito:       pointer(options.Incognito),
		ModelOverwrites: options.ModelOverwrites,
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
	if options.ToolTimeout > 0 {
		milliseconds := int(options.ToolTimeout / time.Millisecond)
		request.ToolTimeoutMs = &milliseconds
	}

	forked, err := api.ForkSessionWithResponse(ctx, s.ID(), request)
	if err != nil {
		return nil, fmt.Errorf("client: forking the session %s: %w", s.ID(), err)
	}
	if forked.JSON201 == nil {
		return nil, failure("forking the session "+s.ID(), forked.HTTPResponse, forked.Body)
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
	if !s.Voice.Started() {
		return nil, fmt.Errorf("client: the session %s is held in writing, so there is no "+
			"call to join", s.ID())
	}

	connected, err := s.client.stream()
	if err != nil {
		return nil, err
	}
	call := s.Voice.call.Load()
	return &Video{Client: connected, Call: connected.Video().Call(call.Type, call.Id)}, nil
}

// Voice is the agent talking on a session's call: agent:<session id>, or a call named for it.
// Typed and spoken turns are one conversation, with one history and one channel, whether
// voice is on or not.
type Voice struct {
	client    *Client
	sessionID string
	on        atomic.Bool
	// call is the call the router last said voice is on.
	call atomic.Pointer[acceleration.SessionCall]
}

// VoiceOption changes how voice is started.
type VoiceOption func(*acceleration.StartSessionVoiceRequest)

// OnCall has the agent join an existing Stream call, such as the default call the people are
// already on, rather than the one voice was last on or agent:<session id>. Naming another
// call while voice is on is refused. Server side only.
func OnCall(callType, callID string) VoiceOption {
	return func(request *acceleration.StartSessionVoiceRequest) {
		request.Call = &acceleration.SessionCall{Type: callType, Id: callID}
	}
}

// Start has the agent join the session's call and carry the conversation on there. Starting
// voice that is already on does nothing.
func (v *Voice) Start(ctx context.Context, opts ...VoiceOption) (*acceleration.Session, error) {
	api, err := v.client.api()
	if err != nil {
		return nil, err
	}
	var request acceleration.StartSessionVoiceRequest
	for _, opt := range opts {
		opt(&request)
	}
	started, err := api.StartSessionVoiceWithResponse(ctx, v.sessionID, request)
	if err != nil {
		return nil, fmt.Errorf("client: starting voice on the session %s: %w", v.sessionID, err)
	}
	if started.JSON200 == nil {
		return nil, failure("starting voice on the session "+v.sessionID, started.HTTPResponse, started.Body)
	}
	v.on.Store(true)
	v.call.Store(&acceleration.SessionCall{Type: started.JSON200.CallType, Id: started.JSON200.CallId})
	return started.JSON200, nil
}

// Stop takes the agent off the call and carries the conversation on in writing. Stopping
// voice that is off does nothing.
func (v *Voice) Stop(ctx context.Context) (*acceleration.Session, error) {
	api, err := v.client.api()
	if err != nil {
		return nil, err
	}
	stopped, err := api.StopSessionVoiceWithResponse(ctx, v.sessionID)
	if err != nil {
		return nil, fmt.Errorf("client: stopping voice on the session %s: %w", v.sessionID, err)
	}
	if stopped.JSON200 == nil {
		return nil, failure("stopping voice on the session "+v.sessionID, stopped.HTTPResponse, stopped.Body)
	}
	v.on.Store(false)
	return stopped.JSON200, nil
}

// Started reports whether the agent is on the call, as this process last saw it.
func (v *Voice) Started() bool { return v.on.Load() }

// split reads a type:id pair, falling back to a default type for a bare id.
func split(pair, fallback string) (string, string) {
	for at := range len(pair) {
		if pair[at] == ':' {
			return pair[:at], pair[at+1:]
		}
	}
	return fallback, pair
}
