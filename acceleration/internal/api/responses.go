package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/video"
	"github.com/danielgtaylor/huma/v2"
)

// errNoStore is what the response paths say where nothing was written down. A turn's items are
// read back from Postgres, so a deployment without one can start a response but has nothing
// to show afterwards.
var errNoStore = notConfigured("this deployment does not record what sessions said")

// maxVideos bounds the clips on one response, so that at video.MaxFrames each a turn stays
// inside the hundred images the strictest vision provider takes in one request.
const maxVideos = 2

// sizeOfResponse is the bytes of the text, images and videos a response carries.
func sizeOfResponse(body *CreateResponseRequest) int {
	size := len(body.Text)
	for _, image := range value(body.Images) {
		size += len(image.Url)
	}
	for _, clip := range value(body.Videos) {
		size += len(clip.Url)
	}
	for _, file := range value(body.Files) {
		size += len(file.Url)
	}
	return size
}

// createResponse asks the agent something and names the turn it answers as.
//
// It is respond with an id back, which is the whole of the difference. Without one a caller
// following a particular answer has to watch every event on the socket and work out which
// belong to the turn it asked for; with one it can ask for that turn's items instead.
func (s *Server) createResponse(ctx context.Context, request *createResponseRequest) (*createResponseResponse, error) {
	// A response is asked of a running session rather than a stored one: a call that ended
	// can be read and forked, but not talked to. A chat that ended is reopened.
	leaveUncosted(ctx)
	var files []FileSource
	if request.Body != nil {
		files = value(request.Body.Files)
	}
	if request.Body != nil {
		if len(files) > 0 && request.Body.Audio != nil {
			return nil, errDocumentWithVoiceNote
		}
		// An inline file is held to its own limit (fetchDocument), which the route's body limit
		// makes room for; the rest of the request is held to the usual one.
		size := sizeOfResponse(request.Body)
		if len(files) > 0 && strings.HasPrefix(files[0].Url, "data:") {
			size -= len(files[0].Url)
		}
		if err := refuseOversizeWithoutAudio(request.Body.Audio, size); err != nil {
			return nil, err
		}
	}
	if request.Body != nil && request.Body.Replaces != nil {
		return s.branchResponse(ctx, request.Id, request.Body)
	}
	found, sent, failure := s.sessionToAnswer(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	defer sent()
	if request.Body == nil || (request.Body.Text == "" && request.Body.Audio == nil && len(value(request.Body.Images)) == 0 && len(files) == 0) {
		return nil, invalidRequest("there is nothing to answer")
	}

	images := make([]wireImage, 0, len(value(request.Body.Images)))
	for _, sent := range value(request.Body.Images) {
		images = append(images, wireImage{URL: sent.Url, Detail: string(value(sent.Detail))})
	}
	parts, err := imagesFromWire(images)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	action := value(request.Body.UiAction)
	if !conversation.ValidUIAction(action) {
		return nil, invalidRequest(fmt.Sprintf("a ui_action names message_id, format and name, in at most %d bytes", conversation.MaxUIAction))
	}
	// The file is read once the request is otherwise known to be good, and before anything
	// else is fetched: screening it may be billed, and a request refused after it would be.
	var document *sentDocument
	if len(files) > 0 {
		retry := found.Durable() && found.Taken(value(request.Body.RequestId))
		if !retry {
			// Whatever would refuse the question is asked first, as before hearing a voice
			// note: reading the file pays for its pictures and its screen.
			if len(value(request.Body.Videos)) > maxVideos {
				return nil, invalidRequest(fmt.Sprintf("at most %d videos go with one response", maxVideos))
			}
			asked := options.LLM{MaxOutputTokens: request.Body.MaxOutputTokens, ReasoningEffort: value(request.Body.ReasoningEffort)}
			if _, err := found.CheckRespond(ctx, value(request.Body.RequestId), request.Body.Text, len(parts) > 0, asked); err != nil {
				return nil, refusedResponse(err)
			}
		}
		sent, failure := s.readDocument(ctx, found, files[0], retry)
		if failure != nil {
			return nil, failure
		}
		document = &sent
		if !retry {
			ctx = agent.WithDocument(ctx, sent.read)
		}
	}

	videos := value(request.Body.Videos)
	if len(videos) > maxVideos {
		return nil, invalidRequest(fmt.Sprintf("at most %d videos go with one response", maxVideos))
	}
	if len(videos) > 0 {
		if request.Body.Audio != nil {
			// With a voice note the question is checked as it is before the note is heard
			// (withHeardNote), the agent's refusal first, so the order of refusals is the same
			// whether or not videos go with it.
			asked := options.LLM{MaxOutputTokens: request.Body.MaxOutputTokens, ReasoningEffort: value(request.Body.ReasoningEffort)}
			if _, err := found.CheckRespond(ctx, value(request.Body.RequestId), request.Body.Text, true, asked); err != nil {
				return nil, refusedResponse(err)
			}
		}
		sample, err := found.SampleVideos(ctx, value(request.Body.RequestId), request.Body.Text)
		if err != nil {
			return nil, refusedResponse(err)
		}
		if !sample {
			// The retry of a question already taken is answered with its receipt, so its
			// videos are not fetched again.
			videos = nil
		}
	}
	for index, sent := range videos {
		frames, length, err := video.Frames(ctx, sent.Url, value(sent.MaxFrames))
		if err != nil {
			return nil, invalidRequest(err.Error())
		}
		for _, frame := range frames {
			frame.Image.Caption = fmt.Sprintf("video %d, %.1fs of %.1fs", index+1, frame.At.Seconds(), length.Seconds())
			parts = append(parts, frame.Image)
		}
	}

	asked := options.LLM{MaxOutputTokens: request.Body.MaxOutputTokens, ReasoningEffort: value(request.Body.ReasoningEffort)}
	commandID := value(request.Body.RequestId)
	text, notes, failure := s.withHeardNote(ctx, found, commandID, request.Body.Text, request.Body.Audio, len(parts) > 0, asked)
	if failure != nil {
		return nil, failure
	}

	var responseID string
	if commandID != "" && found.Durable() {
		var file *sandbox.File
		if document != nil {
			file = &document.file
		}
		_, responseID, err = found.RespondCommandWithAction(ctx, commandID, text, "", parts, asked, action, file, notes...)
	} else {
		told := text
		// A session not kept in Stream Chat has no reply to read the words of.
		if note := conversation.DescribeUIAction(action, ""); note != "" {
			told += "\n\n" + note
		}
		responseID, err = found.Respond(ctx, told, parts, asked, parts[:len(images)]...)
	}
	if err != nil {
		return nil, refusedResponse(err)
	}
	if len(videos) > 0 {
		found.SawVideo()
	}
	if document != nil {
		s.keepDocument(ctx, found, responseID, *document)
	}
	if responseID == "" {
		// The turn is being answered, it just has no name. A session that records nothing
		// has no row to hand back, and a native one lets the model decide what a turn is.
		// Refusing would be wrong -- the agent is answering -- so the id is empty and the
		// caller reads the socket, which is what it would have done before this existed.
		return &createResponseResponse{Body: AgentResponse{SessionId: found.ID(), Status: AgentResponseStatusRunning,
			CreatedAt: found.CreatedAt()}}, nil
	}

	return &createResponseResponse{Body: AgentResponse{Id: responseID, SessionId: found.ID(), Said: &text,
		Status: AgentResponseStatusRunning, CreatedAt: time.Now().UTC()}}, nil
}

// branchResponse is createResponse with replaces: a regenerate of the replaced response's
// reply without text, an edit of its question with it.
func (s *Server) branchResponse(ctx context.Context, id string, body *CreateResponseRequest) (*createResponseResponse, error) {
	replaces, commandID := *body.Replaces, value(body.RequestId)
	switch {
	case replaces == "":
		return nil, invalidRequest("replaces names the response to regenerate or edit")
	case commandID == "":
		return nil, invalidRequest("a request_id is required with replaces")
	case body.Audio != nil || len(value(body.Images)) > 0 || len(value(body.Videos)) > 0 || len(value(body.Files)) > 0 || body.UiAction != nil:
		return nil, invalidRequest("a regenerate or an edit takes text alone: images, audio, videos, files and ui_action cannot go with replaces")
	}
	found, sent, failure := s.sessionToAnswer(ctx, id)
	if failure != nil {
		return nil, failure
	}
	defer sent()

	asked := options.LLM{MaxOutputTokens: body.MaxOutputTokens, ReasoningEffort: value(body.ReasoningEffort)}
	said := body.Text
	var branched session.Branched
	var err error
	if body.Text == "" {
		// The question is asked again as it was.
		if s.store != nil {
			if replaced, err := s.store.SessionResponse(ctx, OwnerFrom(ctx).CustomerID, found.ID(), replaces); err == nil {
				said = replaced.Said
			}
		}
		branched, err = found.Regenerate(ctx, commandID, replaces, asked)
	} else {
		branched, err = found.Edit(ctx, commandID, replaces, body.Text, "", asked)
	}
	if err != nil {
		if refused := branchRefused(err); refused != nil {
			return nil, refused
		}
		return nil, refusedResponse(err)
	}
	if branched.ResponseID == "" {
		// A retry of a request already taken, or a turn with no name: answered as a duplicate
		// request id is, with no id.
		return &createResponseResponse{Body: AgentResponse{SessionId: found.ID(), Status: AgentResponseStatusRunning,
			CreatedAt: found.CreatedAt()}}, nil
	}

	answer := AgentResponse{Id: branched.ResponseID, SessionId: found.ID(), Status: AgentResponseStatusRunning,
		CreatedAt: time.Now().UTC()}
	if said != "" {
		answer.Said = &said
	}
	if branched.ParentResponseID != "" {
		answer.ParentResponseId = &branched.ParentResponseID
	}
	if branched.Count > 1 {
		answer.Branch = &ResponseBranch{Of: branched.Of, Index: branched.Index, Count: branched.Count}
	}
	if branched.Rewound {
		answer.RewoundTo = &RewoundTo{MessageId: branched.RewoundTo.MessageID, Inclusive: branched.RewoundTo.Inclusive}
	}
	return &createResponseResponse{Body: answer}, nil
}

// refusedResponse is the answer to a question its session refused before the turn started.
func refusedResponse(err error) error {
	switch {
	case errors.Is(err, session.ErrSwitchUnfinished):
		return errSwitchUnfinished
	case errors.Is(err, conversation.ErrCommandConflict):
		return conflict(err.Error())
	case errors.Is(err, agent.ErrCannotSeeImages):
		return notConfigured(agent.ErrCannotSeeImages.Error())
	case errors.Is(err, agent.ErrNoRoomForDocument):
		return documentTooLarge("the conversation has no room left for the file: another was sent at the same time")
	default:
		return invalidRequest(err.Error())
	}
}

// listResponses returns a session's turns, oldest first.
func (s *Server) listResponses(ctx context.Context, request *listResponsesRequest) (*listResponsesResponse, error) {
	found, failure := s.storedOrLiveSession(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	if s.store == nil {
		return nil, errNoStore
	}

	after, err := decodeCursor[store.ResponsePosition](request.Cursor.ptr())
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	limit := store.SessionLimit(value(request.Limit.ptr()))
	rows, err := s.store.SessionResponses(ctx, OwnerFrom(ctx).CustomerID, found.ID(), limit, after)
	if err != nil {
		return nil, err
	}

	rows, more := page(rows, limit)
	listed := AgentResponsePage{Items: make([]AgentResponse, 0, len(rows)), HasMore: more}
	for _, row := range rows {
		listed.Items = append(listed.Items, responseOf(row))
	}
	if more {
		last := rows[len(rows)-1]
		listed.NextCursor = encodeCursor(store.ResponsePosition{CreatedAt: last.CreatedAt, ID: last.ID})
	}
	return &listResponsesResponse{Body: listed}, nil
}

// listResponseItems returns what the agent did, in the order it happened.
//
// Flat across turns rather than a list per turn, because that is how a conversation reads
// and how it gets rendered: the question, what the agent did about it, what it said, then
// the next question. Naming a response narrows it to that one turn.
func (s *Server) listResponseItems(ctx context.Context, request *listResponseItemsRequest) (*listResponseItemsResponse, error) {
	found, failure := s.storedOrLiveSession(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	if s.store == nil {
		return nil, errNoStore
	}

	after, err := decodeCursor[store.ItemPosition](request.Cursor.ptr())
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	limit := store.ItemLimit(value(request.Limit.ptr()))
	rows, err := s.store.SessionItems(ctx, OwnerFrom(ctx).CustomerID, found.ID(),
		value(request.ResponseId.ptr()), limit, after)
	if err != nil {
		return nil, err
	}

	rows, more := page(rows, limit)
	listed := AgentResponseItemPage{Items: make([]AgentResponseItem, 0, len(rows)), HasMore: more}
	for _, row := range rows {
		listed.Items = append(listed.Items, itemOf(row))
	}
	if more {
		last := rows[len(rows)-1]
		listed.NextCursor = encodeCursor(store.ItemPosition{
			At: last.At, ResponseID: last.ResponseID, Ordinal: last.Ordinal,
		})
	}
	return &listResponseItemsResponse{Body: listed}, nil
}

// responseOf renders one turn.
func responseOf(row store.AgentResponse) AgentResponse {
	rendered := AgentResponse{
		Id: row.ID, SessionId: row.SessionID,
		Status: AgentResponseStatus(row.Status), CreatedAt: row.CreatedAt,
		FinishedAt: row.FinishedAt,
	}
	if row.ParentResponseID != "" {
		rendered.ParentResponseId = &row.ParentResponseID
	}
	if row.Said != "" {
		rendered.Said = &row.Said
	}
	if row.Error != "" {
		rendered.Error = &row.Error
	}
	return rendered
}

// itemOf renders one thing the agent did.
func itemOf(row store.AgentResponseItem) AgentResponseItem {
	rendered := AgentResponseItem{
		ResponseId: row.ResponseID, Ordinal: row.Ordinal,
		Kind: AgentResponseItemKind(row.Kind), At: row.At,
	}
	if row.SessionID != "" {
		rendered.SessionId = &row.SessionID
	}
	if row.Text != "" {
		rendered.Text = &row.Text
	}
	if row.ToolName != "" {
		rendered.ToolName = &row.ToolName
	}
	if len(row.Payload) > 0 {
		payload := row.Payload
		rendered.Payload = &payload
	}
	return rendered
}

// registerResponses declares the operations served in responses.go.
func (s *Server) registerResponses(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listResponses",
		Method:      http.MethodGet,
		Path:        "/v1/agents/sessions/{id}/responses",
		Summary:     "The turns the agent took in a session",
		Description: "Oldest first, which read in order are the conversation. This is the shape of it rather " +
			"than the text: what was asked, whether the turn finished, and how long it took. The " +
			"items endpoint is what carries what happened inside each one.",
		Extensions: map[string]any{clientAccessibleExtension: true},
		Responses: map[string]*huma.Response{
			"200": {Description: "The session's turns, oldest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listResponses)
	huma.Register(api, huma.Operation{
		OperationID:     "createResponse",
		Method:          http.MethodPost,
		Path:            "/v1/agents/sessions/{id}/responses",
		MaxBodyBytes:    maxInlineAudioBody,
		BodyReadTimeout: uploadReadTimeout,
		Summary:         "Ask the agent something and get a handle on the answer",
		Description: "The same thing respond does, with an id back. That is the whole difference and the " +
			"reason this exists: respond returns nothing, so a caller that wants to follow one " +
			"particular turn has to watch the socket and guess which events belong to it. With an id " +
			"it can ask for that turn's items instead.\n" +
			"It returns as soon as the turn has started, not when it has finished. A model takes " +
			"seconds and a request that waited them out would time out on anything long enough to be " +
			"worth asking.\n" +
			"A voice note in audio is transcribed before the turn starts, and what was said is the " +
			"question, after any text. A voice note that cannot be heard is refused and no turn starts.\n" +
			"With replaces, in a conversation kept in Stream Chat, it regenerates or edits an earlier " +
			"response instead of asking something new. No text answers the replaced response's question " +
			"again, beside its reply; text asks it in place of that question. Chat hides the conversation " +
			"from there on, which is kept as a branch (listResponseSiblings, getResponseBranch, " +
			"activateResponse), and the answer says where the channel was rewound to (rewound_to), so the " +
			"acting device prunes its view at once. A request_id is required, and images, audio, videos " +
			"and ui_action cannot go with replaces. A conversation has at most 100 edit and regenerate " +
			"points; past them it is a 409 branch_limit_reached.",
		Extensions:    map[string]any{clientAccessibleExtension: true},
		DefaultStatus: http.StatusAccepted,
		Responses: map[string]*huma.Response{
			"202": {Description: "The agent is answering"},
			"409": errorResponse("The request id was already accepted with different content, or a regenerate or an edit found the conversation changed or out of room"),
			"503": errorResponse("A regenerate or an edit found the conversation's queued messages cannot be written to Stream Chat now"),
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.createResponse)
	huma.Register(api, huma.Operation{
		OperationID: "listResponseItems",
		Method:      http.MethodGet,
		Path:        "/v1/agents/sessions/{id}/responses/items",
		Summary:     "What the agent did, turn by turn, in the order it happened",
		Description: "One flat stream across every turn rather than a list per turn, because that is how a " +
			"conversation reads and how it is rendered: the question, what the agent did about it, " +
			"what it said, then the next question. Naming a response narrows it to that turn.\n" +
			"Deltas are not here. A hundred fragments of one sentence are the sentence, and keeping " +
			"them would make this mostly punctuation; a caller watching a turn happen reads the " +
			"deltas off the events socket, and a caller reading one back wants the shape of it.\n" +
			"Nothing is returned for an incognito session, which has no items to return.",
		Extensions: map[string]any{clientAccessibleExtension: true},
		Responses: map[string]*huma.Response{
			"200": {Description: "The items, oldest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listResponseItems)
}

type listResponsesRequest struct {
	Id     string                `path:"id" doc:"The session, as returned when it was created."`
	Limit  optionalParam[int]    `query:"limit" doc:"Up to 200. Omitted is 25." minimum:"1" maximum:"200"`
	Cursor optionalParam[string] "query:\"cursor\" doc:\"The `next_cursor` of the previous page, sent with the same filters. Omitted is the first page.\""
}

type listResponsesResponse struct {
	Body AgentResponsePage
}

type createResponseRequest struct {
	Id   string                 `path:"id" doc:"The session, as returned when it was created."`
	Body *CreateResponseRequest `required:"true"`
}

type createResponseResponse struct {
	Body AgentResponse
}

type listResponseItemsRequest struct {
	Id         string                `path:"id" doc:"The session, as returned when it was created."`
	ResponseId optionalParam[string] `query:"response_id" doc:"Narrow to one turn's items. Omitted is every turn in the session."`
	Limit      optionalParam[int]    `query:"limit" doc:"Up to 1000. Omitted is 200." minimum:"1" maximum:"1000"`
	Cursor     optionalParam[string] "query:\"cursor\" doc:\"The `next_cursor` of the previous page, sent with the same filters. Omitted is the first page.\""
}

type listResponseItemsResponse struct {
	Body AgentResponseItemPage
}

// AgentResponse is the AgentResponse schema.
type AgentResponse struct {
	CreatedAt        time.Time           `json:"created_at"`
	Error            *string             `json:"error,omitempty"`
	FinishedAt       *time.Time          `json:"finished_at,omitempty"`
	Id               string              `json:"id"`
	ParentResponseId *string             `json:"parent_response_id,omitempty" doc:"The response this one was answered after, absent for a session's first."`
	Branch           *ResponseBranch     `json:"branch,omitempty" doc:"Where a regenerate or an edit stands among its alternatives, on the answer to one. Absent when it is the only one."`
	RewoundTo        *RewoundTo          `json:"rewound_to,omitempty" doc:"Where a regenerate or an edit rewound the conversation's channel to, on the answer to one that hid something: the acting device drops that message, when inclusive, and every message after it."`
	Said             *string             `json:"said,omitempty" doc:"What the person asked, which is the first item of every response."`
	SessionId        string              `json:"session_id"`
	Status           AgentResponseStatus `json:"status" doc:"cancelled is a turn the caller interrupted, which is a different thing from one that failed: nothing went wrong, and what had already been said still counts. refused is a turn the prompt injection screen refused: a rewind or a fork carries what it asked forward as withheld." enum:"running,completed,failed,cancelled,refused"`
}

// AgentResponseStatus is the AgentResponseStatus schema.
type AgentResponseStatus string

// Defines values for AgentResponseStatus.
const (
	AgentResponseStatusCancelled AgentResponseStatus = "cancelled"
	AgentResponseStatusCompleted AgentResponseStatus = "completed"
	AgentResponseStatusFailed    AgentResponseStatus = "failed"
	AgentResponseStatusRefused   AgentResponseStatus = "refused"
	AgentResponseStatusRunning   AgentResponseStatus = "running"
)

// Valid indicates whether the value is a known member of the AgentResponseStatus enum.
func (e AgentResponseStatus) Valid() bool {
	switch e {
	case AgentResponseStatusCancelled:
		return true
	case AgentResponseStatusCompleted:
		return true
	case AgentResponseStatusFailed:
		return true
	case AgentResponseStatusRefused:
		return true
	case AgentResponseStatusRunning:
		return true
	default:
		return false
	}
}

// AgentResponseItem is the AgentResponseItem schema.
type AgentResponseItem struct {
	At         time.Time               `json:"at"`
	Kind       AgentResponseItemKind   `json:"kind" enum:"said,thought,tool_call,tool_result,answer,blocked,error"`
	Ordinal    int                     `json:"ordinal" doc:"The position within the response, assigned by the writer rather than by the database, so items keep the order they happened in."`
	Payload    *map[string]interface{} `json:"payload,omitempty" doc:"Whatever the kind carries that text cannot: a tool's arguments, a guardrail's reason, the id that ties a call to its result. A tool item whose output a tool output retention took out has redacted: true, no arguments and no text, and error: redacted in place of an error's wording."`
	ResponseId string                  `json:"response_id"`
	SessionId  *string                 `json:"session_id,omitempty"`
	Text       *string                 `json:"text,omitempty"`
	ToolName   *string                 `json:"tool_name,omitempty"`
}

func (*AgentResponseItem) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["ordinal"].Format = ""
	return schema
}

// AgentResponseItemKind is the AgentResponseItemKind schema.
type AgentResponseItemKind string

// Defines values for AgentResponseItemKind.
const (
	AgentResponseItemKindAnswer     AgentResponseItemKind = "answer"
	AgentResponseItemKindBlocked    AgentResponseItemKind = "blocked"
	AgentResponseItemKindError      AgentResponseItemKind = "error"
	AgentResponseItemKindSaid       AgentResponseItemKind = "said"
	AgentResponseItemKindThought    AgentResponseItemKind = "thought"
	AgentResponseItemKindToolCall   AgentResponseItemKind = "tool_call"
	AgentResponseItemKindToolResult AgentResponseItemKind = "tool_result"
)

// Valid indicates whether the value is a known member of the AgentResponseItemKind enum.
func (e AgentResponseItemKind) Valid() bool {
	switch e {
	case AgentResponseItemKindAnswer:
		return true
	case AgentResponseItemKindBlocked:
		return true
	case AgentResponseItemKindError:
		return true
	case AgentResponseItemKindSaid:
		return true
	case AgentResponseItemKindThought:
		return true
	case AgentResponseItemKindToolCall:
		return true
	case AgentResponseItemKindToolResult:
		return true
	default:
		return false
	}
}

// AgentResponseItemPage is the AgentResponseItemPage schema.
type AgentResponseItemPage struct {
	HasMore    bool                `json:"has_more"`
	Items      []AgentResponseItem `json:"items" nullable:"false"`
	NextCursor *string             "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}

// AgentResponsePage is the AgentResponsePage schema.
type AgentResponsePage struct {
	HasMore    bool            `json:"has_more"`
	Items      []AgentResponse `json:"items" nullable:"false"`
	NextCursor *string         "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}

// CreateResponseRequest is the CreateResponseRequest schema.
type CreateResponseRequest struct {
	RequestId       *string         `json:"request_id,omitempty" doc:"Generated and sent by the SDKs, one per question, so a retry of the same question is answered once. Required for personal persistent text conversations, and ignored by a session not kept in Stream Chat. With one, images are data URIs, at most 8, uploaded to the conversation's channel and shown on the person's message, a voice note is uploaded and shown there too, and the frames of videos are shown to the model alone. A retry with the same id, text, images and voice note starts no second turn, returns no id and does not fetch its videos again: videos are not part of what a retry is matched on. On a session a dispatch worker opened for a message handed out from a channel, that message is already there: nothing is written or uploaded for it, and a reused id is answered as a duplicate even with other text or images." pattern:"^[A-Za-z0-9_-]{1,128}$"`
	Files           *[]FileSource   `json:"files,omitempty" doc:"A file to answer questions about: a PDF or a Word document. The router reads it as text before the turn starts, and the agent answers from it on this turn and on every turn of the conversation after it, including a conversation carried on later or forked by the same person. In a conversation kept in Stream Chat the person's message shows the file. At most 100 pages, and at most 120,000 characters of text together with the files the conversation already holds, or a quarter of what the model reads at once. Only in a person's own conversation: refused in a thread, in a channel others write in, in a session a worker opened for a channel's message, and by a speech-to-speech session. Not with a voice note. Refused before any turn starts when it cannot be read, is too large, or reads as a prompt injection under the organization's policy." maxItems:"1"`
	Audio           *AudioSource    `json:"audio,omitempty" doc:"A voice note to answer. The router transcribes it and answers what was said, after text when both are sent; text may then be empty. In a conversation kept in Stream Chat the person's message carries the transcript and shows the voice note. Refused before any turn starts when it is not audio, has no words in it, or the deployment transcribes nothing in the agent's region."`
	Images          *[]ImageSource  `json:"images,omitempty"`
	Replaces        *string         `json:"replaces,omitempty" doc:"An earlier response of a conversation kept in Stream Chat to regenerate or edit. With no text its question is answered again; with text, the text is asked in its place. Requires request_id, and takes no images, audio, videos, files or ui_action."`
	MaxOutputTokens *int            `json:"max_output_tokens,omitempty" doc:"Caps this answer, reasoning included, and the replies that finish it after its tools. Omitted keeps the session's." minimum:"1"`
	ReasoningEffort *string         `json:"reasoning_effort,omitempty" doc:"How long the model may think before this answer, and the replies that finish it after its tools. One of the efforts the session's model accepts, such as none, minimal, low, medium, high or max; any other is a 400. Omitted keeps the session's."`
	Text            string          `json:"text" doc:"What to answer, as though it had been said. May be empty when images or audio carry the question. Images with no text are shown on the person's message in a session kept in Stream Chat, so they are then data URIs, at most 8, with or without a request_id."`
	UiAction        *map[string]any `json:"ui_action,omitempty" doc:"What the person did in the generated UI of an earlier reply, such as tapping a place or submitting a form, which the text says in their words: an object naming the reply (message_id), the UI's format and the action (name), with what it carries (payload), of at most 8 KB as JSON. Their message carries it as custom.ai_ui_action, and the model is told it after the text."`
	Videos          *[]VideoSource  `json:"videos,omitempty" doc:"Recorded clips to show the agent. The router samples evenly spaced frames from each and hands them to the vision skill with their timestamps, which is how every vision model is shown a video, since none of the ones routed here take one whole." maxItems:"2"`
}

func (*CreateResponseRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["max_output_tokens"].Format = ""
	return schema
}

// ItemLimit is the ItemLimit schema.
type ItemLimit = int
