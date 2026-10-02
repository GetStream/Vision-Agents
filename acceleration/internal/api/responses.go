package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/video"
	"github.com/danielgtaylor/huma/v2"
)

// noStore is what the response paths say where nothing was written down. A turn's items are
// read back from Postgres, so a deployment without one can start a response but has nothing
// to show afterwards.
const noStore = "this deployment does not record what sessions said"

// maxVideos bounds the clips on one response, so that at video.MaxFrames each a turn stays
// inside the hundred images the strictest vision provider takes in one request.
const maxVideos = 2

// createResponse asks the agent something and names the turn it answers as.
type AgentResponsePage struct {
	Items      []AgentResponse `json:"items"`
	HasMore    bool            `json:"has_more"`
	NextCursor *string         "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}

type AgentResponseStatus string

const (
	AgentResponseStatusCancelled AgentResponseStatus = "cancelled"
	AgentResponseStatusCompleted AgentResponseStatus = "completed"
	AgentResponseStatusFailed    AgentResponseStatus = "failed"
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
	case AgentResponseStatusRunning:
		return true
	default:
		return false
	}
}

type AgentResponse struct {
	Id         string              `json:"id"`
	SessionId  string              `json:"session_id"`
	Said       *string             `json:"said,omitempty" doc:"What the person asked, which is the first item of every response."`
	Status     AgentResponseStatus `json:"status" doc:"cancelled is a turn the caller interrupted, which is a different thing from one that failed: nothing went wrong, and what had already been said still counts." enum:"running,completed,failed,cancelled"`
	Error      *string             `json:"error,omitempty"`
	CreatedAt  time.Time           `json:"created_at"`
	FinishedAt *time.Time          `json:"finished_at,omitempty"`
}

type CreateResponseRequest struct {
	Text      string         `json:"text" doc:"What to answer, as though it had been said."`
	Images    *[]ImageSource `json:"images,omitempty"`
	Videos    *[]VideoSource `json:"videos,omitempty" doc:"Recorded clips to show the agent. The router samples evenly spaced frames from each and hands them to the vision skill with their timestamps, which is how every vision model is shown a video, since none of the ones routed here take one whole." maxItems:"2"`
	CommandId *string        `json:"command_id,omitempty" doc:"Required for personal persistent text conversations, and text only. Reuse this ID and identical text for retries; a retry starts no second turn and returns no id." pattern:"^[A-Za-z0-9_-]{1,128}$"`
}

type ImageSourceDetail string

const (
	ImageSourceDetailAuto ImageSourceDetail = "auto"
	ImageSourceDetailHigh ImageSourceDetail = "high"
	ImageSourceDetailLow  ImageSourceDetail = "low"
)

// Valid indicates whether the value is a known member of the ImageSourceDetail enum.
func (e ImageSourceDetail) Valid() bool {
	switch e {
	case ImageSourceDetailAuto:
		return true
	case ImageSourceDetailHigh:
		return true
	case ImageSourceDetailLow:
		return true
	default:
		return false
	}
}

type ImageSource struct {
	Url    string             `json:"url" doc:"Absolute HTTP(S) URL or base64 image data URI."`
	Detail *ImageSourceDetail `json:"detail,omitempty" enum:"auto,low,high"`
}

type VideoSource struct {
	Url       string `json:"url" doc:"Public HTTPS URL or base64 video data URI, such as data:video/mp4;base64,.... At most 50 MB either way. The router fetches a URL itself, and refuses one that resolves to a private or loopback address."`
	MaxFrames *int   `json:"max_frames,omitempty" doc:"How many frames to sample, evenly spaced across the clip. Default 8." minimum:"1" maximum:"32"`
}

type AgentResponseItemPage struct {
	Items      []AgentResponseItem `json:"items"`
	HasMore    bool                `json:"has_more"`
	NextCursor *string             "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}

type AgentResponseItemKind string

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

type AgentResponseItem struct {
	ResponseId string                  `json:"response_id"`
	Ordinal    int                     `json:"ordinal" doc:"The position within the response, assigned by the writer rather than by the database, so items keep the order they happened in."`
	SessionId  *string                 `json:"session_id,omitempty"`
	Kind       AgentResponseItemKind   `json:"kind" enum:"said,thought,tool_call,tool_result,answer,blocked,error"`
	Text       *string                 `json:"text,omitempty"`
	ToolName   *string                 `json:"tool_name,omitempty"`
	Payload    *map[string]interface{} `json:"payload,omitempty" doc:"Whatever the kind carries that text cannot: a tool's arguments, a guardrail's reason, the id that ties a call to its result."`
	At         time.Time               `json:"at"`
}

type listResponsesRequest struct {
	ID     string                `path:"id" doc:"The session, as returned when it was created."`
	Limit  optionalParam[int]    `query:"limit" doc:"Up to 200. Omitted is 25." minimum:"1" maximum:"200"`
	Cursor optionalParam[string] "query:\"cursor\" doc:\"The `next_cursor` of the previous page, sent with the same filters. Omitted is the first page.\""
}

type agentResponsePageResponse struct {
	Body AgentResponsePage
}

type createResponseRequest struct {
	ID   string `path:"id" doc:"The session, as returned when it was created."`
	Body CreateResponseRequest
}

type agentResponseResponse struct {
	Body AgentResponse
}

type listResponseItemsRequest struct {
	ID         string                `path:"id" doc:"The session, as returned when it was created."`
	ResponseID optionalParam[string] `query:"response_id" doc:"Narrow to one turn's items. Omitted is every turn in the session."`
	Limit      optionalParam[int]    `query:"limit" doc:"Up to 1000. Omitted is 200." minimum:"1" maximum:"1000"`
	Cursor     optionalParam[string] "query:\"cursor\" doc:\"The `next_cursor` of the previous page, sent with the same filters. Omitted is the first page.\""
}

type agentResponseItemPageResponse struct {
	Body AgentResponseItemPage
}

func (s *Server) registerResponses(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listResponses",
		Method:      http.MethodGet,
		Path:        "/v1/agents/sessions/{id}/responses",
		Summary:     "The turns the agent took in a session",
		Description: "Oldest first, which read in order are the conversation. This is the shape of it " +
			"rather than the text: what was asked, whether the turn finished, and how long " +
			"it took. The items endpoint is what carries what happened inside each one.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The session's turns, oldest first"},
		},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.listResponses)
	huma.Register(api, huma.Operation{
		OperationID: "createResponse",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions/{id}/responses",
		Summary:     "Ask the agent something and get a handle on the answer",
		Description: "The same thing respond does, with an id back. That is the whole difference and " +
			"the reason this exists: respond returns nothing, so a caller that wants to " +
			"follow one particular turn has to watch the socket and guess which events " +
			"belong to it. With an id it can ask for that turn's items instead.\n" +
			"It returns as soon as the turn has started, not when it has finished. A model " +
			"takes seconds and a request that waited them out would time out on anything " +
			"long enough to be worth asking.",
		DefaultStatus: http.StatusAccepted,
		Responses: map[string]*huma.Response{
			"202": {Description: "The agent is answering"},
			"409": errorResponse("The command ID was already accepted with different content"),
		},
		Errors:       []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
		Extensions:   map[string]any{clientAccessibleExtension: true},
		MaxBodyBytes: largeBody,
	}, s.createResponse)
	huma.Register(api, huma.Operation{
		OperationID: "listResponseItems",
		Method:      http.MethodGet,
		Path:        "/v1/agents/sessions/{id}/responses/items",
		Summary:     "What the agent did, turn by turn, in the order it happened",
		Description: "One flat stream across every turn rather than a list per turn, because that is " +
			"how a conversation reads and how it is rendered: the question, what the agent " +
			"did about it, what it said, then the next question. Naming a response narrows " +
			"it to that turn.\n" +
			"Deltas are not here. A hundred fragments of one sentence are the sentence, and " +
			"keeping them would make this mostly punctuation; a caller watching a turn " +
			"happen reads the deltas off the events socket, and a caller reading one back " +
			"wants the shape of it.\n" +
			"Nothing is returned for an incognito session, which has no items to return.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The items, oldest first"},
		},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.listResponseItems)
}

// It is respond with an id back, which is the whole of the difference. Without one a caller
// following a particular answer has to watch every event on the socket and work out which
// belong to the turn it asked for; with one it can ask for that turn's items instead.
func (s *Server) createResponse(ctx context.Context, request *createResponseRequest) (*agentResponseResponse, error) {
	// A response is asked of a running session rather than a stored one: a conversation that
	// ended can be read and forked, but not talked to.
	found, failure := s.session(ctx, request.ID)
	if failure != nil {
		if failure.status == unauthorized {
			return nil, huma.Error401Unauthorized(missingCustomer().Error)
		}
		return nil, huma.Error404NotFound(failure.message)
	}
	if request.Body.Text == "" {
		return nil, huma.Error400BadRequest("there is nothing to answer")
	}

	images := make([]wireImage, 0, len(value(request.Body.Images)))
	for _, sent := range value(request.Body.Images) {
		images = append(images, wireImage{URL: sent.Url, Detail: string(value(sent.Detail))})
	}
	parts, err := imagesFromWire(images)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	videos := value(request.Body.Videos)
	if len(videos) > maxVideos {
		return nil, huma.Error400BadRequest(fmt.Sprintf("at most %d videos go with one response", maxVideos))
	}
	if len(videos) > 0 && value(request.Body.CommandId) != "" {
		return nil, huma.Error400BadRequest("a command ID carries text only")
	}
	for index, sent := range videos {
		frames, length, err := video.Frames(ctx, sent.Url, value(sent.MaxFrames))
		if err != nil {
			return nil, huma.Error400BadRequest(err.Error())
		}
		for _, frame := range frames {
			frame.Image.Caption = fmt.Sprintf("video %d, %.1fs of %.1fs", index+1, frame.At.Seconds(), length.Seconds())
			parts = append(parts, frame.Image)
		}
	}

	var responseID string
	if id := value(request.Body.CommandId); id != "" {
		if len(parts) > 0 {
			return nil, huma.Error400BadRequest("a command ID carries text only")
		}
		_, responseID, err = found.RespondCommand(ctx, id, request.Body.Text, "")
		if errors.Is(err, conversation.ErrCommandConflict) {
			return nil, huma.Error409Conflict(err.Error())
		}
	} else {
		responseID, err = found.Respond(ctx, request.Body.Text, parts)
	}
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	if len(videos) > 0 {
		found.SawVideo()
	}
	if responseID == "" {
		// The turn is being answered, it just has no name. A session that records nothing
		// has no row to hand back, and a native one lets the model decide what a turn is.
		// Refusing would be wrong -- the agent is answering -- so the id is empty and the
		// caller reads the socket, which is what it would have done before this existed.
		return &agentResponseResponse{Body: AgentResponse{
			SessionId: found.ID(), Status: AgentResponseStatusRunning,
			CreatedAt: found.CreatedAt(),
		}}, nil
	}

	said := request.Body.Text
	return &agentResponseResponse{Body: AgentResponse{
		Id: responseID, SessionId: found.ID(), Said: &said,
		Status: AgentResponseStatusRunning, CreatedAt: time.Now().UTC(),
	}}, nil
}

// listResponses returns a session's turns, oldest first.
func (s *Server) listResponses(ctx context.Context, request *listResponsesRequest) (*agentResponsePageResponse, error) {
	found, failure := s.storedOrLiveSession(ctx, request.ID)
	if failure != nil {
		if failure.status == unauthorized {
			return nil, huma.Error401Unauthorized(missingCustomer().Error)
		}
		return nil, huma.Error404NotFound(failure.message)
	}
	if s.store == nil {
		return nil, huma.Error404NotFound(noStore)
	}

	after, err := decodeCursor[store.ResponsePosition](request.Cursor.Ptr())
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	limit := store.SessionLimit(request.Limit.Value)
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
	return &agentResponsePageResponse{Body: listed}, nil
}

// listResponseItems returns what the agent did, in the order it happened.
//
// Flat across turns rather than a list per turn, because that is how a conversation reads
// and how it gets rendered: the question, what the agent did about it, what it said, then
// the next question. Naming a response narrows it to that one turn.
func (s *Server) listResponseItems(ctx context.Context, request *listResponseItemsRequest) (*agentResponseItemPageResponse, error) {
	found, failure := s.storedOrLiveSession(ctx, request.ID)
	if failure != nil {
		if failure.status == unauthorized {
			return nil, huma.Error401Unauthorized(missingCustomer().Error)
		}
		return nil, huma.Error404NotFound(failure.message)
	}
	if s.store == nil {
		return nil, huma.Error404NotFound(noStore)
	}

	after, err := decodeCursor[store.ItemPosition](request.Cursor.Ptr())
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	limit := store.ItemLimit(request.Limit.Value)
	rows, err := s.store.SessionItems(ctx, OwnerFrom(ctx).CustomerID, found.ID(),
		request.ResponseID.Value, limit, after)
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
	return &agentResponseItemPageResponse{Body: listed}, nil
}

// responseOf renders one turn.
func responseOf(row store.AgentResponse) AgentResponse {
	rendered := AgentResponse{
		Id: row.ID, SessionId: row.SessionID,
		Status: AgentResponseStatus(row.Status), CreatedAt: row.CreatedAt,
		FinishedAt: row.FinishedAt,
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
