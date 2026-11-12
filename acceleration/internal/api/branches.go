package api

import (
	"cmp"
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Sibling and branch pages: a response has a handful of alternatives and a branch at most
// what an activation sends again, so both are read whole and cut here.
const (
	defaultSiblingLimit = 25
	maxSiblingLimit     = 100
	defaultBranchLimit  = 100
	maxBranchLimit      = 200
)

// errBranchLimit is a regenerate or an edit refused because the conversation has as many
// edit and regenerate points as Chat keeps for its channel.
var errBranchLimit = APIError{Type: ErrorTypeConflict, Code: codeBranchLimitReached,
	Message: "this conversation has reached its limit of 100 edit and regenerate points; continue in a new chat"}

// errBranchTooLong is an activation that would send more messages again than one may.
var errBranchTooLong = APIError{Type: ErrorTypeConflict, Code: codeBranchTooLong,
	Message: "this branch is too long to bring back: it is over 100 messages; continue in a new chat"}

// errBranchGone is a hidden branch whose copies were deleted.
var errBranchGone = APIError{Type: ErrorTypeGone, Code: codeBranchUnavailable,
	Message: "this branch is no longer kept, so it can be neither read nor brought back"}

// errSwitchUnfinished is a turn refused while a regenerate, an edit or an activation the
// session began is left unfinished after Chat may have rewound.
var errSwitchUnfinished = conflict("the conversation's last regenerate, edit or branch switch did not finish; retry that request, or reopen the conversation")

// errUnknownResponse is a response the session does not have.
var errUnknownResponse = notFound("no such response in this session")

// unsupportedBranches says why a conversation cannot have branches, by the session's reason.
var unsupportedBranches = map[string]string{
	conversation.UnsupportedThread:     "a conversation in a thread cannot be regenerated or edited",
	conversation.UnsupportedDispatched: "a conversation a dispatch worker answers cannot be regenerated or edited",
	conversation.UnsupportedChannel:    "only a session's own conversation can be regenerated or edited, not one in another channel",
	conversation.UnsupportedShared:     "a conversation shared with the members of its channel cannot be regenerated or edited",
	conversation.UnsupportedNotKept:    "only a conversation kept in Stream Chat can be regenerated or edited; an incognito session keeps none",
	conversation.UnsupportedCall:       "a session on a call cannot be regenerated or edited while it is on the call",
	conversation.UnsupportedNative:     "a session on a speech-to-speech model cannot be regenerated or edited: the model keeps its own context",
	conversation.UnsupportedUnmarked:   "this response was answered before responses could be regenerated or edited, so its messages cannot be found",
}

// branchRefused is the answer to a regenerate, an edit or an activation the session refused,
// nil for an error that is none of those.
func branchRefused(err error) error {
	if unsupported, ok := errors.AsType[*conversation.BranchesUnsupportedError](err); ok {
		message, known := unsupportedBranches[unsupported.Reason]
		if !known {
			message = unsupported.Error()
		}
		return invalidRequest(message)
	}
	switch {
	case errors.Is(err, session.ErrSwitchUnfinished):
		return errSwitchUnfinished
	case errors.Is(err, conversation.ErrConversationUnwritable):
		return unavailable("the conversation's earlier messages cannot be written to Stream Chat now, so it cannot be switched; try again shortly")
	case errors.Is(err, conversation.ErrBranchLimit):
		return errBranchLimit
	case errors.Is(err, conversation.ErrBranchTooLong):
		return errBranchTooLong
	case errors.Is(err, session.ErrBranchGone):
		return errBranchGone
	case errors.Is(err, conversation.ErrBranchUnavailable):
		return notConfigured("regenerate and edit are not enabled for this app: its Stream Chat app needs the agent rewind feature")
	case errors.Is(err, conversation.ErrBranchConflict):
		return conflict("the conversation changed while it was being switched; read it again and retry")
	case errors.Is(err, conversation.ErrCommandConflict):
		return conflict(conversation.ErrCommandConflict.Error())
	case errors.Is(err, conversation.ErrCommandNotFound):
		// The conversation is not the caller's: they are told it does not exist.
		return errUnknownSession
	case errors.Is(err, store.ErrUnknownResponse):
		return errUnknownResponse
	}
	return nil
}

// listResponseSiblings returns the alternatives to one response: the other answers to its
// question, or the other questions asked in its place.
func (s *Server) listResponseSiblings(ctx context.Context, request *listResponseSiblingsRequest) (*listResponseSiblingsResponse, error) {
	found, failure := s.branchedSession(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	of := value(request.Of.ptr())
	if of == "" {
		of = store.SiblingsOfReply
	}
	after, err := decodeCursor[store.ResponsePosition](request.Cursor.ptr())
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	customer := OwnerFrom(ctx).CustomerID
	siblings, err := s.store.Siblings(ctx, customer, found.ID(), request.ResponseId, of)
	if errors.Is(err, store.ErrUnknownResponse) {
		return nil, errUnknownResponse
	}
	if err != nil {
		return nil, err
	}

	start := 0
	if after != nil {
		start = len(siblings)
		for i, sibling := range siblings {
			if positionAfter(sibling.CreatedAt, sibling.ID, after.CreatedAt, after.ID) {
				start = i
				break
			}
		}
	}
	rows, more := page(siblings[start:], clampLimit(value(request.Limit.ptr()), defaultSiblingLimit, maxSiblingLimit))
	listed := ResponseSiblingPage{Items: make([]ResponseSibling, 0, len(rows)), HasMore: more}
	for i, row := range rows {
		active, err := s.siblingActive(ctx, customer, found.ID(), row, of)
		if err != nil {
			return nil, err
		}
		sibling := ResponseSibling{Id: row.ID, CreatedAt: row.CreatedAt, Active: active, Index: start + i + 1}
		if row.Said != "" {
			sibling.Said = &row.Said
		}
		listed.Items = append(listed.Items, sibling)
	}
	if more {
		last := rows[len(rows)-1]
		listed.NextCursor = encodeCursor(store.ResponsePosition{CreatedAt: last.CreatedAt, ID: last.ID})
	}
	return &listResponseSiblingsResponse{Body: listed}, nil
}

// siblingActive is whether the conversation goes through a sibling: the response itself for
// an answer, and any answer to it for a question, which may have been regenerated since.
func (s *Server) siblingActive(ctx context.Context, customer, sessionID string, sibling store.AgentResponse, of string) (bool, error) {
	if sibling.RewoundAt == nil || of == store.SiblingsOfReply {
		return sibling.RewoundAt == nil, nil
	}
	answers, err := s.store.Siblings(ctx, customer, sessionID, sibling.ID, store.SiblingsOfReply)
	if err != nil {
		return false, err
	}
	return slices.ContainsFunc(answers, func(answer store.AgentResponse) bool { return answer.RewoundAt == nil }), nil
}

// getResponseBranch returns the messages a hidden branch would show, from where it leaves the
// conversation down to its newest response.
func (s *Server) getResponseBranch(ctx context.Context, request *getResponseBranchRequest) (*getResponseBranchResponse, error) {
	found, failure := s.branchedSession(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	after, err := decodeCursor[branchPosition](request.Cursor.ptr())
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	active, shown, err := session.BranchShown(ctx, s.store, OwnerFrom(ctx).CustomerID, found.ID(), request.ResponseId)
	if err != nil {
		if refused := branchRefused(err); refused != nil {
			return nil, refused
		}
		return nil, err
	}
	listed := BranchMessagePage{Active: active, Items: []BranchMessage{}}
	if active {
		return &getResponseBranchResponse{Body: listed}, nil
	}

	start := 0
	if after != nil {
		start = len(shown)
		for i, m := range shown {
			if positionAfter(m.ChatCreatedAt, m.OriginalMessageID, after.At, after.ID) {
				start = i
				break
			}
		}
	}
	rows, more := page(shown[start:], clampLimit(value(request.Limit.ptr()), defaultBranchLimit, maxBranchLimit))
	listed.HasMore = more
	for _, row := range rows {
		listed.Items = append(listed.Items, branchMessageOf(row))
	}
	if more {
		last := rows[len(rows)-1]
		listed.NextCursor = encodeCursor(branchPosition{At: last.ChatCreatedAt, ID: last.OriginalMessageID})
	}
	return &getResponseBranchResponse{Body: listed}, nil
}

// activateResponse brings a hidden branch back onto the conversation.
func (s *Server) activateResponse(ctx context.Context, request *activateResponseRequest) (*activateResponseResponse, error) {
	leaveUncosted(ctx)
	found, sent, failure := s.sessionToAnswer(ctx, request.Id)
	if failure != nil {
		return nil, failure
	}
	defer sent()

	activated, err := found.Activate(ctx, request.ResponseId)
	if err != nil {
		if refused := branchRefused(err); refused != nil {
			return nil, refused
		}
		return nil, err
	}
	answer := ActivatedBranch{LeafResponseId: activated.LeafResponseID, AlreadyActive: activated.AlreadyActive,
		Resent: activated.Resent}
	if !activated.AlreadyActive && activated.RewoundTo.MessageID != "" {
		answer.RewoundTo = &RewoundTo{MessageId: activated.RewoundTo.MessageID, Inclusive: activated.RewoundTo.Inclusive}
	}
	return &activateResponseResponse{Body: answer}, nil
}

// branchedSession is a session whose branches are read: one the caller may read, on a
// deployment that keeps responses.
func (s *Server) branchedSession(ctx context.Context, id string) (session.Found, error) {
	found, failure := s.storedOrLiveSession(ctx, id)
	if failure != nil {
		return session.Found{}, failure
	}
	// A conversation kept in Chat is its owner's alone, as it is to every other operation
	// on a running session.
	if found.Live != nil && !canReadSession(ctx, found.Live.Spec()) {
		return session.Found{}, errUnknownSession
	}
	if s.store == nil {
		return session.Found{}, errNoStore
	}
	return found, nil
}

// positionAfter is whether (at, id) sorts after the cursor's (afterAt, afterID).
func positionAfter(at time.Time, id string, afterAt time.Time, afterID string) bool {
	return cmp.Or(at.Compare(afterAt), strings.Compare(id, afterID)) > 0
}

// clampLimit is the page size for the limit asked for: fallback when none was, at most most.
func clampLimit(asked, fallback, most int) int {
	if asked <= 0 {
		return fallback
	}
	return min(asked, most)
}

// branchPosition is the last message of a branch page.
type branchPosition struct {
	At time.Time `json:"t"`
	ID string    `json:"id"`
}

// branchMessageOf renders a copy of a hidden message as Chat would show it.
func branchMessageOf(row store.AgentBranchMessage) BranchMessage {
	rendered := BranchMessage{Id: row.OriginalMessageID, CreatedAt: row.ChatCreatedAt, UserId: row.AuthorID,
		Attachments: []map[string]any{}, Custom: map[string]any{}}
	rendered.Text, _ = row.Message["text"].(string)
	if custom, ok := row.Message["custom"].(map[string]any); ok {
		rendered.Custom = custom
		runtime, _ := custom["support_runtime"].(map[string]any)
		if role, _ := runtime["role"].(string); role != "" {
			rendered.Role = &role
		}
	}
	if attachments, ok := row.Message["attachments"].([]any); ok {
		for _, attachment := range attachments {
			if typed, ok := attachment.(map[string]any); ok {
				rendered.Attachments = append(rendered.Attachments, typed)
			}
		}
	}
	return rendered
}

// registerBranches declares the operations served in branches.go.
func (s *Server) registerBranches(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listResponseSiblings",
		Method:      http.MethodGet,
		Path:        "/v1/agents/sessions/{id}/responses/{response_id}/siblings",
		Summary:     "The alternatives to one response: its regenerations, or the edits of its question",
		Description: "What a ‹ i/n › control pages through. of=reply lists every answer to the response's " +
			"question, the response's own included; of=question lists the questions asked at the same " +
			"point of the conversation, the original and its edits, each as the first response that " +
			"answered it. Oldest first, which is the order of their index.\n" +
			"active marks the one the conversation goes through now. Reading a sibling's messages is " +
			"getResponseBranch, and making it the conversation is activateResponse. A response answered " +
			"before branches were kept has no siblings but itself.",
		Extensions: map[string]any{clientAccessibleExtension: true},
		Responses: map[string]*huma.Response{
			"200": {Description: "The response's siblings, oldest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listResponseSiblings)
	huma.Register(api, huma.Operation{
		OperationID: "getResponseBranch",
		Method:      http.MethodGet,
		Path:        "/v1/agents/sessions/{id}/responses/{response_id}/branch",
		Summary:     "The messages of a hidden branch, to show in place of the conversation's",
		Description: "A regenerate or an edit hides the conversation from that point on in Stream Chat " +
			"and keeps what it hid as a branch. This reads a branch back: its messages from where it " +
			"leaves the conversation down to the newest response under the one named, oldest first, " +
			"shaped as Chat messages so a client renders them the same way. Nothing is written to Chat " +
			"and the model is not told.\n" +
			"A response on the conversation is shown by the channel itself, so it answers active: " +
			"true and no items. A branch whose copies were deleted, as a memory truncate deletes " +
			"them, is a 410 branch_unavailable.",
		Extensions: map[string]any{clientAccessibleExtension: true},
		Responses: map[string]*huma.Response{
			"200": {Description: "The branch's messages, oldest first"},
			"410": errorResponse("The branch's copies were deleted"),
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getResponseBranch)
	huma.Register(api, huma.Operation{
		OperationID: "activateResponse",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions/{id}/responses/{response_id}/activate",
		Summary:     "Make a hidden branch the conversation again",
		Description: "The conversation from where the branch leaves it is hidden in Stream Chat and kept as " +
			"a branch of its own, and the branch ending at the newest response under the one named is " +
			"sent again: new messages, with new ids, sent with skip_push and carrying " +
			"custom.resent_from, the id of the message each copies. The model answers from that branch " +
			"from then on. Call it before asking anything while a hidden branch is shown.\n" +
			"A branch already on the conversation changes nothing and answers already_active, so a " +
			"retry is safe. rewound_to says where the channel was rewound to, which the acting device " +
			"prunes its own view from: the message itself is gone too when inclusive.\n" +
			"At most 100 messages are sent again; a longer branch is a 409 branch_too_long, and a " +
			"conversation with no room left for another hidden range a 409 branch_limit_reached.",
		Extensions: map[string]any{clientAccessibleExtension: true},
		Responses: map[string]*huma.Response{
			"200": {Description: "The branch is the conversation"},
			"409": errorResponse("The conversation changed under the switch, or the branch is too long or out of room"),
			"410": errorResponse("The branch's copies were deleted"),
			"503": errorResponse("The conversation's queued messages cannot be written to Stream Chat now"),
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.activateResponse)
}

type listResponseSiblingsRequest struct {
	Id         string                `path:"id" doc:"The session, as returned when it was created."`
	ResponseId string                `path:"response_id" doc:"The response whose alternatives to list, itself among them."`
	Of         optionalParam[string] `query:"of" enum:"reply,question" doc:"reply lists the answers to the response's question; question lists the questions asked in its place. Omitted is reply."`
	Limit      optionalParam[int]    `query:"limit" doc:"Up to 100. Omitted is 25." minimum:"1" maximum:"100"`
	Cursor     optionalParam[string] "query:\"cursor\" doc:\"The `next_cursor` of the previous page, sent with the same filters. Omitted is the first page.\""
}

type listResponseSiblingsResponse struct {
	Body ResponseSiblingPage
}

type getResponseBranchRequest struct {
	Id         string                `path:"id" doc:"The session, as returned when it was created."`
	ResponseId string                `path:"response_id" doc:"A response of the branch, as listResponseSiblings names it."`
	Limit      optionalParam[int]    `query:"limit" doc:"Up to 200. Omitted is 100." minimum:"1" maximum:"200"`
	Cursor     optionalParam[string] "query:\"cursor\" doc:\"The `next_cursor` of the previous page. Omitted is the first page.\""
}

type getResponseBranchResponse struct {
	Body BranchMessagePage
}

type activateResponseRequest struct {
	Id         string `path:"id" doc:"The session, as returned when it was created."`
	ResponseId string `path:"response_id" doc:"A response of the branch to bring back. The branch ends at the newest response under it."`
}

type activateResponseResponse struct {
	Body ActivatedBranch
}

// ResponseSibling is one alternative to a response.
type ResponseSibling struct {
	Id        string    `json:"id" doc:"The response, which getResponseBranch reads and activateResponse brings back."`
	Said      *string   `json:"said,omitempty" doc:"The question the response answers."`
	CreatedAt time.Time `json:"created_at"`
	Active    bool      `json:"active" doc:"Whether the conversation goes through this sibling now: for a question, through any answer to it."`
	Index     int       `json:"index" doc:"Its place among the siblings, from 1, oldest first: the i of ‹ i/n ›."`
}

func (*ResponseSibling) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["index"].Format = ""
	return schema
}

// ResponseSiblingPage is a page of a response's siblings.
type ResponseSiblingPage struct {
	HasMore    bool              `json:"has_more"`
	Items      []ResponseSibling `json:"items" nullable:"false"`
	NextCursor *string           "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}

// BranchMessage is a message of a hidden branch, as Stream Chat showed it.
type BranchMessage struct {
	Id          string           `json:"id" doc:"The id the message had in Stream Chat."`
	Text        string           `json:"text"`
	Attachments []map[string]any `json:"attachments" nullable:"false"`
	Custom      map[string]any   `json:"custom" doc:"The message's custom data, as Stream Chat held it, with support_branch saying where it stands among its alternatives now."`
	CreatedAt   time.Time        `json:"created_at"`
	UserId      string           `json:"user_id" doc:"Who wrote it: the person, or the agent's Stream user."`
	Role        *string          `json:"role,omitempty" doc:"user or assistant, for a message the router wrote. Absent for one somebody else wrote in the channel."`
}

// BranchMessagePage is a page of a branch's messages.
type BranchMessagePage struct {
	Active     bool            `json:"active" doc:"The response is on the conversation, which the channel shows: there are no items."`
	HasMore    bool            `json:"has_more"`
	Items      []BranchMessage `json:"items" nullable:"false"`
	NextCursor *string         "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}

// ActivatedBranch is what bringing a branch back did.
type ActivatedBranch struct {
	LeafResponseId string     `json:"leaf_response_id" doc:"The newest response of the branch, which the next question is answered after."`
	AlreadyActive  bool       `json:"already_active" doc:"The branch was already the conversation, and nothing changed."`
	Resent         int        `json:"resent" doc:"How many messages were sent again."`
	RewoundTo      *RewoundTo `json:"rewound_to,omitempty" doc:"Where the channel was rewound to. Absent when nothing changed."`
}

func (*ActivatedBranch) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["resent"].Format = ""
	return schema
}

// RewoundTo is where a switch rewound a conversation's channel to.
type RewoundTo struct {
	MessageId string `json:"message_id" doc:"The message the channel was rewound to. Every message after it is hidden."`
	Inclusive bool   `json:"inclusive" doc:"The message itself is hidden too."`
}

// ResponseBranch is where a response stands among its alternatives.
type ResponseBranch struct {
	Of    string `json:"of" enum:"reply,question" doc:"reply counts the answers to one question, a regenerate's; question the questions asked at one point, an edit's."`
	Index int    `json:"index" doc:"Its place, from 1, oldest first."`
	Count int    `json:"count" doc:"How many there are."`
}

func (*ResponseBranch) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["index"].Format = ""
	schema.Properties["count"].Format = ""
	return schema
}
