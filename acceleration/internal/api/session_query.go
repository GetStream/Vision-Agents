package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SessionQuery is which of the caller's sessions to list, and in what order. It is written
// the way Stream's query endpoints are: a filter of fields, a sort of {field, direction}.
type SessionQuery struct {
	Filter *SessionFilter `json:"filter,omitempty"`
	Sort   []SessionSort  `json:"sort,omitempty" maxItems:"1" doc:"Omitted is updated_at, or relevance for a text search."`
	Limit  int            `json:"limit,omitempty" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 25."`
	Cursor string         `json:"cursor,omitempty" doc:"The next_cursor of the previous page, sent with the same filter and sort. Omitted is the first page."`
}

func (*SessionQuery) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.AdditionalProperties = false
	return schema
}

// SessionFilter narrows a session query. Fields are ANDed.
type SessionFilter struct {
	Text      *TextMatch `json:"text,omitempty" doc:"Full text over the title, description, project and agent name. Sorted by relevance, and not combined with project_id."`
	ProjectID *Equals    `json:"project_id,omitempty"`
	Agent     *Equals    `json:"agent,omitempty" doc:"The agent name the session was opened against."`
	AgentID   *Equals    `json:"agent_id,omitempty" doc:"The agent id the session was created with, which names its transcript channel."`
	UserID    *Equals    `json:"user_id,omitempty" doc:"Whose sessions to list. Only a server-side caller may set it: an end user is narrowed to their own whatever they ask for."`
	Modality  *Equals    `json:"modality,omitempty" doc:"text, voice or video: how the user took part."`
	State     *Equals    `json:"state,omitempty" doc:"live or ended, as each session reports its state."`
}

func (*SessionFilter) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Which sessions to list. A field not listed here is refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

// TextMatch is a full-text match, Stream's $q.
type TextMatch struct {
	Q string `json:"$q" minLength:"1" doc:"Quoted phrases and bare words both work, and punctuation is taken rather than refused."`
}

func (*TextMatch) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.AdditionalProperties = false
	return schema
}

// Equals matches one value exactly, written bare or as {"$eq": value}.
type Equals string

func (e *Equals) UnmarshalJSON(raw []byte) error {
	var bare string
	if json.Unmarshal(raw, &bare) == nil {
		*e = Equals(bare)
		return nil
	}
	var operator struct {
		Eq string `json:"$eq"`
	}
	if err := json.Unmarshal(raw, &operator); err != nil {
		return err
	}
	*e = Equals(operator.Eq)
	return nil
}

func (Equals) Schema(registry huma.Registry) *huma.Schema {
	schema := &huma.Schema{
		Description: `Matches one value exactly: "value" is short for {"$eq": "value"}.`,
		OneOf: []*huma.Schema{
			{Type: huma.TypeString},
			{
				Type:                 huma.TypeObject,
				Properties:           map[string]*huma.Schema{"$eq": {Type: huma.TypeString}},
				Required:             []string{"$eq"},
				AdditionalProperties: false,
			},
		},
	}
	schema.PrecomputeMessages()
	registry.Map()["Equals"] = schema
	return &huma.Schema{Ref: "#/components/schemas/Equals"}
}

// SessionSort is what a session query is ordered by.
type SessionSort struct {
	Field     SessionSortField `json:"field"`
	Direction int              `json:"direction,omitempty" enum:"-1" default:"-1" doc:"-1, descending. Ascending is not offered."`
}

func (*SessionSort) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.AdditionalProperties = false
	return schema
}

// SessionSortField is a key a session query sorts on.
type SessionSortField string

const (
	SessionSortUpdatedAt SessionSortField = "updated_at"
	SessionSortRelevance SessionSortField = "relevance"
)

func (SessionSortField) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "SessionSortField",
		"updated_at is the most recently active first. relevance is the best match first, "+
			"and only sorts a text search.",
		string(SessionSortUpdatedAt), string(SessionSortRelevance))
}

type querySessionsRequest struct {
	Body *SessionQuery
}

type querySessionsResponse struct {
	Body SessionPage
}

func (s *Server) registerSessionQuery(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "querySessions",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions/query",
		Summary:     "List or search the caller's sessions",
		Description: "Three queries are supported, each over the sessions still running and " +
			"the ones that ended:\n\n" +
			"- every session, sorted by `updated_at`\n" +
			"- a text search, `{\"text\": {\"$q\": \"billing\"}}`, sorted by `relevance`\n" +
			"- one project's, `{\"project_id\": \"health\"}`, sorted by `updated_at`\n\n" +
			"`agent`, `agent_id`, `user_id`, `modality` and `state` narrow any of them. A backend gets its customer's " +
			"sessions; an end user gets their own, whatever they ask for, and an anonymous " +
			"caller who named nobody gets none.\n\n" +
			"The search reads what a person named the conversation, not what was said in it. " +
			"There is no total: counting every conversation costs more than the page.",
		Responses: map[string]*huma.Response{
			"200": {Description: "A page of sessions"},
		},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.querySessions)
}

// querySessions lists or searches the caller's sessions.
func (s *Server) querySessions(ctx context.Context, request *querySessionsRequest) (*querySessionsResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.sessions == nil {
		return &querySessionsResponse{Body: SessionPage{Items: []Session{}}}, nil
	}

	var sent SessionQuery
	if request.Body != nil {
		sent = *request.Body
	}
	query, err := sessionQueryOf(ctx, sent)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	var found []session.Found
	if query.text != "" {
		found, err = s.sessions.Search(ctx, OwnerFrom(ctx), query.text, query.filter)
	} else {
		found, err = s.sessions.Query(ctx, OwnerFrom(ctx), query.filter)
	}
	if err != nil {
		return nil, err
	}
	return &querySessionsResponse{Body: sessionPageOf(found, query.filter.Limit, query.sort)}, nil
}

// sessionQuery is a SessionQuery checked and turned into what the manager takes.
type sessionQuery struct {
	text   string
	sort   SessionSortField
	filter store.SessionFilter
}

// sessionCursor is a page position and the sort it belongs to, so a cursor from one order
// is refused by another rather than paging it from the wrong place.
type sessionCursor struct {
	store.SessionPosition
	Sort SessionSortField `json:"s"`
}

// sessionQueryOf refuses the queries that are not one of the three, and a user id from
// anybody but a backend.
//
// The manager narrows an end user to their own sessions again; two checks is the right
// number for something that decides whose conversations a stranger can read.
func sessionQueryOf(ctx context.Context, sent SessionQuery) (sessionQuery, error) {
	var filter SessionFilter
	if sent.Filter != nil {
		filter = *sent.Filter
	}
	query := sessionQuery{sort: SessionSortUpdatedAt, filter: store.SessionFilter{Limit: sent.Limit}}
	if filter.Text != nil {
		query.text = filter.Text.Q
		query.sort = SessionSortRelevance
	}
	query.filter.Project = string(value(filter.ProjectID))
	query.filter.AgentName = string(value(filter.Agent))
	query.filter.AgentID = string(value(filter.AgentID))
	query.filter.Modality = string(value(filter.Modality))
	state := SessionState(value(filter.State))
	switch state {
	case Live:
		query.filter.State = store.SessionRunning
	case Ended:
		query.filter.State = store.SessionClosed
	}

	switch {
	case query.filter.Modality != "" && !SessionModality(query.filter.Modality).Valid():
		return sessionQuery{}, errors.New("modality is text, voice or video")
	case state != "" && !state.Valid():
		return sessionQuery{}, errors.New("state is live or ended")
	case query.text != "" && query.filter.Project != "":
		return sessionQuery{}, errors.New("a text search covers every project, so it cannot be combined with project_id")
	case len(sent.Sort) > 0 && sent.Sort[0].Field != query.sort:
		if query.text != "" {
			return sessionQuery{}, errors.New("a text search is sorted by relevance")
		}
		return sessionQuery{}, errors.New("only a text search is sorted by relevance")
	}

	if requested := string(value(filter.UserID)); requested != "" {
		if KindFrom(ctx) != auth.KindServer {
			return sessionQuery{}, errors.New("only a server-side caller may list another user's sessions")
		}
		query.filter.UserID = requested
	}

	cursor, err := decodeCursor[sessionCursor](&sent.Cursor)
	if err != nil {
		return sessionQuery{}, err
	}
	if cursor != nil {
		if cursor.Sort != query.sort {
			return sessionQuery{}, errBadCursor
		}
		query.filter.Cursor = &cursor.SessionPosition
	}
	return query, nil
}

// sessionPageOf renders a query's results as a page, with the cursor to the next one.
func sessionPageOf(found []session.Found, limit int, sort SessionSortField) SessionPage {
	kept, more := page(found, store.SessionLimit(limit))
	rendered := SessionPage{Items: sessionsOf(kept), HasMore: more}
	if more {
		rendered.NextCursor = encodeCursor(sessionCursor{kept[len(kept)-1].Position(), sort})
	}
	return rendered
}
