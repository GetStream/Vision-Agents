package api

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// noStore is what the response paths say where nothing was written down. A turn's items are
// read back from Postgres, so a deployment without one can start a response but has nothing
// to show afterwards.
const noStore = "this deployment does not record what sessions said"

// CreateResponse asks the agent something and names the turn it answers as.
//
// It is respond with an id back, which is the whole of the difference. Without one a caller
// following a particular answer has to watch every event on the socket and work out which
// belong to the turn it asked for; with one it can ask for that turn's items instead.
func (s *Server) CreateResponse(ctx context.Context, request CreateResponseRequestObject) (CreateResponseResponseObject, error) {
	// A response is asked of a running session rather than a stored one: a conversation that
	// ended can be read and forked, but not talked to.
	found, failure := s.session(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return CreateResponse401JSONResponse{missingCustomer()}, nil
		}
		return CreateResponse404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}
	if request.Body == nil || request.Body.Text == "" {
		return CreateResponse400JSONResponse{badRequest("there is nothing to answer")}, nil
	}

	images := make([]wireImage, 0, len(value(request.Body.Images)))
	for _, sent := range value(request.Body.Images) {
		images = append(images, wireImage{URL: sent.Url, Detail: string(value(sent.Detail))})
	}
	parts, err := imagesFromWire(images)
	if err != nil {
		return CreateResponse400JSONResponse{badRequest(err.Error())}, nil
	}

	responseID, err := found.Respond(ctx, request.Body.Text, parts)
	if err != nil {
		return CreateResponse400JSONResponse{badRequest(err.Error())}, nil
	}
	if responseID == "" {
		// The turn is being answered, it just has no name. A session that records nothing
		// has no row to hand back, and a native one lets the model decide what a turn is.
		// Refusing would be wrong -- the agent is answering -- so the id is empty and the
		// caller reads the socket, which is what it would have done before this existed.
		return CreateResponse202JSONResponse{
			SessionId: found.ID(), Status: AgentResponseStatusRunning,
			CreatedAt: found.CreatedAt(),
		}, nil
	}

	said := request.Body.Text
	return CreateResponse202JSONResponse{
		Id: responseID, SessionId: found.ID(), Said: &said,
		Status: AgentResponseStatusRunning, CreatedAt: time.Now().UTC(),
	}, nil
}

// ListResponses returns a session's turns, oldest first.
func (s *Server) ListResponses(ctx context.Context, request ListResponsesRequestObject) (ListResponsesResponseObject, error) {
	found, failure := s.storedOrLiveSession(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return ListResponses401JSONResponse{missingCustomer()}, nil
		}
		return ListResponses404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}
	if s.store == nil {
		return ListResponses404JSONResponse{NotFoundJSONResponse{Error: noStore}}, nil
	}

	rows, err := s.store.SessionResponses(ctx, OwnerFrom(ctx).CustomerID, found.ID(),
		value(request.Params.Limit), value(request.Params.Offset))
	if err != nil {
		return nil, err
	}

	listed := make([]AgentResponse, 0, len(rows))
	for _, row := range rows {
		listed = append(listed, responseOf(row))
	}
	return ListResponses200JSONResponse(listed), nil
}

// ListResponseItems returns what the agent did, in the order it happened.
//
// Flat across turns rather than a list per turn, because that is how a conversation reads
// and how it gets rendered: the question, what the agent did about it, what it said, then
// the next question. Naming a response narrows it to that one turn.
func (s *Server) ListResponseItems(ctx context.Context, request ListResponseItemsRequestObject) (ListResponseItemsResponseObject, error) {
	found, failure := s.storedOrLiveSession(ctx, request.Id)
	if failure != nil {
		if failure.status == unauthorized {
			return ListResponseItems401JSONResponse{missingCustomer()}, nil
		}
		return ListResponseItems404JSONResponse{NotFoundJSONResponse{Error: failure.message}}, nil
	}
	if s.store == nil {
		return ListResponseItems404JSONResponse{NotFoundJSONResponse{Error: noStore}}, nil
	}

	rows, err := s.store.SessionItems(ctx, OwnerFrom(ctx).CustomerID, found.ID(),
		value(request.Params.ResponseId), value(request.Params.Limit), value(request.Params.Offset))
	if err != nil {
		return nil, err
	}

	listed := make([]AgentResponseItem, 0, len(rows))
	for _, row := range rows {
		listed = append(listed, itemOf(row))
	}
	return ListResponseItems200JSONResponse(listed), nil
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
