package api

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// noKnowledgeURLs is what these paths say on a deployment that cannot honour a
// subscription: it takes a database to remember one and a reader to fetch the page.
const noKnowledgeURLs = "knowledge urls are not available: no database or no way to read a page configured"

// unknownKnowledgeURL is what a caller is told about a page that is not theirs, which is
// the same thing they are told about one that never existed.
const unknownKnowledgeURL = "no such knowledge url"

// ListKnowledgeUrls returns the pages the calling customer's knowledge bases are filled
// from, newest first.
func (s *Server) ListKnowledgeUrls(
	ctx context.Context, request ListKnowledgeUrlsRequestObject,
) (ListKnowledgeUrlsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListKnowledgeUrls401JSONResponse{missingCustomer()}, nil
	}
	if s.pages == nil {
		return ListKnowledgeUrls400JSONResponse{badRequest(noKnowledgeURLs)}, nil
	}

	namespace := ""
	if request.Params.Namespace != nil {
		namespace = *request.Params.Namespace
	}

	stored, err := s.pages.List(ctx, customerID, namespace)
	if err != nil {
		return nil, err
	}

	listed := make([]KnowledgeUrl, 0, len(stored))
	for _, page := range stored {
		listed = append(listed, knowledgeURLOf(page))
	}
	return ListKnowledgeUrls200JSONResponse(listed), nil
}

// AddKnowledgeUrl subscribes a knowledge base to a page and reads it.
//
// A page that could not be read is a 201 with the row in the failed state rather than an
// error: the subscription was made, and what went wrong reading it is on the row where the
// caller can see it and try again.
func (s *Server) AddKnowledgeUrl(
	ctx context.Context, request AddKnowledgeUrlRequestObject,
) (AddKnowledgeUrlResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return AddKnowledgeUrl401JSONResponse{missingCustomer()}, nil
	}
	if s.pages == nil {
		return AddKnowledgeUrl400JSONResponse{badRequest(noKnowledgeURLs)}, nil
	}
	if request.Body == nil {
		return AddKnowledgeUrl400JSONResponse{badRequest("a request body is required")}, nil
	}

	page, err := s.pages.Add(ctx, customerID, request.Body.Namespace, request.Body.Url)
	if err != nil {
		return AddKnowledgeUrl400JSONResponse{badRequest(err.Error())}, nil
	}
	return AddKnowledgeUrl201JSONResponse(knowledgeURLOf(page)), nil
}

// GetKnowledgeUrl returns one page.
func (s *Server) GetKnowledgeUrl(
	ctx context.Context, request GetKnowledgeUrlRequestObject,
) (GetKnowledgeUrlResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetKnowledgeUrl401JSONResponse{missingCustomer()}, nil
	}
	if s.pages == nil {
		return GetKnowledgeUrl400JSONResponse{badRequest(noKnowledgeURLs)}, nil
	}

	page, err := s.pages.Get(ctx, customerID, request.Id)
	if err != nil {
		return GetKnowledgeUrl404JSONResponse{NotFoundJSONResponse{Error: unknownKnowledgeURL}}, nil
	}
	return GetKnowledgeUrl200JSONResponse(knowledgeURLOf(page)), nil
}

// DeleteKnowledgeUrl stops filling a knowledge base from a page, and removes the passages
// it wrote.
func (s *Server) DeleteKnowledgeUrl(
	ctx context.Context, request DeleteKnowledgeUrlRequestObject,
) (DeleteKnowledgeUrlResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return DeleteKnowledgeUrl401JSONResponse{missingCustomer()}, nil
	}
	if s.pages == nil {
		return DeleteKnowledgeUrl400JSONResponse{badRequest(noKnowledgeURLs)}, nil
	}

	if err := s.pages.Remove(ctx, customerID, request.Id); err != nil {
		return DeleteKnowledgeUrl404JSONResponse{NotFoundJSONResponse{Error: unknownKnowledgeURL}}, nil
	}
	return DeleteKnowledgeUrl204Response{}, nil
}

// IndexKnowledgeUrl reads a page again.
func (s *Server) IndexKnowledgeUrl(
	ctx context.Context, request IndexKnowledgeUrlRequestObject,
) (IndexKnowledgeUrlResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return IndexKnowledgeUrl401JSONResponse{missingCustomer()}, nil
	}
	if s.pages == nil {
		return IndexKnowledgeUrl400JSONResponse{badRequest(noKnowledgeURLs)}, nil
	}

	page, err := s.pages.Reindex(ctx, customerID, request.Id)
	if err != nil {
		return IndexKnowledgeUrl404JSONResponse{NotFoundJSONResponse{Error: unknownKnowledgeURL}}, nil
	}
	return IndexKnowledgeUrl200JSONResponse(knowledgeURLOf(page)), nil
}

// knowledgeURLOf is the stored row as the API describes it.
func knowledgeURLOf(page store.KnowledgeURL) KnowledgeUrl {
	described := KnowledgeUrl{
		Id:            page.ID,
		Namespace:     page.Namespace,
		Url:           page.URL,
		State:         KnowledgeUrlState(page.State),
		Passages:      page.Passages,
		LastIndexedAt: page.LastIndexedAt,
		CreatedAt:     page.CreatedAt,
		UpdatedAt:     page.UpdatedAt,
	}
	if page.Title != "" {
		described.Title = &page.Title
	}
	if page.Error != "" {
		described.Error = &page.Error
	}
	return described
}
