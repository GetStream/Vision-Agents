package api

import (
	"context"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
)

// Search answers one question out of what is true now.
//
// It is the one routed modality with no socket: a question and its answer are one round
// trip, and holding a connection open between them would buy nothing. Everything else is
// the same as the three that do have one - a config to take the options from, per-call
// overrides, failover down the candidate list, and a row recording what it cost.
func (s *Server) Search(ctx context.Context, request SearchRequestObject) (SearchResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return Search401JSONResponse{missingCustomer()}, nil
	}
	if s.streams == nil || s.streams.Search == nil {
		return Search404JSONResponse{NotFoundJSONResponse{Error: "this deployment does not route search"}}, nil
	}
	if request.Body == nil {
		return Search400JSONResponse{badRequest("a request body is required")}, nil
	}
	if strings.TrimSpace(request.Body.Query) == "" {
		return Search400JSONResponse{badRequest("there is nothing to look for")}, nil
	}

	config, err := s.routerOptions(ctx, customerID, value(request.Body.ConfigId))
	if err != nil {
		return Search400JSONResponse{badRequest(err.Error())}, nil
	}
	held := config.Search.Merge(searchOptionsOf(request.Body.Options))

	tags := tagsUnder(config, request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return Search400JSONResponse{badRequest(err.Error())}, nil
	}

	session, err := s.streams.Search.Start(ctx, searchrouter.Request{
		CustomerID:    customerID,
		Tags:          tags,
		Target:        held.Route(),
		LanguageHints: nil,
		Options:       held,
	})
	if err != nil {
		return Search400JSONResponse{badRequest(err.Error())}, nil
	}
	defer session.Close()

	found, err := session.Search(ctx, search.Query{
		Text:           request.Body.Query,
		Limit:          count(held.Results),
		IncludeDomains: held.IncludeDomains,
		ExcludeDomains: held.ExcludeDomains,
		Category:       held.Category,
		MaxAgeHours:    count(held.MaxAgeHours),
		Location:       held.Location,
		Contents:       held.Contents,
		OutputSchema:   held.OutputSchema,
	})
	if err != nil {
		return Search400JSONResponse{badRequest(err.Error())}, nil
	}

	results := make([]SearchResult, 0, len(found.Documents))
	for _, document := range found.Documents {
		score := float32(document.Score)
		results = append(results, SearchResult{
			Title: optional(document.Title),
			Url:   document.URL,
			Text:  optional(document.Text),
			Score: &score,
		})
	}
	return Search200JSONResponse{
		Provider: session.Provider(),
		Model:    session.Model(),
		Answer:   optional(found.Answer),
		Results:  results,
	}, nil
}
