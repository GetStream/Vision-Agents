package api

import (
	"context"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/danielgtaylor/huma/v2"
)

// search answers one question out of what is true now.
type SearchRequest struct {
	ConfigId *string            `json:"config_id,omitempty" doc:"A stored router config to take the options from. Anything named here as well overrides that one field of it."`
	Query    string             `json:"query" doc:"The question, in the caller's own words." example:"perioperative antibiotic guidance"`
	Options  *SearchOptions     `json:"options,omitempty"`
	Tags     *map[string]string `json:"tags,omitempty"`
}

type SearchAnswer struct {
	Provider string         `json:"provider"`
	Model    string         `json:"model"`
	Answer   *string        `json:"answer,omitempty" doc:"The provider's own summary, where it offers one. It is what a voice agent wants: a sentence to say rather than a page to read."`
	Results  []SearchResult `json:"results" doc:"The sources behind it, most relevant first."`
}

type SearchResult struct {
	Title *string  `json:"title,omitempty"`
	Url   string   `json:"url"`
	Text  *string  `json:"text,omitempty" doc:"The relevant extract, which is what a model reads."`
	Score *float32 `json:"score,omitempty" doc:"How relevant the provider judged it."`
}

type searchRequest struct {
	Body SearchRequest
}

type searchAnswerResponse struct {
	Body SearchAnswer
}

func (s *Server) registerSearch(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "search",
		Method:      http.MethodPost,
		Path:        "/v1/search",
		Summary:     "Answer a question out of what is true now",
		Description: "The fourth routed modality, reachable on its own rather than only as a tool an " +
			"agent reaches for. One question, one answer: routed, failed over and billed " +
			"like the rest, and with no socket because nothing arrives in pieces.",
		Responses: map[string]*huma.Response{
			"200": {Description: "What the provider found"},
		},
		Errors:     []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
		Extensions: map[string]any{clientAccessibleExtension: true},
	}, s.search)
}

// It is the one routed modality with no socket: a question and its answer are one round
// trip, and holding a connection open between them would buy nothing. Everything else is
// the same as the three that do have one - a config to take the options from, per-call
// overrides, failover down the candidate list, and a row recording what it cost.
func (s *Server) search(ctx context.Context, request *searchRequest) (*searchAnswerResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.streams == nil || s.streams.Search == nil {
		return nil, huma.Error404NotFound("this deployment does not route search")
	}
	if strings.TrimSpace(request.Body.Query) == "" {
		return nil, huma.Error400BadRequest("there is nothing to look for")
	}

	config, err := s.routerOptions(ctx, customerID, value(request.Body.ConfigId))
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	held := config.Search.Merge(searchOptionsOf(request.Body.Options))

	tags := tagsSent(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	session, err := s.streams.Search.Start(ctx, searchrouter.Request{
		CustomerID:    customerID,
		Tags:          tags,
		Target:        held.Route(),
		LanguageHints: nil,
		Options:       held,
	})
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
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
		return nil, huma.Error400BadRequest(err.Error())
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
	return &searchAnswerResponse{Body: SearchAnswer{
		Provider: session.Provider(),
		Model:    session.Model(),
		Answer:   optional(found.Answer),
		Results:  results,
	}}, nil
}
