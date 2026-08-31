// Package tavily answers questions out of Tavily's search API.
//
// Tavily publishes no Go SDK, so this speaks its REST API directly: one call that searches
// and summarises in the same round trip. That summary is why this provider rather than a
// plain web index: an agent mid-sentence cannot read ten pages and decide what matters, and
// a caller on the phone wants the sentence, not the results.
package tavily

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
)

// ProviderName is how this provider is named in stats.
const ProviderName = "tavily"

const apiKeyEnvVar = "TAVILY_API_KEY"

// The depths Tavily searches at, which are this provider's models. Basic searches the
// index; advanced also crawls the pages it finds, which roughly doubles the latency for an
// answer somebody is waiting on out loud.
const (
	ModelBasic    = "basic"
	ModelAdvanced = "advanced"
)

const (
	defaultBaseURL = "https://api.tavily.com"
	defaultLimit   = 5
	// defaultTimeout is short because the search happens while somebody is waiting on the
	// phone. An answer that has not arrived by now is worth less than the pause it costs.
	defaultTimeout = 6 * time.Second
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// Options configures a Provider. The key falls back to the environment, the way every
// other provider in this service is configured.
type Options struct {
	// APIKey defaults to TAVILY_API_KEY.
	APIKey string
	// Model is the depth to search at. Empty means basic.
	Model string
	// BaseURL replaces the default host, which is what a test or a proxy needs.
	BaseURL string
	// Limit caps how many results one search returns. Zero means five.
	Limit int
	// Timeout bounds one search. It happens mid-conversation, so it cannot hang a call.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
	Logger     *slog.Logger
}

// Provider is Tavily's search. It satisfies search.Provider.
type Provider struct {
	apiKey  string
	model   string
	baseURL string
	limit   int
	client  *http.Client
	logger  *slog.Logger
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("tavily: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = ModelBasic
	}
	if options.Model != ModelBasic && options.Model != ModelAdvanced {
		return nil, fmt.Errorf("tavily: %q is not a depth this searches at", options.Model)
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.Limit <= 0 {
		options.Limit = defaultLimit
	}
	if options.Timeout <= 0 {
		options.Timeout = defaultTimeout
	}
	if options.HTTPClient == nil {
		options.HTTPClient = &http.Client{Timeout: options.Timeout}
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	return &Provider{
		apiKey:  options.APIKey,
		model:   options.Model,
		baseURL: strings.TrimSuffix(options.BaseURL, "/"),
		limit:   options.Limit,
		client:  options.HTTPClient,
		logger:  options.Logger,
	}, nil
}

// Provider implements search.Provider.
func (p *Provider) Provider() string { return ProviderName }

// Model is the depth this searches at.
func (p *Provider) Model() string { return p.model }

// Start opens nothing: the key was checked when this was built.
func (p *Provider) Start(context.Context) error { return nil }

// Close releases nothing: the provider holds no connection of its own.
func (p *Provider) Close() error { return nil }

// searchRequest is what goes upstream. The answer is asked for in the same call, because a
// second round trip to summarise would cost more than the search did.
type searchRequest struct {
	Query         string `json:"query"`
	MaxResults    int    `json:"max_results"`
	IncludeAnswer bool   `json:"include_answer"`
	SearchDepth   string `json:"search_depth"`
}

// searchResponse is what comes back.
type searchResponse struct {
	Answer  string `json:"answer"`
	Results []struct {
		Title   string  `json:"title"`
		URL     string  `json:"url"`
		Content string  `json:"content"`
		Score   float64 `json:"score"`
	} `json:"results"`
}

// Search answers the question out of what is true now.
func (p *Provider) Search(ctx context.Context, query search.Query) (search.Result, error) {
	if err := query.Validate(); err != nil {
		return search.Result{}, err
	}

	limit := query.Limit
	if limit <= 0 {
		limit = p.limit
	}
	payload, err := json.Marshal(searchRequest{
		Query:         strings.TrimSpace(query.Text),
		MaxResults:    limit,
		IncludeAnswer: true,
		SearchDepth:   p.model,
	})
	if err != nil {
		return search.Result{}, fmt.Errorf("tavily: encode search: %w", err)
	}

	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, p.baseURL+"/search", bytes.NewReader(payload))
	if err != nil {
		return search.Result{}, fmt.Errorf("tavily: build search: %w", err)
	}
	request.Header.Set("Authorization", "Bearer "+p.apiKey)
	request.Header.Set("Content-Type", "application/json")

	response, err := p.client.Do(request)
	if err != nil {
		return search.Result{}, fmt.Errorf("tavily: search: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return search.Result{}, fmt.Errorf("tavily: search returned %d: %s",
			response.StatusCode, strings.TrimSpace(string(body)))
	}

	var decoded searchResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		return search.Result{}, fmt.Errorf("tavily: decode search: %w", err)
	}

	found := search.Result{Answer: decoded.Answer}
	for _, result := range decoded.Results {
		found.Documents = append(found.Documents, search.Document{
			Title: result.Title,
			URL:   result.URL,
			Text:  result.Content,
			Score: result.Score,
		})
	}
	return found, nil
}
