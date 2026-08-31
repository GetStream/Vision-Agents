// Package perplexity answers questions out of Perplexity's two web APIs.
//
// Perplexity publishes no Go SDK, so this speaks its REST API directly. Which of the two
// endpoints is used is the model: `search` is the ranked index, which returns pages and no
// opinion about them, while a `sonar` model reads those pages and writes the sentence. The
// second is what a caller on the phone actually wants and is also a language model in the
// middle of a conversation, so which one is worth its latency is a routing decision rather
// than something decided here.
package perplexity

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
const ProviderName = "perplexity"

const apiKeyEnvVar = "PERPLEXITY_API_KEY"

// ModelSearch is the ranked index: results, no summary, no model in the way.
const ModelSearch = "search"

const (
	defaultBaseURL = "https://api.perplexity.ai"
	defaultLimit   = 5
	// defaultTimeout is short because the search happens while somebody is waiting on the
	// phone. An answer that has not arrived by now is worth less than the pause it costs.
	// A sonar model reads pages before it answers, so it needs more of it than the index.
	defaultTimeout      = 6 * time.Second
	defaultSonarTimeout = 12 * time.Second
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// Options configures a Provider. The key falls back to the environment, the way every
// other provider in this service is configured.
type Options struct {
	// APIKey defaults to PERPLEXITY_API_KEY.
	APIKey string
	// Model is `search` for the ranked index, or a sonar model to have one write the
	// answer. Empty means search.
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

// Provider is Perplexity's search. It satisfies search.Provider.
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
		return nil, errors.New("perplexity: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = ModelSearch
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.Limit <= 0 {
		options.Limit = defaultLimit
	}
	if options.Timeout <= 0 {
		options.Timeout = defaultTimeout
		if options.Model != ModelSearch {
			options.Timeout = defaultSonarTimeout
		}
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

// Model is which of Perplexity's two endpoints answers, and with which model when it is
// the one that writes an answer.
func (p *Provider) Model() string { return p.model }

// Start opens nothing: the key was checked when this was built.
func (p *Provider) Start(context.Context) error { return nil }

// Close releases nothing: the provider holds no connection of its own.
func (p *Provider) Close() error { return nil }

// Search answers the question out of what is true now.
func (p *Provider) Search(ctx context.Context, query search.Query) (search.Result, error) {
	if err := query.Validate(); err != nil {
		return search.Result{}, err
	}

	limit := query.Limit
	if limit <= 0 {
		limit = p.limit
	}
	if p.model == ModelSearch {
		return p.index(ctx, strings.TrimSpace(query.Text), limit)
	}
	return p.answer(ctx, strings.TrimSpace(query.Text))
}

type indexRequest struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results"`
}

type indexResponse struct {
	Results []struct {
		Title   string `json:"title"`
		URL     string `json:"url"`
		Snippet string `json:"snippet"`
	} `json:"results"`
}

// index asks the ranked index. It reports no relevance score, so the order is the only
// thing saying which result mattered, and Prompt already renders them in it.
func (p *Provider) index(ctx context.Context, question string, limit int) (search.Result, error) {
	var decoded indexResponse
	err := p.call(ctx, "/search", indexRequest{Query: question, MaxResults: limit}, &decoded)
	if err != nil {
		return search.Result{}, err
	}

	var found search.Result
	for _, result := range decoded.Results {
		found.Documents = append(found.Documents, search.Document{
			Title: result.Title,
			URL:   result.URL,
			Text:  result.Snippet,
		})
	}
	return found, nil
}

type completionRequest struct {
	Model    string    `json:"model"`
	Messages []message `json:"messages"`
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type completionResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	SearchResults []struct {
		Title   string `json:"title"`
		URL     string `json:"url"`
		Snippet string `json:"snippet"`
	} `json:"search_results"`
	// Citations is the older shape, which is bare URLs. It is read when search_results is
	// absent so an answer never comes back with nothing to attribute it to.
	Citations []string `json:"citations"`
}

// answer has a sonar model read the pages and write the sentence.
func (p *Provider) answer(ctx context.Context, question string) (search.Result, error) {
	var decoded completionResponse
	err := p.call(ctx, "/chat/completions", completionRequest{
		Model:    p.model,
		Messages: []message{{Role: "user", Content: question}},
	}, &decoded)
	if err != nil {
		return search.Result{}, err
	}

	var found search.Result
	if len(decoded.Choices) > 0 {
		found.Answer = strings.TrimSpace(decoded.Choices[0].Message.Content)
	}
	for _, result := range decoded.SearchResults {
		found.Documents = append(found.Documents, search.Document{
			Title: result.Title,
			URL:   result.URL,
			Text:  result.Snippet,
		})
	}
	if len(found.Documents) == 0 {
		for _, citation := range decoded.Citations {
			found.Documents = append(found.Documents, search.Document{URL: citation})
		}
	}
	return found, nil
}

// call posts to Perplexity and decodes what comes back.
func (p *Provider) call(ctx context.Context, path string, body, into any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("perplexity: encode %s: %w", path, err)
	}

	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, p.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("perplexity: build %s: %w", path, err)
	}
	request.Header.Set("Authorization", "Bearer "+p.apiKey)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("perplexity: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("perplexity: %s returned %d: %s",
			path, response.StatusCode, strings.TrimSpace(string(detail)))
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("perplexity: decode %s: %w", path, err)
	}
	return nil
}
