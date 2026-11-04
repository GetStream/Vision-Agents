// Package perplexity answers questions out of Perplexity's two web APIs.
//
// Perplexity publishes no Go SDK, so this speaks its REST API directly. Which of the two
// endpoints is used is the model: `search` is the ranked index, which returns pages and no
// opinion about them, while `sonar` searches through the Agent API and writes the sentence. The
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
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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
	// Model is `search` for the ranked index, or `sonar` to have it write the answer.
	// Empty means search.
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
	keys := keypool.FromEnv(options.APIKey, apiKeyEnvVar)
	if keys.Len() == 0 {
		return nil, stack.Wrap(errors.New("perplexity: " + apiKeyEnvVar + " is required"))
	}
	options.APIKey = keys.Peek()
	if options.Model == "" {
		options.Model = ModelSearch
	}
	if _, ok := agentModels[options.Model]; !ok && options.Model != ModelSearch {
		return nil, stack.Wrap(fmt.Errorf("perplexity: %q is neither the search index nor sonar", options.Model))
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
		client:  keys.WrapClient(options.HTTPClient, keypool.Bearer),
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
	return p.answer(ctx, strings.TrimSpace(query.Text), limit)
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

// agentModels is the Agent API's name for each sonar model this adapter serves. Sonar
// Chat Completions ended on 2026-09-27; the Agent API bills the model's tokens and each
// web search apart (https://docs.perplexity.ai/docs/getting-started/pricing, read
// 2026-11-04).
var agentModels = map[string]string{"sonar": "perplexity/sonar"}

// answerInstructions keeps the answer short enough to say: without them the model writes
// for a page.
const answerInstructions = "Answer in one to three plain sentences, with no markdown, headings or lists."

type agentRequest struct {
	Model        string `json:"model"`
	Input        string `json:"input"`
	Instructions string `json:"instructions"`
	Tools        []tool `json:"tools"`
	// ToolChoice makes the model search: offered the tool, it may answer from what it
	// knows instead.
	ToolChoice tool `json:"tool_choice"`
}

type tool struct {
	Type string `json:"type"`
	// MaxResults caps the results one search collects, 1 to 50.
	MaxResults int `json:"max_results,omitempty"`
}

type agentResponse struct {
	// Output is one item per step: search_results for what a search found, message for
	// the answer.
	Output []struct {
		Type    string `json:"type"`
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Snippet string `json:"snippet"`
		} `json:"results"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output"`
	Usage struct {
		// InputTokens counts the cached ones too.
		InputTokens        int64 `json:"input_tokens"`
		InputTokensDetails struct {
			CachedTokens int64 `json:"cached_tokens"`
		} `json:"input_tokens_details"`
		OutputTokens     int64 `json:"output_tokens"`
		ToolCallsDetails map[string]struct {
			Invocation int64 `json:"invocation"`
		} `json:"tool_calls_details"`
		// Cost is what Perplexity charged for the answer, in dollars.
		Cost struct {
			TotalCost float64 `json:"total_cost"`
		} `json:"cost"`
	} `json:"usage"`
}

// searches is how many web searches the answer ran. The tool is asked for as web_search
// and reported as search_web.
func (r agentResponse) searches() int64 {
	return r.Usage.ToolCallsDetails["search_web"].Invocation + r.Usage.ToolCallsDetails["web_search"].Invocation
}

// maxAgentResults is the most results the Agent API's web search collects in one call.
const maxAgentResults = 50

// answer has a sonar model search the web and write the sentence, keeping at most limit of
// the pages it read as the sources.
func (p *Provider) answer(ctx context.Context, question string, limit int) (search.Result, error) {
	var decoded agentResponse
	err := p.call(ctx, "/v1/agent", agentRequest{
		Model:        agentModels[p.model],
		Input:        question,
		Instructions: answerInstructions,
		Tools:        []tool{{Type: "web_search", MaxResults: min(limit, maxAgentResults)}},
		ToolChoice:   tool{Type: "web_search"},
	}, &decoded)
	if err != nil {
		return search.Result{}, err
	}

	cached := min(decoded.Usage.InputTokensDetails.CachedTokens, decoded.Usage.InputTokens)
	found := search.Result{
		VendorMicros: int64(math.Round(decoded.Usage.Cost.TotalCost * 1_000_000)),
		Usage: pricing.Quantities{
			pricing.InputTokens:       decoded.Usage.InputTokens - cached,
			pricing.CachedInputTokens: cached,
			pricing.OutputTokens:      decoded.Usage.OutputTokens,
			pricing.Requests:          decoded.searches(),
		},
	}
	var answer []string
	for _, item := range decoded.Output {
		switch item.Type {
		case "search_results":
			for _, result := range item.Results {
				found.Documents = append(found.Documents, search.Document{
					Title: result.Title,
					URL:   result.URL,
					Text:  result.Snippet,
				})
			}
		case "message":
			for _, content := range item.Content {
				if content.Type == "output_text" {
					answer = append(answer, content.Text)
				}
			}
		}
	}
	found.Answer = strings.TrimSpace(strings.Join(answer, ""))
	// The cap is per search, and the model may search more than once.
	found.Documents = found.Documents[:min(len(found.Documents), limit)]
	return found, nil
}

// call posts to Perplexity and decodes what comes back.
func (p *Provider) call(ctx context.Context, path string, body, into any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return stack.Wrap(fmt.Errorf("perplexity: encode %s: %w", path, err))
	}

	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, p.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return stack.Wrap(fmt.Errorf("perplexity: build %s: %w", path, err))
	}
	request.Header.Set("Authorization", "Bearer "+p.apiKey)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	response, err := p.client.Do(request)
	if err != nil {
		return stack.Wrap(fmt.Errorf("perplexity: %s: %w", path, err))
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return stack.Wrap(fmt.Errorf("perplexity: %s returned %d: %s",
			path, response.StatusCode, strings.TrimSpace(string(detail))))
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return stack.Wrap(fmt.Errorf("perplexity: decode %s: %w", path, err))
	}
	return nil
}
