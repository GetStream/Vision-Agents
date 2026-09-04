// Package exa answers questions out of Exa's search API, and reads pages out of its
// contents API.
//
// Exa publishes no Go SDK, so this speaks its REST API directly. It is the one provider
// here that does both halves of what this service needs from the web: /search finds out
// which pages bear on a question, and /contents reads a page somebody already named. The
// second is what fills a knowledge base from a URL, because Exa returns a crawled page as
// markdown with the navigation and the advertising already taken out.
package exa

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
const ProviderName = "exa"

const apiKeyEnvVar = "EXA_API_KEY"

// The ways Exa will search, which are this provider's models. Fast trades a little recall
// for latency, which is the trade a live conversation wants; auto lets Exa decide between
// its neural and keyword indexes, which is better and slower.
const (
	ModelFast    = "fast"
	ModelAuto    = "auto"
	ModelNeural  = "neural"
	ModelKeyword = "keyword"
)

var models = map[string]struct{}{
	ModelFast:    {},
	ModelAuto:    {},
	ModelNeural:  {},
	ModelKeyword: {},
}

const (
	defaultBaseURL = "https://api.exa.ai"
	defaultLimit   = 5
	// defaultTimeout is short because the search happens while somebody is waiting on the
	// phone. An answer that has not arrived by now is worth less than the pause it costs.
	defaultTimeout = 6 * time.Second
	// crawlTimeoutMs is how long Exa may spend fetching a page it has no fresh copy of.
	// Reading a URL is not on anybody's conversation, so it can wait for the crawl that a
	// search could not.
	crawlTimeoutMs = 12_000
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// Options configures a Provider. The key falls back to the environment, the way every
// other provider in this service is configured.
type Options struct {
	// APIKey defaults to EXA_API_KEY.
	APIKey string
	// Model is how Exa should search. Empty means fast.
	Model string
	// BaseURL replaces the default host, which is what a test or a proxy needs.
	BaseURL string
	// Limit caps how many results one search returns. Zero means five.
	Limit int
	// Timeout bounds one call. A provider used to read pages wants a longer one than a
	// provider used mid-conversation, since a live crawl takes seconds.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
	Logger     *slog.Logger
}

// Provider is Exa's search. It satisfies search.Provider and search.Reader.
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
		return nil, errors.New("exa: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = ModelFast
	}
	if _, ok := models[options.Model]; !ok {
		return nil, fmt.Errorf("exa: %q is not a way this searches", options.Model)
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

// Model is how this searches.
func (p *Provider) Model() string { return p.model }

// Start opens nothing: the key was checked when this was built.
func (p *Provider) Start(context.Context) error { return nil }

// Close releases nothing: the provider holds no connection of its own.
func (p *Provider) Close() error { return nil }

// searchRequest is what goes upstream. Highlights rather than whole pages, because the
// answer is about to be read out and a model handed five full articles will read them.
type searchRequest struct {
	Query      string         `json:"query"`
	Type       string         `json:"type"`
	NumResults int            `json:"numResults"`
	Contents   searchContents `json:"contents"`
	// IncludeDomains and ExcludeDomains are how Exa narrows where an answer comes from.
	IncludeDomains []string `json:"includeDomains,omitempty"`
	ExcludeDomains []string `json:"excludeDomains,omitempty"`
	// Category names the slice of the index to search.
	Category string `json:"category,omitempty"`
	// StartPublishedDate is how Exa expresses recency: everything published since then.
	StartPublishedDate string `json:"startPublishedDate,omitempty"`
	// UserLocation is a two-letter country code.
	UserLocation string `json:"userLocation,omitempty"`
}

type searchContents struct {
	// Text is the whole page, which a caller asks for when something other than a voice
	// is going to read it.
	Text       bool              `json:"text,omitempty"`
	Highlights *highlightOptions `json:"highlights,omitempty"`
	Summary    *summaryOptions   `json:"summary,omitempty"`
}

// summaryOptions asks Exa to write a summary of each page rather than quote from it.
type summaryOptions struct {
	Query string `json:"query,omitempty"`
}

// highlightOptions asks for the excerpts that bear on the question rather than the ones
// that happen to open the page.
type highlightOptions struct {
	Query string `json:"query"`
}

type searchResponse struct {
	Results []struct {
		Title      string   `json:"title"`
		URL        string   `json:"url"`
		Text       string   `json:"text"`
		Highlights []string `json:"highlights"`
		Summary    string   `json:"summary"`
		Score      float64  `json:"score"`
	} `json:"results"`
}

// Search answers the question out of what is true now.
//
// Exa writes no summary of its own, so Result.Answer is left empty and the sources speak
// for themselves. That is what makes it the fast option: nothing here waits on a model.
func (p *Provider) Search(ctx context.Context, query search.Query) (search.Result, error) {
	if err := query.Validate(); err != nil {
		return search.Result{}, err
	}

	limit := query.Limit
	if limit <= 0 {
		limit = p.limit
	}
	question := strings.TrimSpace(query.Text)

	request := searchRequest{
		Query:          question,
		Type:           p.model,
		NumResults:     limit,
		Contents:       contentsFor(question, query.Contents),
		IncludeDomains: query.IncludeDomains,
		ExcludeDomains: query.ExcludeDomains,
		Category:       query.Category,
		UserLocation:   strings.ToUpper(query.Location),
	}
	if query.MaxAgeHours > 0 {
		since := time.Now().UTC().Add(-time.Duration(query.MaxAgeHours) * time.Hour)
		request.StartPublishedDate = since.Format(time.RFC3339)
	}

	var decoded searchResponse
	if err := p.call(ctx, "/search", request, &decoded); err != nil {
		return search.Result{}, err
	}

	var found search.Result
	for _, result := range decoded.Results {
		text := strings.TrimSpace(strings.Join(result.Highlights, "\n\n"))
		if text == "" {
			text = strings.TrimSpace(result.Summary)
		}
		if text == "" {
			text = strings.TrimSpace(result.Text)
		}
		found.Documents = append(found.Documents, search.Document{
			Title: result.Title,
			URL:   result.URL,
			Text:  text,
			Score: result.Score,
		})
	}
	return found, nil
}

// contentsFor is what to return alongside each hit. Highlights are the default because
// the answer is about to be read out and a model handed five full articles will read
// them; a caller that named something else asked for it on purpose.
func contentsFor(question string, wanted []string) searchContents {
	if len(wanted) == 0 {
		return searchContents{Highlights: &highlightOptions{Query: question}}
	}

	contents := searchContents{}
	for _, want := range wanted {
		switch strings.ToLower(want) {
		case "text":
			contents.Text = true
		case "highlights":
			contents.Highlights = &highlightOptions{Query: question}
		case "summary":
			contents.Summary = &summaryOptions{Query: question}
		}
	}
	return contents
}

type contentsRequest struct {
	URLs             []string `json:"urls"`
	Text             bool     `json:"text"`
	LivecrawlTimeout int      `json:"livecrawlTimeout"`
}

type contentsResponse struct {
	Results []struct {
		URL   string `json:"url"`
		Title string `json:"title"`
		Text  string `json:"text"`
	} `json:"results"`
	// Statuses is how this endpoint reports a page it could not read: the request itself
	// succeeds, and the URL that failed says so here.
	Statuses []struct {
		ID     string `json:"id"`
		Status string `json:"status"`
		Error  struct {
			Tag            string `json:"tag"`
			HTTPStatusCode int    `json:"httpStatusCode"`
		} `json:"error"`
	} `json:"statuses"`
}

// Read returns the page as markdown, which is Exa's default format.
func (p *Provider) Read(ctx context.Context, url string) (search.Page, error) {
	url = strings.TrimSpace(url)
	if url == "" {
		return search.Page{}, errors.New("exa: there is no page to read")
	}

	var decoded contentsResponse
	err := p.call(ctx, "/contents", contentsRequest{
		URLs:             []string{url},
		Text:             true,
		LivecrawlTimeout: crawlTimeoutMs,
	}, &decoded)
	if err != nil {
		return search.Page{}, err
	}

	// The status is checked before the results, because a page that could not be read
	// comes back as an empty result rather than as an absent one, and "this page is
	// blank" is not what went wrong.
	for _, status := range decoded.Statuses {
		if status.Status == "error" {
			return search.Page{}, fmt.Errorf("exa: could not read %s: %s", url, status.Error.Tag)
		}
	}
	if len(decoded.Results) == 0 {
		return search.Page{}, fmt.Errorf("exa: nothing came back for %s", url)
	}

	result := decoded.Results[0]
	if strings.TrimSpace(result.Text) == "" {
		return search.Page{}, fmt.Errorf("exa: there is nothing to read at %s", url)
	}
	return search.Page{URL: url, Title: result.Title, Text: result.Text}, nil
}

// call posts to Exa and decodes what comes back.
func (p *Provider) call(ctx context.Context, path string, body, into any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("exa: encode %s: %w", path, err)
	}

	request, err := http.NewRequestWithContext(
		ctx, http.MethodPost, p.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("exa: build %s: %w", path, err)
	}
	request.Header.Set("x-api-key", p.apiKey)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	response, err := p.client.Do(request)
	if err != nil {
		return fmt.Errorf("exa: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("exa: %s returned %d: %s",
			path, response.StatusCode, strings.TrimSpace(string(detail)))
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("exa: decode %s: %w", path, err)
	}
	return nil
}
