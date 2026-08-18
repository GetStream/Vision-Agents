// Package mem0 stores an agent's memories in mem0's hosted platform.
//
// mem0 publishes no Go SDK, so this speaks its v3 REST API directly: one call to hand a
// conversation over and one to ask what is known. Extraction is asynchronous on mem0's
// side, so adding returns an event id rather than the memories it will produce.
package mem0

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

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
)

const apiKeyEnvVar = "MEM0_API_KEY"

const (
	defaultBaseURL = "https://api.mem0.ai"
	defaultTopK    = 5
	defaultTimeout = 10 * time.Second
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// Options configures a Store. The key falls back to the environment, the way every other
// provider in this service is configured.
type Options struct {
	// APIKey defaults to MEM0_API_KEY.
	APIKey string
	// BaseURL defaults to the hosted platform. A self-hosted deployment overrides it.
	BaseURL string
	// Timeout bounds one call. Recall is on the join path, so it cannot hang a call.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
	Logger     *slog.Logger
}

// Store is a mem0 memory store. It satisfies memory.Store.
type Store struct {
	apiKey  string
	baseURL string
	client  *http.Client
	logger  *slog.Logger
}

// New validates the options and returns a Store.
func New(options Options) (*Store, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("mem0: " + apiKeyEnvVar + " is required")
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
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

	return &Store{
		apiKey:  options.APIKey,
		baseURL: strings.TrimSuffix(options.BaseURL, "/"),
		client:  options.HTTPClient,
		logger:  options.Logger,
	}, nil
}

// Recall returns what mem0 knows that bears on the query, most relevant first.
func (s *Store) Recall(ctx context.Context, query memory.Query) ([]memory.Memory, error) {
	if err := query.Scope.Validate(); err != nil {
		return nil, err
	}

	limit := query.Limit
	if limit <= 0 {
		limit = defaultTopK
	}

	// v3 wants the entity ids inside filters; at the top level they are rejected.
	body := searchRequest{
		Query:   query.Text,
		Filters: filtersFor(query.Scope),
		TopK:    limit,
	}

	var response searchResponse
	if err := s.call(ctx, "/v3/memories/search/", body, &response); err != nil {
		return nil, err
	}

	recalled := make([]memory.Memory, 0, len(response.Results))
	for _, result := range response.Results {
		if result.Memory == "" {
			continue
		}
		recalled = append(recalled, memory.Memory{ID: result.ID, Text: result.Memory, Score: result.Score})
	}
	return recalled, nil
}

// Remember hands a conversation to mem0 to learn from. Extraction runs on mem0's side, so
// this returns as soon as the work is queued rather than when memories exist.
func (s *Store) Remember(ctx context.Context, scope memory.Scope, messages []llm.Message) error {
	if err := scope.Validate(); err != nil {
		return err
	}
	if len(messages) == 0 {
		return nil
	}

	body := addRequest{
		Messages: make([]message, 0, len(messages)),
		UserID:   scope.UserID,
		AppID:    scope.AppID,
	}
	for _, said := range messages {
		if said.Content == "" {
			continue
		}
		body.Messages = append(body.Messages, message{Role: string(said.Role), Content: said.Content})
	}
	if len(body.Messages) == 0 {
		return nil
	}

	var response addResponse
	if err := s.call(ctx, "/v3/memories/add/", body, &response); err != nil {
		return err
	}
	s.logger.Debug("queued a conversation to remember", "event", response.EventID, "status", response.Status)
	return nil
}

// Provider is the name this store is recorded under.
func (s *Store) Provider() string { return "mem0" }

// Client exposes the HTTP client, so a caller can reach parts of mem0's API this does not
// wrap without building a second client.
func (s *Store) Client() *http.Client { return s.client }

// Close releases nothing: the store holds no connection of its own.
func (s *Store) Close() error { return nil }

func (s *Store) call(ctx context.Context, path string, body, into any) error {
	encoded, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("mem0: encode %s: %w", path, err)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+path, bytes.NewReader(encoded))
	if err != nil {
		return fmt.Errorf("mem0: %s: %w", path, err)
	}
	request.Header.Set("Authorization", "Token "+s.apiKey)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	response, err := s.client.Do(request)
	if err != nil {
		return fmt.Errorf("mem0: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("mem0: %s: %s: %s", path, response.Status, strings.TrimSpace(string(detail)))
	}

	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("mem0: decode %s: %w", path, err)
	}
	return nil
}

// filtersFor scopes a search to who the memories belong to. An app id narrows it further,
// which is what keeps two deployments sharing one mem0 account apart.
func filtersFor(scope memory.Scope) map[string]string {
	filters := make(map[string]string, len(scope.Extra)+2)
	// The caller's own labels go in first, so neither of the two identities below can be
	// overwritten by one: a filter that could rewrite the user id would read somebody
	// else's memories.
	for key, value := range scope.Extra {
		filters[key] = value
	}
	filters["user_id"] = scope.UserID
	if scope.AppID != "" {
		filters["app_id"] = scope.AppID
	}
	return filters
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type addRequest struct {
	Messages []message `json:"messages"`
	UserID   string    `json:"user_id"`
	AppID    string    `json:"app_id,omitempty"`
}

type addResponse struct {
	EventID string `json:"event_id"`
	Status  string `json:"status"`
}

type searchRequest struct {
	Query   string            `json:"query"`
	Filters map[string]string `json:"filters"`
	TopK    int               `json:"top_k,omitempty"`
}

type searchResponse struct {
	Results []struct {
		ID     string  `json:"id"`
		Memory string  `json:"memory"`
		Score  float64 `json:"score"`
	} `json:"results"`
}
