// Package mem0 stores an agent's memories in mem0's hosted platform.
//
// mem0 publishes no Go SDK, so this speaks its REST API directly: one call to hand a
// conversation over, one to ask what is known, and a list and a batch delete to forget.
// Extraction is asynchronous on mem0's side, so adding returns an event id rather than the
// memories it will produce.
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
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tracing"
)

const apiKeyEnvVar = "MEM0_API_KEY"

const (
	defaultBaseURL = "https://api.mem0.ai"
	defaultTopK    = 5
	defaultTimeout = 10 * time.Second
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// deletePageSize is how many memories are deleted per batch, well under mem0's 1000.
const deletePageSize = 100

// The calls this store makes, as its metrics, spans and log name them.
const (
	opSearch = "search"
	opAdd    = "add"
	opList   = "list"
	opDelete = "delete"
)

// How a call ended, as its metrics name it.
const (
	outcomeOK        = "ok"
	outcomeHTTPError = "http_error"
	outcomeTransport = "transport_error"
	outcomeDecode    = "decode_error"
)

// tracer records each call to mem0 under the caller's span.
var tracer = tracing.Tracer("memory/mem0")

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

// Store is a mem0 memory store. It satisfies memory.Backend: mem0 bills by request, so every
// call says how many it made.
type Store struct {
	apiKey  string
	baseURL string
	client  *http.Client
	logger  *slog.Logger
}

// New validates the options and returns a Store.
func New(options Options) (*Store, error) {
	keys := keypool.FromEnv(options.APIKey, apiKeyEnvVar)
	if keys.Len() == 0 {
		return nil, errors.New("mem0: " + apiKeyEnvVar + " is required")
	}
	options.APIKey = keys.Peek()
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
		client:  keys.WrapClient(options.HTTPClient, keypool.Header("Authorization", "Token ")),
		logger:  options.Logger,
	}, nil
}

// Recall returns what mem0 knows that bears on the query, most relevant first.
func (s *Store) Recall(ctx context.Context, query memory.Query) (memory.Recalled, error) {
	if err := query.Scope.Validate(); err != nil {
		return memory.Recalled{}, err
	}

	limit := query.Limit
	if limit <= 0 {
		limit = defaultTopK
	}
	text := query.Text
	if strings.TrimSpace(text) == "" {
		// Joining recalls context before the first question; v3 rejects a blank query.
		text = "Relevant facts, preferences, and context from previous conversations."
	}

	// v3 wants the entity ids inside filters; at the top level they are rejected.
	body := searchRequest{
		Query:   text,
		Filters: filtersFor(query.Scope),
		TopK:    limit,
	}

	var response searchResponse
	if err := s.call(ctx, opSearch, query.Scope, http.MethodPost, "/v3/memories/search/", body, &response); err != nil {
		return memory.Recalled{}, err
	}

	recalled := memory.Recalled{Memories: make([]memory.Memory, 0, len(response.Results)), Billed: memory.Billed{Requests: 1}}
	for _, result := range response.Results {
		if result.Memory == "" {
			continue
		}
		recalled.Memories = append(recalled.Memories, memory.Memory{ID: result.ID, Text: result.Memory, Score: result.Score})
	}
	return recalled, nil
}

// Remember hands a conversation to mem0 to learn from. Extraction runs on mem0's side, so
// this returns as soon as the work is queued rather than when memories exist.
func (s *Store) Remember(ctx context.Context, scope memory.Scope, messages []llm.Message) (memory.Billed, error) {
	if err := scope.Validate(); err != nil {
		return memory.Billed{}, err
	}
	if len(messages) == 0 {
		return memory.Billed{}, nil
	}

	body := addRequest{
		Messages: make([]message, 0, len(messages)),
		UserID:   scope.UserID,
		AppID:    scope.AppID,
		AgentID:  scope.AgentID,
		RunID:    scope.RunID,
		// The caller's labels are written as metadata, which is the only place a search
		// can later filter on them.
		Metadata: scope.Extra,
	}
	for _, said := range messages {
		if said.Content == "" {
			continue
		}
		body.Messages = append(body.Messages, message{Role: string(said.Role), Content: said.Content})
	}
	if len(body.Messages) == 0 {
		return memory.Billed{}, nil
	}

	var response addResponse
	if err := s.call(ctx, opAdd, scope, http.MethodPost, "/v3/memories/add/", body, &response); err != nil {
		return memory.Billed{}, err
	}
	s.logger.Debug("queued a conversation to remember", "event", response.EventID, "status", response.Status)
	return memory.Billed{Requests: 1}, nil
}

// Truncate deletes everything mem0 knows about one user of one app.
func (s *Store) Truncate(ctx context.Context, appID, userID string) (memory.Billed, error) {
	if appID == "" || userID == "" {
		return memory.Billed{}, stack.Wrap(errors.New("mem0: truncating needs an app id and a user id"))
	}
	return s.deleteMatching(ctx, memory.Scope{AppID: appID, UserID: userID},
		map[string]any{"app_id": appID, "user_id": userID})
}

// ForgetRun deletes what one session of one app taught mem0.
func (s *Store) ForgetRun(ctx context.Context, appID, runID string) (memory.Billed, error) {
	if appID == "" || runID == "" {
		return memory.Billed{}, stack.Wrap(errors.New("mem0: forgetting a session needs an app id and a run id"))
	}
	return s.deleteMatching(ctx, memory.Scope{AppID: appID, RunID: runID},
		map[string]any{"app_id": appID, "run_id": runID})
}

// deleteMatching deletes every memory the filters match, a page at a time, and counts the
// requests it made: each listing and each batch delete. who is whose memories they are, for
// the log.
//
// mem0's delete-all endpoint takes one entity and ignores the rest, so asked for an app and
// a run it deletes the whole app. Listing by filter and deleting by id is the only way to
// delete exactly what was asked for.
func (s *Store) deleteMatching(ctx context.Context, who memory.Scope, filters map[string]any) (memory.Billed, error) {
	var billed memory.Billed
	deleted := map[string]bool{}
	for {
		var page listResponse
		path := fmt.Sprintf("/v3/memories/?page=1&page_size=%d", deletePageSize)
		if err := s.call(ctx, opList, who, http.MethodPost, path, listRequest{Filters: filters}, &page); err != nil {
			return billed, err
		}
		billed.Requests++
		if len(page.Results) == 0 {
			if len(deleted) == 0 {
				emptyLists.Inc()
			}
			return billed, nil
		}

		batch := batchDeleteRequest{Memories: make([]batchMemory, 0, len(page.Results))}
		for _, found := range page.Results {
			// A memory listed again after being deleted would loop here for ever.
			if deleted[found.ID] {
				return billed, stack.Wrap(fmt.Errorf("mem0: memory %s is still listed after being deleted", found.ID))
			}
			deleted[found.ID] = true
			batch.Memories = append(batch.Memories, batchMemory{MemoryID: found.ID})
		}
		var response batchDeleteResponse
		if err := s.call(ctx, opDelete, who, http.MethodDelete, "/v1/batch/", batch, &response); err != nil {
			return billed, err
		}
		billed.Requests++
	}
}

// Provider is the name this store is recorded under.
func (s *Store) Provider() string { return "mem0" }

// Client exposes the HTTP client, so a caller can reach parts of mem0's API this does not
// wrap without building a second client.
func (s *Store) Client() *http.Client { return s.client }

// Close releases nothing: the store holds no connection of its own.
func (s *Store) Close() error { return nil }

// call sends one request to mem0 and records it as a metric, a span and a debug line. A
// failure is not logged here: every caller logs the error returned, which carries what mem0
// answered.
func (s *Store) call(ctx context.Context, op string, who memory.Scope, method, path string, body, into any) error {
	encoded, err := json.Marshal(body)
	if err != nil {
		return stack.Wrap(fmt.Errorf("mem0: encode %s: %w", path, err))
	}

	ctx, span := tracer.Start(ctx, "mem0."+op, trace.WithAttributes(attribute.String("mem0.op", op)))
	defer span.End()
	started := time.Now()
	status, outcome, err := s.send(ctx, method, path, encoded, into)
	elapsed := time.Since(started)

	statusLabel := "none"
	if status != 0 {
		statusLabel = strconv.Itoa(status)
		span.SetAttributes(attribute.Int("http.response.status_code", status))
	}
	calls.WithLabelValues(op, statusLabel, outcome).Inc()
	callSeconds.WithLabelValues(op, outcome).Observe(elapsed.Seconds())

	fields := []any{"op", op, "app_id", who.AppID}
	if who.UserID != "" {
		fields = append(fields, "user_id", who.UserID)
	}
	if who.RunID != "" {
		fields = append(fields, "run_id", who.RunID)
	}
	fields = append(fields, "status", statusLabel, "outcome", outcome, "latency_ms", elapsed.Milliseconds())

	switch {
	case outcome == outcomeHTTPError:
		// The status alone, not the error: mem0's body may repeat what it was sent.
		span.SetStatus(codes.Error, http.StatusText(status))
	case err != nil:
		tracing.Fail(span, err)
	default:
		if count, counted := resultCount(into); counted {
			span.SetAttributes(attribute.Int("mem0.results", count))
			fields = append(fields, "results", count)
		}
	}
	s.logger.Debug("called mem0", fields...)
	return err
}

// send makes one request and decodes mem0's answer into into. The status is mem0's, zero
// when it never answered.
func (s *Store) send(ctx context.Context, method, path string, encoded []byte, into any) (int, string, error) {
	request, err := http.NewRequestWithContext(ctx, method, s.baseURL+path, bytes.NewReader(encoded))
	if err != nil {
		return 0, outcomeTransport, stack.Wrap(fmt.Errorf("mem0: %s: %w", path, err))
	}
	request.Header.Set("Authorization", "Token "+s.apiKey)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	response, err := s.client.Do(request)
	if err != nil {
		return 0, outcomeTransport, stack.Wrap(fmt.Errorf("mem0: %s: %w", path, err))
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		// mem0 answers a spent billing-period quota with 429 as well as a burst, so the
		// error names both.
		limited := ""
		if response.StatusCode == http.StatusTooManyRequests {
			limited = " (mem0 quota or rate limit)"
		}
		return response.StatusCode, outcomeHTTPError, stack.Wrap(fmt.Errorf("mem0: %s: %s%s: %s",
			path, response.Status, limited, strings.TrimSpace(string(detail))))
	}

	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return response.StatusCode, outcomeDecode, stack.Wrap(fmt.Errorf("mem0: decode %s: %w", path, err))
	}
	return response.StatusCode, outcomeOK, nil
}

// resultCount is how many memories a search or a list answered with. Other answers hold
// none, and say so.
func resultCount(into any) (int, bool) {
	switch decoded := into.(type) {
	case *searchResponse:
		return len(decoded.Results), true
	case *listResponse:
		return len(decoded.Results), true
	}
	return 0, false
}

// filtersFor scopes a search to who the memories belong to. An app id narrows it further,
// which is what keeps two deployments sharing one mem0 account apart.
//
// The agent and run ids are left out. mem0 files what it learns about the user with no
// agent id, so filtering on one finds nothing, and filtering on the run would recall only
// what this session already knows.
func filtersFor(scope memory.Scope) map[string]any {
	filters := map[string]any{
		"user_id": scope.UserID,
		"app_id":  scope.AppID,
	}
	// v3 refuses unknown keys at the top level, so the caller's labels can only ever
	// narrow the search through metadata, never rewrite the ids above.
	if len(scope.Extra) > 0 {
		filters["metadata"] = scope.Extra
	}
	return filters
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type addRequest struct {
	Messages []message         `json:"messages"`
	UserID   string            `json:"user_id"`
	AppID    string            `json:"app_id"`
	AgentID  string            `json:"agent_id"`
	RunID    string            `json:"run_id"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type addResponse struct {
	EventID string `json:"event_id"`
	Status  string `json:"status"`
}

type searchRequest struct {
	Query   string         `json:"query"`
	Filters map[string]any `json:"filters"`
	TopK    int            `json:"top_k,omitempty"`
}

type searchResponse struct {
	Results []struct {
		ID     string  `json:"id"`
		Memory string  `json:"memory"`
		Score  float64 `json:"score"`
	} `json:"results"`
}

type listRequest struct {
	Filters map[string]any `json:"filters"`
}

type listResponse struct {
	Results []struct {
		ID string `json:"id"`
	} `json:"results"`
}

type batchMemory struct {
	MemoryID string `json:"memory_id"`
}

type batchDeleteRequest struct {
	Memories []batchMemory `json:"memories"`
}

type batchDeleteResponse struct {
	Message string `json:"message"`
}
