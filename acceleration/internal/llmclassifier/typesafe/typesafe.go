// Package typesafe asks TypeSafe's System One models for typed judgements.
//
// It is not an LLM provider and deliberately does not pretend to be one. There is no stream
// and no generated text: a request carries a state and a map of named questions, and the
// answer to each is a value with a probability distribution behind it. That is the whole of
// the API, which is why this speaks its REST endpoint directly rather than wrapping an SDK.
//
// Jev is the model; TypeSafe is the vendor, which is why this is registered under the
// vendor's name with the model beside it, the way every other provider here is.
package typesafe

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

	"github.com/GetStream/Vision-Agents/acceleration/internal/llmclassifier"
)

// ProviderName is how this is named in stats and configuration.
const ProviderName = "typesafe"

const apiKeyEnvVar = "TYPESAFE_API_KEY"

// baseURLEnvVar overrides the endpoint, which a proxy or a test needs.
const baseURLEnvVar = "TYPESAFE_BASE_URL"

const defaultBaseURL = "https://api.typesafe.ai"

// DefaultModel is the alias for the newest stable System One model. It moves when a release
// ships, so anything with thresholds tuned against one version should name that version.
const DefaultModel = "jev-latest"

// defaultTimeout bounds one request. This is meant for the live path, where a judgement that
// has not arrived is worth less than the pause it is costing.
const defaultTimeout = 6 * time.Second

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// StatusError is a request the API refused. It carries the status so a caller can tell a bad
// question from a rate limit and back off rather than give up.
type StatusError struct {
	StatusCode int
	Body       string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("typesafe: the API returned %d: %s", e.StatusCode, e.Body)
}

// Retryable reports whether waiting and asking again is worth it, which is a rate limit or an
// overloaded service and nothing else. A malformed question does not improve on a second try.
func (e *StatusError) Retryable() bool {
	// 503 is what TypeSafe answers with model_unavailable while a model is being moved
	// or is out of capacity. It is the overloaded case rather than a rejected question,
	// and it comes back on its own.
	return e.StatusCode == http.StatusTooManyRequests ||
		e.StatusCode == http.StatusServiceUnavailable ||
		e.StatusCode == 529
}

// Options configures a Client. The key falls back to the environment, the way every other
// provider in this service is configured.
type Options struct {
	// APIKey defaults to TYPESAFE_API_KEY.
	APIKey string
	// Model is which System One model answers. Empty means DefaultModel.
	Model string
	// BaseURL replaces the default host, which is what a test or a proxy needs.
	BaseURL string
	// Timeout bounds one request. Zero means defaultTimeout.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
	Logger     *slog.Logger
}

// Client asks one System One model for judgements.
type Client struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
	logger  *slog.Logger
}

// New validates the options and returns a Client.
func New(options Options) (*Client, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("typesafe: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(baseURLEnvVar)
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

	return &Client{
		apiKey:  options.APIKey,
		model:   options.Model,
		baseURL: strings.TrimSuffix(options.BaseURL, "/"),
		client:  options.HTTPClient,
		logger:  options.Logger,
	}, nil
}

// Provider is how this is named in stats.
func (c *Client) Provider() string { return ProviderName }

// Model is the model or alias this asks.
func (c *Client) Model() string { return c.model }

// Start opens nothing. There is no session to establish: every judgement is one request,
// and a key that does not work says so when the first one is asked rather than now.
func (c *Client) Start(ctx context.Context) error { return nil }

// Close releases nothing, for the same reason.
func (c *Client) Close() error { return nil }

// HTTPClient hands back the client underneath, so a caller who needs a transport, a proxy or
// a header this does not standardise is not stuck behind it.
func (c *Client) HTTPClient() *http.Client { return c.client }

// request is what goes upstream.
type request struct {
	State     any                     `json:"state"`
	Model     string                  `json:"model"`
	Questions map[string]wireQuestion `json:"questions"`
}

// wireQuestion is a question as the API takes it.
type wireQuestion struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

// wireAnswer is an answer as the API returns it. Which field carries the answer depends on
// the question's type, and the names are the vendor's rather than ours.
type wireAnswer struct {
	Type          string             `json:"type"`
	Chosen        string             `json:"choice"`
	Yes           float64            `json:"noul"`
	Level         float64            `json:"score"`
	Legend        map[string]string  `json:"legend"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
}

// wireResponse is one request's answers.
type wireResponse struct {
	Model   string                `json:"model"`
	Answers map[string]wireAnswer `json:"answers"`
	Usage   struct {
		InputTokens  int64 `json:"input_tokens"`
		OutputTokens int64 `json:"output_tokens"`
	} `json:"usage"`
}

// Classify puts every question to the model at once and hands back the answers.
//
// They are all asked together on purpose. Questions in one request are evaluated in parallel
// and cannot see each other's answers, so they cost the state's tokens once between them
// rather than once each. That makes asking a question whose answer may turn out to be
// irrelevant close to free, and it is why a caller should ask everything it might need and
// let its own code decide what applies.
func (c *Client) Classify(
	ctx context.Context, asked llmclassifier.Request,
) (llmclassifier.Result, error) {
	if err := asked.Validate(); err != nil {
		return llmclassifier.Result{}, err
	}

	questions := make(map[string]wireQuestion, len(asked.Questions))
	for id, question := range asked.Questions {
		questions[id] = wireQuestion{
			Type:         string(question.Type),
			Instructions: question.Instructions,
			Criteria:     question.Criteria,
		}
	}

	payload, err := json.Marshal(request{State: asked.State, Model: c.model, Questions: questions})
	if err != nil {
		return llmclassifier.Result{}, fmt.Errorf("typesafe: encode questions: %w", err)
	}

	httpRequest, err := http.NewRequestWithContext(
		ctx, http.MethodPost, c.baseURL+"/v1/systemone", bytes.NewReader(payload))
	if err != nil {
		return llmclassifier.Result{}, fmt.Errorf("typesafe: build request: %w", err)
	}
	httpRequest.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpRequest.Header.Set("Content-Type", "application/json")

	httpResponse, err := c.client.Do(httpRequest)
	if err != nil {
		return llmclassifier.Result{}, fmt.Errorf("typesafe: classify: %w", err)
	}
	defer httpResponse.Body.Close()

	if httpResponse.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(httpResponse.Body, errorBodyLimit))
		return llmclassifier.Result{}, &StatusError{
			StatusCode: httpResponse.StatusCode,
			Body:       strings.TrimSpace(string(body)),
		}
	}

	var decoded wireResponse
	if err := json.NewDecoder(httpResponse.Body).Decode(&decoded); err != nil {
		return llmclassifier.Result{}, fmt.Errorf("typesafe: decode answers: %w", err)
	}

	result := llmclassifier.Result{
		Model:   decoded.Model,
		Answers: make(map[string]llmclassifier.Answer, len(asked.Questions)),
		Usage: llmclassifier.Usage{
			InputTokens:  decoded.Usage.InputTokens,
			OutputTokens: decoded.Usage.OutputTokens,
		},
	}
	for id := range asked.Questions {
		answer, answered := decoded.Answers[id]
		if !answered {
			return llmclassifier.Result{}, fmt.Errorf("typesafe: %q was not answered", id)
		}
		result.Answers[id] = llmclassifier.Answer{
			Type:          llmclassifier.QuestionType(answer.Type),
			Chosen:        answer.Chosen,
			Yes:           answer.Yes,
			Level:         answer.Level,
			Legend:        answer.Legend,
			Probabilities: answer.Probabilities,
			Confidence:    answer.Confidence,
		}
	}
	return result, nil
}
