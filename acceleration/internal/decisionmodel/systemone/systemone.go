// Package systemone speaks the System One wire protocol, which is what every decision model
// vendor here answers on.
//
// TypeSafe published it for Jev, and OpenRouter and Perplexity adopted it as is: a request
// carries a state and a map of named questions, and the answer to each is a value with a
// probability distribution behind it. There is no stream and no generated text, which is
// why this speaks the REST endpoint directly rather than wrapping an SDK. What differs
// between vendors is only where the endpoint is and which key opens it, and that is an
// Endpoint.
package systemone

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

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// defaultTimeout bounds one request. This is meant for the live path, where a judgement that
// has not arrived is worth less than the pause it is costing.
const defaultTimeout = 6 * time.Second

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// Endpoint is one vendor's System One API.
type Endpoint struct {
	// Provider is the stable name used in stats and configuration, e.g. "typesafe".
	Provider string
	// APIKeyEnvVar is where the key is read from when none is given.
	APIKeyEnvVar string
	// BaseURLEnvVar overrides BaseURL, which a proxy or a test needs.
	BaseURLEnvVar string
	BaseURL       string
	// Path is where the questions are posted, e.g. "/v1/systemone".
	Path string
	// DefaultModel is asked when the options name none.
	DefaultModel string
}

// StatusError is a request the API refused. It carries the status so a caller can tell a bad
// question from a rate limit and back off rather than give up.
type StatusError struct {
	Provider   string
	StatusCode int
	Body       string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("%s: the API returned %d: %s", e.Provider, e.StatusCode, e.Body)
}

// Unwrap says which of decisionmodel's failures this is, so a caller can back off without
// knowing a vendor's status codes. A status that is neither is the request's own fault.
func (e *StatusError) Unwrap() error {
	switch e.StatusCode {
	case http.StatusTooManyRequests:
		return decisionmodel.ErrRateLimited
	case http.StatusServiceUnavailable, 529:
		return decisionmodel.ErrUnavailable
	}
	return nil
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

// Options configures a Client. The key falls back to the endpoint's environment variable,
// the way every other provider in this service is configured.
type Options struct {
	// APIKey defaults to the endpoint's APIKeyEnvVar.
	APIKey string
	// Model is which model answers. Empty means the endpoint's DefaultModel.
	Model string
	// BaseURL replaces the endpoint's host, which is what a test or a proxy needs.
	BaseURL string
	// Timeout bounds one request. Zero means defaultTimeout.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
	Logger     *slog.Logger
}

// Client asks one model on one vendor's endpoint for judgements.
type Client struct {
	provider string
	apiKey   string
	model    string
	url      string
	client   *http.Client
	logger   *slog.Logger
}

// New validates the options and returns a Client for endpoint.
func New(endpoint Endpoint, options Options) (*Client, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(endpoint.APIKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New(endpoint.Provider + ": " + endpoint.APIKeyEnvVar + " is required"))
	}
	if options.Model == "" {
		options.Model = endpoint.DefaultModel
	}
	if options.Model == "" {
		return nil, stack.Wrap(errors.New(endpoint.Provider + ": a model is required"))
	}
	if options.BaseURL == "" && endpoint.BaseURLEnvVar != "" {
		options.BaseURL = os.Getenv(endpoint.BaseURLEnvVar)
	}
	if options.BaseURL == "" {
		options.BaseURL = endpoint.BaseURL
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
		provider: endpoint.Provider,
		apiKey:   options.APIKey,
		model:    options.Model,
		url:      strings.TrimSuffix(options.BaseURL, "/") + endpoint.Path,
		client:   options.HTTPClient,
		logger:   options.Logger,
	}, nil
}

// Provider is how this is named in stats.
func (c *Client) Provider() string { return c.provider }

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

// request is what goes upstream. Perplexity refuses a field it does not know, so nothing
// goes here that every vendor does not take.
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
// the question's type, and the names are the protocol's rather than ours.
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
	ctx context.Context, asked decisionmodel.Request,
) (decisionmodel.Result, error) {
	if err := asked.Validate(); err != nil {
		return decisionmodel.Result{}, stack.Wrap(err)
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
		return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: encode questions: %w", c.provider, err))
	}

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(payload))
	if err != nil {
		return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: build request: %w", c.provider, err))
	}
	httpRequest.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpRequest.Header.Set("Content-Type", "application/json")

	httpResponse, err := c.client.Do(httpRequest)
	if err != nil {
		return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: classify: %w: %w", c.provider, decisionmodel.ErrUnavailable, err))
	}
	defer httpResponse.Body.Close()

	if httpResponse.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(httpResponse.Body, errorBodyLimit))
		return decisionmodel.Result{}, stack.Wrap(&StatusError{
			Provider:   c.provider,
			StatusCode: httpResponse.StatusCode,
			Body:       strings.TrimSpace(string(body)),
		})
	}

	var decoded wireResponse
	if err := json.NewDecoder(httpResponse.Body).Decode(&decoded); err != nil {
		return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: decode answers: %w", c.provider, err))
	}

	result := decisionmodel.Result{
		Model:   decoded.Model,
		Answers: make(map[string]decisionmodel.Answer, len(asked.Questions)),
		Usage: decisionmodel.Usage{
			InputTokens:  decoded.Usage.InputTokens,
			OutputTokens: decoded.Usage.OutputTokens,
		},
	}
	for id := range asked.Questions {
		answer, answered := decoded.Answers[id]
		if !answered {
			return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: %q was not answered", c.provider, id))
		}
		result.Answers[id] = decisionmodel.Answer{
			Type:          decisionmodel.QuestionType(answer.Type),
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
