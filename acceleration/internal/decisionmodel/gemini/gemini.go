// Package gemini asks Google's Gemini Decision models for typed judgements.
//
// Google's Decisions API asks the same three questions as System One and answers them in
// the same fields, but the request is its own: the state is `input`, a noul's yes and no
// are `true_description` and `false_description`, the key goes in `x-goog-api-key`, and
// usage is counted as `total_input_tokens`. So this is a decisionmodel.Provider of its own
// rather than a systemone.Endpoint, and the translation stays in this package.
package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ProviderName is how this is named in stats and configuration, the name every other
// Gemini API provider here goes by.
const ProviderName = "gemini"

const (
	apiKeyEnvVar  = "GOOGLE_API_KEY"
	baseURLEnvVar = "GEMINI_BASE_URL"
	defaultURL    = "https://generativelanguage.googleapis.com"
	path          = "/v1beta/decisions"
	// DefaultModel is the only model the Decisions API serves.
	DefaultModel = "gemini-decision-1"
)

// defaultTimeout bounds one request, as systemone does: on the live path a judgement that
// has not arrived is worth less than the pause it costs.
const defaultTimeout = 6 * time.Second

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// StatusError is a request the API refused, with its status so a caller can tell a bad
// question from a rate limit.
type StatusError struct {
	StatusCode int
	Body       string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("%s: the API returned %d: %s", ProviderName, e.StatusCode, e.Body)
}

// Unwrap says which of decisionmodel's failures this is. Google answers an overloaded or
// failing backend with 500, 503 or 504 and says to retry all three. A 402 is a project
// out of prepaid credit, which waiting does not fix.
func (e *StatusError) Unwrap() error {
	switch e.StatusCode {
	case http.StatusTooManyRequests:
		return decisionmodel.ErrRateLimited
	case http.StatusInternalServerError, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return decisionmodel.ErrUnavailable
	}
	return nil
}

// Options configures a Client. The key falls back to GOOGLE_API_KEY, which the other
// Gemini providers read too.
type Options struct {
	APIKey string
	// Model is which model answers. Empty means DefaultModel.
	Model string
	// BaseURL replaces the API host, which is what a test or a proxy needs. Empty reads
	// GEMINI_BASE_URL, then the Gemini API's own.
	BaseURL string
	// Timeout bounds one request. Zero means defaultTimeout.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
}

// Client asks one Gemini Decision model for judgements. It satisfies decisionmodel.Provider.
type Client struct {
	apiKey string
	model  string
	url    string
	client *http.Client
}

// New validates the options and returns a Client.
func New(options Options) (*Client, error) {
	keys := keypool.FromEnv(options.APIKey, apiKeyEnvVar)
	if keys.Len() == 0 {
		return nil, stack.Wrap(errors.New(ProviderName + ": " + apiKeyEnvVar + " is required"))
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(baseURLEnvVar)
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultURL
	}
	if options.Timeout <= 0 {
		options.Timeout = defaultTimeout
	}
	if options.HTTPClient == nil {
		options.HTTPClient = &http.Client{Timeout: options.Timeout}
	}

	return &Client{
		apiKey: keys.Peek(),
		model:  options.Model,
		url:    strings.TrimSuffix(options.BaseURL, "/") + path,
		client: keys.WrapClient(options.HTTPClient, keypool.Header("x-goog-api-key", "")),
	}, nil
}

// Provider is how this is named in stats.
func (c *Client) Provider() string { return ProviderName }

// Model is the model this asks.
func (c *Client) Model() string { return c.model }

// Start opens nothing: every judgement is one request.
func (c *Client) Start(context.Context) error { return nil }

// Close releases nothing, for the same reason.
func (c *Client) Close() error { return nil }

type request struct {
	Model     string                  `json:"model"`
	Input     string                  `json:"input"`
	Questions map[string]wireQuestion `json:"questions"`
}

type wireQuestion struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

// noulCriteria is what yes and no mean, under Google's names for them.
type noulCriteria struct {
	True  string `json:"true_description,omitempty"`
	False string `json:"false_description,omitempty"`
}

type wireAnswer struct {
	Type          string             `json:"type"`
	Chosen        string             `json:"choice"`
	Yes           float64            `json:"noul"`
	Level         float64            `json:"score"`
	Legend        map[string]string  `json:"legend"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
}

type wireResponse struct {
	Answers map[string]wireAnswer `json:"answers"`
	Usage   struct {
		InputTokens  int64 `json:"total_input_tokens"`
		OutputTokens int64 `json:"total_output_tokens"`
	} `json:"usage"`
	PromptFeedback struct {
		BlockReason string `json:"block_reason"`
	} `json:"prompt_feedback"`
}

// input is the state as text. Google reads a JSON object put in `input` as content blocks
// rather than data, so a state with parts goes as its JSON, which keeps the names a
// question points at in backticks.
func input(state any) (string, error) {
	if text, ok := state.(string); ok {
		return text, nil
	}
	encoded, err := json.Marshal(state)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

// criteria is a question's criteria as Google takes them. Only a noul's differ.
func criteria(question decisionmodel.Question) any {
	if question.Type != decisionmodel.TypeNoul {
		return question.Criteria
	}
	meanings, _ := question.Criteria.(map[string]string)
	if meanings["true"] == "" && meanings["false"] == "" {
		return nil
	}
	return noulCriteria{True: meanings["true"], False: meanings["false"]}
}

// Classify puts every question to the model in one request and hands back the answers.
func (c *Client) Classify(
	ctx context.Context, asked decisionmodel.Request,
) (decisionmodel.Result, error) {
	if err := asked.Validate(); err != nil {
		return decisionmodel.Result{}, stack.Wrap(err)
	}
	state, err := input(asked.State)
	if err != nil {
		return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: encode state: %w", ProviderName, err))
	}
	if strings.TrimSpace(state) == "" {
		return decisionmodel.Result{}, stack.Wrap(errors.New(ProviderName + ": the state is empty"))
	}

	questions := make(map[string]wireQuestion, len(asked.Questions))
	for id, question := range asked.Questions {
		questions[id] = wireQuestion{
			Type:         string(question.Type),
			Instructions: question.Instructions,
			Criteria:     criteria(question),
		}
	}
	payload, err := json.Marshal(request{Model: c.model, Input: state, Questions: questions})
	if err != nil {
		return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: encode questions: %w", ProviderName, err))
	}

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(payload))
	if err != nil {
		return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: build request: %w", ProviderName, err))
	}
	httpRequest.Header.Set("x-goog-api-key", c.apiKey)
	httpRequest.Header.Set("Content-Type", "application/json")

	httpResponse, err := c.client.Do(httpRequest)
	if err != nil {
		return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: classify: %w: %w", ProviderName, decisionmodel.ErrUnavailable, err))
	}
	defer httpResponse.Body.Close()

	if httpResponse.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(httpResponse.Body, errorBodyLimit))
		return decisionmodel.Result{}, stack.Wrap(&StatusError{
			StatusCode: httpResponse.StatusCode,
			Body:       strings.TrimSpace(string(body)),
		})
	}

	var decoded wireResponse
	if err := json.NewDecoder(httpResponse.Body).Decode(&decoded); err != nil {
		return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: decode answers: %w", ProviderName, err))
	}
	if reason := decoded.PromptFeedback.BlockReason; reason != "" {
		return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: the input was blocked (%s)", ProviderName, reason))
	}

	// Google does not say which version answered. The model asked is a version rather
	// than an alias, so it is that one.
	result := decisionmodel.Result{
		Model:   c.model,
		Answers: make(map[string]decisionmodel.Answer, len(asked.Questions)),
		Usage: decisionmodel.Usage{
			InputTokens:  decoded.Usage.InputTokens,
			OutputTokens: decoded.Usage.OutputTokens,
		},
	}
	for id := range asked.Questions {
		answer, answered := decoded.Answers[id]
		if !answered {
			return decisionmodel.Result{}, stack.Wrap(fmt.Errorf("%s: %q was not answered", ProviderName, id))
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
