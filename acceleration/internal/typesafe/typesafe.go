// Package typesafe asks TypeSafe's System One models for typed judgements.
//
// It is not an LLM provider and deliberately does not pretend to be one. There is no stream
// and no generated text: a request carries a state and a map of named questions, and the
// answer to each is a value with a probability distribution behind it. That is the whole of
// the API, which is why this speaks its REST endpoint directly rather than wrapping an SDK.
//
// What it is for here is the flow controller. Deciding whether a caller has finished, whether
// they were talking to the agent, and whether they have taken the floor is three small
// classifications that a model returning JSON has to be talked into and a model returning a
// distribution simply answers. Whether it answers them better is
// [the flow controller benchmark](../../../.factory/evals/flow-controller.md)'s question.
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

// QuestionType is the shape of the answer a question asks for.
type QuestionType string

const (
	// TypeChoice picks one of a named set of options.
	TypeChoice QuestionType = "choice"
	// TypeScore places the state along ordered levels.
	TypeScore QuestionType = "score"
	// TypeNoul answers yes or no, as the probability of yes.
	TypeNoul QuestionType = "noul"
)

// Question is one judgement to make about the state.
//
// Build one with Choice, Score or Noul. Criteria means something different for each of the
// three, and the constructors are what keep the pairing honest.
type Question struct {
	Type         QuestionType `json:"type"`
	Instructions string       `json:"instructions"`
	// Criteria is the possible answers: the options of a choice, the ordered levels of a
	// score, or what yes and no mean for a noul. Omitted where a noul needs no gloss.
	Criteria any `json:"criteria,omitempty"`
}

// Choice asks which of options fits. Each option carries a description of what it covers, or
// an empty string where the name says it. Include something for "none of these" whenever the
// options may not cover an input, because the model can only answer with what it was given.
func Choice(instructions string, options map[string]string) Question {
	criteria := make(map[string]*string, len(options))
	for option, described := range options {
		if described == "" {
			criteria[option] = nil
			continue
		}
		criteria[option] = &described
	}
	return Question{Type: TypeChoice, Instructions: instructions, Criteria: criteria}
}

// Score asks where the state falls along levels, given in order. Each level has to describe a
// concrete situation and stand on its own, because the answer can land between two of them.
func Score(instructions string, levels []string) Question {
	return Question{Type: TypeScore, Instructions: instructions, Criteria: levels}
}

// Noul asks a yes or no question. Yes and no describe what each end means, and either may be
// empty when the question says it plainly enough.
func Noul(instructions, yes, no string) Question {
	question := Question{Type: TypeNoul, Instructions: instructions}
	if yes != "" || no != "" {
		question.Criteria = map[string]string{"true": yes, "false": no}
	}
	return question
}

// Answer is one judgement. Which fields carry it depends on Type: Choice fills Chosen,
// Probabilities and Confidence, Score fills Level, Legend, Probabilities and Confidence, and
// Noul fills Yes alone, because the probability is the answer and has no confidence beside it.
type Answer struct {
	Type QuestionType `json:"type"`
	// Chosen is the likeliest option of a choice.
	Chosen string `json:"choice"`
	// Yes is the probability a noul is true, from 0 to 1.
	Yes float64 `json:"noul"`
	// Level is where a score landed, which may be between two of the levels asked about.
	Level float64 `json:"score"`
	// Legend names a score's levels by their index, as decimal strings.
	Legend map[string]string `json:"legend"`
	// Probabilities is the distribution the answer came from: options for a choice, level
	// indices for a score. They sum to one.
	Probabilities map[string]float64 `json:"probabilities"`
	// Confidence says how peaked that distribution is, not whether acting on it is safe.
	Confidence float64 `json:"confidence"`
}

// Usage is what one request cost. Only input tokens are billed.
type Usage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
}

// Response is the answers to one request, under the ids they were asked under.
type Response struct {
	// Model is the version that answered, which is worth recording when the request named
	// an alias.
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   Usage             `json:"usage"`
}

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
	return e.StatusCode == http.StatusTooManyRequests || e.StatusCode == 529
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

// Model is the model or alias this asks.
func (c *Client) Model() string { return c.model }

// HTTPClient hands back the client underneath, so a caller who needs a transport, a proxy or
// a header this does not standardise is not stuck behind it.
func (c *Client) HTTPClient() *http.Client { return c.client }

// request is what goes upstream.
type request struct {
	State     any                 `json:"state"`
	Model     string              `json:"model"`
	Questions map[string]Question `json:"questions"`
}

// Ask puts every question to the model at once and hands back the answers.
//
// They are all asked together on purpose. Questions in one request are evaluated in parallel
// and cannot see each other's answers, so they cost the state's tokens once between them
// rather than once each. That makes asking a question whose answer may turn out to be
// irrelevant close to free, and it is why a caller should ask everything it might need and
// let its own code decide what applies.
//
// State is a string for plain text, or anything that marshals to JSON for something with
// parts a question can name.
func (c *Client) Ask(
	ctx context.Context, state any, questions map[string]Question,
) (Response, error) {
	if len(questions) == 0 {
		return Response{}, errors.New("typesafe: at least one question is required")
	}
	for id, question := range questions {
		if strings.TrimSpace(question.Instructions) == "" {
			return Response{}, fmt.Errorf("typesafe: question %q has no instructions", id)
		}
	}

	payload, err := json.Marshal(request{State: state, Model: c.model, Questions: questions})
	if err != nil {
		return Response{}, fmt.Errorf("typesafe: encode questions: %w", err)
	}

	httpRequest, err := http.NewRequestWithContext(
		ctx, http.MethodPost, c.baseURL+"/v1/systemone", bytes.NewReader(payload))
	if err != nil {
		return Response{}, fmt.Errorf("typesafe: build request: %w", err)
	}
	httpRequest.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpRequest.Header.Set("Content-Type", "application/json")

	httpResponse, err := c.client.Do(httpRequest)
	if err != nil {
		return Response{}, fmt.Errorf("typesafe: ask: %w", err)
	}
	defer httpResponse.Body.Close()

	if httpResponse.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(httpResponse.Body, errorBodyLimit))
		return Response{}, &StatusError{
			StatusCode: httpResponse.StatusCode,
			Body:       strings.TrimSpace(string(body)),
		}
	}

	var decoded Response
	if err := json.NewDecoder(httpResponse.Body).Decode(&decoded); err != nil {
		return Response{}, fmt.Errorf("typesafe: decode answers: %w", err)
	}
	for id := range questions {
		if _, answered := decoded.Answers[id]; !answered {
			return Response{}, fmt.Errorf("typesafe: %q was not answered", id)
		}
	}
	return decoded, nil
}
