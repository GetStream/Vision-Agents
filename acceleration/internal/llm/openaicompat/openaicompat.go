// Package openaicompat implements the llm contract over an OpenAI-compatible chat
// completions endpoint.
//
// OpenAI's own API, Baseten's Model APIs and a vLLM deployment all speak the same
// protocol, so they share one implementation and differ only in base URL, credentials and
// whatever extra request fields they understand. Each provider package wraps this one with
// its own name, defaults and environment variables.
package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/packages/respjson"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// eventBuffer is how many events may queue before a slow consumer applies backpressure.
const eventBuffer = 64

// defaultTimeout bounds one completion. A conversational turn that takes this long has
// already failed as far as the listener is concerned.
const defaultTimeout = 2 * time.Minute

// reasoningField is where models that think stream it. It is not part of the OpenAI
// schema, so it arrives as an extra field rather than on the delta struct.
const reasoningField = "reasoning_content"

// Options configures a provider served over an OpenAI-compatible endpoint.
type Options struct {
	// Provider is the stable name used in stats, e.g. "deepseek".
	Provider string
	// Model is the identifier sent upstream, e.g. "deepseek-ai/DeepSeek-V4-Flash-0731".
	Model string
	// StatsModel is the shorter name recorded in stats. It defaults to Model, which is
	// what a provider serving one model per name wants.
	StatsModel string
	APIKey     string
	// BaseURL is the endpoint root, up to and including /v1.
	BaseURL string
	// Reasoning declares that this model streams its thinking.
	Reasoning bool
	// ExtraBody carries request fields outside the OpenAI schema, such as a vLLM chat
	// template's arguments.
	ExtraBody map[string]any
	// Timeout bounds one completion.
	Timeout time.Duration
	// HTTPClient replaces the default transport.
	HTTPClient option.HTTPClient
	Logger     *slog.Logger
}

// LLM is a provider reached over an OpenAI-compatible endpoint.
type LLM struct {
	options Options
	client  openai.Client
	emitter *llm.Emitter
	logger  *slog.Logger

	// ctx is the session's lifetime; every completion derives from it so Close stops
	// everything in flight.
	ctx    context.Context
	cancel context.CancelFunc

	mu sync.Mutex
	// inFlight cancels completions that have not settled, keyed by completion id.
	inFlight map[string]context.CancelFunc
	closed   bool
	running  sync.WaitGroup
}

// New builds a provider. It performs no network access; Start does that.
func New(options Options) (*LLM, error) {
	if options.Provider == "" {
		return nil, errors.New("openaicompat: provider name is required")
	}
	if options.Model == "" {
		return nil, errors.New("openaicompat: model is required")
	}
	if options.APIKey == "" {
		return nil, fmt.Errorf("openaicompat: %s: api key is required", options.Provider)
	}
	if options.BaseURL == "" {
		return nil, fmt.Errorf("openaicompat: %s: base url is required", options.Provider)
	}
	if options.StatsModel == "" {
		options.StatsModel = options.Model
	}
	if options.Timeout == 0 {
		options.Timeout = defaultTimeout
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	clientOptions := []option.RequestOption{
		option.WithAPIKey(options.APIKey),
		option.WithBaseURL(options.BaseURL),
	}
	if options.HTTPClient != nil {
		clientOptions = append(clientOptions, option.WithHTTPClient(options.HTTPClient))
	}

	return &LLM{
		options:  options,
		client:   openai.NewClient(clientOptions...),
		emitter:  llm.NewEmitter(eventBuffer),
		logger:   options.Logger.With("provider", options.Provider, "model", options.StatsModel),
		inFlight: map[string]context.CancelFunc{},
	}, nil
}

// Client exposes the underlying SDK client, so anything this package does not standardise
// -- tools, structured output, embeddings -- is still one call away.
func (l *LLM) Client() *openai.Client { return &l.client }

// Start makes the provider ready. There is no connection to open for a request-per-turn
// protocol, so this only fixes the session's lifetime.
func (l *LLM) Start(ctx context.Context) error {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return fmt.Errorf("openaicompat: %s: provider is closed", l.options.Provider)
	}
	if l.ctx != nil {
		l.mu.Unlock()
		return nil
	}
	l.ctx, l.cancel = context.WithCancel(ctx)
	l.mu.Unlock()

	l.emitter.Send(llm.Connected{
		Provider: l.options.Provider,
		Model:    l.options.StatsModel,
		At:       time.Now(),
	})
	return nil
}

// Respond asks for a completion and streams it back through Events. It returns as soon as
// the request is on its way.
func (l *LLM) Respond(request llm.Request) error {
	if len(request.Messages) == 0 {
		return fmt.Errorf("openaicompat: %s: a request needs at least one message", l.options.Provider)
	}

	completion := llm.NewCompletion(request.ID)

	l.mu.Lock()
	if l.closed || l.ctx == nil {
		l.mu.Unlock()
		return fmt.Errorf("openaicompat: %s: provider is not started", l.options.Provider)
	}
	ctx, cancel := context.WithTimeout(l.ctx, l.options.Timeout)
	l.inFlight[completion.ID] = cancel
	l.running.Add(1)
	l.mu.Unlock()

	l.emitter.Send(llm.CompletionStarted{
		CompletionID: completion.ID,
		Provider:     l.options.Provider,
		Model:        l.options.StatsModel,
		At:           time.Now(),
	})

	go func() {
		defer l.running.Done()
		defer cancel()
		l.stream(ctx, request, completion)
	}()
	return nil
}

// Interrupt abandons the named completions, or every completion in flight when given
// none. The ones that had already produced text still settle, and still report what they
// cost.
func (l *LLM) Interrupt(completionIDs ...string) error {
	l.mu.Lock()
	cancels := make([]context.CancelFunc, 0, len(l.inFlight))
	if len(completionIDs) == 0 {
		for _, cancel := range l.inFlight {
			cancels = append(cancels, cancel)
		}
	}
	for _, id := range completionIDs {
		if cancel, ok := l.inFlight[id]; ok {
			cancels = append(cancels, cancel)
		}
	}
	l.mu.Unlock()

	for _, cancel := range cancels {
		cancel()
	}
	return nil
}

// Events carries deltas and completion boundaries. It is closed by Close.
func (l *LLM) Events() <-chan llm.Event { return l.emitter.Events() }

// Close abandons anything in flight and closes the event channel.
func (l *LLM) Close() error {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return nil
	}
	l.closed = true
	cancel := l.cancel
	l.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	// Wait for the streams to emit their final events before the channel closes, so a
	// consumer draining Events sees every completion settle.
	l.running.Wait()

	l.emitter.Send(llm.Disconnected{
		Provider: l.options.Provider,
		Model:    l.options.StatsModel,
		Reason:   "closed by caller",
		Clean:    true,
		At:       time.Now(),
	})
	l.emitter.Close()
	return nil
}

// Provider is the stable provider name used in stats.
func (l *LLM) Provider() string { return l.options.Provider }

// Model is the model identifier used in stats.
func (l *LLM) Model() string { return l.options.StatsModel }

// Reasoning reports whether this model streams its thinking.
func (l *LLM) Reasoning() bool { return l.options.Reasoning }

// stream consumes the response and turns it into events. It always settles the completion,
// so a caller waiting on CompletionComplete is never left hanging.
func (l *LLM) stream(ctx context.Context, request llm.Request, completion *llm.Completion) {
	interrupted := false
	defer func() {
		l.settle(completion.ID)
		l.emitter.Send(completion.Complete(l.options.Provider, l.options.StatsModel, interrupted))
	}()

	stream := l.client.Chat.Completions.NewStreaming(ctx, l.params(request), l.requestOptions()...)
	defer stream.Close()

	for stream.Next() {
		chunk := stream.Current()

		if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
			// Some providers repeat a cumulative usage frame on every chunk rather than
			// sending one at the end, so the tracker keeps the last one it is told.
			completion.Usage(
				chunk.Usage.PromptTokens,
				chunk.Usage.PromptTokensDetails.CachedTokens,
				chunk.Usage.CompletionTokens,
				chunk.Usage.CompletionTokensDetails.ReasoningTokens,
			)
		}

		for _, choice := range chunk.Choices {
			if text := choice.Delta.Content; text != "" {
				l.emitter.Send(completion.Delta(text))
			}
			if thinking := reasoning(choice.Delta.JSON.ExtraFields); thinking != "" {
				l.emitter.Send(completion.Reasoning(thinking))
			}
			if choice.FinishReason != "" {
				completion.Finish(choice.FinishReason)
			}
		}
	}

	if err := stream.Err(); err != nil {
		// A cancelled stream is barge-in or shutdown, not a provider failure, so it
		// settles as interrupted instead of being recorded as an error.
		if errors.Is(err, context.Canceled) {
			interrupted = true
			return
		}
		l.emitter.Send(llm.Error{
			Provider:     l.options.Provider,
			Model:        l.options.StatsModel,
			CompletionID: completion.ID,
			Err:          err,
			Context:      "stream",
		})
	}
}

// params builds the upstream request.
func (l *LLM) params(request llm.Request) openai.ChatCompletionNewParams {
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(request.Messages)+1)
	if request.Instructions != "" {
		messages = append(messages, openai.SystemMessage(request.Instructions))
	}
	for _, message := range request.Messages {
		switch message.Role {
		case llm.System:
			messages = append(messages, openai.SystemMessage(message.Content))
		case llm.Assistant:
			messages = append(messages, openai.AssistantMessage(message.Content))
		default:
			messages = append(messages, openai.UserMessage(message.Content))
		}
	}

	params := openai.ChatCompletionNewParams{
		Model:    l.options.Model,
		Messages: messages,
		// Usage is asked for explicitly, because without it there is nothing to bill.
		StreamOptions: openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: param.NewOpt(true),
		},
	}
	if request.MaxTokens > 0 {
		params.MaxCompletionTokens = param.NewOpt(int64(request.MaxTokens))
	}
	if request.Temperature != nil {
		params.Temperature = param.NewOpt(*request.Temperature)
	}
	return params
}

// requestOptions applies the provider's extra body fields.
func (l *LLM) requestOptions() []option.RequestOption {
	if len(l.options.ExtraBody) == 0 {
		return nil
	}

	options := make([]option.RequestOption, 0, len(l.options.ExtraBody))
	for key, value := range l.options.ExtraBody {
		options = append(options, option.WithJSONSet(key, value))
	}
	return options
}

// settle forgets a completion that is no longer in flight.
func (l *LLM) settle(completionID string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.inFlight, completionID)
}

// reasoning reads the non-standard thinking field off a delta, returning empty when the
// provider did not send one.
//
// Valid is deliberately not consulted: it tracks the presence of fields the SDK declares,
// and an extra field is by definition not one of those, so it always reports false here.
func reasoning(extra map[string]respjson.Field) string {
	field, ok := extra[reasoningField]
	if !ok {
		return ""
	}

	var text string
	if err := json.Unmarshal([]byte(field.Raw()), &text); err != nil {
		return ""
	}
	return text
}
