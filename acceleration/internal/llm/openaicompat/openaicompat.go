// Package openaicompat implements the llm contract over an OpenAI-compatible chat
// completions endpoint.
//
// Baseten's Model APIs, a vLLM deployment and Google's compatibility shim all speak the
// same protocol, so they share one implementation and differ only in base URL, credentials
// and whatever extra request fields they understand. Each provider package wraps this one
// with its own name, defaults and environment variables.
//
// The contract above this is shaped like the Responses API, and chat completions is the
// older and smaller of the two. What it cannot express -- a stored response to continue
// from, a cache key, a conversation held by the provider -- is dropped rather than
// approximated, and the capabilities this reports say so.
package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/packages/respjson"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/shared"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// defaultTimeout bounds one response. A conversational turn that takes this long has
// already failed as far as the listener is concerned.
const defaultTimeout = 2 * time.Minute

// reasoningField is where models that think stream it. It is not part of the OpenAI
// schema, so it arrives as an extra field rather than on the delta struct.
const reasoningField = "reasoning_content"

// A signing provider nests its signature under extra_content, keyed by vendor. Gemini is
// the one that does, and it refuses a replayed call that comes back without it.
const (
	signatureField  = "extra_content"
	signatureVendor = "google"
	signatureName   = "thought_signature"
)

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
	// Capabilities is what this model accepts. Store, Conversations and PromptCacheKey
	// are forced off, because chat completions has nowhere to put them.
	Capabilities llm.Capabilities
	// RequestFields builds the request fields outside the OpenAI schema, such as a vLLM
	// chat template's arguments or a reasoning effort the endpoint spells its own way.
	// It is given the effort already resolved against the model's capabilities.
	RequestFields func(params llm.ResponseParams, effort string) map[string]any
	// Timeout bounds one response.
	Timeout time.Duration
	// HTTPClient replaces the default transport.
	HTTPClient option.HTTPClient
	Logger     *slog.Logger
}

// LLM is a provider reached over an OpenAI-compatible endpoint.
type LLM struct {
	options Options
	client  openai.Client
	logger  *slog.Logger

	mu sync.Mutex
	// inFlight cancels the responses that have not settled, so closing the provider
	// abandons them rather than leaving them to their own timeouts.
	inFlight map[uint64]context.CancelFunc
	nextID   atomic.Uint64
	closed   bool
}

// New builds a provider. It performs no network access.
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
	options.Capabilities.Store = false
	options.Capabilities.Conversations = false
	options.Capabilities.PromptCacheKey = false
	options.Capabilities.CacheTTLs = nil

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
		logger:   options.Logger.With("provider", options.Provider, "model", options.StatsModel),
		inFlight: map[uint64]context.CancelFunc{},
	}, nil
}

// Client exposes the underlying SDK client, so anything this package does not standardise
// -- embeddings, moderation, the responses endpoint itself -- is still one call away.
func (l *LLM) Client() *openai.Client { return &l.client }

// Create asks for one response and returns the stream it arrives on.
func (l *LLM) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	if len(params.Input) == 0 {
		return nil, fmt.Errorf("openaicompat: %s: a request needs at least one input message", l.options.Provider)
	}
	if err := l.options.Capabilities.Validate(params); err != nil {
		return nil, fmt.Errorf("openaicompat: %s: %w", l.options.Provider, err)
	}

	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return nil, fmt.Errorf("openaicompat: %s: provider is closed", l.options.Provider)
	}
	requestCtx, cancel := context.WithTimeout(ctx, l.options.Timeout)
	id := l.nextID.Add(1)
	l.inFlight[id] = cancel
	l.mu.Unlock()

	upstream := l.client.Chat.Completions.NewStreaming(
		requestCtx, l.params(params), l.requestOptions(params)...,
	)
	return llm.NewStream(
		llm.StreamOptions{
			ResponseID: params.ID,
			Provider:   l.options.Provider,
			Model:      l.options.StatsModel,
		},
		&puller{llm: l, id: id, upstream: upstream, cancel: cancel},
	), nil
}

// Close abandons everything in flight.
func (l *LLM) Close() error {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return nil
	}
	l.closed = true
	cancels := make([]context.CancelFunc, 0, len(l.inFlight))
	for _, cancel := range l.inFlight {
		cancels = append(cancels, cancel)
	}
	l.mu.Unlock()

	for _, cancel := range cancels {
		cancel()
	}
	return nil
}

// Provider is the stable provider name used in stats.
func (l *LLM) Provider() string { return l.options.Provider }

// Model is the model identifier used in stats.
func (l *LLM) Model() string { return l.options.StatsModel }

// Capabilities is what this model accepts.
func (l *LLM) Capabilities() llm.Capabilities { return l.options.Capabilities }

// forget releases a response that has settled.
func (l *LLM) forget(id uint64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.inFlight, id)
}

// puller turns the chat completion chunks into what a Stream reports.
type puller struct {
	llm      *LLM
	id       uint64
	upstream *ssestream.Stream[openai.ChatCompletionChunk]
	cancel   context.CancelFunc

	err  error
	done bool
}

// Advance reads one chunk and records what it carried.
func (p *puller) Advance(w *llm.ResponseWriter) bool {
	if p.done {
		return false
	}
	if !p.upstream.Next() {
		p.finish(w)
		return false
	}

	chunk := p.upstream.Current()
	if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
		// Some providers repeat a cumulative usage frame on every chunk rather than
		// sending one at the end, so the writer keeps the last one it is told.
		w.Usage(llm.Usage{
			InputTokens: chunk.Usage.PromptTokens,
			InputTokensDetails: llm.InputTokensDetails{
				CachedTokens: chunk.Usage.PromptTokensDetails.CachedTokens,
			},
			OutputTokens: chunk.Usage.CompletionTokens,
			OutputTokensDetails: llm.OutputTokensDetails{
				ReasoningTokens: chunk.Usage.CompletionTokensDetails.ReasoningTokens,
			},
			TotalTokens: chunk.Usage.TotalTokens,
		})
	}

	for _, choice := range chunk.Choices {
		w.OutputText(choice.Delta.Content)
		w.ReasoningText(reasoning(choice.Delta.JSON.ExtraFields))
		for _, call := range choice.Delta.ToolCalls {
			w.FunctionCall(
				call.Index,
				call.ID,
				call.Function.Name,
				call.Function.Arguments,
				signature(call.JSON.ExtraFields),
			)
		}
		if reason := incomplete(choice.FinishReason); reason != "" {
			w.Incomplete(reason)
		}
	}
	return true
}

// Err is the provider failure that ended the stream, if there was one.
func (p *puller) Err() error { return p.err }

// Close abandons the response. It only cancels: the upstream is released by the goroutine
// reading it, once the cancellation has unblocked it.
func (p *puller) Close() error {
	p.cancel()
	return nil
}

// finish releases the upstream and works out what ended it.
func (p *puller) finish(w *llm.ResponseWriter) {
	p.done = true

	if err := p.upstream.Err(); err != nil && !errors.Is(err, context.Canceled) {
		p.err = err
	} else if err != nil {
		// A cancelled stream is barge-in or shutdown, not a provider failure.
		w.Cancelled()
	}

	p.upstream.Close()
	p.cancel()
	p.llm.forget(p.id)
}

// params builds the upstream request from what the Responses-shaped one asked for.
func (l *LLM) params(request llm.ResponseParams) openai.ChatCompletionNewParams {
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(request.Input)+1)
	if request.Instructions != "" {
		messages = append(messages, openai.SystemMessage(request.Instructions))
	}
	for _, message := range request.Input {
		switch message.Role {
		case llm.System:
			messages = append(messages, openai.SystemMessage(message.Content))
		case llm.Assistant:
			messages = append(messages, assistantMessage(message))
		case llm.ToolResult:
			messages = append(messages, openai.ToolMessage(message.Content, message.ToolCallID))
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
	if len(request.Tools) > 0 {
		params.Tools = tools(request.Tools)
	}
	if request.ToolChoice != "" {
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt(request.ToolChoice),
		}
	}
	if request.Text.Format == llm.FormatJSONObject {
		params.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
		}
	}
	if request.MaxOutputTokens > 0 {
		params.MaxCompletionTokens = param.NewOpt(int64(request.MaxOutputTokens))
	}
	if request.Temperature != nil {
		params.Temperature = param.NewOpt(*request.Temperature)
	}
	return params
}

// tools renders the tools a request offers. A tool with no parameters is sent without a
// schema, which is how the protocol spells "this one takes no arguments".
func tools(offered []llm.Tool) []openai.ChatCompletionToolUnionParam {
	rendered := make([]openai.ChatCompletionToolUnionParam, 0, len(offered))
	for _, tool := range offered {
		definition := shared.FunctionDefinitionParam{Name: tool.Name}
		if tool.Description != "" {
			definition.Description = param.NewOpt(tool.Description)
		}
		if len(tool.Parameters) > 0 {
			definition.Parameters = shared.FunctionParameters(tool.Parameters)
		}
		rendered = append(rendered, openai.ChatCompletionFunctionTool(definition))
	}
	return rendered
}

// assistantMessage replays a turn the model took.
//
// A turn that called a tool has to carry the calls it made, because the provider matches
// each tool result against one and rejects a conversation where a result answers nothing.
func assistantMessage(message llm.Message) openai.ChatCompletionMessageParamUnion {
	if len(message.ToolCalls) == 0 {
		return openai.AssistantMessage(message.Content)
	}

	calls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(message.ToolCalls))
	for _, call := range message.ToolCalls {
		function := &openai.ChatCompletionMessageFunctionToolCallParam{
			ID: call.ID,
			Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
				Name:      call.Name,
				Arguments: call.Arguments,
			},
		}
		if call.Signature != "" {
			function.SetExtraFields(map[string]any{
				signatureField: map[string]any{
					signatureVendor: map[string]any{
						signatureName: call.Signature,
					},
				},
			})
		}
		calls = append(calls, openai.ChatCompletionMessageToolCallUnionParam{OfFunction: function})
	}

	assistant := openai.ChatCompletionAssistantMessageParam{ToolCalls: calls}
	if message.Content != "" {
		assistant.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
			OfString: param.NewOpt(message.Content),
		}
	}
	return openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant}
}

// requestOptions applies the request fields outside the OpenAI schema.
func (l *LLM) requestOptions(request llm.ResponseParams) []option.RequestOption {
	if l.options.RequestFields == nil {
		return nil
	}

	fields := l.options.RequestFields(request, l.options.Capabilities.Effort(request))
	options := make([]option.RequestOption, 0, len(fields))
	for key, value := range fields {
		options = append(options, option.WithJSONSet(key, value))
	}
	return options
}

// incomplete translates a finish reason into why a response stopped early, returning empty
// for the model simply having finished.
func incomplete(reason string) string {
	switch reason {
	case "length":
		return llm.ReasonMaxOutputTokens
	case "content_filter":
		return llm.ReasonContentFilter
	default:
		return ""
	}
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

// signature reads the state a signing provider attached to a tool call, returning empty
// for the providers that do not sign.
func signature(extra map[string]respjson.Field) string {
	field, ok := extra[signatureField]
	if !ok {
		return ""
	}

	var content struct {
		Google struct {
			ThoughtSignature string `json:"thought_signature"`
		} `json:"google"`
	}
	if err := json.Unmarshal([]byte(field.Raw()), &content); err != nil {
		return ""
	}
	return content.Google.ThoughtSignature
}
