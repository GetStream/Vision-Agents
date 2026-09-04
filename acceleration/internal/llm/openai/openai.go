// Package openai reaches OpenAI's own Responses API.
//
// This is the one provider on the real endpoint rather than the chat completions shim the
// others share, which is what earns it the things chat completions has nowhere to put: a
// reasoning effort alongside tools rather than instead of them, a stored response to
// continue from, and a cache key that lets one agent's instructions be written to the
// cache once and read back on every turn of every call it takes.
package openai

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "openai"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "OPENAI_API_KEY"

// defaultBaseURL is OpenAI's endpoint.
const defaultBaseURL = "https://api.openai.com/v1"

// defaultModel is used when the caller names no model. The cheapest, fastest tier is the
// right default for a conversation; a caller who wants a better answer names one.
const defaultModel = "gpt-5.6-luna"

// defaultTimeout bounds one response. A conversational turn that takes this long has
// already failed as far as the listener is concerned.
const defaultTimeout = 2 * time.Minute

// cacheTTL is the only prefix lifetime the API currently takes.
const cacheTTL = 30 * time.Minute

// modelCapabilities is what each model family accepts.
//
// The 5.6 family is the first to take max, and the first where reasoning and tools can be
// asked for together. Anything older is offered the smaller set it answers to, which is
// the whole reason this is a table rather than one list.
var modelCapabilities = map[string]llm.Capabilities{
	"gpt-5.6": {
		ReasoningEfforts: []string{"none", "low", "medium", "high", "xhigh", "max"},
		DefaultEffort:    "none",
		StreamsReasoning: false,
		Verbosities:      []string{"low", "medium", "high"},
		Store:            true,
		Conversations:    true,
		PromptCacheKey:   true,
		CacheTTLs:        []time.Duration{cacheTTL},
	},
	"gpt-5.5": {
		ReasoningEfforts: []string{"none", "minimal", "low", "medium", "high", "xhigh"},
		DefaultEffort:    "none",
		Verbosities:      []string{"low", "medium", "high"},
		Store:            true,
		Conversations:    true,
		PromptCacheKey:   true,
		CacheTTLs:        []time.Duration{cacheTTL},
	},
	"gpt-5": {
		ReasoningEfforts: []string{"minimal", "low", "medium", "high"},
		DefaultEffort:    "minimal",
		Verbosities:      []string{"low", "medium", "high"},
		Store:            true,
		Conversations:    true,
		PromptCacheKey:   true,
	},
}

// fallbackCapabilities is what a model this table has never heard of is assumed to do.
// Storing and caching are safe to assume on this API; reasoning is not.
var fallbackCapabilities = llm.Capabilities{
	Store:          true,
	Conversations:  true,
	PromptCacheKey: true,
}

// Options configures the provider.
type Options struct {
	APIKey string
	Model  string
	// BaseURL overrides the endpoint, which is useful for a gateway in front of OpenAI.
	BaseURL string
	// ReasoningEffort is what the model is asked for when a request names none. Empty
	// leaves the model family's own default, which for a conversation is the least
	// thinking it will do.
	ReasoningEffort string
	// Timeout bounds one response.
	Timeout    time.Duration
	HTTPClient option.HTTPClient
	Logger     *slog.Logger
}

// LLM is OpenAI reached over the Responses API.
type LLM struct {
	options      Options
	capabilities llm.Capabilities
	client       openai.Client
	logger       *slog.Logger

	mu sync.Mutex
	// inFlight cancels the responses that have not settled, so closing the provider
	// abandons them rather than leaving them to their own timeouts.
	inFlight map[uint64]context.CancelFunc
	nextID   atomic.Uint64
	closed   bool
}

// New builds the provider, reading the API key from the environment when it is not given.
func New(options Options) (*LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("openai: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		options.Model = defaultModel
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	if options.Timeout == 0 {
		options.Timeout = defaultTimeout
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	capabilities := capabilitiesFor(options.Model)
	if options.ReasoningEffort != "" {
		if err := capabilities.Validate(
			llm.ResponseParams{Reasoning: llm.ReasoningParams{Effort: options.ReasoningEffort}},
		); err != nil {
			return nil, fmt.Errorf("openai: %s: %w", options.Model, err)
		}
		capabilities.DefaultEffort = options.ReasoningEffort
	}

	clientOptions := []option.RequestOption{
		option.WithAPIKey(options.APIKey),
		option.WithBaseURL(options.BaseURL),
	}
	if options.HTTPClient != nil {
		clientOptions = append(clientOptions, option.WithHTTPClient(options.HTTPClient))
	}

	return &LLM{
		options:      options,
		capabilities: capabilities,
		client:       openai.NewClient(clientOptions...),
		logger:       options.Logger.With("provider", ProviderName, "model", options.Model),
		inFlight:     map[uint64]context.CancelFunc{},
	}, nil
}

// Client exposes the underlying SDK client, so anything this package does not standardise
// is still one call away.
func (l *LLM) Client() *openai.Client { return &l.client }

// Create asks for one response and returns the stream it arrives on.
func (l *LLM) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	if len(params.Input) == 0 && params.PreviousResponseID == "" && params.Conversation == "" {
		return nil, errors.New("openai: a request needs input, a previous response or a conversation")
	}
	if err := l.capabilities.Validate(params); err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}

	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return nil, errors.New("openai: provider is closed")
	}
	requestCtx, cancel := context.WithTimeout(ctx, l.options.Timeout)
	id := l.nextID.Add(1)
	l.inFlight[id] = cancel
	l.mu.Unlock()

	upstream := l.client.Responses.NewStreaming(requestCtx, l.params(params))
	return llm.NewStream(
		llm.StreamOptions{
			ResponseID: params.ID,
			Provider:   ProviderName,
			Model:      l.options.Model,
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
func (l *LLM) Provider() string { return ProviderName }

// Model is the model identifier used in stats.
func (l *LLM) Model() string { return l.options.Model }

// Capabilities is what this model accepts.
func (l *LLM) Capabilities() llm.Capabilities { return l.capabilities }

// forget releases a response that has settled.
func (l *LLM) forget(id uint64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.inFlight, id)
}

// capabilitiesFor looks a model up by the longest family prefix that matches it, so a
// dated snapshot such as gpt-5.6-sol-2026-02-11 is recognised as its family.
func capabilitiesFor(model string) llm.Capabilities {
	best := ""
	for family := range modelCapabilities {
		if strings.HasPrefix(model, family) && len(family) > len(best) {
			best = family
		}
	}
	if best == "" {
		return fallbackCapabilities
	}
	return modelCapabilities[best]
}

// puller turns the Responses stream events into what a Stream reports.
type puller struct {
	llm      *LLM
	id       uint64
	upstream *ssestream.Stream[responses.ResponseStreamEventUnion]
	cancel   context.CancelFunc

	// names remembers the call each function_call output item belongs to, because the
	// argument deltas that follow carry only the item id.
	names map[string]call

	err  error
	done bool
}

// call is what an output item said about a function call before its arguments arrived.
type call struct {
	index  int64
	callID string
	name   string
}

// Advance reads one event and records what it carried.
func (p *puller) Advance(w *llm.ResponseWriter) bool {
	if p.done {
		return false
	}
	if !p.upstream.Next() {
		p.finish(w)
		return false
	}

	event := p.upstream.Current()
	switch event.Type {
	case "response.created", "response.in_progress":
		w.SetProviderResponseID(event.Response.ID)

	case "response.output_text.delta":
		w.OutputText(event.Delta)

	case "response.reasoning_text.delta", "response.reasoning_summary_text.delta":
		w.ReasoningText(event.Delta)

	case "response.output_item.added":
		if event.Item.Type == "function_call" {
			p.remember(event.ItemID, event.OutputIndex, event.Item.CallID, event.Item.Name)
			w.FunctionCall(event.OutputIndex, event.Item.CallID, event.Item.Name, "", "")
		}

	case "response.function_call_arguments.delta":
		known := p.names[event.ItemID]
		w.FunctionCall(known.index, "", "", event.Delta, "")

	case "response.completed", "response.incomplete", "response.failed":
		p.settle(w, event.Response)

	case "error":
		p.err = fmt.Errorf("openai: %s", event.Message)
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

// remember records which call an output item is, so its argument deltas can be attributed.
func (p *puller) remember(itemID string, index int64, callID, name string) {
	if p.names == nil {
		p.names = map[string]call{}
	}
	p.names[itemID] = call{index: index, callID: callID, name: name}
}

// settle records the terminal response, which is where usage and the reason it stopped
// arrive.
func (p *puller) settle(w *llm.ResponseWriter, response responses.Response) {
	w.SetProviderResponseID(response.ID)
	w.Usage(llm.Usage{
		InputTokens: response.Usage.InputTokens,
		InputTokensDetails: llm.InputTokensDetails{
			CachedTokens:     response.Usage.InputTokensDetails.CachedTokens,
			CacheWriteTokens: response.Usage.InputTokensDetails.CacheWriteTokens,
		},
		OutputTokens: response.Usage.OutputTokens,
		OutputTokensDetails: llm.OutputTokensDetails{
			ReasoningTokens: response.Usage.OutputTokensDetails.ReasoningTokens,
		},
		TotalTokens: response.Usage.TotalTokens,
	})

	switch response.Status {
	case "incomplete":
		w.Incomplete(string(response.IncompleteDetails.Reason))
	case "failed":
		message := "the model failed to answer"
		if response.Error.Message != "" {
			message = response.Error.Message
		}
		p.err = fmt.Errorf("openai: %s", message)
	}
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

// params builds the upstream request.
func (l *LLM) params(request llm.ResponseParams) responses.ResponseNewParams {
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(l.options.Model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: l.input(request)},
	}
	if request.MaxOutputTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(request.MaxOutputTokens))
	}
	if request.Temperature != nil {
		params.Temperature = param.NewOpt(*request.Temperature)
	}
	if effort := l.capabilities.Effort(request); effort != "" {
		params.Reasoning = shared.ReasoningParam{Effort: shared.ReasoningEffort(effort)}
	}
	if request.Text.Verbosity != "" {
		params.Text.Verbosity = responses.ResponseTextConfigVerbosity(request.Text.Verbosity)
	}
	if request.Text.Format == llm.FormatJSONObject {
		params.Text.Format = responses.ResponseFormatTextConfigUnionParam{
			OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
		}
	}
	if len(request.Tools) > 0 {
		params.Tools = tools(request.Tools)
	}
	if request.ToolChoice != "" {
		params.ToolChoice = responses.ResponseNewParamsToolChoiceUnion{
			OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptions(request.ToolChoice)),
		}
	}
	if request.Store {
		params.Store = param.NewOpt(true)
	}
	if request.PreviousResponseID != "" {
		params.PreviousResponseID = param.NewOpt(request.PreviousResponseID)
	}
	if request.Conversation != "" {
		params.Conversation = responses.ResponseNewParamsConversationUnion{
			OfString: param.NewOpt(request.Conversation),
		}
	}
	if request.PromptCacheKey != "" {
		params.PromptCacheKey = param.NewOpt(request.PromptCacheKey)
	}
	if options := l.cacheOptions(request.PromptCacheOptions); options != nil {
		params.PromptCacheOptions = *options
	}
	if len(request.Metadata) > 0 {
		params.Metadata = shared.Metadata(request.Metadata)
	}
	return params
}

// input renders the conversation as the items the API takes.
//
// The instructions travel as a developer message rather than in the top-level instructions
// field, because that is the only place a cache breakpoint can be put: the boundary the
// whole agent shares is the end of what it was told to be, and a breakpoint attaches to a
// content part.
func (l *LLM) input(request llm.ResponseParams) responses.ResponseInputParam {
	items := make(responses.ResponseInputParam, 0, len(request.Input)+1)
	if request.Instructions != "" {
		items = append(items, l.instructions(request))
	}

	for _, message := range request.Input {
		switch message.Role {
		case llm.System:
			items = append(items, responses.ResponseInputItemParamOfMessage(
				message.Content, responses.EasyInputMessageRoleDeveloper))
		case llm.Assistant:
			items = append(items, assistant(message)...)
		case llm.ToolResult:
			items = append(items, responses.ResponseInputItemParamOfFunctionCallOutput(
				message.ToolCallID, message.Content))
		default:
			items = append(items, responses.ResponseInputItemParamOfMessage(
				message.Content, responses.EasyInputMessageRoleUser))
		}
	}
	return items
}

// instructions renders the system prompt, marking the end of it as a reusable prefix when
// the caller asked for explicit breakpoints.
func (l *LLM) instructions(request llm.ResponseParams) responses.ResponseInputItemUnionParam {
	text := responses.ResponseInputTextParam{Text: request.Instructions}
	if request.PromptCacheOptions.Mode == llm.CacheExplicit {
		text.PromptCacheBreakpoint = responses.ResponseInputTextPromptCacheBreakpointParam{}
	}
	return responses.ResponseInputItemParamOfInputMessage(
		responses.ResponseInputMessageContentListParam{{OfInputText: &text}},
		string(responses.EasyInputMessageRoleDeveloper),
	)
}

// assistant replays a turn the model took. The calls it made are items of their own here,
// because the result sent back answers one of them by id and the API rejects a
// conversation where a result answers nothing.
func assistant(message llm.Message) []responses.ResponseInputItemUnionParam {
	items := make([]responses.ResponseInputItemUnionParam, 0, len(message.ToolCalls)+1)
	if message.Content != "" {
		items = append(items, responses.ResponseInputItemParamOfMessage(
			message.Content, responses.EasyInputMessageRoleAssistant))
	}
	for _, made := range message.ToolCalls {
		items = append(items, responses.ResponseInputItemParamOfFunctionCall(
			made.Arguments, made.ID, made.Name))
	}
	return items
}

// tools renders the tools a request offers.
func tools(offered []llm.Tool) []responses.ToolUnionParam {
	rendered := make([]responses.ToolUnionParam, 0, len(offered))
	for _, tool := range offered {
		function := &responses.FunctionToolParam{Name: tool.Name}
		if tool.Description != "" {
			function.Description = param.NewOpt(tool.Description)
		}
		if len(tool.Parameters) > 0 {
			function.Parameters = tool.Parameters
		}
		rendered = append(rendered, responses.ToolUnionParam{OfFunction: function})
	}
	return rendered
}

// cacheOptions renders the request's cache policy, returning nil when it asked for nothing
// the model takes.
//
// A TTL the API does not offer is dropped rather than refused: it is a hint about how long
// a prefix should stay warm, and getting the provider's own lifetime instead is not a
// failure worth ending a call over.
func (l *LLM) cacheOptions(policy llm.PromptCacheOptions) *responses.ResponseNewParamsPromptCacheOptions {
	options := responses.ResponseNewParamsPromptCacheOptions{}
	set := false

	if policy.Mode != "" {
		options.Mode = string(policy.Mode)
		set = true
	}
	if policy.TTL > 0 {
		for _, offered := range l.capabilities.CacheTTLs {
			if offered == policy.TTL {
				options.Ttl = shortDuration(policy.TTL)
				set = true
				break
			}
		}
	}
	if !set {
		return nil
	}
	return &options
}

// shortDuration spells a lifetime the way the API does, as 30m rather than 30m0s.
func shortDuration(ttl time.Duration) string {
	if ttl%time.Hour == 0 {
		return fmt.Sprintf("%dh", int(ttl/time.Hour))
	}
	return fmt.Sprintf("%dm", int(ttl/time.Minute))
}
