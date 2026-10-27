package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// defaultTimeout bounds one response, as it does on the compatibility path.
const defaultTimeout = 2 * time.Minute

// defaultMaxTokens is what a request that names no cap may write, thinking included. The
// Messages API requires one, and every current Claude model writes up to 128k.
const defaultMaxTokens = 64000

// LLM is Claude over the Messages API.
//
// Every request caches its prefix: the tools, the system prompt, and the conversation up to
// the turn being answered each end on a cache_control breakpoint (see breakpoints), so a
// tool loop reads each earlier round from the cache instead of paying for it again.
//
// It does what the compatibility endpoint did and no more. Not carried over: a reasoning
// effort (refused, as before), a replayed turn's thinking (dropped, as before; adaptive
// thinking does not require it), temperature (Claude 4.7 and later refuse any but the
// default), JSON modes (the endpoint ignored them; callers already check what comes back),
// and the one-hour cache TTL (its writes bill at twice the input rate, which Price does
// not model).
type LLM struct {
	model        string
	client       anthropic.Client
	capabilities llm.Capabilities

	mu sync.Mutex
	// inFlight cancels the responses that have not settled, so closing the provider
	// abandons them rather than leaving them to their own timeouts.
	inFlight map[uint64]context.CancelFunc
	nextID   atomic.Uint64
	closed   bool
}

func newMessages(options Options) (*LLM, error) {
	keys := keypool.FromEnv(options.APIKey)
	if keys.Len() == 0 {
		return nil, stack.Wrap(errors.New("anthropic: api key is required"))
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	clientOptions := []option.RequestOption{
		// The client reads ANTHROPIC_* itself unless told not to, which would let a stray
		// ANTHROPIC_AUTH_TOKEN or ANTHROPIC_BASE_URL override what was configured here.
		option.WithoutEnvironmentDefaults(),
		option.WithAPIKey(keys.Peek()),
		option.WithHTTPClient(keys.WrapDoer(nil, keypool.Header("X-Api-Key", ""))),
	}
	if options.BaseURL != "" {
		clientOptions = append(clientOptions, option.WithBaseURL(options.BaseURL))
	}
	if options.WorkspaceID != "" {
		clientOptions = append(clientOptions, option.WithHeader(workspaceHeader, options.WorkspaceID))
	}
	return &LLM{
		model:  options.Model,
		client: anthropic.NewClient(clientOptions...),
		// Every current Claude model reads images. No effort is declared yet, so a request
		// naming one is refused rather than answered at the model's default.
		capabilities: llm.Capabilities{InputModalities: []string{llm.ModalityImage}},
		inFlight:     map[uint64]context.CancelFunc{},
	}, nil
}

// Client exposes the underlying SDK client.
func (l *LLM) Client() *anthropic.Client { return &l.client }

// Provider is the stable provider name used in stats.
func (l *LLM) Provider() string { return ProviderName }

// Model is the model identifier used in stats.
func (l *LLM) Model() string { return l.model }

// Capabilities is what this model accepts.
func (l *LLM) Capabilities() llm.Capabilities { return l.capabilities }

// Create asks for one response and returns the stream it arrives on.
func (l *LLM) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	if len(params.Input) == 0 {
		return nil, stack.Wrap(errors.New("anthropic: a request needs at least one input message"))
	}
	if err := l.capabilities.Validate(params); err != nil {
		return nil, stack.Wrap(fmt.Errorf("anthropic: %w", err))
	}

	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return nil, stack.Wrap(errors.New("anthropic: provider is closed"))
	}
	requestCtx, cancel := context.WithTimeout(ctx, defaultTimeout)
	id := l.nextID.Add(1)
	l.inFlight[id] = cancel
	l.mu.Unlock()

	upstream := l.client.Messages.NewStreaming(requestCtx, l.params(params))
	if err := upstream.Err(); err != nil {
		_ = upstream.Close()
		cancel()
		l.forget(id)
		return nil, stack.Wrap(err)
	}
	return llm.NewStream(
		llm.StreamOptions{ResponseID: params.ID, Provider: ProviderName, Model: l.model},
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

func (l *LLM) forget(id uint64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	delete(l.inFlight, id)
}

// puller turns the Messages stream into what a Stream reports.
type puller struct {
	llm      *LLM
	id       uint64
	upstream *ssestream.Stream[anthropic.MessageStreamEventUnion]
	cancel   context.CancelFunc

	// usage is what the stream has reported so far. message_start carries the prompt and
	// message_delta the running totals, so each frame only overwrites what it states.
	usage anthropic.MessageDeltaUsage
	err   error
	done  bool
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
	case "message_start":
		usage := event.Message.Usage
		p.record(w, anthropic.MessageDeltaUsage{
			InputTokens:              usage.InputTokens,
			CacheCreationInputTokens: usage.CacheCreationInputTokens,
			CacheReadInputTokens:     usage.CacheReadInputTokens,
			OutputTokens:             usage.OutputTokens,
			OutputTokensDetails:      usage.OutputTokensDetails,
		})
	case "content_block_start":
		if block := event.ContentBlock; block.Type == "tool_use" {
			w.FunctionCall(event.Index, block.ID, block.Name, "", "")
		}
	case "content_block_delta":
		switch delta := event.Delta; delta.Type {
		case "text_delta":
			w.OutputText(delta.Text)
		case "thinking_delta":
			w.ReasoningText(delta.Thinking)
		case "input_json_delta":
			w.FunctionCall(event.Index, "", "", delta.PartialJSON, "")
		}
	case "message_delta":
		p.record(w, event.Usage)
		if reason := incomplete(event.Delta.StopReason); reason != "" {
			w.Incomplete(reason)
		}
	}
	return true
}

// record folds a usage frame into what the response has used. Anthropic counts the cached
// and the written parts of the prompt apart from input_tokens; the router counts the whole
// prompt, with those parts as details of it.
func (p *puller) record(w *llm.ResponseWriter, frame anthropic.MessageDeltaUsage) {
	if frame.InputTokens > 0 {
		p.usage.InputTokens = frame.InputTokens
	}
	if frame.CacheCreationInputTokens > 0 {
		p.usage.CacheCreationInputTokens = frame.CacheCreationInputTokens
	}
	if frame.CacheReadInputTokens > 0 {
		p.usage.CacheReadInputTokens = frame.CacheReadInputTokens
	}
	if frame.OutputTokens > 0 {
		p.usage.OutputTokens = frame.OutputTokens
	}
	if frame.OutputTokensDetails.ThinkingTokens > 0 {
		p.usage.OutputTokensDetails = frame.OutputTokensDetails
	}
	w.Usage(llm.Usage{
		InputTokens: p.usage.InputTokens + p.usage.CacheCreationInputTokens + p.usage.CacheReadInputTokens,
		InputTokensDetails: llm.InputTokensDetails{
			CachedTokens:     p.usage.CacheReadInputTokens,
			CacheWriteTokens: p.usage.CacheCreationInputTokens,
		},
		OutputTokens:        p.usage.OutputTokens,
		OutputTokensDetails: llm.OutputTokensDetails{ReasoningTokens: p.usage.OutputTokensDetails.ThinkingTokens},
	})
}

// Err is the provider failure that ended the stream, if there was one.
func (p *puller) Err() error { return stack.Wrap(p.err) }

// Close abandons the response. The upstream is released by the goroutine reading it.
func (p *puller) Close() error {
	p.cancel()
	return nil
}

// finish releases the upstream and works out what ended it.
func (p *puller) finish(w *llm.ResponseWriter) {
	p.done = true
	if err := p.upstream.Err(); err != nil && !errors.Is(err, context.Canceled) {
		p.err = named(err)
	} else if err != nil {
		// A cancelled stream is barge-in or shutdown, not a provider failure.
		w.Cancelled()
	}
	p.upstream.Close()
	p.cancel()
	p.llm.forget(p.id)
}

// named keeps the error type Anthropic gave a failure inside a stream. The types that name its
// own side are transient: overloaded, rate limited, timed out or failed on its side. Every
// other type (an invalid request, credentials, permission, billing) is the request's own.
func named(err error) error {
	refused, ok := errors.AsType[*anthropic.Error](err)
	if !ok || refused.Type() == "" {
		return err
	}
	transient := false
	switch refused.Type() {
	case anthropic.ErrorTypeOverloadedError, anthropic.ErrorTypeRateLimitError,
		anthropic.ErrorTypeTimeoutError, anthropic.ErrorTypeAPIError:
		transient = true
	}
	return &llm.ProviderFailure{Err: err, Code: string(refused.Type()), Transient: transient}
}

// incomplete translates a stop reason into why a response stopped early, returning empty
// for the model having finished or stopped to call a tool.
func incomplete(reason anthropic.StopReason) string {
	switch reason {
	case anthropic.StopReasonMaxTokens, anthropic.StopReasonModelContextWindowExceeded:
		return llm.ReasonMaxOutputTokens
	case anthropic.StopReasonRefusal:
		return llm.ReasonContentFilter
	default:
		return ""
	}
}

// params builds the Messages request from the Responses-shaped one.
func (l *LLM) params(request llm.ResponseParams) anthropic.MessageNewParams {
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(l.model),
		MaxTokens: defaultMaxTokens,
	}
	if request.MaxOutputTokens > 0 {
		params.MaxTokens = int64(request.MaxOutputTokens)
	}
	if request.Instructions != "" {
		params.System = append(params.System, anthropic.TextBlockParam{Text: request.Instructions})
	}

	input := openaicompat.StrictToolOrder(request.Input)
	// System turns that open the input join the system prompt; one later in the
	// conversation is told to the model as a note on the user's side, since the API takes
	// a system turn mid-conversation only in places this history does not guarantee.
	for len(input) > 0 && input[0].Role == llm.System {
		if text := messageText(input[0]); text != "" {
			params.System = append(params.System, anthropic.TextBlockParam{Text: text})
		}
		input = input[1:]
	}
	for _, message := range input {
		params.Messages = appendMessage(params.Messages, message)
	}

	for _, tool := range request.Tools {
		definition := anthropic.ToolParam{
			Name:        tool.Name,
			InputSchema: inputSchema(tool.Parameters),
		}
		if tool.Description != "" {
			definition.Description = param.NewOpt(tool.Description)
		}
		params.Tools = append(params.Tools, anthropic.ToolUnionParam{OfTool: &definition})
	}
	switch request.ToolChoice {
	case "auto":
		params.ToolChoice = anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{}}
	case "none":
		params.ToolChoice = anthropic.ToolChoiceUnionParam{OfNone: &anthropic.ToolChoiceNoneParam{}}
	case "required":
		// Claude Opus 5.5 and Sonnet 5.5 answer this with a 400, which is still better than
		// answering it as auto.
		params.ToolChoice = anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}
	}

	breakpoints(&params, request.PromptCacheOptions.Mode)
	return params
}

// maxBreakpoints is how many cache_control markers Anthropic accepts in one request.
const maxBreakpoints = 4

// breakpoints marks where the prefix of this request is cached: after the tools, after the
// system prompt, at the end of the previous request, and at the end of this one.
//
// The last is what the next round of a tool loop reads; the previous one is where the last
// request wrote, which a read finds even when this round added more blocks than the
// API looks back over for an earlier entry. Explicit mode marks only the tools and the
// system prompt. A prefix shorter than the model's minimum is not cached, without error.
func breakpoints(params *anthropic.MessageNewParams, mode llm.CacheMode) {
	marks := 0
	mark := func(control *anthropic.CacheControlEphemeralParam) {
		if control == nil || marks == maxBreakpoints {
			return
		}
		*control = anthropic.NewCacheControlEphemeralParam()
		marks++
	}
	if n := len(params.Tools); n > 0 {
		mark(params.Tools[n-1].GetCacheControl())
	}
	if n := len(params.System); n > 0 {
		mark(&params.System[n-1].CacheControl)
	}
	if mode == llm.CacheExplicit {
		return
	}
	last := len(params.Messages) - 1
	if previous := previousTurn(params.Messages); previous >= 0 {
		mark(lastBlock(params.Messages[previous]))
	}
	if last >= 0 {
		mark(lastBlock(params.Messages[last]))
	}
}

// previousTurn is the index of the last user turn before the latest assistant turn, which is
// where the request that turn answered ended, or -1 when there was none.
func previousTurn(messages []anthropic.MessageParam) int {
	assistant := false
	for i := len(messages) - 1; i >= 0; i-- {
		switch {
		case messages[i].Role == anthropic.MessageParamRoleAssistant:
			assistant = true
		case assistant:
			return i
		}
	}
	return -1
}

func lastBlock(message anthropic.MessageParam) *anthropic.CacheControlEphemeralParam {
	if len(message.Content) == 0 {
		return nil
	}
	return message.Content[len(message.Content)-1].GetCacheControl()
}

// appendMessage adds one turn, merging it into the turn before when both are the same
// side's, since a tool call's results and what the caller said after them are one user
// turn to the API.
func appendMessage(messages []anthropic.MessageParam, message llm.Message) []anthropic.MessageParam {
	role := anthropic.MessageParamRoleUser
	var blocks []anthropic.ContentBlockParamUnion
	switch message.Role {
	case llm.Assistant:
		role = anthropic.MessageParamRoleAssistant
		if text := messageText(message); text != "" {
			blocks = append(blocks, anthropic.NewTextBlock(text))
		}
		for _, call := range message.ToolCalls {
			arguments := json.RawMessage(call.Arguments)
			if !json.Valid(arguments) {
				arguments = json.RawMessage("{}")
			}
			blocks = append(blocks, anthropic.NewToolUseBlock(call.ID, arguments, call.Name))
		}
	case llm.ToolResult:
		result := anthropic.ToolResultBlockParam{ToolUseID: message.ToolCallID}
		if text := messageText(message); text != "" {
			result.Content = append(result.Content, anthropic.ToolResultBlockParamContentUnion{OfText: &anthropic.TextBlockParam{Text: text}})
		}
		for _, part := range message.Parts {
			if part.Image != nil {
				result.Content = append(result.Content, anthropic.ToolResultBlockParamContentUnion{OfImage: image(*part.Image)})
			}
		}
		blocks = append(blocks, anthropic.ContentBlockParamUnion{OfToolResult: &result})
	default:
		if len(message.Parts) == 0 {
			if message.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(message.Content))
			}
			break
		}
		for _, part := range message.Parts {
			switch {
			case part.Image != nil:
				blocks = append(blocks, anthropic.ContentBlockParamUnion{OfImage: image(*part.Image)})
			case part.Text != "":
				blocks = append(blocks, anthropic.NewTextBlock(part.Text))
			}
		}
	}
	// An empty turn is a 400, and says nothing anyway.
	if len(blocks) == 0 {
		return messages
	}
	if n := len(messages); n > 0 && messages[n-1].Role == role {
		messages[n-1].Content = append(messages[n-1].Content, blocks...)
		return messages
	}
	return append(messages, anthropic.MessageParam{Role: role, Content: blocks})
}

func image(part llm.ImagePart) *anthropic.ImageBlockParam {
	if part.URL != "" {
		return &anthropic.ImageBlockParam{Source: anthropic.ImageBlockParamSourceUnion{OfURL: &anthropic.URLImageSourceParam{URL: part.URL}}}
	}
	mime := part.MIME
	if mime == "" {
		mime = "image/jpeg"
	}
	return &anthropic.ImageBlockParam{Source: anthropic.ImageBlockParamSourceUnion{OfBase64: &anthropic.Base64ImageSourceParam{
		Data:      base64.StdEncoding.EncodeToString(part.Data),
		MediaType: anthropic.Base64ImageSourceMediaType(mime),
	}}}
}

// inputSchema carries a tool's JSON Schema whole. Only properties and required have fields
// of their own; the rest of the schema rides along as extra fields.
func inputSchema(schema map[string]any) anthropic.ToolInputSchemaParam {
	rendered := anthropic.ToolInputSchemaParam{ExtraFields: map[string]any{}}
	for key, value := range schema {
		switch key {
		case "type":
		case "properties":
			rendered.Properties = value
		case "required":
			if names, ok := value.([]string); ok {
				rendered.Required = names
				continue
			}
			rendered.ExtraFields[key] = value
		default:
			rendered.ExtraFields[key] = value
		}
	}
	return rendered
}

func messageText(message llm.Message) string {
	if message.Content != "" {
		return message.Content
	}
	return llm.TextOf(message.Parts)
}
