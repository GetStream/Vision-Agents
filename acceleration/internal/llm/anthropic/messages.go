package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"slices"
	"strings"
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
// A reasoning effort goes out as output_config.effort on the models that take one
// (efforts), and a replayed turn's thinking as the signed blocks it came as (withThinking).
// Not carried over from the compatibility endpoint: temperature (Claude 4.7 and later
// refuse any but the default), JSON modes (the endpoint ignored them; callers already check
// what comes back), and the one-hour cache TTL (its writes bill at twice the input rate, which Price does
// not model).
type LLM struct {
	model        string
	client       anthropic.Client
	capabilities llm.Capabilities
	// systemTurns is whether this model takes a system message in the middle of the
	// conversation (systemTurnModels).
	systemTurns bool

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
		// Every current Claude model reads images. High is the API's own default effort,
		// so naming it changes nothing for a request that names none.
		capabilities: llm.Capabilities{InputModalities: []string{llm.ModalityImage}, ReasoningEfforts: efforts(options.Model), DefaultEffort: defaultEffort(options.Model), ToolChoices: toolChoices(options.Model)},
		systemTurns:  takesSystemTurns(options.Model),
		inFlight:     map[uint64]context.CancelFunc{},
	}, nil
}

// unforcedToolModels answer tool_choice any or tool with a 400, so a request that needs a
// tool called is refused before it goes out rather than halfway through a call.
var unforcedToolModels = []string{"claude-fable-5-2"}

// toolChoices are the tool choices a model honours, or nil for all of them.
func toolChoices(model string) []string {
	if slices.Contains(unforcedToolModels, datedModel.ReplaceAllString(model, "")) {
		return []string{"auto", "none"}
	}
	return nil
}

// systemTurnModels take a system message after a user turn mid-conversation and follow it,
// without disturbing the cached prefix before it. The API takes one from more models than
// these, but Sonnet 5.5 followed one in 6 of 9 tries and Haiku 5.5 in 1 of 8, against every
// try when it was told on the user's side; the Opus models followed every one. Any other
// model, measured or not, is told on the user's side, as it always was.
var systemTurnModels = []string{"claude-opus-5-5", "claude-opus-5"}

// xhighModels and maxModels take the efforts above high, by the API's effort page
// (platform.claude.com/docs/en/build-with-claude/effort): xhigh and max on the first, max
// alone on the second. Any other Claude model gets low, medium and high.
var (
	xhighModels = []string{
		"claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5", "claude-haiku-5-5",
		"claude-opus-4-8", "claude-opus-4-7",
	}
	maxModels = []string{"claude-mythos-preview", "claude-opus-4-6", "claude-sonnet-4-6"}
)

// efforts are the output_config efforts a model takes.
func efforts(model string) []string {
	if !strings.HasPrefix(model, "claude-") {
		return nil
	}
	family := datedModel.ReplaceAllString(model, "")
	isFamily := func(prefix string) bool { return strings.HasPrefix(family, prefix) }
	switch {
	case slices.ContainsFunc(xhighModels, isFamily):
		return []string{"low", "medium", "high", "xhigh", "max"}
	case slices.ContainsFunc(maxModels, isFamily):
		return []string{"low", "medium", "high", "max"}
	}
	return []string{"low", "medium", "high"}
}

// defaultEffort is the API's own default for a model that takes an effort.
func defaultEffort(model string) string {
	if efforts(model) == nil {
		return ""
	}
	return "high"
}

// datedModel is a model name pinned to a snapshot date, which is the same model.
var datedModel = regexp.MustCompile(`-\d{8}$`)

func takesSystemTurns(model string) bool {
	return slices.Contains(systemTurnModels, datedModel.ReplaceAllString(model, ""))
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
	// counted is a message_delta read: message_start's output count is about 1, and only a
	// delta's is what the response generated.
	counted bool
	err     error
	done    bool

	// thinking is the thinking block being streamed, recorded when it stops: a block cut
	// off before its signature cannot be sent back.
	thinking     *llm.ReasoningBlock
	thinkingText strings.Builder
	// calls, text and last are where the response stands: the tool calls begun, the bytes
	// of text written and the type of the latest block. A block is recorded at that place.
	calls int
	text  int
	last  string
	// textAt is where the latest text block began, so one that ends empty is noticed: a
	// replay has no block to put back for it.
	textAt int
	// unreplayable is a response laid out as a replay cannot rebuild it from one text and
	// its calls (replayable); no block from that point on is recorded.
	unreplayable bool
	// held are the blocks after a tool call, recorded only once the response ends: text
	// arriving after a call is put back before the calls, inside the prefix they were signed
	// with, so it withholds them all.
	held []llm.ReasoningBlock
	// argued is whether the tool call being streamed has had arguments.
	argued bool
}

// keep records a finished block, or holds it until the end when it follows a tool call.
func (p *puller) keep(w *llm.ResponseWriter, block llm.ReasoningBlock) {
	if block.AfterToolCalls > 0 {
		p.held = append(p.held, block)
		return
	}
	w.ReasoningBlock(block)
}

// place marks where a block of thinking starting now stands: after the text written so far
// while no tool call has started, after the calls otherwise (a replay puts all the text
// before the calls).
func (p *puller) place() llm.ReasoningBlock {
	if p.calls > 0 {
		return llm.ReasoningBlock{AfterToolCalls: p.calls}
	}
	return llm.ReasoningBlock{AfterText: p.text}
}

// replayable reports whether a text block starting now still leaves the response one a
// replay rebuilds exactly. The socket carries a turn's text as one string ahead of its calls,
// so text after a call, or two text blocks with no thinking between them to split them, would
// come back moved or merged, and thinking after them would be sent with a prefix Claude never
// wrote.
func (p *puller) replayable() bool {
	return p.calls == 0 && p.last != "text"
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
		switch block := event.ContentBlock; block.Type {
		case "text":
			p.unreplayable = p.unreplayable || !p.replayable()
			p.textAt = p.text
			if p.calls > 0 {
				p.held = nil
			}
		case "tool_use":
			w.FunctionCall(event.Index, block.ID, block.Name, "", "")
			p.calls++
			p.argued = false
		case "thinking":
			if !p.unreplayable {
				placed := p.place()
				p.thinking = &placed
				p.thinkingText.Reset()
			}
		case "redacted_thinking":
			p.unreplayable = p.unreplayable || block.Data == ""
			if !p.unreplayable {
				placed := p.place()
				placed.Redacted = block.Data
				p.keep(w, placed)
			}
		}
		p.last = string(event.ContentBlock.Type)
	case "content_block_stop":
		switch {
		case p.last == "text" && p.text == p.textAt:
			p.unreplayable = true
		case p.last == "tool_use" && !p.argued:
			// A call that takes nothing streams no arguments; replayed, it needs an object.
			w.FunctionCall(event.Index, "", "", "{}", "")
		}
		// A block that cannot be sent back leaves a gap, and the thinking sent back has to
		// be an unbroken run: none after it is kept.
		if p.thinking != nil {
			if p.thinking.Signature == "" {
				p.unreplayable = true
			} else {
				p.thinking.Text = p.thinkingText.String()
				p.keep(w, *p.thinking)
			}
		}
		p.thinking = nil
	case "content_block_delta":
		switch delta := event.Delta; delta.Type {
		case "text_delta":
			w.OutputText(delta.Text)
			p.text += len(delta.Text)
		case "thinking_delta":
			w.ReasoningText(delta.Thinking)
			if p.thinking != nil {
				p.thinkingText.WriteString(delta.Thinking)
			}
		case "signature_delta":
			if p.thinking != nil {
				p.thinking.Signature = delta.Signature
			}
		case "input_json_delta":
			w.FunctionCall(event.Index, "", "", delta.PartialJSON, "")
			p.argued = p.argued || delta.PartialJSON != ""
		}
	case "message_delta":
		p.counted = true
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
		OutputUnfinished:    !p.counted,
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
	for _, block := range p.held {
		w.ReasoningBlock(block)
	}
	switch err := p.upstream.Err(); {
	case err == nil:
	case errors.Is(err, context.Canceled):
		// A cancelled stream is barge-in or shutdown, not a provider failure.
		w.Cancelled()
	case errors.Is(err, context.DeadlineExceeded):
		// Ours or the caller's: the provider billed what it generated until then.
		w.TimedOut()
		p.err = err
	default:
		p.err = named(err)
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
	return &llm.ProviderFailure{Err: err, Transient: transient}
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
	// Only a named effort goes out: unnamed, the API applies its own (DefaultEffort).
	if effort := request.Reasoning.Effort; effort != "" {
		params.OutputConfig = anthropic.OutputConfigParam{Effort: anthropic.OutputConfigEffort(effort)}
	}

	input := openaicompat.StrictToolOrder(request.Input)
	// System turns that open the input join the system prompt.
	for len(input) > 0 && input[0].Role == llm.System {
		if text := messageText(input[0]); text != "" {
			params.System = append(params.System, anthropic.TextBlockParam{Text: text})
		}
		input = input[1:]
	}
	for i, message := range input {
		if message.Role == llm.System && l.systemTurns && systemTurnFits(params.Messages, input[i+1:]) {
			params.Messages = appendSystem(params.Messages, message)
			continue
		}
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
	last := lastMarkable(params.Messages)
	if previous := previousTurn(params.Messages); previous >= 0 && previous != last {
		mark(lastBlock(params.Messages[previous]))
	}
	if last >= 0 {
		mark(lastBlock(params.Messages[last]))
	}
}

// previousTurn is the index of the last user turn before the latest assistant turn, which is
// where the request that turn answered ended, or -1 when there was none. A system message
// between them is passed over: it is never marked (lastMarkable).
func previousTurn(messages []anthropic.MessageParam) int {
	assistant := false
	for i := len(messages) - 1; i >= 0; i-- {
		switch {
		case messages[i].Role == anthropic.MessageParamRoleAssistant:
			assistant = true
		case messages[i].Role == anthropic.MessageParamRoleSystem:
		case assistant:
			return i
		}
	}
	return -1
}

// lastMarkable is the index of the last message a breakpoint may go on, or -1. A system
// message in the conversation is not one: what it says is for this turn of the reply, and
// the cached prefix is what comes before it.
func lastMarkable(messages []anthropic.MessageParam) int {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != anthropic.MessageParamRoleSystem {
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

// systemTurnFits reports whether a system message can go here as one: the API takes it only
// straight after a user turn, and only last or before the assistant's turn. Anywhere else it
// is told on the user's side.
func systemTurnFits(before []anthropic.MessageParam, after []llm.Message) bool {
	n := len(before)
	if n == 0 || before[n-1].Role != anthropic.MessageParamRoleUser && before[n-1].Role != anthropic.MessageParamRoleSystem {
		return false
	}
	if before[n-1].Role == anthropic.MessageParamRoleSystem && (n < 2 || before[n-2].Role != anthropic.MessageParamRoleUser) {
		return false
	}
	for _, next := range after {
		// A turn sent as nothing is not there to follow the system message.
		if next.Role == llm.System || rendersEmpty(next) {
			continue
		}
		return next.Role == llm.Assistant
	}
	return true
}

// rendersEmpty reports whether appendMessage sends nothing for a message, as it does for one
// with no text, image or tool call: an empty turn is a 400.
func rendersEmpty(message llm.Message) bool {
	if message.Role == llm.ToolResult || len(message.ToolCalls) > 0 || message.HasImage() {
		return false
	}
	return messageText(message) == ""
}

// appendSystem adds a system message, joined to one just before it.
func appendSystem(messages []anthropic.MessageParam, message llm.Message) []anthropic.MessageParam {
	text := messageText(message)
	if text == "" {
		return messages
	}
	block := anthropic.NewTextBlock(text)
	if n := len(messages); n > 0 && messages[n-1].Role == anthropic.MessageParamRoleSystem {
		messages[n-1].Content = append(messages[n-1].Content, block)
		return messages
	}
	return append(messages, anthropic.MessageParam{Role: anthropic.MessageParamRoleSystem, Content: []anthropic.ContentBlockParamUnion{block}})
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
		text := messageText(message)
		var calls []anthropic.ContentBlockParamUnion
		for _, call := range message.ToolCalls {
			arguments := json.RawMessage(call.Arguments)
			if !json.Valid(arguments) {
				arguments = json.RawMessage("{}")
			}
			calls = append(calls, anthropic.NewToolUseBlock(call.ID, arguments, call.Name))
		}
		if text != "" || len(calls) > 0 {
			blocks = withThinking(text, calls, message.ReasoningBlocks)
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

// withThinking lays out an assistant turn with its thinking back where the model wrote it:
// a block before any call goes after the bytes of text that preceded it (AfterText), which
// splits the text into the blocks it came as, and a block after a call goes straight after
// that call. The API refuses a turn whose thinking was moved, edited or left out between two
// kept blocks, so an unsigned block (which it refuses too) is the only one dropped. Positions
// are checked where they arrive (the socket's respond frame); the bounds here only keep a
// slice in range.
func withThinking(text string, calls []anthropic.ContentBlockParamUnion, thinking []llm.ReasoningBlock) []anthropic.ContentBlockParamUnion {
	out := make([]anthropic.ContentBlockParamUnion, 0, 1+len(calls)+len(thinking))
	written, next := 0, 0
	writeText := func(upTo int) {
		if upTo = min(upTo, len(text)); upTo > written {
			out = append(out, anthropic.NewTextBlock(text[written:upTo]))
			written = upTo
		}
	}
	think := func(after int) {
		for ; next < len(thinking) && min(thinking[next].AfterToolCalls, len(calls)) <= after; next++ {
			block := thinking[next]
			if block.Redacted == "" && block.Signature == "" {
				continue
			}
			if after == 0 {
				writeText(block.AfterText)
			}
			if block.Redacted != "" {
				out = append(out, anthropic.NewRedactedThinkingBlock(block.Redacted))
			} else {
				out = append(out, anthropic.NewThinkingBlock(block.Signature, block.Text))
			}
		}
	}
	think(0)
	writeText(len(text))
	for i, call := range calls {
		out = append(out, call)
		think(i + 1)
	}
	return out
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
