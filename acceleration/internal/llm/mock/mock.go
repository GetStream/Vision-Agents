// Package mock implements the llm.LLM contract without a vendor behind it, for load
// tests that want a cascaded conversation without spending tokens.
//
// It answers the same sentence to everything, a word at a time, after a wait: what a
// conversation downstream is sensitive to is when the first token arrives and how fast
// the rest follow, and both are settings here. Nothing is billed, so a run of thousands
// of turns costs nothing and shows up in no usage.
package mock

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "mock"

// DefaultModel is what the catalogue declares.
const DefaultModel = "mock-llm"

// DefaultAnswer is what every response says. It is a couple of seconds of speech, which
// is the length of reply a voice agent gives.
const DefaultAnswer = "We are open from nine in the morning until six in the evening on " +
	"Saturdays, and I can book you a table at any time in between."

// The router asks itself two things in JSON on the way to a reply, and a sentence is not
// an answer to either. A caller is waiting on both, so both are answered here rather than
// left to a real model a load test was meant to keep out of the way.
const (
	// flowAnswer is what the flow controller is told about every turn: the words were
	// meant for the agent, and speech arriving while it talks does not stop it. A load
	// test is about the turns happening at all, not about who holds the floor.
	flowAnswer = `{"disposition":"respond","floor":"continue"}`
	// emptyObject answers everything else that asked for an object, such as naming the
	// conversation. It satisfies the format without claiming a field, which leaves the
	// caller with what it had.
	emptyObject = `{}`
)

const (
	// defaultTimeToFirstTokenMs is the wait before the first word, in the range the fast
	// hosted models answer in.
	defaultTimeToFirstTokenMs = 250
	// defaultTokensPerSecond is how fast the rest follow.
	defaultTokensPerSecond = 80.0
)

// Options configures the provider.
type Options struct {
	Model string
	// Answer is what every response says. Empty uses DefaultAnswer.
	Answer string
	// TimeToFirstTokenMs is the wait before the first word, and TokensPerSecond how fast
	// the rest follow.
	TimeToFirstTokenMs int
	TokensPerSecond    float64
	Logger             *slog.Logger
}

// LLM is a model that answers one sentence to everything.
type LLM struct {
	model              string
	answer             string
	timeToFirstTokenMs int
	tokensPerSecond    float64
	logger             *slog.Logger
}

// New returns a provider.
func New(options Options) (*LLM, error) {
	if options.TokensPerSecond < 0 {
		return nil, errors.New("mock: a token rate cannot be negative")
	}

	provider := &LLM{
		model:              options.Model,
		answer:             options.Answer,
		timeToFirstTokenMs: options.TimeToFirstTokenMs,
		tokensPerSecond:    options.TokensPerSecond,
		logger:             options.Logger,
	}
	if provider.model == "" {
		provider.model = DefaultModel
	}
	if provider.answer == "" {
		provider.answer = DefaultAnswer
	}
	if provider.timeToFirstTokenMs == 0 {
		provider.timeToFirstTokenMs = defaultTimeToFirstTokenMs
	}
	if provider.tokensPerSecond == 0 {
		provider.tokensPerSecond = defaultTokensPerSecond
	}
	if provider.logger == nil {
		provider.logger = slog.Default()
	}
	return provider, nil
}

// Create answers whatever it was asked, a word at a time.
func (l *LLM) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	if err := l.Capabilities().Validate(params); err != nil {
		return nil, err
	}

	ctx, abandon := context.WithCancel(ctx)
	return llm.NewStream(llm.StreamOptions{
		ResponseID: params.ID,
		Provider:   ProviderName,
		Model:      l.model,
	}, &puller{
		ctx:     ctx,
		abandon: abandon,
		words:   strings.SplitAfter(l.answerTo(params), " "),
		first:   time.Duration(l.timeToFirstTokenMs) * time.Millisecond,
		gap:     time.Duration(float64(time.Second) / l.tokensPerSecond),
	}), nil
}

// answerTo is what this request is answered with: the configured sentence, unless an
// object was asked for, in which case a sentence would not parse.
func (l *LLM) answerTo(params llm.ResponseParams) string {
	if params.Text.Format != llm.FormatJSONObject {
		return l.answer
	}
	if params.Purpose == "flow" {
		return flowAnswer
	}
	return emptyObject
}

// Provider is the stable provider name used in stats.
func (l *LLM) Provider() string { return ProviderName }

// Model is the model identifier used in stats.
func (l *LLM) Model() string { return l.model }

// Capabilities is what this model accepts, which is a prompt and nothing else.
func (l *LLM) Capabilities() llm.Capabilities { return llm.Capabilities{} }

// Close releases the provider. There is nothing upstream to release.
func (l *LLM) Close() error { return nil }

// puller hands the answer over one word at a time, at the pace the provider was
// configured with.
type puller struct {
	ctx     context.Context
	abandon context.CancelFunc
	words   []string
	first   time.Duration
	gap     time.Duration
	said    int
}

// Advance writes the next word, after the wait it would have taken to generate.
func (p *puller) Advance(w *llm.ResponseWriter) bool {
	if p.said >= len(p.words) {
		// Nothing is billed: a mocked turn must not show up in anybody's usage.
		w.Usage(llm.Usage{})
		return false
	}

	wait := p.gap
	if p.said == 0 {
		wait = p.first
	}
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-timer.C:
	case <-p.ctx.Done():
		return false
	}

	w.OutputText(p.words[p.said])
	p.said++
	return true
}

// Err is the failure that ended the stream. There is none: a mocked response either
// answers or is abandoned.
func (p *puller) Err() error { return nil }

// Close abandons the response, which is what barge-in does.
func (p *puller) Close() error {
	p.abandon()
	return nil
}
