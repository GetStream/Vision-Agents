package llmrouter

import (
	"context"
	"errors"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// lateAfter is how long the rigs below let a reply say nothing before it is asked again.
const lateAfter = 1200 * time.Millisecond

// rig is a router over some candidates of one target, alpha/quick first and beta/steady behind
// it, whose answers the test writes as it goes. It runs in a synctest bubble, so the clock is
// the bubble's and a test moves it by sleeping.
type rig struct {
	session *Session

	mu      sync.Mutex
	headers map[string]time.Duration
	failing map[string]error
	scripts map[string][]*llmtest.Script
	timings []llm.CallTiming
}

// pace is one candidate as the registry builds it.
type pace struct {
	rig         *rig
	name, model string

	mu      sync.Mutex
	scripts []*llmtest.Script
}

func (p *pace) Start(context.Context) error { return nil }

func (p *pace) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	p.rig.mu.Lock()
	wait, failure := p.rig.headers[p.name], p.rig.failing[p.name]
	p.rig.mu.Unlock()
	if wait > 0 {
		select {
		case <-time.After(wait):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if failure != nil {
		return nil, failure
	}
	script := llmtest.New(llm.StreamOptions{ResponseID: params.ID, Provider: p.name, Model: p.model})
	p.mu.Lock()
	p.scripts = append(p.scripts, script)
	p.mu.Unlock()
	p.rig.mu.Lock()
	p.rig.scripts[p.name] = append(p.rig.scripts[p.name], script)
	p.rig.mu.Unlock()
	return script.Stream(), nil
}

func (p *pace) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, script := range p.scripts {
		script.Done()
	}
	return nil
}

func (p *pace) Provider() string               { return p.name }
func (p *pace) Model() string                  { return p.model }
func (p *pace) Capabilities() llm.Capabilities { return llm.Capabilities{} }

// newRig opens a session on a target made of the named candidates, in the order given, and
// hedges its replies after the given time.
func newRig(t *testing.T, hedge time.Duration, candidates ...string) *rig {
	t.Helper()
	r := &rig{
		headers: map[string]time.Duration{},
		failing: map[string]error{},
		scripts: map[string][]*llmtest.Script{},
	}
	models := map[string]string{"alpha": "quick", "beta": "steady"}
	registry := NewRegistry()
	var config routing.ModalityConfig
	var members []string
	for _, name := range candidates {
		registry.Register(name, func(spec routing.Spec) (Provider, error) {
			return &pace{rig: r, name: name, model: spec.Model}, nil
		})
		config.Providers = append(config.Providers, routing.ProviderConfig{
			Provider: name, Model: models[name], Languages: []string{"en"},
		})
		members = append(members, name+"/"+models[name])
	}
	config.Aliases = map[string]routing.Alias{"voice": {Prefer: members[0], Only: members}}
	router, err := New(Options{Config: config, Registry: registry, ReplyHedge: hedge})
	require.NoError(t, err)
	r.session, err = router.Start(context.Background(), Request{CustomerID: "acme", Target: "voice"})
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, r.session.Close())
		router.Close()
	})
	return r
}

// slow makes a candidate take that long to answer a request with its headers.
func (r *rig) slow(name string, d time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.headers[name] = d
}

// fail makes a candidate refuse every request.
func (r *rig) fail(name string, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.failing[name] = err
}

// params is a reply, which reports the model calls made for it to the rig.
func (r *rig) params() llm.ResponseParams {
	return llm.ResponseParams{
		ID: "turn-1", Purpose: replyPurpose, TurnID: "turn-1", Input: prompt(),
		OnTiming: func(timing llm.CallTiming) {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.timings = append(r.timings, timing)
		},
	}
}

// reply is a request the test has put to the session, which is answered when its first content
// arrives.
type reply struct {
	stream *llm.Stream
	err    error
	done   chan struct{}
}

func (r *rig) ask(ctx context.Context, params llm.ResponseParams) *reply {
	asked := &reply{done: make(chan struct{})}
	go func() {
		defer close(asked.done)
		asked.stream, asked.err = r.session.Create(ctx, params)
	}()
	synctest.Wait()
	return asked
}

func (r *reply) answered() bool {
	select {
	case <-r.done:
		return true
	default:
		return false
	}
}

// pass moves the clock on, and returns once everything that could react to it has.
func pass(d time.Duration) {
	time.Sleep(d)
	synctest.Wait()
}

// say has a candidate stream some of its answer, and returns once everything has reacted.
func (r *rig) say(name, text string) {
	r.script(name).OutputText(text)
	synctest.Wait()
}

// requests is how many requests a candidate has been asked.
func (r *rig) requests(name string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.scripts[name])
}

// script is the answer a candidate is writing to the request it was asked.
func (r *rig) script(name string) *llmtest.Script {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.scripts[name][0]
}

// called is which providers a model call was reported for, and whether it went well.
func (r *rig) called() map[string]bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	called := map[string]bool{}
	for _, timing := range r.timings {
		called[timing.Provider] = timing.Success
	}
	return called
}

// finish ends a candidate's answer and reads out what the caller was given, the way the agent
// does.
func (r *rig) finish(name string, asked *reply) llm.Response {
	r.script(name).Done()
	return drain(asked.stream)
}

func TestAReplyThatSaysSomethingOnTimeIsNotHedged(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, lateAfter, "alpha", "beta")
		asked := r.ask(context.Background(), r.params())

		pass(lateAfter - time.Millisecond)
		r.say("alpha", "Hello")
		require.True(t, asked.answered())
		require.NoError(t, asked.err)

		pass(time.Second)
		require.Zero(t, r.requests("beta"), "nothing was late, so nothing was asked twice")
		require.Equal(t, "alpha", r.finish("alpha", asked).Provider)
	})
}

func TestALateReplyIsAskedOfAnotherCandidateAndTheHedgeCanWin(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, lateAfter, "alpha", "beta")
		asked := r.ask(context.Background(), r.params())

		pass(lateAfter - time.Millisecond)
		require.Zero(t, r.requests("beta"))
		pass(time.Millisecond)
		require.Equal(t, 1, r.requests("beta"), "the same request was asked again once it was late")
		require.False(t, asked.answered())

		pass(300 * time.Millisecond)
		r.say("beta", "Hello")
		require.True(t, asked.answered())
		require.NoError(t, asked.err)

		require.True(t, r.script("alpha").Abandoned(), "the original was stopped the moment it lost")
		require.Equal(t, llm.StatusCancelled, r.script("alpha").Stream().Response().Status)
		response := r.finish("beta", asked)
		require.Equal(t, "beta", response.Provider)
		require.Equal(t, "Hello", response.OutputText)
		require.False(t, r.script("beta").Abandoned())
		require.Equal(t, map[string]bool{"alpha": true, "beta": true}, r.called(),
			"both calls were reported, the original as one that was cut short")
	})
}

func TestALateReplyThatTheOriginalStillAnswersFirstCancelsTheHedge(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, lateAfter, "alpha", "beta")
		asked := r.ask(context.Background(), r.params())

		pass(lateAfter + 300*time.Millisecond)
		require.Equal(t, 1, r.requests("beta"))
		r.say("alpha", "Hello")
		require.True(t, asked.answered())

		require.True(t, r.script("beta").Abandoned(), "the hedge was stopped the moment it lost")
		require.Equal(t, llm.StatusCancelled, r.script("beta").Stream().Response().Status)
		require.Equal(t, "alpha", r.finish("alpha", asked).Provider)
		require.Equal(t, map[string]bool{"alpha": true, "beta": true}, r.called())
		r.session.mu.Lock()
		held := len(r.session.children)
		r.session.mu.Unlock()
		require.Zero(t, held, "the hedge's session was let go of with its stream")
	})
}

func TestAToolCallDecidesTheRaceAsTextDoes(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, lateAfter, "alpha", "beta")
		asked := r.ask(context.Background(), r.params())

		pass(lateAfter)
		r.script("alpha").ReasoningText("Thinking about it")
		synctest.Wait()
		require.False(t, asked.answered(), "thinking is not an answer")
		r.script("beta").ToolCalls(llm.ToolCall{ID: "call_1", Name: "transfer", Arguments: "{}"})
		synctest.Wait()
		require.True(t, asked.answered())

		response := r.finish("beta", asked)
		require.Equal(t, "beta", response.Provider)
		require.Equal(t, "transfer", response.ToolCalls[0].Name)
	})
}

func TestARequestStillBeingAskedIsHedgedToo(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, lateAfter, "alpha", "beta")
		r.slow("alpha", 10*time.Second)
		asked := r.ask(context.Background(), r.params())

		pass(lateAfter)
		require.Zero(t, r.requests("alpha"), "the original has not been answered at all yet")
		r.say("beta", "Hello")

		require.True(t, asked.answered())
		require.Equal(t, "beta", r.finish("beta", asked).Provider)
		require.Zero(t, r.requests("alpha"), "the original was cancelled while it was still being asked")
		require.Equal(t, map[string]bool{"alpha": false, "beta": true}, r.called(),
			"a request cancelled before it was answered is reported as any cancelled one is")
	})
}

func TestNothingIsHedgedWithoutAnotherCandidate(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, lateAfter, "alpha")
		asked := r.ask(context.Background(), r.params())

		pass(3 * lateAfter)
		require.False(t, asked.answered())
		r.say("alpha", "Late, but alone")

		require.True(t, asked.answered())
		require.NoError(t, asked.err)
		require.Equal(t, "Late, but alone", r.finish("alpha", asked).OutputText)
		require.Equal(t, map[string]bool{"alpha": true}, r.called())
	})
}

func TestAHedgeThatFailsLeavesTheOriginalGoing(t *testing.T) {
	for name, trouble := range map[string]func(*rig){
		"when it cannot be asked": func(r *rig) {
			r.fail("beta", errors.New("beta is down"))
			pass(lateAfter)
		},
		"when its answer fails": func(r *rig) {
			pass(lateAfter)
			r.script("beta").Fail(errors.New("beta stopped"), "stream")
			synctest.Wait()
		},
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				r := newRig(t, lateAfter, "alpha", "beta")
				asked := r.ask(context.Background(), r.params())

				trouble(r)
				require.False(t, asked.answered())
				require.False(t, r.script("alpha").Abandoned(), "the original was left alone")
				r.say("alpha", "Hello")
				require.True(t, asked.answered())
				require.NoError(t, asked.err)

				require.Equal(t, "alpha", r.finish("alpha", asked).Provider)
				require.Equal(t, map[string]bool{"alpha": true, "beta": false}, r.called(),
					"the hedge was reported as the failure it was")
			})
		})
	}
}

func TestWhenBothFailTheOriginalsOutcomeStands(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, lateAfter, "alpha", "beta")
		r.fail("beta", errors.New("beta is down"))
		asked := r.ask(context.Background(), r.params())

		pass(2 * lateAfter)
		r.script("alpha").Fail(errors.New("alpha stopped"), "stream")
		synctest.Wait()

		require.True(t, asked.answered())
		require.NoError(t, asked.err)
		require.Equal(t, llm.StatusFailed, drain(asked.stream).Status, "the caller sees the failure it would have seen")
	})
}

func TestARequestIsNotHedgedAtZeroOrWhenItIsNotAReply(t *testing.T) {
	for name, change := range map[string]func(*llm.ResponseParams){
		"at zero":                    func(*llm.ResponseParams) {},
		"for the flow controller":    func(p *llm.ResponseParams) { p.Purpose = "flow" },
		"for a held conversation":    func(p *llm.ResponseParams) { p.PreviousResponseID = "resp_1" },
		"for a provider's own state": func(p *llm.ResponseParams) { p.Conversation = "conv_1" },
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				after := lateAfter
				if name == "at zero" {
					after = 0
				}
				r := newRig(t, after, "alpha", "beta")
				params := r.params()
				change(&params)
				asked := r.ask(context.Background(), params)

				pass(5 * lateAfter)
				require.Zero(t, r.requests("beta"))
				r.say("alpha", "Hello")
				require.True(t, asked.answered())
				require.Equal(t, "alpha", r.finish("alpha", asked).Provider)
			})
		})
	}
}

func TestACallerWhoHangsUpDuringTheRaceStopsBothRequests(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, lateAfter, "alpha", "beta")
		ctx, hangUp := context.WithCancel(context.Background())
		asked := r.ask(ctx, r.params())
		pass(lateAfter)

		hangUp()
		synctest.Wait()

		require.True(t, asked.answered())
		require.ErrorIs(t, asked.err, context.Canceled)
		require.True(t, r.script("alpha").Abandoned())
		require.True(t, r.script("beta").Abandoned())
		require.Equal(t, map[string]bool{"alpha": true, "beta": true}, r.called())
	})
}

func TestAHedgeGoesToADifferentModelBeforeADifferentProvider(t *testing.T) {
	asked := routing.ProviderConfig{Provider: "gemini", Model: "flash"}
	available := live.Health{Available: true}
	candidate := func(provider, model string, health live.Health) routing.Candidate {
		return routing.Candidate{Config: routing.ProviderConfig{Provider: provider, Model: model}, Health: health}
	}

	ordered := hedgeCandidates([]routing.Candidate{
		candidate("gemini", "flash", available),
		candidate("together", "flash", available),
		candidate("gemini", "lite", available),
		candidate("openai", "mini", available),
		candidate("xai", "grok", live.Health{}),
		candidate("openai", "nano", available),
	}, asked)

	names := make([]string, 0, len(ordered))
	for _, c := range ordered {
		names = append(names, c.Config.Name())
	}
	require.Equal(t, []string{"openai/mini", "openai/nano", "gemini/lite", "together/flash"}, names,
		"never the one asked, never an unavailable one, and the ranking decides between equals")
}
