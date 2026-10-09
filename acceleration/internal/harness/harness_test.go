package harness

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	llmoptions "github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// settleFor is how long a test waits for an expectation to become true. The flow crosses
// several goroutines, so the alternative to waiting is asserting on a race.
const settleFor = 3 * time.Second

var errModelDown = errors.New("the model is down")

// stubLLM answers whatever it is asked with whatever the test has queued, and only when
// the test says so, which is what lets a task be caught mid-flight.
type stubLLM struct {
	mu    sync.Mutex
	asked []llm.ResponseParams
	// answers maps a response id to what comes back. A request with no answer stays in
	// flight until the test settles it.
	answers map[string]string
	// automatic answers every request with this text, for tests that do not care which
	// response is which.
	automatic string
	// calls are tool calls to ask for, one queue entry per request, which is what lets a
	// test answer with a tool once and with words the next time it is asked.
	calls [][]llm.ToolCall
	// failing makes every request report a provider failure before it settles.
	failing bool
	// holdCreate, if set, is waited on after the request is recorded and before a stream
	// is returned, so a test can Cancel while Create has not come back.
	holdCreate <-chan struct{}

	// scripts are the responses handed out, so a test can settle one and see which were
	// abandoned. order remembers which came first.
	scripts map[string]*llmtest.Script
	order   []string
	// capabilities is what the model says it accepts, which is nothing unless a test says.
	capabilities llm.Capabilities
}

func newStubLLM() *stubLLM {
	return &stubLLM{answers: map[string]string{}, scripts: map[string]*llmtest.Script{}}
}

func (s *stubLLM) Start(context.Context) error { return nil }

func (s *stubLLM) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	s.mu.Lock()
	s.asked = append(s.asked, params)
	hold := s.holdCreate
	s.mu.Unlock()

	if hold != nil {
		select {
		case <-hold:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	script := llmtest.New(llm.StreamOptions{
		ResponseID: params.ID,
		Provider:   s.Provider(),
		Model:      s.Model(),
	})

	s.mu.Lock()
	s.scripts[params.ID] = script
	s.order = append(s.order, params.ID)
	answer, queued := s.answers[params.ID]
	if !queued && s.automatic != "" {
		answer, queued = s.automatic, true
	}
	failing := s.failing
	var calls []llm.ToolCall
	if len(s.calls) > 0 {
		calls, s.calls = s.calls[0], s.calls[1:]
	}
	s.mu.Unlock()

	switch {
	case failing:
		script.Fail(errModelDown, "stream")
	case len(calls) > 0:
		script.OutputText(answer)
		script.ToolCalls(calls...)
		script.Done()
	case queued:
		script.OutputText(answer)
		script.Done()
	}
	return script.Stream(), nil
}

// settle finishes a response, which is how a test controls when an answer lands.
func (s *stubLLM) settle(responseID, text string) {
	s.mu.Lock()
	script := s.scripts[responseID]
	s.mu.Unlock()

	script.OutputText(text)
	script.Done()
}

func (s *stubLLM) Close() error {
	s.mu.Lock()
	scripts := make([]*llmtest.Script, 0, len(s.scripts))
	for _, script := range s.scripts {
		scripts = append(scripts, script)
	}
	s.mu.Unlock()

	for _, script := range scripts {
		script.Done()
	}
	return nil
}

func (s *stubLLM) Provider() string               { return "stub" }
func (s *stubLLM) Model() string                  { return "stub-model" }
func (s *stubLLM) Capabilities() llm.Capabilities { return s.capabilities }

func (s *stubLLM) requests() []llm.ResponseParams {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]llm.ResponseParams(nil), s.asked...)
}

// interrupted is every response whose stream the code under test closed, oldest first.
func (s *stubLLM) interrupted() []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	var abandoned []string
	for _, id := range s.order {
		if s.scripts[id].Abandoned() {
			abandoned = append(abandoned, id)
		}
	}
	return abandoned
}

// stubConfig is one provider, which is all these tests need from routing.
func stubConfig() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{
			Provider:  "stub",
			Model:     "stub-model",
			Languages: []string{"en"},
			Realtime:  true,
			// ContextWindow is far past what any test sends unless it means to fill it.
			ContextWindow: 100_000,
		}},
		Aliases: map[string]routing.Alias{
			"en-low-latency": {Languages: []string{"en"}, RequireRealtime: true},
		},
	}
}

// testSkills are two skills with no deadline worth tripping over, so a test decides when
// work finishes rather than the clock.
func testSkills() Skills {
	return Skills{Skills: []Skill{
		{Name: "think", Description: "hard questions", Instructions: "think it through", Deadline: time.Minute},
		{Name: "recall", Description: "earlier in the call", Instructions: "read the transcript", Deadline: time.Minute},
	}}
}

type HarnessSuite struct {
	suite.Suite
	ctx context.Context

	fast *stubLLM
	slow *stubLLM
	// tools is what the next harness offers the fast model.
	tools Tools
	// box is where the next harness's subagent may run code. Nil is the usual case.
	box *stubSandbox
	// shelf, when set, is where files the subagent's code hands back are published.
	shelf *shelf
	// skills are what the next harness offers.
	skills Skills

	harness *Harness
	events  *collector
}

func TestHarnessSuite(t *testing.T) {
	suite.Run(t, new(HarnessSuite))
}

func (s *HarnessSuite) SetupTest() {
	s.ctx = context.Background()
	s.tools = Tools{}
	s.box = nil
	s.shelf = nil
	s.skills = testSkills()
}

// collector drains a harness's events for the life of one test, because the emitter
// applies backpressure on a reader that stops.
type collector struct {
	mu     sync.Mutex
	events []Event
	done   chan struct{}
}

func collect(h *Harness) *collector {
	drained := &collector{done: make(chan struct{})}
	go func() {
		defer close(drained.done)
		for event := range h.Events() {
			drained.mu.Lock()
			drained.events = append(drained.events, event)
			drained.mu.Unlock()
		}
	}()
	return drained
}

func (c *collector) seen() []Event {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]Event(nil), c.events...)
}

// session starts a routed session over a stub provider.
func (s *HarnessSuite) session(provider *stubLLM) *llmrouter.Session {
	return stubSession(&s.Suite, s.ctx, provider)
}

// stubSession starts a routed session over a stub provider.
func stubSession(s *suite.Suite, ctx context.Context, provider *stubLLM) *llmrouter.Session {
	logger := slog.New(slog.DiscardHandler)

	registry := llmrouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (llmrouter.Provider, error) { return provider, nil })
	router, err := llmrouter.New(llmrouter.Options{
		Config: stubConfig(), Registry: registry, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	session, err := router.Start(ctx, llmrouter.Request{CustomerID: "acme", Target: "en-low-latency"})
	s.Require().NoError(err)
	return session
}

// build returns a harness over two stub models, with or without a subagent. Whatever is in
// tools is offered, which is nothing unless a test set it first.
func (s *HarnessSuite) build(delegating bool) {
	s.fast = newStubLLM()
	options := Options{
		Model:  s.session(s.fast),
		Skills: s.skills,
		Tools:  s.tools,
		Logger: slog.New(slog.DiscardHandler),
	}
	if delegating {
		s.slow = newStubLLM()
		options.Subagent = s.session(s.slow)
	}
	if s.box != nil {
		options.Sandbox = s.box
	}
	if s.shelf != nil {
		options.Publish = s.shelf.publish
	}

	harness, err := New(options)
	s.Require().NoError(err)
	s.harness = harness

	s.events = collect(harness)
	s.T().Cleanup(func() { <-s.events.done })
	s.T().Cleanup(func() { _ = harness.Close() })
}

// respond asks the harness to answer a turn.
func (s *HarnessSuite) respond(turnID, text string) {
	s.answer(Turn{
		ID:           turnID,
		Instructions: "be brief",
		History:      []llm.Message{{Role: llm.User, Content: text}},
	})
}

// answer asks the harness to answer a turn and drains the reply on its own goroutine, as
// the agent does. A reply the test has not queued stays in flight, which is the point.
func (s *HarnessSuite) answer(turn Turn) {
	stream, err := s.harness.Respond(s.ctx, turn)
	s.Require().NoError(err)

	drained := make(chan struct{})
	go func() {
		defer close(drained)
		for stream.Next() {
		}
	}()
	s.T().Cleanup(func() {
		_ = stream.Close()
		<-drained
	})
}

// reply feeds a whole model reply through the filter and returns what would be spoken.
func (s *HarnessSuite) reply(turnID string, deltas ...string) string {
	var speech strings.Builder
	for _, delta := range deltas {
		speech.WriteString(s.harness.Filter(turnID, delta))
	}
	speech.WriteString(s.harness.Flush())
	return speech.String()
}

func (s *HarnessSuite) eventually(condition func() bool, message string) {
	s.Require().Eventually(condition, settleFor, 5*time.Millisecond, message)
}

// awaitSettled waits for the given number of tasks to have finished, since a task settles
// on the manager's own goroutine rather than on the one that abandoned it.
func (s *HarnessSuite) awaitSettled(count int) []Settled {
	s.eventually(func() bool { return len(settledIn(s.events.seen())) == count },
		"the tasks never settled")
	return settledIn(s.events.seen())
}

// awaitDelegated waits for the given number of handovers to have been reported, since
// the collector sees them on its own goroutine.
func (s *HarnessSuite) awaitDelegated(count int) []Delegated {
	s.eventually(func() bool { return len(delegatedIn(s.events.seen())) == count },
		"the work was never handed over")
	return delegatedIn(s.events.seen())
}

// awaitToolRequests waits for the given number of tool calls to have been reported, since
// the collector sees them on its own goroutine.
func (s *HarnessSuite) awaitToolRequests(count int) []ToolRequested {
	s.eventually(func() bool { return len(toolsRequestedIn(s.events.seen())) == count },
		"the tool calls were never reported")
	return toolsRequestedIn(s.events.seen())
}

func toolsRequestedIn(events []Event) []ToolRequested {
	var requested []ToolRequested
	for _, event := range events {
		if typed, ok := event.(ToolRequested); ok {
			requested = append(requested, typed)
		}
	}
	return requested
}

func settledIn(events []Event) []Settled {
	var settled []Settled
	for _, event := range events {
		if typed, ok := event.(Settled); ok {
			settled = append(settled, typed)
		}
	}
	return settled
}

func delegatedIn(events []Event) []Delegated {
	var delegated []Delegated
	for _, event := range events {
		if typed, ok := event.(Delegated); ok {
			delegated = append(delegated, typed)
		}
	}
	return delegated
}

func compactedIn(events []Event) []Compacted {
	var compacted []Compacted
	for _, event := range events {
		if typed, ok := event.(Compacted); ok {
			compacted = append(compacted, typed)
		}
	}
	return compacted
}

func longHistory() []llm.Message {
	history := make([]llm.Message, 0, compactionMinMessages)
	for index := range compactionMinMessages / 2 {
		history = append(history,
			llm.Message{Role: llm.User, Content: fmt.Sprintf("question %d", index)},
			llm.Message{Role: llm.Assistant, Content: fmt.Sprintf("answer %d", index)},
		)
	}
	return history
}

func (s *HarnessSuite) TestAModelIsRequired() {
	_, err := New(Options{})

	s.ErrorContains(err, "model session")
}

func (s *HarnessSuite) TestASkillWithoutADescriptionIsRefused() {
	// A skill the fast model is told nothing about is one it can never know to ask for.
	_, err := New(Options{
		Model:  &llmrouter.Session{},
		Skills: Skills{Skills: []Skill{{Name: "think", Instructions: "go on then"}}},
	})

	s.ErrorContains(err, "description")
}

func (s *HarnessSuite) TestTheModelIsToldWhatItMayHandOver() {
	s.build(true)

	s.respond("turn-1", "hello")

	s.Require().Len(s.fast.requests(), 1)
	instructions := s.fast.requests()[0].Instructions
	s.Contains(instructions, "be brief", "the agent's own instructions come first")
	s.Contains(instructions, "think: hard questions")
	s.Contains(instructions, "<ask skill=", "and how to ask for it")
}

func (s *HarnessSuite) TestAReplyToAToolResultThinksAndACallerTurnDoesNot() {
	s.build(false)
	s.fast.capabilities = llm.Capabilities{ReasoningEfforts: []string{"none", "low", "medium"}}

	s.respond("turn-1", "a table for four at 7:30")
	s.answer(Turn{
		ID:        "tool-1",
		History:   []llm.Message{{Role: llm.User, Content: "a table for four at 7:30"}},
		AfterTool: true,
	})

	s.Require().Len(s.fast.requests(), 2)
	s.Empty(s.fast.requests()[0].Reasoning.Effort, "the caller is waiting on every word")
	s.Equal("low", s.fast.requests()[1].Reasoning.Effort,
		"at none the model asks to book a free table instead of booking it")
}

func (s *HarnessSuite) TestToolsAreOfferedToTheFastModel() {
	s.tools = testTools()
	s.build(false)

	s.respond("turn-1", "put me through to someone")

	s.Require().Len(s.fast.requests(), 1)
	offered := s.fast.requests()[0].Tools
	s.Require().Len(offered, 2)
	s.Equal("transfer", offered[0].Name)
	s.Equal("hand the caller to a human", offered[0].Description)
	s.NotEmpty(offered[0].Parameters, "without a schema the model cannot fill the arguments in")
}

func (s *HarnessSuite) TestWithoutToolsTheRequestOffersNone() {
	// A model handed an empty toolbox still answers as though it had one, so a request
	// with nothing to offer must carry no tools rather than an empty list.
	s.build(false)

	s.respond("turn-1", "hello")

	s.Require().Len(s.fast.requests(), 1)
	s.Nil(s.fast.requests()[0].Tools)
}

func (s *HarnessSuite) TestTheModelIsToldHowToUseTheToolsItIsOffered() {
	s.tools = testTools()
	s.build(false)

	s.respond("turn-1", "put me through to someone")

	s.Require().Len(s.fast.requests(), 1)
	instructions := s.fast.requests()[0].Instructions
	s.Contains(instructions, "be brief", "the agent's own instructions come first")
	s.Contains(instructions, s.tools.Prompt())
	s.Contains(instructions, "same turn", "a tool is called once what it requires is known")
}

func (s *HarnessSuite) TestTheReplyToAToolResultIsToldHowToUseToolsToo() {
	// What keeps a hold phrase from being said again once the result is back is the same
	// instruction, so the reply that follows a result must carry it as the first reply did.
	s.tools = testTools()
	s.build(false)

	s.respond("turn-1", "a table for four at 7:30")
	s.answer(Turn{
		ID:           "tool-1",
		Instructions: "be brief",
		History:      []llm.Message{{Role: llm.User, Content: "a table for four at 7:30"}},
		AfterTool:    true,
	})

	s.Require().Len(s.fast.requests(), 2)
	s.Contains(s.fast.requests()[1].Instructions, "After a result, answer from it")
	s.Equal(s.fast.requests()[0].Instructions, s.fast.requests()[1].Instructions)
}

func (s *HarnessSuite) TestWithoutToolsTheModelIsToldNothingAboutThem() {
	s.build(true)

	s.respond("turn-1", "hello")

	s.Require().Len(s.fast.requests(), 1)
	s.NotContains(s.fast.requests()[0].Instructions, usePolicy)
	s.Equal("be brief\n\n"+s.skills.Prompt()+"\n\n"+spokenDelivery, s.fast.requests()[0].Instructions)
}

func (s *HarnessSuite) TestAPreviewIsToldWhatTheReplyItBecomesIsTold() {
	// A preview is only taken over by the reply when the two were asked the same thing, so
	// whatever the model is told must come out of one place.
	for _, test := range []struct {
		name       string
		tools      Tools
		delegating bool
	}{
		{"with tools", testTools(), false},
		{"with tools and a colleague", testTools(), true},
		{"without tools", Tools{}, true},
	} {
		s.Run(test.name, func() {
			s.SetupTest()
			s.tools = test.tools
			s.build(test.delegating)
			turn := Turn{
				ID:           "turn-1",
				Instructions: "be brief",
				History:      []llm.Message{{Role: llm.User, Content: "a table for two"}},
				Note:         "the caller was hard to hear",
			}

			stream, err := s.harness.Preview(s.ctx, turn)
			s.Require().NoError(err)
			s.T().Cleanup(func() { _ = stream.Close() })
			s.answer(turn)

			requests := s.fast.requests()
			s.Require().Len(requests, 2)
			s.Equal(requests[1].Instructions, requests[0].Instructions)
			s.Equal(requests[1].Tools, requests[0].Tools)
			s.Equal(test.tools.Prompt() != "", strings.Contains(requests[0].Instructions, usePolicy),
				"the model is told how to use its tools when it has some, and not otherwise")
		})
	}
}

func (s *HarnessSuite) TestAPreviewCannotHideWorkAlreadyRunning() {
	s.build(true)
	model := s.harness.PreviewModel()
	turn := Turn{
		ID:           "turn-2",
		Instructions: "be brief",
		History:      []llm.Message{{Role: llm.User, Content: "is it done yet?"}},
	}
	stream, err := s.harness.Preview(s.ctx, turn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = stream.Close() })

	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.eventually(func() bool { return s.harness.Delegating() }, "the task never started")

	s.False(s.harness.AdoptPreview(turn, model), "the preview did not know about the running work")
	s.Nil(s.harness.PreviewModel(), "running work prevents starting a preview")
	preview, err := s.harness.Preview(s.ctx, turn)
	s.Error(err)
	s.Nil(preview)
	s.answer(turn)
	requests := s.fast.requests()
	s.Contains(requests[len(requests)-1].Instructions, "still working on the think")
}

func (s *HarnessSuite) TestAToolCallIsReportedForSomebodyElseToRun() {
	// The harness cannot transfer a call it does not know exists, so what it does with a
	// tool call is say that one was asked for.
	s.tools = testTools()
	s.build(false)

	s.harness.Requested("turn-1", []llm.ToolCall{
		{ID: "call-1", Name: "transfer", Arguments: `{"to":"+15550001111"}`},
	})

	requested := s.awaitToolRequests(1)
	s.Equal("turn-1", requested[0].TurnID)
	s.Equal("transfer", requested[0].Call.Name)
	s.Equal(`{"to":"+15550001111"}`, requested[0].Call.Arguments)
}

func (s *HarnessSuite) TestAToolThatWasNeverOfferedIsDropped() {
	// Models invent tools, and whoever runs them should not have to know which names are
	// real before acting on one.
	s.tools = testTools()
	s.build(false)

	s.harness.Requested("turn-1", []llm.ToolCall{
		{ID: "call-1", Name: "hang_up", Arguments: "{}"},
		{ID: "call-2", Name: "press", Arguments: `{"digits":"1"}`},
	})

	requested := s.awaitToolRequests(1)
	s.Equal("press", requested[0].Call.Name, "only the real one survives")
}

func (s *HarnessSuite) TestATurnThatCalledAToolIsAnsweredWithTheResult() {
	// The provider matches each result against the call it answers, so the turn the model
	// took has to be replayed with the calls still on it.
	s.tools = testTools()
	s.build(false)

	s.answer(Turn{
		ID:           "turn-2",
		Instructions: "be brief",
		History: []llm.Message{
			{Role: llm.User, Content: "put me through"},
			{
				Role:      llm.Assistant,
				Content:   "One moment.",
				ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "transfer", Arguments: `{"to":"+1555"}`}},
			},
			{Role: llm.ToolResult, Content: "transferred", ToolCallID: "call-1"},
		},
	})

	s.Require().Len(s.fast.requests(), 1)
	sent := s.fast.requests()[0].Input
	s.Require().Len(sent, 3)
	s.Require().Len(sent[1].ToolCalls, 1)
	s.Equal("call-1", sent[1].ToolCalls[0].ID)
	s.Equal(llm.ToolResult, sent[2].Role)
	s.Equal("call-1", sent[2].ToolCallID)
}

func (s *HarnessSuite) TestWithoutASubagentNothingIsOffered() {
	// Skills mean nothing without someone to run them, so offering them would only invite
	// the model to write requests nobody answers.
	s.build(false)

	s.respond("turn-1", "hello")

	s.Require().Len(s.fast.requests(), 1)
	s.Equal("be brief", s.fast.requests()[0].Instructions)
}

func (s *HarnessSuite) TestARequestForHelpIsDelegatedAndNotSpoken() {
	s.build(true)
	s.respond("turn-1", "what is 15% of 84.20")

	spoken := s.reply("turn-1", `Let me check that. <ask skill="think">15% of 84.20</ask>`)

	s.Equal("Let me check that. ", spoken, "the caller hears the filler, never the request")

	s.eventually(func() bool { return len(s.slow.requests()) == 1 }, "the subagent was never asked")
	asked := s.slow.requests()[0]
	s.Equal("think it through", asked.Instructions, "the skill's own instructions")
	s.Require().Len(asked.Input, 2)
	s.Equal("what is 15% of 84.20", asked.Input[0].Content, "the conversation it was asked in")
	s.Equal("15% of 84.20", asked.Input[1].Content)

	s.Equal("think", s.awaitDelegated(1)[0].Skill)
}

func (s *HarnessSuite) TestASkillOfferedByNameIsReadAsItIsWhenItIsUsed() {
	// The fast model only ever sees the index, so a skill's instructions are read when the
	// subagent runs it, and an edit made mid-conversation is what the next use answers under.
	var mu sync.Mutex
	stored := "refund within 30 days"
	s.skills = Skills{
		Skills: []Skill{{Name: "refund", Description: "what a caller is owed", Deadline: time.Minute}},
		Load: func(_ context.Context, name string) (string, error) {
			mu.Lock()
			defer mu.Unlock()
			return name + ": " + stored, nil
		},
	}
	s.build(true)
	s.respond("turn-1", "can I get my money back")

	s.Require().NotContains(s.fast.requests()[0].Instructions, "30 days",
		"the fast model sees what a skill is for, never its instructions")
	s.reply("turn-1", `One moment. <ask skill="refund">bought 20 days ago</ask>`)
	s.eventually(func() bool { return len(s.slow.requests()) == 1 }, "the subagent was never asked")
	s.Equal("refund: refund within 30 days", s.slow.requests()[0].Instructions)

	mu.Lock()
	stored = "refund within 14 days"
	mu.Unlock()
	_, err := s.harness.Delegate("refund", "bought 20 days ago", "turn-2", nil, nil)
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.slow.requests()) == 2 }, "the subagent was never asked again")
	s.Equal("refund: refund within 14 days", s.slow.requests()[1].Instructions)
}

func (s *HarnessSuite) TestASkillThatCannotBeReadFailsRatherThanRunningWithoutInstructions() {
	s.skills = Skills{
		Skills: []Skill{{Name: "refund", Description: "what a caller is owed", Deadline: time.Minute}},
		Load: func(context.Context, string) (string, error) {
			return "", errors.New("the skill was deleted")
		},
	}
	s.build(true)
	s.respond("turn-1", "can I get my money back")

	_, err := s.harness.Delegate("refund", "bought 20 days ago", "turn-1", nil, nil)
	s.Require().NoError(err)

	settled := s.awaitSettled(1)[0]
	s.Equal(Failed, settled.State)
	s.Empty(s.slow.requests(), "nothing was asked without the skill's instructions")
}

func (s *HarnessSuite) TestTheModelAskingAgainDoesNotReplaceTheCallersImages() {
	s.build(true)
	s.respond("turn-1", "what is in this picture")
	picture := llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xFF, 0xD8, 0xFF}}
	_, err := s.harness.Delegate("think", "what is in this picture", "turn-1",
		[]llm.ContentPart{{Image: &picture}}, nil)
	s.Require().NoError(err)

	s.reply("turn-1", `Let me look. <ask skill="think">describe the picture</ask>`)

	s.eventually(func() bool { return len(s.slow.requests()) == 1 }, "the subagent was never asked")
	s.True(s.slow.requests()[0].HasImage(), "the task that ran is the one with the picture")
	// The task's request can reach the subagent before its event reaches the sink.
	s.eventually(func() bool { return len(delegatedIn(s.events.seen())) == 1 }, "the picture's task was never reported")
	for _, settled := range settledIn(s.events.seen()) {
		s.NotEqual(ReasonSuperseded, settled.Result.Reason, "the picture's task was replaced")
	}
	s.Len(delegatedIn(s.events.seen()), 1)
}

func (s *HarnessSuite) TestATurnsImagesAreHandedToTheModelBesideTheWordsTheyCameWith() {
	s.build(true)
	picture := llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xFF, 0xD8, 0xFF}, Caption: "the error dialog"}

	s.answer(Turn{
		ID:           "turn-1",
		Instructions: "be brief",
		History:      []llm.Message{{Role: llm.User, Content: "what does this say"}},
		Images:       []llm.ImagePart{picture},
	})

	asked := s.fast.requests()
	s.Require().Len(asked, 1)
	s.Equal([]llm.Message{{Role: llm.User, Parts: []llm.ContentPart{
		{Text: "what does this say"}, {Text: "the error dialog"}, {Image: &picture},
	}}}, asked[0].Input)
}

func (s *HarnessSuite) TestAColleagueIsHandedTheConversationWithoutAReplysImages() {
	s.build(true)
	s.answer(Turn{
		ID:           "turn-1",
		Instructions: "be brief",
		History:      []llm.Message{{Role: llm.User, Content: "what does this say"}},
		Images:       []llm.ImagePart{{MIME: "image/jpeg", Data: []byte{0xFF, 0xD8, 0xFF}}},
	})

	_, err := s.harness.Delegate("think", "work out what the caller should do", "turn-1", nil, nil)
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.slow.requests()) == 1 }, "the subagent was never asked")
	s.False(s.slow.requests()[0].HasImage(), "a subagent that may not see was handed the picture")
}

func (s *HarnessSuite) TestASkillIsOfferedOnlyWithASubagentToRunIt() {
	s.build(false)
	s.False(s.harness.Offers("think"), "nothing would run it")

	s.build(true)
	s.True(s.harness.Offers("think"))
	s.False(s.harness.Offers("vision"), "the harness was given no such skill")
}

func (s *HarnessSuite) TestCompleteIdentifiersAreNotHandedToAColleague() {
	s.build(true)
	s.respond("turn-1", "Maya Chen, date of birth March 4 1987, member ID ABC123456")

	spoken := s.reply("turn-1", `A B C 1 2 3 4 5 6. <ask skill="think">read that back</ask>`)

	s.Equal("A B C 1 2 3 4 5 6. ", spoken)
	s.True(s.harness.Pending(), "the fast model still owes the caller a tool call")
	s.Empty(s.slow.requests(), "a complete identifier must not wait on the subagent")
}

func (s *HarnessSuite) TestIdentifiersSaidInWordsAreNotHandedToAColleagueEither() {
	for _, said := range []string{
		"a table for two at seven thirty",
		"my number is five one two five five five zero one four two",
	} {
		s.Run(said, func() {
			s.SetupTest()
			s.build(true)
			s.respond("turn-1", said)

			spoken := s.reply("turn-1", `Booking it. <ask skill="think">read that back</ask>`)

			s.Equal("Booking it. ", spoken)
			s.True(s.harness.Pending(), "the fast model still owes the caller a tool call")
			s.Empty(s.slow.requests(), "a time or a number said in words is complete as well")
		})
	}
}

func (s *HarnessSuite) TestATurnNobodyPromptedHasSomethingToAnswer() {
	// Work coming back is a turn nobody asked for, so the conversation ends with the
	// agent's own reply rather than a caller's sentence. Asked to follow its own turn,
	// Gemini refuses the request outright -- "requests ending with a model turn are not
	// supported" -- and the caller never hears what came back.
	s.build(true)

	s.answer(Turn{
		ID:           "turn-1",
		Instructions: "be brief",
		History: []llm.Message{
			{Role: llm.User, Content: "travel advice for Boulder?"},
			{Role: llm.Assistant, Content: "Let me look into that."},
		},
	})

	asked := s.fast.requests()[0].Input
	s.Equal(llm.User, asked[len(asked)-1].Role, "the model was asked to follow its own turn")
	s.Len(s.harness.history, 2, "what was added is the request's, not the conversation's")
}

func (s *HarnessSuite) TestAnAnsweredTurnIsSentAsItStands() {
	s.build(true)

	s.respond("turn-1", "what is 15% of 84.20")

	asked := s.fast.requests()[0].Input
	s.Require().Len(asked, 1, "there was already something to answer")
	s.Equal("what is 15% of 84.20", asked[0].Content)
}

func (s *HarnessSuite) TestDelegatingDoesNotWaitForTheAnswer() {
	// This is the whole point: the fast model keeps talking while the slow one works.
	s.build(true)
	s.respond("turn-1", "what is 15% of 84.20")

	spoken := s.reply("turn-1", `One moment. <ask skill="think">15% of 84.20</ask> Nearly there.`)

	s.Equal("One moment.  Nearly there.", spoken)
	s.eventually(func() bool { return s.harness.Delegating() }, "the task never started")
	s.Empty(settledIn(s.events.seen()), "nothing waited for it to finish")
}

func (s *HarnessSuite) TestAnAnswerIsFoldedIntoTheNextThingTheModelIsAsked() {
	s.build(true)
	s.slow.automatic = "It is 12.63."
	s.respond("turn-1", "what is 15% of 84.20")

	s.reply("turn-1", `Let me check. <ask skill="think">15% of 84.20</ask>`)
	s.awaitSettled(1)

	s.respond("turn-2", "")

	s.Require().Len(s.fast.requests(), 2)
	s.Contains(s.fast.requests()[1].Instructions, "It is 12.63.")
	s.Contains(s.fast.requests()[1].Instructions, "Tell the caller")
}

func (s *HarnessSuite) TestAnAnswerIsOnlyToldOnce() {
	s.build(true)
	s.slow.automatic = "It is 12.63."
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.awaitSettled(1)

	s.respond("turn-2", "")
	s.harness.Remember(llm.Response{ID: "turn-2", Status: llm.StatusCompleted})
	s.respond("turn-3", "and what about tax")

	s.Require().Len(s.fast.requests(), 3)
	s.NotContains(s.fast.requests()[2].Instructions, "12.63",
		"an answer already spoken is not repeated on every later turn")
	s.False(s.harness.Pending())
}

func (s *HarnessSuite) TestAnAnswerIsSummarisedForSomebodyListening() {
	s.build(true)
	s.slow.automatic = "It is 12.63."
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.awaitSettled(1)

	s.respond("turn-2", "")

	s.Contains(s.fast.requests()[1].Instructions, spokenDelivery)
	s.NotContains(s.fast.requests()[1].Instructions, writtenDelivery)
}

func (s *HarnessSuite) TestAnAnswerIsGivenInFullToSomebodyReading() {
	s.build(true)
	s.harness.options.Text = true
	s.slow.automatic = "It is 12.63."
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.awaitSettled(1)

	s.respond("turn-2", "")

	s.Contains(s.fast.requests()[1].Instructions, writtenDelivery)
	s.NotContains(s.fast.requests()[1].Instructions, spokenDelivery)
}

func (s *HarnessSuite) TestAnAgentWithNothingToHandOverIsNotToldHowToHandItOver() {
	s.build(false)

	s.respond("turn-1", "hello")

	s.NotContains(s.fast.requests()[0].Instructions, spokenDelivery)
}

func (s *HarnessSuite) TestTheModelIsToldWhatItsColleagueIsStillWorkingOn() {
	s.build(true)
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.eventually(func() bool { return s.harness.Delegating() }, "the task never started")

	s.respond("turn-2", "is it done yet?")

	s.Contains(s.fast.requests()[1].Instructions, "still working on the think")
	s.Contains(s.fast.requests()[1].Instructions, `<drop skill="think"/>`,
		"the model is given the tag to write, not a placeholder to fill in")
}

func (s *HarnessSuite) TestDroppingAToolIsPassedOnToWhoeverRunsIt() {
	s.tools = testTools()
	s.build(false)

	s.harness.Filter("turn-1", `<drop skill="press"/>`)

	s.eventually(func() bool {
		return slices.ContainsFunc(s.events.seen(), func(event Event) bool { return event == ToolDropped{Name: "press"} })
	}, "the tool was never dropped")
}

func (s *HarnessSuite) TestAnAnswerCutOffBeforeItWasHeardIsStillOwed() {
	s.build(true)
	s.slow.automatic = "It is 12.63."
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.awaitSettled(1)

	s.respond("turn-2", "")
	s.False(s.harness.Pending(), "the reply carrying it is on its way")
	s.harness.Release("turn-2")
	s.True(s.harness.Pending(), "the caller talked over it, so they never heard it")

	s.respond("turn-3", "sorry, go on")
	s.Contains(s.fast.requests()[2].Instructions, "12.63")
}

func (s *HarnessSuite) TestAnAnswerWhoseReplyFailedIsStillOwed() {
	s.build(true)
	s.slow.automatic = "It is 12.63."
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.awaitSettled(1)

	s.respond("turn-2", "")
	s.harness.Remember(llm.Response{ID: "turn-2", Status: llm.StatusFailed})

	s.True(s.harness.Pending())
	s.respond("turn-3", "")
	s.Contains(s.fast.requests()[2].Instructions, "12.63")
}

func (s *HarnessSuite) TestAnAnswerInAReplyThatNeverFinishedGoesWithTheNextOne() {
	// A reply started before the ruling and then dropped never finishes, and nothing says
	// so. The reply asked for after it is the one the caller hears.
	s.build(true)
	s.slow.automatic = "It is 12.63."
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.awaitSettled(1)

	s.respond("turn-2", "")
	s.respond("turn-2", "")

	s.Contains(s.fast.requests()[2].Instructions, "12.63")
	s.harness.Remember(llm.Response{ID: "turn-2", Status: llm.StatusCompleted})
	s.False(s.harness.Pending())
}

func (s *HarnessSuite) TestASettledTaskReportsWhatIsOwedToTheCaller() {
	s.build(true)
	s.slow.automatic = "It is 12.63."
	s.respond("turn-1", "what is 15% of 84.20")

	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Equal(Done, settled.State)
	s.Equal("It is 12.63.", settled.Text)
	s.True(settled.Actionable(), "the caller is owed the answer they were told was coming")
	s.Positive(settled.ElapsedMs)
}

func (s *HarnessSuite) TestASubagentThatNeedsMoreAsksThroughTheAgent() {
	// A subagent that guesses at a missing detail is worse than one that has the agent
	// ask, so it says what it needs and the agent puts it in the caller's language.
	s.build(true)
	s.slow.automatic = "NEED: which date did you want?"
	s.respond("turn-1", "is there a table free")

	s.reply("turn-1", `Let me look. <ask skill="think">table availability</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Equal("which date did you want?", settled.Question)
	s.Empty(settled.Text, "a question is not an answer")
	s.True(settled.Actionable())

	s.respond("turn-2", "")
	s.Contains(s.fast.requests()[1].Instructions, "which date did you want?")
	s.Contains(s.fast.requests()[1].Instructions, "Ask them")
}

func (s *HarnessSuite) TestTheReplyCarryingAnAnswerCannotHandTheWorkBack() {
	// Handed the colleague's question, the model hands it straight back rather than
	// asking the caller — often word for word. The colleague asks again, that answer
	// earns another turn, and the two talk to each other while the caller listens.
	s.build(true)
	s.slow.automatic = "NEED: which date did you want?"
	s.respond("turn-1", "is there a table free")
	s.reply("turn-1", `Let me look. <ask skill="think">table availability</ask>`)
	s.awaitSettled(1)

	s.respond("turn-2", "")
	spoken := s.reply("turn-2",
		`<ask skill="think">which date did you want?</ask>Which date did you want?`)

	s.Equal("Which date did you want?", spoken, "the caller is asked instead")
	s.Require().Never(func() bool { return len(s.slow.requests()) > 1 },
		200*time.Millisecond, 10*time.Millisecond,
		"the colleague was asked its own question")
}

func (s *HarnessSuite) TestAnswersToOneSkillDoNotBlockHandingOverAnother() {
	// Only the work the reply was written to report is refused. A colleague coming back
	// is often exactly when the next piece of work becomes worth doing.
	s.build(true)
	s.slow.automatic = "the table is free at eight"
	s.respond("turn-1", "is there a table free")
	s.reply("turn-1", `<ask skill="think">table availability</ask>`)
	s.awaitSettled(1)

	s.respond("turn-2", "")
	s.reply("turn-2", `Eight works. <ask skill="recall">what name did they book under</ask>`)

	handed := s.awaitDelegated(2)
	s.Equal("recall", handed[1].Skill)
}

func (s *HarnessSuite) TestTheReplyCarryingAColleaguesQuestionIsOfferedNoTools() {
	// A colleague asks for what only the caller can say, so there is nothing to look up.
	// Left holding a tool the model reaches for one and narrates the reaching, and the
	// caller hears a second promise to check without ever being asked the question.
	s.tools = testTools()
	s.build(true)
	s.slow.automatic = "NEED: which date did you want?"
	s.respond("turn-1", "is there a table free")
	s.reply("turn-1", `Let me look. <ask skill="think">table availability</ask>`)
	s.awaitSettled(1)

	s.respond("turn-2", "")

	s.Require().Len(s.fast.requests(), 2)
	s.NotEmpty(s.fast.requests()[0].Tools)
	s.Empty(s.fast.requests()[1].Tools, "there is nobody to ask but the caller")
	s.Contains(s.fast.requests()[0].Instructions, usePolicy)
	s.NotContains(s.fast.requests()[1].Instructions, usePolicy, "nor how to use what it does not have")
}

func (s *HarnessSuite) TestATurnMadeToAnswerIsOfferedNoToolsAndToldWhy() {
	s.tools = testTools()
	s.build(true)

	s.answer(Turn{
		ID:           "tool-9",
		Instructions: "be brief",
		History:      []llm.Message{{Role: llm.User, Content: "where is my order"}},
		AfterTool:    true,
		Answers:      true,
	})

	s.Require().Len(s.fast.requests(), 1)
	s.Empty(s.fast.requests()[0].Tools, "a tool offered is a tool reached for")
	s.Contains(s.fast.requests()[0].Instructions, answerNow)
}

func (s *HarnessSuite) TestTheReplyCarryingAColleaguesAnswerKeepsItsTools() {
	// Only a question leaves nothing to look up. An answer coming back is often exactly
	// when acting on it becomes possible.
	s.tools = testTools()
	s.build(true)
	s.slow.automatic = "the table is free at eight"
	s.respond("turn-1", "is there a table free")
	s.reply("turn-1", `<ask skill="think">table availability</ask>`)
	s.awaitSettled(1)

	s.respond("turn-2", "")

	s.Require().Len(s.fast.requests(), 2)
	s.NotEmpty(s.fast.requests()[1].Tools)
	s.Contains(s.fast.requests()[1].Instructions, usePolicy)
}

func (s *HarnessSuite) TestANewerRequestSupersedesTheOneItReplaces() {
	// The caller has said something since, so the older question was asked about a
	// conversation that no longer exists.
	s.build(true)
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.eventually(func() bool { return len(s.slow.requests()) == 1 }, "the first task never started")
	first := s.slow.requests()[0].ID

	s.respond("turn-2", "actually make it 20%")
	s.reply("turn-2", `<ask skill="think">20% of 84.20</ask>`)

	s.eventually(func() bool { return len(s.slow.requests()) == 2 }, "the second task never started")
	s.Contains(s.slow.interrupted(), first, "the subagent was never told to stop the first")
	s.True(s.harness.Delegating(), "the second task should still be running")

	settled := s.awaitSettled(1)
	s.Equal(Cancelled, settled[0].State)
	s.Equal(ReasonSuperseded, settled[0].Reason)
	s.False(settled[0].Actionable(), "nobody was still waiting on the question that changed")
}

func (s *HarnessSuite) TestARevisedConversationCancelsWorkFromItsOldTurn() {
	s.build(true)
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.eventually(func() bool { return s.harness.Delegating() }, "the task never started")

	s.harness.CancelTurn("turn-1", ReasonSuperseded)

	settled := s.awaitSettled(1)[0]
	s.Equal(Cancelled, settled.State)
	s.Equal(ReasonSuperseded, settled.Reason)
	s.False(settled.Actionable())
}

func (s *HarnessSuite) TestTheModelCanDropWorkTheCallerHasMovedPast() {
	s.build(true)
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.eventually(func() bool { return s.harness.Delegating() }, "the task never started")

	s.respond("turn-2", "never mind")
	spoken := s.reply("turn-2", `Sure, forget it. <drop skill="think"/>`)

	s.Equal("Sure, forget it. ", spoken)
	s.Equal(ReasonDropped, s.awaitSettled(1)[0].Reason)
	s.False(s.harness.Delegating())
}

func (s *HarnessSuite) TestADroppedAnswerIsNeverToldToTheCaller() {
	s.build(true)
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.eventually(func() bool { return s.harness.Delegating() }, "the task never started")

	s.reply("turn-1", `<drop skill="think"/>`)
	s.awaitSettled(1)
	s.respond("turn-2", "something else")

	s.Require().Len(s.fast.requests(), 2)
	s.NotContains(s.fast.requests()[1].Instructions, "has come back",
		"work the caller has moved past leaves nothing to say")
}

func (s *HarnessSuite) TestWorkThatOutlivesItsDeadlineIsAbandoned() {
	s.build(true)
	s.harness.options.Skills = Skills{Skills: []Skill{
		{Name: "think", Description: "hard questions", Instructions: "go on", Deadline: 20 * time.Millisecond},
	}}
	s.respond("turn-1", "what is 15% of 84.20")

	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)

	s.Equal(ReasonDeadline, s.awaitSettled(1)[0].Reason)
	s.False(s.harness.Delegating())
}

func (s *HarnessSuite) TestWorkThatRanOutOfTimeStillOwesTheCallerAWord() {
	// The caller asked and is still waiting: nothing replaced the work and they never moved
	// on from it. Going quiet leaves them holding a question the agent has given up on,
	// which is a call that answers "anything else?" to a question never answered.
	s.build(true)
	s.harness.options.Skills = Skills{Skills: []Skill{
		{Name: "think", Description: "hard questions", Instructions: "go on", Deadline: 20 * time.Millisecond},
	}}
	s.respond("turn-1", "how is traffic on I-70")

	s.reply("turn-1", `<ask skill="think">traffic on I-70</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Require().Equal(ReasonDeadline, settled.Reason)
	s.True(settled.Actionable(), "the caller is owed the news that the answer is not coming")
	s.True(s.harness.Pending(), "and the next turn has to say so")

	s.respond("turn-2", "")
	s.Require().Len(s.fast.requests(), 2)
	s.Contains(s.fast.requests()[1].Instructions, "did not come back")
}

func (s *HarnessSuite) TestWorkTheCallerMovedPastOwesThemNothing() {
	// The other cancellations are the premise being gone, and nobody is waiting on those.
	s.False(Result{State: Cancelled, Reason: ReasonSuperseded}.Actionable())
	s.False(Result{State: Cancelled, Reason: ReasonDropped}.Actionable())
	s.False(Result{State: Cancelled, Reason: ReasonClosed}.Actionable())
}

func (s *HarnessSuite) TestOnlyAsMuchWorkAsWasAllowedRunsAtOnce() {
	s.build(true)
	s.harness.options.Tasks = 1
	s.harness.tasks.limit = 1
	s.respond("turn-1", "two things")

	s.reply("turn-1", `<ask skill="think">the first</ask><ask skill="recall">the second</ask>`)

	s.eventually(func() bool { return len(s.slow.requests()) == 1 }, "the first task never started")
	s.Len(s.awaitDelegated(1), 1, "the second was refused rather than queued")
	s.Equal(1, s.harness.tasks.Running())
}

func (s *HarnessSuite) TestASkillTheModelInventedIsIgnored() {
	s.build(true)
	s.respond("turn-1", "hello")

	spoken := s.reply("turn-1", `Sure. <ask skill="teleport">do the thing</ask>`)

	s.Equal("Sure. ", spoken, "an invented request is still not spoken")
	s.Empty(s.slow.requests(), "and is not sent anywhere")
	s.Empty(delegatedIn(s.events.seen()))
	s.True(s.harness.Pending(), "the caller was told an answer was coming")

	s.respond("turn-2", "")
	s.Contains(s.fast.requests()[1].Instructions, "no skill named teleport")
}

func (s *HarnessSuite) TestASkillThatNamesAToolIsAskedAsOne() {
	// Voice models write skill tags for tools they were offered, then wait for a
	// colleague who is not coming. The body of the tag is the argument the tool needs.
	s.tools = testTools()
	s.build(true)
	s.respond("turn-1", "press 1 for sales")

	spoken := s.reply("turn-1", `One moment. <ask skill="press">1</ask>`)

	s.Equal("One moment. ", spoken)
	s.Empty(s.slow.requests(), "a tool is not a colleague")
	s.Empty(delegatedIn(s.events.seen()))

	asked := s.harness.TakeAsked()
	s.Require().Len(asked, 1)
	s.Equal("press", asked[0].Name)
	s.JSONEq(`{"digits":"1"}`, asked[0].Arguments)
	s.NotEmpty(asked[0].ID)

	s.harness.Requested("turn-1", asked)
	requested := s.awaitToolRequests(1)
	s.Equal("press", requested[0].Call.Name)
	s.JSONEq(`{"digits":"1"}`, requested[0].Call.Arguments)
}

func (s *HarnessSuite) TestAnAbandonedReplyDoesNotKeepTheToolsItAskedFor() {
	s.tools = testTools()
	s.build(false)

	s.harness.Filter("turn-1", `<ask skill="press">1</ask>`)
	s.harness.Reset()

	s.Empty(s.harness.TakeAsked())
}

func (s *HarnessSuite) TestWorkThatFailsStillTellsTheCallerSomething() {
	// The caller was told an answer was coming, so silence is the one thing that is not
	// an option.
	s.build(true)
	s.slow.failing = true
	s.respond("turn-1", "what is 15% of 84.20")

	s.reply("turn-1", `Let me check. <ask skill="think">15% of 84.20</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Equal(Failed, settled.State)
	s.ErrorIs(settled.Err, errModelDown)
	s.True(settled.Actionable())

	s.respond("turn-2", "")
	s.Contains(s.fast.requests()[1].Instructions, "think you asked for failed")
}

func (s *HarnessSuite) TestWorkThatComesBackEmptyIsReportedAsFailed() {
	// An empty answer is not actionable as an answer, so it used to settle in silence:
	// the caller was promised something and heard nothing.
	s.build(true)
	s.slow.automatic = " "
	s.respond("turn-1", "render the clip")

	s.reply("turn-1", `<ask skill="think">render the clip</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Equal(Failed, settled.State)
	s.True(settled.Actionable())
	s.respond("turn-2", "")
	s.Contains(s.fast.requests()[1].Instructions, "think you asked for failed")
}

func (s *HarnessSuite) TestAColdLargePrefixIsCompactedPrivately() {
	s.build(true)
	s.slow.automatic = "The caller is planning dinner for Friday."
	history := longHistory()

	started, err := s.harness.MaybeCompact(history, compactionMinTokens, 0)
	s.Require().NoError(err)
	s.True(started)

	s.eventually(func() bool { return len(compactedIn(s.events.seen())) == 1 },
		"the conversation was never compacted")
	compacted := compactedIn(s.events.seen())[0]
	s.Equal(history[:len(history)-compactionKeepRecent], compacted.Prefix)
	s.Equal("The caller is planning dinner for Friday.", compacted.Summary)
	s.Empty(settledIn(s.events.seen()), "private maintenance is not a caller-facing task")
	s.False(s.harness.Delegating(), "the caller is not waiting for private maintenance")
}

func (s *HarnessSuite) TestCompactionKeepsWholeTurns() {
	s.build(true)
	s.slow.automatic = "The caller asked about their calendar."
	history := append(longHistory(),
		llm.Message{Role: llm.User, Content: "ok check it"},
		llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "calendar__list_tools"}}},
		llm.Message{Role: llm.ToolResult, ToolCallID: "c1", Content: "{}"},
		llm.Message{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{ID: "c2", Name: "calendar__call_tool"}}},
		llm.Message{Role: llm.ToolResult, ToolCallID: "c2", Content: "{}"},
		llm.Message{Role: llm.Assistant, Content: "Nothing today."},
		llm.Message{Role: llm.User, Content: "and linear?"},
	)

	started, err := s.harness.MaybeCompact(history, compactionMinTokens, 0)
	s.Require().NoError(err)
	s.True(started)

	s.eventually(func() bool { return len(compactedIn(s.events.seen())) == 1 },
		"the conversation was never compacted")
	prefix := compactedIn(s.events.seen())[0].Prefix
	s.Equal(llm.User, history[len(prefix)].Role,
		"the history kept opens on the caller, not on a tool call cut from its turn")
	s.Equal("ok check it", history[len(prefix)].Content)
}

func (s *HarnessSuite) TestAnEffectivePrefixCacheKeepsVerbatimHistory() {
	s.build(true)

	started, err := s.harness.MaybeCompact(
		longHistory(),
		compactionMinTokens,
		int64(float64(compactionMinTokens)*compactionCacheRatio),
	)
	s.Require().NoError(err)
	s.False(started)

	s.Empty(s.slow.requests(), "cached history is cheaper and more faithful than a summary")
}

func (s *HarnessSuite) TestAPromptNearingTheContextWindowIsCompactedEvenWhenCached() {
	// A cache that pays does not make room, so once the prompt has used most of the window
	// the history is summarised before the next turn is refused for being too long.
	s.build(true)

	started, err := s.harness.MaybeCompact(longHistory(), 79_999, 79_999)
	s.Require().NoError(err)
	s.False(started, "under 80% of the window, a cached history stays verbatim")

	started, err = s.harness.MaybeCompact(longHistory(), 80_000, 80_000)
	s.Require().NoError(err)
	s.True(started)
	s.eventually(func() bool { return len(s.slow.requests()) == 1 }, "the summary was never asked for")
}

func (s *HarnessSuite) TestShortHistoryIsNotCompacted() {
	s.build(true)

	started, err := s.harness.MaybeCompact(
		longHistory()[:compactionMinMessages-1],
		compactionMinTokens,
		0,
	)
	s.Require().NoError(err)
	s.False(started)

	s.Empty(s.slow.requests())
}

func (s *HarnessSuite) TestClosingAbandonsWorkNobodyWillHear() {
	s.build(true)
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.eventually(func() bool { return s.harness.Delegating() }, "the task never started")

	s.Require().NoError(s.harness.Close())
	// Closing ends the event stream, so draining it is what makes what was reported on
	// the way out settled fact rather than a race.
	<-s.events.done

	s.Equal(0, s.harness.tasks.Running())
	settled := settledIn(s.events.seen())
	s.Require().Len(settled, 1)
	s.Equal(ReasonClosed, settled[0].Reason)
}

func (s *HarnessSuite) TestClosingTwiceIsSafe() {
	s.build(true)

	s.NoError(s.harness.Close())
	s.NoError(s.harness.Close())
}

func (s *HarnessSuite) TestClosingEndsTheEventStream() {
	s.build(true)

	s.Require().NoError(s.harness.Close())

	select {
	case _, open := <-s.harness.Events():
		s.False(open, "the channel closes so a consumer's range loop ends")
	case <-time.After(settleFor):
		s.Fail("the event channel stayed open")
	}
}

func (s *HarnessSuite) TestResettingForgetsAnInterruptedReply() {
	s.build(true)
	s.respond("turn-1", "what is 15% of 84.20")
	s.harness.Filter("turn-1", `Let me check. <ask skill="think">15% of`)

	s.harness.Reset()

	s.Empty(s.slow.requests(), "an abandoned reply never finished asking")
	s.Equal("Hello.", s.reply("turn-2", "Hello."), "and does not leak into the next turn")
}

func (s *HarnessSuite) TestOpeningTheSubagentDoesNotBlockDelegationOrSpeech() {
	fast := newStubLLM()
	fast.automatic = "I can still talk."
	h, err := New(Options{
		Model:        s.session(fast),
		OpenSubagent: func(ctx context.Context) (*llmrouter.Session, error) { <-ctx.Done(); return nil, ctx.Err() },
		Skills:       Skills{Skills: []Skill{{Name: "vision", Description: "inspect", Instructions: "describe", Deadline: time.Minute}}},
	})
	s.Require().NoError(err)
	events := collect(h)
	defer func() { s.NoError(h.Close()); <-events.done }()
	accepted := make(chan error, 1)
	go func() { _, err := h.Delegate("vision", "look", "turn-1", nil, nil); accepted <- err }()
	select {
	case err := <-accepted:
		s.Require().NoError(err)
	case <-time.After(time.Second):
		s.T().Fatal("subagent startup blocked acceptance")
	}
	reply, err := h.Respond(s.ctx, Turn{ID: "turn-2", History: []llm.Message{{Role: llm.User, Content: "hello"}}})
	s.Require().NoError(err)
	for reply.Next() {
	}
	s.Equal("I can still talk.", reply.Response().OutputText)
	s.True(h.Delegating())
}

func (s *HarnessSuite) TestSlowProviderHeadersDoNotBlockTaskAcceptance() {
	provider := newStubLLM()
	hold := make(chan struct{})
	provider.holdCreate = hold
	m := newManager(s.session(provider), 2, nil, llmoptions.LLM{}, slog.New(slog.DiscardHandler))
	defer m.Close()
	accepted := make(chan string, 1)
	go func() { id, _ := m.Create(testSkills().Skills[0], "question", nil, "turn", false); accepted <- id }()
	var id string
	select {
	case id = <-accepted:
		s.NotEmpty(id)
	case <-time.After(time.Second):
		s.T().Fatal("provider headers blocked task acceptance")
	}
	m.Cancel(id, ReasonDropped)
	select {
	case result := <-m.Results():
		s.Equal(Cancelled, result.State)
		s.Equal(ReasonDropped, result.Reason)
	case <-time.After(time.Second):
		s.T().Fatal("cancellation did not stop provider startup")
	}
}
