package harness

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// settleFor is how long a test waits for an expectation to become true. The flow crosses
// several goroutines, so the alternative to waiting is asserting on a race.
const settleFor = 3 * time.Second

var errModelDown = errors.New("the model is down")

// stubLLM answers whatever it is asked with whatever the test has queued, and only when
// the test says so, which is what lets a task be caught mid-flight.
type stubLLM struct {
	emitter *llm.Emitter

	mu    sync.Mutex
	asked []llm.Request
	// abandoned is every completion id an interrupt named.
	abandoned []string
	// answers maps a completion id to what comes back. A request with no answer stays in
	// flight until the test settles it.
	answers map[string]string
	// automatic answers every request with this text, for tests that do not care which
	// completion is which.
	automatic string
	// calls are tool calls to ask for, one queue entry per request, which is what lets a
	// test answer with a tool once and with words the next time it is asked.
	calls [][]llm.ToolCall
	// failing makes every request report a provider failure before it settles.
	failing bool
}

func newStubLLM() *stubLLM {
	return &stubLLM{emitter: llm.NewEmitter(64), answers: map[string]string{}}
}

func (s *stubLLM) Start(context.Context) error { return nil }

func (s *stubLLM) Respond(request llm.Request) error {
	s.mu.Lock()
	s.asked = append(s.asked, request)
	answer, queued := s.answers[request.ID]
	if !queued && s.automatic != "" {
		answer, queued = s.automatic, true
	}
	failing := s.failing
	var calls []llm.ToolCall
	if len(s.calls) > 0 {
		calls, s.calls = s.calls[0], s.calls[1:]
	}
	s.mu.Unlock()

	s.emitter.Send(llm.CompletionStarted{CompletionID: request.ID, At: time.Now()})
	if failing {
		s.emitter.Send(llm.Error{CompletionID: request.ID, Err: errModelDown, Context: "stream"})
		s.emitter.Send(llm.CompletionComplete{CompletionID: request.ID})
		return nil
	}
	if len(calls) > 0 {
		s.emitter.Send(llm.CompletionComplete{
			CompletionID: request.ID,
			Text:         answer,
			ToolCalls:    calls,
		})
		return nil
	}
	if queued {
		s.settle(request.ID, answer, false)
	}
	return nil
}

// settle finishes a completion, which is how a test controls when an answer lands.
func (s *stubLLM) settle(completionID, text string, interrupted bool) {
	s.emitter.Send(llm.CompletionComplete{
		CompletionID: completionID,
		Text:         text,
		Interrupted:  interrupted,
	})
}

func (s *stubLLM) Interrupt(completionIDs ...string) error {
	s.mu.Lock()
	s.abandoned = append(s.abandoned, completionIDs...)
	s.mu.Unlock()

	// A real provider settles an abandoned completion rather than dropping it, because
	// the tokens it already generated were still billed.
	for _, id := range completionIDs {
		s.settle(id, "", true)
	}
	return nil
}

func (s *stubLLM) Events() <-chan llm.Event { return s.emitter.Events() }

func (s *stubLLM) Close() error {
	s.emitter.Close()
	return nil
}

func (s *stubLLM) Provider() string { return "stub" }
func (s *stubLLM) Model() string    { return "stub-model" }
func (s *stubLLM) Reasoning() bool  { return false }

func (s *stubLLM) requests() []llm.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]llm.Request(nil), s.asked...)
}

func (s *stubLLM) interrupted() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.abandoned...)
}

// stubConfig is one provider, which is all these tests need from routing.
func stubConfig() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{
			Provider:  "stub",
			Model:     "stub-model",
			Languages: []string{"en"},
			Realtime:  true,
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
	registry.Register("stub", func(routing.Spec) (llm.LLM, error) { return provider, nil })
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
		Skills: testSkills(),
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

	harness, err := New(options)
	s.Require().NoError(err)
	s.harness = harness

	s.events = collect(harness)
	s.T().Cleanup(func() { <-s.events.done })
	s.T().Cleanup(func() { _ = harness.Close() })
}

// respond asks the harness to answer a turn.
func (s *HarnessSuite) respond(turnID, text string) {
	s.Require().NoError(s.harness.Respond(Turn{
		ID:           turnID,
		Instructions: "be brief",
		History:      []llm.Message{{Role: llm.User, Content: text}},
	}))
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

	s.Require().NoError(s.harness.Respond(Turn{
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
	}))

	s.Require().Len(s.fast.requests(), 1)
	sent := s.fast.requests()[0].Messages
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
	s.Require().Len(asked.Messages, 2)
	s.Equal("what is 15% of 84.20", asked.Messages[0].Content, "the conversation it was asked in")
	s.Equal("15% of 84.20", asked.Messages[1].Content)

	s.Equal("think", s.awaitDelegated(1)[0].Skill)
}

func (s *HarnessSuite) TestATurnNobodyPromptedHasSomethingToAnswer() {
	// Work coming back is a turn nobody asked for, so the conversation ends with the
	// agent's own reply rather than a caller's sentence. Asked to follow its own turn,
	// Gemini refuses the request outright -- "requests ending with a model turn are not
	// supported" -- and the caller never hears what came back.
	s.build(true)

	s.Require().NoError(s.harness.Respond(Turn{
		ID:           "turn-1",
		Instructions: "be brief",
		History: []llm.Message{
			{Role: llm.User, Content: "travel advice for Boulder?"},
			{Role: llm.Assistant, Content: "Let me look into that."},
		},
	}))

	asked := s.fast.requests()[0].Messages
	s.Equal(llm.User, asked[len(asked)-1].Role, "the model was asked to follow its own turn")
	s.Len(s.harness.history, 2, "what was added is the request's, not the conversation's")
}

func (s *HarnessSuite) TestAnAnsweredTurnIsSentAsItStands() {
	s.build(true)

	s.respond("turn-1", "what is 15% of 84.20")

	asked := s.fast.requests()[0].Messages
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
	s.respond("turn-3", "and what about tax")

	s.Require().Len(s.fast.requests(), 3)
	s.NotContains(s.fast.requests()[2].Instructions, "12.63",
		"an answer already spoken is not repeated on every later turn")
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
	s.Contains(s.fast.requests()[1].Instructions, "could not find out")
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
