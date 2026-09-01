package simulation

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// script is what the models in a test are going to say. The caller, the judge and the
// rewriter all reach the same router and are told apart by what they were told to be, which
// is also how a test says which of them it is answering for.
type script struct {
	mu sync.Mutex
	// caller is answered in order, one per turn.
	caller   []string
	judge    string
	rewrites string
	asked    []llm.Request
}

func (s *script) next(request llm.Request) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.asked = append(s.asked, request)

	switch {
	case strings.HasPrefix(request.Instructions, judgeInstructions):
		return s.judge
	case strings.HasPrefix(request.Instructions, expandInstructions):
		return s.rewrites
	case len(s.caller) > 0:
		said := s.caller[0]
		s.caller = s.caller[1:]
		return said
	default:
		return `{"say": "", "done": true}`
	}
}

func (s *script) requests() []llm.Request {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]llm.Request(nil), s.asked...)
}

// scriptedModel is one model reading from the script. Each is its own because a router
// hands out a session per Start, and two sessions sharing an emitter would each consume the
// other's answers.
type scriptedModel struct {
	script  *script
	emitter *llm.Emitter
}

func (m *scriptedModel) Start(context.Context) error { return nil }

func (m *scriptedModel) Respond(request llm.Request) error {
	answer := m.script.next(request)
	go func() {
		m.emitter.Send(llm.CompletionStarted{CompletionID: request.ID, At: time.Now()})
		m.emitter.Send(llm.CompletionComplete{CompletionID: request.ID, Text: answer})
	}()
	return nil
}

func (m *scriptedModel) Interrupt(...string) error { return nil }
func (m *scriptedModel) Events() <-chan llm.Event  { return m.emitter.Events() }
func (m *scriptedModel) Provider() string          { return "scripted" }
func (m *scriptedModel) Model() string             { return "scripted-model" }
func (m *scriptedModel) Reasoning() bool           { return false }

func (m *scriptedModel) Close() error {
	m.emitter.Close()
	return nil
}

// scriptedTransport is an agent that has already decided what it is going to say, which is
// what testing the loop above it needs rather than an agent.
type scriptedTransport struct {
	replies  []string
	greeting string
	fail     error

	mu    sync.Mutex
	heard []string
}

func (t *scriptedTransport) Session() *session.Session { return nil }

func (t *scriptedTransport) Opening() string { return t.greeting }

func (t *scriptedTransport) Say(_ context.Context, text string) (store.SimulationLine, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.heard = append(t.heard, text)

	if len(t.replies) == 0 {
		if t.fail != nil {
			return store.SimulationLine{}, t.fail
		}
		return store.SimulationLine{Text: "I see."}, nil
	}
	reply := t.replies[0]
	t.replies = t.replies[1:]
	return store.SimulationLine{Text: reply}, nil
}

func (t *scriptedTransport) Close() error { return nil }

func (t *scriptedTransport) said() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.heard...)
}

type SimulationSuite struct {
	suite.Suite
	ctx    context.Context
	script *script
	router *llmrouter.Router
}

func TestSimulationSuite(t *testing.T) {
	suite.Run(t, new(SimulationSuite))
}

func (s *SimulationSuite) SetupTest() {
	s.ctx = context.Background()
	s.script = &script{}

	registry := llmrouter.NewRegistry()
	registry.Register("scripted", func(routing.Spec) (llm.LLM, error) {
		return &scriptedModel{script: s.script, emitter: llm.NewEmitter(64)}, nil
	})
	router, err := llmrouter.New(llmrouter.Options{
		Config:   scriptedConfig(),
		Registry: registry,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	s.router = router
}

// caller opens a persona over the scripted router.
func (s *SimulationSuite) caller(brief string) *caller {
	persona, err := newCaller(s.ctx, s.router, llmrouter.Request{
		CustomerID: "customer-1", Target: "scripted/scripted-model",
	}, brief)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = persona.Close() })
	return persona
}

func (s *SimulationSuite) TestTheConversationEndsWhenTheCallerHasAskedEverythingItCameToAsk() {
	s.script.caller = []string{
		`{"say": "I would like a pasta bolognese.", "done": false}`,
		`{"say": "Actually make it a pepperoni pizza.", "done": false}`,
		`{"say": "", "done": true}`,
	}
	over := &scriptedTransport{replies: []string{"One bolognese.", "Changed to pepperoni."}}

	so, why, err := exchange(s.ctx, s.caller("order some food"), over, 12)

	s.Require().NoError(err)
	s.Equal(store.EndedComplete, why)
	s.Equal(2, so.turns())
	s.Len(so, 4)
	s.Equal("I would like a pasta bolognese.", so[0].Text)
	s.True(so[0].Caller)
	s.Equal("One bolognese.", so[1].Text)
	s.False(so[1].Caller)
	s.Equal([]string{"I would like a pasta bolognese.", "Actually make it a pepperoni pizza."}, over.said())
}

func (s *SimulationSuite) TestAConversationStopsAtTheTurnLimitWhenTheCallerNeverFinishes() {
	s.script.caller = []string{
		`{"say": "One.", "done": false}`,
		`{"say": "Two.", "done": false}`,
		`{"say": "Three.", "done": false}`,
		`{"say": "Four.", "done": false}`,
	}
	over := &scriptedTransport{}

	so, why, err := exchange(s.ctx, s.caller("keep talking"), over, 3)

	s.Require().NoError(err)
	s.Equal(store.EndedTurns, why)
	s.Equal(3, so.turns())
}

func (s *SimulationSuite) TestAConversationThatFailedPartWayThroughKeepsWhatWasSaidBeforeIt() {
	s.script.caller = []string{
		`{"say": "Is anyone there?", "done": false}`,
		`{"say": "Hello?", "done": false}`,
	}
	over := &scriptedTransport{replies: []string{"Hello."}, fail: errors.New("the agent did not answer")}

	so, why, err := exchange(s.ctx, s.caller("say hello"), over, 12)

	s.Require().Error(err)
	s.Equal(store.EndedFailed, why)
	// The failed turn is still in the transcript: what the caller said before nobody
	// answered is exactly what makes the failure readable.
	s.Len(so, 3)
	s.Equal("Hello?", so[2].Text)
}

func (s *SimulationSuite) TestACallerWithNothingLeftToSayHasFinishedRatherThanFailed() {
	s.script.caller = []string{`{"say": "   ", "done": false}`}

	so, why, err := exchange(s.ctx, s.caller("ask nothing"), &scriptedTransport{}, 12)

	s.Require().NoError(err)
	s.Equal(store.EndedComplete, why)
	s.Empty(so)
}

func (s *SimulationSuite) TestTheCallerIsNeverToldItIsTalkingToAnAgent() {
	s.script.caller = []string{`{"say": "", "done": true}`}

	_, _, err := exchange(s.ctx, s.caller("order a pizza"), &scriptedTransport{}, 12)
	s.Require().NoError(err)

	asked := s.script.requests()
	s.Require().NotEmpty(asked)
	s.Contains(asked[0].Instructions, "order a pizza")
	s.Contains(asked[0].Instructions, "Never mention that this is a test")
	s.True(asked[0].JSON, "the caller is asked for a decision, so it has to answer in JSON")
}

func (s *SimulationSuite) TestTheCallerAnswersTheGreetingRatherThanTalkingOverIt() {
	s.script.caller = []string{`{"say": "Hi, one pizza please.", "done": false}`, `{"say": "", "done": true}`}
	over := &scriptedTransport{greeting: "Northwind, how can I help?", replies: []string{"Certainly."}}

	so, why, err := exchange(s.ctx, s.caller("order a pizza"), over, 12)

	s.Require().NoError(err)
	s.Equal(store.EndedComplete, why)
	s.Require().Len(so, 3)
	s.Equal("Northwind, how can I help?", so[0].Text)
	s.False(so[0].Caller)

	// The caller was handed the greeting before it decided what to say, which is the whole
	// reason a greeting is in the transcript rather than only on the wire.
	asked := s.script.requests()
	s.Require().NotEmpty(asked)
	s.Contains(asked[0].Messages[0].Content, "Northwind, how can I help?")
}

func (s *SimulationSuite) TestAFencedRulingIsStillUnderstood() {
	ruled, err := parseVerdict("```json\n{\"passed\": true, \"reason\": \"They ordered it.\", \"score\": 5}\n```")

	s.Require().NoError(err)
	s.True(ruled.Passed)
	s.Equal(5, ruled.Score)
}

func (s *SimulationSuite) TestARulingWithNoReasonIsRefusedRatherThanBelieved() {
	_, err := parseVerdict(`{"passed": true, "reason": "  "}`)

	s.Require().Error(err)
	s.Contains(err.Error(), "without saying why")
}

func (s *SimulationSuite) TestTheJudgeIsAskedOnlyTheQuestionItWasGiven() {
	s.script.judge = `{"passed": false, "reason": "No delivery time was agreed.", "score": 4}`

	ruled, err := rule(s.ctx, s.router, llmrouter.Request{
		CustomerID: "customer-1", Target: "scripted/scripted-model",
	}, "judge-1", "was an order placed for 8pm?", said{
		{Caller: true, Text: "One pizza please."},
		{Text: "Certainly."},
	})

	s.Require().NoError(err)
	s.False(ruled.Passed)
	s.Equal(4, ruled.Score)

	asked := s.script.requests()
	s.Require().NotEmpty(asked)
	content := asked[len(asked)-1].Messages[0].Content
	s.Contains(content, "was an order placed for 8pm?")
	s.Contains(content, "Caller: One pizza please.")
	s.Contains(content, "Agent: Certainly.")
}

func (s *SimulationSuite) TestRewritingKeepsAtMostAsManyWaysOfAskingAsWereWanted() {
	s.script.rewrites = `{"variations": ["one", "two", "three", "four"]}`

	rewrites, err := expand(s.ctx, s.router, llmrouter.Request{
		CustomerID: "customer-1", Target: "scripted/scripted-model",
	}, "expand-1", "order a pizza", 2)

	s.Require().NoError(err)
	s.Equal([]string{"one", "two"}, rewrites)
}

func (s *SimulationSuite) TestFewerWaysOfAskingThanWereWantedIsWhatTheRunUses() {
	s.script.rewrites = `{"variations": ["one", "  ", "two"]}`

	rewrites, err := expand(s.ctx, s.router, llmrouter.Request{
		CustomerID: "customer-1", Target: "scripted/scripted-model",
	}, "expand-1", "order a pizza", 9)

	s.Require().NoError(err)
	s.Equal([]string{"one", "two"}, rewrites)
}

func (s *SimulationSuite) TestARunnerNeedsSomewhereToWriteAModelAndAnAgentToTest() {
	_, err := New(Options{})
	s.Require().Error(err)
	s.Contains(err.Error(), "database")

	_, err = New(Options{Store: &store.Store{}})
	s.Require().Error(err)
	s.Contains(err.Error(), "session manager")
}

func (s *SimulationSuite) TestARunPassesOnlyIfEveryOneOfItsConversationsDid() {
	done := context.Background()

	s.Equal(store.SimulationPassed, ended(done, store.SimulationRun{Cases: 3, Passed: 3}))
	s.Equal(store.SimulationFailed, ended(done, store.SimulationRun{Cases: 3, Passed: 2, Failed: 1}))
	// A conversation that never got as far as a ruling is not a conversation that failed,
	// so a run missing one of them is neither passed nor failed.
	s.Equal(store.SimulationErrored, ended(done, store.SimulationRun{Cases: 3, Passed: 2}))

	stopped, cancel := context.WithCancel(context.Background())
	cancel()
	s.Equal(store.SimulationCancelled, ended(stopped, store.SimulationRun{Cases: 3, Passed: 1}))
}

// scriptedConfig is one provider, which is all these tests need from routing.
func scriptedConfig() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{
			Provider:  "scripted",
			Model:     "scripted-model",
			Languages: []string{"en"},
			Realtime:  true,
		}},
		Aliases: map[string]routing.Alias{
			"llm-fast": {Languages: []string{"en"}, RequireRealtime: true},
		},
	}
}
