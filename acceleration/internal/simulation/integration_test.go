//go:build integration

// The whole feature, against real models and a real database.
//
// What is asserted is that the judge distinguishes an agent that did the thing from one
// that did not. A suite that only ran the happy case would pass just as well against a
// judge that said yes to everything, which is the failure worth catching.
package simulation

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// The sprint's own scenario, which is the one worth proving the feature on.
const (
	orderScenario = "In 3 steps do the following: place an order for pasta bolognese, " +
		"after the order is handled change your mind and change it into pepperoni pizza. " +
		"Tell them to deliver at 8pm."
	orderAssertion = "Was an order placed for a pepperoni pizza to be delivered at 8pm?"

	// takesOrders does what it is asked. changesNothing is the same agent with one thing
	// taken away, which is what the judge has to notice.
	takesOrders = "You take food orders for Northwind Pizza over the phone. Keep your " +
		"answers to one or two sentences. Confirm each thing the caller asks for as you " +
		"agree to it, and read the whole order back when they are done."
	changesNothing = "You take food orders for Northwind Pizza over the phone. Keep your " +
		"answers to one or two sentences. Once an order has been placed it is final: " +
		"politely refuse to change any item on it, and refuse to set a delivery time. " +
		"Never agree to a change."
)

// The two scenarios a simulation is described with: one settled by what the agent says, and
// one settled only by what it did.
const (
	capitalScenario  = "Ask what the capital of France is."
	capitalAssertion = "Should mention Paris."
	answersQuestions = "You answer general knowledge questions in one short sentence."
	// withholdsAnswers is the same agent, never giving the answer away.
	withholdsAnswers = "You are a quiz host. Whatever you are asked, never give the answer, " +
		"never name a city, and never confirm or deny a guess: reply only with a vague hint."

	changedOrderScenario = "First tell the agent you want pasta bolognese. Ask how long the " +
		"wait is. After that change your order into a pizza hawaii."
	changedOrderAssertion   = "Verify that an order was placed for pizza hawaii."
	ordersThroughTheKitchen = "You take food orders for Northwind Pizza. Keep your answers to " +
		"one or two sentences. The wait is 25 minutes. An order is only placed once you have " +
		"called place_order with it, so call it for every order and every change before you " +
		"confirm anything to the caller."
	// ordersInWords confirms the same orders without ever placing one, which a judge
	// reading only the words would not be able to tell apart.
	ordersInWords = "You take food orders for Northwind Pizza. Keep your answers to one or " +
		"two sentences. The wait is 25 minutes. Confirm orders and changes in words. Never " +
		"call place_order or any other tool."
)

const suiteWithin = 10 * time.Minute

// counter is a worker hosting the place_order tool, for the customers a test opened it to.
// An agent config has no tools of its own, so this is how the agent under test gets one.
type counter struct {
	mu     sync.Mutex
	open   map[string]bool
	orders map[string][]string
}

func (c *counter) serve(customerID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.open[customerID] = true
}

func (c *counter) HostedTools(customerID, _ string) ([]dispatch.Tool, time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.open[customerID] {
		return nil, 0
	}
	return []dispatch.Tool{{
		Name: placeOrder.Name, Description: placeOrder.Description, Parameters: placeOrder.Parameters,
	}}, time.Minute
}

func (c *counter) RunHosted(_ context.Context, customerID, _ string, call dispatch.ToolCall) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.orders[customerID] = append(c.orders[customerID], call.Arguments)
	return fmt.Sprintf("order %d placed: %s", len(c.orders[customerID]), call.Arguments), nil
}

func (c *counter) placed(customerID string) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.orders[customerID]...)
}

type IntegrationSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	runner     *Runner
	llm        *llmrouter.Router
	counter    *counter
	customerID string
}

func TestIntegrationSuite(t *testing.T) {
	suite.Run(t, new(IntegrationSuite))
}

func (s *IntegrationSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	if os.Getenv("OPENAI_API_KEY") == "" {
		s.T().Skip("OPENAI_API_KEY not set")
	}

	var cancel context.CancelFunc
	s.ctx, cancel = context.WithTimeout(context.Background(), suiteWithin)
	s.T().Cleanup(cancel)

	logger := slog.New(slog.DiscardHandler)

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(s.ctx))
	s.T().Cleanup(func() { _ = pgStore.Close() })
	s.store = pgStore

	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	reasoner, err := llmrouter.New(llmrouter.Options{
		Config: config[routing.LLM], Registry: llmrouter.DefaultRegistry(), Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)
	s.llm = reasoner

	// A conversation in writing joins nothing and says nothing out loud, but a manager
	// insists on knowing how it would have done both.
	transcriber, err := sttrouter.New(sttrouter.Options{
		Config: config[routing.STT], Registry: sttrouter.DefaultRegistry(), Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)

	speaker, err := ttsrouter.New(ttsrouter.Options{
		Config: config[routing.TTS], Registry: ttsrouter.DefaultRegistry(), Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	sessions, err := session.NewManager(session.ManagerOptions{
		LLM: reasoner, STT: transcriber, TTS: speaker, Logger: logger,
		Edge: func(context.Context, session.Spec, streamapp.Bound, *slog.Logger) (agent.Edge, error) {
			return nil, errors.New("there is no call to join")
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = sessions.Shutdown() })
	s.counter = &counter{open: map[string]bool{}, orders: map[string][]string{}}
	sessions.HostTools(s.counter)

	runner, err := New(Options{
		Store: pgStore, Sessions: sessions, LLM: reasoner,
		TTS: speaker, STT: transcriber, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(runner.Close)
	s.runner = runner
}

func (s *IntegrationSuite) SetupTest() {
	s.customerID = "customer-" + time.Now().Format("150405.000000000")
}

// agentThat stores an agent under test and returns its config id.
func (s *IntegrationSuite) agentThat(instructions string) string {
	config := store.AgentConfig{
		CustomerID:   s.customerID,
		Name:         "order taker",
		Mode:         store.AgentModeText,
		Instructions: instructions,
		LLM:          "openai/gpt-5.6-luna",
	}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, &config))
	return config.ID
}

// asks stores a simulation and returns its id.
func (s *IntegrationSuite) asks(configID string, variations int) string {
	return s.asksIn(store.SimulationText, configID, variations)
}

func (s *IntegrationSuite) asksIn(mode, configID string, variations int) string {
	return s.simulates(store.Simulation{
		Name:       "changes their mind mid-order",
		Mode:       mode,
		ConfigID:   configID,
		Scenario:   orderScenario,
		Assertion:  orderAssertion,
		Variations: variations,
	})
}

// checks stores a text simulation of one scenario and returns its id.
func (s *IntegrationSuite) checks(configID, scenario, assertion string) string {
	return s.simulates(store.Simulation{
		Name:       scenario,
		Mode:       store.SimulationText,
		ConfigID:   configID,
		Scenario:   scenario,
		Assertion:  assertion,
		Variations: 1,
	})
}

func (s *IntegrationSuite) simulates(simulation store.Simulation) string {
	simulation.CustomerID = s.customerID
	simulation.MaxTurns = 10
	// The suite runs on an OpenAI key, and the default judge is whichever model the
	// quality tier prefers, which is neither guaranteed to be OpenAI's nor quick.
	simulation.JudgeTarget = "openai/gpt-5.6-terra"
	s.Require().NoError(s.store.CreateSimulation(s.ctx, &simulation))
	return simulation.ID
}

// judged runs a simulation of one conversation and returns it.
func (s *IntegrationSuite) judged(simulationID string) store.SimulationCase {
	cases := s.conversations(s.runs(simulationID))
	s.Require().Len(cases, 1)
	return cases[0]
}

// runs starts a simulation and waits for it to be over.
func (s *IntegrationSuite) runs(simulationID string) store.SimulationRun {
	started, err := s.runner.Start(s.ctx, s.customerID, simulationID)
	s.Require().NoError(err)

	var finished store.SimulationRun
	s.Require().Eventually(func() bool {
		run, err := s.store.SimulationRun(s.ctx, s.customerID, started.ID)
		if err != nil || run.State == store.SimulationRunning {
			return false
		}
		finished = run
		return true
	}, 5*time.Minute, time.Second, "waited for the run to be over")
	return finished
}

// conversations reads back what a run said, which is what makes a failure readable.
func (s *IntegrationSuite) conversations(run store.SimulationRun) []store.SimulationCase {
	cases, err := s.store.SimulationCases(s.ctx, run.ID)
	s.Require().NoError(err)
	return cases
}

func (s *IntegrationSuite) TestAnAgentThatTakesTheChangedOrderIsJudgedToHavePassed() {
	run := s.runs(s.asks(s.agentThat(takesOrders), 1))
	cases := s.conversations(run)

	s.Require().Len(cases, 1)
	s.Equal(store.SimulationPassed, cases[0].State, "the judge said: %s", cases[0].Verdict)
	s.Equal(store.SimulationPassed, run.State)
	s.Equal(1, run.Passed)

	// The transcript is the evidence, so it has to be both halves of the conversation.
	s.Greater(cases[0].Turns, 1)
	said := cases[0].Transcript
	s.Require().NotEmpty(said)
	s.True(said[0].Caller, "the caller speaks first when the agent has no greeting")
	s.NotEmpty(cases[0].Verdict)
}

func (s *IntegrationSuite) TestAnAgentThatRefusesToChangeTheOrderIsJudgedToHaveFailed() {
	run := s.runs(s.asks(s.agentThat(changesNothing), 1))
	cases := s.conversations(run)

	s.Require().Len(cases, 1)
	s.Equal(store.SimulationFailed, cases[0].State, "the judge said: %s", cases[0].Verdict)
	s.Equal(store.SimulationFailed, run.State)
	s.Equal(1, run.Failed)
	s.NotEmpty(cases[0].Verdict)
}

func (s *IntegrationSuite) TestAskingSeveralWaysHasSeveralConversationsUnderOneRun() {
	run := s.runs(s.asks(s.agentThat(takesOrders), 4))
	cases := s.conversations(run)

	s.Require().Len(cases, 4)
	s.Equal(4, run.Cases)

	// The scenario as written is always the first way of asking, and the rest are other
	// wordings of the same thing rather than other things.
	s.Equal(orderScenario, cases[0].Scenario)
	for _, one := range cases[1:] {
		s.NotEqual(orderScenario, one.Scenario)
		s.NotEmpty(one.Scenario)
	}
	// Every one of them was actually had, whatever it was judged to be.
	for _, one := range cases {
		s.NotEqual(store.Pending, one.State)
		s.NotEmpty(one.CallID)
	}
}

func (s *IntegrationSuite) TestAnAgentThatNamesTheCapitalOfFranceIsJudgedToHaveMentionedParis() {
	kase := s.judged(s.checks(s.agentThat(answersQuestions), capitalScenario, capitalAssertion))

	s.Equal(store.SimulationPassed, kase.State, "the judge said: %s", kase.Verdict)
}

func (s *IntegrationSuite) TestAnAgentThatNeverNamesACityIsJudgedNotToHaveMentionedParis() {
	kase := s.judged(s.checks(s.agentThat(withholdsAnswers), capitalScenario, capitalAssertion))

	s.Equal(store.SimulationFailed, kase.State, "the judge said: %s", kase.Verdict)
}

func (s *IntegrationSuite) TestAnAgentThatPlacesTheChangedOrderIsJudgedToHavePlacedIt() {
	s.counter.serve(s.customerID)

	kase := s.judged(s.checks(s.agentThat(ordersThroughTheKitchen), changedOrderScenario, changedOrderAssertion))

	s.Equal(store.SimulationPassed, kase.State, "the judge said: %s", kase.Verdict)
	placed := s.counter.placed(s.customerID)
	s.True(slices.ContainsFunc(placed, func(order string) bool {
		return strings.Contains(strings.ToLower(order), "hawai")
	}), "the kitchen got %v", placed)

	// The transcript carries what the agent did, which is what the judge ruled on.
	var used []string
	for _, line := range kase.Transcript {
		for _, tool := range line.Tools {
			used = append(used, tool.Name)
		}
	}
	s.Contains(used, "place_order")
}

func (s *IntegrationSuite) TestAnAgentThatOnlySaysItPlacedTheOrderIsJudgedNotToHavePlacedIt() {
	s.counter.serve(s.customerID)

	kase := s.judged(s.checks(s.agentThat(ordersInWords), changedOrderScenario, changedOrderAssertion))

	s.Require().Empty(s.counter.placed(s.customerID), "the agent was told never to place an order")
	s.Equal(store.SimulationFailed, kase.State, "the judge said: %s", kase.Verdict)
}

func (s *IntegrationSuite) TestAnAgentIsHeardThroughItsOwnVoiceRatherThanItsOwnWords() {
	for _, name := range []string{"DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY"} {
		if os.Getenv(name) == "" {
			s.T().Skip(name + " not set")
		}
	}

	config := store.AgentConfig{
		CustomerID:   s.customerID,
		Name:         "order taker",
		Mode:         store.AgentModeVoice,
		Instructions: takesOrders,
		Greeting:     "Northwind Pizza, what can I get you?",
		LLM:          "openai/gpt-5.6-luna",
	}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, &config))

	run := s.runs(s.asksIn(store.SimulationAudio, config.ID, 1))
	cases := s.conversations(run)

	s.Require().Len(cases, 1)
	said := cases[0].Transcript
	s.Require().NotEmpty(said)

	// The greeting is the agent's, so the conversation opens with it rather than with the
	// caller talking into silence.
	s.False(said[0].Caller)
	s.Greater(cases[0].Turns, 1, "the judge said: %s", cases[0].Verdict)

	// The whole point of running out loud: what is recorded is what a caller heard, and
	// what the agent meant is kept beside it. They should be close but need not match, and
	// a transcript with neither would mean the pipeline was never exercised.
	var overheard bool
	for _, line := range said {
		if !line.Caller && line.Intended != "" {
			overheard = true
		}
	}
	s.True(overheard, "nothing the agent said was heard through its own voice")
}
