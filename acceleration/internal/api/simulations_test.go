//go:build integration

package api

import (
	"net/http"
	"testing"
)

// SimulationsSuite covers writing down a conversation to put an agent through, running it,
// and reading what it came to.
type SimulationsSuite struct {
	RouterSuite
}

func TestSimulationsSuite(t *testing.T) {
	runSuite(t, new(SimulationsSuite))
}

func (s *SimulationsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SimulationsSuite) TestASimulationSurvivesBeingStoredAndReadBack() {
	agent := s.data.createAgentConfig()

	created := s.createSimulation(SimulationRequest{
		Name: "orders", ConfigId: agent.Id,
		Scenario:  "order a pizza and then change your mind",
		Assertion: "did the agent cancel the order?", Variations: pointerTo(3),
		MaxTurns: pointerTo(4), Tags: &map[string]string{"project": "support"},
	})

	s.Require().NotEmpty(created.Id)
	read := s.simulation(created.Id)
	s.Equal("orders", read.Name)
	s.Equal(agent.Id, read.ConfigId)
	s.Equal("order a pizza and then change your mind", read.Scenario)
	s.Equal("did the agent cancel the order?", read.Assertion)
	s.Equal(3, read.Variations)
	s.Equal(4, read.MaxTurns)
	s.Equal(SimulationMode("text"), read.Mode, "text is what a simulation is unless it says otherwise")
}

func (s *SimulationsSuite) TestTheAppsSimulationsAreListedBack() {
	created := s.createSimulation(s.against(s.data.createAgentConfig()))

	var listed []Simulation
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/simulations", nil, &listed))

	named := make([]string, 0, len(listed))
	for _, simulation := range listed {
		named = append(named, simulation.Id)
	}
	s.Contains(named, created.Id)
}

func (s *SimulationsSuite) TestASimulationIsEditedWithoutBeingRewritten() {
	created := s.createSimulation(s.against(s.data.createAgentConfig()))

	edited := s.against(AgentConfig{Id: created.ConfigId})
	edited.Name = "renamed"
	edited.Scenario = "ask for a refund"
	var updated Simulation
	s.Require().Equal(http.StatusOK, s.serverClient.do(
		http.MethodPut, "/v1/agents/simulations/"+created.Id, edited, &updated))

	s.Equal("renamed", updated.Name)
	s.Equal("ask for a refund", s.simulation(created.Id).Scenario)
}

func (s *SimulationsSuite) TestADeletedSimulationIsGone() {
	created := s.createSimulation(s.against(s.data.createAgentConfig()))

	s.Require().Equal(http.StatusNoContent,
		s.serverClient.do(http.MethodDelete, "/v1/agents/simulations/"+created.Id, nil, nil))

	status, _ := s.serverClient.call(http.MethodGet, "/v1/agents/simulations/"+created.Id, nil)
	s.Equal(http.StatusNotFound, status)
}

func (s *SimulationsSuite) TestASimulationWithNothingToCheckIsRefused() {
	agent := s.data.createAgentConfig()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/simulations",
		SimulationRequest{Name: "orders", ConfigId: agent.Id, Scenario: "order a pizza"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "something to check")
}

func (s *SimulationsSuite) TestASimulationAgainstAnAgentThatIsNotThereIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/simulations",
		SimulationRequest{Name: "orders", ConfigId: s.utils.uuid(),
			Scenario: "order a pizza", Assertion: "was one ordered?"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, unknownConfig.message)
}

func (s *SimulationsSuite) TestRunningASimulationStartsTheConversationsItAsksFor() {
	created := s.createSimulation(s.against(s.data.createAgentConfig()))

	var run SimulationRun
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(
		http.MethodPost, "/v1/agents/simulations/"+created.Id+"/run", nil, &run))

	s.Equal(created.Id, run.SimulationId)
	s.Equal(1, run.Cases, "one way of asking, as the simulation was written")
	s.NotEmpty(run.State)
}

func (s *SimulationsSuite) TestARunIsReadBackWithTheConversationsItHad() {
	created := s.createSimulation(s.against(s.data.createAgentConfig()))
	started := s.run(created.Id)

	var read SimulationRun
	s.Require().Equal(http.StatusOK, s.serverClient.do(
		http.MethodGet, "/v1/agents/simulation-runs/"+started.Id, nil, &read))

	s.Equal(started.Id, read.Id)
	s.NotNil(read.Conversations, "one run asked for by itself carries its transcripts")
}

func (s *SimulationsSuite) TestTheLogOfRunsIsNotOneOfTheRuns() {
	// A run named "runs" would otherwise be the log, and the log would never be reachable.
	created := s.createSimulation(s.against(s.data.createAgentConfig()))
	started := s.run(created.Id)

	var log []SimulationRun
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/simulation-runs?simulation_id="+created.Id, nil, &log))

	ran := make([]string, 0, len(log))
	for _, run := range log {
		ran = append(ran, run.Id)
	}
	s.Contains(ran, started.Id)
	s.Nil(log[0].Conversations, "a log of fifty runs is not fifty transcripts")
}

func (s *SimulationsSuite) TestARunNobodyStartedIsNotFound() {
	status, _ := s.serverClient.call(http.MethodGet, "/v1/agents/simulation-runs/"+s.utils.uuid(), nil)

	s.Equal(http.StatusNotFound, status)
}

func (s *SimulationsSuite) TestAnotherAppsSimulationIsNotFound() {
	created := s.createSimulation(s.against(s.data.createAgentConfig()))

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/simulations/"+created.Id, nil, nil)
	})
}

func (s *SimulationsSuite) TestOnlyTheAppsOwnBackendMayWriteASimulation() {
	agent := s.data.createAgentConfig()

	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodPost, "/v1/agents/simulations", s.against(agent))
		return status
	})
}

// against is a simulation to put one agent through, named so that two tests' rows are not
// each other's.
func (s *SimulationsSuite) against(agent AgentConfig) SimulationRequest {
	return SimulationRequest{
		Name: "orders-" + s.utils.uuid(), ConfigId: agent.Id,
		Scenario: "order a pizza", Assertion: "was one ordered?",
	}
}

func (s *SimulationsSuite) createSimulation(request SimulationRequest) Simulation {
	var created Simulation
	s.Require().Equal(http.StatusCreated,
		s.serverClient.do(http.MethodPost, "/v1/agents/simulations", request, &created))
	return created
}

func (s *SimulationsSuite) simulation(id string) Simulation {
	var read Simulation
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/simulations/"+id, nil, &read))
	return read
}

func (s *SimulationsSuite) run(id string) SimulationRun {
	var started SimulationRun
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(
		http.MethodPost, "/v1/agents/simulations/"+id+"/run", nil, &started))
	return started
}
