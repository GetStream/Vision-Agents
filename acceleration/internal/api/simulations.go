package api

import (
	"context"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// What the simulation paths say on a deployment that cannot run one. Writing a simulation
// down only needs a database; having the conversations needs an agent to talk to and a
// model to play the caller and judge them.
const (
	noSimulations        = "simulations are not available: this deployment has no database"
	cannotRunSimulations = "simulations cannot be run here: this deployment has no sessions or model routing"
	unknownSimulation    = "no such simulation"
	unknownSimulationRun = "no such simulation run"
)

// ListSimulations returns the calling customer's simulations, newest first.
func (s *Server) ListSimulations(ctx context.Context, _ ListSimulationsRequestObject) (ListSimulationsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListSimulations401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ListSimulations400JSONResponse{badRequest(noSimulations)}, nil
	}

	stored, err := s.store.CustomerSimulations(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]Simulation, 0, len(stored))
	for _, simulation := range stored {
		listed = append(listed, simulationOf(simulation))
	}
	return ListSimulations200JSONResponse(listed), nil
}

// CreateSimulation writes down a conversation to have with an agent.
func (s *Server) CreateSimulation(ctx context.Context, request CreateSimulationRequestObject) (CreateSimulationResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateSimulation401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return CreateSimulation400JSONResponse{badRequest(noSimulations)}, nil
	}
	if request.Body == nil {
		return CreateSimulation400JSONResponse{badRequest("a request body is required")}, nil
	}

	simulation := storedSimulation(*request.Body, customerID)
	if complaint, bad := simulationComplaint(ctx, s, customerID, simulation); bad {
		return CreateSimulation400JSONResponse{badRequest(complaint)}, nil
	}
	if err := s.store.CreateSimulation(ctx, &simulation); err != nil {
		return CreateSimulation400JSONResponse{badRequest(err.Error())}, nil
	}
	return CreateSimulation201JSONResponse(simulationOf(simulation)), nil
}

// GetSimulation returns one simulation.
func (s *Server) GetSimulation(ctx context.Context, request GetSimulationRequestObject) (GetSimulationResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetSimulation401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetSimulation400JSONResponse{badRequest(noSimulations)}, nil
	}

	simulation, err := s.store.Simulation(ctx, customerID, request.Id)
	if err != nil {
		return GetSimulation404JSONResponse{NotFoundJSONResponse{Error: unknownSimulation}}, nil
	}
	return GetSimulation200JSONResponse(simulationOf(simulation)), nil
}

// UpdateSimulation replaces what a simulation asks. The runs it already has keep their own
// copy of what they tested.
func (s *Server) UpdateSimulation(ctx context.Context, request UpdateSimulationRequestObject) (UpdateSimulationResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return UpdateSimulation401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return UpdateSimulation400JSONResponse{badRequest(noSimulations)}, nil
	}
	if request.Body == nil {
		return UpdateSimulation400JSONResponse{badRequest("a request body is required")}, nil
	}

	existing, err := s.store.Simulation(ctx, customerID, request.Id)
	if err != nil {
		return UpdateSimulation404JSONResponse{NotFoundJSONResponse{Error: unknownSimulation}}, nil
	}

	simulation := storedSimulation(*request.Body, customerID)
	simulation.ID = existing.ID
	if complaint, bad := simulationComplaint(ctx, s, customerID, simulation); bad {
		return UpdateSimulation400JSONResponse{badRequest(complaint)}, nil
	}
	if err := s.store.UpdateSimulation(ctx, &simulation); err != nil {
		return UpdateSimulation400JSONResponse{badRequest(err.Error())}, nil
	}

	simulation.CreatedAt = existing.CreatedAt
	return UpdateSimulation200JSONResponse(simulationOf(simulation)), nil
}

// DeleteSimulation stops a simulation being runnable. The runs that named it are kept.
func (s *Server) DeleteSimulation(ctx context.Context, request DeleteSimulationRequestObject) (DeleteSimulationResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return DeleteSimulation401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return DeleteSimulation400JSONResponse{badRequest(noSimulations)}, nil
	}

	if err := s.store.DeleteSimulation(ctx, customerID, request.Id); err != nil {
		return DeleteSimulation404JSONResponse{NotFoundJSONResponse{Error: unknownSimulation}}, nil
	}
	return DeleteSimulation204Response{}, nil
}

// RunSimulation has the conversations. It returns once the run is written rather than once
// it is over.
func (s *Server) RunSimulation(ctx context.Context, request RunSimulationRequestObject) (RunSimulationResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return RunSimulation401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return RunSimulation400JSONResponse{badRequest(noSimulations)}, nil
	}
	if s.simulations == nil {
		return RunSimulation400JSONResponse{badRequest(cannotRunSimulations)}, nil
	}

	if _, err := s.store.Simulation(ctx, customerID, request.Id); err != nil {
		return RunSimulation404JSONResponse{NotFoundJSONResponse{Error: unknownSimulation}}, nil
	}
	run, err := s.simulations.Start(ctx, customerID, request.Id)
	if err != nil {
		return RunSimulation400JSONResponse{badRequest(err.Error())}, nil
	}
	return RunSimulation202JSONResponse(simulationRunOf(run, nil)), nil
}

// ListSimulationRuns returns what the simulations have come to, newest first.
func (s *Server) ListSimulationRuns(ctx context.Context, request ListSimulationRunsRequestObject) (ListSimulationRunsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListSimulationRuns401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ListSimulationRuns400JSONResponse{badRequest(noSimulations)}, nil
	}

	filter := store.SimulationRunFilter{
		CustomerID:   customerID,
		SimulationID: value(request.Params.SimulationId),
		Limit:        value(request.Params.Limit),
	}
	if request.Params.State != nil {
		filter.State = string(*request.Params.State)
	}

	stored, err := s.store.SimulationRuns(ctx, filter)
	if err != nil {
		return nil, err
	}

	// The conversations are left out of a list: a log of fifty runs is not fifty
	// transcripts, and the one being read is asked for by itself.
	listed := make([]SimulationRun, 0, len(stored))
	for _, run := range stored {
		listed = append(listed, simulationRunOf(run, nil))
	}
	return ListSimulationRuns200JSONResponse(listed), nil
}

// GetSimulationRun returns one run and the conversations it had.
func (s *Server) GetSimulationRun(ctx context.Context, request GetSimulationRunRequestObject) (GetSimulationRunResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetSimulationRun401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetSimulationRun400JSONResponse{badRequest(noSimulations)}, nil
	}

	run, err := s.store.SimulationRun(ctx, customerID, request.Id)
	if err != nil {
		return GetSimulationRun404JSONResponse{NotFoundJSONResponse{Error: unknownSimulationRun}}, nil
	}
	cases, err := s.store.SimulationCases(ctx, run.ID)
	if err != nil {
		return nil, err
	}
	return GetSimulationRun200JSONResponse(simulationRunOf(run, cases)), nil
}

// CancelSimulationRun stops a run, including the conversations already going.
func (s *Server) CancelSimulationRun(ctx context.Context, request CancelSimulationRunRequestObject) (CancelSimulationRunResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CancelSimulationRun401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return CancelSimulationRun400JSONResponse{badRequest(noSimulations)}, nil
	}
	if s.simulations == nil {
		return CancelSimulationRun400JSONResponse{badRequest(cannotRunSimulations)}, nil
	}

	run, err := s.simulations.Cancel(ctx, customerID, request.Id)
	if err != nil {
		return CancelSimulationRun404JSONResponse{NotFoundJSONResponse{Error: unknownSimulationRun}}, nil
	}
	return CancelSimulationRun200JSONResponse(simulationRunOf(run, nil)), nil
}

// storedSimulation maps what was asked for onto what is stored. The customer comes from the
// header rather than the body.
func storedSimulation(request SimulationRequest, customerID string) store.Simulation {
	simulation := store.Simulation{
		CustomerID:   customerID,
		Name:         strings.TrimSpace(request.Name),
		Mode:         store.SimulationText,
		ConfigID:     request.ConfigId,
		Scenario:     strings.TrimSpace(request.Scenario),
		Assertion:    strings.TrimSpace(request.Assertion),
		Variations:   value(request.Variations),
		JudgeTarget:  value(request.JudgeTarget),
		CallerTarget: value(request.CallerTarget),
		CallerTTS:    value(request.CallerTts),
		CallerSTT:    value(request.CallerStt),
		CallerVoice:  value(request.CallerVoice),
		MaxTurns:     value(request.MaxTurns),
	}
	if request.Mode != nil {
		simulation.Mode = string(*request.Mode)
	}
	if request.Tags != nil {
		simulation.Tags = *request.Tags
	}
	return simulation
}

// simulationComplaint is what is wrong with a simulation, in a sentence somebody can act
// on. It is checked when the simulation is written rather than once per conversation at
// whatever hour it was run.
func simulationComplaint(
	ctx context.Context,
	server *Server,
	customerID string,
	simulation store.Simulation,
) (string, bool) {
	switch {
	case simulation.Name == "":
		return "a simulation needs a name", true
	case simulation.ConfigID == "":
		return "a simulation needs an agent to run against", true
	case simulation.Scenario == "":
		return "a simulation needs something to ask", true
	case simulation.Assertion == "":
		return "a simulation needs something to check", true
	}
	if err := routing.Tags(simulation.Tags).Validate(); err != nil {
		return err.Error(), true
	}
	if _, err := server.store.AgentConfig(ctx, customerID, simulation.ConfigID); err != nil {
		return unknownConfig, true
	}
	return "", false
}

// simulationOf renders a simulation for the wire.
func simulationOf(simulation store.Simulation) Simulation {
	rendered := Simulation{
		Id:           simulation.ID,
		Name:         simulation.Name,
		Mode:         SimulationMode(simulation.Mode),
		ConfigId:     simulation.ConfigID,
		Scenario:     simulation.Scenario,
		Assertion:    simulation.Assertion,
		Variations:   simulation.Variations,
		JudgeTarget:  optional(simulation.JudgeTarget),
		CallerTarget: optional(simulation.CallerTarget),
		CallerTts:    optional(simulation.CallerTTS),
		CallerStt:    optional(simulation.CallerSTT),
		CallerVoice:  optional(simulation.CallerVoice),
		MaxTurns:     simulation.MaxTurns,
		CreatedAt:    simulation.CreatedAt,
		UpdatedAt:    &simulation.UpdatedAt,
	}
	if len(simulation.Tags) > 0 {
		tags := simulation.Tags
		rendered.Tags = &tags
	}
	return rendered
}

// simulationRunOf renders a run, with its conversations when they were asked for.
func simulationRunOf(run store.SimulationRun, cases []store.SimulationCase) SimulationRun {
	rendered := SimulationRun{
		Id:           run.ID,
		SimulationId: run.SimulationID,
		State:        SimulationRunState(run.State),
		Cases:        run.Cases,
		Passed:       run.Passed,
		Failed:       run.Failed,
		ConfigId:     optional(run.ConfigID),
		Scenario:     optional(run.Scenario),
		Assertion:    optional(run.Assertion),
		JudgeTarget:  optional(run.JudgeTarget),
		Error:        optional(run.Error),
		StartedAt:    run.StartedAt,
		FinishedAt:   run.FinishedAt,
	}
	if run.Mode != "" {
		mode := SimulationRunMode(run.Mode)
		rendered.Mode = &mode
	}
	if cases != nil {
		conversations := simulationCasesOf(cases)
		rendered.Conversations = &conversations
	}
	return rendered
}

// simulationCasesOf renders a run's conversations.
func simulationCasesOf(cases []store.SimulationCase) []SimulationCase {
	rendered := make([]SimulationCase, 0, len(cases))
	for _, kase := range cases {
		one := SimulationCase{
			Id:         kase.ID,
			Variation:  kase.Variation,
			Scenario:   kase.Scenario,
			State:      SimulationCaseState(kase.State),
			CallId:     optional(kase.CallID),
			Turns:      kase.Turns,
			Passed:     kase.Passed,
			Verdict:    optional(kase.Verdict),
			Score:      kase.Score,
			Error:      optional(kase.Error),
			StartedAt:  kase.StartedAt,
			FinishedAt: kase.FinishedAt,
		}
		if kase.Ended != "" {
			ended := SimulationCaseEnded(kase.Ended)
			one.Ended = &ended
		}
		if len(kase.Transcript) > 0 {
			transcript := simulationLinesOf(kase.Transcript)
			one.Transcript = &transcript
		}
		rendered = append(rendered, one)
	}
	return rendered
}

func simulationLinesOf(lines []store.SimulationLine) []SimulationLine {
	rendered := make([]SimulationLine, 0, len(lines))
	for _, line := range lines {
		rendered = append(rendered, SimulationLine{
			Caller:   line.Caller,
			Text:     line.Text,
			Intended: optional(line.Intended),
			At:       &line.At,
		})
	}
	return rendered
}
