package api

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
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

type SimulationMode string

const (
	SimulationModeAudio SimulationMode = "audio"
	SimulationModeText  SimulationMode = "text"
)

// Valid indicates whether the value is a known member of the SimulationMode enum.
func (e SimulationMode) Valid() bool {
	switch e {
	case SimulationModeAudio:
		return true
	case SimulationModeText:
		return true
	default:
		return false
	}
}

type Simulation struct {
	Id           string             `json:"id"`
	Name         string             `json:"name"`
	Mode         SimulationMode     `json:"mode" enum:"text,audio"`
	ConfigId     string             `json:"config_id"`
	Scenario     string             `json:"scenario"`
	Assertion    string             `json:"assertion"`
	Variations   int                `json:"variations"`
	JudgeTarget  *string            `json:"judge_target,omitempty"`
	CallerTarget *string            `json:"caller_target,omitempty"`
	CallerTts    *string            `json:"caller_tts,omitempty"`
	CallerStt    *string            `json:"caller_stt,omitempty"`
	CallerVoice  *string            `json:"caller_voice,omitempty"`
	MaxTurns     int                `json:"max_turns"`
	Tags         *map[string]string `json:"tags,omitempty"`
	CreatedAt    time.Time          `json:"created_at"`
	UpdatedAt    *time.Time         `json:"updated_at,omitempty"`
}

type SimulationRequestMode string

const (
	SimulationRequestModeAudio SimulationRequestMode = "audio"
	SimulationRequestModeText  SimulationRequestMode = "text"
)

// Valid indicates whether the value is a known member of the SimulationRequestMode enum.
func (e SimulationRequestMode) Valid() bool {
	switch e {
	case SimulationRequestModeAudio:
		return true
	case SimulationRequestModeText:
		return true
	default:
		return false
	}
}

type SimulationRequest struct {
	Name         string                 `json:"name"`
	Mode         *SimulationRequestMode `json:"mode,omitempty" doc:"Text hands the agent the words, which tests everything between hearing and answering. Audio generates speech and runs the whole pipeline, so what is judged is what a caller would actually have heard. Text when left out." enum:"text,audio"`
	ConfigId     string                 `json:"config_id" doc:"The agent being tested."`
	Scenario     string                 `json:"scenario" doc:"What to ask, in your own words and over as many turns as it takes. This is a brief for the caller rather than a script, so it may describe things that depend on what the agent says back."`
	Assertion    string                 `json:"assertion" doc:"What has to be true at the end for the run to have passed."`
	Variations   *int                   `json:"variations,omitempty" doc:"How many ways of asking the same thing one run tries, up to ten. The scenario as written is always the first of them, and one is what left out means."`
	JudgeTarget  *string                `json:"judge_target,omitempty" doc:"The model that rules on the conversations, named the way any other routing target is. Empty takes llm-judge, the deployment's quality-tier default, since nobody is waiting for it."`
	CallerTarget *string                `json:"caller_target,omitempty" doc:"The model that plays the caller. Empty takes llm-scenario-runner, the deployment's fast-tier default."`
	CallerTts    *string                `json:"caller_tts,omitempty" doc:"How the caller speaks. Audio simulations only."`
	CallerStt    *string                `json:"caller_stt,omitempty" doc:"How the caller hears the agent. Audio simulations only."`
	CallerVoice  *string                `json:"caller_voice,omitempty" doc:"The voice the caller speaks in. Audio simulations only."`
	MaxTurns     *int                   `json:"max_turns,omitempty" doc:"How many times the caller may speak, up to two hundred. It is what stops a caller that never decides it is finished. Twelve when left out."`
	Tags         *map[string]string     `json:"tags,omitempty"`
}

type SimulationRunMode string

const (
	SimulationRunModeAudio SimulationRunMode = "audio"
	SimulationRunModeText  SimulationRunMode = "text"
)

// Valid indicates whether the value is a known member of the SimulationRunMode enum.
func (e SimulationRunMode) Valid() bool {
	switch e {
	case SimulationRunModeAudio:
		return true
	case SimulationRunModeText:
		return true
	default:
		return false
	}
}

type SimulationRunState string

const (
	SimulationRunStateCancelled SimulationRunState = "cancelled"
	SimulationRunStateErrored   SimulationRunState = "errored"
	SimulationRunStateFailed    SimulationRunState = "failed"
	SimulationRunStatePassed    SimulationRunState = "passed"
	SimulationRunStateRunning   SimulationRunState = "running"
)

// Valid indicates whether the value is a known member of the SimulationRunState enum.
func (e SimulationRunState) Valid() bool {
	switch e {
	case SimulationRunStateCancelled:
		return true
	case SimulationRunStateErrored:
		return true
	case SimulationRunStateFailed:
		return true
	case SimulationRunStatePassed:
		return true
	case SimulationRunStateRunning:
		return true
	default:
		return false
	}
}

type SimulationRun struct {
	Id            string             `json:"id"`
	SimulationId  string             `json:"simulation_id"`
	State         SimulationRunState `json:"state" doc:"A run passed only if every one of its conversations did. A conversation that never got as far as a ruling leaves the run errored rather than failed." enum:"running,passed,failed,cancelled,errored"`
	Cases         int                `json:"cases" doc:"How many conversations this run is having."`
	Passed        int                `json:"passed"`
	Failed        int                `json:"failed"`
	Mode          *SimulationRunMode `json:"mode,omitempty" enum:"text,audio"`
	ConfigId      *string            `json:"config_id,omitempty"`
	Scenario      *string            `json:"scenario,omitempty"`
	Assertion     *string            `json:"assertion,omitempty" doc:"What was asked of this run, copied when it started. Editing a simulation does not rewrite what an old run tested."`
	JudgeTarget   *string            `json:"judge_target,omitempty"`
	Error         *string            `json:"error,omitempty"`
	StartedAt     time.Time          `json:"started_at"`
	FinishedAt    *time.Time         `json:"finished_at,omitempty"`
	Conversations *[]SimulationCase  `json:"conversations,omitempty" doc:"The conversations this run had. Present when one run is asked for, and left out of a list so that reading the log does not mean reading every transcript."`
}

type SimulationCaseEnded string

const (
	SimulationCaseEndedComplete SimulationCaseEnded = "complete"
	SimulationCaseEndedFailed   SimulationCaseEnded = "failed"
	SimulationCaseEndedTimeout  SimulationCaseEnded = "timeout"
	SimulationCaseEndedTurns    SimulationCaseEnded = "turns"
)

// Valid indicates whether the value is a known member of the SimulationCaseEnded enum.
func (e SimulationCaseEnded) Valid() bool {
	switch e {
	case SimulationCaseEndedComplete:
		return true
	case SimulationCaseEndedFailed:
		return true
	case SimulationCaseEndedTimeout:
		return true
	case SimulationCaseEndedTurns:
		return true
	default:
		return false
	}
}

type SimulationCaseState string

const (
	SimulationCaseStateCancelled SimulationCaseState = "cancelled"
	SimulationCaseStateErrored   SimulationCaseState = "errored"
	SimulationCaseStateFailed    SimulationCaseState = "failed"
	SimulationCaseStatePassed    SimulationCaseState = "passed"
	SimulationCaseStatePending   SimulationCaseState = "pending"
	SimulationCaseStateRunning   SimulationCaseState = "running"
)

// Valid indicates whether the value is a known member of the SimulationCaseState enum.
func (e SimulationCaseState) Valid() bool {
	switch e {
	case SimulationCaseStateCancelled:
		return true
	case SimulationCaseStateErrored:
		return true
	case SimulationCaseStateFailed:
		return true
	case SimulationCaseStatePassed:
		return true
	case SimulationCaseStatePending:
		return true
	case SimulationCaseStateRunning:
		return true
	default:
		return false
	}
}

type SimulationCase struct {
	Id         string               `json:"id"`
	Variation  int                  `json:"variation" doc:"Which way of asking this was, and the order they are listed in."`
	Scenario   string               `json:"scenario" doc:"The wording this conversation used."`
	State      SimulationCaseState  `json:"state" enum:"pending,running,passed,failed,errored,cancelled"`
	CallId     *string              `json:"call_id,omitempty" doc:"The session that held it, which is what the call and transcript paths take. It is written as soon as it exists, so a conversation still going can be watched."`
	Transcript *[]SimulationLine    `json:"transcript,omitempty"`
	Turns      int                  `json:"turns" doc:"How many times the caller spoke."`
	Passed     *bool                `json:"passed,omitempty" doc:"The judge's ruling. Absent when it never got as far as ruling, which is not the same as having ruled against."`
	Verdict    *string              `json:"verdict,omitempty" doc:"What in the conversation decided it."`
	Score      *int                 `json:"score,omitempty" doc:"How sure the judge was, from 1 to 5."`
	Ended      *SimulationCaseEnded `json:"ended,omitempty" doc:"Why the conversation stopped." enum:"complete,turns,timeout,failed"`
	Error      *string              `json:"error,omitempty"`
	StartedAt  time.Time            `json:"started_at"`
	FinishedAt *time.Time           `json:"finished_at,omitempty"`
}

type SimulationLine struct {
	Caller   bool       `json:"caller" doc:"True when the simulated caller said it rather than the agent."`
	Text     string     `json:"text"`
	Intended *string    `json:"intended,omitempty" doc:"What the agent meant to say, where that differs from what the caller heard. Only an audio simulation has both, and the difference is what running one is for."`
	At       *time.Time `json:"at,omitempty"`
}

type simulationListResponse struct {
	Body []Simulation
}

type createSimulationRequest struct {
	Body SimulationRequest
}

type simulationResponse struct {
	Body Simulation
}

type getSimulationRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type updateSimulationRequest struct {
	ID   string `path:"id" doc:"The resource, as returned when it was created."`
	Body SimulationRequest
}

type deleteSimulationRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type runSimulationRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type simulationRunResponse struct {
	Body SimulationRun
}

type listSimulationRunsRequest struct {
	SimulationID optionalParam[string] `query:"simulation_id"`
	State        optionalParam[string] `query:"state" enum:"running,passed,failed,cancelled,errored"`
	Limit        optionalParam[int]    `query:"limit"`
}

type simulationRunListResponse struct {
	Body []SimulationRun
}

type getSimulationRunRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type cancelSimulationRunRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

func (s *Server) registerSimulations(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listSimulations",
		Method:      http.MethodGet,
		Path:        "/v1/agents/simulations",
		Summary:     "The simulations the calling customer has",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's simulations, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listSimulations)
	huma.Register(api, huma.Operation{
		OperationID: "createSimulation",
		Method:      http.MethodPost,
		Path:        "/v1/agents/simulations",
		Summary:     "Write down a conversation to have with an agent",
		Description: "A simulation is a scenario to put an agent through and something that has to be " +
			"true at the end of it. The scenario is a brief rather than a script: it is " +
			"given to a model that plays the caller, reads what the agent says back and " +
			"decides what to say next, so a scenario can say to change your mind once the " +
			"order is handled. Nothing is run until it is.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The simulation was stored"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.createSimulation)
	huma.Register(api, huma.Operation{
		OperationID: "getSimulation",
		Method:      http.MethodGet,
		Path:        "/v1/agents/simulations/{id}",
		Summary:     "One simulation",
		Responses: map[string]*huma.Response{
			"200": {Description: "The simulation"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getSimulation)
	huma.Register(api, huma.Operation{
		OperationID: "updateSimulation",
		Method:      http.MethodPut,
		Path:        "/v1/agents/simulations/{id}",
		Summary:     "Replace a simulation",
		Description: "Every field is written, so the body is what the simulation now asks rather than " +
			"what changed about it. The runs it already has keep their own copy of what they " +
			"tested, so an old result still says what it was a result of.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The simulation as it now is"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.updateSimulation)
	huma.Register(api, huma.Operation{
		OperationID: "deleteSimulation",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/simulations/{id}",
		Summary:     "Delete a simulation",
		Description: "The runs that named it are kept, so the simulation stops being runnable rather " +
			"than stops having existed.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"204": {Description: "The simulation is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteSimulation)
	huma.Register(api, huma.Operation{
		OperationID: "runSimulation",
		Method:      http.MethodPost,
		Path:        "/v1/agents/simulations/{id}/run",
		Summary:     "Have the conversations",
		Description: "Returns once the run is written rather than once it is over: the conversations " +
			"happen after the answer, and the run says how far along they are. A simulation " +
			"asking several ways has them all at once.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		DefaultStatus: http.StatusAccepted,
		Responses: map[string]*huma.Response{
			"202": {Description: "The run is going"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.runSimulation)
	huma.Register(api, huma.Operation{
		OperationID: "listSimulationRuns",
		Method:      http.MethodGet,
		Path:        "/v1/agents/simulation-runs",
		Summary:     "What the simulations have come to, newest first",
		Description: "Without a simulation named this is the log of everything that has been run " +
			"lately, which is the same question as what one simulation has come to, asked of " +
			"all of them.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The runs, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listSimulationRuns)
	huma.Register(api, huma.Operation{
		OperationID: "getSimulationRun",
		Method:      http.MethodGet,
		Path:        "/v1/agents/simulation-runs/{id}",
		Summary:     "One run, with the conversations it had",
		Responses: map[string]*huma.Response{
			"200": {Description: "The run and its conversations"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getSimulationRun)
	huma.Register(api, huma.Operation{
		OperationID: "cancelSimulationRun",
		Method:      http.MethodPost,
		Path:        "/v1/agents/simulation-runs/{id}/cancel",
		Summary:     "Stop a run",
		Description: "Unlike a paused campaign the conversations in flight are ended too: there is " +
			"nobody on the other end of them to be hung up on.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The run is stopping"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.cancelSimulationRun)
}

// listSimulations returns the calling customer's simulations, newest first.
func (s *Server) listSimulations(ctx context.Context, _ *struct{}) (*simulationListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noSimulations)
	}

	stored, err := s.store.CustomerSimulations(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]Simulation, 0, len(stored))
	for _, simulation := range stored {
		listed = append(listed, simulationOf(simulation))
	}
	return &simulationListResponse{Body: listed}, nil
}

// createSimulation writes down a conversation to have with an agent.
func (s *Server) createSimulation(ctx context.Context, request *createSimulationRequest) (*simulationResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noSimulations)
	}

	simulation := storedSimulation(request.Body, customerID)
	if complaint, bad := simulationComplaint(ctx, s, customerID, simulation); bad {
		return nil, huma.Error400BadRequest(complaint)
	}
	if err := s.store.CreateSimulation(ctx, &simulation); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &simulationResponse{Body: simulationOf(simulation)}, nil
}

// getSimulation returns one simulation.
func (s *Server) getSimulation(ctx context.Context, request *getSimulationRequest) (*simulationResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noSimulations)
	}

	simulation, err := s.store.Simulation(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownSimulation)
	}
	return &simulationResponse{Body: simulationOf(simulation)}, nil
}

// updateSimulation replaces what a simulation asks. The runs it already has keep their own
// copy of what they tested.
func (s *Server) updateSimulation(ctx context.Context, request *updateSimulationRequest) (*simulationResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noSimulations)
	}

	existing, err := s.store.Simulation(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownSimulation)
	}

	simulation := storedSimulation(request.Body, customerID)
	simulation.ID = existing.ID
	if complaint, bad := simulationComplaint(ctx, s, customerID, simulation); bad {
		return nil, huma.Error400BadRequest(complaint)
	}
	if err := s.store.UpdateSimulation(ctx, &simulation); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	simulation.CreatedAt = existing.CreatedAt
	return &simulationResponse{Body: simulationOf(simulation)}, nil
}

// deleteSimulation stops a simulation being runnable. The runs that named it are kept.
func (s *Server) deleteSimulation(ctx context.Context, request *deleteSimulationRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noSimulations)
	}

	if err := s.store.DeleteSimulation(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error404NotFound(unknownSimulation)
	}
	return nil, nil
}

// runSimulation has the conversations. It returns once the run is written rather than once
// it is over.
func (s *Server) runSimulation(ctx context.Context, request *runSimulationRequest) (*simulationRunResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noSimulations)
	}
	if s.simulations == nil {
		return nil, huma.Error400BadRequest(cannotRunSimulations)
	}

	if _, err := s.store.Simulation(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error404NotFound(unknownSimulation)
	}
	run, err := s.simulations.Start(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &simulationRunResponse{Body: simulationRunOf(run, nil)}, nil
}

// listSimulationRuns returns what the simulations have come to, newest first.
func (s *Server) listSimulationRuns(ctx context.Context, request *listSimulationRunsRequest) (*simulationRunListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noSimulations)
	}

	filter := store.SimulationRunFilter{
		CustomerID:   customerID,
		SimulationID: request.SimulationID.Value,
		Limit:        request.Limit.Value,
	}
	if request.State.Set {
		filter.State = string(request.State.Value)
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
	return &simulationRunListResponse{Body: listed}, nil
}

// getSimulationRun returns one run and the conversations it had.
func (s *Server) getSimulationRun(ctx context.Context, request *getSimulationRunRequest) (*simulationRunResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noSimulations)
	}

	run, err := s.store.SimulationRun(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownSimulationRun)
	}
	cases, err := s.store.SimulationCases(ctx, run.ID)
	if err != nil {
		return nil, err
	}
	return &simulationRunResponse{Body: simulationRunOf(run, cases)}, nil
}

// cancelSimulationRun stops a run, including the conversations already going.
func (s *Server) cancelSimulationRun(ctx context.Context, request *cancelSimulationRunRequest) (*simulationRunResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noSimulations)
	}
	if s.simulations == nil {
		return nil, huma.Error400BadRequest(cannotRunSimulations)
	}

	run, err := s.simulations.Cancel(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownSimulationRun)
	}
	return &simulationRunResponse{Body: simulationRunOf(run, nil)}, nil
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
