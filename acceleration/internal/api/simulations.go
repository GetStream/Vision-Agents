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
var (
	noSimulations        = notConfigured("simulations are not available: this deployment has no database")
	cannotRunSimulations = notConfigured("simulations cannot be run here: this deployment has no sessions or model routing")
	unknownSimulation    = APIError{
		Type: ErrorTypeNotFound, Code: codeSimulationNotFound,
		Message: "no such simulation",
	}
	unknownSimulationRun = APIError{
		Type: ErrorTypeNotFound, Code: codeSimulationRunNotFound,
		Message: "no such simulation run",
	}
)

// listSimulations returns the calling customer's simulations, newest first.
func (s *Server) listSimulations(ctx context.Context, _ *listSimulationsRequest) (*listSimulationsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noSimulations
	}

	stored, err := s.store.CustomerSimulations(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]Simulation, 0, len(stored))
	for _, simulation := range stored {
		listed = append(listed, simulationOf(simulation))
	}
	return &listSimulationsResponse{Body: listed}, nil
}

// createSimulation writes down a conversation to have with an agent.
func (s *Server) createSimulation(ctx context.Context, request *createSimulationRequest) (*createSimulationResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noSimulations
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}

	simulation := storedSimulation(*request.Body, customerID)
	if failure := simulationComplaint(ctx, s, customerID, simulation); failure != nil {
		return nil, failure
	}
	if err := s.store.CreateSimulation(ctx, &simulation); err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &createSimulationResponse{Body: simulationOf(simulation)}, nil
}

// getSimulation returns one simulation.
func (s *Server) getSimulation(ctx context.Context, request *getSimulationRequest) (*getSimulationResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noSimulations
	}

	simulation, err := s.store.Simulation(ctx, customerID, request.Id)
	if err != nil {
		return nil, unknownSimulation
	}
	return &getSimulationResponse{Body: simulationOf(simulation)}, nil
}

// updateSimulation replaces what a simulation asks. The runs it already has keep their own
// copy of what they tested.
func (s *Server) updateSimulation(ctx context.Context, request *updateSimulationRequest) (*updateSimulationResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noSimulations
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}

	existing, err := s.store.Simulation(ctx, customerID, request.Id)
	if err != nil {
		return nil, unknownSimulation
	}

	simulation := storedSimulation(*request.Body, customerID)
	simulation.ID = existing.ID
	if failure := simulationComplaint(ctx, s, customerID, simulation); failure != nil {
		return nil, failure
	}
	if err := s.store.UpdateSimulation(ctx, &simulation); err != nil {
		return nil, invalidRequest(err.Error())
	}

	simulation.CreatedAt = existing.CreatedAt
	return &updateSimulationResponse{Body: simulationOf(simulation)}, nil
}

// deleteSimulation stops a simulation being runnable. The runs that named it are kept.
func (s *Server) deleteSimulation(ctx context.Context, request *deleteSimulationRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noSimulations
	}

	if err := s.store.DeleteSimulation(ctx, customerID, request.Id); err != nil {
		return nil, unknownSimulation
	}
	return nil, nil
}

// runSimulation has the conversations. It returns once the run is written rather than once
// it is over.
func (s *Server) runSimulation(ctx context.Context, request *runSimulationRequest) (*runSimulationResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noSimulations
	}
	if s.simulations == nil {
		return nil, cannotRunSimulations
	}

	if _, err := s.store.Simulation(ctx, customerID, request.Id); err != nil {
		return nil, unknownSimulation
	}
	run, err := s.simulations.Start(ctx, customerID, request.Id)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	return &runSimulationResponse{Body: simulationRunOf(run, nil)}, nil
}

// listSimulationRuns returns what the simulations have come to, newest first.
func (s *Server) listSimulationRuns(ctx context.Context, request *listSimulationRunsRequest) (*listSimulationRunsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noSimulations
	}

	filter := store.SimulationRunFilter{
		CustomerID:   customerID,
		SimulationID: value(request.SimulationId.ptr()),
		Limit:        value(request.Limit.ptr()),
	}
	if request.State.ptr() != nil {
		filter.State = string(*request.State.ptr())
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
	return &listSimulationRunsResponse{Body: listed}, nil
}

// getSimulationRun returns one run and the conversations it had.
func (s *Server) getSimulationRun(ctx context.Context, request *getSimulationRunRequest) (*getSimulationRunResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noSimulations
	}

	run, err := s.store.SimulationRun(ctx, customerID, request.Id)
	if err != nil {
		return nil, unknownSimulationRun
	}
	cases, err := s.store.SimulationCases(ctx, run.ID)
	if err != nil {
		return nil, err
	}
	return &getSimulationRunResponse{Body: simulationRunOf(run, cases)}, nil
}

// cancelSimulationRun stops a run, including the conversations already going.
func (s *Server) cancelSimulationRun(ctx context.Context, request *cancelSimulationRunRequest) (*cancelSimulationRunResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, missingCustomer
	}
	if s.store == nil {
		return nil, noSimulations
	}
	if s.simulations == nil {
		return nil, cannotRunSimulations
	}

	run, err := s.simulations.Cancel(ctx, customerID, request.Id)
	if err != nil {
		return nil, unknownSimulationRun
	}
	return &cancelSimulationRunResponse{Body: simulationRunOf(run, nil)}, nil
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
) error {
	switch {
	case simulation.Name == "":
		return invalidRequest("a simulation needs a name")
	case simulation.ConfigID == "":
		return invalidRequest("a simulation needs an agent to run against")
	case simulation.Scenario == "":
		return invalidRequest("a simulation needs something to ask")
	case simulation.Assertion == "":
		return invalidRequest("a simulation needs something to check")
	}
	if err := routing.Tags(simulation.Tags).Validate(); err != nil {
		return invalidRequest(err.Error())
	}
	if _, err := server.store.AgentConfig(ctx, customerID, simulation.ConfigID); err != nil {
		return unknownConfig
	}
	return nil
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

// registerSimulations declares the operations served in simulations.go.
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
		Description: "A simulation is a scenario to put an agent through and something that has to be true at " +
			"the end of it. The scenario is a brief rather than a script: it is given to a model " +
			"that plays the caller, reads what the agent says back and decides what to say next, so " +
			"a scenario can say to change your mind once the order is handled. Nothing is run until " +
			"it is.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The simulation was stored"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
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
		Description: "Every field is written, so the body is what the simulation now asks rather than what " +
			"changed about it. The runs it already has keep their own copy of what they tested, so " +
			"an old result still says what it was a result of.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
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
		Description: "The runs that named it are kept, so the simulation stops being runnable rather than " +
			"stops having existed.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusNoContent,
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
		Description: "Returns once the run is written rather than once it is over: the conversations happen " +
			"after the answer, and the run says how far along they are. A simulation asking several " +
			"ways has them all at once.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
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
		Description: "Without a simulation named this is the log of everything that has been run lately, " +
			"which is the same question as what one simulation has come to, asked of all of them.",
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
		Description: "Unlike a paused campaign the conversations in flight are ended too: there is nobody on " +
			"the other end of them to be hung up on.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The run is stopping"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.cancelSimulationRun)
}

type listSimulationsRequest struct{}

type listSimulationsResponse struct {
	Body []Simulation `nullable:"false"`
}

type createSimulationRequest struct {
	Body *SimulationRequest `required:"true"`
}

type createSimulationResponse struct {
	Body Simulation
}

type getSimulationRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getSimulationResponse struct {
	Body Simulation
}

type updateSimulationRequest struct {
	Id   string             `path:"id" doc:"The resource, as returned when it was created."`
	Body *SimulationRequest `required:"true"`
}

type updateSimulationResponse struct {
	Body Simulation
}

type deleteSimulationRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type runSimulationRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type runSimulationResponse struct {
	Body SimulationRun
}

type listSimulationRunsRequest struct {
	SimulationId optionalParam[string]                        `query:"simulation_id"`
	State        optionalParam[ListSimulationRunsParamsState] `query:"state" enum:"running,passed,failed,cancelled,errored"`
	Limit        optionalParam[int]                           `query:"limit"`
}

type listSimulationRunsResponse struct {
	Body []SimulationRun `nullable:"false"`
}

type getSimulationRunRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getSimulationRunResponse struct {
	Body SimulationRun
}

type cancelSimulationRunRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type cancelSimulationRunResponse struct {
	Body SimulationRun
}

// SimulationMode is the SimulationMode schema.
type SimulationMode string

// Defines values for SimulationMode.
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

// SimulationCase is the SimulationCase schema.
type SimulationCase struct {
	CallId     *string              `json:"call_id,omitempty" doc:"The session that held it, which is what the call and transcript paths take. It is written as soon as it exists, so a conversation still going can be watched."`
	Ended      *SimulationCaseEnded `json:"ended,omitempty" doc:"Why the conversation stopped." enum:"complete,turns,timeout,failed"`
	Error      *string              `json:"error,omitempty"`
	FinishedAt *time.Time           `json:"finished_at,omitempty"`
	Id         string               `json:"id"`
	Passed     *bool                `json:"passed,omitempty" doc:"The judge's ruling. Absent when it never got as far as ruling, which is not the same as having ruled against."`
	Scenario   string               `json:"scenario" doc:"The wording this conversation used."`
	Score      *int                 `json:"score,omitempty" doc:"How sure the judge was, from 1 to 5."`
	StartedAt  time.Time            `json:"started_at"`
	State      SimulationCaseState  `json:"state" enum:"pending,running,passed,failed,errored,cancelled"`
	Transcript *[]SimulationLine    `json:"transcript,omitempty"`
	Turns      int                  `json:"turns" doc:"How many times the caller spoke."`
	Variation  int                  `json:"variation" doc:"Which way of asking this was, and the order they are listed in."`
	Verdict    *string              `json:"verdict,omitempty" doc:"What in the conversation decided it."`
}

func (*SimulationCase) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["score"].Format = ""
	schema.Properties["turns"].Format = ""
	schema.Properties["variation"].Format = ""
	return schema
}

// SimulationCaseEnded is the SimulationCaseEnded schema.
type SimulationCaseEnded string

// Defines values for SimulationCaseEnded.
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

// SimulationCaseState is the SimulationCaseState schema.
type SimulationCaseState string

// Defines values for SimulationCaseState.
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

// SimulationLine is the SimulationLine schema.
type SimulationLine struct {
	At       *time.Time `json:"at,omitempty"`
	Caller   bool       `json:"caller" doc:"True when the simulated caller said it rather than the agent."`
	Intended *string    `json:"intended,omitempty" doc:"What the agent meant to say, where that differs from what the caller heard. Only an audio simulation has both, and the difference is what running one is for."`
	Text     string     `json:"text"`
}

// SimulationRun is the SimulationRun schema.
type SimulationRun struct {
	Assertion     *string            `json:"assertion,omitempty" doc:"What was asked of this run, copied when it started. Editing a simulation does not rewrite what an old run tested."`
	Cases         int                `json:"cases" doc:"How many conversations this run is having."`
	ConfigId      *string            `json:"config_id,omitempty"`
	Conversations *[]SimulationCase  `json:"conversations,omitempty" doc:"The conversations this run had. Present when one run is asked for, and left out of a list so that reading the log does not mean reading every transcript."`
	Error         *string            `json:"error,omitempty"`
	Failed        int                `json:"failed"`
	FinishedAt    *time.Time         `json:"finished_at,omitempty"`
	Id            string             `json:"id"`
	JudgeTarget   *string            `json:"judge_target,omitempty"`
	Mode          *SimulationRunMode `json:"mode,omitempty" enum:"text,audio"`
	Passed        int                `json:"passed"`
	Scenario      *string            `json:"scenario,omitempty"`
	SimulationId  string             `json:"simulation_id"`
	StartedAt     time.Time          `json:"started_at"`
	State         SimulationRunState `json:"state" doc:"A run passed only if every one of its conversations did. A conversation that never got as far as a ruling leaves the run errored rather than failed." enum:"running,passed,failed,cancelled,errored"`
}

func (*SimulationRun) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["cases"].Format = ""
	schema.Properties["failed"].Format = ""
	schema.Properties["passed"].Format = ""
	return schema
}

// SimulationRunMode is the SimulationRunMode schema.
type SimulationRunMode string

// Defines values for SimulationRunMode.
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

// SimulationRunState is the SimulationRunState schema.
type SimulationRunState string

// Defines values for SimulationRunState.
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

// ListSimulationRunsParamsState is the ListSimulationRunsParamsState schema.
type ListSimulationRunsParamsState string

// Defines values for ListSimulationRunsParamsState.
const (
	ListSimulationRunsParamsStateCancelled ListSimulationRunsParamsState = "cancelled"
	ListSimulationRunsParamsStateErrored   ListSimulationRunsParamsState = "errored"
	ListSimulationRunsParamsStateFailed    ListSimulationRunsParamsState = "failed"
	ListSimulationRunsParamsStatePassed    ListSimulationRunsParamsState = "passed"
	ListSimulationRunsParamsStateRunning   ListSimulationRunsParamsState = "running"
)

// Valid indicates whether the value is a known member of the ListSimulationRunsParamsState enum.
func (e ListSimulationRunsParamsState) Valid() bool {
	switch e {
	case ListSimulationRunsParamsStateCancelled:
		return true
	case ListSimulationRunsParamsStateErrored:
		return true
	case ListSimulationRunsParamsStateFailed:
		return true
	case ListSimulationRunsParamsStatePassed:
		return true
	case ListSimulationRunsParamsStateRunning:
		return true
	default:
		return false
	}
}
