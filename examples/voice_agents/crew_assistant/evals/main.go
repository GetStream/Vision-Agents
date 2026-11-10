// Command evals runs the crew assistant's simulations on the router and says how each went.
//
// A simulation is a conversation the router holds with the agent, with a model playing the
// team member and another judging the result against simulations/*.yaml. The backend hosts the
// restaurant's tools on the router, simulations included. The phone's three tools that ask the
// person first have no phone in a simulation, so this hosts them under the agent's name while
// the simulations run, each calling the backend the way the phone does, as the user the backend
// acts for in simulations. A phone declares its own, which win, so it isn't affected.
//
// Run configure first, so the router has the simulations, and the backend with the user a
// simulation acts for. What the simulations write goes into that user's schedule and the
// leaders' inbox:
//
//	cd backend && GOWORK=off go run . -simulation-user crew-jordan
//	cd evals && GOWORK=off go run .
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// runsAtOnce is how many simulations run together, which is as many as the router holds for
// one process.
const runsAtOnce = 4

func main() {
	agent := flag.String("agent", "crew_assistant", "the agent to run the simulations of")
	backend := flag.String("backend", "http://localhost:8097", "the crew backend the tools call")
	user := flag.String("user", "crew-jordan", "who the tools call the backend as: the backend's -simulation-user")
	only := flag.String("only", "", "run only the simulations whose name contains this")
	transcripts := flag.Bool("transcripts", false, "print the transcript of every conversation that didn't pass")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *agent, *backend, *user, *only, *transcripts); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, agentName, backendURL, user, only string, transcripts bool) error {
	client, err := stream.Backend{}.Client()
	if err != nil {
		return err
	}
	simulations, err := simulationsOf(ctx, client, agentName, only)
	if err != nil {
		return err
	}

	dispatch, err := stream.NewDispatch(stream.DispatchOptions{
		Logger: slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn})),
	})
	if err != nil {
		return err
	}
	crew := &crewBackend{url: strings.TrimSuffix(backendURL, "/"), user: user}
	if err := crew.signIn(ctx); err != nil {
		return err
	}
	phone, err := phoneTools(crew)
	if err != nil {
		return err
	}
	dispatch.Host(hostedAgent{name: agentName, tools: phone}, 30*time.Second)
	hosting, stopHosting := context.WithCancel(ctx)
	defer stopHosting()
	hosted := make(chan error, 1)
	go func() { hosted <- dispatch.Run(hosting) }()
	if err := waitForWorker(ctx, dispatch, hosted); err != nil {
		return err
	}

	fmt.Printf("running %d simulations of %s, with the phone's tools calling %s as %s\n\n", len(simulations), agentName, backendURL, user)
	results := make([]acceleration.SimulationRun, len(simulations))
	failures := make([]error, len(simulations))
	slots := make(chan struct{}, runsAtOnce)
	var running sync.WaitGroup
	for i, simulation := range simulations {
		running.Add(1)
		go func() {
			defer running.Done()
			slots <- struct{}{}
			defer func() { <-slots }()
			results[i], failures[i] = runSimulation(ctx, client, simulation)
			report(simulation, results[i], failures[i], transcripts)
		}()
	}
	running.Wait()

	passed, held := 0, 0
	for i, result := range results {
		if failures[i] != nil {
			continue
		}
		passed += result.Passed
		held += result.Cases
	}
	fmt.Printf("\n%d of %d conversations passed\n", passed, held)
	if passed < held || errors.Join(failures...) != nil {
		os.Exit(1)
	}
	return nil
}

// simulationsOf are the agent's simulations whose name contains only.
func simulationsOf(ctx context.Context, client *acceleration.ClientWithResponses, agentName, only string) ([]acceleration.Simulation, error) {
	configs, err := client.ListAgentConfigsWithResponse(ctx, &acceleration.ListAgentConfigsParams{Name: &agentName})
	if err != nil {
		return nil, err
	}
	if configs.JSON200 == nil {
		return nil, fmt.Errorf("finding %s: %s %s", agentName, configs.Status(), configs.Body)
	}
	if len(*configs.JSON200) == 0 {
		return nil, fmt.Errorf("the router has no agent %s: run configure first", agentName)
	}
	configID := (*configs.JSON200)[0].Id

	listed, err := client.ListSimulationsWithResponse(ctx)
	if err != nil {
		return nil, err
	}
	if listed.JSON200 == nil {
		return nil, fmt.Errorf("listing the simulations: %s %s", listed.Status(), listed.Body)
	}
	simulations := []acceleration.Simulation{}
	for _, simulation := range *listed.JSON200 {
		if simulation.ConfigId == configID && strings.Contains(strings.ToLower(simulation.Name), strings.ToLower(only)) {
			simulations = append(simulations, simulation)
		}
	}
	if len(simulations) == 0 {
		return nil, fmt.Errorf("%s has no simulations matching %q: run configure first", agentName, only)
	}
	return simulations, nil
}

// waitForWorker waits for the router to have taken the hosted tools.
func waitForWorker(ctx context.Context, dispatch *stream.Dispatch, hosted <-chan error) error {
	deadline := time.After(15 * time.Second)
	for dispatch.WorkerID() == "" {
		select {
		case err := <-hosted:
			return fmt.Errorf("hosting the phone's tools: %w", err)
		case <-deadline:
			return errors.New("the router never took the phone's tools")
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
	select {
	case err := <-hosted:
		return fmt.Errorf("hosting the phone's tools: %w", err)
	case <-time.After(time.Second):
		return nil
	}
}

// runSimulation starts one simulation and waits for it to finish, for at most half an hour.
func runSimulation(ctx context.Context, client *acceleration.ClientWithResponses, simulation acceleration.Simulation) (acceleration.SimulationRun, error) {
	started, err := client.RunSimulationWithResponse(ctx, simulation.Id)
	if err != nil {
		return acceleration.SimulationRun{}, err
	}
	if started.JSON202 == nil {
		return acceleration.SimulationRun{}, fmt.Errorf("starting it: %s %s", started.Status(), started.Body)
	}
	deadline := time.Now().Add(30 * time.Minute)
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return acceleration.SimulationRun{}, ctx.Err()
		case <-time.After(3 * time.Second):
		}
		read, err := client.GetSimulationRunWithResponse(ctx, started.JSON202.Id)
		if err != nil {
			return acceleration.SimulationRun{}, err
		}
		if read.JSON200 == nil {
			return acceleration.SimulationRun{}, fmt.Errorf("reading it: %s %s", read.Status(), read.Body)
		}
		if read.JSON200.State != acceleration.SimulationRunStateRunning {
			return *read.JSON200, nil
		}
	}
	return acceleration.SimulationRun{}, errors.New("it didn't finish within half an hour")
}

// report prints how one simulation went: its score, and what the judge said of each
// conversation that didn't pass.
func report(simulation acceleration.Simulation, run acceleration.SimulationRun, failure error, transcripts bool) {
	var out strings.Builder
	if failure != nil {
		fmt.Fprintf(&out, "ERROR %s: %v\n", simulation.Name, failure)
		fmt.Print(out.String())
		return
	}
	mark := "PASS "
	if run.Passed < run.Cases {
		mark = "FAIL "
	}
	fmt.Fprintf(&out, "%s %d/%d  %s\n", mark, run.Passed, run.Cases, simulation.Name)
	if run.Error != nil {
		fmt.Fprintf(&out, "       %s\n", *run.Error)
	}
	if run.Conversations != nil {
		for _, conversation := range *run.Conversations {
			if conversation.State == acceleration.SimulationCaseStatePassed {
				continue
			}
			why := string(conversation.State)
			if conversation.Verdict != nil {
				why = *conversation.Verdict
			} else if conversation.Error != nil {
				why = *conversation.Error
			}
			fmt.Fprintf(&out, "       #%d %s\n", conversation.Variation+1, why)
			if transcripts && conversation.Transcript != nil {
				for _, line := range *conversation.Transcript {
					who := "crew  "
					if line.Caller {
						who = "caller"
					}
					fmt.Fprintf(&out, "         %s  %s\n", who, strings.ReplaceAll(line.Text, "\n", "\n                 "))
				}
			}
		}
	}
	fmt.Print(out.String())
}

// hostedAgent is the agent the tools are hosted under.
type hostedAgent struct {
	name  string
	tools *tools.Registry
}

func (a hostedAgent) Name() string           { return a.name }
func (a hostedAgent) Tools() *tools.Registry { return a.tools }

// crewBackend asks the crew backend, as the phone's tools do, with the user's Stream token.
type crewBackend struct {
	url   string
	user  string
	token string
}

// signIn gets the user's Stream token, as the phone does.
func (b *crewBackend) signIn(ctx context.Context) error {
	answered, err := b.do(ctx, "stream-token", map[string]string{"user_id": b.user})
	if err != nil {
		return fmt.Errorf("signing in as %s: %w", b.user, err)
	}
	token, _ := answered.(map[string]any)["token"].(string)
	if token == "" {
		return errors.New("the crew backend answered no token")
	}
	b.token = token
	return nil
}

func (b *crewBackend) do(ctx context.Context, path string, body any) (any, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, b.url+"/"+path, bytes.NewReader(encoded))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	if b.token != "" {
		request.Header.Set("Authorization", "Bearer "+b.token)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	read, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode >= 300 {
		return nil, fmt.Errorf("the crew backend answered %d: %s", response.StatusCode, strings.TrimSpace(string(read)))
	}
	var answered any
	if err := json.Unmarshal(read, &answered); err != nil {
		return nil, err
	}
	return answered, nil
}

type (
	swap struct {
		Note       string `json:"note,omitempty" schema:"A short note to the teammate"`
		ShiftID    string `json:"shift_id" schema:"The shift to give away"`
		TeammateID string `json:"teammate_id" schema:"The teammate, from find_coverage"`
	}
	ticket struct {
		Asset    string `json:"asset,omitempty" schema:"The asset id"`
		Details  string `json:"details" schema:"What they saw and already tried"`
		Priority string `json:"priority,omitempty" schema:"low, normal or urgent"`
		Summary  string `json:"summary" schema:"The problem, in a short line"`
	}
	checkIn struct {
		Topic string `json:"topic,omitempty" schema:"What it is about"`
		When  string `json:"when" schema:"When, such as \"Thursday 14:00\""`
		With  string `json:"with" schema:"Who, by id or name, from get_team"`
	}
)

// phoneTools are the phone's tools that ask the person first, in agent/crewTools.ts of
// ExpoCrewAssistant, with the same names and descriptions. They run straight away here, since
// nobody is there to ask.
func phoneTools(crew *crewBackend) (*tools.Registry, error) {
	registry := tools.NewRegistry()
	return registry, errors.Join(
		tools.Register(registry, "request_shift_swap",
			"Asks a teammate to take one of the person’s shifts. The phone asks the person first.",
			func(ctx context.Context, in swap) (any, error) { return crew.do(ctx, "crew/swaps", in) }),
		tools.Register(registry, "create_support_ticket",
			"Drafts a repair ticket for the owner to review and submit. The phone asks the person first.",
			func(ctx context.Context, in ticket) (any, error) { return crew.do(ctx, "crew/tickets", in) }),
		tools.Register(registry, "book_check_in",
			"Books a 1-on-1 check-in with a leader or trainer. The phone asks the person first.",
			func(ctx context.Context, in checkIn) (any, error) { return crew.do(ctx, "crew/check-ins", in) }),
	)
}
