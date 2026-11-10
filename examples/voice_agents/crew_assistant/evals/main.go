// Command evals runs the crew assistant's simulations on the router and says how each went.
//
// A simulation is a conversation the router holds with the agent, with a model playing the
// team member and another judging the result against simulations/*.yaml. The agent's tools
// live on the phone, which a simulation has none of, so this hosts the same tools under the
// agent's name while the simulations run, each calling the backend the way the phone does. A
// phone declares its own tools, which win, so it isn't affected.
//
// Run configure first, so the router has the simulations, and the backend, ideally one of its
// own so the swaps and tickets the simulations make stay out of the demo's team lead view:
//
//	cd backend && GOWORK=off go run . -addr :8096
//	cd evals && GOWORK=off go run . -backend http://localhost:8096
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"net/url"
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
	only := flag.String("only", "", "run only the simulations whose name contains this")
	transcripts := flag.Bool("transcripts", false, "print the transcript of every conversation that didn't pass")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := run(ctx, *agent, *backend, *only, *transcripts); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, agentName, backendURL, only string, transcripts bool) error {
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
	phone, err := phoneTools(crewBackend(strings.TrimSuffix(backendURL, "/")))
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

	fmt.Printf("running %d simulations of %s, with its tools calling %s\n\n", len(simulations), agentName, backendURL)
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

// crewBackend asks the crew backend, as the phone's tools do.
type crewBackend string

func (b crewBackend) get(ctx context.Context, path string, query url.Values) (any, error) {
	if len(query) > 0 {
		path += "?" + query.Encode()
	}
	return b.do(ctx, http.MethodGet, path, nil)
}

func (b crewBackend) post(ctx context.Context, path string, body any) (any, error) {
	return b.do(ctx, http.MethodPost, path, body)
}

func (b crewBackend) do(ctx context.Context, method, path string, body any) (any, error) {
	var sent io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		sent = strings.NewReader(string(encoded))
	}
	request, err := http.NewRequestWithContext(ctx, method, string(b)+"/crew/"+path, sent)
	if err != nil {
		return nil, err
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
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
	none         struct{}
	calloutCheck struct {
		ShiftID  string   `json:"shift_id" schema:"The shift, from get_my_schedule"`
		Symptoms []string `json:"symptoms" schema:"What they said they have, in a few words each, such as \"fever\""`
	}
	coverage struct {
		ShiftID string `json:"shift_id" schema:"The shift to cover"`
	}
	swap struct {
		Note       string `json:"note,omitempty" schema:"A short note to the teammate"`
		ShiftID    string `json:"shift_id" schema:"The shift to give away"`
		TeammateID string `json:"teammate_id" schema:"The teammate, from find_coverage"`
	}
	leaderNote struct {
		Message string `json:"message" schema:"The note, one or two sentences"`
		ShiftID string `json:"shift_id,omitempty" schema:"The shift"`
	}
	handbookQuery struct {
		Query string `json:"query" schema:"What to look for, in a few words"`
	}
	updatesSince struct {
		Since string `json:"since" schema:"The first day, as YYYY-MM-DD"`
	}
	equipment struct {
		Asset string `json:"asset" schema:"The asset id"`
		Code  string `json:"code,omitempty" schema:"The error code it shows, such as E-41"`
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
	reflection struct {
		Energy int      `json:"energy" schema:"From 1 to 5"`
		Themes []string `json:"themes"`
		Win    string   `json:"win,omitempty" schema:"Their win, in a short line"`
	}
)

// phoneTools are the tools the app declares, in agent/crewTools.ts of ExpoCrewAssistant, with
// the same names and descriptions. The ones that ask first on the phone run straight away here,
// since nobody is there to ask.
func phoneTools(crew crewBackend) (*tools.Registry, error) {
	registry := tools.NewRegistry()
	return registry, errors.Join(
		tools.Register(registry, "get_local_context",
			"The person's current date, local time, timezone and language. Call it before answering anything that depends on today or the time.",
			func(context.Context, none) (any, error) {
				now := time.Now()
				zone, _ := now.Zone()
				return map[string]string{"language": "en-US", "local": now.Format(time.RFC1123), "timeZone": zone, "utc": now.UTC().Format(time.RFC3339)}, nil
			}),
		tools.Register(registry, "get_my_schedule",
			"The person's upcoming shifts, each with its id, station, start, end and shift leader.",
			func(ctx context.Context, _ none) (any, error) { return crew.get(ctx, "schedule", nil) }),
		tools.Register(registry, "check_callout",
			"Whether the person may find their own cover for a shift, by the restaurant's call-out policy, given their symptoms. Call it before helping with coverage.",
			func(ctx context.Context, in calloutCheck) (any, error) { return crew.post(ctx, "callout-check", in) }),
		tools.Register(registry, "find_coverage",
			"Teammates trained for the shift’s station and free at that time.",
			func(ctx context.Context, in coverage) (any, error) {
				return crew.get(ctx, "coverage", url.Values{"shift_id": {in.ShiftID}})
			}),
		tools.Register(registry, "request_shift_swap",
			"Asks a teammate to take one of the person’s shifts. The phone asks the person first.",
			func(ctx context.Context, in swap) (any, error) { return crew.post(ctx, "swaps", in) }),
		tools.Register(registry, "notify_shift_leader",
			"Sends the shift's leader a short status note, such as who is out and whether cover was asked for.",
			func(ctx context.Context, in leaderNote) (any, error) { return crew.post(ctx, "leader-notes", in) }),
		tools.Register(registry, "search_handbook",
			"Searches the team handbook: policies, uniform, food safety, allergens, breaks, swaps, time off, meals and values. The passages it finds are the only source for answers about them, and show as the reply's sources.",
			func(ctx context.Context, in handbookQuery) (any, error) {
				return crew.get(ctx, "handbook", url.Values{"q": {in.Query}})
			}),
		tools.Register(registry, "get_updates_since",
			"Announcements and team news since a date.",
			func(ctx context.Context, in updatesSince) (any, error) {
				return crew.get(ctx, "updates", url.Values{"since": {in.Since}})
			}),
		tools.Register(registry, "get_equipment_guide",
			"The quick-fix guide for an error code on a piece of equipment, by its asset id, such as FRY-02, POS-1 or ICE-1.",
			func(ctx context.Context, in equipment) (any, error) {
				query := url.Values{"asset": {in.Asset}}
				if in.Code != "" {
					query.Set("code", in.Code)
				}
				return crew.get(ctx, "equipment", query)
			}),
		tools.Register(registry, "create_support_ticket",
			"Drafts a repair ticket for the owner to review and submit. The phone asks the person first.",
			func(ctx context.Context, in ticket) (any, error) { return crew.post(ctx, "tickets", in) }),
		tools.Register(registry, "get_team",
			"The leadership team, the trainer and the onboarding buddy, with what each does.",
			func(ctx context.Context, _ none) (any, error) { return crew.get(ctx, "team", nil) }),
		tools.Register(registry, "book_check_in",
			"Books a 1-on-1 check-in with a leader or trainer. The phone asks the person first.",
			func(ctx context.Context, in checkIn) (any, error) { return crew.post(ctx, "check-ins", in) }),
		tools.Register(registry, "get_menu",
			"The menu, with what each item contains and notes, for questions and quizzes.",
			func(ctx context.Context, _ none) (any, error) { return crew.get(ctx, "menu", nil) }),
		tools.Register(registry, "save_shift_reflection",
			"Saves the end-of-shift reflection. Only the themes, energy and win reach the leaders, never who wrote it.",
			func(ctx context.Context, in reflection) (any, error) { return crew.post(ctx, "reflections", in) }),
	)
}
