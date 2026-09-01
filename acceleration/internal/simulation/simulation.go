// Package simulation puts an agent through a conversation somebody wrote down and rules on
// how it went.
//
// A simulation is the question "does this agent still do the thing" made runnable. What
// makes it more than a scripted exchange is that the caller is a model too: a scenario that
// says to change your mind once the order is handled only means anything to somebody
// reading the replies, so the caller reads them and decides what to say next. Ten
// variations of a scenario are ten of those conversations, had at once, and a third model
// rules on each against what the customer said had to be true.
//
// The conversations happen in this process, which is the same place sessions already live,
// and a run is claimed by the process that started it: a router that goes down mid-run
// abandons it rather than leaving it for another to pick up.
package simulation

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// What bounds a conversation, layered because each of them fails differently: a caller that
// never finishes, an agent that answers slower every turn, and an agent that stopped
// answering are three problems and only the last of them is quick to notice.
const (
	// replyWithin is how long one turn waits for the agent. Generous: a turn that looks
	// something up legitimately takes most of it.
	replyWithin = 60 * time.Second
	// caseTimeout bounds one whole conversation.
	caseTimeout = 5 * time.Minute
	// runTimeout bounds all of them together.
	runTimeout = 30 * time.Minute
	// writeTimeout bounds the writes a run makes on its way out, which have to happen on a
	// context that cancelling the run did not cancel.
	writeTimeout = 10 * time.Second
)

// How much of this a process does at once. A customer who presses Run on every simulation
// they have should be told to wait rather than open a hundred conversations, and a run of
// ten variations is ten agents talking at once if nothing says otherwise.
const (
	runsAtOnce  = 4
	casesAtOnce = 5
)

// Options configures a Runner. The first three are required: a simulation is a
// conversation, a model to judge it and a row, and it cannot be any of the three alone.
type Options struct {
	Store    *store.Store
	Sessions *session.Manager
	LLM      *llmrouter.Router
	// TTS and STT give the simulated caller a voice and ears. Absent means text
	// simulations only, and an audio one is refused rather than quietly held in writing.
	TTS    *ttsrouter.Router
	STT    *sttrouter.Router
	Logger *slog.Logger
}

// Runner works through the simulation runs that have been started.
type Runner struct {
	store    *store.Store
	sessions *session.Manager
	llm      *llmrouter.Router
	tts      *ttsrouter.Router
	stt      *sttrouter.Router
	logger   *slog.Logger

	mu sync.Mutex
	// running is how a run is cancelled: the entry is the way to stop its conversations.
	running map[string]context.CancelFunc
	closed  bool

	working sync.WaitGroup
}

// New validates the options and returns a Runner. It starts nothing.
func New(options Options) (*Runner, error) {
	if options.Store == nil {
		return nil, errors.New("simulation: a database is required")
	}
	if options.Sessions == nil {
		return nil, errors.New("simulation: a session manager is required")
	}
	if options.LLM == nil {
		return nil, errors.New("simulation: a model is required to play the caller and judge the call")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	return &Runner{
		store:    options.Store,
		sessions: options.Sessions,
		llm:      options.LLM,
		tts:      options.TTS,
		stt:      options.STT,
		logger:   options.Logger,
		running:  map[string]context.CancelFunc{},
	}, nil
}

// Abandon writes off the runs an older process left saying they were running. Nothing else
// will finish them: the conversations were held in that process, and it is gone.
func (r *Runner) Abandon(ctx context.Context) error {
	return r.store.AbandonSimulationRuns(ctx, time.Now().UTC())
}

// Start has the conversations a simulation asks for and returns the run they happen under.
// It returns as soon as the run is written: the conversations outlive the request.
func (r *Runner) Start(ctx context.Context, customerID, id string) (store.SimulationRun, error) {
	simulation, err := r.store.Simulation(ctx, customerID, id)
	if err != nil {
		return store.SimulationRun{}, err
	}
	if simulation.Mode == store.SimulationAudio && (r.tts == nil || r.stt == nil) {
		return store.SimulationRun{}, errors.New(
			"simulation: this deployment cannot give the caller a voice or ears, so it cannot run an audio simulation")
	}
	// The config is read once, here, rather than per conversation: a run is one agent
	// asked the same thing several ways, and editing it halfway through would mean the
	// variations were not comparable.
	config, err := r.store.AgentConfig(ctx, customerID, simulation.ConfigID)
	if err != nil {
		return store.SimulationRun{}, err
	}

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return store.SimulationRun{}, errors.New("simulation: the runner is shut down")
	}
	if len(r.running) >= runsAtOnce {
		r.mu.Unlock()
		return store.SimulationRun{}, fmt.Errorf(
			"simulation: %d runs are already going, so this one would have to wait", runsAtOnce)
	}
	r.mu.Unlock()

	scenarios := r.scenarios(ctx, simulation)

	run := store.SimulationRun{
		CustomerID:   customerID,
		SimulationID: simulation.ID,
		Mode:         simulation.Mode,
		ConfigID:     simulation.ConfigID,
		Scenario:     simulation.Scenario,
		Assertion:    simulation.Assertion,
		JudgeTarget:  simulation.JudgeTarget,
	}
	cases := make([]store.SimulationCase, 0, len(scenarios))
	for i, scenario := range scenarios {
		cases = append(cases, store.SimulationCase{Variation: i, Scenario: scenario})
	}
	if err := r.store.StartSimulationRun(ctx, &run, cases); err != nil {
		return store.SimulationRun{}, err
	}

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return store.SimulationRun{}, errors.New("simulation: the runner is shut down")
	}
	// The run outlives the request that started it, so it takes the background rather than
	// a context that is cancelled the moment the caller is answered.
	loop, cancel := context.WithCancel(context.WithoutCancel(ctx))
	r.running[run.ID] = cancel
	r.mu.Unlock()

	r.working.Add(1)
	go func() {
		defer r.working.Done()
		defer cancel()
		r.run(loop, simulation, config, run, cases)
	}()
	return run, nil
}

// Cancel stops a run. Unlike a paused campaign the conversations in flight are ended too:
// there is nobody on the other end of them to be hung up on.
func (r *Runner) Cancel(ctx context.Context, customerID, id string) (store.SimulationRun, error) {
	run, err := r.store.SimulationRun(ctx, customerID, id)
	if err != nil {
		return store.SimulationRun{}, err
	}

	r.forget(run.ID)
	return run, nil
}

// Close stops every run and waits for the conversations to end.
func (r *Runner) Close() {
	r.mu.Lock()
	r.closed = true
	for id, cancel := range r.running {
		cancel()
		delete(r.running, id)
	}
	r.mu.Unlock()

	r.working.Wait()
}

// scenarios is the ways this run will ask. The scenario as written is always the first of
// them, so a run that asks ten ways still asks the one the customer actually wrote.
//
// Failing to think of the other nine is not a failed run: it is a run of one, which is what
// a simulation without variations is anyway.
func (r *Runner) scenarios(ctx context.Context, simulation store.Simulation) []string {
	scenarios := []string{simulation.Scenario}
	if simulation.Variations <= 1 {
		return scenarios
	}

	rewrites, err := expand(ctx, r.llm, llmrouter.Request{
		CustomerID: simulation.CustomerID,
		Tags:       simulation.Tags,
		Target:     simulation.CallerTarget,
	}, "expand-"+simulation.ID, simulation.Scenario, simulation.Variations-1)
	if err != nil {
		r.logger.Error("could not think of other ways to ask",
			"simulation", simulation.ID, "error", err)
		return scenarios
	}
	return append(scenarios, rewrites...)
}

// run has a simulation's conversations, holding at most its variations at once.
func (r *Runner) run(
	ctx context.Context,
	simulation store.Simulation,
	config store.AgentConfig,
	run store.SimulationRun,
	cases []store.SimulationCase,
) {
	ctx, cancel := context.WithTimeout(ctx, runTimeout)
	defer cancel()
	defer r.forget(run.ID)

	slots := make(chan struct{}, min(len(cases), casesAtOnce))
	var held sync.WaitGroup
	var tally sync.Mutex

	for i, kase := range cases {
		select {
		case slots <- struct{}{}:
		case <-ctx.Done():
			// Nobody is going to have the conversations that had not started, so they are
			// written off here rather than left saying they are about to happen.
			for _, waiting := range cases[i:] {
				waiting.State = store.SimulationCancelled
				r.finish(ctx, waiting)
			}
			held.Wait()
			run.State = ended(ctx, run)
			r.settle(ctx, run)
			return
		}

		held.Add(1)
		go func() {
			defer held.Done()
			defer func() { <-slots }()

			finished := r.play(ctx, simulation, config, kase)
			r.finish(ctx, finished)

			tally.Lock()
			switch finished.State {
			case store.SimulationPassed:
				run.Passed++
			case store.SimulationFailed:
				run.Failed++
			}
			tally.Unlock()
		}()
	}
	held.Wait()

	run.State = ended(ctx, run)
	r.settle(ctx, run)
}

// play holds one conversation and records what the judge made of it.
func (r *Runner) play(
	ctx context.Context,
	simulation store.Simulation,
	config store.AgentConfig,
	kase store.SimulationCase,
) store.SimulationCase {
	ctx, cancel := context.WithTimeout(ctx, caseTimeout)
	defer cancel()

	owner := llmrouter.Request{
		CustomerID: simulation.CustomerID,
		AgentID:    agentID(kase.ID),
		Tags:       tag(simulation.Tags, simulation.ID),
	}

	persona, err := newCaller(ctx, r.llm, llmrouter.Request{
		CustomerID: owner.CustomerID,
		AgentID:    owner.AgentID,
		Tags:       owner.Tags,
		Target:     simulation.CallerTarget,
	}, kase.Scenario)
	if err != nil {
		return errored(kase, err)
	}
	defer persona.Close()

	over, err := r.hold(ctx, simulation, config, kase)
	if err != nil {
		return errored(kase, err)
	}
	defer over.Close()

	kase.CallID = over.Session().ID()
	if err := r.store.StartSimulationCase(ctx, kase.ID, kase.CallID); err != nil {
		r.logger.Error("could not record which call a conversation became",
			"case", kase.ID, "error", err)
	}
	defer r.hang(over)

	so, why, err := exchange(ctx, persona, over, turnsOf(simulation))
	kase.Transcript = so
	kase.Turns = so.turns()
	kase.Ended = why
	if err != nil {
		// The conversation is still judged on what was said before it stopped. A run that
		// ran out of turns is a failure with evidence, and only a run with no evidence at
		// all is an error.
		kase.Error = err.Error()
		if len(so) == 0 {
			return errored(kase, err)
		}
	}

	ruled, err := rule(ctx, r.llm, llmrouter.Request{
		CustomerID: owner.CustomerID,
		AgentID:    owner.AgentID,
		Tags:       owner.Tags,
		Target:     simulation.JudgeTarget,
	}, "judge-"+kase.ID, simulation.Assertion, so)
	if err != nil {
		return errored(kase, err)
	}

	kase.State = store.SimulationFailed
	if ruled.Passed {
		kase.State = store.SimulationPassed
	}
	kase.Passed = &ruled.Passed
	kase.Verdict = ruled.Reason
	if ruled.Score >= 1 && ruled.Score <= 5 {
		kase.Score = &ruled.Score
	}
	return kase
}

// hold opens the conversation, out loud or in writing as the simulation asks.
func (r *Runner) hold(
	ctx context.Context,
	simulation store.Simulation,
	config store.AgentConfig,
	kase store.SimulationCase,
) (transport, error) {
	spec := r.spec(simulation, config, kase)
	if simulation.Mode == store.SimulationAudio {
		return r.converse(ctx, spec, simulation, kase)
	}
	return r.speak(ctx, spec)
}

// spec is the agent under test, as the session manager wants it.
func (r *Runner) spec(
	simulation store.Simulation,
	config store.AgentConfig,
	kase store.SimulationCase,
) session.Spec {
	spec := session.FromConfig(config)
	spec.CustomerID = simulation.CustomerID
	spec.AgentID = agentID(kase.ID)
	if simulation.Mode == store.SimulationAudio {
		// A conversation out loud is a call, even though the call is a loopback with no
		// network in it, and Normalize insists on one having an id.
		spec.Text = false
		spec.CallID = agentID(kase.ID)
	} else {
		spec.Text = true
		// A text session holds no call, and Normalize says so rather than ignoring one.
		spec.CallID = ""
	}
	spec.Tags = tag(spec.Tags, simulation.ID)
	// The judge has already ruled on this conversation, so paying a second model to
	// summarise it afterwards buys nothing. On a run of ten that is ten completions.
	spec.NoReview = true
	return spec
}

// hang ends a conversation through the manager rather than the session, which takes it off
// the list of sessions as well as ending it.
func (r *Runner) hang(over transport) {
	created := over.Session()
	if _, err := r.sessions.Close(created.ID(), created.Spec().CustomerID); err != nil {
		r.logger.Error("could not end a simulated conversation",
			"session", created.ID(), "error", err)
	}
}

// finish records what became of one conversation, off the context the run was cancelled
// with: a stopped run must still write down what it heard.
func (r *Runner) finish(ctx context.Context, kase store.SimulationCase) {
	written, cancel := context.WithTimeout(context.WithoutCancel(ctx), writeTimeout)
	defer cancel()

	if err := r.store.FinishSimulationCase(written, kase); err != nil {
		r.logger.Error("could not record how a conversation went", "case", kase.ID, "error", err)
	}
}

// settle records how the run as a whole came out.
func (r *Runner) settle(ctx context.Context, run store.SimulationRun) {
	written, cancel := context.WithTimeout(context.WithoutCancel(ctx), writeTimeout)
	defer cancel()

	if err := r.store.FinishSimulationRun(written, run); err != nil {
		r.logger.Error("could not record how a run ended", "run", run.ID, "error", err)
	}
}

func (r *Runner) forget(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if cancel, ok := r.running[id]; ok {
		cancel()
		delete(r.running, id)
	}
}

// ended is what a run came to. It passed only if every conversation did, because a
// simulation that holds nine times out of ten is a simulation that does not hold.
func ended(ctx context.Context, run store.SimulationRun) string {
	switch {
	case ctx.Err() != nil && run.Passed+run.Failed < run.Cases:
		return store.SimulationCancelled
	case run.Passed+run.Failed < run.Cases:
		return store.SimulationErrored
	case run.Failed > 0:
		return store.SimulationFailed
	default:
		return store.SimulationPassed
	}
}

// errored is a conversation that never got as far as a ruling, which is not the same as one
// that was ruled against.
func errored(kase store.SimulationCase, err error) store.SimulationCase {
	kase.State = store.SimulationErrored
	kase.Error = err.Error()
	return kase
}

// turnsOf is how many times the caller may speak.
func turnsOf(simulation store.Simulation) int {
	if simulation.MaxTurns > 0 {
		return simulation.MaxTurns
	}
	return 12
}

// agentID names the conversation the way the transcript and the call row will. A simulated
// conversation is still a call, and it is worth being able to find it as one.
func agentID(caseID string) string {
	return "simulation-" + caseID
}

// tag labels what a run spends so it can be told apart from what the same agent costs
// answering real callers.
func tag(configured routing.Tags, simulationID string) routing.Tags {
	tagged := routing.Tags{}
	for key, value := range configured {
		tagged[key] = value
	}
	tagged["simulation"] = simulationID
	return tagged
}
