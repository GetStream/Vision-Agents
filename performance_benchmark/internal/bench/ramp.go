package bench

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/agents"
)

// What a step is held to. Past any of these the router has stopped holding a
// conversation rather than merely slowed down, and the ramp stops.
const (
	// maxFailedSessions is the share of conversations that may fail to open.
	maxFailedSessions = 0.05
	// maxUnansweredTurns is the share of turns the agent may leave unanswered.
	maxUnansweredTurns = 0.10
	// maxReplyP95Ms is the wait at which a conversation has stopped being one.
	maxReplyP95Ms = 3000
)

// settleBetweenSteps lets the sessions of one step close before the next opens, so a
// step is not measured against the one before it still hanging up.
const settleBetweenSteps = 5 * time.Second

// podSample is how often the cluster is asked what the router is doing.
const podSample = 10 * time.Second

// Options is one ramp.
type Options struct {
	// Label names the build, which is what a comparison reads.
	Label string
	// Agent is the stored config the conversations run, and Instructions what it says.
	Agent        string
	Instructions string
	Models       Models
	// Start is how many agents the first step runs, Factor what each step multiplies it
	// by, and Max where to stop whatever is still holding.
	Start  int
	Factor int
	Max    int
	// Hold is how long each step talks for once every conversation in it is open.
	Hold time.Duration
	// JoinRate is how many conversations a step opens a second. Zero opens them all at
	// once. The hosted edge refuses one address past 600 requests a minute and each
	// conversation costs two, so a step of hundreds has to be spread out to measure the
	// router rather than the edge.
	JoinRate float64
	// Cluster is where the router runs, for the pod readings. Its zero value takes none.
	Cluster Cluster
	// Target is the router, for the report.
	Target string
	Logger *slog.Logger
}

// Run ramps until something gives way, and returns what it found.
//
// The agent's configuration is written once and every conversation runs it: a thousand
// stored configs would measure the config endpoint rather than the conversations.
func Run(ctx context.Context, options Options) (Report, error) {
	if err := options.valid(); err != nil {
		return Report{}, err
	}
	if err := writeConfig(ctx, options); err != nil {
		return Report{}, err
	}

	report := Report{
		Label:     options.Label,
		Target:    options.Target,
		Models:    options.Models,
		StartedAt: time.Now(),
		HoldS:     options.Hold.Seconds(),
	}

	for count := options.Start; count <= options.Max; count *= options.Factor {
		options.Logger.Info("ramping", "agents", count, "hold", options.Hold)
		step := runStep(ctx, options, count)
		report.Steps = append(report.Steps, step)

		if step.Broke != "" {
			report.Broke = step.Broke
			options.Logger.Warn("the router stopped keeping up", "agents", count, "broke", step.Broke)
			break
		}
		report.Held = count
		options.Logger.Info("held", "agents", count,
			"reply_p50_ms", step.ReplyMs.P50, "reply_p95_ms", step.ReplyMs.P95,
			"answered", fmt.Sprintf("%d/%d", step.Answered, step.Turns))

		if ctx.Err() != nil {
			break
		}
		sleep(ctx, settleBetweenSteps)
	}
	return report, nil
}

// runStep holds that many conversations at once and measures them together.
func runStep(ctx context.Context, options Options, count int) Step {
	watching, stop := context.WithCancel(ctx)
	pods := make(chan Pods, 1)
	go func() { pods <- options.Cluster.Watch(watching, podSample) }()
	host := watchHost()

	// Every conversation talks until the same moment, so all of them are held together for
	// Hold however long the last took to open.
	var apart time.Duration
	if options.JoinRate > 0 {
		apart = time.Duration(float64(time.Second) / options.JoinRate)
	}
	until := time.Now().Add(time.Duration(count)*apart + options.Hold)

	held := make([]conversation, count)
	var running sync.WaitGroup
	for i := range count {
		running.Add(1)
		go func() {
			defer running.Done()
			sleep(ctx, time.Duration(i)*apart)
			held[i] = Caller{
				Agent:  options.Agent,
				Models: options.Models,
				Index:  i,
				Logger: options.Logger,
			}.Converse(ctx, until)
		}()
	}
	running.Wait()

	stop()
	step := summarise(count, held)
	step.Router = <-pods
	step.Host = host.since()
	step.Broke = broke(step)
	return step
}

// summarise turns what the callers did into the step's numbers.
func summarise(count int, held []conversation) Step {
	step := Step{Agents: count, Sessions: count, Errors: map[string]int{}}
	var sessions, tracks, replies, gaps []float64

	for _, conversation := range held {
		if conversation.failedAt != "" {
			step.Errors[conversation.failedAt]++
		}
		if conversation.sessionMs > 0 {
			sessions = append(sessions, conversation.sessionMs)
		}
		if conversation.trackMs > 0 {
			step.Joined++
			tracks = append(tracks, conversation.trackMs)
		}
		step.DroppedFrames += conversation.dropped
		for _, turn := range conversation.turns {
			step.Turns++
			if !turn.answered() {
				continue
			}
			step.Answered++
			replies = append(replies, turn.replyMs)
			if turn.gapMs > 0 {
				gaps = append(gaps, turn.gapMs)
			}
		}
	}

	step.SessionMs = stat(sessions)
	step.TrackMs = stat(tracks)
	step.ReplyMs = stat(replies)
	step.GapMs = stat(gaps)
	return step
}

// broke says what gave way at this step, and is empty for one that held.
//
// A step that saturated this machine is not a step the router failed: the callers were
// the bottleneck, and saying so is more useful than reporting a ceiling that is really
// the laptop's.
func broke(step Step) string {
	if step.Host.Saturated() {
		return fmt.Sprintf("this machine ran out of room at %.0f%% of %d cores, so the ceiling here is the caller's and not the router's",
			step.Host.Busy*100, step.Host.Cores)
	}
	if step.Router.Restarts > 0 {
		return fmt.Sprintf("the router restarted %d times", step.Router.Restarts)
	}
	if failed := step.Sessions - step.Joined; float64(failed) > maxFailedSessions*float64(step.Sessions) {
		return fmt.Sprintf("%d of %d conversations never got the agent onto the call", failed, step.Sessions)
	}
	if step.Turns > 0 {
		unanswered := step.Turns - step.Answered
		if float64(unanswered) > maxUnansweredTurns*float64(step.Turns) {
			return fmt.Sprintf("%d of %d turns went unanswered", unanswered, step.Turns)
		}
	}
	if step.ReplyMs.P95 > maxReplyP95Ms {
		return fmt.Sprintf("a reply took %.0fms at p95, past the %dms a conversation survives",
			step.ReplyMs.P95, maxReplyP95Ms)
	}
	return ""
}

// writeConfig writes the stored config every conversation of the run starts from.
func writeConfig(ctx context.Context, options Options) error {
	agent, err := agents.New(agents.Options{
		Name:         options.Agent,
		Instructions: options.Instructions,
		LLM:          Caller{Agent: options.Agent, Models: options.Models, Logger: options.Logger}.pipeline(),
		Logger:       options.Logger,
	})
	if err != nil {
		return err
	}
	if _, err := agent.Sync(ctx); err != nil {
		return fmt.Errorf("bench: write the agent config: %w", err)
	}
	return nil
}

func (o Options) valid() error {
	switch {
	case o.Agent == "":
		return errors.New("bench: the conversations need an agent name")
	case o.Start < 1:
		return errors.New("bench: a ramp starts at one agent or more")
	case o.Factor < 2:
		return errors.New("bench: a ramp step multiplies by two or more")
	case o.Max < o.Start:
		return errors.New("bench: the ramp's ceiling is below where it starts")
	case o.Hold <= 0:
		return errors.New("bench: a step has to talk for some length of time")
	case o.Logger == nil:
		return errors.New("bench: a run needs a logger")
	}
	return nil
}

func sleep(ctx context.Context, how time.Duration) {
	timer := time.NewTimer(how)
	defer timer.Stop()
	select {
	case <-timer.C:
	case <-ctx.Done():
	}
}
