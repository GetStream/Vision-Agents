// Command agent joins a Stream call as a voice agent: it listens, thinks and talks back.
//
// It is the demo for all three modalities at once. Every turn goes through the routers, so
// the same failover, health and billing that a direct API call gets applies to a
// conversation. The Opus path is cgo, so pkg-config, libopus, libopusfile and libsoxr must
// be installed to build this.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/agent/streamedge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory/mem0"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

const (
	postgresEnvVar    = "ROUTER_POSTGRES_DSN"
	redisEnvVar       = "ROUTER_REDIS_ADDR"
	configEnvVar      = "ROUTER_CONFIG"
	skillsEnvVar      = "HARNESS_SKILLS"
	toolsEnvVar       = "HARNESS_TOOLS"
	phoneConfigEnvVar = "ROUTER_PHONE_CONFIG"
)

// finishWithin bounds how long the agent is given to finish its last sentence.
const finishWithin = 10 * time.Second

// The person the browser joins the call as, so the agent has somebody to talk to.
const (
	demoUserID   = "demo-caller"
	demoUserName = "Demo caller"
)

func main() {
	options := options{}
	flag.StringVar(&options.callID, "call", "", "Stream call id to join (required)")
	flag.StringVar(&options.callType, "call-type", "default", "Stream call type")
	flag.StringVar(&options.userID, "user", "vision-agent", "user id to join as")
	flag.StringVar(&options.customerID, "customer", "demo", "customer the usage is billed to")
	flag.StringVar(&options.agentID, "agent", "", "agent id transcripts and stats are keyed by, defaults to the call id")
	flag.StringVar(&options.appID, "app", "", "application memories are scoped to")
	flag.Var(&options.tags, "tag", "cost label as key=value, repeat for several")
	flag.StringVar(&options.llmTarget, "llm", "llm-fast", "llm provider/model or shortcut")
	flag.StringVar(&options.sttTarget, "stt", "en-low-latency", "stt provider/model or shortcut")
	flag.StringVar(&options.ttsTarget, "tts", "en-low-latency", "tts provider/model or shortcut")
	flag.StringVar(&options.voice, "voice", "", "provider-specific voice id")
	flag.StringVar(&options.language, "language", "", "language hint, e.g. es")
	flag.StringVar(&options.instructions, "instructions",
		"You are a helpful voice assistant. Keep your answers to one or two sentences.",
		"the system prompt")
	flag.StringVar(&options.greeting, "greeting", "Hi, I'm listening.",
		"said on joining, without going through the model")
	flag.StringVar(&options.subagentTarget, "subagent", "",
		"provider/model or shortcut for the model that does the thinking, empty to answer everything on the voice model")
	flag.StringVar(&options.skillsFile, "skills", os.Getenv(skillsEnvVar),
		"skills the voice model may hand over, empty for the built-in set")
	flag.StringVar(&options.toolsFile, "tools", os.Getenv(toolsEnvVar),
		"tools the voice model may run, empty for the built-in set")
	flag.StringVar(&options.number, "number", "",
		"one of your numbers, which is what a transferred human sees, and what turns transferring on")
	flag.StringVar(&options.vendor, "vendor", "telnyx", "vendor carrying an outbound leg")
	flag.StringVar(&options.vendorCallID, "vendor-call", "",
		"the vendor call id of an outbound leg, which is what lets the agent press digits at a menu")
	flag.BoolVar(&options.navigating, "navigating", false,
		"the agent placed this call, so let recordings finish and answer their menus")
	flag.IntVar(&options.tasks, "tasks", 0, "how much delegated work may run at once")
	flag.BoolVar(&options.backchannel, "backchannel", true,
		"murmur while the caller is still talking, the way a person on the phone does")
	flag.Float64Var(&options.minConfidence, "min-confidence", 0,
		"how sure the transcriber must be for the agent to answer rather than check what was meant")
	flag.BoolVar(&options.demo, "demo", true,
		"open a browser on a link that joins the call, so there is somebody for the agent to talk to")
	verbose := flag.Bool("verbose", false, "log lifecycle events")
	flag.Parse()

	level := slog.LevelWarn
	if *verbose {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(logger)

	if err := run(options, logger); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

type options struct {
	callID       string
	callType     string
	userID       string
	customerID   string
	agentID      string
	appID        string
	tags         routing.TagsFlag
	llmTarget    string
	sttTarget    string
	ttsTarget    string
	voice        string
	language     string
	instructions string
	greeting     string

	subagentTarget string
	skillsFile     string
	toolsFile      string
	tasks          int
	backchannel    bool
	minConfidence  float64
	demo           bool

	number       string
	vendor       string
	vendorCallID string
	navigating   bool
}

// prompt is what the agent is told to be. An agent that placed the call is told how to get
// through whatever answers, ahead of whatever it was told to do once it has.
func (o options) prompt() string {
	if !o.navigating {
		return o.instructions
	}
	return agent.NavigatingInstructions + "\n\n" + o.instructions
}

// duplex is how the agent listens and talks at the same time.
func (o options) duplex() agent.DuplexOptions {
	return agent.DuplexOptions{
		Backchannel:   o.backchannel,
		MinConfidence: o.minConfidence,
	}
}

func (o options) languages() []string {
	if trimmed := strings.TrimSpace(o.language); trimmed != "" {
		return []string{trimmed}
	}
	return nil
}

// agent returns the agent id, falling back to the call id so a conversation always has
// one to store its transcript under.
func (o options) agent() string {
	if trimmed := strings.TrimSpace(o.agentID); trimmed != "" {
		return trimmed
	}
	return o.callID
}

func run(options options, logger *slog.Logger) error {
	if options.callID == "" {
		return errors.New("-call is required")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	routers, cleanup, err := buildRouters(ctx, logger)
	if err != nil {
		return err
	}
	defer cleanup()

	edge, err := streamedge.New(streamedge.Options{
		CallID:   options.callID,
		CallType: options.callType,
		User:     streamedge.User{ID: options.userID, Name: "Vision Agent"},
		Logger:   logger,
	})
	if err != nil {
		return err
	}

	// Without a mem0 key the agent starts every call knowing nothing but its
	// instructions, which is the behaviour before memory existed.
	var remembering memory.Store
	if recall, err := mem0.New(mem0.Options{Logger: logger}); err != nil {
		logger.Debug("not remembering anything between calls", "error", err)
	} else {
		remembering = recall
	}

	// Skills only mean something with a subagent to run them, so loading them is skipped
	// rather than failing when the agent is answering everything itself.
	var skills harness.Skills
	if options.subagentTarget != "" {
		skills, err = harness.LoadSkills(options.skillsFile)
		if err != nil {
			return err
		}
	}

	// Telephony is only wired when the agent has a number to act from: a call in a
	// browser has nowhere to transfer anyone to and no keypad to press.
	line, tools, err := buildTelephony(options, routers.store, logger)
	if err != nil {
		return err
	}

	voiceAgent, err := agent.New(agent.Options{
		Edge:           edge,
		Instructions:   options.prompt(),
		CustomerID:     options.customerID,
		AgentID:        options.agent(),
		CallID:         options.callID,
		Tags:           options.tags.Tags,
		SubagentTarget: options.subagentTarget,
		Skills:         skills,
		Telephony:      line,
		Tools:          tools,
		Tasks:          options.tasks,
		Duplex:         options.duplex(),
		LLM:            routers.llm,
		LLMTarget:      options.llmTarget,
		STT:            routers.stt,
		STTTarget:      options.sttTarget,
		TTS:            routers.tts,
		TTSTarget:      options.ttsTarget,
		Voice:          options.voice,
		LanguageHints:  options.languages(),
		Memory:         remembering,
		AppID:          options.appID,
		Store:          routers.store,
		Live:           routers.live,
		Logger:         logger,
	})
	if err != nil {
		return err
	}
	defer voiceAgent.Close()

	// A voice call leaves nothing behind, so what was said is stored in a chat channel
	// named after the agent. Without Stream credentials the conversation just is not kept.
	transcript, err := chatlog.New(chatlog.Options{
		AgentID: options.agent(),
		Agent:   chatlog.User{ID: options.userID, Name: "Vision Agent"},
		Logger:  logger,
	})
	if err != nil {
		logger.Warn("not storing the transcript", "error", err)
	} else {
		if err := transcript.Start(ctx); err != nil {
			return err
		}
		defer transcript.Close()
	}

	go report(voiceAgent, transcript)

	if err := voiceAgent.Join(ctx); err != nil {
		return err
	}
	fmt.Printf("joined %s:%s as %s\n", options.callType, options.callID, options.userID)
	// Transcription is routed per participant, so there is no one provider to name until
	// somebody speaks.
	fmt.Printf("answering with %s/%s, speaking with %s/%s, listening via %s\n",
		voiceAgent.LLM().Provider(), voiceAgent.LLM().Model(),
		voiceAgent.TTS().Provider(), voiceAgent.TTS().Model(),
		options.sttTarget)
	if options.subagentTarget != "" {
		fmt.Printf("handing the hard parts to %s: %s\n", options.subagentTarget, skillNames(skills))
	}

	if options.demo {
		invite(edge, logger)
	}

	if options.greeting != "" {
		if err := voiceAgent.Say(ctx, options.greeting); err != nil {
			return err
		}
	}

	fmt.Println("talk to it, or Ctrl-C to leave")
	<-ctx.Done()

	// Hanging up mid-sentence is rude, so the last utterance is given a moment to finish.
	finishing, cancel := context.WithTimeout(context.Background(), finishWithin)
	defer cancel()
	if err := voiceAgent.Finish(finishing); err != nil {
		logger.Warn("the agent was still talking when it left", "error", err)
	}

	fmt.Println("\nleaving")
	return voiceAgent.Close()
}

// buildTelephony wires what the agent may do to the call, which is nothing unless it was
// given a number to act from.
//
// The tools are returned alongside because they are only worth offering to a model that can
// run them: a model told it may transfer, on a call with nowhere to transfer to, promises
// the caller a person who never arrives.
func buildTelephony(
	options options,
	numbers *store.Store,
	logger *slog.Logger,
) (agent.Telephony, harness.Tools, error) {
	if strings.TrimSpace(options.number) == "" {
		return nil, harness.Tools{}, nil
	}
	if numbers == nil {
		return nil, harness.Tools{}, fmt.Errorf(
			"transferring needs %s, because it is the record of who holds %s",
			postgresEnvVar, options.number)
	}

	tools, err := harness.LoadTools(options.toolsFile)
	if err != nil {
		return nil, harness.Tools{}, err
	}

	config, err := phone.LoadConfig(os.Getenv(phoneConfigEnvVar))
	if err != nil {
		return nil, harness.Tools{}, err
	}
	stream, err := phone.NewStream(phone.StreamOptions{})
	if err != nil {
		return nil, harness.Tools{}, err
	}
	service, err := phone.NewService(phone.ServiceOptions{
		Registry: vendors.Registry(config),
		Store:    numbers,
		Stream:   stream,
		Logger:   logger,
	})
	if err != nil {
		return nil, harness.Tools{}, err
	}

	return service.Line(phone.LineOptions{
		Owner:        routing.Owner{CustomerID: options.customerID, Tags: options.tags.Tags},
		From:         options.number,
		CallID:       options.callID,
		CallType:     options.callType,
		Vendor:       options.vendor,
		VendorCallID: options.vendorCallID,
	}), tools, nil
}

// invite opens a browser on a link that joins the call, because an agent alone in a call
// has nobody to talk to. A link that cannot be opened is still printed: the conversation
// works, it just needs somebody to click.
func invite(edge *streamedge.Edge, logger *slog.Logger) {
	link, err := edge.DemoURL(streamedge.User{ID: demoUserID, Name: demoUserName})
	if err != nil {
		logger.Warn("not opening a browser on the call", "error", err)
		return
	}

	fmt.Printf("join the call at %s\n", link)
	if err := openBrowser(link); err != nil {
		logger.Warn("could not open a browser, so open the link above instead", "error", err)
	}
}

// openBrowser shows a link in whatever this machine opens links with.
func openBrowser(link string) error {
	switch runtime.GOOS {
	case "darwin":
		return exec.Command("open", link).Start()
	case "windows":
		return exec.Command("rundll32", "url.dll,FileProtocolHandler", link).Start()
	default:
		return exec.Command("xdg-open", link).Start()
	}
}

// report prints the conversation as it happens and stores it. Interim work is left to
// -verbose logging: what a reader wants is what was said.
func report(voiceAgent *agent.Agent, transcript *chatlog.Log) {
	for event := range voiceAgent.Events() {
		if transcript != nil {
			transcript.Record(event)
		}

		switch typed := event.(type) {
		case agent.Heard:
			fmt.Printf("%s: %s\n", speaker(typed), typed.Text)
		case agent.Responded:
			fmt.Printf("agent (%.0fms to first token): %s\n", typed.TimeToFirstTokenMs, typed.Text)
		case agent.Turn:
			fmt.Printf("  turn %.0fms round trip  stt %.0fms  llm %.0fms  tts %.0fms\n",
				typed.RoundtripMs, typed.STTLatencyMs, typed.LLMTTFTMs, typed.TTSTTFBMs)
		case agent.Delegated:
			fmt.Printf("  handed %s to the subagent: %s\n", typed.Skill, typed.Prompt)
		case agent.TaskSettled:
			fmt.Printf("  %s came back in %.0fms: %s%s\n",
				typed.Skill, typed.ElapsedMs, typed.Text, typed.Question)
		case agent.TaskCancelled:
			fmt.Printf("  %s abandoned (%s)\n", typed.Skill, typed.Reason)
		case agent.Transferred:
			fmt.Printf("  transferring to %s: %s\n", typed.To, typed.Summary)
		case agent.Pressed:
			fmt.Printf("  pressed %s\n", typed.Digits)
		case agent.ToolRan:
			if typed.Err != nil {
				fmt.Fprintf(os.Stderr, "  %s failed: %v\n", typed.Tool, typed.Err)
			}
		case agent.Interrupted:
			fmt.Println("(interrupted)")
		case agent.Error:
			fmt.Fprintf(os.Stderr, "%s failed: %v\n", typed.Context, typed.Err)
		}
	}
}

// skillNames lists what the voice model may hand over.
func skillNames(skills harness.Skills) string {
	names := make([]string, 0, len(skills.Skills))
	for _, skill := range skills.Skills {
		names = append(names, skill.Name)
	}
	return strings.Join(names, ", ")
}

func speaker(heard agent.Heard) string {
	if heard.Participant.Name != "" {
		return heard.Participant.Name
	}
	if heard.Participant.UserID != "" {
		return heard.Participant.UserID
	}
	return "someone"
}

// routers is the three modalities an agent needs, plus the store its turns are recorded
// in when one is configured.
type routers struct {
	llm   *llmrouter.Router
	stt   *sttrouter.Router
	tts   *ttsrouter.Router
	store *store.Store
	live  *live.Client
}

// buildRouters wires all three routers, using Postgres and Redis when they are configured.
// The demo is useful without them: it just stops recording usage.
func buildRouters(ctx context.Context, logger *slog.Logger) (routers, func(), error) {
	config, err := routing.LoadConfig(os.Getenv(configEnvVar))
	if err != nil {
		return routers{}, nil, err
	}
	for _, modality := range []routing.Modality{routing.STT, routing.LLM, routing.TTS} {
		if _, ok := config[modality]; !ok {
			return routers{}, nil, fmt.Errorf("the config declares no %s providers", modality)
		}
	}

	var closers []func()
	cleanup := func() {
		for i := len(closers) - 1; i >= 0; i-- {
			closers[i]()
		}
	}

	var pgStore *store.Store
	if dsn := os.Getenv(postgresEnvVar); dsn != "" {
		pgStore, err = store.Open(dsn)
		if err != nil {
			cleanup()
			return routers{}, nil, err
		}
		closers = append(closers, func() { pgStore.Close() })

		if err := pgStore.Migrate(ctx); err != nil {
			cleanup()
			return routers{}, nil, err
		}
	}

	var liveClient *live.Client
	if address := os.Getenv(redisEnvVar); address != "" {
		liveClient, err = live.New(live.Options{Address: address})
		if err != nil {
			cleanup()
			return routers{}, nil, err
		}
		closers = append(closers, liveClient.Close)
	}

	transcriber, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
		Store:    pgStore,
		Live:     liveClient,
		Logger:   logger,
	})
	if err != nil {
		cleanup()
		return routers{}, nil, err
	}
	closers = append(closers, transcriber.Close)

	reasoner, err := llmrouter.New(llmrouter.Options{
		Config:   config[routing.LLM],
		Registry: llmrouter.DefaultRegistry(),
		Store:    pgStore,
		Live:     liveClient,
		Logger:   logger,
	})
	if err != nil {
		cleanup()
		return routers{}, nil, err
	}
	closers = append(closers, reasoner.Close)

	speech, err := ttsrouter.New(ttsrouter.Options{
		Config:   config[routing.TTS],
		Registry: ttsrouter.DefaultRegistry(),
		Store:    pgStore,
		Live:     liveClient,
		Logger:   logger,
	})
	if err != nil {
		cleanup()
		return routers{}, nil, err
	}
	closers = append(closers, speech.Close)

	return routers{
		llm: reasoner, stt: transcriber, tts: speech, store: pgStore, live: liveClient,
	}, cleanup, nil
}
