package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/joho/godotenv"

	"github.com/GetStream/Vision-Agents/benchmark/internal/report"
	"github.com/GetStream/Vision-Agents/benchmark/internal/run"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/synth"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	if err := dispatch(os.Args[1], os.Args[2:]); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: voicebench <synth|run|report> [flags]")
}

func dispatch(cmd string, args []string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	root := findRoot()
	loadDotEnv(root)
	switch cmd {
	case "synth":
		return cmdSynth(ctx, root, args)
	case "run":
		return cmdRun(ctx, root, args)
	case "report":
		return cmdReport(root, args)
	default:
		usage()
		return fmt.Errorf("unknown command %s", cmd)
	}
}

func cmdSynth(ctx context.Context, root string, args []string) error {
	fs := flag.NewFlagSet("synth", flag.ExitOnError)
	pack := fs.String("pack", "", "scenario pack (restaurant, healthcare, telecom). Empty means all.")
	voice := fs.String("voice", os.Getenv("ELEVENLABS_VOICE_ID"), "ElevenLabs voice id")
	if err := fs.Parse(args); err != nil {
		return err
	}
	packs := scenario.Packs()
	if *pack != "" {
		packs = []string{*pack}
	}
	for _, p := range packs {
		scenarios, err := scenario.LoadPack(filepath.Join(root, "scenarios", p))
		if err != nil {
			return err
		}
		if err := synth.Pack(root, *voice, scenarios); err != nil {
			return err
		}
		fmt.Printf("synthesized %s (%d scenarios)\n", p, len(scenarios))
	}
	return nil
}

func cmdRun(ctx context.Context, root string, args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	pack := fs.String("pack", "restaurant", "scenario pack")
	id := fs.String("scenario", "", "run a single scenario id")
	k := fs.Int("k", 3, "trials per scenario")
	callID := fs.String("call-id", "", "Stream call id. Empty generates one per trial")
	callType := fs.String("call-type", "default", "Stream call type")
	agentURL := fs.String("agent-url", "", "Vision Agents HTTP base, POST /calls/{id}/sessions")
	spawnAgent := fs.Bool("spawn-agent", false, "start python -m voicebench_agents for this pack")
	agentPort := fs.Int("agent-port", 8000, "port for --spawn-agent")
	userID := fs.String("user", "voicebench-caller", "Stream user id the harness joins as")
	worldAddr := fs.String("world-addr", "127.0.0.1:8090", "world server bind")
	system := fs.String("system", "vision-agents", "system name in the report")
	out := fs.String("out", "", "output directory")
	skipSTT := fs.Bool("skip-stt", false, "skip Deepgram (fails the trial)")
	skipJudge := fs.Bool("skip-judge", false, "skip LLM judge (fails the trial)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	sum, err := run.Run(ctx, run.Config{
		Root:       root,
		OutDir:     *out,
		Pack:       *pack,
		ScenarioID: *id,
		K:          *k,
		WorldAddr:  *worldAddr,
		CallID:     *callID,
		CallType:   *callType,
		AgentURL:   *agentURL,
		UserID:     *userID,
		System:     *system,
		SpawnAgent: *spawnAgent,
		AgentPort:  *agentPort,
		SkipSTT:    *skipSTT,
		SkipJudge:  *skipJudge,
		Logger:     slog.Default(),
	})
	if len(sum.Calls) > 0 {
		report.FprintTable(os.Stdout, sum)
	}
	return err
}

func cmdReport(root string, args []string) error {
	fs := flag.NewFlagSet("report", flag.ExitOnError)
	dir := fs.String("dir", "", "existing out/<run_id> directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dir == "" {
		return fmt.Errorf("report: --dir is required")
	}
	raw, err := os.ReadFile(filepath.Join(*dir, "summary.json"))
	if err != nil {
		return err
	}
	var sum report.Summary
	if err := json.Unmarshal(raw, &sum); err != nil {
		return err
	}
	report.FprintTable(os.Stdout, sum)
	return os.WriteFile(filepath.Join(*dir, "report.md"), []byte(report.Markdown(sum)), 0o644)
}

func loadDotEnv(root string) {
	for _, dir := range []string{root, filepath.Dir(root)} {
		path := filepath.Join(dir, ".env")
		if st, err := os.Stat(path); err == nil && !st.IsDir() {
			_ = godotenv.Load(path)
			return
		}
	}
}

func findRoot() string {
	if env := os.Getenv("VOICEBENCH_ROOT"); env != "" {
		return env
	}
	wd, _ := os.Getwd()
	for dir := wd; dir != "/"; dir = filepath.Dir(dir) {
		if _, err := os.Stat(filepath.Join(dir, "scenarios")); err == nil {
			return dir
		}
	}
	return wd
}
