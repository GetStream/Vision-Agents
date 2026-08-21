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
	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
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
	fmt.Fprintln(os.Stderr, "usage: voicebench <synth|run|report|calibrate> [flags]")
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
	case "calibrate":
		return cmdCalibrate(root, args)
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
	callID := fs.String("call-id", "", "call id / room name. Empty generates one per trial")
	callType := fs.String("call-type", "default", "Stream call type")
	transport := fs.String("transport", "stream", "media transport (stream, livekit)")
	target := fs.String("target", "", "target system (python, acceleration, livekit)")
	targetURL := fs.String("target-url", "", "target HTTP base URL, or LiveKit URL for --target livekit")
	targetModel := fs.String("target-model", "", "target model identifier for the reproducibility manifest")
	targetVoice := fs.String("target-voice", "", "target voice identifier for the reproducibility manifest")
	spawn := fs.Bool("spawn", false, "start the selected target for this run")
	bin := fs.String("bin", "", "target binary, used by --target acceleration --spawn")
	userID := fs.String("user", "voicebench-caller", "caller user id / LiveKit identity")
	liveKitAgent := fs.String("livekit-agent", "", "LiveKit agent name for dispatch")
	liveKitDeployment := fs.String("livekit-deployment", "", "LiveKit Cloud deployment for dispatch")
	worldAddr := fs.String("world-addr", "127.0.0.1:8090", "world server bind")
	worldURL := fs.String("world-url", "", "world server URL given to the target. Defaults to the bind address")
	system := fs.String("system", "", "system name in the report. Defaults to the selected target.")
	networkProfile := fs.String("network-profile", os.Getenv("VOICEBENCH_NETWORK_PROFILE"), "stable label for the runner region and network setup")
	out := fs.String("out", "", "output directory")
	skipSTT := fs.Bool("skip-stt", false, "skip Deepgram (fails the trial)")
	skipJudge := fs.Bool("skip-judge", false, "skip LLM judge (fails the trial)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *target == "livekit" && *liveKitAgent == "" && !*spawn {
		*liveKitAgent = os.Getenv("LIVEKIT_AGENT_NAME")
	}
	sum, err := run.Run(ctx, run.Config{
		Root:              root,
		OutDir:            *out,
		Pack:              *pack,
		ScenarioID:        *id,
		K:                 *k,
		WorldAddr:         *worldAddr,
		WorldURL:          *worldURL,
		CallID:            *callID,
		CallType:          *callType,
		Transport:         *transport,
		UserID:            *userID,
		System:            *system,
		TargetName:        *target,
		TargetURL:         *targetURL,
		TargetModel:       *targetModel,
		TargetVoice:       *targetVoice,
		TargetBin:         *bin,
		SpawnTarget:       *spawn,
		LiveKitAgentName:  *liveKitAgent,
		LiveKitDeployment: *liveKitDeployment,
		NetworkProfile:    *networkProfile,
		SkipSTT:           *skipSTT,
		SkipJudge:         *skipJudge,
		Logger:            slog.Default(),
	})
	if len(sum.Calls) > 0 {
		report.FprintTable(os.Stdout, sum)
	}
	if err != nil {
		return err
	}
	if invalid := sum.InvalidTrials(); invalid > 0 {
		return fmt.Errorf("run: %d trial(s) produced no verdict", invalid)
	}
	return nil
}

func cmdCalibrate(root string, args []string) error {
	fs := flag.NewFlagSet("calibrate", flag.ExitOnError)
	fixture := fs.String("fixture", filepath.Join(root, "calibration", "judge.json"), "human-labeled judge fixture")
	out := fs.String("out", "", "write the calibration report as JSON")
	minimum := fs.Float64("minimum-agreement", 0.9, "required exact case agreement")
	if err := fs.Parse(args); err != nil {
		return err
	}
	set, err := score.LoadCalibrationSet(*fixture)
	if err != nil {
		return err
	}
	scenarios := map[string]scenario.Scenario{}
	for _, pack := range scenario.Packs() {
		loaded, err := scenario.LoadPack(filepath.Join(root, "scenarios", pack))
		if err != nil {
			return err
		}
		for _, sc := range loaded {
			scenarios[sc.ID] = sc
		}
	}
	calibration := score.CalibrateJudge(set, scenarios, *minimum)
	raw, err := json.MarshalIndent(calibration, "", "  ")
	if err != nil {
		return err
	}
	if *out != "" {
		if err := os.WriteFile(*out, raw, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("judge calibration: %d/%d decisions (%.1f%%), exact cases: %d/%d, critical misses: %d, reviewed by: %q\n",
		calibration.DecisionsAgreed, calibration.Decisions, calibration.AgreementRate*100,
		calibration.CasesAgreed, calibration.Cases, calibration.CriticalMisses, calibration.ReviewedBy)
	if !calibration.LabelsReviewed {
		return fmt.Errorf("judge calibration labels need human review")
	}
	if !calibration.ModelPassed {
		return fmt.Errorf("judge calibration failed")
	}
	return nil
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
