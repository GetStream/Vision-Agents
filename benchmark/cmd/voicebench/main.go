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
	"strings"
	"syscall"
	"time"

	"github.com/joho/godotenv"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/report"
	"github.com/GetStream/Vision-Agents/benchmark/internal/run"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
	"github.com/GetStream/Vision-Agents/benchmark/internal/slack"
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
	fmt.Fprintln(os.Stderr, "usage: voicebench <synth|run|report|calibrate|compare|noise|digest|stt|tts> [flags]")
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
	case "compare":
		return cmdCompare(root, args)
	case "noise":
		return cmdNoise(root, args)
	case "digest":
		return cmdDigest(ctx, args)
	case "stt":
		return cmdSTT(ctx, root, args)
	case "tts":
		return cmdTTS(ctx, root, args)
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
	target := fs.String("target", "", "target system (python, acceleration, accelerated, livekit)")
	targetURL := fs.String("target-url", "", "target HTTP base URL, or LiveKit URL for --target livekit")
	targetModel := fs.String("target-model", "", "target model identifier for the reproducibility manifest")
	targetVoice := fs.String("target-voice", "", "target voice identifier for the reproducibility manifest")
	spawn := fs.Bool("spawn", false, "start the selected target for this run")
	bin := fs.String("bin", "", "router binary, used by --target acceleration|accelerated --spawn")
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
	frozen := fs.Bool("frozen", false, "run only the frozen scenario set used for the trend line")
	short := fs.Bool("short", false, "run only the short scenario set, for quick iteration")
	storeBaseline := fs.Bool("store-baseline", false, "copy summary.json and manifest.json to baselines/<target>/<commit>/")
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
		Frozen:            *frozen,
		Short:             *short,
		Logger:            slog.Default(),
	})
	if len(sum.Calls) > 0 {
		report.FprintTable(os.Stdout, sum)
	}
	var storeErr error
	if *storeBaseline && sum.RunID != "" {
		runDir := filepath.Join(root, "out", sum.RunID)
		if *out != "" {
			runDir = *out
		}
		targetName := sum.Manifest.Target
		if targetName == "" {
			targetName = sum.System
		}
		storeErr = report.StoreBaseline(root, targetName, sum.Manifest.GitCommit, runDir)
	}
	if err != nil {
		return err
	}
	if invalid := sum.InvalidTrials(); invalid > 0 {
		return fmt.Errorf("run: %d trial(s) produced no verdict", invalid)
	}
	return storeErr
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

func cmdCompare(root string, args []string) error {
	fs := flag.NewFlagSet("compare", flag.ExitOnError)
	baseline := fs.String("baseline", "", "run directory or stored target name (baselines/<target>/<commit>)")
	mde := fs.String("mde", "", "noise floor from voicebench noise; flags changes bigger than it")
	out := fs.String("out", "", "write the comparison markdown here")
	if err := fs.Parse(args); err != nil {
		return err
	}
	dirs := fs.Args()
	if *baseline != "" {
		resolved, err := report.ResolveBaseline(root, *baseline)
		if err != nil {
			return err
		}
		dirs = append([]string{resolved}, dirs...)
	}
	if len(dirs) < 2 {
		return fmt.Errorf("compare: need at least two run directories")
	}
	cfg := report.CompareConfig{Baseline: -1}
	if *mde != "" {
		noise, err := report.LoadNoiseFloor(*mde)
		if err != nil {
			return fmt.Errorf("compare: %w", err)
		}
		cfg.MDE = &noise
	}
	if *baseline != "" {
		cfg.Baseline = 0
	}
	for i, dir := range dirs {
		sum, err := report.LoadSummary(dir)
		if err != nil {
			return fmt.Errorf("compare: %s: %w", dir, err)
		}
		label := filepath.Base(dir)
		if sum.System != "" {
			label = sum.System
		}
		if *baseline != "" && i == 0 {
			label = "baseline"
		}
		cfg.Runs = append(cfg.Runs, report.LabeledRun{Label: label, Summary: sum})
	}
	md := report.CompareMarkdown(cfg)
	fmt.Print(md)
	if *out != "" {
		return os.WriteFile(*out, []byte(md), 0o644)
	}
	return nil
}

func cmdNoise(root string, args []string) error {
	fs := flag.NewFlagSet("noise", flag.ExitOnError)
	out := fs.String("out", "", "write the noise floor here (default baselines/<target>/noise-<packs>.json)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	var runs []report.LabeledRun
	for _, dir := range fs.Args() {
		sum, err := report.LoadSummary(dir)
		if err != nil {
			return fmt.Errorf("noise: %s: %w", dir, err)
		}
		runs = append(runs, report.LabeledRun{Label: dir, Summary: sum})
	}
	noise, err := report.MeasureNoise(runs)
	if err != nil {
		return err
	}
	fmt.Print(report.NoiseMarkdown(noise))
	path := *out
	if path == "" {
		path = filepath.Join(root, "baselines", noise.Target, "noise-"+strings.Join(noise.Packs, "+")+".json")
	}
	raw, err := json.MarshalIndent(noise, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	fmt.Printf("\nwrote %s\n", path)
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

func cmdDigest(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("digest", flag.ExitOnError)
	title := fs.String("title", "Voicebench", "headline of the card and the message")
	out := fs.String("out", "", "directory to write voicebench.png and voicebench.html into")
	post := fs.Bool("slack", false, "post to VOICEBENCH_SLACK_CHANNEL as the bot behind VOICEBENCH_SLACK_BOT_TOKEN")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		return fmt.Errorf("digest: need at least one run directory")
	}
	var runs []report.LabeledRun
	for _, dir := range fs.Args() {
		sum, err := report.LoadSummary(dir)
		if err != nil {
			return fmt.Errorf("digest: %s: %w", dir, err)
		}
		label := sum.System
		if label == "" {
			label = filepath.Base(dir)
		}
		runs = append(runs, report.LabeledRun{Label: label, Summary: sum})
	}
	runs = report.MergeRuns(runs)
	digest := report.BuildDigest(runs)
	card, err := digest.PNG(*title)
	if err != nil {
		return fmt.Errorf("digest: draw card: %w", err)
	}
	page, err := digest.HTML(*title, card, runs)
	if err != nil {
		return fmt.Errorf("digest: render report: %w", err)
	}
	text := digest.SlackText(*title)
	fmt.Print(text)
	if *out != "" {
		if err := os.MkdirAll(*out, 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(*out, "voicebench.png"), card, 0o644); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(*out, "voicebench.html"), []byte(page), 0o644); err != nil {
			return err
		}
	}
	if !*post {
		return nil
	}
	client := slack.Client{Token: os.Getenv("VOICEBENCH_SLACK_BOT_TOKEN")}
	return client.Post(ctx, os.Getenv("VOICEBENCH_SLACK_CHANNEL"), text, []slack.File{
		{Name: "voicebench.png", Title: *title, Data: card},
		{Name: "voicebench.html", Title: "Full report", Data: []byte(page)},
	})
}

func cmdSTT(ctx context.Context, root string, args []string) error {
	fs := flag.NewFlagSet("stt", flag.ExitOnError)
	manifest := fs.String("manifest", "", "JSONL of id, reference, and audio (a WAV to stream) or hypothesis (to score as given)")
	var targets stringList
	fs.Var(&targets, "target", "provider/model or shortcut to stream each clip to through the router; repeat for several")
	out := fs.String("out", "", "output directory (default out/stt-<time>)")
	networkProfile := fs.String("network-profile", os.Getenv("VOICEBENCH_NETWORK_PROFILE"), "stable label for the runner region and network setup")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *manifest == "" {
		return fmt.Errorf("stt: --manifest is required")
	}
	dir := *out
	if dir == "" {
		dir = filepath.Join(root, "out", "stt-"+time.Now().UTC().Format("20060102T150405Z"))
	}
	sum, err := run.STT(ctx, run.STTConfig{
		Root:           root,
		Manifest:       *manifest,
		Targets:        targets,
		Out:            dir,
		NetworkProfile: *networkProfile,
		Logger:         slog.Default(),
	})
	if err != nil {
		return err
	}
	fmt.Print(report.STTMarkdown(sum))
	fmt.Printf("\nresults in %s\n", dir)
	for _, target := range sum.STT {
		if target.Failed > 0 {
			return fmt.Errorf("stt: %d clip(s) for %s ended in an error, see clips.jsonl", target.Failed, target.Target)
		}
	}
	return nil
}

// stringList is a flag that may be given more than once.
type stringList []string

func (l *stringList) String() string { return strings.Join(*l, ",") }

func (l *stringList) Set(value string) error {
	*l = append(*l, value)
	return nil
}

func cmdTTS(ctx context.Context, root string, args []string) error {
	fs := flag.NewFlagSet("tts", flag.ExitOnError)
	wav := fs.String("wav", "", "16-bit PCM wav to score for clipping and silence, without synthesizing anything")
	var targets stringList
	fs.Var(&targets, "target", "provider/model or shortcut to speak the corpus with through the router; repeat for several")
	corpus := fs.String("corpus", "", "JSONL of id and text (default every scenario's agent reply lines)")
	voice := fs.String("voice", "", "voice for every target, when the target's default is not wanted")
	out := fs.String("out", "", "output directory (default out/tts-<time>)")
	networkProfile := fs.String("network-profile", os.Getenv("VOICEBENCH_NETWORK_PROFILE"), "stable label for the runner region and network setup")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *wav != "" {
		pcm, err := audio.ReadWAV(*wav)
		if err != nil {
			return err
		}
		health := audio.MeasureHealth(pcm.Samples, pcm.Rate)
		raw, err := json.MarshalIndent(health, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(raw))
		return nil
	}
	if len(targets) == 0 {
		return fmt.Errorf("tts: --target or --wav is required")
	}
	dir := *out
	if dir == "" {
		dir = filepath.Join(root, "out", "tts-"+time.Now().UTC().Format("20060102T150405Z"))
	}
	sum, err := run.TTS(ctx, run.TTSConfig{
		Root:           root,
		Corpus:         *corpus,
		Targets:        targets,
		Voice:          *voice,
		Out:            dir,
		NetworkProfile: *networkProfile,
		Logger:         slog.Default(),
	})
	if err != nil {
		return err
	}
	fmt.Print(report.TTSMarkdown(sum))
	fmt.Printf("\nresults in %s\n", dir)
	for _, target := range sum.TTS {
		if target.Failed > 0 {
			return fmt.Errorf("tts: %d line(s) for %s ended in an error, see clips.jsonl", target.Failed, target.Target)
		}
	}
	return nil
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
