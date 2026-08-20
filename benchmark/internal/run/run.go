package run

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/report"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
	"github.com/GetStream/Vision-Agents/benchmark/internal/synth"
	benchtarget "github.com/GetStream/Vision-Agents/benchmark/internal/target"
	"github.com/GetStream/Vision-Agents/benchmark/internal/world"
)

// Config is a benchmark run.
type Config struct {
	Root       string
	OutDir     string
	Pack       string
	ScenarioID string
	K          int
	WorldAddr  string
	CallID     string
	CallType   string
	AgentURL   string
	UserID     string
	System     string
	SpawnAgent bool
	SpawnAccel bool
	AccelBin   string
	AccelURL   string
	AgentPort  int
	WorldURL   string
	Target     benchtarget.Target
	SkipSTT    bool
	SkipJudge  bool
	Logger     *slog.Logger
}

// Run executes a pack and writes a report.
func Run(ctx context.Context, cfg Config) (report.Summary, error) {
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.K <= 0 {
		cfg.K = 3
	}
	if cfg.Root == "" {
		cfg.Root = "."
	}
	if cfg.WorldAddr == "" {
		cfg.WorldAddr = "127.0.0.1:8090"
	}
	if cfg.System == "" {
		cfg.System = "vision-agents"
	}
	if cfg.SpawnAgent && cfg.SpawnAccel {
		return report.Summary{}, fmt.Errorf("run: --spawn-agent and --spawn-accel cannot both be set")
	}
	if cfg.SpawnAgent {
		if cfg.AgentPort <= 0 {
			cfg.AgentPort = 8000
		}
		if cfg.AgentURL == "" {
			cfg.AgentURL = fmt.Sprintf("http://127.0.0.1:%d", cfg.AgentPort)
		}
	}
	if cfg.SpawnAccel || cfg.AccelURL != "" {
		if cfg.System == "vision-agents" {
			cfg.System = "acceleration"
		}
	}
	if cfg.SpawnAccel {
		if cfg.AccelURL == "" {
			cfg.AccelURL = "http://127.0.0.1:8080"
		}
		if cfg.AccelBin == "" {
			cfg.AccelBin = os.Getenv("ACCEL_ROUTER")
		}
		if cfg.AccelBin == "" {
			return report.Summary{}, fmt.Errorf(
				"run: --accel-bin or ACCEL_ROUTER is required with --spawn-accel (CGO_ENABLED=1 go build -o /tmp/accel-router ./cmd/router)")
		}
	}

	packDir := filepath.Join(cfg.Root, "scenarios", cfg.Pack)
	scenarios, err := scenario.LoadPack(packDir)
	if err != nil {
		return report.Summary{}, err
	}
	if cfg.ScenarioID != "" {
		var filtered []scenario.Scenario
		for _, sc := range scenarios {
			if sc.ID == cfg.ScenarioID {
				filtered = append(filtered, sc)
			}
		}
		if len(filtered) == 0 {
			return report.Summary{}, fmt.Errorf("run: unknown scenario %s", cfg.ScenarioID)
		}
		scenarios = filtered
	}

	worldSrv := world.New(cfg.Logger)
	if err := worldSrv.ListenAndServe(cfg.WorldAddr); err != nil {
		return report.Summary{}, err
	}
	defer worldSrv.Close()
	cfg.WorldURL = "http://" + worldSrv.Addr

	runID := time.Now().UTC().Format("20060102T150405Z")
	out := cfg.OutDir
	if out == "" {
		out = filepath.Join(cfg.Root, "out", runID)
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return report.Summary{}, err
	}

	if cfg.Target == nil {
		cfg.Target = buildTarget(cfg)
	}
	stopTarget, err := cfg.Target.Prepare(ctx)
	if err != nil {
		return report.Summary{}, err
	}
	defer stopTarget()

	var calls []report.CallResult
	for _, sc := range scenarios {
		for trial := 1; trial <= cfg.K; trial++ {
			res, err := runOnce(ctx, cfg, worldSrv, sc, trial, out)
			if err != nil {
				cfg.Logger.Error("trial failed", "scenario", sc.ID, "trial", trial, "err", err)
				res.Error = err.Error()
			}
			calls = append(calls, res)
		}
	}
	sum := report.BuildSummary(cfg.System, runID, cfg.K, calls)
	if err := report.Write(out, sum); err != nil {
		return sum, err
	}
	cfg.Logger.Info("wrote report", "dir", out)
	return sum, nil
}

func buildTarget(cfg Config) benchtarget.Target {
	if cfg.SpawnAccel || cfg.AccelURL != "" {
		return &benchtarget.Acceleration{
			Root:     cfg.Root,
			Pack:     cfg.Pack,
			URL:      cfg.AccelURL,
			Spawn:    cfg.SpawnAccel,
			Bin:      cfg.AccelBin,
			WorldURL: cfg.WorldURL,
			Logger:   cfg.Logger,
		}
	}
	if cfg.SpawnAgent || cfg.AgentURL != "" {
		return &benchtarget.Python{
			Root:     cfg.Root,
			Pack:     cfg.Pack,
			URL:      cfg.AgentURL,
			Spawn:    cfg.SpawnAgent,
			Port:     cfg.AgentPort,
			WorldURL: cfg.WorldURL,
			Logger:   cfg.Logger,
		}
	}
	return benchtarget.Noop{}
}

func runOnce(ctx context.Context, cfg Config, worldSrv *world.Server, sc scenario.Scenario, trial int, out string) (report.CallResult, error) {
	result := report.CallResult{ScenarioID: sc.ID, Pack: sc.Pack, Category: string(sc.Category), Trial: trial}
	callDir := filepath.Join(out, fmt.Sprintf("%s-t%d", sc.ID, trial))
	if err := os.MkdirAll(callDir, 0o755); err != nil {
		return result, err
	}
	result.Dir = callDir
	worldSrv.Seed(sc)

	audioMap := map[string][]int16{}
	for _, text := range sc.SpeechTexts() {
		pcm, err := synth.LoadOrSynth(cfg.Root, "", text)
		if err != nil {
			return result, fmt.Errorf("run: tts required for caller speech: %w", err)
		}
		audioMap[text] = pcm
	}

	rec, err := runWebRTC(ctx, cfg, sc, audioMap, trial)
	if err != nil {
		return result, err
	}

	if err := audio.WriteWAV(filepath.Join(callDir, "caller.wav"), audio.PCM{Rate: rec.Rate, Samples: rec.Caller}); err != nil {
		return result, err
	}
	if err := audio.WriteWAV(filepath.Join(callDir, "agent.wav"), audio.PCM{Rate: rec.Rate, Samples: rec.Agent}); err != nil {
		return result, err
	}
	if err := audio.WriteStereoWAV(filepath.Join(callDir, "mixed.wav"), rec.Rate, rec.Caller, rec.Agent); err != nil {
		return result, err
	}

	sess := worldSrv.Snapshot()
	metrics := score.Metrics{}
	metrics.V2V = score.TimingFromRecording(rec)
	if rec.Rate > 0 {
		metrics.CallDurationMS = len(rec.Caller) * 1000 / rec.Rate
	}
	if sess != nil {
		metrics.ToolCount = len(sess.Tools)
		for _, tool := range sess.Tools {
			if tool.Error != "" {
				metrics.ToolErrorCount++
			}
			metrics.ToolWaitMS += tool.DurationMS
			if tool.DurationMS > metrics.MaxToolWaitMS {
				metrics.MaxToolWaitMS = tool.DurationMS
			}
		}
		score.MarkToolTurns(&metrics, rec, sess.Tools)
	}
	score.SummarizeTiming(&metrics)
	metrics.BargeInStopMS = score.BargeInStopMS(rec)
	metrics.SelectivityHold = score.SelectivityHold(rec)
	metrics.HoldThroughOverlap = score.HoldThroughOverlap(rec)
	metrics.FalseCutoff = score.FalseCutoff(rec)

	agentText := ""
	callerText := ""
	if cfg.SkipSTT {
		metrics.ScoringFail = append(metrics.ScoringFail, "stt")
	} else {
		var err error
		agentText, err = score.TranscribeDeepgram(audio.PCM{Rate: rec.Rate, Samples: rec.Agent})
		if err != nil {
			metrics.ScoringFail = append(metrics.ScoringFail, "stt")
			cfg.Logger.Warn("stt failed", "leg", "agent", "err", err)
		}
		callerText, err = score.TranscribeDeepgram(audio.PCM{Rate: rec.Rate, Samples: rec.Caller})
		if err != nil {
			metrics.ScoringFail = append(metrics.ScoringFail, "stt")
			cfg.Logger.Warn("stt failed", "leg", "caller", "err", err)
		}
	}
	score.WorldGates(&metrics, sc, sess, agentText)

	if cfg.SkipJudge {
		metrics.ScoringFail = append(metrics.ScoringFail, "judge")
	} else {
		var tools []world.ToolCall
		if sess != nil {
			tools = sess.Tools
		}
		verdict, jerr := score.Judge(sc, sc.CallerTranscript(), agentText, tools)
		if jerr != nil {
			metrics.ScoringFail = append(metrics.ScoringFail, "judge")
			cfg.Logger.Warn("judge failed", "err", jerr)
		} else {
			metrics.PolicyFail = verdict.PolicyFail
			metrics.SayDoFail = verdict.SayDoFail
			if sc.Judge.Coherence && !verdict.Coherent {
				metrics.PolicyFail = append(metrics.PolicyFail, "incoherent")
			}
		}
	}

	score.ScoreFiller(&metrics, sc, rec, sess, agentText)
	score.ApplyGates(&metrics, sc)
	result.Metrics = metrics
	result.Passed = metrics.Passed

	if err := writeJSON(filepath.Join(callDir, "metrics.json"), metrics); err != nil {
		return result, err
	}
	if err := writeJSON(filepath.Join(callDir, "result.json"), result); err != nil {
		return result, err
	}
	if sess != nil {
		if err := writeJSON(filepath.Join(callDir, "tools.json"), sess.Tools); err != nil {
			return result, err
		}
		if err := writeJSON(filepath.Join(callDir, "state.json"), sess.State); err != nil {
			return result, err
		}
	}
	if err := writeJSON(filepath.Join(callDir, "transcript.json"), map[string]string{"caller": callerText, "agent": agentText}); err != nil {
		return result, err
	}
	return result, nil
}

func webrtcCallID(cfg Config, sc scenario.Scenario, trial int) string {
	if cfg.CallID != "" {
		if cfg.K > 1 {
			return fmt.Sprintf("%s-t%d", cfg.CallID, trial)
		}
		return cfg.CallID
	}
	id := strings.ReplaceAll(sc.ID, ".", "-")
	return fmt.Sprintf("vb-%s-%d-%s", id, trial, randomToken())
}

func writeJSON(path string, v any) error {
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

func randomToken() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}
