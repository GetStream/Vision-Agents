package run

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
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

// Transports Voicebench can run the scripted caller over.
const (
	transportStream  = "stream"
	transportLiveKit = "livekit"
)

// Config is a benchmark run.
type Config struct {
	Root              string
	OutDir            string
	Pack              string
	ScenarioID        string
	K                 int
	WorldAddr         string
	CallID            string
	CallType          string
	Transport         string
	UserID            string
	System            string
	TargetName        string
	TargetURL         string
	TargetModel       string
	TargetVoice       string
	TargetBin         string
	SpawnTarget       bool
	LiveKitAgentName  string
	LiveKitDeployment string
	WorldURL          string
	NetworkProfile    string
	Frozen            bool
	Target            benchtarget.Target
	SkipSTT           bool
	SkipJudge         bool
	Logger            *slog.Logger
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
	if cfg.TargetName != "" {
		definition, ok := benchtarget.Lookup(cfg.TargetName)
		if !ok {
			return report.Summary{}, fmt.Errorf("run: unknown --target %s", cfg.TargetName)
		}
		cfg.Transport = definition.Transport
		if cfg.System == "" {
			cfg.System = definition.System
		}
	}
	if cfg.System == "" {
		cfg.System = "external"
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
	if cfg.Frozen {
		ids, err := scenario.LoadIDList(scenario.FrozenPath(cfg.Root))
		if err != nil {
			return report.Summary{}, err
		}
		scenarios, err = scenario.Filter(scenarios, ids)
		if err != nil {
			return report.Summary{}, err
		}
	}

	worldSrv := world.New(cfg.Logger)
	if err := worldSrv.ListenAndServe(cfg.WorldAddr); err != nil {
		return report.Summary{}, err
	}
	defer worldSrv.Close()
	if cfg.WorldURL == "" {
		cfg.WorldURL = "http://" + worldSrv.Addr
	}

	if cfg.Target == nil {
		if cfg.TargetName == "" {
			cfg.Target = benchtarget.Noop{}
		} else {
			cfg.Target, err = benchtarget.Build(cfg.TargetName, benchtarget.Config{
				Root:              cfg.Root,
				Pack:              cfg.Pack,
				URL:               cfg.TargetURL,
				Bin:               cfg.TargetBin,
				WorldURL:          cfg.WorldURL,
				Spawn:             cfg.SpawnTarget,
				LiveKitAgentName:  cfg.LiveKitAgentName,
				LiveKitDeployment: cfg.LiveKitDeployment,
				Logger:            cfg.Logger,
			})
			if err != nil {
				return report.Summary{}, fmt.Errorf("run: %w", err)
			}
		}
	}
	stopTarget, err := cfg.Target.Prepare(ctx)
	if err != nil {
		return report.Summary{}, err
	}
	defer stopTarget()

	started := time.Now().UTC()
	runID := started.Format("20060102T150405Z")
	out := cfg.OutDir
	if out == "" {
		out = filepath.Join(cfg.Root, "out", runID)
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return report.Summary{}, err
	}

	var calls []report.CallResult
	for _, sc := range scenarios {
		for trial := 1; trial <= cfg.K; trial++ {
			res, err := runOnce(ctx, cfg, worldSrv, sc, trial, out)
			if err != nil {
				res.Error = err.Error()
				var targetErr targetFailure
				if errors.As(err, &targetErr) {
					cfg.Logger.Error("target failed", "scenario", sc.ID, "trial", trial, "err", err)
					res.Outcome = report.OutcomeFail
					res.Metrics.GateNotes = append(res.Metrics.GateNotes, "target")
				} else {
					cfg.Logger.Error("trial invalid", "scenario", sc.ID, "trial", trial, "err", err)
					res.Outcome = report.OutcomeInvalid
					res.InvalidReason = append(res.InvalidReason, err.Error())
				}
			}
			if persistErr := persistTrialResult(res); persistErr != nil {
				cfg.Logger.Error("persist trial result", "scenario", sc.ID, "trial", trial, "err", persistErr)
				res.Error = persistErr.Error()
				res.Outcome = report.OutcomeInvalid
				res.Passed = false
				res.InvalidReason = append(res.InvalidReason, persistErr.Error())
			}
			calls = append(calls, res)
		}
	}
	sum := report.BuildSummary(cfg.System, runID, cfg.K, calls)
	sum.Started = started
	sum.Manifest = buildManifest(cfg, scenarios)
	if err := report.Write(out, sum); err != nil {
		return sum, err
	}
	cfg.Logger.Info("wrote report", "dir", out)
	return sum, nil
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

	rec, callErr := runWebRTC(ctx, cfg, sc, audioMap, trial)
	if rec.Rate > 0 {
		if err := audio.WriteWAV(filepath.Join(callDir, "caller.wav"), audio.PCM{Rate: rec.Rate, Samples: rec.Caller}); err != nil {
			return result, err
		}
		if err := audio.WriteWAV(filepath.Join(callDir, "agent.wav"), audio.PCM{Rate: rec.Rate, Samples: rec.Agent}); err != nil {
			return result, err
		}
		if err := audio.WriteStereoWAV(filepath.Join(callDir, "mixed.wav"), rec.Rate, rec.Caller, rec.Agent); err != nil {
			return result, err
		}
	}

	sess := worldSrv.Snapshot()
	metrics := score.Metrics{}
	metrics.V2V, metrics.Dropped = score.TimingFromRecording(rec)
	metrics.CallDurationMS = rec.DurationMS()
	metrics.ClockDriftMS = rec.ClockDriftMS()
	metrics.InboundDropped = rec.InboundDropped
	metrics.RequestedSNRDB = rec.RequestedSNRDB
	metrics.MeasuredSNRDB = rec.MeasuredSNRDB
	if sess != nil {
		metrics.WorldContact = sess.Contacted
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
	metrics.OverlapChecks = score.ScoreOverlaps(rec)
	metrics.SelectivityHold = score.SelectivityHold(metrics.OverlapChecks)
	metrics.HoldThroughOverlap = score.HoldThroughOverlap(metrics.OverlapChecks)
	metrics.FalseCutoff = score.FalseCutoff(rec)
	result.Metrics = metrics

	if sess != nil {
		if err := writeJSON(filepath.Join(callDir, "tools.json"), sess.Tools); err != nil {
			return result, err
		}
		if err := writeJSON(filepath.Join(callDir, "state.json"), sess.State); err != nil {
			return result, err
		}
	}
	if err := writeJSON(filepath.Join(callDir, "events.json"), rec.Events); err != nil {
		return result, err
	}
	if callErr != nil {
		return result, callErr
	}

	agentTranscript := score.Transcript{}
	callerTranscript := score.Transcript{}
	if cfg.SkipSTT {
		result.InvalidReason = append(result.InvalidReason, "agent stt skipped")
	} else {
		var err error
		agentTranscript, err = score.TranscribeDeepgram(audio.PCM{Rate: rec.Rate, Samples: rec.Agent})
		if err != nil {
			result.InvalidReason = append(result.InvalidReason, "agent stt: "+err.Error())
			cfg.Logger.Warn("stt failed", "leg", "agent", "err", err)
		}
		callerTranscript, err = score.TranscribeDeepgram(audio.PCM{Rate: rec.Rate, Samples: rec.Caller})
		if err != nil {
			result.Warnings = append(result.Warnings, "diagnostic caller stt failed: "+err.Error())
			cfg.Logger.Warn("diagnostic stt failed", "leg", "caller", "err", err)
		}
	}
	if rec.InboundDropped > 0 {
		result.InvalidReason = append(result.InvalidReason, fmt.Sprintf("inbound audio dropped %d frame(s)", rec.InboundDropped))
	}
	score.WorldGates(&metrics, sc, sess, agentTranscript.Text)
	score.CountConversation(&metrics, rec, agentTranscript.Text)
	if callerTranscript.Text != "" {
		raw := score.ScoreWER(sc.CallerTranscript(), callerTranscript.Text, false)
		norm := score.ScoreWER(sc.CallerTranscript(), callerTranscript.Text, true)
		metrics.CallerWER = raw.WER
		metrics.CallerWERNormalized = norm.WER
	}

	var judgeVerdict *score.JudgeVerdict
	if cfg.SkipJudge {
		result.InvalidReason = append(result.InvalidReason, "judge skipped")
	} else {
		var tools []world.ToolCall
		if sess != nil {
			tools = sess.Tools
		}
		verdict, jerr := score.Judge(sc, sc.CallerTranscript(), agentTranscript.Text, tools)
		if jerr != nil {
			result.InvalidReason = append(result.InvalidReason, "judge: "+jerr.Error())
			cfg.Logger.Warn("judge failed", "err", jerr)
		} else {
			judgeVerdict = &verdict
			metrics.PolicyFail = verdict.PolicyFail
			metrics.SayDoFail = verdict.SayDoFail
			if sc.Judge.Coherence && !verdict.Coherent {
				metrics.PolicyFail = append(metrics.PolicyFail, "incoherent")
			}
		}
	}

	score.ScoreFiller(&metrics, sc, rec, sess, agentTranscript)
	score.ApplyGates(&metrics, sc)
	result.Metrics = metrics
	result.Passed = metrics.Passed && len(result.InvalidReason) == 0
	if len(result.InvalidReason) > 0 {
		result.Outcome = report.OutcomeInvalid
	} else if result.Passed {
		result.Outcome = report.OutcomePass
	} else {
		result.Outcome = report.OutcomeFail
	}
	if !metrics.WorldContact && len(sc.ExpectedTools) > 0 {
		result.Warnings = append(result.Warnings, "target never contacted the world server: verify it received the contract and can reach "+cfg.WorldURL)
	}

	if err := writeJSON(filepath.Join(callDir, "transcript.json"), map[string]score.Transcript{"caller": callerTranscript, "agent": agentTranscript}); err != nil {
		return result, err
	}
	if judgeVerdict != nil {
		if err := writeJSON(filepath.Join(callDir, "judge.json"), judgeVerdict); err != nil {
			return result, err
		}
	}
	return result, nil
}

func persistTrialResult(result report.CallResult) error {
	if err := writeJSON(filepath.Join(result.Dir, "metrics.json"), result.Metrics); err != nil {
		return err
	}
	return writeJSON(filepath.Join(result.Dir, "result.json"), result)
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
