package run

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/report"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
	"github.com/GetStream/Vision-Agents/benchmark/internal/synth"
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
	AgentPort  int
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
	if cfg.SpawnAgent {
		if cfg.AgentPort <= 0 {
			cfg.AgentPort = 8000
		}
		if cfg.AgentURL == "" {
			cfg.AgentURL = fmt.Sprintf("http://127.0.0.1:%d", cfg.AgentPort)
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
	worldURL := "http://" + worldSrv.Addr

	runID := time.Now().UTC().Format("20060102T150405Z")
	out := cfg.OutDir
	if out == "" {
		out = filepath.Join(cfg.Root, "out", runID)
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return report.Summary{}, err
	}

	if cfg.SpawnAgent {
		stop, err := startAgentProc(ctx, cfg, worldURL)
		if err != nil {
			return report.Summary{}, err
		}
		defer stop()
	}

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
	if sess != nil {
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
		verdict, jerr := score.Judge(sc, callerText, agentText, tools)
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
	score.ApplyGates(&metrics)
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

func startAgentProc(ctx context.Context, cfg Config, worldURL string) (func(), error) {
	port := cfg.AgentPort
	cmd := exec.CommandContext(ctx, "uv", "run", "python", "-m", "voicebench_agents", cfg.Pack,
		"--host", "127.0.0.1", "--port", strconv.Itoa(port))
	cmd.Dir = filepath.Join(cfg.Root, "agents")
	env := make([]string, 0, len(os.Environ())+1)
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, "WORLD_URL=") {
			continue
		}
		env = append(env, e)
	}
	cmd.Env = append(env, "WORLD_URL="+worldURL)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("run: spawn agent: %w", err)
	}
	stop := func() {
		if cmd.Process == nil {
			return
		}
		_ = cmd.Process.Signal(os.Interrupt)
		done := make(chan struct{})
		go func() {
			_ = cmd.Wait()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			_ = cmd.Process.Kill()
		}
	}
	readyURL := fmt.Sprintf("http://127.0.0.1:%d/ready", port)
	client := &http.Client{Timeout: time.Second}
	deadline := time.Now().Add(120 * time.Second)
	for time.Now().Before(deadline) {
		if err := ctx.Err(); err != nil {
			stop()
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, readyURL, nil)
		if err != nil {
			stop()
			return nil, err
		}
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				cfg.Logger.Info("spawned agent ready", "url", cfg.AgentURL)
				return stop, nil
			}
		}
		time.Sleep(250 * time.Millisecond)
	}
	stop()
	return nil, fmt.Errorf("run: agent did not become ready at %s", readyURL)
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

func startAgentSession(ctx context.Context, agentURL, callID, callType string) (string, error) {
	base := strings.TrimRight(agentURL, "/")
	body, _ := json.Marshal(map[string]string{"call_type": callType})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/calls/"+callID+"/sessions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("run: start agent session: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("run: start agent session: HTTP %d", resp.StatusCode)
	}
	var parsed struct {
		SessionID string `json:"session_id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", fmt.Errorf("run: start agent session: %w", err)
	}
	if parsed.SessionID == "" {
		return "", fmt.Errorf("run: start agent session: missing session_id")
	}
	return parsed.SessionID, nil
}

func closeAgentSession(ctx context.Context, logger *slog.Logger, agentURL, callID, sessionID string) {
	if sessionID == "" {
		return
	}
	if logger == nil {
		logger = slog.Default()
	}
	base := strings.TrimRight(agentURL, "/")
	url := base + "/calls/" + callID + "/sessions/" + sessionID
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		logger.Warn("close agent session", "err", err)
		return
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		logger.Warn("close agent session", "err", err)
		return
	}
	_ = resp.Body.Close()

	client := &http.Client{Timeout: time.Second}
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if err := ctx.Err(); err != nil {
			return
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return
		}
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusNotFound {
				return
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
	logger.Warn("agent session still open after close", "session", sessionID, "call", callID)
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
