package run

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/GetStream/Vision-Agents/benchmark/internal/report"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
	"github.com/GetStream/Vision-Agents/benchmark/internal/synth"
)

func buildManifest(cfg Config, scenarios []scenario.Scenario) report.RunManifest {
	commit, dirty := gitState(cfg.Root)
	target := cfg.TargetName
	if target == "" {
		target = "external"
	}
	model := cfg.TargetModel
	voice := cfg.TargetVoice
	if model == "" {
		switch {
		case target == "python" && cfg.SpawnTarget:
			model = "gpt-realtime-2"
		case target == "accelerated" && cfg.SpawnTarget:
			model = envOr("VOICEBENCH_MODEL", "")
		case target == "livekit" && cfg.SpawnTarget:
			model = envOr("VOICEBENCH_LIVEKIT_MODEL", "gpt-realtime-2")
		}
	}
	if voice == "" {
		switch {
		case target == "python" && cfg.SpawnTarget:
			voice = "marin"
		case target == "livekit" && cfg.SpawnTarget:
			voice = envOr("VOICEBENCH_LIVEKIT_VOICE", "marin")
		}
	}
	scenarioJSON, _ := json.Marshal(scenarios)
	calibration, _ := score.LoadCalibrationSet(filepath.Join(cfg.Root, "calibration", "judge.json"))
	return report.RunManifest{
		GitCommit:                commit,
		GitDirty:                 dirty,
		ScenarioHash:             digest(scenarioJSON),
		ContractHash:             contractHash(cfg.Root, cfg.Pack),
		Transport:                envOrValue(cfg.Transport, transportStream),
		Target:                   target,
		TargetModel:              model,
		TargetVoice:              voice,
		TargetSTT:                envOr("VOICEBENCH_STT", ""),
		TargetLLM:                envOr("VOICEBENCH_MODEL", model),
		TargetTTS:                envOr("VOICEBENCH_TTS", ""),
		CallerModel:              synth.CallerModel,
		CallerVoice:              synth.VoiceID(""),
		GoVersion:                runtime.Version(),
		NetworkProfile:           cfg.NetworkProfile,
		JudgeCalibrationHash:     calibration.Hash,
		JudgeCalibrationReviewer: calibration.ReviewedBy,
		Command:                  append([]string{"voicebench"}, os.Args[1:]...),
	}
}

func contractHash(root, pack string) string {
	var content []byte
	for _, suffix := range []string{".prompt", ".tools.yaml"} {
		raw, err := os.ReadFile(filepath.Join(root, "agents", "contracts", pack+suffix))
		if err != nil {
			continue
		}
		content = append(content, raw...)
		content = append(content, 0)
	}
	return digest(content)
}

func digest(content []byte) string {
	sum := sha256.Sum256(content)
	return hex.EncodeToString(sum[:])
}

func gitState(root string) (string, bool) {
	commitRaw, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", false
	}
	statusRaw, err := exec.Command("git", "-C", root, "status", "--porcelain", "--", ".").Output()
	return strings.TrimSpace(string(commitRaw)), err == nil && len(statusRaw) > 0
}

func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func envOrValue(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}
