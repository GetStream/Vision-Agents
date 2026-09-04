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
	benchtarget "github.com/GetStream/Vision-Agents/benchmark/internal/target"
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
			model = envOr("VOICEBENCH_MODEL", benchtarget.DefaultAcceleratedModel)
		case target == "livekit" && cfg.SpawnTarget:
			if envOr("VOICEBENCH_LIVEKIT_PIPELINE", benchtarget.DefaultLiveKitPipeline) == "realtime" {
				model = envOr("VOICEBENCH_LIVEKIT_MODEL", benchtarget.DefaultLiveKitRealtimeModel)
			} else {
				model = envOr("VOICEBENCH_LIVEKIT_MODEL", benchtarget.DefaultLiveKitModel)
			}
		}
	}
	if voice == "" {
		switch {
		case target == "python" && cfg.SpawnTarget:
			voice = "marin"
		case target == "livekit" && cfg.SpawnTarget:
			if envOr("VOICEBENCH_LIVEKIT_PIPELINE", benchtarget.DefaultLiveKitPipeline) == "realtime" {
				voice = envOr("VOICEBENCH_LIVEKIT_VOICE", benchtarget.DefaultLiveKitRealtimeVoice)
			} else {
				voice = envOr("VOICEBENCH_LIVEKIT_VOICE", benchtarget.DefaultLiveKitVoice)
			}
		}
	}
	stt := envOr("VOICEBENCH_STT", "")
	tts := envOr("VOICEBENCH_TTS", "")
	llm := envOr("VOICEBENCH_MODEL", model)
	subagent := envOr("VOICEBENCH_SUBAGENT", "")
	if target == "accelerated" && cfg.SpawnTarget {
		if stt == "" {
			stt = benchtarget.DefaultAcceleratedSTT
		}
		if tts == "" {
			tts = benchtarget.DefaultAcceleratedTTS
		}
		if llm == "" {
			llm = benchtarget.DefaultAcceleratedModel
		}
		if subagent == "" {
			subagent = benchtarget.DefaultAcceleratedSubagent
		}
	}
	if target == "livekit" && cfg.SpawnTarget && envOr("VOICEBENCH_LIVEKIT_PIPELINE", benchtarget.DefaultLiveKitPipeline) != "realtime" {
		stt = envOr("VOICEBENCH_LIVEKIT_STT", benchtarget.DefaultLiveKitSTT)
		tts = envOr("VOICEBENCH_LIVEKIT_TTS", benchtarget.DefaultLiveKitTTS)
		llm = envOr("VOICEBENCH_LIVEKIT_MODEL", benchtarget.DefaultLiveKitModel)
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
		TargetSTT:                stt,
		TargetLLM:                llm,
		TargetTTS:                tts,
		TargetSubagent:           subagent,
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
	accel := filepath.Join(root, "agents", "accelerated", pack)
	_ = filepath.Walk(accel, func(path string, info os.FileInfo, err error) error {
		if err != nil || info == nil || info.IsDir() {
			return err
		}
		if !strings.HasSuffix(path, ".md") {
			return nil
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		content = append(content, raw...)
		content = append(content, 0)
		return nil
	})
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
