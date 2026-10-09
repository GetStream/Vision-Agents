package run

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

type timelineEntry struct {
	TurnID             string   `json:"turn_id"`
	Interrupted        *bool    `json:"interrupted"`
	SttLatencyMs       *float64 `json:"stt_latency_ms"`
	CadenceMs          *float64 `json:"cadence_ms"`
	DecisionMs         *float64 `json:"decision_ms"`
	ModelToFirstTextMs *float64 `json:"model_to_first_text_ms"`
	TextToTtsMs        *float64 `json:"text_to_tts_ms"`
	TtsToAudioMs       *float64 `json:"tts_to_audio_ms"`
	RoundtripMs        *float64 `json:"roundtrip_ms"`
	SpeechEndToAudioMs *float64 `json:"speech_end_to_audio_ms"`
}

// captureRouterTimeline writes the router's timeline of a call and returns the replies in
// it. The router times every turn as consecutive stages, which is the only place to see
// where the wait between the caller stopping and the agent starting went.
func captureRouterTimeline(cfg Config, callID, callDir string) ([]score.StageTiming, error) {
	if cfg.TargetName != "accelerated" && cfg.TargetName != "acceleration" {
		return nil, nil
	}
	base := routerBase(cfg)
	customer := envOr("STREAM_ACCELERATION_CUSTOMER_ID", "voicebench")
	sessionID, err := resolveHeardCallID(base, customer, callID)
	if err != nil {
		return nil, err
	}
	body, err := getJSON(base+"/v1/agents/calls/"+sessionID+"/timeline", customer)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(callDir, "timeline.json"), body, 0o644); err != nil {
		return nil, err
	}
	var entries []timelineEntry
	if err := json.Unmarshal(body, &entries); err != nil {
		return nil, err
	}
	return replyStages(entries), nil
}

// toolTurnPrefix starts the id of the reply the agent begins when a tool returns.
const toolTurnPrefix = "tool-"

// replyStages keeps the turns that answered something: the ones a caller's words started, and
// the replies the agent started after a tool returned, which are marked as such. The latter
// have no transcript to settle, so they only have the stages from the model on, and count once
// they reached audio. A turn the agent began on its own, such as the greeting, is a reply to
// nobody.
func replyStages(entries []timelineEntry) []score.StageTiming {
	ms := func(v *float64) int {
		if v == nil {
			return 0
		}
		return int(math.Round(*v))
	}
	var stages []score.StageTiming
	for _, entry := range entries {
		tool := strings.HasPrefix(entry.TurnID, toolTurnPrefix)
		if tool {
			if entry.TtsToAudioMs == nil {
				continue
			}
		} else if entry.SttLatencyMs == nil || entry.RoundtripMs == nil {
			continue
		}
		stages = append(stages, score.StageTiming{
			TurnID:          entry.TurnID,
			Tool:            tool,
			STTMs:           ms(entry.SttLatencyMs),
			CadenceMs:       ms(entry.CadenceMs),
			DecisionMs:      ms(entry.DecisionMs),
			ModelToTextMs:   ms(entry.ModelToFirstTextMs),
			TextToTTSMs:     ms(entry.TextToTtsMs),
			TTSToAudioMs:    ms(entry.TtsToAudioMs),
			RoundtripMs:     ms(entry.RoundtripMs),
			SpeechToAudioMs: ms(entry.SpeechEndToAudioMs),
			Interrupted:     entry.Interrupted != nil && *entry.Interrupted,
		})
	}
	return stages
}
