package chatlog

import (
	"cmp"
	"fmt"
	"math"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
)

// timings is how long each stage of a turn took, shown on the reply to it so that somebody
// talking to the agent can see how fast each step was without reading its logs.
type timings struct {
	// line is written after the reply's text.
	line string
	// fields are the same legs in whole milliseconds, for a client that would rather read
	// them than parse the line.
	fields map[string]any
}

// timingsOf describes a finished turn. It reports false for one with nothing to say: it was
// not cut off, and none of its legs happened.
//
// The figure the line leads with is the delay the caller felt, from the end of their speech to
// the first sound of the reply they could hear, or to the first audio published where the edge
// does not say when that was. A leg that did not happen is left out, not shown as zero.
func timingsOf(turn agent.Turn) (timings, bool) {
	fields := map[string]any{"interrupted": turn.Interrupted}
	leg := func(key string, ms float64) int {
		rounded := int(math.Round(ms))
		if rounded <= 0 {
			return 0
		}
		fields[key] = rounded
		return rounded
	}

	total := leg("voice_to_voice_ms", cmp.Or(turn.SpeechEndToAudibleMs, turn.SpeechEndToAudioMs, turn.RoundtripMs))
	leg("roundtrip_ms", turn.RoundtripMs)
	leg("speech_end_to_audio_ms", turn.SpeechEndToAudioMs)
	leg("speech_end_to_audible_ms", turn.SpeechEndToAudibleMs)
	stt := leg("stt_ms", turn.STTLatencyMs)
	cadence := leg("cadence_ms", turn.CadenceMs)
	decision := leg("decision_ms", turn.DecisionMs)
	model := leg("model_to_first_text_ms", turn.ModelToFirstTextMs)
	ttft := leg("llm_ttft_ms", turn.LLMTTFTMs)
	toTTS := leg("text_to_tts_ms", turn.TextToTTSMs)
	voice := leg("tts_to_audio_ms", turn.TTSToAudioMs)
	leg("tts_ttfb_ms", turn.TTSTTFBMs)
	hold := leg("reply_hold_ms", turn.ReplyHoldMs)
	leg("first_frame_queued_ms", turn.FirstFrameQueuedMs)
	leg("first_audible_frame_ms", turn.FirstAudibleFrameMs)
	// Both run from the same moment, so what lies between is the edge taking the audio that was
	// published and playing it.
	var edge int
	if turn.RoundtripMs > 0 && turn.FirstAudibleFrameMs > 0 {
		edge = leg("publish_to_audible_ms", turn.FirstAudibleFrameMs-turn.RoundtripMs)
	}
	leg("audio_out_ms", turn.AudioOutMs)
	leg("audio_dropped_ms", turn.AudioDroppedMs)

	var parts []string
	if turn.Interrupted {
		parts = append(parts, "interrupted")
	}
	if total > 0 {
		parts = append(parts, fmt.Sprintf("%d ms", total))
	}
	shown := func(label string, ms int) {
		if ms > 0 {
			parts = append(parts, fmt.Sprintf("%s %d", label, ms))
		}
	}
	shown("stt", stt)
	shown("wait", cadence)
	shown("eot", decision)
	switch {
	case model > 0 && ttft > 0:
		parts = append(parts, fmt.Sprintf("llm %d (ttft %d)", model, ttft))
	case model > 0:
		shown("llm", model)
	default:
		shown("ttft", ttft)
	}
	shown("→tts", toTTS)
	shown("tts", voice)
	shown("audio", edge)
	shown("hold", hold)

	if len(parts) == 0 {
		return timings{}, false
	}
	return timings{line: conversation.TimingsMark + " " + strings.Join(parts, " · "), fields: fields}, true
}
