package chatlog

import (
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
	// fields are the same figures in whole milliseconds, for a client that would rather read
	// them than parse the line.
	fields map[string]any
}

// timingsOf describes a finished turn as the wait the caller felt and the three stages it is
// made of, which add up to it:
//
//   - reply: from the end of the caller's speech to the first audible frame of the reply, or to
//     the first audio published where the edge does not say when that was heard;
//   - eou, end of utterance: from the end of their speech until the turn was committed to (the
//     transcriber's settling, the cadence wait and the floor decision);
//   - llm: from the commitment to the reply's first text, which is nothing when a reply started
//     beside the decision was ready by then;
//   - tts: from the first text to the first audible frame (handing it to the voice, the voice's
//     first audio, any hold for the caller to be quiet, and the edge playing it out).
//
// The providers' own first-token and first-byte waits follow, because work started early hides
// them inside the stages. It reports false for a turn with nothing to say: it was not cut off,
// and none of it happened. A figure that did not happen is left out, not shown as zero.
func timingsOf(turn agent.Turn) (timings, bool) {
	ms := func(v float64) int { return max(0, int(math.Round(v))) }

	reply, heard := ms(turn.SpeechEndToAudibleMs), true
	if reply == 0 {
		reply, heard = ms(turn.SpeechEndToAudioMs), false
	}
	stt := ms(turn.STTLatencyMs)
	if reply == 0 && turn.RoundtripMs > 0 {
		// The roundtrip starts at the transcript, so the transcriber's settling is not in it.
		reply, stt = ms(turn.RoundtripMs), 0
	}
	wait, eot := ms(turn.CadenceMs), ms(turn.DecisionMs)
	eou, llm := stt+wait+eot, ms(turn.ModelToFirstTextMs)
	tts := ms(turn.TextToTTSMs + turn.TTSToAudioMs)
	if heard && turn.FirstAudibleFrameMs > 0 {
		tts = ms(turn.FirstAudibleFrameMs - turn.CadenceMs - turn.DecisionMs - turn.ModelToFirstTextMs)
	}
	ttft, ttfb, hold := ms(turn.LLMTTFTMs), ms(turn.TTSTTFBMs), ms(turn.ReplyHoldMs)

	fields := map[string]any{"interrupted": turn.Interrupted}
	set := func(key string, v int) {
		if v > 0 {
			fields[key] = v
		}
	}
	set("reply_ms", reply)
	if reply > 0 {
		fields["reply_heard"] = heard
	}
	set("eou_ms", eou)
	set("stt_ms", stt)
	set("wait_ms", wait)
	set("eot_ms", eot)
	set("llm_ms", llm)
	set("tts_ms", tts)
	set("llm_ttft_ms", ttft)
	set("tts_ttfb_ms", ttfb)
	set("hold_ms", hold)

	var stages []string
	for _, stage := range []struct {
		name string
		ms   int
	}{{"eou", eou}, {"llm", llm}, {"tts", tts}} {
		if stage.ms > 0 {
			stages = append(stages, fmt.Sprintf("%s %d", stage.name, stage.ms))
		}
	}
	var parts []string
	if turn.Interrupted {
		parts = append(parts, "interrupted")
	}
	switch {
	case reply > 0 && len(stages) > 0:
		parts = append(parts, fmt.Sprintf("reply %d ms = %s", reply, strings.Join(stages, " + ")))
	case reply > 0:
		parts = append(parts, fmt.Sprintf("reply %d ms", reply))
	default:
		parts = append(parts, stages...)
	}
	for _, provider := range []struct {
		name string
		ms   int
	}{{"ttft", ttft}, {"ttfb", ttfb}, {"hold", hold}} {
		if provider.ms > 0 {
			parts = append(parts, fmt.Sprintf("%s %d", provider.name, provider.ms))
		}
	}

	if len(parts) == 0 {
		return timings{}, false
	}
	return timings{line: conversation.TimingsMark + " " + strings.Join(parts, " · "), fields: fields}, true
}
