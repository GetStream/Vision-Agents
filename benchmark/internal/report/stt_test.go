package report

import (
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

func TestSummarizeSTTPoolsWordsAndTimesOnlyWhatReturned(t *testing.T) {
	clip := func(ref, hyp string, returned bool, settle int) STTClip {
		return STTClip{Reference: ref, Hypothesis: hyp, Returned: returned, Timed: true, ToSettleMs: settle, ToFirstWordsMs: settle / 2, WhileSpeaking: 2,
			Provider: "deepgram", Model: "nova-3",
			Raw: score.ScoreWER(ref, hyp, false), Normalized: score.ScoreWER(ref, hyp, true)}
	}
	sum := SummarizeSTT("deepgram/nova-3", []STTClip{
		clip("one two three four five six seven eight nine ten", "one two three four five six seven eight nine ten", true, 200),
		clip("hello there", "hello", true, 400),
		clip("anything at all", "", false, 0),
	})
	if sum.PooledWER != 4.0/15 {
		t.Errorf("pooled WER = %v, want 4/15: errors over reference words, not a mean of clips", sum.PooledWER)
	}
	if sum.ReturnedRate != 2.0/3 || sum.PerfectRate != 1.0/3 {
		t.Errorf("returned %v, perfect %v", sum.ReturnedRate, sum.PerfectRate)
	}
	if sum.TimedClips != 2 || sum.ToSettleP50Ms != 200 || sum.ToSettleP95Ms != 400 {
		t.Errorf("a clip that returned nothing has no TTFS to pool: %+v", sum)
	}
	if strings.Join(sum.ProvidersAnswering, ",") != "deepgram/nova-3" {
		t.Errorf("answering = %v", sum.ProvidersAnswering)
	}
}

func TestSTTAndTTSPostOneLinePerTarget(t *testing.T) {
	stt := STTSlackText("Voicebench STT", Summary{STT: []STTSummary{{Target: "deepgram/flux-general-en", Clips: 48, PooledWER: 0.031, PooledWERRaw: 0.05, TimedClips: 48, ToSettleP50Ms: 1700, ToSettleP95Ms: 2400, ToFirstWordsP50Ms: 1100}}})
	if !strings.Contains(stt, "• deepgram/flux-general-en: WER 3.1% (raw 5.0%) · TTFS P50 1.70 s, P95 2.40 s · first words P50 1.10 s") {
		t.Fatalf("stt text:\n%s", stt)
	}
	tts := TTSSlackText("Voicebench TTS", Summary{TTS: []TTSSummary{{Target: "elevenlabs/eleven_v4_turbo", Clips: 40, TTFBP50Ms: 700, TTFBP95Ms: 900, RTFP50: 0.19, PooledWER: 0.05, Grade: HealthGood, Good: 40, Failed: 1}}})
	if !strings.Contains(tts, "• elevenlabs/eleven_v4_turbo: TTFB P50 700 ms") || !strings.Contains(tts, "health good (40/0/0) · 1 failed") {
		t.Fatalf("tts text:\n%s", tts)
	}
}
