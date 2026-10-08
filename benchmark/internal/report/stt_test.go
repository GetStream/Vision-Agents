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
