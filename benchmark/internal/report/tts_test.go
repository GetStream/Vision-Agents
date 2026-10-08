package report

import (
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
)

func TestGradeHealthNamesWhatIsWrong(t *testing.T) {
	healthy := audio.Health{DurationMS: 2000, Peak: 12000, LeadSilenceMS: 40, TailSilenceMS: 100, SilenceRatio: 0.1}
	if grade, issues := GradeHealth(healthy); grade != HealthGood || issues != nil {
		t.Fatalf("healthy clip = %s %v", grade, issues)
	}
	slow := healthy
	slow.LeadSilenceMS = 600
	if grade, _ := GradeHealth(slow); grade != HealthWarn {
		t.Fatalf("600 ms before speech should warn, got %s", grade)
	}
	clipped := healthy
	clipped.ClipFraction = 0.01
	if grade, issues := GradeHealth(clipped); grade != HealthFail || len(issues) != 1 {
		t.Fatalf("1%% clipped should fail, got %s %v", grade, issues)
	}
	if grade, _ := GradeHealth(audio.Health{}); grade != HealthFail {
		t.Fatal("no audio is a failed clip")
	}
}

func TestSummarizeTTSGradesTheShareOfHealthyClips(t *testing.T) {
	var clips []TTSClip
	for i := range 40 {
		clips = append(clips, TTSClip{Returned: true, Grade: HealthGood, TTFBMs: 100 + i, SynthesisMs: 400, RTF: 0.2})
	}
	clips = append(clips, TTSClip{Grade: HealthFail, Error: "provider refused"})
	sum := SummarizeTTS("x/y", clips)
	if sum.Good != 40 || sum.Fail != 1 || sum.Failed != 1 || sum.Grade != HealthWarn {
		t.Fatalf("40 of 41 healthy is under 99%%, so warn: %+v", sum)
	}
	if sum.TimedClips != 40 || sum.TTFBP50Ms != 119 || sum.RTFP50 != 0.2 {
		t.Fatalf("latency is over the clips that returned audio: %+v", sum)
	}
}
