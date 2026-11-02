package audio

import "testing"

func TestMeasureHealthFindsLeadingSilence(t *testing.T) {
	rate := Rate
	samples := make([]int16, rate) // 1s of silence
	speech := make([]int16, rate/5)
	for i := range speech {
		speech[i] = 8000
	}
	samples = append(samples, speech...)
	samples = append(samples, make([]int16, rate)...)
	got := MeasureHealth(samples, rate)
	if got.LeadSilenceMS < 800 {
		t.Fatalf("lead %d", got.LeadSilenceMS)
	}
	if got.TailSilenceMS < 800 {
		t.Fatalf("tail %d", got.TailSilenceMS)
	}
	if got.ClipFraction != 0 {
		t.Fatalf("clip %v", got.ClipFraction)
	}
}

func TestMeasureHealthCountsClipping(t *testing.T) {
	samples := []int16{32767, 32767, 0, 0}
	got := MeasureHealth(samples, Rate)
	if got.ClipFraction < 0.4 {
		t.Fatalf("clip %v", got.ClipFraction)
	}
}
