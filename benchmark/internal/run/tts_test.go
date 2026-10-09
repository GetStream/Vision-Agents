package run

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/report"
)

func TestLoadCorpusDefaultsToTheScenariosAgentReplies(t *testing.T) {
	lines, err := loadCorpus(filepath.Join("..", ".."), "")
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	var restaurant bool
	for _, line := range lines {
		if seen[line.Text] {
			t.Fatalf("%q is spoken twice", line.Text)
		}
		seen[line.Text] = true
		restaurant = restaurant || strings.HasPrefix(line.ID, "restaurant.")
	}
	if len(lines) < 10 || !restaurant {
		t.Fatalf("expected every pack's agent lines, got %d", len(lines))
	}
}

func TestLoadCorpusReadsJSONL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.jsonl")
	if err := os.WriteFile(path, []byte("{\"id\":\"a\",\"text\":\"Ticket T-1842 is open.\"}\n\n{\"text\":\"Ticket T-1842 is open.\"}\n{\"text\":\"Your table is at 7:30.\"}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	lines, err := loadCorpus("", path)
	if err != nil || len(lines) != 2 || lines[0].ID != "a" || lines[1].ID != "4" {
		t.Fatalf("lines = %+v, %v", lines, err)
	}
}

// tone is a clip of steady speech-level signal, which the health check hears as voice.
func tone(rate, ms int) []int16 {
	samples := make([]int16, rate*ms/1000)
	for i := range samples {
		if i%2 == 0 {
			samples[i] = 8000
		} else {
			samples[i] = -8000
		}
	}
	return samples
}

func TestScoreSpokenTimesTheUtteranceAndRoundTripsIt(t *testing.T) {
	line := TTSLine{ID: "a", Text: "Your table is ready."}
	out := spoken{Samples: tone(24_000, 2000), Rate: 24_000, TTFBMs: 180, SynthesisMs: 500}
	heardAt := 0
	clip := scoreSpoken(line, "inworld/tts-2-flash", out, func(pcm audio.PCM) (string, error) {
		heardAt = pcm.Rate
		return "your table is ready", nil
	})
	if !clip.Returned || clip.TTFBMs != 180 || clip.AudioMs != 2000 || clip.RTF != 0.25 {
		t.Fatalf("latency = %+v", clip)
	}
	if heardAt != 24_000 || clip.Normalized.WER != 0 || clip.Grade != report.HealthGood {
		t.Fatalf("round trip and health = %+v", clip)
	}
}

func TestScoreSpokenWithoutAudioFailsWithoutTranscribing(t *testing.T) {
	clip := scoreSpoken(TTSLine{ID: "a", Text: "hello"}, "x/y", spoken{Error: "provider refused"}, func(audio.PCM) (string, error) {
		t.Fatal("nothing came back, so there is nothing to transcribe")
		return "", nil
	})
	if clip.Returned || clip.Error != "provider refused" || clip.Grade != report.HealthFail {
		t.Fatalf("clip = %+v", clip)
	}
}
