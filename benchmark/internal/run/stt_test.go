package run

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/report"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/synth"
)

func TestScoreClipTimesFromTheSpeechNotTheFile(t *testing.T) {
	row := STTClipRow{ID: "c1", Reference: "table for four at seven thirty"}
	voice := speech{StartMs: 400, EndMs: 2400}
	clip := scoreClip(row, "deepgram/nova-3", voice, []heard{
		{AtMs: 300, Text: "", Provider: "deepgram", Model: "nova-3"},
		{AtMs: 900, Text: "table"},
		{AtMs: 1800, Text: "table for four"},
		{AtMs: 2600, Text: "table for four at", Final: true},
		{AtMs: 2900, Text: "seven thirty", Final: true},
	})
	if clip.ToFirstWordsMs != 500 {
		t.Errorf("first words = %d ms, want 500: the empty interim is not words, and the clock starts at the speech", clip.ToFirstWordsMs)
	}
	if clip.ToSettleMs != 500 {
		t.Errorf("TTFS = %d ms, want 500: from the end of the speech to the last settled transcript", clip.ToSettleMs)
	}
	if clip.WhileSpeaking != 2 {
		t.Errorf("while speaking = %d, want 2", clip.WhileSpeaking)
	}
	if !clip.Returned || clip.Hypothesis != "table for four at seven thirty" || clip.Normalized.WER != 0 {
		t.Errorf("the settled transcripts make the hypothesis: %+v", clip)
	}
	if clip.Provider != "deepgram" || clip.Model != "nova-3" {
		t.Errorf("answering model = %s/%s", clip.Provider, clip.Model)
	}
}

func TestScoreClipWithoutASettledTranscriptReturnsNothing(t *testing.T) {
	clip := scoreClip(STTClipRow{ID: "c1", Reference: "hello there"}, "x/y", speech{EndMs: 1000}, []heard{
		{AtMs: 500, Text: "hello"},
		{AtMs: 1200, Error: "provider hung up"},
	})
	if clip.Returned || clip.Error != "provider hung up" || clip.Normalized.Deletions != 2 {
		t.Fatalf("an interim alone is not a transcript, and the error is kept: %+v", clip)
	}
}

func TestParseClipRowsNeedsAudioToStream(t *testing.T) {
	given := []byte(`{"id":"a","reference":"hi","hypothesis":"hi"}`)
	if _, err := parseClipRows(given, "/clips", true); err == nil || !strings.Contains(err.Error(), "needs audio") {
		t.Fatalf("a target with no audio to stream must be refused, got %v", err)
	}
	rows, err := parseClipRows([]byte(`{"id":"a","reference":"hi","audio":"a.wav"}`+"\n"), "/clips", true)
	if err != nil || rows[0].Audio != "/clips/a.wav" {
		t.Fatalf("audio resolves against the manifest: %+v, %v", rows, err)
	}
	if _, err := parseClipRows([]byte("{\"id\":\"a\",\"reference\":\"x\",\"hypothesis\":\"x\"}\n{\"id\":\"a\",\"reference\":\"y\",\"hypothesis\":\"y\"}"), "/", false); err == nil {
		t.Fatal("a duplicate id must be refused")
	}
}

func TestSTTScoresGivenHypothesesIntoASummary(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "clips.jsonl")
	rows := `{"id":"a","reference":"It's fifty dollars for four","hypothesis":"it is fifty dollars for four"}
{"id":"b","reference":"see you saturday","hypothesis":"see you sunday"}
`
	if err := os.WriteFile(manifest, []byte(rows), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "out")
	sum, err := STT(context.Background(), STTConfig{Root: dir, Manifest: manifest, Out: out})
	if err != nil {
		t.Fatal(err)
	}
	if sum.Kind != report.KindSTT || sum.Manifest.DatasetHash == "" || len(sum.STT) != 1 {
		t.Fatalf("summary = %+v", sum)
	}
	if got := sum.STT[0]; got.Clips != 2 || got.Substitutions != 1 || got.TimedClips != 0 {
		t.Fatalf("given hypotheses are scored but not timed: %+v", got)
	}
	raw, err := os.ReadFile(filepath.Join(out, "summary.json"))
	if err != nil {
		t.Fatal(err)
	}
	var written report.Summary
	if err := json.Unmarshal(raw, &written); err != nil || written.Kind != "stt" {
		t.Fatalf("summary.json kind = %q, %v", written.Kind, err)
	}
	clips, err := os.ReadFile(filepath.Join(out, "clips.jsonl"))
	if err != nil || strings.Count(string(clips), "\n") != 2 {
		t.Fatalf("clips.jsonl should hold one line per clip: %q, %v", clips, err)
	}
}

func TestScenarioClipsAreTheCallersLinesWithTheirScripts(t *testing.T) {
	root := t.TempDir()
	for _, pack := range scenario.Packs() {
		if err := os.MkdirAll(filepath.Join(root, "scenarios", pack), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	yaml := `id: restaurant.short_call
pack: restaurant
category: checklist
turns:
  - {id: intro, text: "Table for four at 7:30, name Alvarez."}
  - {id: again, text: "Table for four at 7:30, name Alvarez."}
  - {id: cough, overlap_sound: cough}
  - {id: go_ahead, text: "Yes, please go ahead."}
`
	if err := os.WriteFile(filepath.Join(root, "scenarios", "restaurant", "short_call.yaml"), []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}
	// Every pack has scenarios; these say a line already in the set.
	for _, pack := range []string{"healthcare", "telecom"} {
		other := "id: " + pack + ".yes\npack: " + pack + "\ncategory: checklist\nturns:\n  - {id: go_ahead, text: \"Yes, please go ahead.\"}\n"
		if err := os.WriteFile(filepath.Join(root, "scenarios", pack, "yes.yaml"), []byte(other), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// The caller audio is already cached, as it is after any run, so nothing is synthesized.
	for _, text := range []string{"Table for four at 7:30, name Alvarez.", "Yes, please go ahead."} {
		path := synth.CachePath(root, "", text)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := audio.WriteWAV(path, audio.PCM{Rate: audio.Rate, Samples: audio.Tone(audio.Rate/2, 200, 8000)}); err != nil {
			t.Fatal(err)
		}
	}
	rows, err := ScenarioClips(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0].Reference != "Table for four at 7:30, name Alvarez." || rows[0].ID != "restaurant.short_call#1" {
		t.Fatalf("one clip per distinct line, its script as reference: %+v", rows)
	}
	if _, err := os.Stat(rows[1].Audio); err != nil {
		t.Fatalf("the clip is the cached caller audio: %v", err)
	}
	if _, err := ScenarioClips(root, []string{"restaurant.other"}); err == nil {
		t.Fatal("a set with none of these scenarios has no clips")
	}
}
