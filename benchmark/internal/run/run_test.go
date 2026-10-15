package run

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/report"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
	"github.com/GetStream/Vision-Agents/benchmark/internal/target"
)

func TestAnUnseededRunKeepsScenarioOrder(t *testing.T) {
	sum, ids := runOrder(t, Config{})
	want := []string{"a-t1", "a-t2", "b-t1", "b-t2", "c-t1", "c-t2", "d-t1", "d-t2", "e-t1", "e-t2"}
	if !slices.Equal(ids, want) {
		t.Fatalf("order = %v, want %v", ids, want)
	}
	if sum.Manifest.Seed != nil {
		t.Fatal("unseeded runs must not acquire a seed")
	}
}

func TestASeedReplaysTheSameScenarioOrder(t *testing.T) {
	seed := uint64(0)
	first, ids := runOrder(t, Config{Seed: &seed})
	second, replay := runOrder(t, Config{Seed: &seed})
	if !slices.Equal(ids, replay) || first.Manifest.ScenarioHash != second.Manifest.ScenarioHash {
		t.Fatalf("same seed changed the order: %v vs %v", ids, replay)
	}
	raw, err := json.Marshal(first.Manifest)
	if err != nil || !strings.Contains(string(raw), `"seed":0`) {
		t.Fatalf("zero seed missing from manifest: %s, %v", raw, err)
	}
}

func TestChangingTheSeedReordersWithoutDroppingTrials(t *testing.T) {
	zero, one := uint64(0), uint64(1)
	_, first := runOrder(t, Config{Seed: &zero})
	_, second := runOrder(t, Config{Seed: &one})
	_, original := runOrder(t, Config{})
	if slices.Equal(first, second) || slices.Equal(first, original) {
		t.Fatalf("seed did not change the order: %v vs %v", first, second)
	}
	slices.Sort(first)
	slices.Sort(second)
	if !slices.Equal(first, original) || !slices.Equal(second, original) {
		t.Fatal("shuffling lost or duplicated trials")
	}
}

func TestSeededShortRunsBalanceTypesAndRotateVariants(t *testing.T) {
	seen := map[string]bool{}
	for seed := uint64(0); seed < 10; seed++ {
		sum, ids := runOrder(t, Config{Seed: &seed, Short: true})
		_, replay := runOrder(t, Config{Seed: &seed, Short: true})
		if !slices.Equal(ids, replay) {
			t.Fatalf("seed %d changed selection: %v vs %v", seed, ids, replay)
		}
		counts := map[string]int{}
		for _, call := range sum.Calls {
			counts[call.Category]++
			seen[call.ScenarioID] = true
		}
		if len(sum.Calls) != 6 || counts["golden"] != 2 || counts["interrupt"] != 2 || counts["checklist"] != 2 {
			t.Fatalf("seed %d changed the type weights: %v", seed, counts)
		}
	}
	if len(seen) != 5 {
		t.Fatalf("different seeds did not cover the full pool: %v", seen)
	}
}

func TestUnseededShortRunsKeepTheFixedList(t *testing.T) {
	_, ids := runOrder(t, Config{Short: true})
	want := []string{"a-t1", "a-t2", "b-t1", "b-t2", "c-t1", "c-t2"}
	if !slices.Equal(ids, want) {
		t.Fatalf("unseeded short selection = %v, want %v", ids, want)
	}
}

func runOrder(t *testing.T, cfg Config) (report.Summary, []string) {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "scenarios", "restaurant")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for i, category := range []string{"golden", "interrupt", "checklist", "checklist", "checklist"} {
		id := string(rune('a' + i))
		data := fmt.Sprintf("id: %s\npack: restaurant\ncategory: %s\nturns:\n  - id: noise\n    overlap_sound: cough\n", id, category)
		if err := os.WriteFile(filepath.Join(dir, id+".yaml"), []byte(data), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "scenarios", "short.txt"), []byte("a\nb\nc\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Exercise scheduling and persisted reports without provider calls or synthesized speech.
	cfg.Root, cfg.Pack, cfg.K = root, "restaurant", 2
	cfg.WorldAddr, cfg.Transport = "127.0.0.1:0", "unavailable"
	sum, err := Run(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, call := range sum.Calls {
		ids = append(ids, fmt.Sprintf("%s-t%d", call.ScenarioID, call.Trial))
	}
	return sum, ids
}

func TestLoadRestaurantPack(t *testing.T) {
	root := findTestRoot(t)
	scenarios, err := scenario.LoadPack(filepath.Join(root, "scenarios", "restaurant"))
	if err != nil {
		t.Fatal(err)
	}
	if len(scenarios) < 4 {
		t.Fatalf("got %d scenarios", len(scenarios))
	}
}

func TestWebRTCCallID(t *testing.T) {
	sc := scenario.Scenario{ID: "restaurant.golden"}
	if got := webrtcCallID(Config{CallID: "fixed", K: 1}, sc, 1); got != "fixed" {
		t.Fatalf("got %s", got)
	}
	if got := webrtcCallID(Config{CallID: "fixed", K: 3}, sc, 2); got != "fixed-t2" {
		t.Fatalf("got %s", got)
	}
	got := webrtcCallID(Config{}, sc, 1)
	if !strings.HasPrefix(got, "vb-restaurant-golden-1-") {
		t.Fatalf("got %s", got)
	}
}

func TestWebRTCCallType(t *testing.T) {
	for _, name := range []string{target.AccelerationName, target.AcceleratedName} {
		if got := webrtcCallType(Config{TargetName: name, CallType: "default"}); got != "agent" {
			t.Fatalf("%s: got %s, want agent: the router joins agent:<session id>", name, got)
		}
	}
	if got := webrtcCallType(Config{TargetName: "python"}); got != "default" {
		t.Fatalf("got %s", got)
	}
	if got := webrtcCallType(Config{TargetName: "python", CallType: "audio_room"}); got != "audio_room" {
		t.Fatalf("got %s", got)
	}
}

func TestBuildManifestRecordsComparableInputs(t *testing.T) {
	manifest := buildManifest(Config{
		Root:           findTestRoot(t),
		Pack:           "restaurant",
		Transport:      "stream",
		TargetName:     "python",
		TargetModel:    "custom-model",
		TargetVoice:    "custom-voice",
		NetworkProfile: "us-west-wired",
	}, []scenario.Scenario{{ID: "restaurant.golden", Pack: "restaurant", Category: scenario.Golden}})
	if manifest.ScenarioHash == "" || manifest.ContractHash == "" {
		t.Fatalf("missing hashes: %+v", manifest)
	}
	if manifest.TargetModel != "custom-model" || manifest.TargetVoice != "custom-voice" {
		t.Fatalf("target configuration missing: %+v", manifest)
	}
	if manifest.CallerModel == "" || manifest.CallerVoice == "" {
		t.Fatalf("caller configuration missing: %+v", manifest)
	}
	if manifest.NetworkProfile != "us-west-wired" {
		t.Fatalf("network profile missing: %+v", manifest)
	}
	if manifest.JudgeCalibrationHash == "" {
		t.Fatalf("judge calibration fingerprint missing: %+v", manifest)
	}
	if len(manifest.Command) == 0 || manifest.Command[0] != "voicebench" {
		t.Fatalf("command is not reproducible: %+v", manifest.Command)
	}
}

func TestBuildManifestAcceleratedDefaults(t *testing.T) {
	t.Setenv("VOICEBENCH_MODEL", "")
	t.Setenv("VOICEBENCH_STT", "")
	t.Setenv("VOICEBENCH_TTS", "")
	t.Setenv("VOICEBENCH_SUBAGENT", "")
	manifest := buildManifest(Config{
		Root:        findTestRoot(t),
		Pack:        "restaurant",
		TargetName:  "accelerated",
		SpawnTarget: true,
	}, []scenario.Scenario{{ID: "restaurant.golden", Pack: "restaurant", Category: scenario.Golden}})
	if manifest.TargetSTT != "deepgram/flux-general-en" {
		t.Fatalf("stt %q", manifest.TargetSTT)
	}
	if manifest.TargetTTS != "elevenlabs/eleven_v4_turbo" {
		t.Fatalf("tts %q", manifest.TargetTTS)
	}
	if manifest.TargetModel != "gemma/gemma-4-26B-A4B-it" || manifest.TargetLLM != "gemma/gemma-4-26B-A4B-it" {
		t.Fatalf("model %q llm %q", manifest.TargetModel, manifest.TargetLLM)
	}
	if manifest.TargetSubagent != "openai/gpt-6.1-sol" {
		t.Fatalf("subagent %q", manifest.TargetSubagent)
	}
}

func TestBuildManifestSTSIgnoresTheCascade(t *testing.T) {
	t.Setenv("VOICEBENCH_STS", "")
	t.Setenv("VOICEBENCH_MODEL", "gemma/gemma-4-26B-A4B-it")
	t.Setenv("VOICEBENCH_STT", "deepgram/flux-general-en")
	t.Setenv("VOICEBENCH_TTS", "elevenlabs/eleven_v4_turbo")
	t.Setenv("VOICEBENCH_SUBAGENT", "openai/gpt-6.1-sol")
	manifest := buildManifest(Config{
		Root:        findTestRoot(t),
		Pack:        "restaurant",
		TargetName:  "sts",
		SpawnTarget: true,
	}, []scenario.Scenario{{ID: "restaurant.golden", Pack: "restaurant", Category: scenario.Golden}})
	if manifest.TargetModel != "openai/gpt-live-1" || manifest.TargetLLM != "openai/gpt-live-1" {
		t.Fatalf("model %q llm %q", manifest.TargetModel, manifest.TargetLLM)
	}
	if manifest.TargetSTT != "" || manifest.TargetTTS != "" || manifest.TargetSubagent != "" {
		t.Fatalf("stt %q tts %q subagent %q", manifest.TargetSTT, manifest.TargetTTS, manifest.TargetSubagent)
	}
}

func TestBuildManifestRecordsTheScoringASRAndNormalizer(t *testing.T) {
	cfg := Config{
		Root:       findTestRoot(t),
		Pack:       "restaurant",
		TargetName: "accelerated",
	}
	scenarios := []scenario.Scenario{{ID: "restaurant.golden", Pack: "restaurant", Category: scenario.Golden}}

	scored := buildManifest(cfg, scenarios)
	if scored.ScoringASR != score.ScoringASR || scored.NormalizerVersion != score.NormalizerVersion {
		t.Fatalf("asr %q normalizer %q", scored.ScoringASR, scored.NormalizerVersion)
	}

	cfg.SkipSTT = true
	skipped := buildManifest(cfg, scenarios)
	if skipped.ScoringASR != "" || skipped.NormalizerVersion != "" {
		t.Fatalf("no transcript was scored, so both fields must be absent: asr %q normalizer %q",
			skipped.ScoringASR, skipped.NormalizerVersion)
	}
}

func TestBuildManifestLiveKitDefaultsToRealtime(t *testing.T) {
	t.Setenv("VOICEBENCH_LIVEKIT_PIPELINE", "")
	t.Setenv("VOICEBENCH_LIVEKIT_MODEL", "")
	t.Setenv("VOICEBENCH_LIVEKIT_STT", "")
	t.Setenv("VOICEBENCH_LIVEKIT_TTS", "")
	t.Setenv("VOICEBENCH_LIVEKIT_VOICE", "")
	t.Setenv("VOICEBENCH_MODEL", "")
	manifest := buildManifest(Config{
		Root:        findTestRoot(t),
		Pack:        "healthcare",
		TargetName:  "livekit",
		SpawnTarget: true,
	}, []scenario.Scenario{{ID: "healthcare.golden", Pack: "healthcare", Category: scenario.Golden}})
	if manifest.TargetModel != "gpt-realtime-2" || manifest.TargetVoice != "marin" {
		t.Fatalf("model %q voice %q", manifest.TargetModel, manifest.TargetVoice)
	}
	if manifest.TargetSTT != "" || manifest.TargetTTS != "" {
		t.Fatalf("realtime carries no separate stt/tts: stt %q tts %q", manifest.TargetSTT, manifest.TargetTTS)
	}
}

func TestBuildManifestLiveKitInferenceDefaults(t *testing.T) {
	t.Setenv("VOICEBENCH_LIVEKIT_PIPELINE", "inference")
	t.Setenv("VOICEBENCH_LIVEKIT_MODEL", "")
	t.Setenv("VOICEBENCH_LIVEKIT_STT", "")
	t.Setenv("VOICEBENCH_LIVEKIT_TTS", "")
	t.Setenv("VOICEBENCH_LIVEKIT_VOICE", "")
	t.Setenv("VOICEBENCH_MODEL", "gemini/gemini-3.5-flash-lite")
	manifest := buildManifest(Config{
		Root:        findTestRoot(t),
		Pack:        "healthcare",
		TargetName:  "livekit",
		SpawnTarget: true,
	}, []scenario.Scenario{{ID: "healthcare.golden", Pack: "healthcare", Category: scenario.Golden}})
	if manifest.TargetSTT != "google/gemini-3.5-transcribe-live" {
		t.Fatalf("stt %q", manifest.TargetSTT)
	}
	if manifest.TargetTTS != "inworld/inworld-tts-2-flash" {
		t.Fatalf("tts %q", manifest.TargetTTS)
	}
	if manifest.TargetModel != "google/gemini-3.5-flash-lite" || manifest.TargetLLM != "google/gemini-3.5-flash-lite" {
		t.Fatalf("model %q llm %q", manifest.TargetModel, manifest.TargetLLM)
	}
	if manifest.TargetVoice != "Ashley" {
		t.Fatalf("voice %q", manifest.TargetVoice)
	}
}

func TestWebRTCJoinFailsWithoutCredentials(t *testing.T) {
	t.Setenv("STREAM_API_KEY", "")
	t.Setenv("STREAM_API_SECRET", "")
	t.Setenv("STREAM_USER_TOKEN", "")
	_, err := runWebRTC(context.Background(), Config{}, scenario.Scenario{ID: "restaurant.golden"}, nil, 1, "", nil)
	if err == nil {
		t.Fatal("expected error")
	}
}

func findTestRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	dir := wd
	for {
		if _, err := os.Stat(filepath.Join(dir, "scenarios", "restaurant")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("scenarios not found")
		}
		dir = parent
	}
}

func TestProgressSaysWhichCallAndHowItWent(t *testing.T) {
	var b strings.Builder
	progress(&b, 3, 8, "restaurant.selectivity", 1, 1, "started")
	failed := report.CallResult{Outcome: report.OutcomeFail}
	failed.Metrics.V2VP50 = 2280
	failed.Metrics.ToolCount = 1
	failed.Metrics.ExpectedToolFail = []string{"create_reservation not called"}
	progress(&b, 3, 8, "restaurant.selectivity", 1, 1, callProgress(failed))
	progress(&b, 4, 8, "restaurant.golden", 2, 3, callProgress(report.CallResult{Outcome: report.OutcomePass}))
	progress(nil, 1, 1, "ignored", 1, 1, "started")

	want := "voicebench: [3/8] restaurant.selectivity: started\n" +
		"voicebench: [3/8] restaurant.selectivity: fail · reply P50 2.28 s · 1 tool — create_reservation not called\n" +
		"voicebench: [4/8] restaurant.golden #2: pass · 0 tools\n"
	if b.String() != want {
		t.Fatalf("progress:\n%s\nwant:\n%s", b.String(), want)
	}
}

func TestALongTurnPlaysItsSentencesWithThePausesBetween(t *testing.T) {
	turn := scenario.Turn{ID: "request", Segments: []scenario.Segment{
		{Text: "first", PauseAfterMS: 500},
		{Text: "second", PauseAfterMS: 250},
		{Text: "last", PauseAfterMS: 1000},
	}}
	clips := map[string][]int16{"first": {1, 1}, "second": {2}, "last": {3, 3, 3}}
	pcm := longTurnAudio(turn, clips)
	// 2 + 500 ms + 1 + 250 ms + 3, and nothing after the last sentence.
	if want := 2 + 8000 + 1 + 4000 + 3; len(pcm) != want {
		t.Fatalf("samples = %d, want %d", len(pcm), want)
	}
	if pcm[0] != 1 || pcm[2] != 0 || pcm[8002] != 2 || pcm[len(pcm)-1] != 3 {
		t.Fatal("the sentences should play in order with silence between them")
	}
}
