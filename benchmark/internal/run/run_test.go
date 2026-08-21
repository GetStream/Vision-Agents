package run

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
)

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

func TestWebRTCJoinFailsWithoutCredentials(t *testing.T) {
	t.Setenv("STREAM_API_KEY", "")
	t.Setenv("STREAM_API_SECRET", "")
	t.Setenv("STREAM_USER_TOKEN", "")
	_, err := runWebRTC(context.Background(), Config{}, scenario.Scenario{ID: "restaurant.golden"}, nil, 1)
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
