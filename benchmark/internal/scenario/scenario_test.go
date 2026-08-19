package scenario

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "golden.yaml")
	body := []byte(`
id: restaurant.golden
pack: restaurant
category: golden
name: patio four
turns:
  - id: intro
    text: "Table for four Saturday at 7:30."
    trigger:
      kind: after_agent_turn
      delay_ms: 400
end_state:
  - path: reservation.party_size
    eq: 4
entities:
  - name: time
    value: "7:30"
    in_speech: true
    in_tools: true
`)
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	s, err := LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if s.ID != "restaurant.golden" || len(s.Turns) != 1 {
		t.Fatalf("%+v", s)
	}
	if s.Turns[0].Trigger.DelayMS != 400 {
		t.Fatalf("delay %d", s.Turns[0].Trigger.DelayMS)
	}
}

func TestValidateDuringAgent(t *testing.T) {
	s := Scenario{
		ID: "x", Pack: "restaurant", Category: Checklist,
		Turns: []Turn{{
			ID:           "cough",
			OverlapSound: "cough",
			Trigger:      Trigger{Kind: TriggerDuringAgent, AfterMS: 400},
		}},
	}
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestValidateRejectsEmptyTurns(t *testing.T) {
	s := Scenario{ID: "x", Pack: "restaurant", Category: Golden}
	if err := s.Validate(); err == nil {
		t.Fatal("expected error")
	}
}

func TestMatchValue(t *testing.T) {
	cases := []struct {
		text, value string
		want        bool
	}{
		{"party of six tonight", "6", true},
		{"party of 6 tonight", "six", true},
		{"Saturday at 7:30 patio", "7:30", true},
		{"Saturday at 07:30 patio", "7:30", true},
		{"table at 7 30", "7:30", true},
		{"seven thirty this Saturday", "7:30", true},
		{"call (512) 555-0142 please", "512-555-0142", true},
		{"digits 5125550142", "512-555-0142", true},
		{"name is Alvarez peanut", "Alvarez", true},
		{"name is alvarez peanut", "Alvarez", true},
		{"nothing relevant here", "xyz", false},
	}
	for _, tc := range cases {
		got := MatchValue(tc.text, tc.value)
		if got != tc.want {
			t.Errorf("MatchValue(%q, %q)=%v want %v", tc.text, tc.value, got, tc.want)
		}
	}
}

func TestLoadPacks(t *testing.T) {
	root := findRepoRoot(t)
	for _, pack := range Packs() {
		scenarios, err := LoadPack(filepath.Join(root, "scenarios", pack))
		if err != nil {
			t.Fatalf("%s: %v", pack, err)
		}
		if len(scenarios) < 4 {
			t.Fatalf("%s: %d scenarios", pack, len(scenarios))
		}
	}
}

func findRepoRoot(t *testing.T) string {
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
