package score

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
)

// TestGoldRepliesSatisfyEntitySpeech keeps scenarios honest: the gold reference reply must
// itself clear the entity_speech gate, so a scenario cannot demand a read-back that its own
// reference answer never performs.
func TestGoldRepliesSatisfyEntitySpeech(t *testing.T) {
	root := repoRoot(t)
	for _, pack := range scenario.Packs() {
		scenarios, err := scenario.LoadPack(filepath.Join(root, "scenarios", pack))
		if err != nil {
			t.Fatalf("%s: %v", pack, err)
		}
		for _, sc := range scenarios {
			if len(sc.AgentReplies) == 0 {
				t.Errorf("%s: no agent_replies to score", sc.ID)
				continue
			}
			gold := strings.Join(sc.AgentReplies, " ")
			if fails := EntityInSpeech(gold, sc.Entities); len(fails) > 0 {
				t.Errorf("%s: gold reply misses %v", sc.ID, fails)
			}
		}
	}
}

func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
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
