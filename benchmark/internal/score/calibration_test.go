package score

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadCalibrationSet(t *testing.T) {
	root := findCalibrationRoot(t)
	set, err := LoadCalibrationSet(filepath.Join(root, "calibration", "judge.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(set.Cases) < 12 || set.Hash == "" {
		t.Fatalf("calibration set is incomplete: %+v", set)
	}
	critical := 0
	for _, item := range set.Cases {
		if item.Expected.Critical {
			critical++
		}
	}
	if critical < 3 {
		t.Fatalf("critical safety coverage = %d", critical)
	}
}

func findCalibrationRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "calibration", "judge.json")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("calibration fixture not found")
		}
		dir = parent
	}
}
