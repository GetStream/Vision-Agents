package report

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// ResolveBaseline maps a --baseline value to a run directory that has summary.json.
// A path that already holds a summary is used as-is. A target name looks under
// <root>/baselines/<target>/<commit>/ and picks the newest commit.
func ResolveBaseline(root, spec string) (string, error) {
	if spec == "" {
		return "", fmt.Errorf("compare: a baseline path or target name is required")
	}
	if dir, ok := summaryDir(spec); ok {
		return dir, nil
	}
	if root != "" {
		if dir, ok := summaryDir(filepath.Join(root, spec)); ok {
			return dir, nil
		}
		picked, err := newestBaseline(filepath.Join(root, "baselines", spec))
		if err != nil {
			return "", err
		}
		if picked != "" {
			return picked, nil
		}
	}
	return "", fmt.Errorf("compare: no summary.json under %s", spec)
}

// StoreBaseline copies summary.json and manifest.json from a run into
// <root>/baselines/<target>/<commit>/. A second pack for the same commit is merged
// into the stored summary so the frozen set can be stored one vertical at a time.
func StoreBaseline(root, target, commit, runDir string) error {
	if target == "" || commit == "" {
		return fmt.Errorf("store baseline: target and commit are required")
	}
	dest := filepath.Join(root, "baselines", target, commit)
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return err
	}
	if _, exists := summaryDir(dest); exists {
		existing, err := LoadSummary(dest)
		if err != nil {
			return err
		}
		incoming, err := LoadSummary(runDir)
		if err != nil {
			return err
		}
		merged := mergeSummaries(existing, incoming)
		raw, err := json.MarshalIndent(merged, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dest, "summary.json"), append(raw, '\n'), 0o644); err != nil {
			return err
		}
		return copyFile(filepath.Join(runDir, "manifest.json"), filepath.Join(dest, "manifest.json"))
	}
	for _, name := range []string{"summary.json", "manifest.json"} {
		if err := copyFile(filepath.Join(runDir, name), filepath.Join(dest, name)); err != nil {
			return err
		}
	}
	return nil
}

func mergeSummaries(base, extra Summary) Summary {
	calls := append(append([]CallResult(nil), base.Calls...), extra.Calls...)
	k := base.K
	if extra.K > k {
		k = extra.K
	}
	system := base.System
	if system == "" {
		system = extra.System
	}
	runID := extra.RunID
	if runID == "" {
		runID = base.RunID
	}
	sum := BuildSummary(system, runID, k, calls)
	sum.Manifest = extra.Manifest
	if sum.Manifest.GitCommit == "" {
		sum.Manifest = base.Manifest
	}
	return sum
}

func newestBaseline(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	type candidate struct {
		path    string
		written time.Time
	}
	var found []candidate
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		if _, ok := summaryDir(path); !ok {
			continue
		}
		info, err := os.Stat(filepath.Join(path, "summary.json"))
		if err != nil {
			continue
		}
		found = append(found, candidate{path: path, written: info.ModTime()})
	}
	if len(found) == 0 {
		return "", nil
	}
	sort.Slice(found, func(i, j int) bool {
		return found[i].written.After(found[j].written)
	})
	return found[0].path, nil
}

func summaryDir(dir string) (string, bool) {
	if _, err := os.Stat(filepath.Join(dir, "summary.json")); err == nil {
		return dir, true
	}
	return "", false
}

func copyFile(from, to string) error {
	src, err := os.Open(from)
	if err != nil {
		return err
	}
	defer src.Close()
	dst, err := os.Create(to)
	if err != nil {
		return err
	}
	defer dst.Close()
	if _, err := io.Copy(dst, src); err != nil {
		return err
	}
	return dst.Close()
}
