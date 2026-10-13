// Package bench ramps voice agents against a router until it stops keeping up, and
// reports what it took.
package bench

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Report is one ramp: every step it ran, and the largest one that held.
type Report struct {
	// Label names the build this ran against, which is what a comparison reads.
	Label string `json:"label"`
	// Target is the router the agents ran on.
	Target string `json:"target"`
	// Models are the three the agents were configured with, so a report against real
	// providers is never mistaken for a mocked one.
	Models    Models    `json:"models"`
	StartedAt time.Time `json:"started_at"`
	HoldS     float64   `json:"hold_s"`
	Steps     []Step    `json:"steps"`
	// Held is the largest number of agents that met every criterion, and Broke says what
	// gave way at the step after it.
	Held  int    `json:"held"`
	Broke string `json:"broke"`
}

// Models are the three a cascaded agent is built from.
type Models struct {
	STT string `json:"stt"`
	TTS string `json:"tts"`
	LLM string `json:"llm"`
}

// Step is one level of the ramp: this many agents, all talking at once.
type Step struct {
	Agents int `json:"agents"`
	// Sessions and Joined are how many conversations were asked for and how many got as
	// far as hearing the agent on the call.
	Sessions int `json:"sessions"`
	Joined   int `json:"joined"`
	// Turns is how many times a caller stopped talking, and Answered how many of those
	// the agent said something back to.
	Turns    int `json:"turns"`
	Answered int `json:"answered"`
	// SessionMs is how long creating a conversation took, TrackMs how long after that the
	// agent's audio appeared, and ReplyMs the wait a caller felt between finishing a
	// sentence and hearing the answer start.
	SessionMs Stat `json:"session_ms"`
	TrackMs   Stat `json:"track_ms"`
	ReplyMs   Stat `json:"reply_ms"`
	// GapMs is the longest silence inside a reply, which is what a starved speaker
	// sounds like.
	GapMs Stat `json:"gap_ms"`
	// DroppedFrames is inbound audio the caller could not keep up with. Anything above
	// zero means the measurement is the laptop's and not the router's.
	DroppedFrames int `json:"dropped_frames"`
	// Errors counts what failed, by the step of a conversation it failed at.
	Errors map[string]int `json:"errors,omitempty"`
	// Router is what the pods were doing while this step ran, and Host what this machine
	// was doing.
	Router Pods    `json:"router"`
	Host   HostCPU `json:"host"`
	// Broke is empty for a step that held, and otherwise says what gave way.
	Broke string `json:"broke,omitempty"`
}

// Stat is a distribution, in milliseconds.
type Stat struct {
	Count int     `json:"count"`
	P50   float64 `json:"p50"`
	P95   float64 `json:"p95"`
	P99   float64 `json:"p99"`
	Max   float64 `json:"max"`
}

// HostCPU is what this machine was doing while a step ran. A run that saturated the
// caller measured the laptop rather than the router.
type HostCPU struct {
	// Busy is the share of this machine's cores the benchmark used, from 0 to 1.
	Busy float64 `json:"busy"`
	// Cores is how many it had.
	Cores int `json:"cores"`
}

// Saturated reports whether this machine ran out of room, which makes the step's
// latencies the caller's own.
func (h HostCPU) Saturated() bool { return h.Busy > 0.85 }

// stat summarises a set of measurements.
func stat(values []float64) Stat {
	if len(values) == 0 {
		return Stat{}
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	return Stat{
		Count: len(sorted),
		P50:   percentile(sorted, 0.50),
		P95:   percentile(sorted, 0.95),
		P99:   percentile(sorted, 0.99),
		Max:   sorted[len(sorted)-1],
	}
}

// percentile is the nearest-rank value, which needs no interpolation to explain.
func percentile(sorted []float64, at float64) float64 {
	rank := int(math.Ceil(at*float64(len(sorted)))) - 1
	if rank < 0 {
		rank = 0
	}
	if rank >= len(sorted) {
		rank = len(sorted) - 1
	}
	return sorted[rank]
}

// Save writes the report as JSON and as the table people read.
func (r Report) Save(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "report.json"), raw, 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "report.md"), []byte(r.Markdown()), 0o644)
}

// Load reads a report from a run directory, or from the file itself.
func Load(path string) (Report, error) {
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		path = filepath.Join(path, "report.json")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return Report{}, err
	}
	var report Report
	if err := json.Unmarshal(raw, &report); err != nil {
		return Report{}, fmt.Errorf("bench: read %s: %w", path, err)
	}
	return report, nil
}

// Markdown is the table a reader scans: one row per step, and what broke underneath.
func (r Report) Markdown() string {
	var out strings.Builder
	fmt.Fprintf(&out, "# %s\n\n", r.title())
	fmt.Fprintf(&out, "Target %s, %s / %s / %s, %.0fs a step, %s.\n\n",
		r.Target, r.Models.STT, r.Models.TTS, r.Models.LLM, r.HoldS,
		r.StartedAt.Format(time.RFC3339))
	fmt.Fprintf(&out, "**Held %d agents.** %s\n\n", r.Held, r.verdict())

	fmt.Fprintln(&out, "| agents | joined | answered | reply p50 | reply p95 | reply p99 | gap p95 | session p95 | router cpu | router mem | host cpu |")
	fmt.Fprintln(&out, "|---|---|---|---|---|---|---|---|---|---|---|")
	for _, step := range r.Steps {
		fmt.Fprintf(&out, "| %d | %d/%d | %d/%d | %s | %s | %s | %s | %s | %s | %s | %.0f%% |\n",
			step.Agents, step.Joined, step.Sessions, step.Answered, step.Turns,
			ms(step.ReplyMs.P50), ms(step.ReplyMs.P95), ms(step.ReplyMs.P99),
			ms(step.GapMs.P95), ms(step.SessionMs.P95),
			step.Router.CPU(), step.Router.Memory(), step.Host.Busy*100)
	}

	for _, step := range r.Steps {
		if len(step.Errors) == 0 && step.Broke == "" && step.DroppedFrames == 0 {
			continue
		}
		fmt.Fprintf(&out, "\n## %d agents\n\n", step.Agents)
		if step.Broke != "" {
			fmt.Fprintf(&out, "- broke: %s\n", step.Broke)
		}
		if step.DroppedFrames > 0 {
			fmt.Fprintf(&out, "- %d inbound frames dropped: the callers could not keep up, so these latencies are this machine's\n",
				step.DroppedFrames)
		}
		for _, where := range sortedKeys(step.Errors) {
			fmt.Fprintf(&out, "- %s: %d\n", where, step.Errors[where])
		}
	}
	return out.String()
}

func (r Report) title() string {
	if r.Label == "" {
		return "Voice agent load"
	}
	return "Voice agent load: " + r.Label
}

func (r Report) verdict() string {
	if r.Broke == "" {
		return "Nothing broke inside the range it was given, so the ceiling is higher than this run looked."
	}
	return "At the step above, " + r.Broke + "."
}

// Compare is the before and after of one change, step by step.
func Compare(before, after Report) string {
	var out strings.Builder
	fmt.Fprintf(&out, "# %s vs %s\n\n", label(before, "before"), label(after, "after"))
	fmt.Fprintf(&out, "Held %d then %d agents.\n\n", before.Held, after.Held)

	fmt.Fprintln(&out, "| agents | reply p50 | reply p95 | session p95 | answered | router cpu |")
	fmt.Fprintln(&out, "|---|---|---|---|---|---|")
	indexed := map[int]Step{}
	for _, step := range before.Steps {
		indexed[step.Agents] = step
	}
	for _, step := range after.Steps {
		was, ran := indexed[step.Agents]
		if !ran {
			fmt.Fprintf(&out, "| %d | %s | %s | %s | %s | %s |\n", step.Agents,
				ms(step.ReplyMs.P50), ms(step.ReplyMs.P95), ms(step.SessionMs.P95),
				answered(step), step.Router.CPU())
			continue
		}
		fmt.Fprintf(&out, "| %d | %s | %s | %s | %s | %s |\n", step.Agents,
			change(was.ReplyMs.P50, step.ReplyMs.P50),
			change(was.ReplyMs.P95, step.ReplyMs.P95),
			change(was.SessionMs.P95, step.SessionMs.P95),
			answered(was)+" to "+answered(step),
			was.Router.CPU()+" to "+step.Router.CPU())
	}
	return out.String()
}

func label(report Report, fallback string) string {
	if report.Label == "" {
		return fallback
	}
	return report.Label
}

// change reads as the two numbers and what moved between them, since a percentage alone
// hides whether either was worth caring about.
func change(was, now float64) string {
	if was == 0 || now == 0 {
		return ms(was) + " to " + ms(now)
	}
	return fmt.Sprintf("%s to %s (%+.0f%%)", ms(was), ms(now), (now-was)/was*100)
}

func answered(step Step) string {
	if step.Turns == 0 {
		return "-"
	}
	return fmt.Sprintf("%.0f%%", float64(step.Answered)/float64(step.Turns)*100)
}

func ms(value float64) string {
	if value == 0 {
		return "-"
	}
	return fmt.Sprintf("%.0fms", value)
}

func sortedKeys(counts map[string]int) []string {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
