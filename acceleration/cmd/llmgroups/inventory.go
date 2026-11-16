package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"slices"
	"time"
)

// measurement is one model at one effort on the suite. Latency and token figures are over
// the calls that answered; pass rates count an error as a fail, since a turn that errors is
// a turn the caller did not get.
type measurement struct {
	Model  string `json:"-"`
	Effort string `json:"-"`
	// Routed is a run at the effort router.yaml builds the model with, which is the one a
	// group's ranking reads; a sweep's other efforts are for the report's comparison.
	Routed      bool   `json:"-"`
	Suite       string `json:"suite"`
	Date        string `json:"date"`
	Calls       int    `json:"calls"`
	Errors      int    `json:"errors"`
	ErrorSample string `json:"error_sample,omitempty"`
	TTFTP50Ms   int64  `json:"ttft_p50_ms"`
	TTFTP95Ms   int64  `json:"ttft_p95_ms"`
	// TTFTLive is over the turns a conversation is made of (a spoken reply, a tool call, a
	// small JSON answer), without the maths, where a thinking model spends seconds before
	// its first word that a caller's question would not cost it.
	TTFTLiveP50Ms       int64            `json:"ttft_live_p50_ms"`
	TTFTLiveP95Ms       int64            `json:"ttft_live_p95_ms"`
	TotalP50Ms          int64            `json:"total_p50_ms"`
	TotalP95Ms          int64            `json:"total_p95_ms"`
	InputTokensMean     float64          `json:"input_tokens_mean"`
	CachedTokensMean    float64          `json:"cached_tokens_mean"`
	OutputTokensMean    float64          `json:"output_tokens_mean"`
	ReasoningTokensMean float64          `json:"reasoning_tokens_mean"`
	PassRate            float64          `json:"pass_rate"`
	PassByKind          map[kind]float64 `json:"pass_by_kind"`
}

func summarise(s subject, calls []call) measurement {
	m := measurement{
		Model: s.config.Name(), Effort: s.effort, Routed: s.asked == "", Suite: suiteVersion,
		Date: time.Now().UTC().Format(time.DateOnly), Calls: len(calls), PassByKind: map[kind]float64{},
	}
	var ttft, live, total []int64
	var input, cached, output, reasoning, passed int64
	attempts, passes := map[kind]int{}, map[kind]int{}
	for _, c := range calls {
		attempts[c.kind]++
		if c.err != "" {
			m.Errors++
			if m.ErrorSample == "" {
				m.ErrorSample = firstLine(c.err)
			}
			continue
		}
		ttft = append(ttft, c.ttftMs)
		if c.kind == kindVoice || c.kind == kindTool || c.kind == kindJSON {
			live = append(live, c.ttftMs)
		}
		total = append(total, c.totalMs)
		input += c.input
		cached += c.cached
		output += c.output
		reasoning += c.reasoning
		if c.correct {
			passed++
			passes[c.kind]++
		}
	}
	for k, n := range attempts {
		m.PassByKind[k] = round(float64(passes[k])/float64(n), 3)
	}
	if len(calls) > 0 {
		m.PassRate = round(float64(passed)/float64(len(calls)), 3)
	}
	if answered := len(ttft); answered > 0 {
		m.TTFTP50Ms, m.TTFTP95Ms = percentile(ttft, 0.5), percentile(ttft, 0.95)
		m.TotalP50Ms, m.TotalP95Ms = percentile(total, 0.5), percentile(total, 0.95)
		if len(live) > 0 {
			m.TTFTLiveP50Ms, m.TTFTLiveP95Ms = percentile(live, 0.5), percentile(live, 0.95)
		}
		n := float64(answered)
		m.InputTokensMean = round(float64(input)/n, 1)
		m.CachedTokensMean = round(float64(cached)/n, 1)
		m.OutputTokensMean = round(float64(output)/n, 1)
		m.ReasoningTokensMean = round(float64(reasoning)/n, 1)
	}
	return m
}

// percentile is the nearest-rank percentile, which on a few dozen calls is as precise as
// the sample allows.
func percentile(values []int64, q float64) int64 {
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	index := int(q*float64(len(sorted))+0.5) - 1
	return sorted[max(0, min(index, len(sorted)-1))]
}

func round(v float64, places int) float64 {
	scale := 1.0
	for range places {
		scale *= 10
	}
	return float64(int64(v*scale+0.5)) / scale
}

// inventory is the committed file, kept as loose JSON so the fields the Python side writes
// (Artificial Analysis numbers, notes) survive a bench run untouched.
type inventory struct {
	raw map[string]any
}

func readInventory(path string) (*inventory, error) {
	inv := &inventory{raw: map[string]any{}}
	if path == "" {
		return inv, nil
	}
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return inv, nil
	}
	if err != nil {
		return nil, fmt.Errorf("llmgroups: read inventory: %w", err)
	}
	if err := json.Unmarshal(data, &inv.raw); err != nil {
		return nil, fmt.Errorf("llmgroups: parse inventory: %w", err)
	}
	return inv, nil
}

func (inv *inventory) entry(name string) map[string]any {
	models, _ := inv.raw["models"].(map[string]any)
	if models == nil {
		models = map[string]any{}
		inv.raw["models"] = models
	}
	entry, _ := models[name].(map[string]any)
	if entry == nil {
		entry = map[string]any{}
		models[name] = entry
	}
	return entry
}

func (inv *inventory) measured(name, effort string) bool {
	measured, _ := inv.entry(name)["measured"].(map[string]any)
	m, _ := measured[effort].(map[string]any)
	return m != nil && m["suite"] == suiteVersion
}

// record keeps a measurement unless every call failed and an earlier one answered: an
// outage on the day of a run says nothing about the model.
func (inv *inventory) record(m measurement) {
	entry := inv.entry(m.Model)
	if m.Routed {
		entry["routed_effort"] = m.Effort
	}
	measured, _ := entry["measured"].(map[string]any)
	if measured == nil {
		measured = map[string]any{}
		entry["measured"] = measured
	}
	if m.Errors == m.Calls {
		if earlier, _ := measured[m.Effort].(map[string]any); earlier != nil && earlier["errors"] != earlier["calls"] {
			fmt.Fprintf(os.Stderr, "kept the earlier %s@%s: every call failed (%s)\n", m.Model, m.Effort, m.ErrorSample)
			return
		}
	}
	data, _ := json.Marshal(m)
	var loose map[string]any
	_ = json.Unmarshal(data, &loose)
	measured[m.Effort] = loose
}

func (inv *inventory) write(path string) error {
	data, err := json.MarshalIndent(inv.raw, "", "  ")
	if err != nil {
		return fmt.Errorf("llmgroups: encode inventory: %w", err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("llmgroups: write inventory: %w", err)
	}
	return nil
}
