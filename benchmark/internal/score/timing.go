package score

import (
	"math"
	"sort"
	"strings"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/caller"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/world"
)

const (
	HumanBandMinMS   = 300
	HumanBandMaxMS   = 700
	MaxBargeInStopMS = 800
)

// Timing is one voice-to-voice gap.
type Timing struct {
	TurnID string `json:"turn_id"`
	V2VMS  int    `json:"v2v_ms"`
	Tool   bool   `json:"tool,omitempty"`
}

// Metrics is the per-call scorecard.
type Metrics struct {
	V2V                []Timing `json:"v2v"`
	V2VP50             int      `json:"v2v_p50_ms"`
	V2VP95             int      `json:"v2v_p95_ms"`
	V2VMax             int      `json:"v2v_max_ms"`
	NonToolP50         int      `json:"non_tool_p50_ms"`
	SpikeCount         int      `json:"spike_count"`
	InHumanBandPct     float64  `json:"in_human_band_pct"`
	FalseCutoff        int      `json:"false_cutoff"`
	BargeInStopMS      int      `json:"barge_in_stop_ms"`
	SelectivityHold    bool     `json:"selectivity_hold"`
	HoldThroughOverlap bool     `json:"hold_through_overlap"`
	FillerBeforeMS     int      `json:"filler_before_tool_ms"`
	FillerHeard        bool     `json:"filler_heard"`
	FillerNonBlocking  bool     `json:"filler_non_blocking"`
	FillerFail         []string `json:"filler_fail"`
	ScoringFail        []string `json:"scoring_fail"`
	EndStateFail       []string `json:"end_state_fail"`
	ToolOrderFail      []string `json:"tool_order_fail"`
	EntityToolFail     []string `json:"entity_tool_fail"`
	EntitySpeechFail   []string `json:"entity_speech_fail"`
	PolicyFail         []string `json:"policy_fail"`
	SayDoFail          []string `json:"say_do_fail"`
	Passed             bool     `json:"passed"`
	GateNotes          []string `json:"gate_notes"`
}

// TimingFromRecording measures V2V from caller-turn end to the next agent onset.
func TimingFromRecording(rec caller.Result) []Timing {
	rate := rec.Rate
	if rate <= 0 {
		rate = audio.Rate
	}
	var out []Timing
	for _, ev := range rec.Events {
		if ev.BargeIn || ev.Overlap {
			continue
		}
		cEnd := ev.RecEndMs
		next := firstOnsetAfter(rec.Agent, rate, cEnd, audio.DefaultSpeechThreshold)
		if next < 0 {
			continue
		}
		out = append(out, Timing{TurnID: ev.TurnID, V2VMS: next - cEnd})
	}
	return out
}

func firstOnsetAfter(samples []int16, rate, afterMs int, threshold float64) int {
	if rate <= 0 || len(samples) == 0 {
		return -1
	}
	frame := rate / 50
	if frame < 1 {
		frame = 1
	}
	start := afterMs * rate / 1000
	if start < 0 {
		start = 0
	}
	start = start / frame * frame
	inSpeech := false
	if start >= frame {
		inSpeech = audio.FrameEnergy(samples[start-frame:start]) >= threshold
	}
	for i := start; i+frame <= len(samples); i += frame {
		e := audio.FrameEnergy(samples[i : i+frame])
		if e >= threshold {
			if !inSpeech {
				return i * 1000 / rate
			}
			continue
		}
		inSpeech = false
	}
	return -1
}

// MarkToolTurns flags V2V samples whose reply gap overlaps a tool call.
func MarkToolTurns(m *Metrics, rec caller.Result, tools []world.ToolCall) {
	if rec.StartedAt.IsZero() || len(tools) == 0 {
		return
	}
	endByID := map[string]int{}
	for _, ev := range rec.Events {
		endByID[ev.TurnID] = ev.RecEndMs
	}
	for i, t := range m.V2V {
		cEnd, ok := endByID[t.TurnID]
		if !ok {
			continue
		}
		onset := cEnd + t.V2VMS
		for _, tool := range tools {
			startMs := int(tool.Started.Sub(rec.StartedAt).Milliseconds())
			endMs := int(tool.Ended.Sub(rec.StartedAt).Milliseconds())
			if startMs < onset && endMs > cEnd {
				m.V2V[i].Tool = true
				break
			}
		}
	}
}

// SummarizeTiming fills P50/P95/max. Human-band and spikes use non-tool turns.
func SummarizeTiming(m *Metrics) {
	vals := make([]int, 0, len(m.V2V))
	nonTool := make([]int, 0, len(m.V2V))
	for _, t := range m.V2V {
		if t.V2VMS < 0 {
			continue
		}
		vals = append(vals, t.V2VMS)
		if !t.Tool {
			nonTool = append(nonTool, t.V2VMS)
		}
	}
	if len(vals) == 0 {
		return
	}
	sort.Ints(vals)
	m.V2VP50 = percentile(vals, 50)
	m.V2VP95 = percentile(vals, 95)
	m.V2VMax = vals[len(vals)-1]
	bandSrc := nonTool
	if len(bandSrc) == 0 {
		bandSrc = vals
	} else {
		sort.Ints(nonTool)
		m.NonToolP50 = percentile(nonTool, 50)
	}
	inBand := 0
	p50 := m.NonToolP50
	if p50 == 0 {
		p50 = m.V2VP50
	}
	for _, v := range bandSrc {
		if v >= HumanBandMinMS && v <= HumanBandMaxMS {
			inBand++
		}
		if p50 > 0 && v > 2*p50 {
			m.SpikeCount++
		}
	}
	m.InHumanBandPct = 100 * float64(inBand) / float64(len(bandSrc))
}

func percentile(sorted []int, p int) int {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(float64(p)/100*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// BargeInStopMS is time from barge-in start until agent energy drops.
func BargeInStopMS(rec caller.Result) int {
	var barge caller.Event
	found := false
	for _, ev := range rec.Events {
		if ev.BargeIn && ev.Kind == scenario.TriggerBargeIn {
			barge = ev
			found = true
			break
		}
	}
	if !found {
		return -1
	}
	spans := audio.DetectSpeech(rec.Agent, rec.Rate, audio.DefaultSpeechThreshold, 80)
	for _, s := range spans {
		if s.StartMs < barge.RecStartMs && s.EndMs > barge.RecStartMs {
			stop := s.EndMs - barge.RecStartMs
			if stop < 0 {
				return 0
			}
			return stop
		}
	}
	return -1
}

// SelectivityHold is true when a cough/backchannel did not produce a new agent turn.
func SelectivityHold(rec caller.Result) bool {
	var overlap *caller.Event
	for i := range rec.Events {
		ev := rec.Events[i]
		if ev.BargeIn && ev.Kind != scenario.TriggerBargeIn && ev.Kind != scenario.TriggerDuringAgent {
			overlap = &rec.Events[i]
			break
		}
	}
	if overlap == nil {
		return true
	}
	agent := audio.DetectSpeech(rec.Agent, rec.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	for _, s := range agent {
		if s.StartMs >= overlap.RecStartMs && s.StartMs <= overlap.RecEndMs+200 {
			return false
		}
	}
	return true
}

// HoldThroughOverlap is true when a mid-speech cough/talker did not stop the agent.
func HoldThroughOverlap(rec caller.Result) bool {
	var overlap *caller.Event
	for i := range rec.Events {
		ev := rec.Events[i]
		if ev.Kind == scenario.TriggerDuringAgent {
			overlap = &rec.Events[i]
			break
		}
	}
	if overlap == nil {
		return true
	}
	agent := audio.DetectSpeech(rec.Agent, rec.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	for _, s := range agent {
		if s.StartMs < overlap.RecStartMs && s.EndMs > overlap.RecEndMs+150 {
			return true
		}
	}
	return false
}

// FalseCutoff counts agent starts that overlap a non-barge caller utterance.
func FalseCutoff(rec caller.Result) int {
	callerSpans := audio.DetectSpeech(rec.Caller, rec.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	agentSpans := audio.DetectSpeech(rec.Agent, rec.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	n := 0
	for _, a := range agentSpans {
		for _, c := range callerSpans {
			if a.StartMs > c.StartMs+80 && a.StartMs < c.EndMs-80 {
				n++
			}
		}
	}
	return n
}

var fillerPhrases = []string{
	"one moment", "one sec", "one second", "just a moment", "just a sec",
	"hold on", "hang on", "checking", "let me check", "look that up",
}

// ContainsFiller reports whether the transcript has a filler phrase.
func ContainsFiller(text string) bool {
	low := strings.ToLower(text)
	for _, p := range fillerPhrases {
		if strings.Contains(low, p) {
			return true
		}
	}
	return false
}

// DelayedToolNames returns tools that the scenario delays, preferring expected_tools order.
func DelayedToolNames(sc scenario.Scenario) []string {
	var names []string
	seen := map[string]bool{}
	for _, t := range sc.ExpectedTools {
		if sc.ToolDelayMS[t.Name] > 0 && !seen[t.Name] {
			names = append(names, t.Name)
			seen[t.Name] = true
		}
	}
	var extra []string
	for name, ms := range sc.ToolDelayMS {
		if ms > 0 && !seen[name] {
			extra = append(extra, name)
		}
	}
	sort.Strings(extra)
	return append(names, extra...)
}

// ScoreFiller checks filler speech during a delayed tool window.
func ScoreFiller(m *Metrics, sc scenario.Scenario, rec caller.Result, sess *world.Session, agentText string) {
	names := DelayedToolNames(sc)
	if len(names) == 0 {
		return
	}
	name := names[0]
	m.FillerHeard = ContainsFiller(agentText)
	if !m.FillerHeard {
		m.FillerFail = append(m.FillerFail, "no filler speech")
	}
	if sess == nil {
		m.FillerFail = append(m.FillerFail, "no session")
		return
	}
	var tool *world.ToolCall
	for i := range sess.Tools {
		if sess.Tools[i].Name == name {
			tool = &sess.Tools[i]
			break
		}
	}
	if tool == nil {
		m.FillerFail = append(m.FillerFail, "delayed tool "+name+" not called")
		return
	}
	if rec.StartedAt.IsZero() {
		return
	}
	startMs := int(tool.Started.Sub(rec.StartedAt).Milliseconds())
	endMs := int(tool.Ended.Sub(rec.StartedAt).Milliseconds())
	if startMs < 0 {
		startMs = 0
	}
	onset := firstOnsetAfter(rec.Agent, rec.Rate, startMs, audio.DefaultSpeechThreshold)
	if onset < 0 {
		onset = firstOnsetAfter(rec.Agent, rec.Rate, max(0, startMs-2000), audio.DefaultSpeechThreshold)
	}
	if onset >= 0 && onset < endMs {
		m.FillerNonBlocking = true
		gap := onset - startMs
		if gap < 0 {
			gap = 0
		}
		m.FillerBeforeMS = gap
	}
	if !m.FillerNonBlocking {
		m.FillerFail = append(m.FillerFail, "blocked until tool returned")
	}
}

// EntityInSpeech reports missing spoken entities.
func EntityInSpeech(agentTranscript string, entities []scenario.Entity) []string {
	var fails []string
	for _, e := range entities {
		if !e.InSpeech {
			continue
		}
		if !scenario.MatchValue(agentTranscript, e.Value) {
			fails = append(fails, e.Name+"="+e.Value)
		}
	}
	return fails
}

// ApplyGates sets Passed from hard AND-gates.
func ApplyGates(m *Metrics) {
	var notes []string
	if len(m.EndStateFail) > 0 {
		notes = append(notes, "end_state")
	}
	if len(m.PolicyFail) > 0 {
		notes = append(notes, "policy")
	}
	if len(m.EntityToolFail) > 0 {
		notes = append(notes, "entity_tools")
	}
	if len(m.EntitySpeechFail) > 0 {
		notes = append(notes, "entity_speech")
	}
	if len(m.ToolOrderFail) > 0 {
		notes = append(notes, "tool_order")
	}
	if len(m.SayDoFail) > 0 {
		notes = append(notes, "say_do")
	}
	if len(m.FillerFail) > 0 {
		notes = append(notes, "filler")
	}
	if len(m.ScoringFail) > 0 {
		notes = append(notes, m.ScoringFail...)
	}
	if m.BargeInStopMS > MaxBargeInStopMS {
		notes = append(notes, "barge_in")
	}
	if !m.SelectivityHold {
		notes = append(notes, "selectivity")
	}
	if !m.HoldThroughOverlap {
		notes = append(notes, "hold")
	}
	m.GateNotes = notes
	m.Passed = len(notes) == 0
}

// PassAtK is true if any of k trials passed.
func PassAtK(passed []bool) bool {
	for _, p := range passed {
		if p {
			return true
		}
	}
	return false
}

// PassHatK is true if every trial passed.
func PassHatK(passed []bool) bool {
	if len(passed) == 0 {
		return false
	}
	for _, p := range passed {
		if !p {
			return false
		}
	}
	return true
}

// WorldGates fills deterministic checks from the mock DB.
func WorldGates(m *Metrics, sc scenario.Scenario, sess *world.Session, agentText string) {
	if sess == nil {
		m.EndStateFail = []string{"no session"}
		return
	}
	m.EndStateFail = world.CheckAssertions(sess.State, sc.EndState)
	m.ToolOrderFail = world.CheckToolOrder(sess.Tools, sc.ToolOrder)
	m.EntityToolFail = world.EntityInTools(sess.Tools, sc.Entities)
	m.EntitySpeechFail = EntityInSpeech(agentText, sc.Entities)
}
