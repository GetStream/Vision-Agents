package score

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/caller"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/world"
)

const (
	HumanBandMinMS   = 300
	HumanBandMaxMS   = 700
	MaxBargeInStopMS = 800
	// UtteranceMergeGapMS joins speech spans separated by less than this, so a reply
	// that pauses between sentences is still one utterance for selectivity and hold.
	UtteranceMergeGapMS = 700
	// FillerLeadInMS is how long before a delayed tool started a filler still counts.
	// The reply that asks for a tool is spoken before the request is handed over, so
	// the words meant to fill the pause are already in flight when the window opens.
	// The non-blocking half of the filler gate has always allowed for that; the phrase
	// half allows for the same, so one late tool timestamp cannot fail only one of them.
	FillerLeadInMS = 2000
	// BargeInMergeGapMS is the longest pause the barge-in stop edge reads through. A
	// silence longer than this is already audible as the agent having stopped.
	BargeInMergeGapMS = 300
	// MaxToolSilenceMS is how long a delayed tool may leave the caller hearing nothing.
	// It matches the agent's own workingGap, which is the point the implementation
	// promises to speak up rather than keep waiting in silence.
	MaxToolSilenceMS = 800
	// bargeAlignmentMS is how far after the barge an utterance may still start and count
	// as the one that was interrupted, absorbing frame quantisation.
	bargeAlignmentMS = 120
)

// Why a barge-in has no stop measurement. Collapsing these into one unmeasured value
// made an agent that answered late look the same as one with nothing to interrupt.
const (
	BargeMeasured     = ""
	BargeNoEvent      = "no_barge_event"
	BargeStartedAfter = "started_after"
	BargeAlreadyQuiet = "already_quiet"
)

// Timing is one voice-to-voice gap.
type Timing struct {
	TurnID string `json:"turn_id"`
	V2VMS  int    `json:"v2v_ms"`
	Tool   bool   `json:"tool,omitempty"`
}

// DroppedTurn is a scripted turn that produced no V2V sample, and why. Without it a call that
// measured three turns and a call that measured one look identical in the report.
type DroppedTurn struct {
	TurnID string `json:"turn_id"`
	Reason string `json:"reason"`
}

// OverlapCheck records how the agent handled one non-directed overlap sound.
type OverlapCheck struct {
	TurnID      string `json:"turn_id"`
	Sound       string `json:"sound"`
	Continued   bool   `json:"continued"`
	StartedTurn bool   `json:"started_turn"`
}

// Why a turn produced no V2V sample.
const (
	DropBargeIn = "barge_in"
	DropOverlap = "overlap"
	DropNoOnset = "no_onset"
)

// Metrics is the per-call scorecard.
type Metrics struct {
	V2V                 []Timing       `json:"v2v"`
	Dropped             []DroppedTurn  `json:"dropped_turns,omitempty"`
	CallDurationMS      int            `json:"call_duration_ms"`
	ToolCount           int            `json:"tool_count"`
	ToolErrorCount      int            `json:"tool_error_count"`
	WorldContact        bool           `json:"world_contact"`
	ToolWaitMS          int            `json:"tool_wait_ms"`
	MaxToolWaitMS       int            `json:"max_tool_wait_ms"`
	V2VP50              int            `json:"v2v_p50_ms"`
	V2VP95              int            `json:"v2v_p95_ms"`
	V2VMax              int            `json:"v2v_max_ms"`
	NonToolP50          int            `json:"non_tool_p50_ms"`
	SpikeCount          int            `json:"spike_count"`
	InHumanBandPct      float64        `json:"in_human_band_pct"`
	FalseCutoff         int            `json:"false_cutoff"`
	ClockDriftMS        int            `json:"clock_drift_ms"`
	InboundDropped      int            `json:"inbound_dropped"`
	AgentJitterMaxMS    int            `json:"agent_jitter_max_ms"`
	RequestedSNRDB      float64        `json:"requested_snr_db,omitempty"`
	MeasuredSNRDB       float64        `json:"measured_snr_db,omitempty"`
	BargeInStopMS       int            `json:"barge_in_stop_ms"`
	BargeInReason       string         `json:"barge_in_reason,omitempty"`
	OverlapChecks       []OverlapCheck `json:"overlap_checks,omitempty"`
	SelectivityHold     bool           `json:"selectivity_hold"`
	HoldThroughOverlap  bool           `json:"hold_through_overlap"`
	FillerBeforeMS      int            `json:"filler_before_tool_ms"`
	FillerHeard         bool           `json:"filler_heard"`
	FillerSilenceMS     int            `json:"filler_silence_ms"`
	FillerNonBlocking   bool           `json:"filler_non_blocking"`
	FillerFail          []string       `json:"filler_fail"`
	EndStateFail        []string       `json:"end_state_fail"`
	ExpectedToolFail    []string       `json:"expected_tool_fail"`
	ToolOrderFail       []string       `json:"tool_order_fail"`
	EntityToolFail      []string       `json:"entity_tool_fail"`
	EntitySpeechFail    []string       `json:"entity_speech_fail"`
	PolicyFail          []string       `json:"policy_fail"`
	SayDoFail           []string       `json:"say_do_fail"`
	Passed              bool           `json:"passed"`
	GateNotes           []string       `json:"gate_notes"`
	CallerTurns         int            `json:"caller_turns"`
	AgentTurns          int            `json:"agent_turns"`
	HeardUtterances     int            `json:"heard_utterances"`
	HeardIgnored        int            `json:"heard_ignored"`
	AgentWords          int            `json:"agent_words"`
	CallerWER           float64        `json:"caller_wer,omitempty"`
	CallerWERNormalized float64        `json:"caller_wer_normalized,omitempty"`
	ExtraTools          []string       `json:"extra_tools,omitempty"`
}

// TimingFromRecording measures V2V from caller-turn end to the next agent onset, and reports
// the turns it could not measure. A turn played over a still-speaking agent has no meaningful
// reply gap; one with no later onset never drew a reply at all.
func TimingFromRecording(rec caller.Result) ([]Timing, []DroppedTurn) {
	rate := rec.Rate
	if rate <= 0 {
		rate = audio.Rate
	}
	var out []Timing
	var dropped []DroppedTurn
	for _, ev := range rec.Events {
		switch {
		case ev.BargeIn:
			dropped = append(dropped, DroppedTurn{TurnID: ev.TurnID, Reason: DropBargeIn})
			continue
		case ev.Overlap:
			dropped = append(dropped, DroppedTurn{TurnID: ev.TurnID, Reason: DropOverlap})
			continue
		}
		cEnd := ev.RecEndMs
		next := firstOnsetAfter(rec.Agent, rate, cEnd, audio.DefaultSpeechThreshold)
		if next < 0 {
			dropped = append(dropped, DroppedTurn{TurnID: ev.TurnID, Reason: DropNoOnset})
			continue
		}
		out = append(out, Timing{TurnID: ev.TurnID, V2VMS: next - cEnd})
	}
	return out, dropped
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
			startMs := rec.SampleMs(tool.Started)
			endMs := rec.SampleMs(tool.Ended)
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
	m.V2VP50 = Percentile(vals, 50)
	m.V2VP95 = Percentile(vals, 95)
	m.V2VMax = vals[len(vals)-1]
	bandSrc := nonTool
	if len(bandSrc) == 0 {
		bandSrc = vals
	} else {
		sort.Ints(nonTool)
		m.NonToolP50 = Percentile(nonTool, 50)
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

// CountConversation fills turn and word counts from the recording and agent transcript.
func CountConversation(m *Metrics, rec caller.Result, agentText string) {
	for _, ev := range rec.Events {
		if ev.Text {
			m.CallerTurns++
		}
	}
	m.AgentTurns = len(mergeUtterances(audio.DetectSpeech(rec.Agent, rec.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs), UtteranceMergeGapMS))
	m.AgentWords = len(strings.Fields(agentText))
}

// Percentile is the nearest-rank percentile of an already sorted slice. It is the only
// percentile convention in the benchmark: report and board both call it, so a P50 means the
// same thing wherever it is printed.
func Percentile(sorted []int, p int) int {
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

// BargeInStopMS is time from barge-in start until agent energy drops, with the reason
// there is no such time when there is not.
//
// The stop edge is read off spans merged only across pauses too short to hear as
// stopping. Merging at UtteranceMergeGapMS, which selectivity and hold do want, joins
// the interrupted reply to the one answering the correction, and reports a reply that
// stopped promptly as one that ran on.
func BargeInStopMS(rec caller.Result) (int, string) {
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
		return -1, BargeNoEvent
	}
	raw := audio.DetectSpeech(rec.Agent, rec.Rate, audio.DefaultSpeechThreshold, 80)
	for _, span := range mergeUtterances(raw, BargeInMergeGapMS) {
		if span.StartMs <= barge.RecStartMs+bargeAlignmentMS && span.EndMs > barge.RecStartMs {
			return span.EndMs - barge.RecStartMs, BargeMeasured
		}
	}
	for _, span := range mergeUtterances(raw, UtteranceMergeGapMS) {
		if span.StartMs > barge.RecStartMs+bargeAlignmentMS {
			return -1, BargeStartedAfter
		}
	}
	return -1, BargeAlreadyQuiet
}

// ScoreOverlaps evaluates every non-directed sound in the script.
func ScoreOverlaps(rec caller.Result) []OverlapCheck {
	agent := mergeUtterances(audio.DetectSpeech(rec.Agent, rec.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs), UtteranceMergeGapMS)
	var checks []OverlapCheck
	for _, event := range rec.Events {
		if event.OverlapSound == "" {
			continue
		}
		check := OverlapCheck{TurnID: event.TurnID, Sound: event.OverlapSound}
		for _, span := range agent {
			if span.StartMs >= event.RecStartMs && span.StartMs <= event.RecEndMs+200 {
				check.StartedTurn = true
			}
			if event.Kind == scenario.TriggerDuringAgent && span.StartMs < event.RecStartMs && span.EndMs >= event.RecEndMs {
				check.Continued = true
			}
		}
		checks = append(checks, check)
	}
	return checks
}

// SelectivityHold is true when no overlap sound produced a new agent turn.
func SelectivityHold(checks []OverlapCheck) bool {
	for _, check := range checks {
		if check.StartedTurn {
			return false
		}
	}
	return true
}

// HoldThroughOverlap is true when the agent continued through every mid-speech overlap.
func HoldThroughOverlap(checks []OverlapCheck) bool {
	for _, check := range checks {
		if !check.Continued {
			return false
		}
	}
	return true
}

// mergeUtterances joins speech spans separated by less than maxGapMS so a reply that
// pauses between sentences still scores as one utterance.
func mergeUtterances(spans []audio.Span, maxGapMS int) []audio.Span {
	if len(spans) == 0 {
		return nil
	}
	merged := []audio.Span{spans[0]}
	for _, span := range spans[1:] {
		last := &merged[len(merged)-1]
		if span.StartMs-last.EndMs < maxGapMS {
			last.EndMs = span.EndMs
			continue
		}
		merged = append(merged, span)
	}
	return merged
}

// FalseCutoff counts agent starts inside scripted, non-barge caller utterances.
// Script intervals remain authoritative when the caller leg also contains a noise bed.
func FalseCutoff(rec caller.Result) int {
	agentSpans := mergeUtterances(audio.DetectSpeech(rec.Agent, rec.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs), UtteranceMergeGapMS)
	n := 0
	for _, agent := range agentSpans {
		for _, event := range rec.Events {
			if event.BargeIn || event.OverlapSound != "" {
				continue
			}
			if agent.StartMs > event.RecStartMs+80 && agent.StartMs < event.RecEndMs-80 {
				n++
				break
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
	for _, phrase := range fillerPhrases {
		if strings.Contains(low, phrase) {
			return true
		}
	}
	return false
}

func containsTimedFiller(words []TranscriptWord, startMS, endMS int) bool {
	for _, phrase := range fillerPhrases {
		parts := strings.Fields(phrase)
		for i := 0; i+len(parts) <= len(words); i++ {
			if words[i].StartMS < startMS || words[i].StartMS >= endMS {
				continue
			}
			matched := true
			for j, part := range parts {
				if normalizedWord(words[i+j].Text) != part || words[i+j].StartMS >= endMS {
					matched = false
					break
				}
			}
			if matched {
				return true
			}
		}
	}
	return false
}

func normalizedWord(word string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return unicode.ToLower(r)
		}
		return -1
	}, word)
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

// ScoreFiller checks that a delayed tool does not leave the caller in silence.
//
// The gate used to require one of a set of stall phrases inside the tool window, which
// scored the words rather than the property. Measured on restaurant.tool_filler, an agent
// that read the booking back across the whole 3 s lookup — never leaving a pause, and
// saying something more useful than "one moment" — failed, and a contract rewritten to put
// the stall phrase first passed it while losing four other scenarios: the substantive reply
// moved into the post-tool turn, where noise could cut it. Silence is what a caller
// notices, so silence is what fails. FillerHeard stays recorded, as a description of how
// the wait was covered rather than a requirement.
func ScoreFiller(m *Metrics, sc scenario.Scenario, rec caller.Result, sess *world.Session, transcript Transcript) {
	names := DelayedToolNames(sc)
	if len(names) == 0 {
		return
	}
	name := names[0]
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
	startMs := rec.SampleMs(tool.Started)
	endMs := rec.SampleMs(tool.Ended)
	if startMs < 0 {
		startMs = 0
	}
	m.FillerHeard = containsTimedFiller(transcript.Words, startMs-FillerLeadInMS, endMs)
	m.FillerSilenceMS = longestSilence(rec, startMs, endMs)
	if m.FillerSilenceMS > MaxToolSilenceMS {
		m.FillerFail = append(m.FillerFail, fmt.Sprintf("caller heard nothing for %d ms while %s ran", m.FillerSilenceMS, name))
	}
	onset := firstOnsetAfter(rec.Agent, rec.Rate, startMs, audio.DefaultSpeechThreshold)
	if onset < 0 {
		onset = firstOnsetAfter(rec.Agent, rec.Rate, max(0, startMs-FillerLeadInMS), audio.DefaultSpeechThreshold)
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

// longestSilence is the longest stretch of the window the agent said nothing in, reading
// through pauses short enough to hear as one utterance rather than as dead air.
func longestSilence(rec caller.Result, startMs, endMs int) int {
	if endMs <= startMs {
		return 0
	}
	spans := mergeUtterances(audio.DetectSpeech(rec.Agent, rec.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs), BargeInMergeGapMS)
	longest := 0
	quietFrom := startMs
	for _, span := range spans {
		if span.EndMs <= startMs || span.StartMs >= endMs {
			continue
		}
		if gap := span.StartMs - quietFrom; gap > longest {
			longest = gap
		}
		if span.EndMs > quietFrom {
			quietFrom = span.EndMs
		}
	}
	if gap := endMs - quietFrom; gap > longest {
		longest = gap
	}
	return longest
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
func ApplyGates(m *Metrics, sc scenario.Scenario) {
	var notes []string
	if len(m.EndStateFail) > 0 {
		notes = append(notes, "end_state")
	}
	if len(m.ExpectedToolFail) > 0 {
		notes = append(notes, "expected_tools")
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
	if sc.HasBargeIn() && m.BargeInStopMS < 0 {
		notes = append(notes, "barge_in")
	} else if m.BargeInStopMS > MaxBargeInStopMS {
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
	m.ExpectedToolFail = world.CheckExpectedTools(sess.Tools, sc.ExpectedTools)
	m.ToolOrderFail = world.CheckToolOrder(sess.Tools, sc.ToolOrder)
	m.EntityToolFail = world.EntityInTools(sess.Tools, sc.Entities)
	m.EntitySpeechFail = EntityInSpeech(agentText, sc.Entities)
	m.ExtraTools = ExtraToolNames(sess.Tools, sc.ExpectedTools)
}

// ExtraToolNames are distinct tools the agent called that the scenario did not expect.
func ExtraToolNames(tools []world.ToolCall, expected []scenario.ExpectedTool) []string {
	if len(expected) == 0 {
		return nil
	}
	want := map[string]bool{}
	for _, tool := range expected {
		want[tool.Name] = true
	}
	var extra []string
	seen := map[string]bool{}
	for _, tool := range tools {
		if want[tool.Name] || seen[tool.Name] || tool.Name == "" {
			continue
		}
		seen[tool.Name] = true
		extra = append(extra, tool.Name)
	}
	sort.Strings(extra)
	return extra
}
