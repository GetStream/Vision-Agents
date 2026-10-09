package report

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/score"
)

// Why a call failed, at the level someone fixing it would start from.
const (
	// CauseHeard is the caller saying a value the agent's speech-to-text never heard.
	CauseHeard = "heard wrong"
	// CauseDid is the agent hearing what it needed and acting wrongly, or not at all.
	CauseDid = "did wrong"
	// CauseSaid is the agent saying something it should not: a policy or say-do break, or a
	// value it read back wrong.
	CauseSaid = "said wrong"
	// CauseTurns is the agent taking or holding the floor wrongly: talking over the caller,
	// not stopping for a barge-in, no filler while a tool runs.
	CauseTurns = "turn-taking"
	// CauseInfra is a call that produced no verdict, or a target that failed under it.
	CauseInfra = "infra"
)

// causeOrder is how causes are listed, most actionable first.
var causeOrder = []string{CauseHeard, CauseDid, CauseSaid, CauseTurns, CauseInfra}

// Failure is one failed check, what it checked, and why it failed.
type Failure struct {
	Cause   string
	Gate    string
	Message string
	// Cascade marks a failure that follows from an earlier one, such as a booking not made
	// because availability was never checked.
	Cascade bool
}

// CallerLine is one scripted caller line and what the agent's speech-to-text made of it.
type CallerLine struct {
	TurnID string
	Script string
	Heard  string
	// Missed are the scenario's values in the line that the agent did not hear.
	Missed []string
}

// AgentTurn is one turn the router took, what the agent said in it and where the time went.
type AgentTurn struct {
	Said        string
	AfterTool   bool
	Interrupted bool
	Stages      []Stage
	Models      []string
}

// Stage is one part of a turn's time.
type Stage struct {
	Name string
	Ms   int
}

// ToolUse is one tool call the agent made and what came back.
type ToolUse struct {
	Name       string
	Args       string
	Result     string
	DurationMs int
	Error      string
}

// CallDetail is everything known about how one call went, read from its artifacts.
type CallDetail struct {
	Failures []Failure
	Caller   []CallerLine
	Agent    []AgentTurn
	Tools    []ToolUse
	// ToolNotes say how well the tools were used: arguments right, repeats, made-up names.
	ToolNotes  []string
	TurnTaking []string
	JudgeNotes string
}

// Cause is the call's main reason for failing: that of its first failure that is not a
// consequence of another, or "" when it passed.
func (d CallDetail) Cause() string {
	for _, failure := range d.Failures {
		if !failure.Cascade {
			return failure.Cause
		}
	}
	if len(d.Failures) > 0 {
		return d.Failures[0].Cause
	}
	return ""
}

// LoadCallDetail reads a call's artifacts from call.Dir. A missing file leaves its part
// empty, so a call from a target that records less still reads as far as it goes. sc is the
// call's scenario, for its scripted lines and the values to listen for, or nil.
func LoadCallDetail(call CallResult, sc *scenario.Scenario) CallDetail {
	var heard []heardLine
	readJSON(call.Dir, "heard.json", &heard)
	heardText := settled(heard)

	script := ""
	if sc != nil {
		script = callerScript(*sc)
	}
	detail := CallDetail{
		Failures:   classify(call, strings.Join(heardText, " "), script),
		ToolNotes:  toolNotes(call.Metrics),
		TurnTaking: turnTaking(call.Metrics),
	}
	if sc != nil {
		detail.Caller = callerLines(*sc, heardText)
	}

	var timeline []timelineTurn
	readJSON(call.Dir, "timeline.json", &timeline)
	sort.SliceStable(timeline, func(i, j int) bool { return timeline[i].StartedAt.Before(timeline[j].StartedAt) })
	for _, turn := range timeline {
		detail.Agent = append(detail.Agent, turn.agentTurn())
	}

	var tools []toolCall
	readJSON(call.Dir, "tools.json", &tools)
	for _, tool := range tools {
		detail.Tools = append(detail.Tools, ToolUse{
			Name: tool.Name, Args: compactJSON(tool.Args, 240), Result: compactJSON(tool.Result, 240),
			DurationMs: tool.DurationMS, Error: tool.Error,
		})
	}

	var judge struct {
		Notes string `json:"notes"`
	}
	readJSON(call.Dir, "judge.json", &judge)
	detail.JudgeNotes = judge.Notes
	return detail
}

// classify turns every failed gate into a failure with its cause. A value the agent got
// wrong is put down to hearing when the caller's script says it and the agent's
// speech-to-text never heard it, so a misheard name is not blamed on the model that wrote
// down what it was given. A value the caller only implies ("no allergies" for none) is not.
func classify(call CallResult, heard, script string) []Failure {
	var out []Failure
	if callOutcome(call) == OutcomeInvalid {
		for _, reason := range call.InvalidReason {
			out = append(out, Failure{Cause: CauseInfra, Gate: "invalid", Message: reason})
		}
		if call.Error != "" && len(call.InvalidReason) == 0 {
			out = append(out, Failure{Cause: CauseInfra, Gate: "invalid", Message: call.Error})
		}
		return out
	}
	if containsNote(call.Metrics.GateNotes, "target") {
		out = append(out, Failure{Cause: CauseInfra, Gate: "target", Message: call.Error})
	}
	for _, gate := range gateDetails(call.Metrics) {
		failure := Failure{Gate: gate.Gate, Message: gate.Message, Cascade: gate.Cascade}
		switch gate.Gate {
		case "policy", "say_do":
			failure.Cause = CauseSaid
		case "filler", "barge_in", "selectivity", "hold", "false_cutoff":
			failure.Cause = CauseTurns
		case "entity_speech":
			failure.Cause = CauseSaid
		default:
			failure.Cause = CauseDid
		}
		if value := wantedValue(gate.Gate, gate.Message); value != "" && heard != "" &&
			heardValue(script, value) && !heardValue(heard, value) {
			failure.Cause = CauseHeard
			failure.Message += fmt.Sprintf(" (never heard %q)", value)
		}
		out = append(out, failure)
	}
	return out
}

// wantedValue is the value a failed check expected, when it names one: "x want V got W" for
// world state and tool arguments, "name=V" for entities.
func wantedValue(gate, message string) string {
	if strings.HasPrefix(gate, "entity_") {
		if _, value, ok := strings.Cut(message, "="); ok {
			return value
		}
		return ""
	}
	if _, rest, ok := strings.Cut(message, " want "); ok {
		if value, _, ok := strings.Cut(rest, " got "); ok && value != "<nil>" {
			return value
		}
	}
	return ""
}

// heardValue is whether heard contains value, reading spoken digits and spelled letters
// ("five one two", "a l v a r e z") as the characters they stand for.
func heardValue(heard, value string) bool {
	if scenario.MatchValue(heard, value) {
		return true
	}
	want := alnum(value)
	if len(want) < 3 {
		return false
	}
	return strings.Contains(alnum(spelled(heard)), want)
}

var digitWords = map[string]string{
	"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
	"five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

// spelled joins runs of spoken digits and single letters into the string they spell.
func spelled(text string) string {
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	var out []string
	run := ""
	for _, word := range words {
		if digit, ok := digitWords[word]; ok {
			run += digit
			continue
		}
		if len(word) == 1 {
			run += word
			continue
		}
		if run != "" {
			out = append(out, run)
			run = ""
		}
		out = append(out, word)
	}
	if run != "" {
		out = append(out, run)
	}
	return strings.Join(out, " ")
}

func alnum(text string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// callerScript is every line the caller says, as one text.
func callerScript(sc scenario.Scenario) string {
	var lines []string
	for _, turn := range sc.Turns {
		lines = append(lines, turn.Text)
	}
	return strings.Join(lines, " ")
}

// callerLines pairs each scripted caller line with the settled transcript that reads most
// like it, and names the scenario's values in the line that the agent did not hear.
func callerLines(sc scenario.Scenario, heard []string) []CallerLine {
	var out []CallerLine
	for _, turn := range sc.Turns {
		if strings.TrimSpace(turn.Text) == "" {
			continue
		}
		line := CallerLine{TurnID: turn.ID, Script: turn.Text}
		best := 2.0
		for _, text := range heard {
			if wer := score.ScoreWER(turn.Text, text, true).WER; wer < best {
				best, line.Heard = wer, text
			}
		}
		for _, entity := range sc.Entities {
			if scenario.MatchValue(turn.Text, entity.Value) && !heardValue(line.Heard, entity.Value) {
				line.Missed = append(line.Missed, entity.Value)
			}
		}
		out = append(out, line)
	}
	return out
}

// toolNotes is how well the agent used its tools, none of which gates a pass.
func toolNotes(m score.Metrics) []string {
	var out []string
	if m.ArgsExpected > 0 {
		out = append(out, fmt.Sprintf("%d of %d expected arguments right", m.ArgsRight, m.ArgsExpected))
	}
	if len(m.RepeatedTools) > 0 {
		out = append(out, "called more than once: "+strings.Join(m.RepeatedTools, ", "))
	}
	if len(m.ExtraTools) > 0 {
		out = append(out, "called tools the scenario does not need: "+strings.Join(m.ExtraTools, ", "))
	}
	if len(m.UnknownTools) > 0 {
		out = append(out, "called tools that do not exist: "+strings.Join(m.UnknownTools, ", "))
	}
	return out
}

// turnTaking is what the recording says about how the agent took and held the floor,
// including what is reported without gating a pass.
func turnTaking(m score.Metrics) []string {
	var out []string
	if m.FalseCutoff > 0 {
		out = append(out, fmt.Sprintf("talked over the caller %d time(s)", m.FalseCutoff))
	}
	if m.BargeInStopMS > 0 {
		out = append(out, fmt.Sprintf("stopped %d ms after the caller barged in (limit %d ms)", m.BargeInStopMS, score.MaxBargeInStopMS))
	}
	for _, dropped := range m.Dropped {
		out = append(out, fmt.Sprintf("no reply time for turn %s: %s", dropped.TurnID, strings.ReplaceAll(dropped.Reason, "_", " ")))
	}
	if len(m.FillerFail) == 0 && m.FillerHeard {
		out = append(out, "said a filler while a tool ran")
	}
	return out
}

type heardLine struct {
	Kind string    `json:"kind"`
	Said string    `json:"said"`
	At   time.Time `json:"at"`
}

// settled is the transcripts the router acted on, or every transcript when none settled.
func settled(heard []heardLine) []string {
	var answers, all []string
	for _, line := range heard {
		all = append(all, line.Said)
		if line.Kind == "answer" {
			answers = append(answers, line.Said)
		}
	}
	if len(answers) > 0 {
		return answers
	}
	return all
}

type timelineTurn struct {
	TurnID      string    `json:"turn_id"`
	StartedAt   time.Time `json:"started_at"`
	Heard       string    `json:"heard"`
	Interrupted bool      `json:"interrupted"`
	DecisionMs  float64   `json:"decision_ms"`
	ModelToText float64   `json:"model_to_first_text_ms"`
	TextToTTS   float64   `json:"text_to_tts_ms"`
	TTSToAudio  float64   `json:"tts_to_audio_ms"`
	RoundtripMs float64   `json:"roundtrip_ms"`
	ModelCalls  []struct {
		Model    string  `json:"model"`
		Purpose  string  `json:"purpose"`
		TTFTMs   float64 `json:"ttft_ms"`
		Duration float64 `json:"duration_ms"`
		Success  bool    `json:"success"`
	} `json:"model_calls"`
}

// agentTurn reads a router turn. Its "heard" is what the caller heard: the agent's words.
func (t timelineTurn) agentTurn() AgentTurn {
	turn := AgentTurn{
		Said:        t.Heard,
		AfterTool:   strings.HasPrefix(t.TurnID, "tool-"),
		Interrupted: t.Interrupted,
	}
	for _, stage := range []Stage{
		{"decision", int(t.DecisionMs)},
		{"model to first text", int(t.ModelToText)},
		{"text to TTS", int(t.TextToTTS)},
		{"TTS to audio", int(t.TTSToAudio)},
		{"roundtrip", int(t.RoundtripMs)},
	} {
		if stage.Ms > 0 {
			turn.Stages = append(turn.Stages, stage)
		}
	}
	for _, call := range t.ModelCalls {
		model := fmt.Sprintf("%s %s: first token %d ms, %d ms", call.Purpose, call.Model, int(call.TTFTMs), int(call.Duration))
		if !call.Success {
			model += ", failed"
		}
		turn.Models = append(turn.Models, model)
	}
	return turn
}

type toolCall struct {
	Name       string `json:"name"`
	Args       any    `json:"args"`
	Result     any    `json:"result"`
	DurationMS int    `json:"duration_ms"`
	Error      string `json:"error"`
}

func readJSON(dir, name string, into any) {
	if dir == "" {
		return
	}
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return
	}
	_ = json.Unmarshal(raw, into)
}

func compactJSON(value any, limit int) string {
	if value == nil {
		return ""
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	text := string(raw)
	if len(text) > limit {
		text = text[:limit] + "…"
	}
	return text
}

// CauseCount is how many failed calls came down to one cause.
type CauseCount struct {
	Cause string
	Calls int
}

// CauseCounts counts failed calls by their main cause, in causeOrder.
func CauseCounts(details []CallDetail) []CauseCount {
	counts := map[string]int{}
	for _, detail := range details {
		if cause := detail.Cause(); cause != "" {
			counts[cause]++
		}
	}
	var out []CauseCount
	for _, cause := range causeOrder {
		if counts[cause] > 0 {
			out = append(out, CauseCount{Cause: cause, Calls: counts[cause]})
		}
	}
	return out
}

// CausesText is the counts as they read in a message: "2 did wrong, 1 heard wrong".
func CausesText(counts []CauseCount) string {
	parts := make([]string, 0, len(counts))
	for _, count := range counts {
		parts = append(parts, fmt.Sprintf("%d %s", count.Calls, count.Cause))
	}
	return strings.Join(parts, ", ")
}
