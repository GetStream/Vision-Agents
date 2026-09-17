package harness

import (
	"embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"testing"
	"unicode"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// flowState is one of the situations the conversation can be in when a judgement is due.
//
// They are the twelve in [the conversation](../../../.factory/features/conversation.md), less
// the recovery from a wait that has gone on too long, which no model is asked about because it
// is decided from a clock, and plus the half-said number the prompt singles out. Several
// states want the same thing done, and that is the point: the state says what kind of
// situation it is, so a model that mistakes a menu for an unfinished sentence can be told
// apart from one that mistakes background chatter for a question.
type flowState string

const (
	// The floor is free and a turn has settled.
	stateRespond    flowState = "respond"
	stateWait       flowState = "wait"
	stateWaitDigits flowState = "wait-digits"
	stateWaitMenu   flowState = "wait-menu"
	stateClarify    flowState = "clarify"
	stateIgnore     flowState = "ignore"
	// The agent holds the floor and somebody talks over it.
	stateStop              flowState = "stop"
	stateShorten           flowState = "shorten"
	stateContinueAck       flowState = "continue-ack"
	stateContinueNoise     flowState = "continue-noise"
	stateContinueEcho      flowState = "continue-echo"
	stateContinueElsewhere flowState = "continue-elsewhere"
)

// flowStates are the states in the order a report reads best: the floor free first, then
// somebody talking over the agent.
var flowStates = []flowState{
	stateRespond, stateWait, stateWaitDigits, stateWaitMenu, stateClarify, stateIgnore,
	stateStop, stateShorten,
	stateContinueAck, stateContinueNoise, stateContinueEcho, stateContinueElsewhere,
}

// flowOutcome is what the agent ends up doing, which is what a benchmark can grade.
//
// A disposition and a floor are not separately gradeable, because the conversation reads them
// together and one overrides the other: an ignore takes the floor decision away. So a model is
// scored on what its two answers would have made the agent do.
type flowOutcome string

const (
	// outcomeAnswer replies to the caller.
	outcomeAnswer flowOutcome = "answer"
	// outcomeClarify replies with a short question instead of an answer.
	outcomeClarify flowOutcome = "answer-clarify"
	// outcomeWait leaves the words to settle further.
	outcomeWait flowOutcome = "wait"
	// outcomeIgnore drops the words.
	outcomeIgnore flowOutcome = "ignore"
	// outcomeInterrupt abandons the reply being spoken.
	outcomeInterrupt flowOutcome = "interrupt"
	// outcomeShorten stops generating and lets the audio already sent play out.
	outcomeShorten flowOutcome = "shorten"
	// outcomeContinue keeps the floor and answers what was said over it afterwards.
	outcomeContinue flowOutcome = "continue"
)

// flowOutcomes are every outcome, for the columns of a confusion matrix.
var flowOutcomes = []flowOutcome{
	outcomeAnswer, outcomeClarify, outcomeWait, outcomeIgnore,
	outcomeInterrupt, outcomeShorten, outcomeContinue,
}

// flowCase is one labelled situation.
type flowCase struct {
	ID    string    `json:"id"`
	State flowState `json:"state"`
	// Contract names the agent's own instructions in the set's Contracts, because what the
	// agent was told to do is part of deciding whether words were meant for it.
	Contract string `json:"contract"`
	History  []struct {
		Speaker string `json:"speaker"`
		Text    string `json:"text"`
	} `json:"history"`
	Participant   string `json:"participant"`
	Heard         string `json:"heard"`
	AgentSpeaking bool   `json:"agent_speaking"`
	AgentSaid     string `json:"agent_said"`
	AnotherVoice  bool   `json:"another_voice"`
	Unfinished    bool   `json:"unfinished"`
	// Expect is what the agent should do, which is what is graded.
	Expect flowOutcome `json:"expect"`
}

// flowSet is the labelled set.
type flowSet struct {
	Version   int               `json:"version"`
	Contracts map[string]string `json:"contracts"`
	Cases     []flowCase        `json:"cases"`
}

//go:embed testdata/flowbench.json
var flowBenchFS embed.FS

// loadFlowSet reads the labelled set and refuses one that would score something other than
// what it claims to.
func loadFlowSet() (flowSet, error) {
	raw, err := flowBenchFS.ReadFile("testdata/flowbench.json")
	if err != nil {
		return flowSet{}, fmt.Errorf("harness: read the flow set: %w", err)
	}
	var set flowSet
	if err := json.Unmarshal(raw, &set); err != nil {
		return flowSet{}, fmt.Errorf("harness: decode the flow set: %w", err)
	}
	return set, set.validate()
}

func (s flowSet) validate() error {
	if len(s.Cases) == 0 {
		return fmt.Errorf("harness: the flow set has no cases")
	}
	known := map[flowState]bool{}
	for _, state := range flowStates {
		known[state] = true
	}
	seen := map[string]bool{}
	for _, one := range s.Cases {
		if seen[one.ID] {
			return fmt.Errorf("harness: %q appears twice in the flow set", one.ID)
		}
		seen[one.ID] = true
		if !known[one.State] {
			return fmt.Errorf("harness: %q is in state %q, which is not one of the twelve",
				one.ID, one.State)
		}
		if _, ok := s.Contracts[one.Contract]; !ok {
			return fmt.Errorf("harness: %q names contract %q, which the set does not carry",
				one.ID, one.Contract)
		}
		if wanted := s.expected(one.State); one.Expect != wanted {
			return fmt.Errorf("harness: %q is in state %q, which wants %q rather than %q",
				one.ID, one.State, wanted, one.Expect)
		}
		if one.AgentSpeaking != (one.AgentSaid != "") {
			return fmt.Errorf("harness: %q disagrees with itself about whether the agent "+
				"is speaking", one.ID)
		}
		if one.Unfinished && !one.AgentSpeaking {
			return fmt.Errorf("harness: %q is a provisional ask with a quiet floor, which "+
				"is not a state the conversation reaches", one.ID)
		}
	}
	return nil
}

// expected is what a state wants done about it.
func (s flowSet) expected(state flowState) flowOutcome {
	switch state {
	case stateRespond:
		return outcomeAnswer
	case stateClarify:
		return outcomeClarify
	case stateWait, stateWaitDigits, stateWaitMenu:
		return outcomeWait
	// Background chatter over the agent must not stop it and must not be answered when it
	// does stop, which is an ignore rather than a continue.
	case stateIgnore, stateContinueElsewhere:
		return outcomeIgnore
	case stateStop:
		return outcomeInterrupt
	case stateShorten:
		return outcomeShorten
	default:
		return outcomeContinue
	}
}

// counts says how many cases each state has, so a report can say what a percentage is of.
func (s flowSet) counts() map[flowState]int {
	counts := map[flowState]int{}
	for _, one := range s.Cases {
		counts[one.State]++
	}
	return counts
}

// turn is the case as the production flow controller is asked about it, so the incumbent is
// measured on the prompt it actually runs rather than on a benchmark's paraphrase of it.
func (c flowCase) turn(id string, contracts map[string]string) FlowTurn {
	history := make([]llm.Message, 0, len(c.History))
	for _, said := range c.History {
		role := llm.User
		if said.Speaker == "agent" {
			role = llm.Assistant
		}
		history = append(history, llm.Message{Role: role, Content: said.Text})
	}
	return FlowTurn{
		ID:           id,
		Instructions: contracts[c.Contract],
		History:      history,
		Participant:  c.Participant,
		Text:         c.Heard,
		Speaking:     c.AgentSpeaking,
		Reply:        c.AgentSaid,
		Unfinished:   c.Unfinished,
		AnotherVoice: c.AnotherVoice,
	}
}

// state is the case as structured input, with its parts named so a question can point at one
// by path rather than describing it again in prose.
func (c flowCase) state(contracts map[string]string) map[string]any {
	conversation := make([]map[string]string, 0, len(c.History))
	for _, said := range c.History {
		conversation = append(conversation,
			map[string]string{"speaker": said.Speaker, "text": said.Text})
	}
	state := map[string]any{
		"agent_was_told":       contracts[c.Contract],
		"conversation":         conversation,
		"speaker":              c.Participant,
		"heard":                c.Heard,
		"agent_speaking":       c.AgentSpeaking,
		"speaker_has_finished": !c.Unfinished,
		"different_voice":      c.AnotherVoice,
	}
	if c.AgentSaid != "" {
		state["agent_has_said"] = c.AgentSaid
	}
	return state
}

// outcome is what the conversation would do with a disposition and a floor, given who held the
// floor when the words arrived. It is the reading converse.Ruled and converse.overlapRuled
// take, which is what makes an answer gradeable as behaviour rather than as JSON.
func (c flowCase) outcome(disposition Disposition, floor Floor) flowOutcome {
	if !c.AgentSpeaking {
		switch disposition {
		case Respond:
			return outcomeAnswer
		case Clarify:
			return outcomeClarify
		case Ignore:
			return outcomeIgnore
		default:
			return outcomeWait
		}
	}

	// Speech that was not meant for the agent cannot take the floor from it. Mid-utterance
	// that is all an ignore can do, because words still arriving are not answerable either
	// way; once they have settled an ignore drops them rather than queueing them.
	if disposition == Ignore {
		if c.Unfinished {
			return outcomeContinue
		}
		return outcomeIgnore
	}
	switch floor {
	case Stop:
		return outcomeInterrupt
	case Shorten:
		return outcomeShorten
	default:
		return outcomeContinue
	}
}

// caughtBeforeTheModel mirrors the agent's overlapNoise, which recognises a cough and the
// shorter fillers and returns without asking the controller anything.
//
// It is a copy rather than a call because the agent package imports this one, so the
// dependency cannot run the other way. Its only job is to keep the noise cases in the set
// about the model: if overlapNoise grows to cover one of them, this stops matching and
// TestTheNoiseCasesAreOnesTheModelActuallyDecides says so.
func caughtBeforeTheModel(text string) bool {
	var kept strings.Builder
	for _, symbol := range strings.ToLower(text) {
		if unicode.IsLetter(symbol) || unicode.IsDigit(symbol) || unicode.IsSpace(symbol) {
			kept.WriteRune(symbol)
		}
	}
	stripped := strings.Join(strings.Fields(kept.String()), " ")
	if stripped == "" {
		return false
	}
	if strings.Contains(stripped, "cough") || strings.Contains(stripped, "ahem") {
		return true
	}
	switch stripped {
	case "huh", "uh", "mm", "hm", "hmm":
		return true
	}
	return false
}

// percentile returns the value a fraction of the way through a set of measurements, and sorts
// what it is handed on the way.
func percentile(measured []float64, fraction float64) float64 {
	if len(measured) == 0 {
		return 0
	}
	sort.Float64s(measured)
	return measured[int(fraction*float64(len(measured)-1))]
}

type FlowSetSuite struct {
	suite.Suite
	set flowSet
}

func TestFlowSetSuite(t *testing.T) {
	suite.Run(t, new(FlowSetSuite))
}

func (s *FlowSetSuite) SetupSuite() {
	set, err := loadFlowSet()
	s.Require().NoError(err)
	s.set = set
}

func (s *FlowSetSuite) TestEveryStateIsRepresentedEnoughToMeanSomething() {
	// One case per state measures a model's luck. Ten measures the state.
	counts := s.set.counts()
	for _, state := range flowStates {
		s.GreaterOrEqual(counts[state], 10, "state %q is thinly covered", state)
	}
	s.Len(counts, len(flowStates), "the set covers states the benchmark does not name")
}

func (s *FlowSetSuite) TestTheNoiseCasesAreOnesTheModelActuallyDecides() {
	// overlapNoise recognises a cough and the shorter fillers before the controller is asked
	// at all, so a noise case it catches measures the word list rather than the model. The
	// interesting non-speech is the kind that does not match a fixed list.
	for _, one := range s.set.Cases {
		if one.State != stateContinueNoise {
			continue
		}
		s.False(caughtBeforeTheModel(one.Heard),
			"%q is caught in code, so no model is ever asked about it", one.ID)
	}
}

func (s *FlowSetSuite) TestTheFloorIsFreeInHalfTheCasesAndHeldInTheOther() {
	// The two halves ask different questions of a model, and a set weighted towards either
	// one reports an accuracy that is mostly about that half.
	var speaking int
	for _, one := range s.set.Cases {
		if one.AgentSpeaking {
			speaking++
		}
	}
	s.Equal(len(s.set.Cases)-speaking, speaking, "the halves are unbalanced")
}

func (s *FlowSetSuite) TestAnIgnoreCannotTakeTheFloorHoweverItRules() {
	// Whatever the controller says about the floor, words that were not for the agent leave
	// it talking. This is the override in overlapRuled, and grading depends on it.
	overlapping := flowCase{AgentSpeaking: true, AgentSaid: "still talking", Unfinished: true}
	settled := flowCase{AgentSpeaking: true, AgentSaid: "still talking"}

	s.Equal(outcomeContinue, overlapping.outcome(Ignore, Stop))
	s.Equal(outcomeContinue, overlapping.outcome(Ignore, Shorten))
	s.Equal(outcomeIgnore, settled.outcome(Ignore, Stop))
}

func (s *FlowSetSuite) TestTheFloorIsWhatDecidesWhileTheAgentIsTalking() {
	overlapping := flowCase{AgentSpeaking: true, AgentSaid: "still talking", Unfinished: true}

	s.Equal(outcomeInterrupt, overlapping.outcome(Wait, Stop))
	s.Equal(outcomeShorten, overlapping.outcome(Wait, Shorten))
	s.Equal(outcomeContinue, overlapping.outcome(Wait, Continue))
}

func (s *FlowSetSuite) TestTheDispositionIsWhatDecidesWhileTheAgentIsQuiet() {
	quiet := flowCase{}

	s.Equal(outcomeAnswer, quiet.outcome(Respond, Continue))
	s.Equal(outcomeClarify, quiet.outcome(Clarify, Continue))
	s.Equal(outcomeWait, quiet.outcome(Wait, Continue))
	s.Equal(outcomeIgnore, quiet.outcome(Ignore, Continue))
	s.Equal(outcomeAnswer, quiet.outcome(Respond, Stop),
		"a floor decision about a floor nobody holds decides nothing")
}

func (s *FlowSetSuite) TestACaseTheProductionPromptCannotSeeIsRefused() {
	broken := flowSet{
		Contracts: map[string]string{"restaurant": "take bookings"},
		Cases: []flowCase{{
			ID: "quiet-overlap", State: stateStop, Contract: "restaurant",
			Heard: "no, wait", Unfinished: true, Expect: outcomeInterrupt,
		}},
	}

	s.ErrorContains(broken.validate(), "quiet floor")
}

func (s *FlowSetSuite) TestALabelThatDisagreesWithItsStateIsRefused() {
	broken := flowSet{
		Contracts: map[string]string{"restaurant": "take bookings"},
		Cases: []flowCase{{
			ID: "mislabelled", State: stateWaitMenu, Contract: "restaurant",
			Heard: "For billing, press three.", Expect: outcomeIgnore,
		}},
	}

	s.ErrorContains(broken.validate(), "wait")
}

func (s *FlowSetSuite) TestTheProductionQuestionCarriesWhatTheModelNeedsToDecide() {
	// The overlap states are only decidable if the model is told what the agent is in the
	// middle of saying, and the ignore states only if it is told about the second voice.
	for _, one := range s.set.Cases {
		asked := flowQuestion(one.turn(one.ID, s.set.Contracts))
		s.Contains(asked, one.Heard, "%q does not put the words to the model", one.ID)
		if one.AgentSaid != "" {
			s.Contains(asked, one.AgentSaid,
				"%q does not say what the agent is in the middle of", one.ID)
		}
		if one.AnotherVoice {
			s.Contains(asked, "different voice",
				"%q does not mention the second voice", one.ID)
		}
		if one.Unfinished {
			s.Contains(asked, "Decide only the floor",
				"%q is provisional but asks for a disposition too", one.ID)
		}
	}
}
