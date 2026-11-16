package bench

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

func TestRampSuite(t *testing.T) { suite.Run(t, new(RampSuite)) }

type RampSuite struct {
	suite.Suite
}

// held is a step where everything went well, for a test to spoil one part of.
func (s *RampSuite) held() Step {
	return summarise(4, []conversation{
		s.talked(4), s.talked(4), s.talked(4), s.talked(4),
	})
}

// talked is one caller that got onto the call and had that many answered turns.
func (s *RampSuite) talked(turns int) conversation {
	held := conversation{sessionMs: 120, trackMs: 400}
	for range turns {
		held.turns = append(held.turns, turn{replyMs: 500})
	}
	return held
}

func (s *RampSuite) TestAStepWhereEverythingWorkedHeld() {
	step := s.held()

	s.Equal(4, step.Joined)
	s.Equal(16, step.Answered)
	s.Equal(16, step.Turns)
	s.Empty(broke(step))
}

func (s *RampSuite) TestASlowReplyBreaksTheStep() {
	step := s.held()
	step.ReplyMs.P95 = maxReplyP95Ms + 1

	s.Contains(broke(step), "past the")
}

func (s *RampSuite) TestUnansweredTurnsBreakTheStep() {
	step := summarise(2, []conversation{s.talked(9), {sessionMs: 120, trackMs: 400, turns: []turn{{}, {}}}})

	s.Equal(9, step.Answered)
	s.Equal(11, step.Turns)
	s.Contains(broke(step), "went unanswered")
}

func (s *RampSuite) TestConversationsThatNeverGotTheAgentOnTheCallBreakTheStep() {
	step := summarise(4, []conversation{
		s.talked(4), s.talked(4), s.talked(4),
		{sessionMs: 120, failedAt: "agent.track"},
	})

	s.Equal(3, step.Joined)
	s.Equal(1, step.Errors["agent.track"])
	s.Contains(broke(step), "never got the agent onto the call")
}

func (s *RampSuite) TestARestartedRouterBreaksTheStep() {
	step := s.held()
	step.Router = Pods{Readings: 3, Restarts: 1}

	s.Contains(broke(step), "restarted")
}

func (s *RampSuite) TestASaturatedMachineIsReportedAsTheCallersFault() {
	step := s.held()
	step.Host = HostCPU{Busy: 0.95, Cores: 10}

	s.Contains(broke(step), "the caller's and not the router's")
}

func (s *RampSuite) TestAnUnansweredTurnHasNoReplyTimeOfItsOwn() {
	step := summarise(1, []conversation{{sessionMs: 1, trackMs: 1, turns: []turn{{replyMs: 300}, {}}}})

	s.Equal(1, step.ReplyMs.Count, "a turn nobody answered has no wait to put in the percentiles")
	s.InDelta(300, step.ReplyMs.P50, 0.1)
}

func (s *RampSuite) TestTheProbesAudioIsSummarised() {
	step := summarise(2, []conversation{
		{sessionMs: 1, trackMs: 1, turns: []turn{
			{replyMs: 300, scored: true, quality: toneQuality{snrDb: 40, windows: 50}},
			{replyMs: 300, scored: true, quality: toneQuality{snrDb: 12, windows: 50, glitches: 2, dropouts: 3}},
		}},
		{sessionMs: 1, trackMs: 1, turns: []turn{{replyMs: 300}}},
	})

	s.Equal(Audio{Replies: 2, SNRP50Db: 12, SNRMinDb: 12, Windows: 100, Glitches: 2, Dropouts: 3}, step.Audio)
}

func (s *RampSuite) TestAudioThatBreaksUpBreaksTheStep() {
	step := s.held()
	step.Audio = Audio{Replies: 8, Windows: 400, Glitches: 1}
	s.Empty(broke(step), "one glitch is a packet the network lost")

	step.Audio.Glitches = 12
	s.Contains(broke(step), "audio broke up in 3.0%")
}

func (s *RampSuite) TestAGappyReplyIsReported() {
	step := summarise(1, []conversation{{sessionMs: 1, trackMs: 1, turns: []turn{{replyMs: 300, gapMs: 900}}}})

	s.InDelta(900, step.GapMs.P95, 0.1)
}

func (s *RampSuite) TestARunRefusesARampThatGoesNowhere() {
	for name, options := range map[string]Options{
		"no agent":      {Start: 1, Factor: 2, Max: 4},
		"no step":       {Agent: "a", Start: 1, Factor: 1, Max: 4},
		"ceiling below": {Agent: "a", Start: 8, Factor: 2, Max: 4},
	} {
		s.Errorf(options.valid(), "%s should be refused", name)
	}
}

func (s *RampSuite) TestTheReportSaysWhatHeldAndWhatBroke() {
	report := Report{
		Label:  "before",
		Target: "https://router",
		Models: Models{STT: "mock/mock-stt", TTS: "mock/mock-tts", LLM: "mock/mock-llm"},
		Held:   8,
		Broke:  "the router restarted 1 times",
		Steps:  []Step{{Agents: 8, Sessions: 8, Joined: 8, Turns: 10, Answered: 10}},
	}

	rendered := report.Markdown()

	s.Contains(rendered, "Held 8 agents")
	s.Contains(rendered, "the router restarted 1 times")
	s.Contains(rendered, "mock/mock-stt")
}

func (s *RampSuite) TestAComparisonShowsWhatMovedAtEachStep() {
	before := Report{Label: "before", Held: 8, Steps: []Step{{Agents: 8, Turns: 10, Answered: 10, ReplyMs: Stat{P50: 500, P95: 900}}}}
	after := Report{Label: "after", Held: 16, Steps: []Step{
		{Agents: 8, Turns: 10, Answered: 10, ReplyMs: Stat{P50: 400, P95: 700}},
		{Agents: 16, Turns: 20, Answered: 20, ReplyMs: Stat{P50: 450, P95: 800}},
	}}

	rendered := Compare(before, after)

	s.Contains(rendered, "Held 8 then 16 agents")
	s.Contains(rendered, "500ms to 400ms (-20%)")
	s.Contains(rendered, "| 16 |", "a step the earlier run never reached is still shown")
	s.False(strings.Contains(rendered, "NaN"))
}

func (s *RampSuite) TestKubectlQuantitiesAreReadInTheUnitTheyAreWrittenIn() {
	s.Equal(250, quantity("250m", "m"))
	s.Equal(384, quantity("384Mi", "Mi"))
	s.Zero(quantity("", "m"))
}
