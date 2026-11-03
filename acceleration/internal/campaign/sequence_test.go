package campaign

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type SequenceSuite struct {
	suite.Suite
}

func TestSequenceSuite(t *testing.T) {
	suite.Run(t, new(SequenceSuite))
}

func (s *SequenceSuite) TestTheDefaultSequenceCallsFourTimesOverTwentyDays() {
	seq := DefaultSequence("cfg")

	var delays []int
	for _, a := range seq.First().Attempts {
		delays = append(delays, a.DelayMinutes)
	}
	s.Equal([]int{0, 4320, 10080, 14400}, delays)
	s.True(seq.DoubleDial)
	s.Equal("cfg", seq.First().ConfigID)

	success, ok := seq.First().Find(OutcomeSuccess)
	s.True(ok)
	s.Equal(EndSuccessful, success.Next.End)
	notInterested, ok := seq.First().Find(OutcomeNotInterested)
	s.True(ok)
	s.Equal(EndUnsuccessful, notInterested.Next.End)

	s.NoError(seq.Validate())
	seq.Steps[0].Goal = "Confirm the visit"
	s.NoError(seq.Ready())
}

func (s *SequenceSuite) TestAStepWithElevenAttemptsIsRefused() {
	seq := DefaultSequence("cfg")
	seq.Steps[0].Attempts = nil
	for i := 0; i < 11; i++ {
		delay := 1
		if i == 0 {
			delay = 0
		}
		seq.Steps[0].Attempts = append(seq.Steps[0].Attempts, Attempt{DelayMinutes: delay})
	}

	s.ErrorContains(seq.Validate(), "attempts")
}

func (s *SequenceSuite) TestAStepNeedsAFirstAttemptRightAwayAndLaterOnesAfterADelay() {
	seq := DefaultSequence("cfg")
	seq.Steps[0].Attempts = []Attempt{{DelayMinutes: 5}}
	s.ErrorContains(seq.Validate(), "first attempt")

	seq = DefaultSequence("cfg")
	seq.Steps[0].Attempts = []Attempt{{DelayMinutes: 0}, {DelayMinutes: 0}}
	s.ErrorContains(seq.Validate(), "delay")

	seq = DefaultSequence("cfg")
	seq.Steps[0].Attempts = nil
	s.ErrorContains(seq.Validate(), "attempts")
}

func (s *SequenceSuite) TestBranchingToAnotherStepIsRefusedForNow() {
	seq := DefaultSequence("cfg")
	seq.Steps[0].Outcomes[0].Next = Next{Step: "step_2"}

	s.ErrorContains(seq.Validate(), "branching")
}

func (s *SequenceSuite) TestEveryRouteMustEndSomewhere() {
	seq := DefaultSequence("cfg")
	seq.Steps[0].Other = Route{}
	s.Error(seq.Validate())

	seq = DefaultSequence("cfg")
	seq.Steps[0].Outcomes[0].Next = Next{End: "maybe"}
	s.Error(seq.Validate())
}

func (s *SequenceSuite) TestDoNotCallIsAlwaysAChoiceAndCannotBeRedefined() {
	seq := DefaultSequence("cfg")

	var found *Outcome
	for _, c := range seq.First().Choices() {
		if c.ID == OutcomeDoNotCall {
			found = &c
		}
	}
	s.Require().NotNil(found)
	s.Equal("The person asked not to be called or contacted again.", found.Description)
	s.Equal(EndUnsuccessful, found.Next.End)

	again, ok := seq.First().Find(OutcomeDoNotCall)
	s.True(ok)
	s.Equal(*found, again)

	seq.Steps[0].Outcomes = append(seq.Steps[0].Outcomes, Outcome{ID: OutcomeDoNotCall, Name: "x", Description: "x", Next: Next{End: EndSuccessful}})
	s.ErrorContains(seq.Validate(), OutcomeDoNotCall)
}

func (s *SequenceSuite) TestOtherAndNoResponseCannotBeCustomOutcomes() {
	for _, id := range []string{OutcomeOther, OutcomeNoResponse} {
		seq := DefaultSequence("cfg")
		seq.Steps[0].Outcomes = append(seq.Steps[0].Outcomes, Outcome{ID: id, Name: "x", Description: "x", Next: Next{End: EndSuccessful}})
		s.ErrorContains(seq.Validate(), id)
	}
}

func (s *SequenceSuite) TestOutcomeIDsMustBeUniqueAndWellFormed() {
	seq := DefaultSequence("cfg")
	seq.Steps[0].Outcomes = append(seq.Steps[0].Outcomes, seq.Steps[0].Outcomes[0])
	s.ErrorContains(seq.Validate(), "twice")

	for _, bad := range []string{"", "Won", "1won", "has space", "a-b"} {
		seq = DefaultSequence("cfg")
		seq.Steps[0].Outcomes[0].ID = bad
		s.Error(seq.Validate(), bad)
	}
}

func (s *SequenceSuite) TestNoResponseCannotEndSuccessfully() {
	seq := DefaultSequence("cfg")
	seq.Steps[0].NoResponse.Next = Next{End: EndSuccessful}

	s.ErrorContains(seq.Validate(), "no response")
}

func (s *SequenceSuite) TestASequenceHasExactlyOneStepCalledStepOne() {
	seq := DefaultSequence("cfg")
	seq.Steps = nil
	s.Error(seq.Validate())

	seq = DefaultSequence("cfg")
	seq.Steps = append(seq.Steps, seq.Steps[0])
	s.Error(seq.Validate())

	seq = DefaultSequence("cfg")
	seq.Steps[0].ID = "intro"
	s.ErrorContains(seq.Validate(), "step_1")
}

func (s *SequenceSuite) TestLaunchNeedsADescriptionForEveryOutcome() {
	seq := DefaultSequence("cfg")
	seq.Steps[0].Goal = "Confirm the visit"
	seq.Steps[0].Outcomes = append(seq.Steps[0].Outcomes, Outcome{ID: "callback", Name: "Callback", Next: Next{End: EndSuccessful}})

	s.NoError(seq.Validate())
	s.ErrorContains(seq.Ready(), "description")

	seq.Steps[0].Outcomes[2].Description = "Asked to be called later"
	seq.Steps[0].Outcomes[2].Name = ""
	s.ErrorContains(seq.Ready(), "name")
}

func (s *SequenceSuite) TestLaunchNeedsAnAgentConfig() {
	seq := DefaultSequence("")

	s.NoError(seq.Validate())
	s.ErrorContains(seq.Ready(), "agent")
}

func (s *SequenceSuite) TestGoalIsRequiredToLaunch() {
	seq := DefaultSequence("cfg")

	for _, goal := range []string{"", "   "} {
		seq.Steps[0].Goal = goal
		s.NoError(seq.Validate())
		s.ErrorContains(seq.Ready(), "goal")
	}
	seq.Steps[0].Goal = "Confirm the visit"
	s.NoError(seq.Ready())
}

func (s *SequenceSuite) TestGoalOverTheLimitIsRefused() {
	seq := DefaultSequence("cfg")

	seq.Steps[0].Goal = strings.Repeat("é", MaxGoal)
	s.NoError(seq.Validate())
	seq.Steps[0].Goal = strings.Repeat("é", MaxGoal+1)
	s.ErrorContains(seq.Validate(), "goal")
}

func (s *SequenceSuite) TestSequenceWithoutGoalStillParses() {
	seq, err := ParseSequence([]byte(`{"steps":[{"id":"step_1","config_id":"cfg","attempts":[{"delay_minutes":0}],"outcomes":[],"other":{"next":{"end":"unsuccessful"}},"no_response":{"next":{"end":"unsuccessful"}}}],"double_dial":false}`))

	s.Require().NoError(err)
	s.Empty(seq.First().Goal)
}

func (s *SequenceSuite) TestASequenceSurvivesJSON() {
	want := DefaultSequence("cfg")
	raw, err := json.Marshal(want)
	s.Require().NoError(err)

	got, err := ParseSequence(raw)
	s.Require().NoError(err)
	s.Equal(want, got)
}

func (s *SequenceSuite) TestParsingRefusesWhatIsNotASequence() {
	_, err := ParseSequence([]byte(`{"steps": "no"}`))
	s.Error(err)
	_, err = ParseSequence([]byte(`{"steps":[],"surprise":1}`))
	s.Error(err)
	_, err = ParseSequence([]byte(`{"steps":[]} {"steps":[]}`))
	s.Error(err)
}

func (s *SequenceSuite) TestTheLegacySequenceIsWhatTheMigrationWrote() {
	raw, err := json.Marshal(LegacySequence("cfg"))
	s.Require().NoError(err)

	s.JSONEq(`{"steps":[{"id":"step_1","config_id":"cfg","attempts":[{"delay_minutes":0}],"outcomes":[],"other":{"next":{"end":"unsuccessful"}},"no_response":{"next":{"end":"unsuccessful"}}}],"double_dial":false}`, string(raw))

	legacy := LegacySequence("cfg")
	s.NoError(legacy.Validate())
	s.ErrorContains(legacy.Ready(), "goal")
}
