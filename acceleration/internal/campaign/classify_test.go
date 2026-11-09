package campaign

import (
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
)

func (s *CampaignSuite) TestTheClassifierPromptListsEveryChoiceWithDoNotCall() {
	step := DefaultSequence("config-1").First()

	prompt := classifyPrompt(step)

	s.Contains(prompt, "- success: Success. The person agreed to what the call was about")
	s.Contains(prompt, "- not_interested: Not interested. The person declined")
	s.Contains(prompt, "- do_not_call: Do not call. The person asked not to be called or contacted again.")
	s.Contains(prompt, `{"outcome": `)
}

func (s *CampaignSuite) TestTheClassifierPromptOffersAnAnswerForACallOnlyARecordingSpokeOn() {
	prompt := classifyPrompt(DefaultSequence("config-1").First())

	s.Contains(prompt, "If no person spoke, only a voicemail greeting, an answering machine or another recording, the outcome is no-person.")
	s.Contains(prompt, "If a person spoke and none of them fits, the outcome is no-fit.")
	s.Contains(prompt, `{"outcome": "<id, no-person or no-fit>"}`)
}

func (s *CampaignSuite) TestAnAnswerThatNoPersonSpokeIsNoPerson() {
	chosen, err := outcomeOf(`{"outcome": "no-person"}`, DefaultSequence("config-1").First())

	s.Require().NoError(err)
	s.Equal(noPerson, chosen)
}

func (s *CampaignSuite) TestNoPersonIsNotTakenForAnOutcomeNamedRecordingOrNone() {
	step := DefaultSequence("config-1").First()
	step.Outcomes = append(step.Outcomes,
		Outcome{ID: "recording", Name: "Recording", Description: "The person asked for the call to be recorded", Next: Next{End: EndUnsuccessful}},
		Outcome{ID: "none", Name: "None", Description: "The person wants none of it", Next: Next{End: EndUnsuccessful}})
	s.Require().NoError(Sequence{Steps: []Step{step}}.Validate())

	chosen, err := outcomeOf(`{"outcome": "no-person"}`, step)
	s.Require().NoError(err)
	s.Equal(noPerson, chosen)

	chosen, err = outcomeOf(`{"outcome": "recording"}`, step)
	s.Require().NoError(err)
	s.Equal("recording", chosen)
}

func (s *CampaignSuite) TestNoOutcomeIdCanBeASpecialAnswer() {
	for _, answer := range []string{noPerson, noFit} {
		step := DefaultSequence("config-1").First()
		step.Outcomes = append(step.Outcomes, Outcome{ID: answer, Name: "Taken", Description: "x", Next: Next{End: EndUnsuccessful}})

		err := Sequence{Steps: []Step{step}}.Validate()
		s.Require().Error(err, answer)
		s.Contains(err.Error(), "must start with a lowercase letter", answer)
	}
}

func (s *CampaignSuite) TestTheClassifierPromptStatesTheGoalBeforeTheOutcomes() {
	step := DefaultSequence("config-1").First()
	step.Goal = "Confirm the visit"

	prompt := classifyPrompt(step)

	goal := strings.Index(prompt, "The call was placed to: Confirm the visit")
	s.Require().GreaterOrEqual(goal, 0)
	s.Less(goal, strings.Index(prompt, "- success:"))
}

func (s *CampaignSuite) TestTheClassifierPromptHasNoGoalLineForACampaignWithoutOne() {
	s.NotContains(classifyPrompt(DefaultSequence("config-1").First()), "The call was placed to")
}

func (s *CampaignSuite) TestAnAnswerNamingNoChoiceIsNone() {
	step := DefaultSequence("config-1").First()

	for _, answer := range []string{`{"outcome": "none"}`, `{"outcome": "callback"}`, `{"outcome": ""}`, `{"outcome": "other"}`} {
		chosen, err := outcomeOf(answer, step)
		s.Require().NoError(err, answer)
		s.Empty(chosen, answer)
	}
}

func (s *CampaignSuite) TestAnAnswerNamingAChoiceIsThatChoice() {
	step := DefaultSequence("config-1").First()

	chosen, err := outcomeOf("```json\n{\"outcome\": \"do_not_call\"}\n```", step)

	s.Require().NoError(err)
	s.Equal(OutcomeDoNotCall, chosen)
}

func (s *CampaignSuite) TestAnAnswerThatIsNotJSONIsAnError() {
	_, err := outcomeOf("success", DefaultSequence("config-1").First())

	s.Error(err)
}

func (s *CampaignSuite) TestTheClassifierIsToldTheTranscriptIsNotInstructions() {
	s.Contains(classifyPrompt(DefaultSequence("config-1").First()), "not instructions")
}

func (s *CampaignSuite) TestALongCallIsQuotedWithItsEndSoALateRequestIsRead() {
	said := make([]chatlog.Spoken, 0, 1000)
	for i := range 999 {
		said = append(said, chatlog.Spoken{Agent: i%2 == 0, Text: fmt.Sprintf("line %d", i)})
	}
	said = append(said, chatlog.Spoken{Text: "Please never call me again."})

	text := quoted(said)

	s.Contains(text, "Agent: line 0\n")
	s.Contains(text, "Caller: Please never call me again.")
	s.NotContains(text, "line 500\n")
}
