package campaign

func (s *CampaignSuite) TestTheClassifierPromptListsEveryChoiceWithDoNotCall() {
	step := DefaultSequence("config-1").First()

	prompt := classifyPrompt(step)

	s.Contains(prompt, "- success: Success. The person agreed to what the call was about")
	s.Contains(prompt, "- not_interested: Not interested. The person declined")
	s.Contains(prompt, "- do_not_call: Do not call. The person asked not to be called or contacted again.")
	s.Contains(prompt, `{"outcome": `)
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
