package lcm

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type ClassifierSuite struct {
	suite.Suite
}

func TestClassifierSuite(t *testing.T) {
	suite.Run(t, new(ClassifierSuite))
}

func (s *ClassifierSuite) TestARequestHasToAskSomething() {
	s.ErrorContains(Request{State: "anything"}.Validate(), "at least one question")
	s.NoError(Request{
		State:     "anything",
		Questions: map[string]Question{"heard": Noul("Did anyone speak?", "", "")},
	}.Validate())
}

func (s *ClassifierSuite) TestAQuestionIsNamedInTheRefusalItCaused() {
	// Every question in a request is asked at once, so a refusal that did not say which
	// one was wrong would leave a caller with several to check by hand.
	err := Request{
		State: "anything",
		Questions: map[string]Question{
			"heard": Noul("Did anyone speak?", "", ""),
			"floor": {Type: TypeChoice, Instructions: "  "},
		},
	}.Validate()

	s.ErrorContains(err, "floor")
}

func (s *ClassifierSuite) TestAQuestionAskingForAnAnswerOfNoKnownShapeIsRefused() {
	err := Request{
		State:     "anything",
		Questions: map[string]Question{"floor": {Type: "vibes", Instructions: "Who has it?"}},
	}.Validate()

	s.ErrorContains(err, "vibes")
}

func (s *ClassifierSuite) TestAnOptionWithNoGlossIsStillAnOption() {
	// An option a classifier was not given cannot be answered with, so one the caller
	// thought needed no description has to survive as an option rather than be dropped.
	question := Choice("Who should hold the floor?", map[string]string{
		"stop":     "A correction or a direct interruption.",
		"continue": "",
	})

	criteria, ok := question.Criteria.(map[string]*string)
	s.Require().True(ok)
	s.Require().Contains(criteria, "continue")
	s.Nil(criteria["continue"])
	s.Require().NotNil(criteria["stop"])
	s.Equal("A correction or a direct interruption.", *criteria["stop"])
}

func (s *ClassifierSuite) TestANoulSaysWhatYesAndNoMeanOnlyWhenTold() {
	// An empty gloss sent as an empty string is a description of nothing, which is worse
	// than the question standing on its own.
	s.Nil(Noul("Is anyone shouting?", "", "").Criteria)

	told := Noul("Is this a recorded menu?", "A recording listing options.", "A person talking.")
	s.Equal(map[string]string{
		"true":  "A recording listing options.",
		"false": "A person talking.",
	}, told.Criteria)
}
