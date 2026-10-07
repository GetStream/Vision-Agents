package harness

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type SpokenSuite struct {
	suite.Suite
}

func TestSpokenSuite(t *testing.T) {
	suite.Run(t, new(SpokenSuite))
}

func (s *SpokenSuite) TestAClockTimeSaidInWordsIsRecognised() {
	for _, text := range []string{
		"a table for two at seven thirty",
		"Seven Thirty please",
		"make it eight fifteen",
		"seven forty-five works",
		"seven forty five works",
		"nine oh five",
		"nine o five",
		"seven o'clock",
		"seven oclock",
		"seven o clock",
		"around twelve thirty tomorrow",
		"half past seven",
		"quarter past eight",
		"quarter to nine",
		"quarter after six",
		"ten to eight",
		"twenty past seven",
		"twenty five past seven",
		"five to eight",
		"seven pm",
		"seven p.m.",
		"seven p m",
		"nine a.m. on Friday",
		"7 30 tonight",
		"7 pm",
		"half past 7",
		"at eleven fifty",
	} {
		s.True(spokenNumbers(text), text)
	}
}

func (s *SpokenSuite) TestADigitRunSaidInWordsIsRecognised() {
	for _, text := range []string{
		"five one two five five five zero one four two",
		"my pin is four eight two one",
		"it is one two three four five six",
		"five five five oh one four two",
		"Five, One, Two, Five, Five, Five, Zero, One, Four, Two",
		"double five oh one four two",
		"triple five zero one four two",
		"call me on 5 1 2 5 5 5 0 1 4 2",
		"it is four 8 two one",
	} {
		s.True(spokenNumbers(text), text)
	}
}

func (s *SpokenSuite) TestWordsThatAreNotATimeOrAnIdentifierAreLeftAlone() {
	for _, text := range []string{
		"",
		"hello",
		"a table for two",
		"a table for two at seven",
		"seven people",
		"we are four",
		"three or four of us",
		"one two three",
		"two kids, thirty minutes at most",
		"oh I see, oh oh oh oh",
		"I have two kids and one dog, plus one more",
		"twenty minutes from now",
		"it was a good one, thanks",
		"quarter of the table",
		"half the group",
		"to seven",
		"past seven",
		"it takes ten minutes",
		"about 3 of them",
		"double the portion please",
		"a, m and p",
	} {
		s.False(spokenNumbers(text), text)
	}
}

func (s *SpokenSuite) TestOnlyWhatTheCallerSaidLastCounts() {
	history := []llm.Message{
		{Role: llm.User, Content: "seven thirty"},
		{Role: llm.Assistant, Content: "and how many of you?"},
		{Role: llm.User, Content: "four of us"},
	}

	s.False(identifiersAlreadyComplete(history))
	s.True(identifiersAlreadyComplete(history[:1]))
	s.True(identifiersAlreadyComplete([]llm.Message{{Role: llm.User, Content: "it is at 7:30"}}),
		"digits are still recognised")
	s.True(identifiersAlreadyComplete([]llm.Message{{Role: llm.User, Content: "my member id is ABC123456"}}))
	s.True(identifiersAlreadyComplete([]llm.Message{
		{Role: llm.User, Content: "seven thirty"}, {Role: llm.Assistant, Content: "ok"},
	}), "the last thing the caller said was a time, whatever the agent said after it")
}

func (s *SpokenSuite) TestRecognisingNumbersAllocatesNothing() {
	for _, text := range []string{
		"a table for two at seven thirty, quarter to eight at the latest",
		"five one two five five five zero one four two",
		"no numbers worth speaking of in this sentence at all, just a long run of ordinary words",
	} {
		s.Zero(testing.AllocsPerRun(100, func() { spokenNumbers(text) }), text)
	}
}
