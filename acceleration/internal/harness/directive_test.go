package harness

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type DirectiveSuite struct {
	suite.Suite
}

func TestDirectiveSuite(t *testing.T) {
	suite.Run(t, new(DirectiveSuite))
}

// scan feeds a reply to a scanner one delta at a time and reports everything that came
// out, which is how the caller sees it: speech goes to the voice as it arrives and
// directives are acted on the moment they close.
func scan(deltas ...string) (string, []directive) {
	var reader scanner
	var speech strings.Builder
	var found []directive

	for _, delta := range deltas {
		spoken, directives := reader.Add(delta)
		speech.WriteString(spoken)
		found = append(found, directives...)
	}
	speech.WriteString(reader.Flush())
	return speech.String(), found
}

func (s *DirectiveSuite) TestAReplyWithoutDirectivesIsSpokenWhole() {
	speech, directives := scan("Hello there. ", "How are you?")

	s.Equal("Hello there. How are you?", speech)
	s.Empty(directives)
}

func (s *DirectiveSuite) TestARequestForHelpIsTakenOutOfWhatIsSpoken() {
	speech, directives := scan(
		`Let me check that. <ask skill="think">what is 15% of 84.20</ask> One moment.`,
	)

	s.Equal("Let me check that.  One moment.", speech,
		"the caller hears the filler, never the request")
	s.Require().Len(directives, 1)
	s.Equal(kindAsk, directives[0].kind)
	s.Equal("think", directives[0].skill)
	s.Equal("what is 15% of 84.20", directives[0].body)
}

func (s *DirectiveSuite) TestATagSplitAcrossDeltasIsStillFound() {
	// A model emits a few characters at a time, so a tag almost never arrives whole.
	speech, directives := scan(
		"Let me check. ", "<a", "sk sk", `ill="th`, `ink">what is 15% `, "of 84.20</a", "sk> Right.",
	)

	s.Equal("Let me check.  Right.", speech)
	s.Require().Len(directives, 1)
	s.Equal("think", directives[0].skill)
	s.Equal("what is 15% of 84.20", directives[0].body)
}

func (s *DirectiveSuite) TestSpeechIsReleasedBeforeTheTagItPrecedes() {
	// The caller is listening to the gap, so text before a tag must not wait for the tag
	// to finish arriving.
	var reader scanner

	spoken, directives := reader.Add(`Let me check that. <ask skill="thi`)

	s.Equal("Let me check that. ", spoken, "the filler is spoken while the request is still arriving")
	s.Empty(directives)
}

func (s *DirectiveSuite) TestOnlyTheDirectiveIsHeldBack() {
	var reader scanner
	reader.Add(`One moment. <ask skill="think">`)

	spoken, directives := reader.Add("what is 15% of 84.20")

	s.Empty(spoken, "the body of a request is never spoken, not even in part")
	s.Empty(directives, "and the request is not made until it closes")
}

func (s *DirectiveSuite) TestADropCancelsWorkThatNoLongerMatters() {
	speech, directives := scan(`Never mind that. <drop skill="think"/> What else?`)

	s.Equal("Never mind that.  What else?", speech)
	s.Require().Len(directives, 1)
	s.Equal(kindDrop, directives[0].kind)
	s.Equal("think", directives[0].skill)
	s.Empty(directives[0].body)
}

func (s *DirectiveSuite) TestSeveralDirectivesInOneReplyAllLand() {
	speech, directives := scan(
		`<drop skill="recall"/>Sure. <ask skill="think">add it up</ask> and <ask skill="explain">why</ask>.`,
	)

	s.Equal("Sure.  and .", speech)
	s.Require().Len(directives, 3)
	s.Equal("recall", directives[0].skill)
	s.Equal("add it up", directives[1].body)
	s.Equal("why", directives[2].body)
}

func (s *DirectiveSuite) TestADirectiveThatNeverClosesIsDroppedRatherThanSpoken() {
	// Half a request is not a request, and its body was never meant to be heard.
	speech, directives := scan(`Let me check. <ask skill="think">what is 15% of`)

	s.Equal("Let me check. ", speech)
	s.Empty(directives)
}

func (s *DirectiveSuite) TestAnAngleBracketThatIsNotATagIsSpoken() {
	speech, directives := scan("anything under ", "<", " 5 pounds qualifies")

	s.Equal("anything under < 5 pounds qualifies", speech)
	s.Empty(directives)
}

func (s *DirectiveSuite) TestAWordThatMerelyStartsLikeATagIsSpoken() {
	speech, directives := scan("I was <asking> about the booking")

	s.Equal("I was <asking> about the booking", speech,
		"a longer word is not a directive, however it starts")
	s.Empty(directives)
}

func (s *DirectiveSuite) TestAnUnrelatedTagIsSpoken() {
	speech, directives := scan("the <b>bold</b> option")

	s.Equal("the <b>bold</b> option", speech)
	s.Empty(directives)
}

func (s *DirectiveSuite) TestALoneBracketDoesNotSilenceTheAgent() {
	// Waiting forever for a tag that is never coming would cost the caller the rest of
	// the sentence, which is worse than saying a bracket out loud.
	var reader scanner
	reader.Add("<")

	spoken, _ := reader.Add(strings.Repeat("a", tagLimit+1))

	s.Contains(spoken, "aaa", "the reply carries on once a bracket has gone on too long to be a tag")
}

func (s *DirectiveSuite) TestResetForgetsAnInterruptedReply() {
	var reader scanner
	reader.Add(`Let me check. <ask skill="think">what is`)

	reader.Reset()
	speech, directives := reader.Add("Something else entirely.")

	s.Equal("Something else entirely.", speech, "the abandoned request does not leak into the next turn")
	s.Empty(directives)
}

func (s *DirectiveSuite) TestFlushSpeaksTextHeldForATagThatNeverCame() {
	var reader scanner
	spoken, _ := reader.Add("that costs <")

	s.Equal("that costs ", spoken)
	s.Equal("<", reader.Flush(), "text held back on a guess is still text")
}

func (s *DirectiveSuite) TestASkillIsNotRequiredToBeQuoted() {
	// An unquoted attribute is a model getting the syntax slightly wrong. The request is
	// still kept out of the caller's ears; it just cannot say which skill it wanted.
	speech, directives := scan("<ask skill=think>work it out</ask> done")

	s.Equal(" done", speech)
	s.Require().Len(directives, 1)
	s.Empty(directives[0].skill)
}
