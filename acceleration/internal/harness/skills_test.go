package harness

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type SkillsSuite struct {
	suite.Suite
}

func TestSkillsSuite(t *testing.T) {
	suite.Run(t, new(SkillsSuite))
}

func (s *SkillsSuite) TestTheBuiltInSkillsAreUsable() {
	skills, err := DefaultSkills()

	s.Require().NoError(err)
	s.NotEmpty(skills.Skills)
	for _, skill := range skills.Skills {
		s.NotEmpty(skill.Description, "%s must say what it is for", skill.Name)
		s.NotEmpty(skill.Instructions, "%s must say how to do it", skill.Name)
		s.Positive(skill.Deadline, "%s must give up eventually", skill.Name)
		s.Contains(skill.Instructions, needPrefix,
			"%s must tell the subagent how to ask for what only the caller knows", skill.Name)
	}
}

func (s *SkillsSuite) TestASkillWithoutADeadlineGetsOne() {
	// Work with no deadline would keep a caller waiting through small talk forever.
	skills, err := parseSkills([]byte(
		"skills:\n  - name: think\n    description: hard questions\n    instructions: go on\n"))

	s.Require().NoError(err)
	s.Equal(defaultDeadline, skills.Skills[0].Deadline)
}

func (s *SkillsSuite) TestADeclaredDeadlineIsKept() {
	skills, err := parseSkills([]byte(
		"skills:\n  - name: think\n    description: hard\n    instructions: go on\n    deadline: 5s\n"))

	s.Require().NoError(err)
	s.Equal(5*time.Second, skills.Skills[0].Deadline)
}

func (s *SkillsSuite) TestASkillDeclaredTwiceIsRefused() {
	_, err := parseSkills([]byte(
		"skills:\n" +
			"  - name: think\n    description: a\n    instructions: b\n" +
			"  - name: think\n    description: c\n    instructions: d\n"))

	s.ErrorContains(err, "declared twice")
}

func (s *SkillsSuite) TestTheModelIsToldWhatEachSkillIsForAndHowToAsk() {
	prompt := testSkills().Prompt()

	s.Contains(prompt, "- think: hard questions")
	s.Contains(prompt, "- recall: earlier in the call")
	s.Contains(prompt, `<ask skill="name">`)
	s.Contains(prompt, `<drop skill="name"/>`)
	s.Contains(prompt, "never spoken aloud", "or the model would read its own request out")
	s.NotContains(prompt, "think it through",
		"the fast model is told what a skill is for, never how the subagent does it")
}

func (s *SkillsSuite) TestNoSkillsMeansNothingIsAddedToThePrompt() {
	s.Empty(Skills{}.Prompt())
}

func (s *SkillsSuite) TestLookupFindsASkillByName() {
	skill, found := testSkills().Lookup("recall")
	s.True(found)
	s.Equal("read the transcript", skill.Instructions)

	_, found = testSkills().Lookup("teleport")
	s.False(found)
}

func (s *SkillsSuite) TestAMissingFileIsReported() {
	_, err := LoadSkills(filepath.Join(s.T().TempDir(), "nope.yaml"))

	s.ErrorContains(err, "read skills")
}

func (s *SkillsSuite) TestNoPathMeansTheBuiltInSet() {
	skills, err := LoadSkills("")

	s.Require().NoError(err)
	s.NotEmpty(skills.Skills)
}

func (s *SkillsSuite) TestASubagentThatNeedsSomethingIsRecognised() {
	text, question := answer("NEED: which date did you want?")
	s.Empty(text)
	s.Equal("which date did you want?", question)

	text, question = answer("  It is 12.63.  ")
	s.Equal("It is 12.63.", text)
	s.Empty(question)
}
