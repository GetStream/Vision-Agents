package harness

import (
	"context"
	"path/filepath"
	"strings"
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
	s.Contains(prompt, "complete identifiers",
		"a complete read-back is the fast model's, not a three-second colleague")
	s.NotContains(prompt, "think it through",
		"the fast model is told what a skill is for, never how the subagent does it")
	s.Contains(prompt, "in your own words")
	s.NotContains(strings.ToLower(prompt), "let me check",
		"an example phrase is the one every hand-over opens with")
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

func (s *SkillsSuite) TestASkillsInstructionsAreReadByName() {
	instructions, err := testSkills().Instructions(context.Background(), "recall")
	s.Require().NoError(err)
	s.Equal("read the transcript", instructions)

	_, err = testSkills().Instructions(context.Background(), "teleport")
	s.ErrorContains(err, `no skill called "teleport"`)
	s.ErrorContains(err, "think, recall", "the model is told which names it may use")
}

func (s *SkillsSuite) TestAStoredSkillIsReadWhenItIsLoaded() {
	// A stored skill is held by name and description; its instructions are read when used,
	// so one edited mid-conversation is loaded as it now is.
	read := 0
	skills := Skills{
		Skills: []Skill{{Name: "sales-moderation", Description: "moderation pricing"}},
		Load: func(_ context.Context, name string) (Loaded, error) {
			read++
			return Loaded{Instructions: "instructions for " + name}, nil
		},
	}

	instructions, err := skills.Instructions(context.Background(), "sales-moderation")
	s.Require().NoError(err)
	s.Equal("instructions for sales-moderation", instructions)
	s.Equal(1, read)
}

func (s *SkillsSuite) TestOnlyAFileThatIsNotADocumentIsAScript() {
	s.Equal([]string{"scripts/fill.py", "scripts/merge.go"},
		Scripts([]string{"reference/forms.md", "scripts/merge.go", "NOTES.TXT", "scripts/fill.py", "guide.mdx"}))
	s.Empty(Scripts([]string{"reference/forms.md"}))
}

func (s *SkillsSuite) TestASkillWithScriptsNeedsASandbox() {
	err := NeedsSandbox("pdf", []string{"reference/forms.md", "scripts/fill.py"}, false)
	s.ErrorContains(err, `skill "pdf" runs scripts (scripts/fill.py), so the agent needs a sandbox`)

	s.NoError(NeedsSandbox("pdf", []string{"scripts/fill.py"}, true))
	s.NoError(NeedsSandbox("pdf", []string{"reference/forms.md"}, false), "a reference is read, not run")
}

func (s *SkillsSuite) TestASkillsFileIsReadByItsPath() {
	skills := Skills{
		Skills: []Skill{{Name: "pdf", Description: "fill PDF forms", Files: []string{"reference/forms.md"}}},
		Load: func(context.Context, string) (Loaded, error) {
			return Loaded{Instructions: "fill it", Files: map[string]string{"reference/forms.md": "# Forms"}}, nil
		},
	}

	content, err := skills.File(context.Background(), "pdf", "./reference/forms.md")
	s.Require().NoError(err)
	s.Equal("# Forms", content)

	_, err = skills.File(context.Background(), "pdf", "reference/missing.md")
	s.ErrorContains(err, "it has: reference/forms.md", "the model is told which files there are")
}

func (s *SkillsSuite) TestLoadToolOffersEachSkillByNameAndDescription() {
	skills := Skills{Skills: []Skill{{Name: "sales-chat", Description: "Chat pricing"}, {Name: "sales-video", Description: "Video pricing"}}}
	tool := skills.LoadTool()

	s.Equal(LoadSkillTool, tool.Name)
	s.Contains(tool.Description, "- sales-chat: Chat pricing")
	s.Contains(tool.Description, "- sales-video: Video pricing")
	s.Contains(tool.Description, "before any other tool")
	s.Equal([]string{"sales-chat", "sales-video"},
		tool.Parameters["properties"].(map[string]any)["name"].(map[string]any)["enum"])
	s.NoError(Tools{Tools: []Tool{tool}}.Validate())
}

func (s *SkillsSuite) TestTextSkillsAcceptCompleteWrittenRequests() {
	skills := Skills{Skills: []Skill{{Name: "sales-chat", Description: "Chat product evaluation"}, {Name: "sales-video", Description: "Video product evaluation"}}}
	prompt := skills.TextPrompt()
	s.Contains(prompt, `<ask skill="name">`)
	s.Contains(prompt, "sales-chat")
	s.Contains(prompt, "sales-video")
	s.NotContains(prompt, "truncated")
	s.Contains(skills.Prompt(), "truncated", "voice behavior is preserved")
}
