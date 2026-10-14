package agent

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// salesSkills is a skill set as a synced agent directory leaves it: held by name and
// description, its instructions read when one is loaded.
func salesSkills() harness.Skills {
	return harness.Skills{
		Skills: []harness.Skill{{Name: "sales-moderation", Description: "Moderation pricing and positioning"}},
		Load: func(_ context.Context, name string) (harness.Loaded, error) {
			return harness.Loaded{
				Instructions: "Quote the pricing page for " + name,
				Files:        map[string]string{"reference/tiers.md": "Tier 1 is per item."},
			}, nil
		},
	}
}

// loadsSkill runs one load_skill call on a joined text agent and returns what it reported.
func (s *AgentSuite) loadsSkill(name string) ToolRan {
	return s.loadsSkillWith(`{"name":"` + name + `"}`)
}

// loadsSkillWith runs one load_skill call with these arguments.
func (s *AgentSuite) loadsSkillWith(arguments string) ToolRan {
	s.model.reply = []string{"Moderation is priced per item."}
	call := llm.ToolCall{ID: "skill", Name: harness.LoadSkillTool, Arguments: arguments}
	s.agent.mu.Lock()
	s.agent.history = []llm.Message{
		{Role: llm.User, Content: "how much does moderation cost"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{call}},
	}
	s.agent.mu.Unlock()
	s.agent.runTool(harness.ToolRequested{TurnID: "skill-turn", Call: call})
	return s.awaitToolRan()
}

func (s *AgentSuite) TestAWrittenConversationLoadsItsSkillsItself() {
	// A text session has no subagent: the model holding the conversation reads the skill
	// and goes on to the knowledge base or the web with the tools it already has.
	s.skills = salesSkills()
	s.joinText()

	tool, offered := s.agent.availableTools().Lookup(harness.LoadSkillTool)
	s.Require().True(offered)
	s.Contains(tool.Description, "sales-moderation: Moderation pricing and positioning")

	ran := s.loadsSkill("sales-moderation")
	s.Require().NoError(ran.Err)
	var loaded map[string]string
	s.Require().NoError(json.Unmarshal([]byte(ran.Result), &loaded))
	s.Equal("sales-moderation", loaded["name"])
	s.True(strings.HasPrefix(loaded["instructions"], "Quote the pricing page for sales-moderation"))
	s.Contains(loaded["instructions"], "- reference/tiers.md", "the files are listed, not handed over")
	s.NotContains(loaded["instructions"], "Tier 1 is per item.")
}

func (s *AgentSuite) TestAWrittenConversationReadsASkillsFileWhenItAsksForOne() {
	s.skills = salesSkills()
	s.joinText()

	ran := s.loadsSkillWith(`{"name":"sales-moderation","path":"reference/tiers.md"}`)
	s.Require().NoError(ran.Err)
	var loaded map[string]string
	s.Require().NoError(json.Unmarshal([]byte(ran.Result), &loaded))
	s.Equal("reference/tiers.md", loaded["path"])
	s.Equal("Tier 1 is per item.", loaded["content"])
}

func (s *AgentSuite) TestAnUnknownSkillIsToldToTheModelWithTheNamesItMayUse() {
	s.skills = salesSkills()
	s.joinText()

	ran := s.loadsSkill("sales-teleport")
	s.Require().Error(ran.Err)
	s.Contains(ran.Result, "did not work")
	s.Contains(ran.Result, "sales-moderation")
}

func (s *AgentSuite) TestSkillsHandedToASubagentAreNotLoaded() {
	s.skills = salesSkills()
	s.subagent = newStubLLM()
	s.joinText()

	_, offered := s.agent.availableTools().Lookup(harness.LoadSkillTool)
	s.False(offered, "a sandboxed text session still hands its skills over")
}

func (s *AgentSuite) TestACallDoesNotLoadSkills() {
	s.skills = salesSkills()
	s.join(false)

	_, offered := s.agent.availableTools().Lookup(harness.LoadSkillTool)
	s.False(offered)
}

func (s *AgentSuite) TestNoSkillsMeansNothingToLoad() {
	s.joinText()

	_, offered := s.agent.availableTools().Lookup(harness.LoadSkillTool)
	s.False(offered)
}

func (s *AgentSuite) TestACallersOwnLoadSkillKeepsTheName() {
	s.ownsTools("the caller's own")
	s.tools.Tools = append(s.tools.Tools, harness.Tool{Name: harness.LoadSkillTool, Description: "mine"})
	s.skills = salesSkills()
	s.joinText()

	tools := s.agent.availableTools()
	s.NoError(tools.Validate(), "the name is offered once")
	tool, _ := tools.Lookup(harness.LoadSkillTool)
	s.Equal("mine", tool.Description)

	ran := s.loadsSkill("sales-moderation")
	s.Equal("the caller's own", ran.Result)
}
