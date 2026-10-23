package agent

import (
	"context"
	"encoding/json"
	"strings"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
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

// quoteSkills is a skill that came as a folder with a script in it.
func quoteSkills() harness.Skills {
	return harness.Skills{
		Skills: []harness.Skill{{Name: "quote", Description: "Price a quote"}},
		Load: func(context.Context, string) (harness.Loaded, error) {
			return harness.Loaded{Instructions: "Run the script.", Files: map[string]string{"scripts/quote.py": "print(42)"}}, nil
		},
	}
}

// codeBox runs code as a sandbox would, recording what it was handed. generation is which
// sandbox it is, 1 unless the test makes it another.
type codeBox struct {
	mu         sync.Mutex
	result     sandbox.Result
	ran        []string
	written    []string
	generation int
}

func (b *codeBox) Ready(context.Context) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return max(b.generation, 1), nil
}

// remade is the sandbox made again, as after its provider deleted it.
func (b *codeBox) remade() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.generation = max(b.generation, 1) + 1
}

func (b *codeBox) Run(_ context.Context, code string, _ []string) (sandbox.Result, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.ran = append(b.ran, code)
	return b.result, nil
}

func (b *codeBox) Write(_ context.Context, path string, _ []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.written = append(b.written, path)
	return nil
}

func (b *codeBox) Close() error { return nil }

func (b *codeBox) writes() []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]string(nil), b.written...)
}

// codeShelf publishes files at a URL named after them, as a conversation's channel would.
type codeShelf struct{}

func (codeShelf) publish(_ context.Context, file sandbox.File) (sandbox.Attachment, error) {
	return sandbox.Attachment{Name: file.Name, MIME: file.MIME, URL: "https://cdn.example/" + file.Name, Size: len(file.Data)}, nil
}

// runsTool runs one call on a joined agent and returns what the n-th tool to settle reported.
func (s *AgentSuite) runsTool(call llm.ToolCall, n int) ToolRan {
	s.agent.runTool(harness.ToolRequested{TurnID: "turn-" + call.ID, Call: call})
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == n }, "the tool never settled")
	return toolsRanIn(s.reported())[n-1]
}

func runCodeCall(id, code string) llm.ToolCall {
	arguments, _ := json.Marshal(map[string]any{"code": code, "files": []string{"/tmp/quote.png"}})
	return llm.ToolCall{ID: id, Name: sandbox.ToolName, Arguments: string(arguments)}
}

func (s *AgentSuite) TestAWrittenConversationWithASandboxRunsItsSkillsScriptsItself() {
	s.skills = quoteSkills()
	s.box = &codeBox{result: sandbox.Result{Output: "42\n"}}
	s.joinText()

	tools := s.agent.availableTools()
	s.NoError(tools.Validate())
	_, loads := tools.Lookup(harness.LoadSkillTool)
	_, runs := tools.Lookup(sandbox.ToolName)
	s.True(loads)
	s.True(runs, "with no subagent the model holding the conversation runs the code")

	loaded := s.runsTool(llm.ToolCall{ID: "load", Name: harness.LoadSkillTool, Arguments: `{"name":"quote"}`}, 1)
	s.Require().NoError(loaded.Err)
	s.Contains(loaded.Result, "skills/quote/", "the model is told where the scripts are")

	ran := s.runsTool(runCodeCall("run-1", "import subprocess"), 2)
	s.Require().NoError(ran.Err)
	s.Contains(ran.Result, "42")
	s.Equal([]string{"skills/quote/scripts/quote.py"}, s.box.writes(), "the script is there before the code runs")

	s.runsTool(runCodeCall("run-2", "import subprocess"), 3)
	s.Len(s.box.writes(), 1, "and is not written again for the next run")
}

func (s *AgentSuite) TestANewSandboxGetsTheSkillsAgain() {
	s.skills = quoteSkills()
	s.box = &codeBox{result: sandbox.Result{Output: "42\n"}}
	s.joinText()
	loaded := s.runsTool(llm.ToolCall{ID: "load", Name: harness.LoadSkillTool, Arguments: `{"name":"quote"}`}, 1)
	s.Require().NoError(loaded.Err)
	s.runsTool(runCodeCall("run-1", "import subprocess"), 2)
	s.box.remade()

	ran := s.runsTool(runCodeCall("run-2", "import subprocess"), 3)

	s.Require().NoError(ran.Err)
	s.Equal([]string{"skills/quote/scripts/quote.py", "skills/quote/scripts/quote.py"}, s.box.writes(),
		"a sandbox made again holds nothing written into the last")
}

func (s *AgentSuite) TestFilesAWrittenConversationsCodeHandsBackRideOnTheToolResult() {
	s.box = &codeBox{result: sandbox.Result{Files: []sandbox.File{{Name: "quote.png", MIME: "image/png", Data: []byte("png")}}}}
	s.shelf = &codeShelf{}
	s.joinText()

	ran := s.runsTool(runCodeCall("run-1", "render()"), 1)
	s.Require().NoError(ran.Err)
	s.Equal([]sandbox.Attachment{{Name: "quote.png", MIME: "image/png", URL: "https://cdn.example/quote.png", Size: 3}}, ran.Files)
	s.Contains(ran.Result, "attached to the reply the person sees: quote.png")
}

func (s *AgentSuite) TestAWrittenConversationWithoutASandboxRunsNoCode() {
	s.skills = quoteSkills()
	s.joinText()

	_, runs := s.agent.availableTools().Lookup(sandbox.ToolName)
	s.False(runs)
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
