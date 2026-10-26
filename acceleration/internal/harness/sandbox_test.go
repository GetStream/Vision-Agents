package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

var errNoSandbox = errors.New("the sandbox is unreachable")

// stubSandbox records the code it was asked to run and answers with whatever the test set.
type stubSandbox struct {
	mu      sync.Mutex
	ran     []string
	outputs [][]string
	// results are answered one per run before result is.
	results []sandbox.Result
	result  sandbox.Result
	err     error
	closed  bool
	// written are the files put into it, in order, by path.
	written []string
	files   map[string]string
	// remakeAfter, when set, has the sandbox made again once that many runs have run, as its
	// provider deleting it would.
	remakeAfter int
}

func (s *stubSandbox) Ready(context.Context) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.remakeAfter > 0 && len(s.ran) >= s.remakeAfter {
		return 2, nil
	}
	return 1, nil
}

func (s *stubSandbox) Write(_ context.Context, path string, data []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.written = append(s.written, path)
	if s.files == nil {
		s.files = map[string]string{}
	}
	s.files[path] = string(data)
	return nil
}

func (s *stubSandbox) writes() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.written...)
}

func (s *stubSandbox) Run(_ context.Context, code string, outputs []string) (sandbox.Result, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ran = append(s.ran, code)
	s.outputs = append(s.outputs, outputs)
	if len(s.results) > 0 {
		next := s.results[0]
		s.results = s.results[1:]
		return next, s.err
	}
	return s.result, s.err
}

func (s *stubSandbox) asked() [][]string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([][]string(nil), s.outputs...)
}

// shelf publishes files as a conversation would, at a URL named after them, or refuses to.
type shelf struct {
	mu        sync.Mutex
	published []sandbox.File
	err       error
}

func (s *shelf) publish(_ context.Context, file sandbox.File) (sandbox.Attachment, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return sandbox.Attachment{}, s.err
	}
	s.published = append(s.published, file)
	return sandbox.Attachment{Name: file.Name, MIME: file.MIME, URL: "https://cdn.example/" + file.Name, Size: len(file.Data)}, nil
}

func (s *shelf) files() []sandbox.File {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]sandbox.File(nil), s.published...)
}

func (s *stubSandbox) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}

func (s *stubSandbox) code() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.ran...)
}

// runCodeFor is the model asking to run a piece of Python and have files back.
func runCodeFor(id, code string, files ...string) [][]llm.ToolCall {
	arguments, _ := json.Marshal(map[string]any{"code": code, "files": files})
	return [][]llm.ToolCall{{{ID: id, Name: sandbox.ToolName, Arguments: string(arguments)}}}
}

// runCode is the model asking to run a piece of Python.
func runCode(id, code string) [][]llm.ToolCall {
	return [][]llm.ToolCall{{{
		ID:        id,
		Name:      sandbox.ToolName,
		Arguments: `{"code":` + strconv.Quote(code) + `}`,
	}}}
}

func (s *HarnessSuite) TestOnlyTheSubagentIsOfferedSomewhereToRunCode() {
	// The fast model is holding a conversation. Running code takes seconds it does not
	// have, so the tool exists on the other side of the handover or not at all.
	s.box = &stubSandbox{}
	s.build(true)
	s.respond("turn-1", "what is 15% of 84.20")

	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)

	s.Nil(s.fast.requests()[0].Tools, "the model on the live path is offered nothing")
	s.eventually(func() bool { return len(s.slow.requests()) == 1 }, "the subagent was never asked")
	offered := s.slow.requests()[0].Tools
	s.Require().Len(offered, 1)
	s.Equal(sandbox.ToolName, offered[0].Name)
	s.NotEmpty(offered[0].Parameters, "without a schema the model cannot fill the arguments in")
}

func (s *HarnessSuite) TestWithoutASandboxTheSubagentIsOfferedNoTools() {
	s.build(true)
	s.respond("turn-1", "what is 15% of 84.20")

	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)

	s.eventually(func() bool { return len(s.slow.requests()) == 1 }, "the subagent was never asked")
	s.Nil(s.slow.requests()[0].Tools)
}

func (s *HarnessSuite) TestCodeTheSubagentWroteRunsAndItsOutputIsAnsweredWith() {
	s.box = &stubSandbox{result: sandbox.Result{Output: "12.63\n"}}
	s.build(true)
	s.slow.calls = runCode("call-1", "print(84.20 * 0.15)")
	s.slow.automatic = "It is 12.63."
	s.respond("turn-1", "what is 15% of 84.20")

	s.reply("turn-1", `Let me check. <ask skill="think">15% of 84.20</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Equal(Done, settled.State)
	s.Equal("It is 12.63.", settled.Text, "the answer is what the model said after running it")
	s.Equal([]string{"print(84.20 * 0.15)"}, s.box.code())

	s.Require().Len(s.slow.requests(), 2, "the task was put again with what the code said")
	asked := s.slow.requests()[1]
	s.Equal(s.slow.requests()[0].ID, asked.ID, "and it is still the same task")
	last := asked.Input[len(asked.Input)-1]
	s.Equal(llm.ToolResult, last.Role)
	s.Equal("call-1", last.ToolCallID)
	s.Equal("12.63\n", last.Content)
}

func (s *HarnessSuite) TestCodeThatCouldNotBeRunIsDescribedRatherThanHidden() {
	// The subagent asked for this mid-thought. It can only do something sensible about
	// code that did not run if it is told that it did not run.
	s.box = &stubSandbox{err: errNoSandbox}
	s.build(true)
	s.slow.calls = runCode("call-1", "print(1)")
	s.slow.automatic = "I could not work it out."
	s.respond("turn-1", "what is 15% of 84.20")

	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)

	s.awaitSettled(1)
	s.Require().Len(s.slow.requests(), 2)
	last := s.slow.requests()[1].Input[len(s.slow.requests()[1].Input)-1]
	s.Contains(last.Content, errNoSandbox.Error())
}

func (s *HarnessSuite) TestCodeThatExitedBadlySaysSo() {
	s.box = &stubSandbox{result: sandbox.Result{Output: "NameError: total", ExitCode: 1}}
	s.build(true)
	s.slow.calls = runCode("call-1", "print(total)")
	s.slow.automatic = "I could not work it out."
	s.respond("turn-1", "what is 15% of 84.20")

	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)

	s.awaitSettled(1)
	s.Require().Len(s.slow.requests(), 2)
	last := s.slow.requests()[1].Input[len(s.slow.requests()[1].Input)-1]
	s.Contains(last.Content, "exited with 1")
	s.Contains(last.Content, "NameError: total")
}

func (s *HarnessSuite) TestAnAnswerAfterCodeThatFailedIsNotPassedOffAsSuccess() {
	// A subagent whose render failed still writes something, and it is usually the script
	// it meant to run. Told only that its colleague came back, the fast model says done.
	s.box = &stubSandbox{err: errNoSandbox}
	s.build(true)
	s.slow.calls = runCode("call-1", "render()")
	s.slow.automatic = "Here is the script: render()"
	s.respond("turn-1", "render the clip")

	s.reply("turn-1", `<ask skill="think">render the clip</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Equal(Done, settled.State)
	s.ErrorContains(settled.Err, errNoSandbox.Error())
	s.respond("turn-2", "")
	s.Contains(s.fast.requests()[1].Instructions, "do not claim anything that code was meant to produce")
}

func (s *HarnessSuite) TestCodeThatWorksOnASecondTryIsAnAnswer() {
	s.box = &stubSandbox{
		results: []sandbox.Result{{Output: "NameError: total", ExitCode: 1}},
		result:  sandbox.Result{Output: "12.63"},
	}
	s.build(true)
	s.slow.calls = append(runCode("call-1", "print(total)"), runCode("call-2", "print(84.20 * 0.15)")...)
	s.slow.automatic = "It is 12.63."
	s.respond("turn-1", "what is 15% of 84.20")

	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Equal(Done, settled.State)
	s.NoError(settled.Err, "the failure was recovered from")
}

func (s *HarnessSuite) TestWorkStillRunningCodeWhenItRunsOutOfRoundsFails() {
	s.box = &stubSandbox{result: sandbox.Result{Output: "not yet"}}
	s.build(true)
	for i := range toolRounds + 1 {
		s.slow.calls = append(s.slow.calls, runCode(fmt.Sprintf("call-%d", i), "try()")...)
	}
	s.respond("turn-1", "render the clip")

	s.reply("turn-1", `<ask skill="think">render the clip</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Equal(Failed, settled.State, "a task that never stopped running code never finished")
	s.True(settled.Actionable())
}

func (s *HarnessSuite) TestWorkAbandonedWhileItsCodeRanStillSettles() {
	// The completion that was running has already been and gone, so nothing else is
	// going to report this task, and a caller waiting on it would wait forever.
	s.box = &stubSandbox{result: sandbox.Result{Output: "12.63"}}
	s.build(true)
	s.slow.calls = runCode("call-1", "print(84.20 * 0.15)")
	s.respond("turn-1", "what is 15% of 84.20")
	s.reply("turn-1", `<ask skill="think">15% of 84.20</ask>`)
	s.eventually(func() bool { return len(s.box.code()) == 1 }, "the code never ran")

	s.harness.CancelTurn("turn-1", ReasonSuperseded)

	settled := s.awaitSettled(1)[0]
	s.Equal(Cancelled, settled.State)
	s.Equal(ReasonSuperseded, settled.Reason)
	s.False(s.harness.Delegating())
}

func (s *HarnessSuite) TestFilesTheCodeHandsBackAreAttachedToTheAnswer() {
	png := sandbox.File{Name: "render.png", MIME: "image/png", Data: []byte("png")}
	s.box = &stubSandbox{result: sandbox.Result{Output: "rendered\n", Files: []sandbox.File{png}}}
	s.shelf = &shelf{}
	s.build(true)
	s.slow.calls = runCodeFor("call-1", "render()", "/tmp/render.png")
	s.slow.automatic = "Here is the teapot."
	s.respond("turn-1", "render a teapot")

	s.reply("turn-1", `<ask skill="think">render a teapot</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Equal(Done, settled.State)
	s.Equal([][]string{{"/tmp/render.png"}}, s.box.asked(), "the sandbox is told which files are wanted")
	s.Equal([]sandbox.File{png}, s.shelf.files())
	s.Equal([]sandbox.Attachment{{Name: "render.png", MIME: "image/png", URL: "https://cdn.example/render.png", Size: 3}}, settled.Files)
	told := s.slow.requests()[1].Input[len(s.slow.requests()[1].Input)-1].Content
	s.Contains(told, "rendered")
	s.Contains(told, "attached to the reply the person sees: render.png")
}

func (s *HarnessSuite) TestAFileRenderedAgainReplacesTheOneBefore() {
	s.box = &stubSandbox{result: sandbox.Result{Files: []sandbox.File{{Name: "render.png", MIME: "image/png", Data: []byte("v1")}}}}
	s.shelf = &shelf{}
	s.build(true)
	s.slow.calls = append(runCodeFor("call-1", "render()", "/tmp/render.png"), runCodeFor("call-2", "render(better=True)", "/tmp/render.png")...)
	s.slow.automatic = "Here it is."
	s.respond("turn-1", "render a teapot")

	s.reply("turn-1", `<ask skill="think">render a teapot</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Len(s.shelf.files(), 2, "both renders were published")
	s.Len(settled.Files, 1, "the answer carries one picture of the teapot, not two")
}

func (s *HarnessSuite) TestWithNowhereToShowFilesTheSubagentIsToldSo() {
	s.box = &stubSandbox{result: sandbox.Result{Files: []sandbox.File{{Name: "render.png", MIME: "image/png", Data: []byte("png")}}}}
	s.build(true)
	s.slow.calls = runCodeFor("call-1", "render()", "/tmp/render.png")
	s.slow.automatic = "I rendered it, but cannot show it here."
	s.respond("turn-1", "render a teapot")

	s.reply("turn-1", `<ask skill="think">render a teapot</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Empty(settled.Files)
	told := s.slow.requests()[1].Input[len(s.slow.requests()[1].Input)-1].Content
	s.Contains(told, "nowhere to show them: render.png")
}

func (s *HarnessSuite) TestAFileThatCouldNotBePublishedIsDescribed() {
	s.box = &stubSandbox{result: sandbox.Result{Files: []sandbox.File{{Name: "render.png", MIME: "image/png", Data: []byte("png")}}}}
	s.shelf = &shelf{err: errors.New("upload refused")}
	s.build(true)
	s.slow.calls = runCodeFor("call-1", "render()", "/tmp/render.png")
	s.slow.automatic = "I could not attach it."
	s.respond("turn-1", "render a teapot")

	s.reply("turn-1", `<ask skill="think">render a teapot</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Empty(settled.Files)
	told := s.slow.requests()[1].Input[len(s.slow.requests()[1].Input)-1].Content
	s.Contains(told, "could not be attached: render.png")
}

func (s *HarnessSuite) TestAFileTheCodeDidNotWriteIsNamed() {
	s.box = &stubSandbox{result: sandbox.Result{Output: "done", Missing: []string{"/tmp/render.png"}}}
	s.shelf = &shelf{}
	s.build(true)
	s.slow.calls = runCodeFor("call-1", "render()", "/tmp/render.png")
	s.slow.automatic = "It did not render."
	s.respond("turn-1", "render a teapot")

	s.reply("turn-1", `<ask skill="think">render a teapot</ask>`)

	s.awaitSettled(1)
	told := s.slow.requests()[1].Input[len(s.slow.requests()[1].Input)-1].Content
	s.Contains(told, "were not written, or were too large to return: /tmp/render.png")
}

// TestCodeRunWithoutTheUsersConnectionTellsTheSubagentToHaveThemConnect: the subagent cannot
// call github__list_tools, so the line also says to tell the user.
func (s *HarnessSuite) TestCodeRunWithoutTheUsersConnectionTellsTheSubagentToHaveThemConnect() {
	s.box = &stubSandbox{result: sandbox.Result{Output: "status: 401", Unconnected: []sandbox.Grant{
		{Binding: "github", Hosts: []string{"api.github.com"}}}}}
	s.build(true)
	s.slow.calls = runCode("call-1", "print(issues())")
	s.slow.automatic = "Connect GitHub first."
	s.respond("turn-1", "list my issues")

	s.reply("turn-1", `<ask skill="think">list my issues</ask>`)

	s.awaitSettled(1)
	told := s.slow.requests()[1].Input[len(s.slow.requests()[1].Input)-1].Content
	s.Equal("status: 401\n\nIf this code needed github (a 401 from api.github.com), the user has not connected it: call "+
		"github__list_tools so they can, then run the code again; if you cannot call it, tell the user they must connect "+
		"github.", told)
}

// TestCodeRunJustAfterTheConnectionTellsTheSubagentToWaitFor401: a token on its way to the
// sandbox may still be missing from this run.
func (s *HarnessSuite) TestCodeRunJustAfterTheConnectionTellsTheSubagentToWaitFor401() {
	s.box = &stubSandbox{result: sandbox.Result{Output: "status: 401", Connected: []string{"github"}}}
	s.build(true)
	s.slow.calls = runCode("call-1", "print(issues())")
	s.slow.automatic = "Try again shortly."
	s.respond("turn-1", "list my issues")

	s.reply("turn-1", `<ask skill="think">list my issues</ask>`)

	s.awaitSettled(1)
	told := s.slow.requests()[1].Input[len(s.slow.requests()[1].Input)-1].Content
	s.Equal("status: 401\n\ngithub was just connected: if this run got 401, wait 20 seconds and run the code again.", told)
}

// library answers questions as the knowledge base or the web would, recording what was
// asked and on which turn.
type library struct {
	mu     sync.Mutex
	answer string
	err    error
	asked  []string
	turns  []string
}

func (l *library) ask(_ context.Context, turnID, query string) (string, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.asked = append(l.asked, query)
	l.turns = append(l.turns, turnID)
	return l.answer, l.err
}

func (l *library) questions() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.asked...)
}

func askFor(id, tool, query string) []llm.ToolCall {
	return []llm.ToolCall{{ID: id, Name: tool, Arguments: `{"query":` + strconv.Quote(query) + `}`}}
}

func (s *HarnessSuite) TestTheSubagentLooksThingsUpAndSearchesAcrossRounds() {
	s.knowledge = &library{answer: "The handbook says nothing about GPUs."}
	s.web = &library{answer: "The H200 shipped in 2024."}
	s.build(true)
	s.slow.calls = [][]llm.ToolCall{askFor("call-1", "lookup", "latest GPU"), askFor("call-2", "search", "latest Nvidia GPU")}
	s.slow.automatic = "The latest is the H200."
	s.respond("turn-1", "what is the latest GPU")

	s.reply("turn-1", `<ask skill="think">what is the latest GPU</ask>`)

	settled := s.awaitSettled(1)[0]
	s.Equal(Done, settled.State)
	s.Equal("The latest is the H200.", settled.Text)
	s.Equal([]string{"latest GPU"}, s.knowledge.questions())
	s.Equal([]string{"latest Nvidia GPU"}, s.web.questions())
	s.Equal([]string{"turn-1"}, s.web.turns, "what it found is told about the turn that asked")

	first := s.slow.requests()[0]
	var offered []string
	for _, tool := range first.Tools {
		offered = append(offered, tool.Name)
	}
	s.Equal([]string{"lookup", "search"}, offered)
	s.Contains(first.Instructions, "knowledge base (lookup) before the web (search)")

	s.Require().Len(s.slow.requests(), 3)
	found := s.slow.requests()[2].Input[len(s.slow.requests()[2].Input)-1]
	s.Equal("call-2", found.ToolCallID)
	s.Equal("The H200 shipped in 2024.", found.Content)
}

func (s *HarnessSuite) TestASearchThatFailedIsToldToTheSubagent() {
	s.web = &library{err: errors.New("search is down")}
	s.build(true)
	s.slow.calls = [][]llm.ToolCall{askFor("call-1", "search", "latest Nvidia GPU")}
	s.slow.automatic = "I could not check."
	s.respond("turn-1", "what is the latest GPU")

	s.reply("turn-1", `<ask skill="think">what is the latest GPU</ask>`)

	s.awaitSettled(1)
	s.Require().Len(s.slow.requests(), 2)
	s.NotContains(s.slow.requests()[0].Instructions, "(lookup)", "there is no order to keep with one source")
	told := s.slow.requests()[1].Input[len(s.slow.requests()[1].Input)-1].Content
	s.Contains(told, "search is down")
}

// pdfSkills is a stored skill that came as a folder, with a reference and a script.
func pdfSkills() Skills {
	return Skills{
		Skills: []Skill{{Name: "pdf", Description: "fill PDF forms", Deadline: time.Minute,
			Files: []string{"reference/forms.md", "scripts/fill.py"}}},
		Load: func(context.Context, string) (Loaded, error) {
			return Loaded{Instructions: "Fill the form.", Files: map[string]string{
				"reference/forms.md": "# Forms\nUse field names as printed.",
				"scripts/fill.py":    "print('filled')",
			}}, nil
		},
	}
}

func readSkillFile(id, path string) [][]llm.ToolCall {
	return [][]llm.ToolCall{{{ID: id, Name: ReadSkillFileTool, Arguments: `{"path":` + strconv.Quote(path) + `}`}}}
}

func (s *HarnessSuite) TestASkillsFileIsReadOnlyWhenTheSubagentAsksForIt() {
	s.skills = pdfSkills()
	s.build(true)
	s.slow.calls = readSkillFile("call-1", "reference/forms.md")
	s.slow.automatic = "Done."
	s.respond("turn-1", "fill in my form")

	s.reply("turn-1", `<ask skill="pdf">fill in my form</ask>`)

	s.Equal(Done, s.awaitSettled(1)[0].State)
	first := s.slow.requests()[0]
	s.Contains(first.Instructions, "- reference/forms.md", "the subagent is told which files there are")
	s.NotContains(first.Instructions, "Use field names as printed.", "and is not handed them all")
	s.Require().Len(first.Tools, 1, "without a sandbox it may read, not run")
	s.Equal(ReadSkillFileTool, first.Tools[0].Name)

	s.Require().Len(s.slow.requests(), 2)
	last := s.slow.requests()[1].Input[len(s.slow.requests()[1].Input)-1]
	s.Equal("# Forms\nUse field names as printed.", last.Content)
}

func (s *HarnessSuite) TestASkillsFilesAreWrittenToTheSandboxOnceBeforeItsCodeRuns() {
	s.box = &stubSandbox{result: sandbox.Result{Output: "filled"}}
	s.skills = pdfSkills()
	s.build(true)
	s.slow.calls = append(runCode("call-1", "import subprocess"), runCode("call-2", "import subprocess")...)
	s.slow.automatic = "Filled."
	s.respond("turn-1", "fill in my form")

	s.reply("turn-1", `<ask skill="pdf">fill in my form</ask>`)

	s.Equal(Done, s.awaitSettled(1)[0].State)
	s.Equal([]string{"skills/pdf/reference/forms.md", "skills/pdf/scripts/fill.py"}, s.box.writes(),
		"every file once, though the code ran twice")
	s.Equal("print('filled')", s.box.files["skills/pdf/scripts/fill.py"])
	s.Len(s.box.code(), 2)
	s.Contains(s.slow.requests()[0].Instructions, "under skills/pdf/")
}

func (s *HarnessSuite) TestASandboxMadeAgainGetsTheSkillsFilesAgain() {
	s.box = &stubSandbox{result: sandbox.Result{Output: "filled"}, remakeAfter: 1}
	s.skills = pdfSkills()
	s.build(true)
	s.slow.calls = append(runCode("call-1", "import subprocess"), runCode("call-2", "import subprocess")...)
	s.slow.automatic = "Filled."
	s.respond("turn-1", "fill in my form")

	s.reply("turn-1", `<ask skill="pdf">fill in my form</ask>`)

	s.Equal(Done, s.awaitSettled(1)[0].State)
	s.Equal([]string{"skills/pdf/reference/forms.md", "skills/pdf/scripts/fill.py",
		"skills/pdf/reference/forms.md", "skills/pdf/scripts/fill.py"}, s.box.writes(),
		"the new sandbox holds nothing written into the last")
}
