package harness

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

var errNoSandbox = errors.New("the sandbox is unreachable")

// stubSandbox records the code it was asked to run and answers with whatever the test set.
type stubSandbox struct {
	mu      sync.Mutex
	ran     []string
	outputs [][]string
	result  sandbox.Result
	err     error
	closed  bool
}

func (s *stubSandbox) Run(_ context.Context, code string, outputs []string) (sandbox.Result, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ran = append(s.ran, code)
	s.outputs = append(s.outputs, outputs)
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
