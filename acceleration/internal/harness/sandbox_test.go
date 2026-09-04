package harness

import (
	"context"
	"errors"
	"strconv"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

var errNoSandbox = errors.New("the sandbox is unreachable")

// stubSandbox records the code it was asked to run and answers with whatever the test set.
type stubSandbox struct {
	mu     sync.Mutex
	ran    []string
	result sandbox.Result
	err    error
	closed bool
}

func (s *stubSandbox) Run(_ context.Context, code string) (sandbox.Result, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ran = append(s.ran, code)
	return s.result, s.err
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
