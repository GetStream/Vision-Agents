package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// stubTelephony is a phone line with no phone network in it: it records what it was asked
// to do, and fails when the test wants to see what a failed transfer does to the
// conversation.
type stubTelephony struct {
	mu           sync.Mutex
	transferred  []string
	tried        []string
	pressed      []string
	transferErr  error
	sendDigitErr error
}

func (t *stubTelephony) Transfer(_ context.Context, to string) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.tried = append(t.tried, to)
	if t.transferErr != nil {
		return t.transferErr
	}
	t.transferred = append(t.transferred, to)
	return nil
}

func (t *stubTelephony) SendDigits(_ context.Context, digits string) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.sendDigitErr != nil {
		return t.sendDigitErr
	}
	t.pressed = append(t.pressed, digits)
	return nil
}

func (t *stubTelephony) handedOver() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.transferred...)
}

// attempted is every transfer asked for, including the ones that failed.
func (t *stubTelephony) attempted() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.tried...)
}

func (t *stubTelephony) keypad() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.pressed...)
}

// onACall gives the agent a phone line and the tools that act on it.
func (s *AgentSuite) onACall() {
	s.line = &stubTelephony{}
	tools, err := harness.DefaultTools()
	s.Require().NoError(err)
	s.tools = tools
}

// asksFor makes the next reply call a tool, alongside whatever it says.
func (s *AgentSuite) asksFor(name, arguments string) {
	s.model.calls = []llm.ToolCall{{ID: "call-1", Name: name, Arguments: arguments}}
}

// transferredIn returns the transfers reported to the caller of Events.
func transferredIn(events []Event) []Transferred {
	var handed []Transferred
	for _, event := range events {
		if typed, ok := event.(Transferred); ok {
			handed = append(handed, typed)
		}
	}
	return handed
}

func toolsRanIn(events []Event) []ToolRan {
	var ran []ToolRan
	for _, event := range events {
		if typed, ok := event.(ToolRan); ok {
			ran = append(ran, typed)
		}
	}
	return ran
}

func (s *AgentSuite) TestToolsAreOnlyOfferedWhenThereIsACallToActOn() {
	// A model told it may transfer, in a call with nowhere to transfer to, promises the
	// caller a person who never arrives.
	tools, err := harness.DefaultTools()
	s.Require().NoError(err)
	s.tools = tools
	s.join(false)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "put me through to someone")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	s.Nil(s.model.requests()[0].Tools, "without telephony there is nothing to offer")
}

func (s *AgentSuite) TestTheModelIsOfferedTheToolsItCanRun() {
	s.onACall()
	s.join(false)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	offered := s.model.requests()[0].Tools
	s.Require().NotEmpty(offered)

	names := make([]string, 0, len(offered))
	for _, tool := range offered {
		names = append(names, tool.Name)
	}
	s.Contains(names, "transfer")
	s.Contains(names, "press")
}

func (s *AgentSuite) TestAColdTransferBringsTheHumanOnAndTheAgentLeaves() {
	s.onACall()
	s.join(false)
	s.model.reply = []string{"Putting you through now."}
	s.asksFor("transfer", `{"to":"+15550001111"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "I want to speak to a person")

	s.eventually(func() bool { return len(s.line.handedOver()) == 1 }, "nobody was dialled")
	s.Equal("+15550001111", s.line.handedOver()[0])

	s.eventually(func() bool { return len(transferredIn(s.reported())) == 1 },
		"the handover was never reported")
	s.Empty(transferredIn(s.reported())[0].Summary, "a cold transfer introduces nobody")

	s.eventually(s.left, "the agent stayed on a call it had handed over")
}

func (s *AgentSuite) TestAWarmTransferIntroducesTheCallerOnceTheHumanIsOn() {
	// The summary is spoken on the call rather than privately, so it only means anything
	// once there is somebody new to hear it.
	s.onACall()
	s.join(false)
	s.model.reply = []string{"One moment."}
	s.asksFor("transfer", `{"to":"+15550001111","summary":"Alice needs a refund on order 12."}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "I want to speak to a person")
	s.eventually(func() bool { return len(s.line.handedOver()) == 1 }, "nobody was dialled")

	s.never(func() bool { return s.spokenText("Alice needs a refund") },
		"the summary was said to an empty seat")

	// The human answering is the agent hearing a voice it has not heard before.
	s.speak(stt.Participant{ID: "human"})

	s.eventually(func() bool { return s.spokenText("Alice needs a refund on order 12.") },
		"the human was never introduced to the caller")
	s.eventually(s.left, "the agent stayed on a call it had handed over")
}

func (s *AgentSuite) TestPressingAMenuOptionDoesNotEndTheCall() {
	s.onACall()
	s.join(false)
	s.model.reply = nil
	s.asksFor("press", `{"digits":"1"}`)
	menu := stt.Participant{ID: "menu"}
	s.speak(menu)

	s.says(menu, "For sales, press one")

	s.eventually(func() bool { return len(s.line.keypad()) == 1 }, "nothing was pressed")
	s.Equal("1", s.line.keypad()[0])

	s.Empty(transferredIn(s.reported()), "pressing a menu option hands nobody over")
	s.never(s.left, "the agent left a call it had only pressed a button on")
}

func (s *AgentSuite) TestATransferThatFailsIsToldToTheModelRatherThanEndingTheCall() {
	// A caller promised a person, on a transfer that did not happen, is owed an apology
	// rather than a silent call.
	s.onACall()
	s.line.transferErr = errors.New("the trunk is down")
	s.join(false)
	s.model.reply = []string{"Putting you through."}
	s.model.then = []string{"Sorry, I could not put you through."}
	s.asksFor("transfer", `{"to":"+15550001111"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "I want a person")

	ran := s.awaitToolRan()
	s.Require().Error(ran.Err)
	s.Contains(ran.Result, "did not work", "the model has to be told so it can say so")

	s.never(s.left, "the agent left a call it never transferred")
	s.Contains(s.history(), llm.Message{
		Role:       llm.ToolResult,
		Content:    ran.Result,
		ToolCallID: "call-1",
	})
	s.eventually(func() bool {
		return s.spokenText("could not put you through")
	}, "the caller was left in silence by a transfer that never happened")
}

func (s *AgentSuite) TestAToolThatKeepsFailingIsNotAnsweredForever() {
	// The apology for a failed tool is a turn like any other, so a model that answers it
	// by trying the tool again would have the agent talking to itself until the money
	// ran out.
	s.onACall()
	s.line.transferErr = errors.New("the trunk is down")
	s.join(false)
	s.model.reply = []string{"Putting you through."}
	s.model.keepCalling = true
	s.asksFor("transfer", `{"to":"+15550001111"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "I want a person")

	// The apology is allowed to reach for the tool once more, in case the model has
	// thought better of the number it used. What it cannot do is keep going.
	s.eventually(func() bool { return len(s.line.attempted()) == 2 },
		"the model never answered the failure at all")
	s.never(func() bool {
		return len(s.line.attempted()) > 2
	}, "a failing transfer was retried without the caller saying anything")
}

func (s *AgentSuite) TestATurnThatCalledAToolIsRememberedWithTheCallOnIt() {
	// The provider matches the result against the call it answers, so the turn cannot be
	// recorded as plain speech.
	s.onACall()
	s.join(false)
	s.model.reply = []string{"One moment."}
	s.asksFor("press", `{"digits":"4"}`)
	menu := stt.Participant{ID: "menu"}
	s.speak(menu)

	s.says(menu, "For accounts, press four")

	s.eventually(func() bool { return len(s.line.keypad()) == 1 }, "nothing was pressed")
	s.eventually(func() bool {
		for _, message := range s.history() {
			if message.Role == llm.ToolResult && message.ToolCallID == "call-1" {
				return true
			}
		}
		return false
	}, "the result never reached the conversation")

	history := s.history()
	var assistant llm.Message
	for _, message := range history {
		if message.Role == llm.Assistant {
			assistant = message
		}
	}
	s.Require().Len(assistant.ToolCalls, 1)
	s.Equal("press", assistant.ToolCalls[0].Name)
}

func (s *AgentSuite) TestAToolTheAgentCannotRunIsRefusedRatherThanIgnored() {
	// The harness drops names it never offered, so what reaches here is a tool that was
	// offered and is not implemented, which the model still has to be told about.
	s.onACall()
	s.join(false)

	s.agent.runTool(harness.ToolRequested{
		TurnID: "turn-1",
		Call:   llm.ToolCall{ID: "call-9", Name: "hang_up", Arguments: "{}"},
	})

	ran := s.awaitToolRan()
	s.Require().Error(ran.Err)
	s.Contains(ran.Result, "did not work")
}

func (s *AgentSuite) TestTransferringWithoutANumberIsRefused() {
	s.onACall()
	s.join(false)

	s.agent.runTool(harness.ToolRequested{
		TurnID: "turn-1",
		Call:   llm.ToolCall{ID: "call-9", Name: "transfer", Arguments: `{"summary":"they want a person"}`},
	})

	s.ErrorContains(s.awaitToolRan().Err, "number to transfer to")
	s.Empty(s.line.handedOver(), "there was nobody to dial")
}

func (s *AgentSuite) TestPressingSomethingUnreadableIsRefused() {
	s.onACall()
	s.join(false)

	s.agent.runTool(harness.ToolRequested{
		TurnID: "turn-1",
		Call:   llm.ToolCall{ID: "call-9", Name: "press", Arguments: "press one please"},
	})

	s.ErrorContains(s.awaitToolRan().Err, "could not read")
	s.Empty(s.line.keypad())
}

// awaitToolRan waits for one tool to have settled and returns what it reported, since the
// collector sees the event on its own goroutine.
func (s *AgentSuite) awaitToolRan() ToolRan {
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 1 },
		"the tool never settled")
	return toolsRanIn(s.reported())[0]
}

// left reports whether the agent is out of the call, which is what it says on the way out.
func (s *AgentSuite) left() bool { return countOf[Left](s.reported()) > 0 }

// never asserts that something stays untrue for long enough to believe it.
func (s *AgentSuite) never(condition func() bool, message string) {
	s.Require().Never(condition, 500*time.Millisecond, 5*time.Millisecond, message)
}

// spokenText reports whether the voice was asked to say something containing the text.
func (s *AgentSuite) spokenText(text string) bool {
	for _, request := range s.voice.spoken() {
		if strings.Contains(request.Text, text) {
			return true
		}
	}
	return false
}

// history is the conversation the agent would send on the next turn.
func (s *AgentSuite) history() []llm.Message {
	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	return append([]llm.Message(nil), s.agent.history...)
}
