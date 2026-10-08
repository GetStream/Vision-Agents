package agent

import (
	"context"
	"errors"
	"fmt"
	"slices"
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

// stubToolRunner stands in for whoever owns the tools this package does not, which in
// production is a caller on the other end of a socket.
type stubToolRunner struct {
	result  string
	err     error
	entered chan struct{}
	release <-chan struct{}

	mu   sync.Mutex
	runs []llm.ToolCall
}

func (r *stubToolRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	r.mu.Lock()
	r.runs = append(r.runs, call)
	entered, release := r.entered, r.release
	r.entered, r.release = nil, nil
	result, err := r.result, r.err
	r.mu.Unlock()
	if entered != nil {
		close(entered)
	}
	if release != nil {
		select {
		case <-release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if err != nil {
		return nil, err
	}
	return llm.TextParts(result), nil
}

func (r *stubToolRunner) asked() []llm.ToolCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]llm.ToolCall(nil), r.runs...)
}

// ownsTools gives the agent a caller with tools of its own, the way a remote session does.
func (s *AgentSuite) ownsTools(result string) {
	s.runner = &stubToolRunner{result: result}
	s.tools = harness.Tools{Tools: []harness.Tool{{
		Name:        "lookup_order",
		Description: "find an order by its number",
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{"order": map[string]any{"type": "string"}},
		},
	}}}
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

func (s *AgentSuite) TestACallersOwnToolsAreOfferedWithoutTelephony() {
	// The gate is per tool rather than per call: a transfer needs a phone line, and a
	// caller's own function needs only somebody to run it.
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the model was never asked")
	offered := s.model.requests()[0].Tools
	s.Require().Len(offered, 1)
	s.Equal("lookup_order", offered[0].Name)
}

func (s *AgentSuite) TestACallersOwnToolIsRunByThemAndItsAnswerReachesTheConversation() {
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Let me check."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return len(s.runner.asked()) == 1 }, "the caller was never asked")
	s.Equal(`{"order":"12"}`, s.runner.asked()[0].Arguments)

	ran := s.awaitToolRan()
	s.Require().NoError(ran.Err)
	s.Equal("order 12 ships tomorrow", ran.Result)
	s.Contains(s.history(), llm.Message{
		Role:       llm.ToolResult,
		Content:    "order 12 ships tomorrow",
		ToolCallID: "call-1",
	})
	s.never(s.left, "answering a question is not a reason to hang up")
}

func (s *AgentSuite) TestWhatAToolFoundOutIsSpokenRatherThanWaitedOn() {
	// A tool result is not something the caller can hear. Somebody asked a question, and
	// leaving the answer sitting in the history until they speak again is silence where
	// they expected to be told.
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Let me check."}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool {
		return s.spokenText("ships tomorrow")
	}, "the caller was left in silence by a tool that worked")
}

func (s *AgentSuite) TestTextToolFollowUpStillProducesAFinalAnswer() {
	s.ownsTools("The source returns the ChatContext value.")
	s.joinText()
	s.model.reply = []string{"It returns the ChatContext value."}
	call := llm.ToolCall{ID: "research-after-docs", Name: "lookup_order", Arguments: `{}`}
	s.agent.mu.Lock()
	s.agent.history = []llm.Message{
		{Role: llm.User, Content: "What does useChatContext return?"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{call}},
	}
	s.agent.mu.Unlock()
	// A tool requested while answering an earlier tool uses the tool turn prefix.
	s.agent.runTool(harness.ToolRequested{TurnID: toolPrefix + "after-docs", Call: call})
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "source research left the text session silent")
	response, _ := firstOf[Responded](s.reported())
	s.Equal("It returns the ChatContext value.", response.Text)
	s.False(response.PendingWork)
}

func (s *AgentSuite) TestAToolReachedForWithoutAWordStillTellsTheCallerToWait() {
	// The fast models do not reliably say anything before they call a tool, and the
	// caller cannot hear one running. Without this they ask a question and get silence,
	// which on a phone is indistinguishable from having been cut off.
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool {
		return s.spokenText("moment") || s.spokenText("check") ||
			s.spokenText("second") || s.spokenText("Bear with me")
	}, "the caller was left in silence while the tool ran")
	s.eventually(func() bool {
		return s.spokenText("ships tomorrow")
	}, "the answer never followed the promise to check")
}

func (s *AgentSuite) TestATurnThatSpokeForItselfIsNotGivenAFiller() {
	// A model that already said what it was doing does not need it said again, and
	// "Let me check. One moment." is a worse answer than either half.
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Let me check."}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return s.spokenText("ships tomorrow") }, "the tool answer never came")
	s.False(s.spokenText("One moment"), "the agent stacked a filler on top of its own words")
}

func (s *AgentSuite) TestAToolsPreSpeechIsSaidInsteadOfTheAgentsOwnFiller() {
	// A connector binding's policy names what to say while its tool runs; the agent says
	// it where it would have picked a phrase itself, and tool_started carries it.
	s.ownsTools("order 12 ships tomorrow")
	s.toolPolicy = func(tool string) ToolPolicy {
		if tool == "lookup_order" {
			return ToolPolicy{PreSpeech: "Let me pull up your order."}
		}
		return ToolPolicy{}
	}
	s.join(false)
	s.model.reply = []string{}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return s.spokenText("Let me pull up your order.") },
		"the binding's phrase was not said while its tool ran")
	s.eventually(func() bool { return s.spokenText("ships tomorrow") }, "the answer never followed")
	for _, phrase := range workingPhrases {
		s.False(s.spokenText(phrase), "the agent said its own filler as well: %q", phrase)
	}
	s.eventually(func() bool { return countOf[ToolStarted](s.reported()) == 1 }, "the tool never started")
	started, _ := firstOf[ToolStarted](s.reported())
	s.Equal("Let me pull up your order.", started.PreSpeech)
}

func (s *AgentSuite) TestAToolWithoutPreSpeechGetsTheAgentsOwnFillerAsBefore() {
	// A tool its binding names no phrase for, or a session with no bindings at all, is
	// TestAToolReachedForWithoutAWordStillTellsTheCallerToWait.
	s.ownsTools("order 12 ships tomorrow")
	s.toolPolicy = func(string) ToolPolicy { return ToolPolicy{} }
	s.join(false)
	s.model.reply = []string{}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return s.spokenText(workingPhrases[0]) }, "the agent's own filler was not said")
	s.eventually(func() bool { return countOf[ToolStarted](s.reported()) == 1 }, "the tool never started")
	started, _ := firstOf[ToolStarted](s.reported())
	s.Empty(started.PreSpeech)
}

func (s *AgentSuite) TestATurnThatSpokeForItselfIsNotGivenPreSpeechEither() {
	// pre_speech takes the filler's place, so it follows the filler's rule: a model that
	// already said what it was doing is not given more words on top.
	s.ownsTools("order 12 ships tomorrow")
	s.toolPolicy = func(string) ToolPolicy { return ToolPolicy{PreSpeech: "Let me pull up your order."} }
	s.join(false)
	s.model.reply = []string{"Let me check."}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return s.spokenText("ships tomorrow") }, "the tool answer never came")
	s.False(s.spokenText("Let me pull up your order."), "the agent stacked pre_speech on top of its own words")
}

func (s *AgentSuite) TestTheHoldPhraseOpeningTheSentenceBeforeACallIsTheOnlyOneInTheWait() {
	// The sentence the use policy asks for before a call opens with the hold phrase and reads
	// the details back in one go. That phrase is the wait's own: the agent adds neither its
	// own filler nor the binding's pre_speech, which would come after the read-back, and the
	// answer to the result adds none, so the caller hears one hold phrase, and it starts the
	// wait rather than ending it.
	s.ownsTools("order 12 ships tomorrow")
	s.toolPolicy = func(string) ToolPolicy { return ToolPolicy{PreSpeech: "Let me pull up your order."} }
	s.join(false)
	s.model.reply = []string{"One moment, I have order twelve for Alvarez."}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return s.spokenText("ships tomorrow") }, "the tool answer never came")
	spoken := s.voice.spoken()
	s.Require().NotEmpty(spoken)
	s.Equal("One moment, I have order twelve for Alvarez.", spoken[0].Text,
		"the hold phrase and the read-back are one utterance, the first the caller hears")
	holds, texts := 0, make([]string, 0, len(spoken))
	for _, request := range spoken {
		texts = append(texts, request.Text)
		for _, phrase := range append([]string{"One moment", "Let me pull up your order."}, workingPhrases...) {
			if strings.Contains(request.Text, phrase) {
				holds++
				break
			}
		}
	}
	s.Equal(1, holds, "one hold phrase to a wait: %q", texts)
}

func (s *AgentSuite) TestPreSpeechIsTheFirstCallsThatNamesOne() {
	// Two calls in one reply: the first names no phrase, the second does, so the second's
	// is said rather than the agent's own.
	s.ownsTools("done")
	s.tools.Tools = append(s.tools.Tools, harness.Tool{Name: "track_parcel", Description: "track a parcel"})
	s.toolPolicy = func(tool string) ToolPolicy {
		if tool == "track_parcel" {
			return ToolPolicy{PreSpeech: "Let me track your parcel."}
		}
		return ToolPolicy{}
	}
	s.join(false)
	s.model.reply = []string{}
	s.model.then = []string{"It ships tomorrow."}
	s.model.calls = []llm.ToolCall{
		{ID: "call-1", Name: "lookup_order", Arguments: `{"order":"12"}`},
		{ID: "call-2", Name: "track_parcel", Arguments: `{}`},
	}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return s.spokenText("Let me track your parcel.") },
		"the second call's phrase was not said")
	s.False(s.spokenText(workingPhrases[0]), "the agent said its own filler instead")
}

// holdingTool answers only when released, whatever its context says, the way the
// dispatcher runs a call of an on_interrupt: wait binding.
type holdingTool struct {
	began   chan struct{}
	release chan struct{}
}

func (r *holdingTool) Run(_ context.Context, _ llm.ToolCall) ([]llm.ContentPart, error) {
	close(r.began)
	<-r.release
	return llm.TextParts("order 12 ships tomorrow"), nil
}

// unanswered names the first tool call in messages whose results do not all come right
// after it, or a result that answers no call right before it: what a provider refuses.
// Empty when there is none.
func unanswered(messages []llm.Message) string {
	for i := 0; i < len(messages); i++ {
		if messages[i].Role == llm.ToolResult {
			return "a result answering no call before it: " + messages[i].ToolCallID
		}
		calls := messages[i].ToolCalls
		if messages[i].Role != llm.Assistant || len(calls) == 0 {
			continue
		}
		var asked, answered []string
		for _, call := range calls {
			asked = append(asked, call.ID)
		}
		for i+1 < len(messages) && messages[i+1].Role == llm.ToolResult {
			i++
			answered = append(answered, messages[i].ToolCallID)
		}
		slices.Sort(asked)
		slices.Sort(answered)
		if !slices.Equal(asked, answered) {
			return fmt.Sprintf("calls %v answered by %v", asked, answered)
		}
	}
	return ""
}

func (s *AgentSuite) TestAnInterruptedWaitToolLeavesItsCallAnsweredForTheNextTurn() {
	// The call goes on after the interruption, so it is answered at once with a note that
	// its result will follow; the next turn is not held, and the result comes as a message
	// of its own once it is in.
	s.ownsTools("")
	s.toolPolicy = func(string) ToolPolicy { return ToolPolicy{Waits: true} }
	s.join(true)
	runner := &holdingTool{began: make(chan struct{}), release: make(chan struct{})}
	release := sync.OnceFunc(func() { close(runner.release) })
	defer release()
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{"Let me check the order."}
	s.model.then = []string{"Fifteen."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "where is my order")
	select {
	case <-runner.began:
	case <-time.After(3 * time.Second):
		s.FailNow("tool did not start")
	}
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}
	s.mutters(participant, "stop cancel that lookup")
	s.eventually(func() bool {
		return slices.ContainsFunc(s.history(), func(m llm.Message) bool { return m.Content == stillRunning })
	},
		"the interrupted call was not answered")
	s.flow.then = nil
	s.says(participant, "what is seven plus eight")
	s.eventually(func() bool { return s.spokenText("Fifteen") }, "the next turn waited on the call")

	s.model.mu.Lock()
	next := s.model.asked[len(s.model.asked)-1].Input
	s.model.mu.Unlock()
	s.Empty(unanswered(next), "the next turn replayed a call without its result")
	s.Equal(llm.User, next[len(next)-1].Role)

	release()
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 1 }, "the call never answered")
	s.eventually(func() bool {
		return strings.Contains(s.history()[len(s.history())-1].Content, "order 12 ships tomorrow")
	},
		"the late result never reached the conversation")
	history := s.history()
	late := history[len(history)-1]
	s.Empty(unanswered(history), "the late result broke the conversation")
	s.Equal(llm.User, late.Role)
	s.Equal(fmt.Sprintf(lateResult, "lookup_order", "call-1")+"\norder 12 ships tomorrow", late.Content)
}

func (s *AgentSuite) TestAWaitToolNobodyInterruptedIsAnsweredOnceAsBefore() {
	s.ownsTools("order 12 ships tomorrow")
	s.toolPolicy = func(string) ToolPolicy { return ToolPolicy{Waits: true} }
	s.join(false)
	s.model.reply = []string{"Let me check."}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	s.eventually(func() bool { return s.spokenText("ships tomorrow") }, "the tool answer never came")
	results := slices.DeleteFunc(s.history(), func(m llm.Message) bool { return m.Role != llm.ToolResult })
	s.Equal([]llm.Message{{Role: llm.ToolResult, ToolCallID: "call-1", Content: "order 12 ships tomorrow"}}, results)
}

// heldCalls answers each call only when its own release channel is closed, whatever its
// context says, the way the dispatcher runs a call of an on_interrupt: wait binding.
type heldCalls struct {
	began   map[string]chan struct{}
	release map[string]chan struct{}
}

func newHeldCalls(ids ...string) *heldCalls {
	r := &heldCalls{began: map[string]chan struct{}{}, release: map[string]chan struct{}{}}
	for _, id := range ids {
		r.began[id] = make(chan struct{})
		r.release[id] = make(chan struct{})
	}
	return r
}

func (r *heldCalls) Run(_ context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	close(r.began[call.ID])
	<-r.release[call.ID]
	return llm.TextParts(call.ID + " result"), nil
}

// interruptsWaitCallsThenCalls starts a turn asking for first, interrupts it once they all
// run, then starts the next turn, which asks for next, and returns once next runs.
func (s *AgentSuite) interruptsWaitCallsThenCalls(runner *heldCalls, first []llm.ToolCall, next llm.ToolCall) {
	s.ownsTools("")
	s.toolPolicy = func(string) ToolPolicy { return ToolPolicy{Waits: true} }
	s.join(true)
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{"Let me check the order."}
	s.model.then = []string{"Checking."}
	s.model.calls = first
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "where is my order")
	for _, call := range first {
		s.began(runner, call.ID)
	}
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}
	s.mutters(participant, "stop cancel that lookup")
	s.eventually(func() bool {
		answered := 0
		for _, m := range s.history() {
			if m.Content == stillRunning {
				answered++
			}
		}
		return answered == len(first)
	}, "the interrupted calls were not answered")
	s.flow.then = nil
	s.model.mu.Lock()
	s.model.calls = []llm.ToolCall{next}
	s.model.keepCalling = true
	s.model.mu.Unlock()
	s.says(participant, "and my other order")
	s.began(runner, next.ID)
	s.model.mu.Lock()
	s.model.keepCalling = false
	s.model.mu.Unlock()
}

// began waits for the call id to start running.
func (s *AgentSuite) began(runner *heldCalls, id string) {
	select {
	case <-runner.began[id]:
	case <-time.After(3 * time.Second):
		s.FailNow("tool did not start: " + id)
	}
}

// lateResultsIn is the content of every lateResult message in messages, in order.
func lateResultsIn(messages []llm.Message) []string {
	var late []string
	for _, m := range messages {
		if m.Role == llm.User && strings.HasPrefix(m.Content, "The lookup_order call (") {
			late = append(late, m.Content)
		}
	}
	return late
}

func (s *AgentSuite) TestALateResultWaitsForTheCallsRunningWhenItComes() {
	// The late result is the caller's message, so it must not come between the next
	// turn's call and that call's result: a provider refuses the call then.
	runner := newHeldCalls("call-1", "call-2")
	s.interruptsWaitCallsThenCalls(runner,
		[]llm.ToolCall{{ID: "call-1", Name: "lookup_order", Arguments: `{"order":"12"}`}},
		llm.ToolCall{ID: "call-2", Name: "lookup_order", Arguments: `{"order":"13"}`})

	close(runner.release["call-1"])
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 1 }, "the first call never answered")
	s.Empty(lateResultsIn(s.history()), "the late result came while the next call was still running")

	close(runner.release["call-2"])
	s.eventually(func() bool {
		s.model.mu.Lock()
		defer s.model.mu.Unlock()
		return len(s.model.asked) == 3
	}, "the second call's result was never sent")

	s.model.mu.Lock()
	asked := append([]llm.ResponseParams(nil), s.model.asked...)
	s.model.mu.Unlock()
	for i, request := range asked {
		s.Empty(unanswered(request.Input), "request %d replayed a call without its result", i)
	}
	last := asked[2].Input
	s.Equal(llm.ToolResult, last[len(last)-2].Role)
	s.Equal("call-2", last[len(last)-2].ToolCallID)
	s.Equal(llm.Message{Role: llm.User, Content: fmt.Sprintf(lateResult, "lookup_order", "call-1") + "\ncall-1 result"},
		last[len(last)-1])
}

func (s *AgentSuite) TestALateResultIsKeptWhenTheNextCallIsInterruptedToo() {
	// The next turn's call is interrupted and answered, which closes the open call: the
	// late result that already came must go into the history then, not wait for a result
	// that is already there.
	runner := newHeldCalls("call-1", "call-2")
	s.interruptsWaitCallsThenCalls(runner,
		[]llm.ToolCall{{ID: "call-1", Name: "lookup_order", Arguments: `{"order":"12"}`}},
		llm.ToolCall{ID: "call-2", Name: "lookup_order", Arguments: `{"order":"13"}`})
	close(runner.release["call-1"])
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 1 }, "the first call never answered")

	participant := stt.Participant{ID: "alice"}
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}
	s.mutters(participant, "stop that too")
	s.eventually(func() bool {
		answered := 0
		for _, m := range s.history() {
			if m.Content == stillRunning {
				answered++
			}
		}
		return answered == 2
	}, "the second call was not answered")
	s.flow.then = nil
	s.model.mu.Lock()
	s.model.then = []string{"Fifteen."}
	s.model.mu.Unlock()
	s.says(participant, "what is seven plus eight")
	s.eventually(func() bool { return s.spokenText("Fifteen") }, "the third turn never spoke")

	s.model.mu.Lock()
	last := s.model.asked[len(s.model.asked)-1].Input
	s.model.mu.Unlock()
	s.Equal([]string{fmt.Sprintf(lateResult, "lookup_order", "call-1") + "\ncall-1 result"}, lateResultsIn(last))

	close(runner.release["call-2"])
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 2 }, "the second call never answered")
}

func (s *AgentSuite) TestALateResultWaitsForEveryCallOfTheNextTurn() {
	// The next turn asks for two calls and one is answered: the history still ends in an
	// open call, so the late result waits for the other.
	s.ownsTools("")
	s.toolPolicy = func(string) ToolPolicy { return ToolPolicy{Waits: true} }
	s.join(true)
	runner := newHeldCalls("call-1", "call-2", "call-3")
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{"Let me check the order."}
	s.model.then = []string{"Checking."}
	s.model.calls = []llm.ToolCall{{ID: "call-1", Name: "lookup_order", Arguments: `{"order":"12"}`}}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "where is my order")
	s.began(runner, "call-1")
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}
	s.mutters(participant, "stop cancel that lookup")
	s.eventually(func() bool {
		for _, m := range s.history() {
			if m.Content == stillRunning {
				return true
			}
		}
		return false
	}, "the call was not answered")
	s.flow.then = nil
	s.model.mu.Lock()
	s.model.calls = []llm.ToolCall{
		{ID: "call-2", Name: "lookup_order", Arguments: `{"order":"13"}`},
		{ID: "call-3", Name: "lookup_order", Arguments: `{"order":"14"}`},
	}
	s.model.keepCalling = true
	s.model.mu.Unlock()
	s.says(participant, "and my other two orders")
	s.began(runner, "call-2")
	s.began(runner, "call-3")
	s.model.mu.Lock()
	s.model.keepCalling = false
	s.model.mu.Unlock()

	close(runner.release["call-2"])
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 1 }, "call-2 never answered")
	close(runner.release["call-1"])
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 2 }, "call-1 never answered")
	s.Empty(lateResultsIn(s.history()), "the late result came while call-3 was still running")

	close(runner.release["call-3"])
	s.eventually(func() bool { return len(lateResultsIn(s.history())) == 1 }, "the late result was never added")
	s.Empty(unanswered(s.history()))
}

func (s *AgentSuite) TestLateResultsHeldBackComeInTheOrderTheyCame() {
	runner := newHeldCalls("call-1", "call-2", "call-3", "call-4")
	s.interruptsWaitCallsThenCalls(runner,
		[]llm.ToolCall{
			{ID: "call-1", Name: "lookup_order", Arguments: `{"order":"12"}`},
			{ID: "call-2", Name: "lookup_order", Arguments: `{"order":"13"}`},
			{ID: "call-3", Name: "lookup_order", Arguments: `{"order":"14"}`},
		},
		llm.ToolCall{ID: "call-4", Name: "lookup_order", Arguments: `{"order":"15"}`})

	close(runner.release["call-2"])
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 1 }, "call-2 never answered")
	close(runner.release["call-1"])
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 2 }, "call-1 never answered")
	close(runner.release["call-4"])
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 3 }, "call-4 never answered")
	close(runner.release["call-3"])
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 4 }, "call-3 never answered")

	late := func(id string) string { return fmt.Sprintf(lateResult, "lookup_order", id) + "\n" + id + " result" }
	s.Equal([]string{late("call-2"), late("call-1"), late("call-3")}, lateResultsIn(s.history()))
	s.Empty(unanswered(s.history()))
}

func (s *AgentSuite) TestPressingAMenuOptionLeavesTheLineQuietForTheMenu() {
	// The digits are the whole point of the tool and the menu is what answers next, so
	// talking over it would be talking to nobody.
	s.onACall()
	s.join(false)
	s.model.reply = []string{"One moment."}
	s.model.then = []string{"I pressed four for you."}
	s.asksFor("press", `{"digits":"4"}`)
	menu := stt.Participant{ID: "menu"}
	s.speak(menu)

	s.says(menu, "For accounts, press four")

	s.eventually(func() bool { return len(s.line.keypad()) == 1 }, "nothing was pressed")
	s.never(func() bool { return s.spokenText("pressed four") },
		"the agent talked over the menu it had just pressed at")
}

func (s *AgentSuite) TestPressingAMenuOptionSilentlyIsNotGivenAFillerEither() {
	// Filling the pause is for a caller who asked a question. A menu did not ask one and
	// is not listening, so a wordless press has to stay wordless.
	s.onACall()
	s.join(false)
	s.model.reply = []string{}
	s.asksFor("press", `{"digits":"4"}`)
	menu := stt.Participant{ID: "menu"}
	s.speak(menu)

	s.says(menu, "For accounts, press four")

	s.eventually(func() bool { return len(s.line.keypad()) == 1 }, "nothing was pressed")
	s.never(func() bool { return s.spokenText("moment") || s.spokenText("check") },
		"the agent talked over the menu it had just pressed at")
}

func (s *AgentSuite) TestATelephonyToolIsNotHandedToTheCallersRunner() {
	// The two that act on the call are this process's to run, so a caller cannot quietly
	// take over what happens when the model says it is transferring somebody.
	s.onACall()
	s.runner = &stubToolRunner{result: "not mine to answer"}
	s.join(false)
	s.model.reply = []string{"Putting you through."}
	s.asksFor("transfer", `{"to":"+15550001111"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "I want a person")

	s.eventually(func() bool { return len(s.line.handedOver()) == 1 }, "nobody was dialled")
	s.Empty(s.runner.asked(), "the caller was asked to run a transfer")
}

func (s *AgentSuite) TestACallersToolThatFailsIsToldToTheModel() {
	s.ownsTools("")
	s.runner.err = errors.New("the orders service is down")
	s.join(false)
	s.model.reply = []string{"Let me check."}
	s.model.then = []string{"Sorry, I cannot look that up right now."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "where is my order")

	ran := s.awaitToolRan()
	s.Require().Error(ran.Err)
	s.Contains(ran.Result, "did not work")
	s.eventually(func() bool {
		return s.spokenText("cannot look that up")
	}, "the caller was left in silence by a tool that failed")
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
	s.T().Helper()
	s.neverWithin(condition, 500*time.Millisecond, message)
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

type cancellationTool struct {
	began   chan struct{}
	stopped chan struct{}
}

func (r *cancellationTool) Run(ctx context.Context, _ llm.ToolCall) ([]llm.ContentPart, error) {
	close(r.began)
	<-ctx.Done()
	close(r.stopped)
	return nil, ctx.Err()
}
func (s *AgentSuite) TestInterruptCancelsActiveTextToolWithoutFollowingUp() {
	s.ownsTools("")
	runner := &cancellationTool{began: make(chan struct{}), stopped: make(chan struct{})}
	s.joinText()
	s.agent.options.ToolRunner = runner
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.agent.runTool(harness.ToolRequested{TurnID: "active-research", Call: llm.ToolCall{ID: "research", Name: "lookup_order", Arguments: `{}`}})
	}()
	select {
	case <-runner.began:
	case <-time.After(time.Second):
		s.FailNow("tool did not start")
	}
	s.agent.Interrupt()
	select {
	case <-runner.stopped:
	case <-time.After(time.Second):
		s.FailNow("tool context was not cancelled")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		s.FailNow("tool did not finish")
	}
	s.Zero(countOf[Responded](s.reported()), "cancelled research must not produce an unsolicited follow-up")
}

func (s *AgentSuite) TestSpokenInterruptionCancelsPendingToolAndAnswersNextTurn() {
	s.ownsTools("")
	s.join(true)
	runner := &cancellationTool{began: make(chan struct{}), stopped: make(chan struct{})}
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{"Let me check the order."}
	s.model.then = []string{"Fifteen."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "where is my order")
	select {
	case <-runner.began:
	case <-time.After(3 * time.Second):
		s.FailNow("tool did not start")
	}
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}
	s.mutters(participant, "stop cancel that lookup")
	select {
	case <-runner.stopped:
	case <-time.After(3 * time.Second):
		s.FailNow("spoken interruption did not cancel the pending tool")
	}
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == 1 }, "cancelled tool did not settle")
	s.Require().Error(toolsRanIn(s.reported())[0].Err)
	s.flow.then = nil
	s.says(participant, "what is seven plus eight")
	s.eventually(func() bool { return s.spokenText("Fifteen") }, "next spoken turn was not answered")
}
