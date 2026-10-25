package agent

import (
	"context"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

func (s *AgentSuite) TestACallThatNeedsApprovalAsksTheQuestionAsWrittenAndDoesNotRun() {
	s.askedToApprove()

	s.never(func() bool { return len(s.runner.asked()) > 0 }, "the call ran before the caller said yes")
	asked := s.historyWith(llm.ToolResult)
	s.Contains(asked, "Book order 12?")
	s.Contains(asked, confirmTool)
	s.Len(s.model.requests(), 1, "the question is said as written, without a turn of the model's")
}

func (s *AgentSuite) TestAYesRunsTheCallOnceWithItsOwnID() {
	s.askedToApprove()
	var approved atomic.Bool
	s.runner.run = func(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
		approved.Store(Approved(ctx))
		return llm.TextParts("order 12 booked"), nil
	}
	s.confirms(`{"tool_call_id":"call-1","allowed":true}`)

	s.says(stt.Participant{ID: "alice"}, "yes please")

	s.eventually(func() bool { return len(s.runner.asked()) == 1 }, "the call never ran")
	ran := s.runner.asked()[0]
	s.Equal("call-1", ran.ID, "the call runs under the id the model gave it")
	s.Equal("lookup_order", ran.Name)
	s.Equal(`{"order":"12"}`, ran.Arguments)
	// Read once the result is back: the stub records a call before it runs it.
	s.eventually(func() bool { return strings.Contains(s.historyWith(llm.ToolResult), "order 12 booked") },
		"the call's result never answered the confirmation")
	s.True(approved.Load(), "the call is marked approved for whoever runs it")
	s.eventually(func() bool { return countOf[ToolApprovalDecided](s.reported()) == 1 }, "the answer was never reported")
	decided, _ := firstOf[ToolApprovalDecided](s.reported())
	s.Equal(ToolApprovalDecided{ID: "call-1", TurnID: decided.TurnID, Allowed: true}, decided)
}

func (s *AgentSuite) TestANoNeverRunsTheCall() {
	s.askedToApprove()
	s.confirms(`{"tool_call_id":"call-1","allowed":false}`)

	s.says(stt.Participant{ID: "alice"}, "no, leave it")

	s.eventually(func() bool { return strings.Contains(s.historyWith(llm.ToolResult), "declined") },
		"the model was never told the caller declined")
	s.never(func() bool { return len(s.runner.asked()) > 0 }, "a declined call ran")
	s.eventually(func() bool { return countOf[ToolApprovalDecided](s.reported()) == 1 }, "the answer was never reported")
	decided, _ := firstOf[ToolApprovalDecided](s.reported())
	s.False(decided.Allowed)
}

func (s *AgentSuite) TestSilenceDeclinesOnceTheWaitHasPassedSinceTheQuestionWasSaid() {
	s.agentAsksAndWaits(300 * time.Millisecond)
	s.voice.mu.Lock()
	s.voice.silent = true
	s.voice.mu.Unlock()
	s.askFor()

	s.eventually(func() bool { return s.spokenText("Book order 12?") }, "the question was never said")
	time.Sleep(600 * time.Millisecond)
	s.False(s.declinedBySilence(), "the wait counted from when the question started, not when it finished")
	finished := time.Now()
	s.voiceFinishes()

	s.eventually(s.declinedBySilence, "silence never declined the call")
	s.GreaterOrEqual(time.Since(finished), 300*time.Millisecond, "declined before the wait had passed")
	s.never(func() bool { return len(s.runner.asked()) > 0 }, "a call nobody allowed ran")
}

func (s *AgentSuite) TestAYesAfterTheWaitAsksTheModelToCallTheToolAgain() {
	s.agentAsksAndWaits(100 * time.Millisecond)
	s.askFor()
	s.eventually(s.declinedBySilence, "silence never declined the call")
	s.confirms(`{"tool_call_id":"call-1","allowed":true}`)

	s.says(stt.Participant{ID: "alice"}, "oh, yes")

	s.eventually(func() bool { return strings.Contains(s.historyWith(llm.ToolResult), "call lookup_order again") },
		"a late yes was not sent back to ask again")
	s.never(func() bool { return len(s.runner.asked()) > 0 }, "a late yes ran the call")
}

func (s *AgentSuite) TestConfirmToolCallIsOfferedOnlyWhenAToolNeedsApproval() {
	s.ownsTools("")
	s.join(true)
	_, offered := s.agent.availableTools().Lookup(confirmTool)
	s.False(offered, "confirm_tool_call was offered with nothing to confirm")

	s.ownsTools("")
	s.approves()
	s.join(true)
	_, offered = s.agent.availableTools().Lookup(confirmTool)
	s.True(offered)
}

func (s *AgentSuite) TestAToolThatNeedsApprovalIsNotAskedAboutInWriting() {
	s.ownsTools("order 12 ships tomorrow")
	s.tools.Tools[0].Approval = &harness.ToolApproval{Title: "Book order {order}?"}
	s.joinText()
	s.agent.runTool(harness.ToolRequested{TurnID: "turn-1", Call: llm.ToolCall{ID: "call-1", Name: "lookup_order", Arguments: `{"order":"12"}`}})

	s.Len(s.runner.asked(), 1, "in writing the app asks, as it always has")
	_, offered := s.agent.availableTools().Lookup(confirmTool)
	s.False(offered)
}

func (s *AgentSuite) TestTheQuestionFillsInTheCallsArguments() {
	for _, c := range []struct{ template, arguments, want string }{
		{"Book flight {flight_id}?", `{"flight_id":"BA 117"}`, "Book flight BA 117?"},
		{"Send {count} seats to {city}?", `{"count":2,"city":"Rome"}`, "Send 2 seats to Rome?"},
		{"Cancel {order}?", `{}`, "Cancel {order}?"},
		{"Cancel {order}?", `not json`, "Cancel {order}?"},
	} {
		s.Equal(c.want, fillQuestion(c.template, c.arguments), c.template+" "+c.arguments)
	}
}

// approves makes lookup_order need the caller's yes, for an agent joined after it.
func (s *AgentSuite) approves() {
	s.tools.Tools[0].Approval = &harness.ToolApproval{Title: "Book order {order}?"}
}

// agentAsksAndWaits is a voice call on which lookup_order needs a yes, waiting wait for one.
func (s *AgentSuite) agentAsksAndWaits(wait time.Duration) {
	s.ownsTools("order 12 booked")
	s.approves()
	s.join(true)
	s.agent.options.ApprovalWait = wait
	s.agent.options.ToolRunner = s.runner
}

// askFor has the caller ask for order 12, which the model calls lookup_order for.
func (s *AgentSuite) askFor() {
	s.model.reply = []string{"Sure."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "book order 12")
}

// askedToApprove is a voice call on which the caller was asked whether to book order 12.
func (s *AgentSuite) askedToApprove() {
	s.agentAsksAndWaits(time.Minute)
	s.askFor()
	s.eventually(func() bool { return s.spokenText("Book order 12?") }, "the question was never said")
	s.replyWentOut()
}

// confirms makes the model's next reply call confirm_tool_call with arguments.
func (s *AgentSuite) confirms(arguments string) {
	s.model.mu.Lock()
	defer s.model.mu.Unlock()
	s.model.calls = []llm.ToolCall{{ID: "call-2", Name: confirmTool, Arguments: arguments}}
	s.model.callsOn = len(s.model.asked) + 1
	s.model.then = []string{"Okay."}
}

// voiceFinishes has a silent voice finish everything it was asked to say.
func (s *AgentSuite) voiceFinishes() {
	s.voice.mu.Lock()
	s.voice.silent = false
	said := slices.Clone(s.voice.said)
	s.voice.mu.Unlock()
	finished := map[string]bool{}
	for _, request := range said {
		if !request.Final || finished[request.ID] {
			continue
		}
		finished[request.ID] = true
		s.voice.emitter.Send(tts.AudioChunk{
			SynthesisID: request.ID,
			Audio:       audio.PcmData{Samples: make([]int16, 160), SampleRate: 16_000, Channels: 1},
		})
		s.voice.emitter.Send(tts.SynthesisComplete{SynthesisID: request.ID, AudioDurationMs: 10, TimeToFirstByteMs: 5})
	}
}

func (s *AgentSuite) declinedBySilence() bool {
	return strings.Contains(s.historyWith(llm.User), "did not say yes")
}

// historyWith is the text of every message in the history with role.
func (s *AgentSuite) historyWith(role llm.Role) string {
	var text []string
	for _, message := range s.history() {
		if message.Role == role {
			text = append(text, message.Content)
		}
	}
	return strings.Join(text, "\n")
}
