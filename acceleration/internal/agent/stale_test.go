package agent

import (
	"errors"
	"slices"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// nash is the caller of session 01a15c2c, whose "two" was answered over their "Hey,".
var nash = stt.Participant{ID: "nash"}

// answering has the caller's words answered by a reply that says nothing yet, and returns its turn.
func (s *AgentSuite) answering(text string) string {
	s.model.reply = nil
	s.speak(nash)
	s.says(nash, text)
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the reply was never asked for")
	return s.model.requests()[0].ID
}

func (s *AgentSuite) TestAReplyTheCallerSaidMoreOverBeforeHearingAnyIsDropped() {
	// Session 01a15c2c: "two" settled over the greeting and was ruled ambiguous, the caller
	// said "Hey," and the reply to "two" started 400 ms later, over them.
	held, release := make(chan struct{}), make(chan struct{})
	s.logs = &logLines{}
	s.joinWithVoice(true, func(voice *stubTTS) tts.TTS { return newHeldCompletionTTS(voice, held, release) })
	s.flow.reply = []string{`{"disposition":"clarify","floor":"stop"}`}
	s.flow.then = []string{`{"disposition":"respond","floor":"continue"}`}
	dropped := testutil.ToFloat64(staleReplies)
	overAgent := testutil.ToFloat64(settledOverAgent.WithLabelValues("1", "true", "interrupt"))
	played, _ := histogramOf("router_agent_played_before_interrupt_seconds", "kind", kindCaller)
	s.Require().NoError(s.agent.Say(s.ctx, "Hi, I'm Nessy. How can I help?"))
	<-held

	two := s.answering("two")
	close(release)
	s.mutters(nash, "Hey,")

	s.eventually(func() bool { return s.model.script(two).Abandoned() }, "the reply to two kept going")
	s.model.writes(two, "Which two do you mean?")
	s.synthesises(two)
	s.says(nash, "Hey. Can you help me find after school care for the kids?")
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the newer words were never answered")
	input := s.model.requests()[1].Input
	s.Equal("two", input[len(input)-2].Content, "the words the dropped reply was for are still in the history")
	s.Equal("Hey. Can you help me find after school care for the kids?", input[len(input)-1].Content)
	s.Never(func() bool { return len(s.edge.heard()) > 0 }, 100*time.Millisecond, 5*time.Millisecond,
		"none of the dropped reply may go out")
	s.Equal(1.0, testutil.ToFloat64(staleReplies)-dropped)
	s.Equal(1.0, testutil.ToFloat64(settledOverAgent.WithLabelValues("1", "true", "interrupt"))-overAgent,
		"one word, final, settled over the greeting and interrupting it")
	unplayed, _ := histogramOf("router_agent_played_before_interrupt_seconds", "kind", kindCaller)
	s.Equal(played, unplayed, "a reply nobody heard did not play")
	logged := s.logs.records("dropped a reply the caller said more over before hearing any of it")
	s.Require().Len(logged, 1)
	s.Equal(two, logged[0]["turn"])
	s.Equal("nash", logged[0]["participant"])
	s.Equal(false, logged[0]["sent_to_voice"])
	s.Contains(logged[0], "gap_ms")
}

func (s *AgentSuite) TestAReplyStartedBeforeTheRulingIsDroppedTheSameWay() {
	s.speculates = true
	s.join(true)
	ruling := make(chan struct{})
	s.flow.holdCreate = ruling
	dropped := testutil.ToFloat64(staleReplies)
	two := s.answering("two")
	close(ruling)
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 }, "the early reply was never used")

	s.mutters(nash, "wait no")

	s.eventually(func() bool { return s.model.script(two).Abandoned() }, "the early reply kept going")
	s.Len(s.model.requests(), 1, "the early reply was the one answering")
	s.Len(s.flow.requests(), 1, "words that drop a reply are not also put to the controller")
	s.Equal(1.0, testutil.ToFloat64(staleReplies)-dropped)
}

func (s *AgentSuite) TestAMurmurOrAWordOfAssentLeavesTheReplyToPlay() {
	s.join(true)
	s.flow.then = []string{`{"disposition":"respond","floor":"continue"}`}
	dropped := testutil.ToFloat64(staleReplies)
	two := s.answering("two")

	for _, murmur := range []string{"mm", "cough cough", "okay"} {
		s.mutters(nash, murmur)
	}
	s.eventually(func() bool { return countOf[Hearing](s.reported()) == 4 }, "the murmurs were never heard")
	s.model.writes(two, "Two of what?")

	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the reply was dropped over a murmur")
	s.Zero(countOf[Interrupted](s.reported()))
	s.Equal(0.0, testutil.ToFloat64(staleReplies)-dropped)
}

func (s *AgentSuite) TestAnotherVoiceLeavesTheReplyToTheController() {
	s.logs = &logLines{}
	s.join(true)
	s.flow.then = []string{`{"disposition":"wait","floor":"continue"}`}
	dropped, over := testutil.ToFloat64(staleReplies), testutil.ToFloat64(talkedOver.WithLabelValues(kindCaller))
	two := s.answering("two")
	bob := stt.Participant{ID: "bob"}
	s.speak(bob)

	s.mutters(bob, "wait no hold on")

	s.eventually(func() bool { return len(s.flow.requests()) == 2 }, "the other voice was never put to the controller")
	s.voice.emitter.Send(tts.SynthesisStarted{SynthesisID: two, Provider: "stub"})
	s.model.writes(two, "Two of what? Say again.")
	s.model.finishes(two)
	s.eventually(func() bool { return len(s.edge.heard()) == 2 }, "the reply was dropped over another voice")
	s.Zero(countOf[Interrupted](s.reported()))
	s.Equal(0.0, testutil.ToFloat64(staleReplies)-dropped)
	s.Equal(1.0, testutil.ToFloat64(talkedOver.WithLabelValues(kindCaller))-over,
		"bob was mid-utterance when the reply began, which is once however much of it went out")
	took := s.logs.records("the voice took an utterance")
	s.Require().Len(took, 1)
	s.Equal(true, took[0]["caller_talking"])
}

func (s *AgentSuite) TestWordsAfterAnotherVoiceTookTheFloorAreNotRuledOnAgain() {
	s.join(true)
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}
	s.answering("two")
	bob := stt.Participant{ID: "bob"}
	s.speak(bob)
	s.mutters(bob, "wait no hold on")
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "the other voice never took the floor")
	decided := countOf[Decided](s.reported())

	s.mutters(nash, "hold on please")

	s.eventually(func() bool { return countOf[Hearing](s.reported()) == 3 }, "the caller was never heard")
	s.Never(func() bool { return countOf[Decided](s.reported()) > decided }, 100*time.Millisecond, 5*time.Millisecond,
		"a reply already abandoned is no reply to drop")
}

func (s *AgentSuite) TestASettledTurnOverTheAgentIsCountedByItsWords() {
	s.join(true)
	s.flow.then = []string{`{"disposition":"wait","floor":"continue"}`}
	before := testutil.ToFloat64(settledOverAgent.WithLabelValues("2-3", "false", "wait"))
	s.heardSomeOf(s.answering("two"))

	s.mutters(nash, "the second")

	s.eventually(func() bool {
		return testutil.ToFloat64(settledOverAgent.WithLabelValues("2-3", "false", "wait")) > before
	}, "two words the transcriber never finalized, ruled while the agent talked, were not counted")
}

func (s *AgentSuite) TestAReplyNobodySaidMoreOverPlaysAsBefore() {
	// Every expected value is what base 318b0352 did with the same turn.
	s.join(true)
	dropped, over := testutil.ToFloat64(staleReplies), testutil.ToFloat64(talkedOver.WithLabelValues(kindCaller))
	quiet := testutil.ToFloat64(settledOverAgent.WithLabelValues("1", "true", "answer"))
	s.speak(nash)

	s.says(nash, "two")

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the reply never played")
	s.eventually(func() bool { return len(s.agent.History()) == 2 }, "the exchange was never kept")
	var spoken []string
	for _, request := range s.voice.spoken() {
		spoken = append(spoken, request.Text)
	}
	s.Equal([]string{"Hello there.", "How are you?", ""}, spoken)
	s.Len(s.edge.heard(), 2)
	s.Zero(countOf[Interrupted](s.reported()))
	s.Equal("two", s.agent.History()[0].Content)
	s.Equal("Hello there. How are you?", s.agent.History()[1].Content)
	s.Equal(0.0, testutil.ToFloat64(staleReplies)-dropped)
	s.Equal(0.0, testutil.ToFloat64(talkedOver.WithLabelValues(kindCaller))-over)
	s.Equal(0.0, testutil.ToFloat64(settledOverAgent.WithLabelValues("1", "true", "answer"))-quiet,
		"a turn settled while the agent was quiet is not one over it")
}

func (s *AgentSuite) TestHowLongAReplyPlayedBeforeAnInterruptionIsMeasured() {
	s.join(true)
	s.flow.then = []string{`{"disposition":"respond","floor":"stop"}`}
	before, _ := histogramOf("router_agent_played_before_interrupt_seconds", "kind", kindCaller)
	overAgent := testutil.ToFloat64(settledOverAgent.WithLabelValues("2-3", "true", "interrupt"))
	two := s.answering("two")
	s.heardSomeOf(two)

	s.says(nash, "stop right there")

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "the reply was never interrupted")
	after, _ := histogramOf("router_agent_played_before_interrupt_seconds", "kind", kindCaller)
	s.Equal(before+1, after)
	s.Equal(1.0, testutil.ToFloat64(settledOverAgent.WithLabelValues("2-3", "true", "interrupt"))-overAgent)
}

func (s *AgentSuite) TestFlowRulingsAreCountedByWhoseAnswerWasUsed() {
	s.join(true)
	primary := testutil.ToFloat64(flowDecisions.WithLabelValues("respond", "primary"))
	failed := testutil.ToFloat64(flowDecisions.WithLabelValues("failed", "none"))
	timed, _ := histogramOf("router_agent_flow_decision_seconds", "ruler", "primary")
	s.speak(nash)

	s.says(nash, "two")
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the reply never played")
	s.flow.mu.Lock()
	s.flow.refuses = errors.New("stub: the controller is down")
	s.flow.mu.Unlock()
	s.says(nash, "three")
	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 2 }, "the second reply never played")

	s.Equal(1.0, testutil.ToFloat64(flowDecisions.WithLabelValues("respond", "primary"))-primary)
	s.Equal(1.0, testutil.ToFloat64(flowDecisions.WithLabelValues("failed", "none"))-failed)
	samples, _ := histogramOf("router_agent_flow_decision_seconds", "ruler", "primary")
	s.Equal(timed+1, samples)
}

func (s *AgentSuite) TestSayingMoreBeforeAnyReplyIsHeardCancelsAToolWhoseBindingAsksForIt() {
	s.ownsTools("")
	s.toolPolicy = func(string) ToolPolicy { return ToolPolicy{Cancels: true} }
	s.join(true)
	s.voice.silent = true
	runner := &cancellationTool{began: make(chan struct{}), stopped: make(chan struct{})}
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{"Let me check the order."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	dropped := testutil.ToFloat64(staleReplies)
	s.speak(nash)
	s.says(nash, "where is my order")
	select {
	case <-runner.began:
	case <-time.After(3 * time.Second):
		s.FailNow("tool did not start")
	}

	s.mutters(nash, "hello?")

	select {
	case <-runner.stopped:
	case <-time.After(3 * time.Second):
		s.FailNow("the binding asked for its call to be cancelled and it kept running")
	}
	s.Equal(1.0, testutil.ToFloat64(staleReplies)-dropped, "the reply was dropped, not talked over")
	s.False(slices.ContainsFunc(s.history(), func(m llm.Message) bool { return strings.HasPrefix(m.Content, stillRunning) }),
		"a cancelled call was answered as still running")
}

func (s *AgentSuite) TestSayingMoreBeforeAnyReplyIsHeardAnswersARunningToolStillRunning() {
	s.ownsTools("")
	s.join(true)
	s.voice.silent = true
	runner := &cancellationTool{began: make(chan struct{}), stopped: make(chan struct{})}
	s.agent.options.ToolRunner = runner
	s.model.reply = []string{"Let me check the order."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	dropped := testutil.ToFloat64(staleReplies)
	s.speak(nash)
	s.says(nash, "where is my order")
	select {
	case <-runner.began:
	case <-time.After(3 * time.Second):
		s.FailNow("tool did not start")
	}

	s.mutters(nash, "hello?")

	s.eventually(func() bool {
		return slices.ContainsFunc(s.history(), func(m llm.Message) bool { return strings.HasPrefix(m.Content, stillRunning) })
	}, "the call the dropped reply made was not answered")
	s.Equal(1.0, testutil.ToFloat64(staleReplies)-dropped, "the reply was dropped, not talked over")
	select {
	case <-runner.stopped:
		s.FailNow("saying more cancelled what the caller had asked for")
	default:
	}
}

func (s *AgentSuite) TestAReplyThatEndedWithNothingSaidIsNoReplyToDrop() {
	s.join(true)
	dropped := testutil.ToFloat64(staleReplies)
	two := s.answering("two")
	s.model.writes(two, "")
	s.model.finishes(two)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never settled")
	s.eventually(func() bool { return s.agent.floor().Quiet }, "the agent never fell quiet")

	s.mutters(nash, "tell me more please")

	s.eventually(func() bool { return countOf[Hearing](s.reported()) == 2 }, "the caller was never heard")
	s.Never(func() bool { return countOf[Interrupted](s.reported()) > 0 }, 200*time.Millisecond, 5*time.Millisecond,
		"a reply that had ended was dropped")
	s.Equal(0.0, testutil.ToFloat64(staleReplies)-dropped)
}

func (s *AgentSuite) TestAssentOfMoreThanOneWordLeavesTheReplyToPlay() {
	for _, assent := range []string{"okay okay", "yeah yeah", "uh huh", "uh-huh", "got it", "thank you", "uh yeah"} {
		s.Run(assent, func() {
			s.False(takesFloor(assent), "assent is somebody listening")
		})
	}
	for _, more := range []string{"okay wait", "yeah the second one", "got it but"} {
		s.Run(more, func() {
			s.True(takesFloor(more), "a word past assent is the caller saying more")
		})
	}
}

func (s *AgentSuite) TestTwoWordsOfAssentLeaveTheReplyToTheController() {
	s.join(true)
	s.flow.then = []string{`{"disposition":"ignore","floor":"continue"}`}
	dropped := testutil.ToFloat64(staleReplies)
	two := s.answering("two")

	s.mutters(nash, "okay okay")

	s.eventually(func() bool { return len(s.flow.requests()) == 2 }, "the assent was never put to the controller")
	s.model.writes(two, "Two of what?")
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "the reply was dropped over assent")
	s.Zero(countOf[Interrupted](s.reported()))
	s.Equal(0.0, testutil.ToFloat64(staleReplies)-dropped)
}
