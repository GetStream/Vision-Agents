package agent

import (
	"errors"
	"fmt"
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
	s.oneDropSince(dropped)
	s.Equal(1.0, testutil.ToFloat64(settledOverAgent.WithLabelValues("1", "true", "interrupt"))-overAgent,
		"one word, final, settled over the greeting and interrupting it")
	unplayed, _ := histogramOf("router_agent_played_before_interrupt_seconds", "kind", kindCaller)
	s.Equal(played, unplayed, "a reply nobody heard did not play")
	logged := s.dropLogged()
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
	s.oneDropSince(dropped)
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
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}

	s.mutters(nash, "hello?")

	select {
	case <-runner.stopped:
	case <-time.After(3 * time.Second):
		s.FailNow("the binding asked for its call to be cancelled and it kept running")
	}
	s.oneDropSince(dropped)
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
	s.flow.then = []string{`{"disposition":"wait","floor":"stop"}`}

	s.mutters(nash, "hello?")

	s.eventually(func() bool {
		return slices.ContainsFunc(s.history(), func(m llm.Message) bool { return strings.HasPrefix(m.Content, stillRunning) })
	}, "the call the dropped reply made was not answered")
	s.oneDropSince(dropped)
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
	for _, assent := range []string{"okay okay", "yeah yeah", "uh huh", "uh-huh", "uh-huh okay", "got it", "thank you", "uh yeah",
		"ok gotcha", "yes yes", "yep yep", "yup sure", "mhm right", "mm-hmm alright", "thanks thanks"} {
		s.Run(assent, func() {
			s.False(takesFloor(assent), "assent is somebody listening")
		})
	}
	for _, lone := range []string{"Mom", "two"} {
		s.Run(lone, func() {
			s.False(takesFloor(lone), "a lone word that is no interruption is somebody waiting to be answered")
		})
	}
	for _, noise := range []string{"huh", "uh", "mm", "hm", "hmm"} {
		s.True(overlapNoise(noise), "%q said alone is noise", noise)
	}
	s.False(overlapNoise("okay"), "assent said alone is still put to the controller over the agent")
	for _, more := range []string{"Hey,", "okay wait", "yeah the second one", "got it but"} {
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

func (s *AgentSuite) TestAResumedUtteranceRestatingAnsweredWordsPlusAMurmurLeavesTheReplyToPlay() {
	s.join(true)
	s.flow.then = []string{`{"disposition":"respond","floor":"continue"}`}
	dropped := testutil.ToFloat64(staleReplies)
	s.model.reply = nil
	s.speak(nash)
	s.ears.emitter.Send(stt.Transcript{Participant: nash, Mode: stt.ModeFinal, Text: "two", Language: "en", Confidence: 1, Utterance: 7})
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the caller was never answered")
	two := s.model.requests()[0].ID

	s.ears.emitter.Send(stt.Transcript{Participant: nash, Mode: stt.ModeReplacement, Text: "two mm", Language: "en", Utterance: 7})

	s.Never(func() bool { return s.model.script(two).Abandoned() || testutil.ToFloat64(staleReplies) > dropped },
		300*time.Millisecond, 5*time.Millisecond, "a murmur restated with the answered words dropped the reply")
}

func (s *AgentSuite) TestAResumedUtteranceRestatingAnsweredWordsPlusNewWordsDropsTheReply() {
	s.join(true)
	s.flow.then = []string{`{"disposition":"respond","floor":"continue"}`}
	dropped := testutil.ToFloat64(staleReplies)
	s.model.reply = nil
	s.speak(nash)
	s.ears.emitter.Send(stt.Transcript{Participant: nash, Mode: stt.ModeFinal, Text: "two", Language: "en", Confidence: 1, Utterance: 7})
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the caller was never answered")
	two := s.model.requests()[0].ID

	s.ears.emitter.Send(stt.Transcript{Participant: nash, Mode: stt.ModeReplacement, Text: "two wait no", Language: "en", Utterance: 7})

	s.eventually(func() bool { return s.model.script(two).Abandoned() }, "the reply was not dropped")
	s.Equal(1.0, testutil.ToFloat64(staleReplies)-dropped)
}

func (s *AgentSuite) TestAReplyHeardBeforeTheDropLandsIsNotDropped() {
	s.join(true)
	dropped := testutil.ToFloat64(staleReplies)
	two := s.answering("two")
	s.heardSomeOf(two)

	s.agent.perform(Action{Kind: ActInterrupt, Unheard: true, Participant: nash, TurnID: two})

	s.Never(func() bool { return countOf[Interrupted](s.reported()) > 0 }, 200*time.Millisecond, 5*time.Millisecond,
		"a reply already heard was dropped as unheard")
	s.Equal(0.0, testutil.ToFloat64(staleReplies)-dropped)
}

func (s *AgentSuite) TestADropLandingAfterAnotherTurnTookTheFloorLeavesThatTurnPlaying() {
	s.join(true)
	dropped := testutil.ToFloat64(staleReplies)
	two := s.answering("two")
	s.agent.mu.Lock()
	s.agent.speakingTurn = "check-in"
	s.agent.mu.Unlock()

	s.agent.perform(Action{Kind: ActInterrupt, Unheard: true, Participant: nash, TurnID: two})

	s.Never(func() bool { return countOf[Interrupted](s.reported()) > 0 }, 200*time.Millisecond, 5*time.Millisecond,
		"a drop meant for another turn interrupted the one speaking")
	s.Equal(0.0, testutil.ToFloat64(staleReplies)-dropped)
}

func (s *AgentSuite) TestAReplyTheCallerSaidMoreOverAfterItReachedTheVoiceIsDropped() {
	s.logs = &logLines{}
	s.join(true)
	s.voice.silent = true
	two := s.answering("two")
	s.model.writes(two, "Which two do you mean?")
	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "the reply never reached the voice")

	s.mutters(nash, "wait no")

	s.eventually(func() bool { return s.model.script(two).Abandoned() }, "the reply kept going")
	logged := s.dropLogged()
	s.Require().Len(logged, 1)
	s.Equal(true, logged[0]["sent_to_voice"])
}

func (s *AgentSuite) TestFinalWordsOverAReplyNobodyHeardDropIt() {
	s.join(true)
	dropped := testutil.ToFloat64(staleReplies)
	two := s.answering("two")

	s.says(nash, "only the vegetarian options")

	s.eventually(func() bool { return s.model.script(two).Abandoned() }, "final words left the reply to play")
	s.oneDropSince(dropped)
}

func (s *AgentSuite) TestTheTurnAfterADroppedReplyIsAskedToAnswerWhatItWasFor() {
	// Voicebench restaurant.interrupt on #1100: the reply to a correction made over a tool's
	// answer was dropped when the caller went on, and their next turn was answered alone, so
	// the gluten-free guest was never booked.
	s.ownsTools("a table for four at seven is free")
	s.join(true)
	s.model.reply = []string{"Let me check."}
	s.model.then = []string{}
	s.asksFor("lookup_order", `{"order":"4"}`)
	s.speak(nash)
	s.says(nash, "A table for four at seven please.")
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the tool's result was never put to the model")
	booked := s.model.requests()[1].ID
	s.model.writes(booked, "A table for four at seven is free.")
	s.heardSomeOf(booked)

	correction := "Wait. Make it six, and one of us is gluten free instead."
	s.says(nash, correction)
	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the correction was never answered")
	corrected := s.model.requests()[2].ID
	s.mutters(nash, "Nine Name is")
	s.eventually(func() bool { return s.model.script(corrected).Abandoned() }, "the reply to the correction kept going")
	s.says(nash, "Nine Name is Alvarez.")
	s.eventually(func() bool { return len(s.model.requests()) == 4 }, "the newer words were never answered")

	newer := s.model.requests()[3]
	s.Contains(newer.Instructions, missedNote(correction), "the correction is owed with the newer words")
	s.NotContains(s.model.requests()[2].Instructions, "not yet answered", "nothing was dropped before the correction")

	s.model.writes(newer.ID, "Six at seven, gluten free noted. Under Alvarez?")
	s.heardSomeOf(newer.ID)
	s.model.finishes(newer.ID)
	s.says(nash, "Can we sit outside?")
	s.eventually(func() bool { return len(s.model.requests()) == 5 }, "the next turn was never answered")
	s.NotContains(s.model.requests()[4].Instructions, "not yet answered", "what was owed was answered once")
}

func (s *AgentSuite) TestATurnAnsweredWithNothingDroppedIsAskedAsBefore() {
	// On base 318b0352 every caller turn is asked with the same instructions.
	s.join(true)
	two := s.answering("two")
	s.model.writes(two, "Two of what?")
	s.model.finishes(two)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never settled")

	s.says(nash, "two tickets please")

	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the next turn was never answered")
	s.Equal(s.model.requests()[0].Instructions, s.model.requests()[1].Instructions)
	s.NotContains(s.model.requests()[1].Instructions, "not yet answered")
}

func (s *AgentSuite) TestWhatADroppedReplyWasForIsOwedToTheSameCallerAlone() {
	s.join(true)
	two := s.answering("two")
	s.mutters(nash, "wait no")
	s.eventually(func() bool { return s.model.script(two).Abandoned() }, "the reply kept going")
	bob := stt.Participant{ID: "bob"}
	s.speak(bob)

	s.says(bob, "Is anyone there?")
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the other voice was never answered")
	s.NotContains(s.model.requests()[1].Instructions, "not yet answered", "bob never said two")
	s.says(nash, "wait no the second one")
	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the caller was never answered")

	s.Contains(s.model.requests()[2].Instructions, missedNote("two"))
}

func (s *AgentSuite) TestEveryDroppedReplyInARowIsOwed() {
	s.join(true)
	two := s.answering("two")
	s.mutters(nash, "wait no")
	s.eventually(func() bool { return s.model.script(two).Abandoned() }, "the first reply kept going")
	s.says(nash, "wait no three")
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the newer words were never answered")
	three := s.model.requests()[1].ID
	s.mutters(nash, "actually")
	s.eventually(func() bool { return s.model.script(three).Abandoned() }, "the second reply kept going")

	s.says(nash, "actually four")

	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the last words were never answered")
	s.Contains(s.model.requests()[2].Instructions, missedNote("two wait no three"))
}

func (s *AgentSuite) TestWhatDroppedRepliesOweKeepsOnlyTheNewestWords() {
	numbered := func(prefix string, from, to int) string {
		var words []string
		for i := from; i <= to; i++ {
			words = append(words, fmt.Sprintf("%s%d", prefix, i))
		}
		return strings.Join(words, " ")
	}
	s.join(true)
	first := s.answering(numbered("w", 1, 80))
	s.mutters(nash, "x1 x2")
	s.eventually(func() bool { return s.model.script(first).Abandoned() }, "the first reply kept going")
	s.says(nash, numbered("x", 1, 80))
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the newer words were never answered")
	second := s.model.requests()[1].ID
	s.mutters(nash, "y1 y2")
	s.eventually(func() bool { return s.model.script(second).Abandoned() }, "the second reply kept going")

	s.says(nash, numbered("y", 1, 10))

	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the last words were never answered")
	s.Contains(s.model.requests()[2].Instructions,
		missedNote(numbered("w", 61, 80)+" "+numbered("x", 1, 80)),
		"the newest hundred words of the two dropped turns are owed, the oldest twenty are not")
}

func (s *AgentSuite) TestAReplyStartedBeforeTheRulingIsNotUsedForWhatADroppedReplyOwes() {
	s.speculates = true
	s.join(true)
	two := s.answering("two")
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 }, "the reply was never started")
	s.mutters(nash, "wait no")
	s.eventually(func() bool { return s.model.script(two).Abandoned() }, "the reply kept going")

	s.says(nash, "wait no three")

	// The reply started early may reach the model after the one written in its place, so
	// the one with the note is looked for among every request, not taken to be the last.
	s.eventually(func() bool {
		return slices.ContainsFunc(s.model.requests(), func(asked llm.ResponseParams) bool {
			return strings.Contains(asked.Instructions, missedNote("two"))
		})
	}, "the turn was answered by a reply written without what it owes")
}

// oneDropSince waits for the stale-reply counter to move past before, then says it moved by
// exactly one. dropUnheard abandons the reply before it counts the drop, so a test that saw
// the reply abandoned can read the counter first.
func (s *AgentSuite) oneDropSince(before float64) {
	s.eventually(func() bool { return testutil.ToFloat64(staleReplies) > before }, "the drop was never counted")
	s.Equal(1.0, testutil.ToFloat64(staleReplies)-before, "the reply was dropped once, not talked over")
}

// dropLogged waits for the line logged when a reply is dropped, which comes after the reply
// is abandoned, and returns what was logged.
func (s *AgentSuite) dropLogged() []map[string]any {
	const line = "dropped a reply the caller said more over before hearing any of it"
	s.eventually(func() bool { return len(s.logs.records(line)) > 0 }, "the drop was never logged")
	return s.logs.records(line)
}

// queuesBehindAReply has the caller's first words answered by a reply that is going out, and
// what they say over it queued until it ends. It returns the reply's turn.
func (s *AgentSuite) queuesBehindAReply(first, over string) string {
	s.model.reply = nil
	s.speak(nash)
	s.says(nash, first)
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the reply was never asked for")
	reply := s.model.requests()[0].ID
	s.heardSomeOf(reply)
	s.flow.then = []string{`{"disposition":"respond","floor":"continue"}`}
	s.says(nash, over)
	s.eventually(func() bool { return countOf[OverlapDecided](s.reported()) == 1 }, "the words over the reply were never queued")
	return reply
}

func (s *AgentSuite) TestAQueuedTurnTheCallerCorrectedSinceIsAnsweredWithTheCorrection() {
	// Voicebench restaurant.interrupt on v0.6.115: "peanut allergy" was queued over the reply,
	// the caller corrected it once the reply was cut, and after answering the correction the
	// agent answered the queued words on their own and asked about the allergy again.
	s.logs = &logLines{}
	s.join(true)
	dropped := testutil.ToFloat64(staleQueuedTurns)
	newer := s.correctsAQueuedTurn()
	s.Equal(queuedCorrection, newer.Input[len(newer.Input)-1].Content)
	s.Contains(newer.Instructions, missedNote("Peanut allergy too."), "the queued words are owed with the correction")
	for _, message := range newer.Input {
		s.NotEqual("Peanut allergy too.", message.Content, "the queued words reach the model in the note, not the history")
	}
	s.model.writes(newer.ID, "Six at seven, gluten free and a peanut allergy noted.")
	s.model.finishes(newer.ID)

	s.Never(func() bool { return len(s.model.requests()) > 2 }, 200*time.Millisecond, 5*time.Millisecond,
		"the queued words were answered again on their own")
	s.Equal(1.0, testutil.ToFloat64(staleQueuedTurns)-dropped)
	logged := s.logs.records("dropped a queued turn the caller has said more after")
	s.Require().Len(logged, 1)
	s.Equal("nash", logged[0]["participant"])
	s.Contains(logged[0]["turn"], "turn-")
	s.Positive(logged[0]["age_ms"], "the queued turn waited behind the reply")
}

// queuedCorrection is what the caller of Voicebench restaurant.interrupt said once the reply was cut.
const queuedCorrection = "Wait. Make it six, and one of us is gluten free instead."

// correctsAQueuedTurn has the caller's "Peanut allergy too." queued behind a reply, then
// their correction answered once the reply ends, and returns the correction's request.
func (s *AgentSuite) correctsAQueuedTurn() llm.ResponseParams {
	reply := s.queuesBehindAReply("A table for four at seven please.", "Peanut allergy too.")
	s.mutters(nash, "Wait.")
	s.eventually(func() bool { return countOf[Hearing](s.reported()) >= 3 }, "the caller was never heard going on")
	s.model.writes(reply, "Seven works.")
	s.model.finishes(reply)
	s.eventually(func() bool {
		return slices.ContainsFunc(s.reported(), func(event Event) bool {
			decided, ok := event.(Decided)
			return ok && decided.Kind == string(ActHold)
		})
	}, "the queued turn did not wait on the caller still talking")

	s.says(nash, queuedCorrection)
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the correction was never answered")
	return s.model.requests()[1]
}

func (s *AgentSuite) TestQueuedWordsTakenInByADroppedReplyAreOwedToTheNextTurn() {
	// The reply to the correction is dropped before any of it went out (AI-1601), so what it
	// was for, the queued words with it, is owed to the caller's next turn.
	s.join(true)
	newer := s.correctsAQueuedTurn()
	s.Require().Contains(newer.Instructions, missedNote("Peanut allergy too."))
	s.mutters(nash, "Under the name")
	s.eventually(func() bool { return s.model.script(newer.ID).Abandoned() }, "the reply to the correction kept going")

	s.says(nash, "Under the name Alvarez.")

	s.eventually(func() bool { return len(s.model.requests()) == 3 }, "the newer words were never answered")
	s.Contains(s.model.requests()[2].Instructions, missedNote("Peanut allergy too. "+queuedCorrection))
}

func (s *AgentSuite) TestAQueuedTurnNobodySaidMoreAfterIsAskedAsBefore() {
	// Every expected value is what base 4588e625 did with the same turns.
	s.join(true)
	dropped := testutil.ToFloat64(staleQueuedTurns)
	reply := s.queuesBehindAReply("A table for four at seven please.", "Peanut allergy too.")

	s.model.writes(reply, "Seven works.")
	s.model.finishes(reply)

	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "the queued turn was never answered")
	queued := s.model.requests()[1]
	s.Equal([]string{"A table for four at seven please.", "Seven works.", "Peanut allergy too."},
		contents(queued.Input))
	s.Equal(s.model.requests()[0].Instructions, queued.Instructions)
	s.NotContains(queued.Instructions, "not yet answered")
	s.Equal(0.0, testutil.ToFloat64(staleQueuedTurns)-dropped)
}

func contents(messages []llm.Message) []string {
	var kept []string
	for _, message := range messages {
		kept = append(kept, message.Content)
	}
	return kept
}
