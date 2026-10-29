package agent

import (
	"context"
	"errors"
	"log/slog"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/suite"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

const interruptedLine = "the agent was interrupted"

// interruptedOnce is the one line an abandoned reply logged.
func (s *AgentSuite) interruptedOnce() map[string]any {
	logged := s.loggedOnce(interruptedLine)
	s.never(func() bool { return len(s.logs.records(interruptedLine)) > 1 }, "one reply abandoned once logged twice")
	return logged
}

// replyGoingOut has the caller's words answered by a reply that has written said and begun
// to go out, and returns its turn.
func (s *AgentSuite) replyGoingOut(said string) string {
	s.model.reply = nil
	s.speak(nash)
	s.says(nash, "explain the menu")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the reply was never asked for")
	turn := s.model.requests()[0].ID
	if said != "" {
		s.model.writes(turn, said)
		s.eventually(func() bool { return s.spokenText(said) }, "the reply never reached the voice")
	}
	s.heardSomeOf(turn)
	return turn
}

func (s *AgentSuite) TestARulingOnSettledWordsSaysWhatTookTheFloorAndFromWhom() {
	s.logs = &logLines{}
	s.join(true)
	s.flow.then = []string{`{"disposition":"respond","floor":"stop"}`}
	turn := s.replyGoingOut("Here is the menu.")

	s.says(nash, "stop right there")

	logged := s.interruptedOnce()
	s.Equal(turn, logged["turn"])
	s.Equal("nash", logged["participant"])
	s.Equal(causeSettled, logged["cause"])
	s.Equal("primary", logged["ruler"])
	s.Equal("stop right there", logged["words"])
	s.Greater(logged["took_ms"], 0.0)
	s.Greater(logged["played_ms"], 0.0, "some of the reply had gone out")
	s.Equal(echoNone, logged["echo"], "the caller's own words")
	s.Equal(0.0, logged["echo_share"])
}

func (s *AgentSuite) TestTheAgentsOwnLineHeardBackIsAnEcho() {
	// Session 01a182c7: the words that cut the agent off were its own line.
	s.logs = &logLines{}
	spans := tracetest.NewSpanRecorder()
	turnSpans().RegisterSpanProcessor(spans)
	defer turnSpans().UnregisterSpanProcessor(spans)
	s.join(true)
	s.flow.then = []string{`{"disposition":"respond","floor":"stop"}`}
	turn := s.replyGoingOut("I can hear you loud and clear.")

	s.says(nash, "I can hear you, loud and clear")

	logged := s.interruptedOnce()
	s.Equal(echoMost, logged["echo"])
	s.Equal(1.0, logged["echo_share"])
	var traced []string
	for _, span := range spans.Ended() {
		if span.Name() != "agent.turn" {
			continue
		}
		for _, attribute := range span.Attributes() {
			if attribute.Key == "turn.id" && attribute.Value.AsString() == turn {
				for _, attribute := range span.Attributes() {
					traced = append(traced, string(attribute.Key)+"="+attribute.Value.Emit())
				}
			}
		}
	}
	s.Subset(traced, []string{"turn.interrupt.cause=settled", "turn.interrupt.ruler=primary", "turn.interrupt.echo=most"})
}

func (s *AgentSuite) TestARulingWhileTheCallerIsStillTalkingSaysSo() {
	s.logs = &logLines{}
	s.join(true)
	turn := s.replyGoingOut("")

	s.mutters(nash, "actually wait")

	logged := s.interruptedOnce()
	s.Equal(turn, logged["turn"])
	s.Equal(causeTalking, logged["cause"])
	s.Equal("primary", logged["ruler"])
	s.Equal("actually wait", logged["words"])
	s.Less(logged["took_ms"], 750.0, "the controller answered in time")
}

func (s *AgentSuite) TestAnOverlapNoModelRuledOnInTimeSaysTheDeadlineStoppedTheAgent() {
	// Prod us-east4, 2026-10-28: every interruption was this, logged as nobody's ruling.
	s.logs = &logLines{}
	s.join(true)
	deadline := testutil.ToFloat64(flowDecisions.WithLabelValues("wait", "deadline"))
	counted := testutil.ToFloat64(interruptions.WithLabelValues(causeTalking, "deadline", echoNoWords))
	turn := s.replyGoingOut("")
	held := make(chan struct{})
	defer close(held)
	s.flow.mu.Lock()
	s.flow.holdCreate = held
	s.flow.mu.Unlock()

	s.mutters(nash, "actually wait")

	logged := s.interruptedOnce()
	s.Equal(turn, logged["turn"])
	s.Equal(causeTalking, logged["cause"])
	s.Equal("deadline", logged["ruler"])
	s.GreaterOrEqual(logged["took_ms"], 750.0)
	s.Equal(1.0, testutil.ToFloat64(flowDecisions.WithLabelValues("wait", "deadline"))-deadline)
	s.Equal(1.0, testutil.ToFloat64(interruptions.WithLabelValues(causeTalking, "deadline", echoNoWords))-counted)
	// The control: what the base sha (b64d358ce) does with the same overlap, measured there.
	s.Equal(1, countOf[Interrupted](s.reported()))
	s.Equal(1, s.voice.interrupted())
	s.Equal(1, s.model.interrupted())
	s.Len(s.flow.requests(), 2)
	s.Len(s.model.requests(), 2)
	overlap, _ := firstOf[OverlapDecided](s.reported())
	s.Equal("stop", overlap.Action)
}

func (s *AgentSuite) TestAReplyDroppedBeforeAnyWentOutSaysSo() {
	s.logs = &logLines{}
	s.join(true)
	s.voice.silent = true
	turn := s.answering("explain the menu")

	s.mutters(nash, "actually wait")

	logged := s.interruptedOnce()
	s.Equal(turn, logged["turn"])
	s.Equal(causeUnheard, logged["cause"])
	s.Equal("none", logged["ruler"])
	s.Equal("actually wait", logged["words"])
	s.Equal(0.0, logged["played_ms"], "none of it went out")
	s.Equal(echoNoWords, logged["echo"], "the reply had said nothing")
	s.NotContains(logged, "echo_share")
}

func (s *AgentSuite) TestAnInterruptByTheAPIHasNoWords() {
	s.logs = &logLines{}
	s.join(true)
	counted := testutil.ToFloat64(interruptions.WithLabelValues(causeAPI, "none", echoNoWords))
	turn := s.replyGoingOut("Here is the menu.")

	s.agent.Interrupt()

	logged := s.interruptedOnce()
	s.Equal(turn, logged["turn"])
	s.Equal(causeAPI, logged["cause"])
	s.Equal("none", logged["ruler"])
	s.Equal("", logged["words"])
	s.Equal(echoNoWords, logged["echo"])
	s.NotContains(logged, "echo_share", "no words is not a share of none")
	s.Equal(1.0, testutil.ToFloat64(interruptions.WithLabelValues(causeAPI, "none", echoNoWords))-counted)
}

func (s *AgentSuite) TestAnUtteranceTheVoiceFinishedAfterItsTurnWasAbandonedSaysSo() {
	// Session 01a182c7, 13:52:20.366Z: the voice had synthesised the whole sentence, so its
	// completion came 1 ms after the interruption, not cut, and was logged interrupted=false.
	s.logs = &logLines{}
	s.join(true)
	turn := s.replyGoingOut("Here is the menu.")
	s.agent.Interrupt()
	s.interruptedOnce()

	s.voice.emitter.Send(tts.SynthesisComplete{SynthesisID: turn, AudioDurationMs: 3520})

	spoke := s.loggedOnce(finishedSpeaking)
	s.Equal(turn, spoke["turn"])
	s.Equal(false, spoke["synthesis_cut"], "the voice finished what it was asked to say")
	s.Equal(true, spoke["turn_abandoned"], "but the caller did not hear it out")
}

// nativeAgent is a speech-to-speech agent logging to s.logs, whose model's events are sent
// on the channel returned. They pass receiveSTS first, as a call's do.
func (s *AgentSuite) nativeAgent() chan<- sts.Event {
	s.logs = &logLines{}
	logger := slog.New(slog.NewJSONHandler(s.logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	agent, err := New(Options{CustomerID: "acme", Edge: newLoopbackEdge(), STS: &stsrouter.Router{},
		STSTarget: "openai/gpt-realtime-2", Logger: logger})
	s.Require().NoError(err)
	agent.ctx, agent.cancel = context.WithCancel(s.ctx)
	s.T().Cleanup(func() { agent.Close() })
	source := make(chan sts.Event, 8)
	ordered := make(chan sts.Event, 8)
	agent.pipe = newPipeline(agent.ctx, true)
	agent.pipe.running.Add(1)
	go agent.consumeSTS(agent.pipe, nil, ordered)
	agent.pipe.running.Add(1)
	go agent.receiveSTS(agent.pipe, source, ordered)
	s.T().Cleanup(func() { close(source) })
	return source
}

func (s *AgentSuite) TestANativeModelStoppingOnItsOwnHasNoWords() {
	source := s.nativeAgent()

	pcm := audio.PcmData{Samples: []int16{1, 2, 3}, SampleRate: 24000, Channels: 1}
	source <- sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()}
	source <- sts.AudioChunk{ResponseID: "r1", Generation: 1, Audio: pcm}
	source <- sts.ResponseComplete{ResponseID: "r1", Generation: 1, Interrupted: true}

	logged := s.interruptedOnce()
	s.Equal("r1", logged["turn"])
	s.Equal(causeProvider, logged["cause"])
	s.Equal("none", logged["ruler"])
	s.Equal("", logged["words"])
	s.Equal(echoNoWords, logged["echo"])
}

func (s *AgentSuite) TestANativeReplyTheModelStoppedAndTheAPIInterruptedIsToldOnce() {
	s.logs = &logLines{}
	logger := slog.New(slog.NewJSONHandler(s.logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	agent, err := New(Options{CustomerID: "acme", Edge: newLoopbackEdge(), STS: &stsrouter.Router{},
		STSTarget: "openai/gpt-realtime-2", Logger: logger})
	s.Require().NoError(err)
	agent.ctx, agent.cancel = context.WithCancel(s.ctx)
	s.T().Cleanup(func() { agent.Close() })
	events := collect(agent)
	source := make(chan sts.Event, 8)
	// Nobody reads it: consumeSTS is behind, as it is while PublishAudio blocks, so the reply
	// is still the one speaking when Agent.Interrupt comes.
	behind := make(chan sts.Event, 8)
	agent.pipe = newPipeline(agent.ctx, true)
	agent.pipe.running.Add(1)
	go agent.receiveSTS(agent.pipe, source, behind)
	s.T().Cleanup(func() { close(source) })
	agent.mu.Lock()
	agent.speakingTurn = "r1"
	agent.mu.Unlock()
	agent.turns.started("r1", kindCaller)
	api := testutil.ToFloat64(interruptions.WithLabelValues(causeAPI, "none", echoNoWords))

	source <- sts.ResponseComplete{ResponseID: "r1", Generation: 1, Interrupted: true}
	s.Equal(causeProvider, s.loggedOnce(interruptedLine)["cause"])
	agent.Interrupt()

	s.eventually(func() bool { return countOf[Interrupted](events.seen()) == 2 },
		"both abandons are reported, as before the line existed")
	s.Equal(causeProvider, s.interruptedOnce()["cause"])
	s.Zero(testutil.ToFloat64(interruptions.WithLabelValues(causeAPI, "none", echoNoWords))-api,
		"the reply the model stopped is not counted again")
}

func (s *AgentSuite) TestALineInterruptedAndThenRefusedIsToldOnce() {
	s.logs = &logLines{}
	s.join(false)
	synthesizing, release := make(chan struct{}), make(chan struct{})
	s.voice.mu.Lock()
	s.voice.synthesizing, s.voice.synthesisRelease = synthesizing, release
	s.voice.refuses = errors.New("socket closed")
	s.voice.mu.Unlock()
	refused := testutil.ToFloat64(interruptions.WithLabelValues(causeVoiceRefused, "none", echoNoWords))
	said := make(chan error, 1)
	go func() { said <- s.agent.Say(s.ctx, "Hi, thanks for getting in touch.") }()

	<-synthesizing
	s.agent.Interrupt()
	close(release)
	s.Require().Error(<-said)

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 2 },
		"both abandons are reported, as before the line existed")
	s.Equal(causeAPI, s.interruptedOnce()["cause"])
	s.Zero(testutil.ToFloat64(interruptions.WithLabelValues(causeVoiceRefused, "none", echoNoWords))-refused,
		"the line the caller interrupted is not counted again")
}

func (s *AgentSuite) TestALineALostVoiceWasSayingSaysTheVoiceWasLost() {
	s.logs = &logLines{}
	s.join(false)
	s.voice.silent = true
	s.replacement = newStubTTS(false)
	s.Require().NoError(s.agent.Say(s.ctx, "Hi, thanks for getting in touch."))
	s.eventually(func() bool { return countOf[Said](s.reported()) == 1 }, "the greeting was never reported")
	said, _ := firstOf[Said](s.reported())

	s.voice.emitter.Send(tts.Error{
		Provider: "stub", Model: "stub-tts", Err: errors.New("quota exceeded"), Context: "read", Fatal: true,
	})

	logged := s.interruptedOnce()
	s.Equal(said.TurnID, logged["turn"])
	s.Equal(causeVoiceLost, logged["cause"])
	s.Equal(echoNoWords, logged["echo"])
}

func (s *AgentSuite) TestALineTheVoiceRefusedSaysSo() {
	s.logs = &logLines{}
	s.join(false)
	s.voice.refuses = errors.New("text too long")

	s.Require().Error(s.agent.Say(s.ctx, "Hi, thanks for getting in touch."))

	logged := s.interruptedOnce()
	said, _ := firstOf[Said](s.reported())
	s.Equal(said.TurnID, logged["turn"])
	s.Equal(causeVoiceRefused, logged["cause"])
}

func TestInterruptionSuite(t *testing.T) { suite.Run(t, new(InterruptionSuite)) }

// InterruptionSuite is how much of the words that took the floor the reply had said itself.
type InterruptionSuite struct{ suite.Suite }

func (s *InterruptionSuite) TestEveryWordTheReplySaidIsAnEcho() {
	share, compared := echoShare("I can hear you loud and clear.", "Yes, I can hear you loud and clear. How can I help?")
	s.True(compared)
	s.Equal(1.0, share)
	s.Equal(echoMost, echoBucket(share, compared))
}

func (s *InterruptionSuite) TestCaseAndPunctuationDoNotHideAnEcho() {
	share, _ := echoShare("CLEAR!", "loud and clear.")
	s.Equal(1.0, share)
}

func (s *InterruptionSuite) TestWordsTheReplyNeverSaidAreNoEcho() {
	share, compared := echoShare("stop right there", "Here is the menu.")
	s.True(compared)
	s.Equal(0.0, share)
	s.Equal(echoNone, echoBucket(share, compared))
}

func (s *InterruptionSuite) TestTwoWordsInThreeAreAnEchoAndOneInTwoAreSome() {
	share, compared := echoShare("can I order", "you can order now")
	s.InDelta(2.0/3, share, 1e-9)
	s.Equal(echoMost, echoBucket(share, compared), "two words in three is where an echo starts")

	share, compared = echoShare("can we order soup", "you can order now")
	s.Equal(0.5, share)
	s.Equal(echoSome, echoBucket(share, compared))
}

func (s *InterruptionSuite) TestOneWordInFourIsSome() {
	share, compared := echoShare("wait I want soup", "you can order now, I think")
	s.Equal(0.25, share)
	s.Equal(echoSome, echoBucket(share, compared), "any word the reply said is some echo")
}

func (s *InterruptionSuite) TestNoWordsAreNotComparedWithAReply() {
	_, compared := echoShare("", "Here is the menu.")
	s.False(compared)
	_, compared = echoShare("...", "Here is the menu.")
	s.False(compared, "punctuation is no words")
	s.Equal(echoNoWords, echoBucket(0, false))
}

func (s *InterruptionSuite) TestWordsAreNotComparedWithAReplyThatSaidNothing() {
	_, compared := echoShare("stop right there", "")
	s.False(compared)
}
