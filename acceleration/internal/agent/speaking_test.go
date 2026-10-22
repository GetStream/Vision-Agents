package agent

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// heardAt is where the Speaking for a synthesis is among events, -1 for none.
func heardAt(events []Event, synthesisID string) int {
	return slices.IndexFunc(events, func(event Event) bool {
		speaking, ok := event.(Speaking)
		return ok && speaking.SynthesisID == synthesisID
	})
}

// spokeAt is where the Spoke for a synthesis is among events, -1 for none.
func spokeAt(events []Event, synthesisID string) int {
	return slices.IndexFunc(events, func(event Event) bool {
		spoke, ok := event.(Spoke)
		return ok && spoke.SynthesisID == synthesisID
	})
}

// sentTo is what the voice was sent for each synthesis.
func sentTo(requests []tts.Request) map[string]string {
	sent := map[string]string{}
	for _, request := range requests {
		sent[request.ID] += request.Text
	}
	return sent
}

// sayQuietly has the agent say a line to a voice that makes no audio of its own, so the
// test sends it.
func (s *AgentSuite) sayQuietly(line string) Said {
	s.voice.mu.Lock()
	s.voice.silent = true
	s.voice.mu.Unlock()
	s.Require().NoError(s.agent.Say(s.ctx, line))
	s.eventually(func() bool { return countOf[Said](s.reported()) == 1 }, "the line was never said")
	said, _ := firstOf[Said](s.reported())
	return said
}

func (s *AgentSuite) TestEachSentenceIsReportedHeardWithItsWordsBeforeItIsReportedSpoken() {
	s.join(false)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 2 }, "the reply was never spoken")
	reported := s.reported()
	sent := sentTo(s.voice.spoken())
	s.Equal(2, countOf[Speaking](reported), "one for each sentence the voice took")
	for index, event := range reported {
		spoke, ok := event.(Spoke)
		if !ok {
			continue
		}
		heard := heardAt(reported, spoke.SynthesisID)
		s.Require().GreaterOrEqual(heard, 0, "%s was never reported heard", spoke.SynthesisID)
		s.Less(heard, index, "it is heard before it is reported spoken")
		speaking := reported[heard].(Speaking)
		s.Equal(spoke.TurnID, speaking.TurnID)
		s.Equal(sent[spoke.SynthesisID], speaking.Text, "the words are exactly what the voice was sent")
		s.EqualValues(len(sent[spoke.SynthesisID]), spoke.Characters, "the characters are the voice's")
		s.False(speaking.At.IsZero())
	}
}

func (s *AgentSuite) TestAGreetingIsReportedHeardWithTheWordsItSays() {
	s.join(true)

	s.Require().NoError(s.agent.Say(s.ctx, "welcome"))

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the greeting was never spoken")
	reported := s.reported()
	said, _ := firstOf[Said](reported)
	speaking, _ := firstOf[Speaking](reported)
	spoke, _ := firstOf[Spoke](reported)
	s.Equal(said.TurnID, speaking.TurnID)
	s.Equal("welcome", speaking.Text)
	s.Equal(speaking.SynthesisID, spoke.SynthesisID)
	s.EqualValues(len("welcome"), spoke.Characters)
	s.Less(heardAt(reported, spoke.SynthesisID), spokeAt(reported, spoke.SynthesisID))
}

func (s *AgentSuite) TestAStreamedReplyIsReportedHeardWithTheStartOfWhatItSays() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the reply was never spoken")
	reported := s.reported()
	spoke, _ := firstOf[Spoke](reported)
	s.Equal(1, countOf[Speaking](reported), "a streaming voice says the whole reply as one utterance")
	speaking, _ := firstOf[Speaking](reported)
	whole := sentTo(s.voice.spoken())[spoke.SynthesisID]
	s.NotEmpty(speaking.Text)
	s.True(strings.HasPrefix(whole, speaking.Text), "%q is not how %q starts", speaking.Text, whole)
	s.Less(heardAt(reported, spoke.SynthesisID), spokeAt(reported, spoke.SynthesisID))
}

func (s *AgentSuite) TestAnUtteranceIsReportedHeardWhenTheTrackTakesItsFirstFrame() {
	s.edgeFactory = func(base *loopbackEdge) Edge {
		return &markedEdge{loopbackEdge: base, trackDelay: 150 * time.Millisecond}
	}
	s.join(true)
	said := s.sayQuietly("welcome")
	published := time.Now()

	s.synthesises(said.TurnID)

	s.eventually(func() bool { return countOf[Speaking](s.reported()) == 1 }, "the line was never reported heard")
	speaking, _ := firstOf[Speaking](s.reported())
	s.GreaterOrEqual(speaking.At.Sub(published), 100*time.Millisecond,
		"reported when it was published rather than when the track took it")
	s.Equal("welcome", speaking.Text)
	s.Zero(countOf[Spoke](s.reported()), "the voice has not finished it")

	s.voice.emitter.Send(tts.SynthesisComplete{SynthesisID: said.TurnID, Characters: 7, AudioDurationMs: 10})

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the line was never spoken")
	reported := s.reported()
	s.Less(heardAt(reported, said.TurnID), spokeAt(reported, said.TurnID))
	s.Equal(1, countOf[Speaking](reported))
}

func (s *AgentSuite) TestAnUtteranceFinishedBeforeTheTrackTakesItIsStillHeardBeforeItIsSpoken() {
	// A short line behind speech already queued is all synthesised before the track gets to
	// it, which is what the stub voice does with every line.
	s.edgeFactory = func(base *loopbackEdge) Edge {
		return &markedEdge{loopbackEdge: base, trackDelay: 150 * time.Millisecond}
	}
	s.join(true)

	s.Require().NoError(s.agent.Say(s.ctx, "welcome"))

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the line was never spoken")
	reported := s.reported()
	spoke, _ := firstOf[Spoke](reported)
	s.Require().GreaterOrEqual(heardAt(reported, spoke.SynthesisID), 0, "the line was never reported heard")
	s.Less(heardAt(reported, spoke.SynthesisID), spokeAt(reported, spoke.SynthesisID))
	s.Never(func() bool { return countOf[Speaking](s.reported()) > 1 }, 300*time.Millisecond, 10*time.Millisecond,
		"the track taking it later reported it heard a second time")
}

func (s *AgentSuite) TestAnUtteranceTalkedOverIsStillReportedHeard() {
	s.join(true)
	said := s.sayQuietly("welcome to the clinic")
	s.synthesises(said.TurnID)
	s.eventually(func() bool { return countOf[Speaking](s.reported()) == 1 }, "the line was never reported heard")

	s.agent.Interrupt()

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "the line was never cut off")
	reported := s.reported()
	s.Less(heardAt(reported, said.TurnID), slices.IndexFunc(reported, func(event Event) bool {
		_, ok := event.(Interrupted)
		return ok
	}))
	s.Zero(countOf[Spoke](reported), "it was not heard in full")
}

func (s *AgentSuite) TestAnUtteranceCutOffBeforeAnyOfItWentOutIsNeverReportedHeard() {
	s.join(true)
	said := s.sayQuietly("welcome to the clinic")
	s.agent.Interrupt()
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "the line was never cut off")

	s.synthesises(said.TurnID)

	s.Never(func() bool { return countOf[Speaking](s.reported()) > 0 }, 200*time.Millisecond, 10*time.Millisecond,
		"audio that was dropped was reported heard")
}

func (s *AgentSuite) TestANativeReplyIsReportedHeardWithItsTranscriptBeforeItIsSpoken() {
	agent, err := New(Options{CustomerID: "acme", Edge: newLoopbackEdge(), STS: &stsrouter.Router{}, STSTarget: "openai/gpt-realtime-2"})
	s.Require().NoError(err)
	agent.ctx, agent.cancel = context.WithCancel(s.ctx)
	s.T().Cleanup(func() { agent.Close() })
	seen := collect(agent)
	source := make(chan sts.Event, 8)
	ordered := make(chan sts.Event, 8)
	agent.pipe = newPipeline(agent.ctx, true)
	agent.pipe.running.Add(2)
	go agent.receiveSTS(agent.pipe, source, ordered)
	go agent.consumeSTS(agent.pipe, nil, ordered)
	s.T().Cleanup(func() { close(source) })

	pcm := audio.PcmData{Samples: []int16{1, 2, 3}, SampleRate: 24000, Channels: 1}
	source <- sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()}
	source <- sts.OutputTranscript{ResponseID: "r1", Mode: stt.ModeDelta, Text: "Hi there."}
	source <- sts.AudioChunk{ResponseID: "r1", Generation: 1, Audio: pcm}
	source <- sts.ResponseComplete{ResponseID: "r1", Generation: 1, AudioDurationMs: 10}

	s.Eventually(func() bool { return countOf[Spoke](seen.seen()) == 1 }, 2*time.Second, time.Millisecond)
	reported := seen.seen()
	speaking, _ := firstOf[Speaking](reported)
	spoke, _ := firstOf[Spoke](reported)
	s.Equal("r1", speaking.TurnID)
	s.Equal("Hi there.", speaking.Text)
	s.Equal("r1", spoke.SynthesisID)
	s.EqualValues(len("Hi there."), spoke.Characters, "a speech-to-speech model's are its transcript's")
	s.Less(heardAt(reported, "r1"), spokeAt(reported, "r1"))
}

// repliesUnderOneTurn has the agent answer, and checks every utterance of the reply is
// reported heard and spoken under the turn its text was written under: the transcript stores
// a reply under the turn its deltas name, and a page pairs that message with how much of it
// has been heard by the turn speaking and spoke name.
func (s *AgentSuite) repliesUnderOneTurn(streamingVoice bool) {
	s.join(streamingVoice)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "hello")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 && countOf[Spoke](s.reported()) > 0 },
		"the reply was never spoken")
	reported := s.reported()
	responded, _ := firstOf[Responded](reported)
	delta, _ := firstOf[ResponseDelta](reported)
	s.Require().NotEmpty(responded.TurnID)
	s.Equal(responded.TurnID, delta.TurnID)
	s.Positive(countOf[Speaking](reported))
	for _, event := range reported {
		switch typed := event.(type) {
		case Speaking:
			s.Equal(responded.TurnID, typed.TurnID, "%s", typed.SynthesisID)
		case Spoke:
			s.Equal(responded.TurnID, typed.TurnID, "%s", typed.SynthesisID)
		}
	}
}

func (s *AgentSuite) TestASentenceAtATimeIsReportedHeardUnderTheTurnTheReplyIsWrittenAs() {
	s.repliesUnderOneTurn(false)
}

func (s *AgentSuite) TestAStreamedReplyIsReportedHeardUnderTheTurnItIsWrittenAs() {
	s.repliesUnderOneTurn(true)
}

// queuedMarkedEdge is a marked edge that also counts its queue, holding as much speech
// waiting as the test says.
type queuedMarkedEdge struct {
	*markedEdge
	mu      sync.Mutex
	waiting time.Duration
}

func (e *queuedMarkedEdge) PlayoutStats() PlayoutStats {
	e.mu.Lock()
	defer e.mu.Unlock()
	return PlayoutStats{Waiting: e.waiting}
}

func (s *AgentSuite) TestAnUtteranceReportedBeforeTheTrackGetsToItSaysHowLongUntilItIsHeard() {
	// Its speaking comes as it finishes being made, behind speech still queued: the page
	// starts it when that has played, not when it was told.
	edge := &queuedMarkedEdge{waiting: 2 * time.Second}
	s.edgeFactory = func(base *loopbackEdge) Edge {
		edge.markedEdge = &markedEdge{loopbackEdge: base, trackDelay: time.Hour}
		return edge
	}
	s.join(true)

	s.Require().NoError(s.agent.Say(s.ctx, "welcome"))

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the line was never spoken")
	speaking, _ := firstOf[Speaking](s.reported())
	spoke, _ := firstOf[Spoke](s.reported())
	s.InDelta(2000-spoke.AudioDurationMs, speaking.DelayMs, 0.001, "the queue ahead of it, less its own audio")
}

func (s *AgentSuite) TestAnUtteranceHeardFromTheStartIsNotDelayed() {
	s.join(true)

	s.Require().NoError(s.agent.Say(s.ctx, "welcome"))

	s.eventually(func() bool { return countOf[Spoke](s.reported()) == 1 }, "the line was never spoken")
	speaking, _ := firstOf[Speaking](s.reported())
	s.Zero(speaking.DelayMs)
}

func (s *AgentSuite) TestAnUtteranceTalkedOverSaysHowMuchOfItWasHeard() {
	s.join(true)
	said := s.sayQuietly("welcome to the clinic")
	s.synthesises(said.TurnID)
	s.synthesises(said.TurnID)
	s.eventually(func() bool { return countOf[Speaking](s.reported()) == 1 }, "the line was never reported heard")

	s.agent.Interrupt()

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "the line was never cut off")
	cut, _ := firstOf[Interrupted](s.reported())
	s.Equal(said.TurnID, cut.SynthesisID)
	s.Positive(cut.HeardMs)
	s.LessOrEqual(cut.HeardMs, 20.0, "never more than the two 10 ms chunks published")
}

func (s *AgentSuite) TestAReplyCutOffBeforeAnyOfItWasHeardSaysNothingOfWhatWasHeard() {
	s.join(true)
	said := s.sayQuietly("welcome to the clinic")

	s.agent.Interrupt()

	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "the line was never cut off")
	cut, _ := firstOf[Interrupted](s.reported())
	s.Equal(said.TurnID, cut.TurnID)
	s.Empty(cut.SynthesisID)
	s.Zero(cut.HeardMs)
}

func (s *AgentSuite) TestAnUtteranceThatFinishedIsNotOpenedAgainByALateDelta() {
	s.join(true)
	s.agent.sending("turn-1", "turn-1", "Hello")
	s.agent.takeUtterance("turn-1")

	s.agent.sending("turn-1", "turn-1", " again")

	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	s.Nil(s.agent.utteranceLocked("turn-1"), "it would wait forever for a completion that never comes")
}

func (s *AgentSuite) TestOnlyTheLatestFinishedUtterancesAreRemembered() {
	s.join(true)
	for index := range finishedKept + 1 {
		s.agent.takeUtterance(fmt.Sprintf("turn-%d", index))
	}

	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	s.Len(s.agent.finished, finishedKept)
	s.NotContains(s.agent.finished, "turn-0")
	s.Contains(s.agent.finished, fmt.Sprintf("turn-%d", finishedKept))
}

func (s *AgentSuite) TestAnUtteranceTheVoiceRefusedIsLetGoOf() {
	s.join(false)
	s.voice.mu.Lock()
	s.voice.refuses = errors.New("stub: refused")
	s.voice.mu.Unlock()

	s.Error(s.agent.speakSentence("turn-1", "Hello there."))

	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	s.Empty(s.agent.heard)
}

func (s *AgentSuite) TestAStreamedDeltaTheVoiceRefusedIsLetGoOf() {
	s.join(true)
	s.voice.mu.Lock()
	s.voice.refuses = errors.New("stub: refused")
	s.voice.mu.Unlock()

	s.Error(s.agent.speakSentence("turn-1", "Hello there."))

	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	s.Empty(s.agent.heard, "only a final request was let go of before")
}

func (s *AgentSuite) TestClosingTheAgentLetsGoOfWhatItWasSaying() {
	s.join(true)
	s.agent.sending("turn-1", "turn-1", "Hello")

	s.Require().NoError(s.agent.Close())

	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	s.Empty(s.agent.heard)
}

func (s *AgentSuite) TestANativeReplyTheSessionDroppedIsLetGoOfWhenTheSessionEnds() {
	agent, err := New(Options{CustomerID: "acme", Edge: newLoopbackEdge(), STS: &stsrouter.Router{}, STSTarget: "openai/gpt-realtime-2"})
	s.Require().NoError(err)
	agent.ctx, agent.cancel = context.WithCancel(s.ctx)
	s.T().Cleanup(func() { agent.Close() })
	collect(agent)
	source := make(chan sts.Event, 8)
	ordered := make(chan sts.Event, 8)
	agent.pipe = newPipeline(agent.ctx, true)
	agent.pipe.running.Add(2)
	go agent.receiveSTS(agent.pipe, source, ordered)
	go agent.consumeSTS(agent.pipe, nil, ordered)

	source <- sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()}
	s.Eventually(func() bool {
		agent.mu.Lock()
		defer agent.mu.Unlock()
		return agent.utteranceLocked("r1") != nil
	}, 2*time.Second, time.Millisecond)

	close(source)

	s.Eventually(func() bool {
		agent.mu.Lock()
		defer agent.mu.Unlock()
		return len(agent.heard) == 0
	}, 2*time.Second, time.Millisecond, "it never completes, so it would be kept forever")
}
