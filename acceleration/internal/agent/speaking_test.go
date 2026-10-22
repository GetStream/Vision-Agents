package agent

import (
	"context"
	"slices"
	"strings"
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
