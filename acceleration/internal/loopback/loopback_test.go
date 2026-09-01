package loopback

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type LoopbackSuite struct {
	suite.Suite
}

func TestLoopbackSuite(t *testing.T) {
	suite.Run(t, new(LoopbackSuite))
}

// half a second of audio, which is long enough that pacing it is measurable and short
// enough that a test waiting for it is not.
func speech(ms int) audio.PcmData {
	return audio.PcmData{
		Samples:    make([]int16, stt.SampleRate*ms/1000),
		SampleRate: stt.SampleRate,
		Channels:   1,
	}
}

func (s *LoopbackSuite) TestSpeechIsHeldOntoUntilItWouldHaveBeenHeard() {
	edge := New(Options{PlayoutAhead: 100 * time.Millisecond})

	started := time.Now()
	s.Require().NoError(edge.PublishAudio(speech(500)))
	took := time.Since(started)

	// Half a second of speech, of which 100ms may be waiting to be heard: publishing has
	// to hold onto the rest rather than hand it over at once. An edge that swallowed it
	// would leave the agent believing it had finished talking before anybody heard it.
	s.Greater(took, 300*time.Millisecond)
	s.True(edge.Talking(), "the last of it has not been heard yet")
}

func (s *LoopbackSuite) TestAnEdgeIsQuietOnceEverythingHasBeenHeard() {
	// Allowing more to be queued than is published means this returns at once, so what is
	// asserted afterwards is the playhead rather than the sleep.
	edge := New(Options{PlayoutAhead: time.Second})

	s.Require().NoError(edge.PublishAudio(speech(150)))

	s.True(edge.Talking(), "speech that has not finished being heard is still speech")
	s.Eventually(func() bool { return !edge.Talking() }, time.Second, 10*time.Millisecond,
		"the floor is given up once the last of it has been heard")
}

func (s *LoopbackSuite) TestSomebodyOnTheOtherEndHearsWhatTheAgentSaid() {
	var heard int
	edge := New(Options{
		PlayoutAhead: time.Second,
		Published:    func(pcm audio.PcmData) { heard += len(pcm.Samples) },
	})

	s.Require().NoError(edge.PublishAudio(speech(100)))

	s.Equal(stt.SampleRate*100/1000, heard)
}

func (s *LoopbackSuite) TestAMicrophoneCarriesTheCallEvenWhenNobodyIsTalking() {
	edge := New(Options{})
	mic := edge.Microphone(stt.Participant{ID: "caller"}, audio.PcmData{})
	defer mic.Stop()

	// A real call carries the room, so an agent waiting for audio to arrive is not left
	// waiting between sentences.
	var packets int
	deadline := time.After(500 * time.Millisecond)
	for packets < 3 {
		select {
		case <-edge.Audio():
			packets++
		case <-deadline:
			s.FailNowf("the call went silent", "got %d packets", packets)
		}
	}
	s.Equal(3, packets)
}

func (s *LoopbackSuite) TestPlayingAnUtteranceReturnsWhenTheSpeakerStoppedTalking() {
	edge := New(Options{Inbound: 256})
	mic := edge.Microphone(stt.Participant{ID: "caller"}, audio.PcmData{})
	defer mic.Stop()

	// Drain, because a microphone whose packets nobody takes stops being paced by the
	// clock and starts being paced by the reader.
	go func() {
		for range edge.Audio() {
		}
	}()

	started := time.Now()
	at := mic.Play(speech(200))

	s.Greater(time.Since(started), 150*time.Millisecond,
		"an utterance is spliced into the call at the rate it is spoken")
	s.False(at.IsZero())
}

func (s *LoopbackSuite) TestAnEdgeThatHasLeftTellsTheAgentTheCallIsOver() {
	edge := New(Options{})

	s.Require().NoError(edge.Leave())

	_, open := <-edge.Audio()
	s.False(open, "the channel closing is what says the call ended")
	// Leaving twice is what a teardown that already happened looks like.
	s.Require().NoError(edge.Leave())
}
