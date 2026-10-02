package socketedge

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type SocketEdgeSuite struct {
	suite.Suite

	edge *Edge

	mu      sync.Mutex
	sent    [][]byte
	cleared int
}

func TestSocketEdgeSuite(t *testing.T) {
	suite.Run(t, new(SocketEdgeSuite))
}

func (s *SocketEdgeSuite) SetupTest() {
	s.sent, s.cleared = nil, 0
	s.edge = New(Options{
		SampleRate: 24_000,
		Caller:     stt.Participant{ID: "caller"},
		Send: func(pcm []byte) error {
			s.mu.Lock()
			s.sent = append(s.sent, pcm)
			s.mu.Unlock()
			return nil
		},
		Cleared: func() {
			s.mu.Lock()
			s.cleared++
			s.mu.Unlock()
		},
	})
	s.Require().NoError(s.edge.Join(context.Background()))
	s.T().Cleanup(func() { _ = s.edge.Leave() })
}

// speech is that much of the agent's voice at the rate a TTS provider hands it over.
func speech(duration time.Duration) audio.PcmData {
	samples := make([]int16, 16_000*int(duration/time.Millisecond)/1000)
	for i := range samples {
		samples[i] = 1000
	}
	return audio.PcmData{Samples: samples, SampleRate: 16_000, Channels: 1}
}

// sentMs is how much speech has gone out, at the socket's 24 kHz.
func (s *SocketEdgeSuite) sentMs() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	total := 0
	for _, frame := range s.sent {
		total += len(frame) / 2
	}
	return total * 1000 / 24_000
}

func (s *SocketEdgeSuite) TestSpeechGoesOutAtThePaceItIsHeard() {
	started := time.Now()
	s.Require().NoError(s.edge.PublishAudio(speech(200 * time.Millisecond)))

	s.Require().Eventually(func() bool { return s.sentMs() >= 200 }, 2*time.Second, 5*time.Millisecond)
	s.GreaterOrEqual(time.Since(started), 180*time.Millisecond, "200 ms of speech sent faster than it is heard")
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Len(s.sent[0], 24_000*20/1000*2, "a frame is 20 ms of 16-bit samples at the socket's rate")
}

func (s *SocketEdgeSuite) TestPublishingWaitsWhileMuchIsAlreadyQueued() {
	started := time.Now()
	s.Require().NoError(s.edge.PublishAudio(speech(time.Second)))

	s.GreaterOrEqual(time.Since(started), 500*time.Millisecond,
		"a second of speech was taken at once, leaving nothing to drop when the caller cuts in")
}

func (s *SocketEdgeSuite) TestDroppingSpeechStopsItWithinAFrame() {
	go func() { _ = s.edge.PublishAudio(speech(2 * time.Second)) }()
	s.Require().Eventually(func() bool { return s.sentMs() >= 100 }, time.Second, 5*time.Millisecond)

	s.edge.DropSpeech()
	at := s.sentMs()
	time.Sleep(200 * time.Millisecond)

	s.LessOrEqual(s.sentMs()-at, 20, "speech went on after it was dropped")
	s.False(s.edge.SpeechPending())
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Equal(1, s.cleared)
}

func (s *SocketEdgeSuite) TestTheCallersAudioReachesTheAgentAt16kHz() {
	frame := audio.PcmData{Samples: make([]int16, 24_000*20/1000), SampleRate: 24_000, Channels: 1}

	s.Require().NoError(s.edge.Hear(frame.Bytes()))

	heard := <-s.edge.Audio()
	s.Equal("caller", heard.Participant.ID)
	s.Equal(stt.SampleRate, heard.Audio.SampleRate)
	s.Len(heard.Audio.Samples, stt.SampleRate*20/1000)
}

func (s *SocketEdgeSuite) TestLeavingEndsTheCallBothWays() {
	s.Require().NoError(s.edge.Leave())

	_, open := <-s.edge.Audio()
	s.False(open, "the agent was not told the call is over")
	s.ErrorIs(s.edge.Hear(make([]byte, 640)), ErrLeft)
	s.ErrorIs(s.edge.PublishAudio(speech(20*time.Millisecond)), ErrLeft)
}
