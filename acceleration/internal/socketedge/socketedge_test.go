package socketedge

import (
	"context"
	"math"
	"slices"
	"strconv"
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
	edge, err := New(Options{
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
	s.Require().NoError(err)
	s.edge = edge
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

// The edge is one call, held once: joining it again after leaving would start a second
// playout on a call already hung up.
func (s *SocketEdgeSuite) TestAnEdgeThatHasLeftIsNotJoinedAgain() {
	s.False(s.edge.Left())
	s.Require().NoError(s.edge.Leave())

	s.ErrorIs(s.edge.Join(context.Background()), ErrLeft)
	s.True(s.edge.Left())
	s.ErrorIs(s.edge.PublishAudio(speech(20*time.Millisecond)), ErrLeft)
}

// Joining a call already joined changes nothing: there is still one playout, and leaving
// stops it once.
func (s *SocketEdgeSuite) TestJoiningTwiceIsOneCall() {
	s.Require().NoError(s.edge.Join(context.Background()))

	s.NoError(s.edge.Leave())
	s.True(s.edge.Left())
}

// unjoined is a 16 kHz edge that sends nowhere and has not been joined.
func (s *SocketEdgeSuite) unjoined() *Edge {
	edge, err := New(Options{Send: func([]byte) error { return nil }})
	s.Require().NoError(err)
	return edge
}

// An edge left before it was joined has no playout to wait for: leaving returns.
func (s *SocketEdgeSuite) TestLeavingAnEdgeNeverJoinedReturns() {
	edge := s.unjoined()
	left := make(chan error, 1)

	go func() { left <- edge.Leave() }()

	select {
	case err := <-left:
		s.NoError(err)
	case <-time.After(2 * time.Second):
		s.Fail("Leave waits for a playout that never started")
	}
	s.ErrorIs(edge.Join(context.Background()), ErrLeft)
}

// A voice started while the same voice is stopped joins, asks and leaves the edge at once:
// whichever wins, the call is held at most once and hung up once.
func (s *SocketEdgeSuite) TestJoinRacingLeaveHangsUpOnce() {
	for range 200 {
		edge := s.unjoined()
		left := make(chan error, 1)
		var wg sync.WaitGroup
		wg.Add(3)
		go func() { defer wg.Done(); _ = edge.Join(context.Background()) }()
		go func() { defer wg.Done(); _ = edge.Join(context.Background()) }()
		go func() { defer wg.Done(); _ = edge.Left() }()
		go func() { left <- edge.Leave() }()
		wg.Wait()

		select {
		case err := <-left:
			s.Require().NoError(err)
		case <-time.After(2 * time.Second):
			s.Require().Fail("Leave hangs")
		}
		s.Require().True(edge.Left())
		s.Require().ErrorIs(edge.Join(context.Background()), ErrLeft)
	}
}

// The socket's read loop hears frames while a DELETE voice leaves on another goroutine: a
// frame heard as the call ends is refused, never sent on the closed audio channel.
func (s *SocketEdgeSuite) TestHearRacingLeaveNeverSendsOnAClosedChannel() {
	for i := range 3000 {
		edge := s.unjoined()
		if i%2 == 0 {
			s.Require().NoError(edge.Join(context.Background()))
		}
		drained := make(chan struct{})
		go func() {
			for range edge.Audio() {
			}
			close(drained)
		}()
		var wg sync.WaitGroup
		for range 4 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for edge.Hear(make([]byte, 640)) == nil {
				}
			}()
		}
		s.Require().NoError(edge.Leave())
		wg.Wait()
		<-drained
	}
}

// A frame waiting for room on the audio channel, which nothing reads, is released by leaving,
// so Leave returns whether or not the edge was joined.
func (s *SocketEdgeSuite) TestLeavingReleasesAHearWaitingToSend() {
	for _, join := range []bool{true, false} {
		edge := s.unjoined()
		if join {
			s.Require().NoError(edge.Join(context.Background()))
		}
		heard := make(chan error, 1)
		go func() {
			for {
				if err := edge.Hear(make([]byte, 640)); err != nil {
					heard <- err
					return
				}
			}
		}()
		s.Require().Eventually(func() bool { return len(edge.inbound) == cap(edge.inbound) }, 2*time.Second, time.Millisecond)

		left := make(chan error, 1)
		go func() { left <- edge.Leave() }()
		select {
		case err := <-left:
			s.Require().NoError(err)
		case <-time.After(2 * time.Second):
			s.Require().Fail("Leave waits for a Hear that is never released")
		}
		s.Require().ErrorIs(<-heard, ErrLeft)
	}
}

// carriedRates are the rates a caller's audio comes in at, from a phone line's 8 kHz to
// WebRTC's 48 kHz.
var carriedRates = []int{8_000, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000}

// edgeAt is an edge joined at the rate given, and what it has sent so far.
func (s *SocketEdgeSuite) edgeAt(rate int) (*Edge, func() [][]byte) {
	var mu sync.Mutex
	var sent [][]byte
	edge, err := New(Options{
		SampleRate: rate,
		Caller:     stt.Participant{ID: "caller"},
		Send: func(pcm []byte) error {
			mu.Lock()
			sent = append(sent, pcm)
			mu.Unlock()
			return nil
		},
	})
	s.Require().NoError(err)
	s.Require().NoError(edge.Join(context.Background()))
	s.T().Cleanup(func() { _ = edge.Leave() })
	return edge, func() [][]byte {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(sent)
	}
}

// The control: 20 ms of the caller at any rate the socket carries reaches the agent as 20 ms
// at 16 kHz, and the agent's speech goes out in 20 ms frames at the socket's rate.
func (s *SocketEdgeSuite) TestEveryCarriedRateIsHeardAndSpokenIn20msFrames() {
	for _, rate := range carriedRates {
		s.Run(strconv.Itoa(rate), func() {
			edge, sent := s.edgeAt(rate)
			frame := audio.PcmData{Samples: make([]int16, rate*20/1000), SampleRate: rate, Channels: 1}

			s.Require().NoError(edge.Hear(frame.Bytes()))
			heard := <-edge.Audio()
			s.Require().NoError(edge.PublishAudio(speech(20 * time.Millisecond)))

			s.Len(heard.Audio.Samples, stt.SampleRate*20/1000)
			s.Require().Eventually(func() bool { return len(sent()) > 0 }, 2*time.Second, 5*time.Millisecond)
			s.Len(sent()[0], rate*20/1000*2, "a frame is 20 ms of 16-bit samples at the socket's rate")
		})
	}
}

// A rate the socket does not carry is refused before any frame is resampled at it, and zero
// still means 16 kHz.
func (s *SocketEdgeSuite) TestNewRefusesARateTheSocketDoesNotCarry() {
	for _, rate := range []int{1, 7_999, 11_025, 48_001, 96_000, math.MaxInt32, math.MaxInt, -1, -16_000, math.MinInt} {
		s.Run(strconv.Itoa(rate), func() {
			edge, err := New(Options{SampleRate: rate, Send: func([]byte) error { return nil }})

			s.Error(err)
			s.Nil(edge)
		})
	}
	edge := s.unjoined()
	s.Require().NoError(edge.Hear(make([]byte, stt.SampleRate*20/1000*2)))
	s.Len((<-edge.Audio()).Audio.Samples, stt.SampleRate*20/1000, "20 ms at 16 kHz is heard as 20 ms")
}
