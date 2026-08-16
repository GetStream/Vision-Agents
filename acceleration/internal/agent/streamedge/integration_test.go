//go:build integration

package streamedge

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// audioArrivesWithin bounds how long a real call is given to carry the first speech. It
// covers joining, negotiating and subscribing, not just the media.
const audioArrivesWithin = 30 * time.Second

type StreamEdgeIntegrationSuite struct {
	suite.Suite
	ctx    context.Context
	callID string
}

func TestStreamEdgeIntegrationSuite(t *testing.T) {
	suite.Run(t, new(StreamEdgeIntegrationSuite))
}

func (s *StreamEdgeIntegrationSuite) SetupSuite() {
	if os.Getenv("STREAM_API_KEY") == "" || os.Getenv("STREAM_API_SECRET") == "" {
		s.T().Skip("STREAM_API_KEY and STREAM_API_SECRET must be set")
	}
	s.ctx = context.Background()
}

func (s *StreamEdgeIntegrationSuite) SetupTest() {
	// A call of its own per test, so one test's participants cannot be heard by another's.
	s.callID = fmt.Sprintf("go-edge-%d", time.Now().UnixNano())
}

// join puts one participant in the test's call.
func (s *StreamEdgeIntegrationSuite) join(userID string) *Edge {
	edge, err := New(Options{CallID: s.callID, User: User{ID: userID, Name: userID}})
	s.Require().NoError(err)
	s.Require().NoError(edge.Join(s.ctx))
	s.T().Cleanup(func() { _ = edge.Leave() })
	return edge
}

// tone is a second of 440 Hz at the rate a voice provider produces, which is loud enough to
// tell from the silence a published track sends when nobody is talking.
func tone(sampleRate int) audio.PcmData {
	samples := make([]int16, sampleRate)
	for i := range samples {
		samples[i] = int16(8000 * math.Sin(2*math.Pi*440*float64(i)/float64(sampleRate)))
	}
	return audio.PcmData{Samples: samples, SampleRate: sampleRate, Channels: 1}
}

// speak publishes a tone repeatedly until the test stops, because a call only carries audio
// while somebody is talking.
func (s *StreamEdgeIntegrationSuite) speak(ctx context.Context, edge *Edge, sampleRate int) {
	go func() {
		speech := tone(sampleRate)
		for ctx.Err() == nil {
			if err := edge.PublishAudio(speech); err != nil {
				return
			}
		}
	}()
}

// hear waits for speech loud enough to have been the tone rather than silence.
func (s *StreamEdgeIntegrationSuite) hear(edge *Edge) agent.InboundAudio {
	deadline := time.After(audioArrivesWithin)
	for {
		select {
		case inbound, open := <-edge.Audio():
			if !open {
				s.FailNow("the edge left before any audio arrived")
			}
			if loudest(inbound.Audio) > 500 {
				return inbound
			}
		case <-deadline:
			s.FailNowf("no audio", "nothing was heard within %s", audioArrivesWithin)
			return agent.InboundAudio{}
		}
	}
}

func loudest(pcm audio.PcmData) int {
	var peak int
	for _, sample := range pcm.Samples {
		level := int(sample)
		if level < 0 {
			level = -level
		}
		peak = max(peak, level)
	}
	return peak
}

func (s *StreamEdgeIntegrationSuite) TestAudioFlowsBothWaysInARealCall() {
	// Two edges in one call is the whole path: PCM is encoded to Opus, published, forwarded
	// by the SFU, subscribed to, decoded and resampled back to what the providers accept.
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	first := s.join("go-edge-first")
	second := s.join("go-edge-second")

	s.speak(ctx, first, 24_000)
	s.speak(ctx, second, 48_000)

	fromFirst := s.hear(second)
	s.Equal("go-edge-first", fromFirst.Participant.UserID)
	s.Equal(stt.SampleRate, fromFirst.Audio.SampleRate,
		"the agent is handed the rate every speech-to-text provider accepts")
	s.Equal(1, fromFirst.Audio.Channels)

	fromSecond := s.hear(first)
	s.Equal("go-edge-second", fromSecond.Participant.UserID)
	s.Equal(stt.SampleRate, fromSecond.Audio.SampleRate)
}

func (s *StreamEdgeIntegrationSuite) TestSomeoneJoiningLaterIsHeard() {
	// Nobody is publishing when the first edge joins, so this only works if the edge
	// subscribes to what is published after it arrived.
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	listener := s.join("go-edge-listener")
	time.Sleep(time.Second)

	talker := s.join("go-edge-latecomer")
	s.speak(ctx, talker, 16_000)

	heard := s.hear(listener)
	s.Equal("go-edge-latecomer", heard.Participant.UserID)
}

func (s *StreamEdgeIntegrationSuite) TestLeavingClosesTheAudio() {
	edge := s.join("go-edge-leaver")

	s.Require().NoError(edge.Leave())

	_, open := <-edge.Audio()
	s.False(open, "the channel closes so the agent's range loop ends")
	s.NoError(edge.Leave(), "leaving twice is safe")
	s.Error(edge.PublishAudio(tone(16_000)), "there is nowhere left to publish")
}
