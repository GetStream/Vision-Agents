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
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
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
	edge, err := New(Options{CallID: s.callID, User: User{ID: userID, Name: userID},
		APIKey: os.Getenv("STREAM_API_KEY"), APISecret: os.Getenv("STREAM_API_SECRET")})
	s.Require().NoError(err)
	s.Require().NoError(edge.Join(s.ctx))
	s.T().Cleanup(func() { _ = edge.Leave() })
	return edge
}

func (s *StreamEdgeIntegrationSuite) TestTheAgentsConnectionComingUpEndsItsStartUp() {
	// Catches the publisher's connection coming up and the media stage of the start-up the
	// edge joined with never timed, timed again by the subscriber's, or timed from another
	// instant than the connection's, such as the start of the join, which reads as no wait.
	timed, waited, failed, err := mediaStartups()
	s.Require().NoError(err)
	startup := agent.NewStartup(agent.StartupPhone, time.Now())
	edge, err := New(Options{CallID: s.callID, User: User{ID: "agent", Name: "agent"},
		APIKey: os.Getenv("STREAM_API_KEY"), APISecret: os.Getenv("STREAM_API_SECRET")})
	s.Require().NoError(err)
	startup.JoinStarted(time.Now())
	s.Require().NoError(edge.Join(agent.WithStartup(s.ctx, startup)))
	joined := time.Now()
	startup.JoinEnded(joined)
	s.T().Cleanup(func() { _ = edge.Leave() })

	s.Eventually(func() bool {
		now, _, _, err := mediaStartups()
		return err == nil && now == timed+1
	}, audioArrivesWithin, 50*time.Millisecond, "the connection coming up was not timed")
	since := time.Since(joined)
	now, waitedNow, failedNow, err := mediaStartups()
	s.Require().NoError(err)
	s.Equal(timed+1, now)
	s.Equal(failed, failedNow)
	// The publisher's connection comes up hundreds of ms after the join (its connected_ms
	// less join_ms in the edge's logs).
	s.Greater(waitedNow-waited, 0.0, "the media stage was timed at no wait")
	s.LessOrEqual(waitedNow-waited, since.Seconds(), "the media stage began before the join ended")
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

// hearMany waits for several stretches of speech, so a test can say who was heard over a
// while rather than who happened to arrive first.
func (s *StreamEdgeIntegrationSuite) hearMany(edge *Edge, count int) []agent.InboundAudio {
	heard := make([]agent.InboundAudio, 0, count)
	for range count {
		heard = append(heard, s.hear(edge))
	}
	return heard
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

	// Catches the media quality of a real call not reaching the metrics: the round trip read
	// from the publisher at Leave (an edge without a call has none), the SFU's receiver reports
	// on the agent's own audio, and the inbound jitter taken with the track's clock rate.
	for _, edge := range []*Edge{first, second} {
		s.Eventually(func() bool { return edge.quality.summary(nil).outboundLoss != nil },
			audioArrivesWithin, 100*time.Millisecond, "the SFU reported on the agent's audio")
	}
	before := mediaSeries()
	s.Require().NoError(first.Leave())
	s.Require().NoError(second.Leave())
	after := mediaSeries()
	for _, series := range []string{"router_agent_media_rtt_seconds/", "router_agent_media_packet_loss_ratio/outbound",
		"router_agent_media_jitter_seconds/outbound", "router_agent_media_jitter_seconds/inbound",
		"router_agent_media_packet_loss_ratio/inbound"} {
		s.Equal(before[series].count+2, after[series].count, series+": one value per edge")
	}
	rtt := "router_agent_media_rtt_seconds/"
	s.Positive(after[rtt].sum-before[rtt].sum, "a round trip took time")
	jitter := "router_agent_media_jitter_seconds/inbound"
	s.Less((after[jitter].sum-before[jitter].sum)/2, 0.02,
		"jitter in seconds; the Opus clock taken as 16 kHz reads every 20 ms step as 40 ms of jitter")
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

func (s *StreamEdgeIntegrationSuite) TestAnotherInstanceOfTheAgentIsNotHeard() {
	// An agent that was restarted without leaving is still in the call, publishing under
	// the same user id. Heard as a caller, the two answer each other until nobody else can
	// get a word in.
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	agentEdge := s.join("go-edge-twin")
	twin := s.join("go-edge-twin")
	caller := s.join("go-edge-outsider")

	s.speak(ctx, twin, 24_000)
	s.speak(ctx, caller, 48_000)

	// Hearing the outsider at all proves the edge is subscribed and decoding, so the twin
	// being absent from what arrived is the guard rather than a call carrying nothing.
	for _, heard := range s.hearMany(agentEdge, 5) {
		s.Equal("go-edge-outsider", heard.Participant.UserID,
			"the agent heard itself, which is the two of them talking in a loop")
	}
}

func (s *StreamEdgeIntegrationSuite) TestLeavingClosesTheAudio() {
	edge := s.join("go-edge-leaver")

	s.Require().NoError(edge.Leave())

	_, open := <-edge.Audio()
	s.False(open, "the channel closes so the agent's range loop ends")
	s.NoError(edge.Leave(), "leaving twice is safe")
	s.Error(edge.PublishAudio(tone(16_000)), "there is nowhere left to publish")
}
