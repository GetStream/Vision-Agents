//go:build integration

package streamedge

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"os"
	"slices"
	"testing"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"
	"github.com/GetStream/getstream-go-webrtc/coordinator"
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
	ctx      context.Context
	callID   string
	callType string
	// pin is the coordinator options every edge joins with: STREAMEDGE_SFU_ID's SFU.
	pin []coordinator.Option
}

func TestStreamEdgeIntegrationSuite(t *testing.T) {
	suite.Run(t, new(StreamEdgeIntegrationSuite))
}

func (s *StreamEdgeIntegrationSuite) SetupSuite() {
	if os.Getenv("STREAM_API_KEY") == "" || os.Getenv("STREAM_API_SECRET") == "" {
		s.T().Skip("STREAM_API_KEY and STREAM_API_SECRET must be set")
	}
	s.ctx = context.Background()
	// The 3RTT local stack's app has no "agent" call type.
	if os.Getenv("LOCAL_STACK") != "" {
		s.callType = "default"
	}
	if callType := os.Getenv("STREAMEDGE_CALL_TYPE"); callType != "" {
		s.callType = callType
	}
	// A staging app's SFUs are reached only pinned, by the full SFU id.
	if sfuID := os.Getenv("STREAMEDGE_SFU_ID"); sfuID != "" {
		s.pin = []coordinator.Option{coordinator.WithJoinQuery(url.Values{"sfu_id": {sfuID}})}
	}
}

func (s *StreamEdgeIntegrationSuite) SetupTest() {
	// A call of its own per test, so one test's participants cannot be heard by another's.
	s.callID = fmt.Sprintf("go-edge-%d", time.Now().UnixNano())
}

// join puts one participant in the test's call.
func (s *StreamEdgeIntegrationSuite) join(userID string) *Edge {
	edge, err := New(Options{CallID: s.callID, CallType: s.callType, User: User{ID: userID, Name: userID},
		coordinatorOptions: s.pin})
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

func (s *StreamEdgeIntegrationSuite) TestTheAgentsJoinReachesMediaWithinItsRoundTripBudget() {
	// Against the 3RTT local stack with fast join (a coordinator with fast_join and SFUs
	// with FastJoin), with every connection of the agent's given a real network's round
	// trip, so its join is counted in round trips as it would be against a remote
	// deployment.
	if os.Getenv("LOCAL_STACK") == "" {
		s.T().Skip(`needs the 3RTT local stack: eval "$(local-stack.sh env)"`)
	}
	// The first join of a new user and the first call on an SFU pay for setup a running
	// agent has already done.
	s.measureJoin(-1)

	measured := make([]joinMeasurement, 0, joinRuns)
	for run := range joinRuns {
		measured = append(measured, s.measureJoin(run))
	}
	s.record(measured)

	publish := median(measured, func(m joinMeasurement) float64 { return m.PublishToMediaMs })
	subscribe := median(measured, func(m joinMeasurement) float64 { return m.SubscribeToMediaMs })
	rtt := median(measured, func(m joinMeasurement) float64 { return m.RTTMs["sfu"] })
	s.T().Logf("median of %d cold joins at %s: publish to media %.0f ms (%.1f RTT), subscribe to media %.0f ms (%.1f RTT), RTT_s %.1f ms",
		joinRuns, joinRTT, publish, publish/rtt, subscribe, subscribe/rtt, rtt)

	for _, m := range measured {
		s.Equal(string(rtc.JoinFlowFast), m.Flow, "run %d fell back to the legacy join", m.Run)
	}
	budget := joinBudgetRTTs*float64(joinRTT.Milliseconds()) + 30
	s.LessOrEqual(publish, budget, "the agent's audio reaches the SFU within %.1f RTT", joinBudgetRTTs)
	s.LessOrEqual(subscribe, budget, "the caller's audio reaches the agent within %.1f RTT", joinBudgetRTTs)
}

// joinRTT is the round trip the measured agent's connections are given, as in the 3RTT
// benches.
const joinRTT = 100 * time.Millisecond

// joinRuns is how many joins a measurement takes the median of.
const joinRuns = 10

// joinBudgetRTTs bounds the median time to media both ways of a cold agent join, in round
// trips, plus 30 ms. The legacy join took about 14.5.
const joinBudgetRTTs = 12.5

// joinMeasurement is one measured join of the agent: the SDK's join trace, and how long
// Edge.Join took.
type joinMeasurement struct {
	Run                int                `json:"run"`
	Flow               string             `json:"flow"`
	JoinMs             float64            `json:"join_ms"`
	PublishToMediaMs   float64            `json:"publish_to_media_ms"`
	SubscribeToMediaMs float64            `json:"subscribe_to_media_ms"`
	RTTMs              map[string]float64 `json:"rtt_ms"`
	CriticalPath       []string           `json:"critical_path"`
	Trace              json.RawMessage    `json:"trace"`
}

// measureJoin puts a caller who is already talking in a call of its own, then joins the
// agent over a delayed network and waits for its join trace, which the SDK reports once
// media flows both ways. Both sides must then hear each other.
func (s *StreamEdgeIntegrationSuite) measureJoin(run int) joinMeasurement {
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()
	s.callID = fmt.Sprintf("go-edge-%d", time.Now().UnixNano())
	caller := s.join("go-edge-caller")
	defer caller.Leave()
	s.speak(ctx, caller, 48_000)

	agentEdge, err := New(Options{CallID: s.callID, CallType: s.callType, User: User{ID: "go-edge-agent", Name: "go-edge-agent"},
		clientOptions: []rtc.Option{rtc.WithNetworkDelay(joinRTT)}})
	s.Require().NoError(err)
	started := time.Now()
	s.Require().NoError(agentEdge.Join(s.ctx))
	joinMs := float64(time.Since(started).Microseconds()) / 1000
	defer agentEdge.Leave()
	s.speak(ctx, agentEdge, 24_000)

	var joined agent.JoinTrace
	select {
	case joined = <-agentEdge.JoinTraces():
	case <-time.After(rtc.JoinTraceTimeout + 5*time.Second):
		s.FailNow("the agent reported no join trace")
	}
	var report struct {
		PublishToMediaMs   *float64           `json:"publish_to_media_ms"`
		SubscribeToMediaMs *float64           `json:"subscribe_to_media_ms"`
		RTTMs              map[string]float64 `json:"rtt_ms"`
		CriticalPath       []string           `json:"critical_path"`
	}
	s.Require().NoError(json.Unmarshal(joined.Trace, &report))
	s.Require().NotNil(report.PublishToMediaMs, "the agent's audio never reached the SFU")
	s.Require().NotNil(report.SubscribeToMediaMs, "the caller's audio never reached the agent")

	s.Equal("go-edge-agent", s.hear(caller).Participant.UserID)
	s.Equal("go-edge-caller", s.hear(agentEdge).Participant.UserID)
	return joinMeasurement{
		Run:                run,
		Flow:               joined.Flow,
		JoinMs:             joinMs,
		PublishToMediaMs:   *report.PublishToMediaMs,
		SubscribeToMediaMs: *report.SubscribeToMediaMs,
		RTTMs:              report.RTTMs,
		CriticalPath:       report.CriticalPath,
		Trace:              joined.Trace,
	}
}

// record writes the measurements as JSON lines to STREAMEDGE_JOIN_OUT, when it is set.
func (s *StreamEdgeIntegrationSuite) record(measured []joinMeasurement) {
	path := os.Getenv("STREAMEDGE_JOIN_OUT")
	if path == "" {
		return
	}
	out, err := os.Create(path)
	s.Require().NoError(err)
	defer out.Close()
	encoder := json.NewEncoder(out)
	for _, m := range measured {
		s.Require().NoError(encoder.Encode(m))
	}
}

func median(measured []joinMeasurement, value func(joinMeasurement) float64) float64 {
	values := make([]float64, 0, len(measured))
	for _, m := range measured {
		values = append(values, value(m))
	}
	slices.Sort(values)
	middle := len(values) / 2
	if len(values)%2 == 0 {
		return (values[middle-1] + values[middle]) / 2
	}
	return values[middle]
}

func (s *StreamEdgeIntegrationSuite) TestLeavingClosesTheAudio() {
	edge := s.join("go-edge-leaver")

	s.Require().NoError(edge.Leave())

	_, open := <-edge.Audio()
	s.False(open, "the channel closes so the agent's range loop ends")
	s.NoError(edge.Leave(), "leaving twice is safe")
	s.Error(edge.PublishAudio(tone(16_000)), "there is nowhere left to publish")
}
