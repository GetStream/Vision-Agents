package streamedge

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"math"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/getstream-go-webrtc/jointrace"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

type StreamEdgeSuite struct {
	suite.Suite
	ctx context.Context
}

func TestStreamEdgeSuite(t *testing.T) {
	suite.Run(t, new(StreamEdgeSuite))
}

func (s *StreamEdgeSuite) SetupTest() {
	s.ctx = context.Background()
	// The credentials are read from the environment, so a machine that has them must not
	// change what these tests mean.
	s.T().Setenv("STREAM_API_KEY", "")
	s.T().Setenv("STREAM_API_SECRET", "")
	s.T().Setenv("STREAM_USER_TOKEN", "")
}

// speech returns a tone at the given rate, which is what a voice provider hands over.
func speech(sampleRate int, durationMs int) audio.PcmData {
	samples := make([]int16, sampleRate*durationMs/1000)
	for i := range samples {
		samples[i] = int16(8000 * math.Sin(2*math.Pi*440*float64(i)/float64(sampleRate)))
	}
	return audio.PcmData{Samples: samples, SampleRate: sampleRate, Channels: 1}
}

// drain takes every frame the speaker has ready, stopping at the first silence.
func (s *StreamEdgeSuite) drain(talker *speaker) [][]byte {
	var frames [][]byte
	for {
		sample, err := talker.NextSample(s.ctx)
		s.Require().NoError(err)
		if bytes.Equal(sample.Data, silenceFrame) {
			return frames
		}
		s.Equal(opusFrameDuration, sample.Duration, "every frame is one Opus frame long")
		frames = append(frames, sample.Data)
	}
}

func (s *StreamEdgeSuite) TestACallIDIsRequired() {
	_, err := New(Options{User: User{ID: "agent"}})

	s.ErrorContains(err, "call id")
}

func (s *StreamEdgeSuite) TestCredentialsAreRequired() {
	_, err := New(Options{CallID: "demo", User: User{ID: "agent"}})

	s.ErrorContains(err, "STREAM_API_KEY")
}

func (s *StreamEdgeSuite) TestATokenOrASecretIsRequired() {
	s.T().Setenv("STREAM_API_KEY", "key")

	_, err := New(Options{CallID: "demo", User: User{ID: "agent"}})

	s.ErrorContains(err, "STREAM_USER_TOKEN")
}

func (s *StreamEdgeSuite) TestCredentialsComeFromTheEnvironment() {
	s.T().Setenv("STREAM_API_KEY", "key")
	s.T().Setenv("STREAM_API_SECRET", "secret")

	edge, err := New(Options{CallID: "demo", User: User{ID: "agent"}})

	s.Require().NoError(err)
	s.Equal("agent", edge.options.CallType, "the call type an agent joins unless one is named")
}

func (s *StreamEdgeSuite) TestWithoutARegionTheCoordinatorPlacesTheAgent() {
	s.T().Setenv("STREAM_REGION", "")

	edge, err := New(Options{CallID: "demo", User: User{ID: "agent"}, APIKey: "key", APISecret: "secret"})

	s.Require().NoError(err)
	s.Equal("auto", edge.location)
}

func (s *StreamEdgeSuite) TestTheRegionBecomesTheNearestAirport() {
	for region, want := range map[string]string{
		"us-east1":  "CHS",
		"us-east-1": "IAD",
		"eu-west-1": "DUB",
		"ams":       "AMS",
		"mars-1":    "auto",
	} {
		edge, err := New(Options{CallID: "demo", User: User{ID: "agent"}, APIKey: "key", APISecret: "secret", Region: region})

		s.Require().NoError(err)
		s.Equal(want, edge.location, region)
	}
}

func (s *StreamEdgeSuite) TestTheRegionComesFromTheEnvironment() {
	s.T().Setenv("STREAM_REGION", "europe-west4")

	edge, err := New(Options{CallID: "demo", User: User{ID: "agent"}, APIKey: "key", APISecret: "secret"})

	s.Require().NoError(err)
	s.Equal("AMS", edge.location)
}

func (s *StreamEdgeSuite) TestTheDemoLinkJoinsTheAgentsCall() {
	s.T().Setenv("STREAM_API_KEY", "key")
	s.T().Setenv("STREAM_API_SECRET", "secret")
	// A developer whose own environment points the demo somewhere else is not what this is
	// about.
	s.T().Setenv("EXAMPLE_BASE_URL", "")
	edge, err := New(Options{CallID: "my call", User: User{ID: "agent"}})
	s.Require().NoError(err)

	link, err := edge.DemoURL(User{ID: "demo-caller"})

	s.Require().NoError(err)
	parsed, err := url.Parse(link)
	s.Require().NoError(err)
	s.Equal("https://getstream.io/video/demos/join/my%20call", parsed.Scheme+"://"+parsed.Host+parsed.EscapedPath())
	s.Equal("key", parsed.Query().Get("api_key"))
	s.Equal("true", parsed.Query().Get("skip_lobby"), "the caller should land in the call, not a lobby")
	s.Equal("demo-caller", parsed.Query().Get("user_name"), "an unnamed caller is named after their id")
	s.NotEmpty(parsed.Query().Get("token"), "the browser joins as somebody the app trusts")
}

func (s *StreamEdgeSuite) TestTheDemoLinkCanPointAtAnotherDeployment() {
	s.T().Setenv("STREAM_API_KEY", "key")
	s.T().Setenv("STREAM_API_SECRET", "secret")
	s.T().Setenv("EXAMPLE_BASE_URL", "https://pronto.getstream.io/")
	edge, err := New(Options{CallID: "demo", User: User{ID: "agent"}})
	s.Require().NoError(err)

	link, err := edge.DemoURL(User{ID: "demo-caller", Name: "Demo caller"})

	s.Require().NoError(err)
	s.True(strings.HasPrefix(link, "https://pronto.getstream.io/join/demo?"), link)
}

func (s *StreamEdgeSuite) TestADemoLinkNeedsASecretToSignAToken() {
	s.T().Setenv("STREAM_API_KEY", "key")
	edge, err := New(Options{CallID: "demo", User: User{ID: "agent"}, UserToken: "token"})
	s.Require().NoError(err)

	_, err = edge.DemoURL(User{ID: "demo-caller"})

	s.ErrorContains(err, "STREAM_API_SECRET")
}

func (s *StreamEdgeSuite) TestSpeechIsEncodedToOpusFrames() {
	// This is the outbound path: the voice's PCM becomes the 20 ms Opus frames the track
	// sends.
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })

	s.Require().NoError(talker.Write(speech(opusSampleRate, 100)))

	frames := s.drain(talker)
	s.Require().Len(frames, 5, "100 ms of speech is five frames")
	for _, frame := range frames {
		s.NotEmpty(frame)
		s.NotEqual(silenceFrame, frame, "the tone is not silence")
	}
}

func (s *StreamEdgeSuite) TestSilenceIsSentWhenThereIsNothingToSay() {
	// The track is published for the whole call, so something has to go out even when the
	// agent is listening rather than talking.
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })

	sample, err := talker.NextSample(s.ctx)

	s.Require().NoError(err)
	s.Equal(silenceFrame, sample.Data)
	s.Equal(opusFrameDuration, sample.Duration)
	s.EqualValues(audioLevelSilent, talker.CurrentAudioLevel())
}

func (s *StreamEdgeSuite) TestTheAudioLevelSaysWhoIsTalking() {
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })
	s.Require().NoError(talker.Write(speech(48_000, 40)))

	_, err := talker.NextSample(s.ctx)
	s.Require().NoError(err)

	s.EqualValues(audioLevelSpeaking, talker.CurrentAudioLevel(),
		"the other participants' clients show the agent as the speaker")
}

// replyChunks is one utterance as a voice provider streams it: a long chunk while the
// provider is ahead of the speaking, then shorter ones as it catches up. The sizes vary
// because that is what makes the end of a reply go missing -- a pipeline that sizes what it
// holds back off the largest chunk it has seen keeps hold of every smaller one after it.
// They add up to less than the playout bound so that writing them does not wait on the track.
var replyChunks = []int{200, 60, 40, 40}

func (s *StreamEdgeSuite) TestTheEndOfAnUtteranceReachesTheCall() {
	// The track carries 48 kHz Opus in whole 20 ms frames, so the end of an utterance that
	// does not land on a frame boundary has nowhere to go until more audio arrives. What is
	// held back is the last thing the caller was meant to hear, and an utterance is the last
	// one before a silence, so it is not made good by the next reply.
	//
	// Which rate the voice speaks at is its own business, and a failover mid-call can change
	// it, so no rate may be the one that loses the end of a reply.
	for _, rate := range []int{16_000, 22_050, 24_000, 44_100, 48_000} {
		talker := newSpeaker(slog.New(slog.DiscardHandler))

		spokenMs := 0
		for _, chunkMs := range replyChunks {
			s.Require().NoError(talker.Write(speech(rate, chunkMs)))
			spokenMs += chunkMs
		}

		heardMs := len(s.drain(talker)) * 20
		s.GreaterOrEqualf(heardMs, spokenMs,
			"at %d Hz only %dms of %dms reached the call", rate, heardMs, spokenMs)
		s.Require().NoError(talker.Close())
	}
}

func (s *StreamEdgeSuite) TestSpeechNotHeardYetIsThrownAwayOnBargeIn() {
	// When the caller takes the floor the agent has to stop being heard, and cancelling the
	// voice only stops what it has not synthesised yet. What it already sent is queued here,
	// and the caller is talked over for as long as that queue is deep.
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })
	s.Require().NoError(talker.Write(speech(24_000, 200)))
	_, err := talker.NextSample(s.ctx)
	s.Require().NoError(err)

	talker.drop()

	s.EqualValues(audioLevelSilent, talker.CurrentAudioLevel(),
		"the other participants' clients still show the agent as talking")
	sample, err := talker.NextSample(s.ctx)
	s.Require().NoError(err)
	s.Equal(silenceFrame, sample.Data, "the abandoned reply is still being heard")
	s.False(talker.pending(), "nothing of the abandoned reply is still waiting to go out")
}

func (s *StreamEdgeSuite) TestTheTailOfAnAbandonedReplyIsNotHeardOnTheNextOne() {
	// Emptying the queue is not enough on its own: what the encoder is holding belongs to
	// the reply being abandoned too, and would otherwise be the first thing the caller hears
	// of the next one.
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })
	_, err := talker.NextSample(s.ctx)
	s.Require().NoError(err)
	s.Require().NoError(talker.Write(speech(24_000, 170)))

	talker.drop()
	s.Require().NoError(talker.Write(speech(24_000, 100)))

	heardMs := len(s.drain(talker)) * 20
	// The flush that ends the reply pads its last part-frame out to a whole one.
	s.LessOrEqual(heardMs, 100+20,
		"the abandoned reply was heard at the start of the next one")
}

func (s *StreamEdgeSuite) TestSpeechNotHeardYetIsReportedAsWaiting() {
	// A voice streams a reply far faster than it is spoken, so when the provider says it
	// has finished, most of the reply is still queued here. Leaving the call throws that
	// queue away, so whoever is about to leave has to be able to ask whether it is empty:
	// without this the caller hears the agent stop short of its last words.
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })

	// Until the track asks for a frame nothing is draining the queue, so what is held is
	// reported as settled rather than leaving a caller waiting on a queue that cannot move.
	s.False(talker.pending(), "a track that is not taking audio has nothing to wait for")
	_, err := talker.NextSample(s.ctx)
	s.Require().NoError(err)

	s.Require().NoError(talker.Write(speech(24_000, 200)))
	s.True(talker.pending(), "the reply has not reached the track yet")

	s.drain(talker)
	s.False(talker.pending(), "the whole reply has gone out")
}

func (s *StreamEdgeSuite) TestLeavingWhileSpeakingIsNotReportedAsWaiting() {
	// Leaving discards the queue, so a caller waiting for it to empty would wait for
	// something that is never going to happen.
	talker := newSpeaker(slog.New(slog.DiscardHandler))

	_, err := talker.NextSample(s.ctx)
	s.Require().NoError(err)
	s.Require().NoError(talker.Write(speech(24_000, 200)))
	s.Require().True(talker.pending())

	s.Require().NoError(talker.Close())

	s.False(talker.pending(), "there is nothing left to be heard once the call is left")
}

func (s *StreamEdgeSuite) TestAnyInputRateIsResampled() {
	// A voice provider's rate is its own business, and a failover mid-call can change it.
	// The resampler holds the first chunk back by design, so speech is written twice here
	// the way a stream of chunks would arrive.
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })

	for _, rate := range []int{16_000, 24_000} {
		s.Require().NoError(talker.Write(speech(rate, 40)))
		s.Require().NoError(talker.Write(speech(rate, 40)))

		s.NotEmptyf(s.drain(talker), "speech at %d Hz never reached the track", rate)
	}
}

func (s *StreamEdgeSuite) TestPublishingIsPacedByWhatIsHeard() {
	// A voice provider streams an utterance far faster than it is spoken. Without this the
	// whole reply would be queued in a moment, and barge-in would arrive too late to stop
	// it.
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })

	written := make(chan error, 1)
	go func() { written <- talker.Write(speech(48_000, 1_000)) }()

	select {
	case <-written:
		s.Fail("a second of speech should not be accepted all at once")
	case <-time.After(100 * time.Millisecond):
	}

	s.NotEmpty(s.drain(talker), "draining the queue lets the rest of the utterance in")
	s.Require().Eventually(func() bool {
		s.drain(talker)
		select {
		case err := <-written:
			s.Require().NoError(err)
			return true
		default:
			return false
		}
	}, 5*time.Second, 10*time.Millisecond, "the utterance never finished being published")
}

func (s *StreamEdgeSuite) TestOnlyMonoSpeechIsAccepted() {
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })

	err := talker.Write(audio.PcmData{Samples: make([]int16, 100), SampleRate: 48_000, Channels: 2})

	s.ErrorContains(err, "mono")
}

func (s *StreamEdgeSuite) TestEmptySpeechIsIgnored() {
	talker := newSpeaker(slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() { _ = talker.Close() })

	s.NoError(talker.Write(audio.PcmData{SampleRate: 48_000, Channels: 1}))
	s.Empty(s.drain(talker))
}

func (s *StreamEdgeSuite) TestLeavingPartWayThroughAFrameIsNotAFailure() {
	// The queue is thrown away on leaving, so the part-frame the encoder is still holding
	// was never going to be heard. The encoder only takes whole frames and says so, and
	// reporting that as a failure makes every call look like it ended badly.
	talker := newSpeaker(slog.New(slog.DiscardHandler))

	// 33 ms is not a whole number of 20 ms frames, at either rate.
	s.Require().NoError(talker.Write(speech(24_000, 33)))

	s.NoError(talker.Close())
}

func (s *StreamEdgeSuite) TestPublishingAfterLeavingFails() {
	talker := newSpeaker(slog.New(slog.DiscardHandler))

	s.Require().NoError(talker.Close())

	s.ErrorContains(talker.Write(speech(48_000, 20)), "left")
	s.NoError(talker.Close(), "closing twice is safe")
}

// joinTimeline is a first join as the SDK records it: the coordinator, the SFU, then the
// publish and subscribe branches in parallel, at 50 ms to every peer.
func joinTimeline(started time.Time) jointrace.Trace {
	at := func(ms float64) time.Time { return started.Add(time.Duration(ms * float64(time.Millisecond))) }
	rec := jointrace.NewRecorder(started)
	rec.SetRTT(jointrace.PeerCoordinator, 50*time.Millisecond)
	rec.SetRTT(jointrace.PeerSFU, 50*time.Millisecond)
	rec.SetRTT(jointrace.PeerUDP, 50*time.Millisecond)
	step := func(name string, after []string, from, to float64, kind jointrace.Kind, peer jointrace.Peer) {
		rec.Add(jointrace.Span{Name: name, After: after, Start: at(from), End: at(to), Kind: kind, Peer: peer})
	}
	step(jointrace.CoordJoin, nil, 0, 200, jointrace.KindNet, jointrace.PeerCoordinator)
	step(jointrace.PCsCreate, []string{jointrace.CoordJoin}, 200, 202, jointrace.KindLocal, jointrace.PeerLocal)
	step(jointrace.SFUWSDial, []string{jointrace.PCsCreate}, 202, 352, jointrace.KindNet, jointrace.PeerSFU)
	step(jointrace.SFUJoin, []string{jointrace.SFUWSDial}, 352, 402, jointrace.KindNet, jointrace.PeerSFU)
	step(jointrace.PubDebounce, []string{jointrace.SFUJoin}, 402, 403, jointrace.KindTimer, jointrace.PeerLocal)
	step(jointrace.PubOffer, []string{jointrace.PubDebounce}, 403, 405, jointrace.KindLocal, jointrace.PeerLocal)
	step(jointrace.PubSetPublisher, []string{jointrace.PubOffer}, 405, 455, jointrace.KindNet, jointrace.PeerSFU)
	step(jointrace.PubICE, []string{jointrace.PubSetPublisher}, 456, 506, jointrace.KindNet, jointrace.PeerUDP)
	step(jointrace.PubDTLS, []string{jointrace.PubICE}, 506, 556, jointrace.KindNet, jointrace.PeerUDP)
	step(jointrace.PubRTP, []string{jointrace.PubDTLS}, 556, 560, jointrace.KindLocal, jointrace.PeerLocal)
	step(jointrace.SubDebounce, []string{jointrace.SFUJoin}, 402, 480, jointrace.KindTimer, jointrace.PeerSFU)
	step(jointrace.SubOffer, []string{jointrace.SubDebounce}, 480, 505, jointrace.KindNet, jointrace.PeerSFU)
	step(jointrace.SubSendAnswer, []string{jointrace.SubOffer}, 505, 556, jointrace.KindNet, jointrace.PeerSFU)
	step(jointrace.SubICE, []string{jointrace.SubSendAnswer}, 507, 557, jointrace.KindNet, jointrace.PeerUDP)
	step(jointrace.SubDTLS, []string{jointrace.SubICE}, 557, 607, jointrace.KindNet, jointrace.PeerUDP)
	step(jointrace.SubRTP, []string{jointrace.SubDTLS}, 607, 630, jointrace.KindNet, jointrace.PeerUDP)
	trace := rec.Trace()
	trace.JoinAt = started
	return trace
}

func (s *StreamEdgeSuite) TestTheJoinTraceIsTheSDKsReportUnchanged() {
	trace := joinTimeline(time.Now())

	joined, err := joinTrace(trace)
	s.Require().NoError(err)

	want, err := json.Marshal(trace)
	s.Require().NoError(err)
	s.JSONEq(string(want), string(joined.Trace), "the agent forwards what the SDK recorded")
	s.Equal("coord.join > pcs.create > sfu.ws.dial > sfu.join > sub.debounce > sub.offer > "+
		"sub.sendanswer > sub.ice > sub.dtls > sub.rtp", joined.CriticalPath,
		"the subscriber finishes last, so its branch is the critical path")
	s.Equal(630.0, joined.CriticalMs)
	s.InDelta(4+3+1+0.5+1.02+0.02+1+0.46, joined.CriticalRTTs, 0.01)
}

func (s *StreamEdgeSuite) TestTheJoinTraceIsReportedOnce() {
	s.T().Setenv("STREAM_API_KEY", "key")
	s.T().Setenv("STREAM_API_SECRET", "secret")
	edge, err := New(Options{CallID: "demo", User: User{ID: "agent"}})
	s.Require().NoError(err)

	go edge.onJoinTrace(joinTimeline(time.Now()))

	select {
	case joined := <-edge.JoinTraces():
		var report jointrace.Report
		s.Require().NoError(json.Unmarshal(joined.Trace, &report))
		s.Len(report.Spans, 16)
		s.Equal(jointrace.SubRTP, report.CriticalPath[len(report.CriticalPath)-1])
	case <-time.After(time.Second):
		s.FailNow("the join trace was not reported")
	}

	s.Require().NoError(edge.Leave())
	_, open := <-edge.JoinTraces()
	s.False(open, "leaving closes the channel")
}
