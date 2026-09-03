package streamedge

import (
	"bytes"
	"context"
	"log/slog"
	"math"
	"net/url"
	"strings"
	"testing"
	"time"

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
	s.Equal("default", edge.options.CallType, "a Stream app has this call type out of the box")
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

	// More than five: an utterance ends with a flush, which is what carries the tail of one
	// out of the pipeline. Here there is no tail to carry, so the last of them are quiet.
	frames := s.drain(talker)
	s.Require().Len(frames, 5+flushFrames, "100 ms of speech is five frames, and a flush ends it")
	for _, frame := range frames[:5] {
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
	// Emptying the queue is not enough on its own: what the pipeline is holding belongs to
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
	s.LessOrEqual(heardMs, 100+flushFrames*20,
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
	// The queue is thrown away on leaving, so the part-frame the pipeline is still holding
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
