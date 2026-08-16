package streamedge

import (
	"bytes"
	"context"
	"math"
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

func (s *StreamEdgeSuite) TestSpeechIsEncodedToOpusFrames() {
	// This is the outbound path: the voice's PCM becomes the 20 ms Opus frames the track
	// sends.
	talker := newSpeaker()
	s.T().Cleanup(func() { _ = talker.Close() })

	s.Require().NoError(talker.Write(speech(opusSampleRate, 100)))

	frames := s.drain(talker)
	s.Len(frames, 5, "100 ms of speech is five frames")
	for _, frame := range frames {
		s.NotEmpty(frame)
		s.NotEqual(silenceFrame, frame, "the tone is not silence")
	}
}

func (s *StreamEdgeSuite) TestSilenceIsSentWhenThereIsNothingToSay() {
	// The track is published for the whole call, so something has to go out even when the
	// agent is listening rather than talking.
	talker := newSpeaker()
	s.T().Cleanup(func() { _ = talker.Close() })

	sample, err := talker.NextSample(s.ctx)

	s.Require().NoError(err)
	s.Equal(silenceFrame, sample.Data)
	s.Equal(opusFrameDuration, sample.Duration)
	s.EqualValues(audioLevelSilent, talker.CurrentAudioLevel())
}

func (s *StreamEdgeSuite) TestTheAudioLevelSaysWhoIsTalking() {
	talker := newSpeaker()
	s.T().Cleanup(func() { _ = talker.Close() })
	s.Require().NoError(talker.Write(speech(48_000, 40)))

	_, err := talker.NextSample(s.ctx)
	s.Require().NoError(err)

	s.EqualValues(audioLevelSpeaking, talker.CurrentAudioLevel(),
		"the other participants' clients show the agent as the speaker")
}

func (s *StreamEdgeSuite) TestAnyInputRateIsResampled() {
	// A voice provider's rate is its own business, and a failover mid-call can change it.
	// The resampler holds the first chunk back by design, so speech is written twice here
	// the way a stream of chunks would arrive.
	talker := newSpeaker()
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
	talker := newSpeaker()
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
	talker := newSpeaker()
	s.T().Cleanup(func() { _ = talker.Close() })

	err := talker.Write(audio.PcmData{Samples: make([]int16, 100), SampleRate: 48_000, Channels: 2})

	s.ErrorContains(err, "mono")
}

func (s *StreamEdgeSuite) TestEmptySpeechIsIgnored() {
	talker := newSpeaker()
	s.T().Cleanup(func() { _ = talker.Close() })

	s.NoError(talker.Write(audio.PcmData{SampleRate: 48_000, Channels: 1}))
	s.Empty(s.drain(talker))
}

func (s *StreamEdgeSuite) TestPublishingAfterLeavingFails() {
	talker := newSpeaker()

	s.Require().NoError(talker.Close())

	s.ErrorContains(talker.Write(speech(48_000, 20)), "left")
	s.NoError(talker.Close(), "closing twice is safe")
}
