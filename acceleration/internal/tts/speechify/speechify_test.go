package speechify

import (
	"bytes"
	"context"
	"io"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

type SpeechifySuite struct {
	suite.Suite
}

func TestSpeechifySuite(t *testing.T) {
	suite.Run(t, new(SpeechifySuite))
}

// newTTS returns a provider that is wired up but never started.
func (s *SpeechifySuite) newTTS(options Options) *TTS {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = provider.Close() })
	return provider
}

// drain returns what the provider has emitted so far without waiting for more.
func (s *SpeechifySuite) drain(provider *TTS) []tts.Event {
	var events []tts.Event
	for {
		select {
		case event := <-provider.Events():
			events = append(events, event)
		default:
			return events
		}
	}
}

// failingReader hands back its audio and then the error a transfer aborted mid-stream ends
// with.
type failingReader struct {
	audio []byte
	err   error
}

func (r *failingReader) Read(buffer []byte) (int, error) {
	if len(r.audio) == 0 {
		return 0, r.err
	}
	read := copy(buffer, r.audio)
	r.audio = r.audio[read:]
	return read, nil
}

func (s *SpeechifySuite) TestNewRequiresAPIKey() {
	s.T().Setenv("SPEECHIFY_API_KEY", "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *SpeechifySuite) TestNewFallsBackToEnv() {
	s.T().Setenv("SPEECHIFY_API_KEY", "from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("from-env", provider.options.APIKey)
}

func (s *SpeechifySuite) TestNewDefaultsToSimba32() {
	provider := s.newTTS(Options{})
	s.Equal(DefaultModel, provider.Model())
	s.Equal(DefaultVoiceID, provider.Voice())
	s.Equal(ProviderName, provider.Provider())
	s.Equal(DefaultSampleRate, provider.SampleRate())
	s.False(provider.Streaming(), "one POST per utterance means whole sentences only")
	s.False(provider.Performs())
}

func (s *SpeechifySuite) TestNewRefusesARateSpeechifyHasNoPcmFor() {
	_, err := New(Options{APIKey: "test-key", SampleRate: 32_000})
	s.ErrorContains(err, "no pcm format at 32000 Hz")
}

func (s *SpeechifySuite) TestSynthesizeFailsBeforeStart() {
	err := s.newTTS(Options{}).Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "not started")
}

func (s *SpeechifySuite) TestSynthesizeFailsAfterClose() {
	provider := s.newTTS(Options{})
	s.Require().NoError(provider.Close())

	err := provider.Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "session closed")
}

func (s *SpeechifySuite) TestAPartialWithoutAnIdIsRejected() {
	provider := s.newTTS(Options{})
	s.Require().NoError(provider.Start(context.Background()))

	err := provider.Synthesize(tts.Request{Text: "hello"})
	s.ErrorContains(err, "a partial request needs an id")
}

func (s *SpeechifySuite) TestAnEmptyUtteranceIsRejected() {
	provider := s.newTTS(Options{})
	s.Require().NoError(provider.Start(context.Background()))

	err := provider.Synthesize(tts.Request{Text: "   ", Final: true})
	s.ErrorContains(err, "nothing to say")
}

func (s *SpeechifySuite) TestStreamCarriesAnOddByteIntoTheNextChunk() {
	provider := s.newTTS(Options{})
	synthesis := tts.NewSynthesis("u1")
	// 4097 bytes is one chunk plus a byte, then the second half of that sample arrives.
	body := io.MultiReader(bytes.NewReader(make([]byte, chunkBytes+1)), bytes.NewReader([]byte{0}))

	interrupted := provider.stream(synthesis, body)

	s.False(interrupted)
	var total int
	for _, event := range s.drain(provider) {
		chunk, ok := event.(tts.AudioChunk)
		s.Require().True(ok)
		s.Equal(DefaultSampleRate, chunk.Audio.SampleRate)
		total += len(chunk.Audio.Samples)
	}
	s.Equal((chunkBytes+2)/2, total, "no sample should be split or lost")
}

func (s *SpeechifySuite) TestAnAbortedTransferIsAnErrorAndCutsTheUtteranceShort() {
	provider := s.newTTS(Options{})
	synthesis := tts.NewSynthesis("u1")
	pcm := audio.PcmData{Samples: make([]int16, 1000), SampleRate: DefaultSampleRate, Channels: 1}

	interrupted := provider.stream(synthesis, &failingReader{audio: pcm.Bytes(), err: io.ErrUnexpectedEOF})

	s.True(interrupted)
	events := s.drain(provider)
	s.Require().Len(events, 2)
	s.IsType(tts.AudioChunk{}, events[0])
	failure, ok := events[1].(tts.Error)
	s.Require().True(ok)
	s.ErrorIs(failure.Err, io.ErrUnexpectedEOF)
	s.Equal("audio", failure.Context)
}

func (s *SpeechifySuite) TestACancelledReadIsBargeInNotAFailure() {
	provider := s.newTTS(Options{})

	interrupted := provider.stream(tts.NewSynthesis("u1"), &failingReader{err: context.Canceled})

	s.True(interrupted)
	s.Empty(s.drain(provider), "barge-in is not an error")
}

func (s *SpeechifySuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newTTS(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	for range provider.Events() {
	}
}

func (s *SpeechifySuite) TestSatisfiesTTSInterface() {
	var _ tts.TTS = s.newTTS(Options{})
	var _ tts.Voiced = s.newTTS(Options{})
}
