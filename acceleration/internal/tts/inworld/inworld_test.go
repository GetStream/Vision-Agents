package inworld

import (
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

type InworldSuite struct {
	suite.Suite
}

func TestInworldSuite(t *testing.T) {
	suite.Run(t, new(InworldSuite))
}

func (s *InworldSuite) newTTS(options Options) *TTS {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

// primed is a provider with an utterance in flight and no socket, so handleMessage can
// be fed server frames without a network.
func (s *InworldSuite) primed(id string) *TTS {
	provider := s.newTTS(Options{})
	provider.started = true
	provider.active[id] = &utterance{tracker: tts.NewSynthesis(id), voice: DefaultVoiceID}
	return provider
}

func (s *InworldSuite) collect(provider *TTS, until func(tts.Event) bool) []tts.Event {
	var events []tts.Event
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				return events
			}
			events = append(events, event)
			if until(event) {
				return events
			}
		case <-deadline:
			s.FailNow("timed out waiting for events")
			return events
		}
	}
}

func (s *InworldSuite) TestNewRequiresAPIKey() {
	s.T().Setenv("INWORLD_API_KEY", "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required")
}

func (s *InworldSuite) TestNewFallsBackToEnv() {
	s.T().Setenv("INWORLD_API_KEY", "from-env")
	s.T().Setenv("INWORLD_VOICE_ID", "voice-from-env")

	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("from-env", provider.options.APIKey)
	s.Equal("voice-from-env", provider.options.VoiceID)
}

func (s *InworldSuite) TestNewDefaultsToFlashAndSarah() {
	s.T().Setenv("INWORLD_VOICE_ID", "")

	provider := s.newTTS(Options{})
	s.Equal(DefaultModel, provider.Model())
	s.Equal(DefaultVoiceID, provider.options.VoiceID)
	s.Equal(DefaultSampleRate, provider.SampleRate())
	s.Equal(ProviderName, provider.Provider())
	s.True(provider.Streaming(), "the model generates from partial text")
	s.False(provider.Performs())
	s.Empty(provider.Prompt())
}

func (s *InworldSuite) TestNewRejectsANonPositiveSampleRate() {
	_, err := New(Options{APIKey: "k", SampleRate: -1})
	s.ErrorContains(err, "sample rate must be positive")
}

func (s *InworldSuite) TestTheEndpointIsTheBidirectionalSocket() {
	s.Contains(s.newTTS(Options{}).url(), streamPath)
}

func (s *InworldSuite) TestSynthesizeFailsBeforeStart() {
	err := s.newTTS(Options{}).Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "not started")
}

func (s *InworldSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newTTS(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	for range provider.Events() {
	}
}

func (s *InworldSuite) TestAudioAndContextClosedSettleTheSynthesis() {
	provider := s.primed("u1")
	defer provider.Close()

	pcm := audio.PcmData{Samples: make([]int16, 2400), SampleRate: DefaultSampleRate, Channels: 1}
	provider.handleMessage(serverFrame{Result: &serverResult{
		ContextID:  "u1",
		AudioChunk: &serverAudio{AudioContent: base64.StdEncoding.EncodeToString(pcm.Bytes())},
	}})
	provider.handleMessage(serverFrame{Result: &serverResult{
		ContextID:     "u1",
		ContextClosed: json.RawMessage(`{}`),
	}})

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})

	var chunks []tts.AudioChunk
	var complete tts.SynthesisComplete
	for _, event := range events {
		switch typed := event.(type) {
		case tts.AudioChunk:
			chunks = append(chunks, typed)
		case tts.SynthesisComplete:
			complete = typed
		}
	}

	s.Require().Len(chunks, 1)
	s.Equal("u1", chunks[0].SynthesisID)
	s.Equal(DefaultSampleRate, chunks[0].Audio.SampleRate)
	s.Equal("u1", complete.SynthesisID)
	s.InDelta(100.0, complete.AudioDurationMs, 1.0)
	s.False(complete.Interrupted)
}

func (s *InworldSuite) TestAudioFromAClosedContextIsDropped() {
	provider := s.primed("live")
	defer provider.Close()

	pcm := audio.PcmData{Samples: make([]int16, 2400), SampleRate: DefaultSampleRate, Channels: 1}
	provider.handleMessage(serverFrame{Result: &serverResult{
		ContextID:  "stale",
		AudioChunk: &serverAudio{AudioContent: base64.StdEncoding.EncodeToString(pcm.Bytes())},
	}})
	provider.handleMessage(serverFrame{Result: &serverResult{
		ContextID:     "live",
		ContextClosed: json.RawMessage(`{}`),
	}})

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})
	for _, event := range events {
		_, isChunk := event.(tts.AudioChunk)
		s.False(isChunk, "audio belonging to a context the agent has left behind was published")
	}
}

func (s *InworldSuite) TestAServerStatusErrorSettlesTheUtterance() {
	provider := s.primed("u1")
	defer provider.Close()

	provider.handleMessage(serverFrame{Result: &serverResult{
		ContextID: "u1",
		Status:    &serverStatus{Code: 3, Message: "Invalid model"},
	}})

	events := s.collect(provider, func(event tts.Event) bool {
		_, done := event.(tts.SynthesisComplete)
		return done
	})

	var failure tts.Error
	for _, event := range events {
		if typed, ok := event.(tts.Error); ok {
			failure = typed
		}
	}
	s.Equal("u1", failure.SynthesisID)
	s.ErrorContains(failure.Err, "Invalid model")
	s.False(failure.Fatal, "one rejected utterance does not end the session")
}
