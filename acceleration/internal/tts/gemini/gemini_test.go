package gemini

import (
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

type GeminiSuite struct {
	suite.Suite
}

func TestGeminiSuite(t *testing.T) {
	suite.Run(t, new(GeminiSuite))
}

func (s *GeminiSuite) newTTS(options Options) *TTS {
	if options.APIKey == "" {
		options.APIKey = "test-key"
	}
	provider, err := New(options)
	s.Require().NoError(err)
	return provider
}

// drain reads what the provider has emitted so far without waiting for more.
func (s *GeminiSuite) drain(provider *TTS) []tts.Event {
	var events []tts.Event
	for {
		select {
		case event := <-provider.Events():
			events = append(events, event)
		case <-time.After(50 * time.Millisecond):
			return events
		}
	}
}

func chunksOf(events []tts.Event) []tts.AudioChunk {
	var chunks []tts.AudioChunk
	for _, event := range events {
		if chunk, ok := event.(tts.AudioChunk); ok {
			chunks = append(chunks, chunk)
		}
	}
	return chunks
}

func audioPart(raw []byte) part {
	return part{InlineData: &blob{
		Data:     base64.StdEncoding.EncodeToString(raw),
		MimeType: "audio/l16; rate=24000; channels=1",
	}}
}

func eventOf(parts ...part) streamEvent {
	return streamEvent{Candidates: []candidate{{Content: &content{Role: "model", Parts: parts}}}}
}

func silence(samples int) []byte {
	return audio.PcmData{Samples: make([]int16, samples), SampleRate: OutputSampleRate, Channels: 1}.Bytes()
}

func (s *GeminiSuite) TestNewRequiresAPIKey() {
	s.T().Setenv("GOOGLE_API_KEY", "")

	_, err := New(Options{})
	s.ErrorContains(err, "api key is required (set GOOGLE_API_KEY)")
}

func (s *GeminiSuite) TestNewReadsTheKeyTheOtherGeminiPackagesRead() {
	s.T().Setenv("GOOGLE_API_KEY", "")
	s.T().Setenv("GEMINI_API_KEY", "from-google-docs")

	_, err := New(Options{})
	s.Error(err, "GEMINI_API_KEY is what Google's docs name, not what this repository uses")

	s.T().Setenv("GOOGLE_API_KEY", "from-env")
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal("from-env", provider.options.APIKey)
}

func (s *GeminiSuite) TestNewDefaultsTo38FlashTTS() {
	provider := s.newTTS(Options{})

	s.Equal("gemini-3.8-flash-tts", provider.Model())
	s.Equal(ProviderName, provider.Provider())
	s.Equal(24_000, provider.SampleRate())
	s.Empty(provider.Voice(), "no voice leaves the model's own")
	s.False(provider.Streaming(), "one request per utterance means whole sentences only")
	s.False(provider.Performs(), "square-bracketed directions would be read out")
	s.Empty(provider.Prompt())
}

func (s *GeminiSuite) TestNewRejectsANonHTTPURL() {
	_, err := New(Options{APIKey: "k", BaseURL: "wss://generativelanguage.googleapis.com"})
	s.ErrorContains(err, "base url must be http:// or https://")
}

func (s *GeminiSuite) TestAudioPartsBecomeChunksAtTheModelsRate() {
	provider := s.newTTS(Options{})
	synthesis := tts.NewSynthesis("u1")

	carry, err := provider.handleEvent(synthesis, eventOf(audioPart(silence(2400)), audioPart(silence(4800))), nil)
	s.Require().NoError(err)
	s.Empty(carry)

	chunks := chunksOf(s.drain(provider))
	s.Require().Len(chunks, 2)
	s.Equal("u1", chunks[0].SynthesisID)
	s.Equal(0, chunks[0].Index)
	s.Equal(1, chunks[1].Index)
	s.Equal(24_000, chunks[0].Audio.SampleRate)
	s.Equal(1, chunks[0].Audio.Channels)
	s.InDelta(100.0, chunks[0].Audio.DurationMs(), 0.01)
	s.InDelta(200.0, chunks[1].Audio.DurationMs(), 0.01)
}

func (s *GeminiSuite) TestASampleSplitAcrossEventsIsJoinedRatherThanShifted() {
	provider := s.newTTS(Options{})
	synthesis := tts.NewSynthesis("u1")
	// Two samples, 0x0102 and 0x0304, little-endian, cut after the third byte.
	whole := []byte{0x02, 0x01, 0x04, 0x03}

	carry, err := provider.handleEvent(synthesis, eventOf(audioPart(whole[:3])), nil)
	s.Require().NoError(err)
	s.Equal([]byte{0x04}, carry)
	_, err = provider.handleEvent(synthesis, eventOf(audioPart(whole[3:])), carry)
	s.Require().NoError(err)

	chunks := chunksOf(s.drain(provider))
	s.Require().Len(chunks, 2)
	s.Equal([]int16{0x0102}, chunks[0].Audio.Samples)
	s.Equal([]int16{0x0304}, chunks[1].Audio.Samples)
}

func (s *GeminiSuite) TestTextPartsAreNotSpoken() {
	provider := s.newTTS(Options{})

	_, err := provider.handleEvent(tts.NewSynthesis("u1"), eventOf(part{Text: ""}, part{Text: "thinking"}), nil)
	s.Require().NoError(err)

	s.Empty(chunksOf(s.drain(provider)))
}

func (s *GeminiSuite) TestTheLastEventStoppingNormallyIsNotAnError() {
	provider := s.newTTS(Options{})
	event := streamEvent{Candidates: []candidate{{Content: &content{Parts: []part{{Text: ""}}}, FinishReason: "STOP"}}}

	_, err := provider.handleEvent(tts.NewSynthesis("u1"), event, nil)
	s.NoError(err)
}

func (s *GeminiSuite) TestAnyOtherStopEndsTheUtteranceWithItsReason() {
	provider := s.newTTS(Options{})
	event := streamEvent{Candidates: []candidate{{FinishReason: "PROHIBITED_CONTENT"}}}

	_, err := provider.handleEvent(tts.NewSynthesis("u1"), event, nil)
	s.ErrorContains(err, "synthesis stopped: PROHIBITED_CONTENT")
}

func (s *GeminiSuite) TestAnErrorEventEndsTheUtterance() {
	provider := s.newTTS(Options{})
	var event streamEvent
	s.Require().NoError(json.Unmarshal(
		[]byte(`{"error":{"code":503,"message":"The model is overloaded.","status":"UNAVAILABLE"}}`), &event))

	_, err := provider.handleEvent(tts.NewSynthesis("u1"), event, nil)
	s.ErrorContains(err, "UNAVAILABLE: The model is overloaded.")
}

func (s *GeminiSuite) TestTheBodyAsksForRawPCMInTheSessionsVoiceAndLanguage() {
	provider := s.newTTS(Options{Voice: "Kore", Language: "en-US"})

	raw, err := json.Marshal(provider.body("Have a wonderful day!", tts.Request{}))
	s.Require().NoError(err)

	s.JSONEq(`{
		"contents": [{"role": "user", "parts": [{"text": "Have a wonderful day!"}]}],
		"generationConfig": {
			"responseModalities": ["AUDIO"],
			"responseFormat": {"audio": {"mimeType": "AUDIO_L16", "sampleRate": 24000}},
			"speechConfig": {"voiceConfig": {"voice": "Kore"}, "languageCode": "en-US"}
		}
	}`, string(raw))
}

func (s *GeminiSuite) TestARequestCanPickItsOwnVoiceAndLanguage() {
	provider := s.newTTS(Options{Voice: "Kore", Language: "en-US"})

	body := provider.body("Bonjour", tts.Request{Voice: "voice_abc123", Language: "fr"})

	s.Require().NotNil(body.GenerationConfig.SpeechConfig)
	s.Equal("voice_abc123", body.GenerationConfig.SpeechConfig.VoiceConfig.Voice)
	s.Equal("fr", body.GenerationConfig.SpeechConfig.LanguageCode)
}

func (s *GeminiSuite) TestNothingAskedLeavesTheVoiceAndLanguageToTheModel() {
	provider := s.newTTS(Options{})

	body := provider.body("hello", tts.Request{})

	s.Nil(body.GenerationConfig.SpeechConfig)
}

func (s *GeminiSuite) TestSynthesizeFailsBeforeStart() {
	err := s.newTTS(Options{}).Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "not started")
}

func (s *GeminiSuite) TestSynthesizeFailsAfterClose() {
	provider := s.newTTS(Options{})
	s.Require().NoError(provider.Close())

	err := provider.Synthesize(tts.Request{Text: "hello", Final: true})
	s.ErrorContains(err, "session closed")
}

func (s *GeminiSuite) TestCloseIsIdempotentAndClosesEvents() {
	provider := s.newTTS(Options{})

	s.Require().NoError(provider.Close())
	s.Require().NoError(provider.Close())

	for range provider.Events() {
	}
}

func (s *GeminiSuite) TestSatisfiesTTSInterface() {
	var _ tts.TTS = s.newTTS(Options{})
	var _ tts.Voiced = s.newTTS(Options{})
}
