package openaicompat

import (
	"embed"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

//go:embed testdata/*.json
var fixtures embed.FS

type BasetenSuite struct{ suite.Suite }

func TestBasetenSuite(t *testing.T) { suite.Run(t, new(BasetenSuite)) }

func (s *BasetenSuite) provider() *basetenRealtime {
	p, err := newBasetenRealtime(Options{BaseURL: "https://example.invalid", APIKey: "test-key"})
	s.Require().NoError(err)
	return p
}

func (s *BasetenSuite) message(name string) basetenMessage {
	raw, err := fixtures.ReadFile("testdata/baseten_" + name + ".json")
	s.Require().NoError(err)
	var message basetenMessage
	s.Require().NoError(json.Unmarshal(raw, &message))
	return message
}

func (s *BasetenSuite) TestRevisionsReplaceTheSameUtteranceUntilItsFinal() {
	p := s.provider()
	p.participant = stt.Participant{ID: "alice"}
	p.handleMessage(s.message("partial"))
	p.handleMessage(s.message("partial"))
	p.handleMessage(s.message("final"))
	p.handleMessage(s.message("partial"))
	first := (<-p.Events()).(stt.Transcript)
	s.Equal("in a quiet village", first.Text)
	s.Equal(stt.ModeReplacement, first.Mode)
	s.Equal("alice", first.Participant.ID)
	s.Equal("en", first.Language)
	s.Equal(int64(1), first.Utterance)
	s.Equal(int64(1), (<-p.Events()).(stt.Transcript).Utterance)
	final := (<-p.Events()).(stt.Transcript)
	s.True(final.Final())
	s.Equal("In a quiet village.", final.Text)
	s.Equal(int64(1), final.Utterance)
	s.Equal(int64(2), (<-p.Events()).(stt.Transcript).Utterance)
}

func (s *BasetenSuite) TestAnEmptyFinalDoesNotInventSpeechOrReuseThePreviousTurn() {
	p := s.provider()
	p.handleMessage(s.message("empty"))
	s.Empty(p.Events())
	p.handleMessage(s.message("partial"))
	s.Equal(int64(2), (<-p.Events()).(stt.Transcript).Utterance)
}

func (s *BasetenSuite) TestAnEmptyFinalWithdrawsTheWordsOfItsOwnPartial() {
	p := s.provider()
	p.handleMessage(s.message("partial"))
	p.handleMessage(s.message("empty"))
	partial := (<-p.Events()).(stt.Transcript)
	final := (<-p.Events()).(stt.Transcript)
	s.True(final.Final())
	s.Empty(final.Text)
	s.Equal(partial.Utterance, final.Utterance)
}

func (s *BasetenSuite) TestAVadFinalDoesNotAcknowledgeTheHangupFlush() {
	p := s.provider()
	p.handleMessage(s.message("final"))
	select {
	case <-p.finished:
		s.FailNow("an ordinary final must not complete the flush")
	default:
	}
	p.handleMessage(s.message("flush"))
	select {
	case <-p.finished:
	default:
		s.FailNow("the explicit flush final must complete the flush")
	}
}

func (s *BasetenSuite) TestServerErrorsAreReportedAsFatal() {
	p := s.provider()
	p.handleMessage(s.message("error"))
	failure := (<-p.Events()).(stt.Error)
	s.True(failure.Fatal)
	s.ErrorContains(failure, "invalid audio format")
}

func (s *BasetenSuite) TestProtocolCanBeSelectedForTheEnvironmentEndpoint() {
	s.T().Setenv("OPENAI_STT_BASE_URL", "https://example.invalid/environments/production")
	s.T().Setenv("OPENAI_STT_API_KEY", "test-key")
	s.T().Setenv("OPENAI_STT_REALTIME_PROTOCOL", "baseten")
	provider, err := NewRealtime(Options{Model: "qwen3-asr-1.7b"})
	s.Require().NoError(err)
	p := provider.(*basetenRealtime)
	s.Equal("wss://example.invalid/environments/production/websocket", p.options.BaseURL)
	s.Equal("test-key", p.options.APIKey)
}

func (s *BasetenSuite) TestExplicitEndpointDoesNotInheritTheEnvironmentProtocolOrKey() {
	s.T().Setenv("OPENAI_STT_API_KEY", "ambient-secret")
	s.T().Setenv("OPENAI_STT_REALTIME_PROTOCOL", "baseten")
	provider, err := NewRealtime(Options{BaseURL: "https://other.invalid/v1"})
	s.Require().NoError(err)
	p := provider.(*Realtime)
	s.Equal("wss://other.invalid/v1/realtime?intent=transcription", p.endpoint)
	s.Empty(p.apiKey)
}

func (s *BasetenSuite) TestAnUnknownProtocolIsRefused() {
	_, err := NewRealtime(Options{BaseURL: "https://example.invalid", RealtimeProtocol: "unknown"})
	s.ErrorContains(err, "unknown realtime protocol")
}

func (s *BasetenSuite) TestUnsupportedBasetenSettingsAreRefused() {
	_, err := NewRealtime(Options{BaseURL: "https://example.invalid", RealtimeProtocol: "baseten", Settings: TranscriptionSettings{Keywords: []string{"Acme"}}})
	s.ErrorContains(err, "does not support")
}
