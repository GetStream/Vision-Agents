//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
)

// sarahOnly is the stub provider's voice library: it lists Sarah and, as ElevenLabs does
// for an id it has never heard of, says no other voice exists.
type sarahOnly struct{}

func (sarahOnly) List(context.Context) ([]voices.Library, error) {
	return []voices.Library{{ID: "stub-sarah", Name: "Sarah - Mature, Reassuring, Confident"}}, nil
}

func (sarahOnly) Preview(context.Context, string) (voices.Speech, error) {
	return voices.Speech{}, nil
}

func (sarahOnly) Exists(context.Context, string) (bool, error) { return false, nil }

// suggestion is what a refusal of "Sara" names.
const suggestion = `the closest in its library is "Sarah - Mature, Reassuring, Confident" (stub-sarah)`

// UnknownVoiceSuite is every way a voice is asked for, the socket, a session's agent and a
// recording, run against a router whose resolver checks the stub provider's library, as
// cmd/router builds it. Every shortcut also reaches plain, a vendor with no library, as
// tts-quality reaches Cartesia and Inworld after ElevenLabs.
type UnknownVoiceSuite struct {
	RouterSuite
}

func TestUnknownVoiceSuite(t *testing.T) {
	suite := new(UnknownVoiceSuite)
	catalogue := voices.NewCatalogue()
	catalogue.Register("stub", sarahOnly{})
	suite.ttsVoices = voices.NewResolver(nil, catalogue)
	suite.ttsConfig = routableConfig()
	suite.ttsConfig.Providers = append(suite.ttsConfig.Providers, routing.ProviderConfig{
		Provider: "plain", Model: "plain-model", Languages: []string{"en"}, Realtime: true, Terms: everyTerm,
	})
	runSuite(t, suite)
}

func (s *UnknownVoiceSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *UnknownVoiceSuite) TestASocketOpenedWithAVoiceTheProviderDoesNotHaveIsRefusedNamingTheClosest() {
	speaking := s.speak("Sara")

	refused := s.await(speaking, "error")

	s.Contains(refused["error"], `stub has no voice "Sara"; `+suggestion)
	s.await(speaking, "closed")
}

func (s *UnknownVoiceSuite) TestASocketOpenedWithAVoiceInTheLibraryStarts() {
	speaking := s.speak("stub-sarah")

	s.Equal("stub", s.await(speaking, "started")["provider"])
}

func (s *UnknownVoiceSuite) TestASessionAskedToSpeakInAVoiceTheProviderDoesNotHaveIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions",
		CreateSessionRequest{Voice: pointerTo("Sara"), StartVoice: pointerTo(true)})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, suggestion)
}

func (s *UnknownVoiceSuite) TestARecordingInAVoiceTheProviderDoesNotHaveFailsNamingTheClosest() {
	var spoken Speech
	// A recording is routed once its job runs, so a refusal is the job's answer, as any
	// other refused recording's is.
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "options": map[string]any{"voice": "Sara"}}, &spoken))

	s.Equal(RecordingStatusFailed, spoken.Status)
	s.Contains(value(spoken.Error), suggestion)
}

// speak opens a text-to-speech socket asking for a voice.
func (s *UnknownVoiceSuite) speak(voice string) *websocket.Conn {
	connection := s.serverClient.opens("/v1/tts/stream")
	s.Require().NoError(connection.WriteJSON(frame{"type": "start", "target": "en-low-latency", "voice": voice}))
	return connection
}
