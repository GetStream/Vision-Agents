//go:build integration

package api

import (
	"context"
	"encoding/base64"
	"net/http"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
)

type VoicesSuite struct {
	RouterSuite
}

func TestVoicesSuite(t *testing.T) {
	runSuite(t, new(VoicesSuite))
}

// SetupTest gives every test an app of its own, because the voices listed are everything
// one customer brought with them.
func (s *VoicesSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *VoicesSuite) TestAVoiceIsRecordedPreparedAndReadBack() {
	created := s.createVoice(map[string]any{
		"name": "founder", "description": "the one from the ad",
	})
	s.Require().NotEmpty(created.Id)
	s.Equal("founder", created.Name)
	s.Empty(value(created.Samples), "a voice starts with nothing recorded")

	recorded := s.record(created.Id, map[string]any{
		"audio":      base64.StdEncoding.EncodeToString([]byte("pretend this is speech")),
		"filename":   "clip.wav",
		"transcript": "hello", "content_type": "audio/wav",
	})
	s.Require().Len(value(recorded.Samples), 1)
	s.EqualValues(22, value(value(recorded.Samples)[0].Bytes))

	prepared := s.prepare(created.Id)
	s.Require().Len(value(prepared.Bindings), 1)
	s.Equal(VoiceBindingStateReady, value(prepared.Bindings)[0].State)
	s.Equal("el-cloned", value(value(prepared.Bindings)[0].ExternalId),
		"a session names the voice, and the provider is asked for its own id")
}

func (s *VoicesSuite) TestVoiceProvidersAreTheOnesThisDeploymentCanCloneWith() {
	var listed VoiceProviders
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/voices/providers", nil, &listed))

	s.Equal([]string{"elevenlabs"}, listed.Providers)
}

func (s *VoicesSuite) TestAPreparedVoiceCanBeHeardThroughItsProvider() {
	created := s.createVoice(map[string]any{"name": "founder"})
	s.record(created.Id, map[string]any{
		"audio":    base64.StdEncoding.EncodeToString([]byte("pretend this is speech")),
		"filename": "clip.wav",
	})
	s.prepare(created.Id)

	var preview VoicePreview
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/voices/"+created.Id+"/preview",
		map[string]any{"provider": "elevenlabs"}, &preview))

	s.Equal("elevenlabs", preview.Provider)
	s.Equal("audio/mpeg", preview.ContentType)
	s.Equal([]byte("spoken"), preview.Audio)
}

func (s *VoicesSuite) TestAVoiceCannotBeHeardThroughAProviderThatDoesNotHaveIt() {
	created := s.createVoice(map[string]any{"name": "founder"})

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/voices/"+created.Id+"/preview", map[string]any{"provider": "elevenlabs"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "not ready with elevenlabs")
}

func (s *VoicesSuite) TestAVoiceWithNothingRecordedCannotBePrepared() {
	created := s.createVoice(map[string]any{"name": "founder"})

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/voices/"+created.Id+"/prepare", map[string]any{})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "add a recording")
}

func (s *VoicesSuite) TestAVoiceNeedsAName() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/voices",
		map[string]any{"name": "  "})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "needs a name")
}

func (s *VoicesSuite) TestADeletedVoiceStopsBeingListed() {
	created := s.createVoice(map[string]any{"name": "founder"})

	s.Require().Equal(http.StatusNoContent,
		s.serverClient.do(http.MethodDelete, "/v1/agents/voices/"+created.Id, nil, nil))

	var listed []Voice
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/voices", nil, &listed))
	s.Empty(listed)
}

func (s *VoicesSuite) TestAVoiceBelongingToAnotherAppIsNotThere() {
	created := s.createVoice(map[string]any{"name": "founder"})

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/voices/"+created.Id, nil, nil)
	})
}

func (s *VoicesSuite) TestOnlyTheAppsOwnBackendMayBringAVoice() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/voices",
			map[string]any{"name": "voice-" + s.utils.uuid()}, nil)
	})
}

func (s *VoicesSuite) TestVoiceProvidersAreRankedSeparatelyFromTranscribers() {
	// A speech-to-text failure must not make the text-to-speech provider look unhealthy.
	s.Require().NoError(s.live.RecordRequest(context.Background(), live.Usage{
		Modality: "stt", CustomerID: s.customerID(),
		Provider: "stub", Model: "stub-model",
		LatencyMs: 150, Success: false,
	}))

	var providers []Provider
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/tts/providers", nil, &providers))
	s.Require().NotEmpty(providers)

	for _, provider := range providers {
		if provider.Model == "stub-model" {
			s.Zero(provider.Health.Errors, "health is keyed by modality")
		}
	}
}

// createVoice brings a voice of the customer's own.
func (s *VoicesSuite) createVoice(body map[string]any) Voice {
	var created Voice
	s.Require().Equal(http.StatusCreated,
		s.serverClient.do(http.MethodPost, "/v1/agents/voices", body, &created))
	return created
}

// record adds a sample to a voice.
func (s *VoicesSuite) record(id string, sample map[string]any) Voice {
	var recorded Voice
	s.Require().Equal(http.StatusCreated,
		s.serverClient.do(http.MethodPost, "/v1/agents/voices/"+id+"/samples", sample, &recorded))
	return recorded
}

// prepare clones a voice at every provider that can hold one.
func (s *VoicesSuite) prepare(id string) Voice {
	var prepared Voice
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodPost, "/v1/agents/voices/"+id+"/prepare", map[string]any{}, &prepared))
	return prepared
}
