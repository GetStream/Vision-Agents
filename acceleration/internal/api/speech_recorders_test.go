//go:build integration

package api

import (
	"context"
	"net/http"
	"slices"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// onlyStreams is what a recording of live-model is told, which names the one that records.
const onlyStreams = "live/live-model can only stream speech, not record it; models that record: stub/stub-model"

// SpeechRecordersSuite routes text to speech over live, a vendor that only streams, as
// Cartesia, Inworld and Breeze do in the built-in config: the live router lists it and the
// recordings router has no recorder for it.
type SpeechRecordersSuite struct {
	RouterSuite
}

func TestSpeechRecordersSuite(t *testing.T) {
	suite := new(SpeechRecordersSuite)
	suite.ttsConfig = routableConfig()
	suite.ttsConfig.Providers = append(suite.ttsConfig.Providers, routing.ProviderConfig{
		Provider: "live", Model: "live-model", Languages: []string{"en"}, Realtime: true, Terms: everyTerm,
	})
	// live-first puts live ahead of stub, as tts-quality puts Cartesia ahead of ElevenLabs
	// v4 Turbo.
	suite.ttsConfig.Aliases["live-first"] = routing.Alias{Languages: []string{"en"}, Prefer: "live/live-model"}
	suite.ttsConfig.Aliases["live-only"] = routing.Alias{Title: "Live only", Only: []string{"live/live-model"}}
	runSuite(t, suite)
}

func (s *SpeechRecordersSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SpeechRecordersSuite) TestASpeechJobPinnedToAModelThatOnlyStreamsIsRefusedBeforeItIsQueued() {
	// On da97ee28 this was accepted, queued and failed: "live/live-model: no factory registered".
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"text": "hello", "options": map[string]any{"target": "live/live-model"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, onlyStreams)
}

func (s *SpeechRecordersSuite) TestATextSpokenInlineOnAModelThatOnlyStreamsIsRefused() {
	// On da97ee28 this was a 202 with status failed.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "options": map[string]any{"target": "live/live-model"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, onlyStreams)
}

func (s *SpeechRecordersSuite) TestARecordingOnAShortcutOfModelsThatOnlyStreamIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "options": map[string]any{"target": "live-only"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, onlyStreams)
}

func (s *SpeechRecordersSuite) TestARecordingOnAShortcutIsServedByTheModelThatRecords() {
	// Probed on da97ee28: completed by stub-model, and no row for live.
	var spoken Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "options": map[string]any{"target": "live-first"}}, &spoken))

	s.Require().Equal(RecordingStatusCompleted, spoken.Status, value(spoken.Error))
	s.Equal("stub-model", value(spoken.Model))
	// Stats are written in the order they were taken, so once the recording's row is in, a
	// row for live would be too.
	s.Require().Eventually(func() bool {
		return slices.ContainsFunc(s.speechStats(), func(row store.Request) bool { return row.Provider == "stub" })
	}, settleFor, 25*time.Millisecond, "the recording was never counted")
	for _, row := range s.speechStats() {
		s.NotEqual("live", row.Provider, "a vendor with no recorder is passed over, not counted as failing")
	}
}

func (s *SpeechRecordersSuite) TestTheRoutesSayWhichTextToSpeechCandidatesCanRecord() {
	var resolved []Candidate
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/tts/routes/live-first", nil, &resolved))

	s.Equal(map[string]bool{"live": false, "stub": true}, canRecord(resolved))

	var routes []Route
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/tts/routes", nil, &routes))
	offered := slices.IndexFunc(routes, func(route Route) bool { return route.Id == "live-only" })
	s.Require().NotEqual(-1, offered, "live-only has a title, so it is offered")
	s.Equal(map[string]bool{"live": false}, canRecord(routes[offered].Candidates))
}

func (s *SpeechRecordersSuite) TestSpeechToTextRoutesSayNothingAboutRecording() {
	var resolved []Candidate
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/stt/routes/en-low-latency", nil, &resolved))

	s.Require().NotEmpty(resolved)
	for _, candidate := range resolved {
		s.Nil(candidate.CanRecord, candidate.Provider)
	}
}

// speechStats are the text-to-speech rows written for this test's app.
func (s *SpeechRecordersSuite) speechStats() []store.Request {
	rows := []store.Request{}
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", s.customerID()).
		Where("modality = ?", "tts").Order("id").Scan(context.Background()))
	return rows
}

// canRecord is what each candidate's vendor says about recording.
func canRecord(candidates []Candidate) map[string]bool {
	said := map[string]bool{}
	for _, candidate := range candidates {
		if candidate.CanRecord != nil {
			said[candidate.Provider] = *candidate.CanRecord
		}
	}
	return said
}
