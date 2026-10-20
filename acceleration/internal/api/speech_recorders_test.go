//go:build integration

package api

import (
	"context"
	"net/http"
	"slices"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SpeechRecordersSuite routes text to speech over live, a vendor that only streams, as
// Inworld and Breeze do in the built-in config: the live router lists it and the
// recordings router has no recorder for it. Cartesia records too.
type SpeechRecordersSuite struct {
	RouterSuite
	// live is the model of the vendor that only streams, new each run: health outlives a
	// run, and errors left by an earlier one would put it behind stub.
	live string
}

func TestSpeechRecordersSuite(t *testing.T) {
	suite := &SpeechRecordersSuite{live: "live/" + uuid.NewString()}
	suite.ttsConfig = routableConfig()
	suite.ttsConfig.Providers = append(suite.ttsConfig.Providers, routing.ProviderConfig{
		Provider: "live", Model: suite.live[len("live/"):], Languages: []string{"en"}, Realtime: true, Terms: everyTerm,
	})
	// live-first puts live ahead of stub, as tts-quality puts Cartesia ahead of ElevenLabs
	// v4 Turbo.
	suite.ttsConfig.Aliases["live-first"] = routing.Alias{Languages: []string{"en"}, Prefer: suite.live}
	suite.ttsConfig.Aliases["live-only"] = routing.Alias{Title: "Live only", Only: []string{suite.live}}
	runSuite(t, suite)
}

func (s *SpeechRecordersSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SpeechRecordersSuite) TestASpeechJobPinnedToAModelThatOnlyStreamsIsRefusedBeforeItIsQueued() {
	// On da97ee28 this was accepted, queued and failed: "<live>: no factory registered".
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"text": "hello", "options": map[string]any{"target": s.live}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, s.onlyStreams())
	s.Zero(s.recordingRows(), "a refused job is not left queued")
}

func (s *SpeechRecordersSuite) TestASpeechJobWithACallbackIsRefusedBeforeItIsQueuedToo() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"text": "hello", "callback": "https://example.com/hook", "options": map[string]any{"target": s.live}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, s.onlyStreams())
	s.Zero(s.recordingRows(), "a refused job is not left queued")
}

func (s *SpeechRecordersSuite) TestATextSpokenInlineOnAModelThatOnlyStreamsIsRefused() {
	// On da97ee28 this was a 202 with status failed.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "options": map[string]any{"target": s.live}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, s.onlyStreams())
}

func (s *SpeechRecordersSuite) TestARecordingOnAShortcutOfModelsThatOnlyStreamIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "options": map[string]any{"target": "live-only"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, s.onlyStreams())
}

func (s *SpeechRecordersSuite) TestARecordingOnAShortcutIsServedByTheModelThatRecords() {
	// Probed on da97ee28: completed by stub-model, and no row for live.
	var resolved []Candidate
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/tts/routes/live-first", nil, &resolved))
	s.Require().Equal("live", resolved[0].Provider, "live is tried first, or this proves nothing")

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

// The controls below take their expected values from da97ee28, the base: an unknown target
// is not this change's to refuse. Check leaves it to Record, which says what it always has.
func (s *SpeechRecordersSuite) TestAnUnknownTargetSpokenInlineStillFailsTheSpeech() {
	var spoken Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "options": map[string]any{"target": "nope/nope"}}, &spoken))

	s.Equal(RecordingStatusFailed, spoken.Status)
	s.Equal(`routing: unknown target "nope/nope"`, value(spoken.Error))
}

func (s *SpeechRecordersSuite) TestAnUnknownTargetAsAJobIsStillQueued() {
	var queued Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"text": "hello", "options": map[string]any{"target": "nope/nope"}}, &queued))

	s.Equal(RecordingStatusQueued, queued.Status)
}

func (s *SpeechRecordersSuite) TestSpeechToTextAndConversationRoutesSayNothingAboutRecording() {
	for _, modality := range []string{"stt", "sts"} {
		var routes []Route
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/"+modality+"/routes", nil, &routes))
		s.Require().NotEmpty(routes, modality+" has a shortcut with a title, so it is listed")
		for _, route := range routes {
			s.Require().NotEmpty(route.Candidates, modality+"/"+route.Id)
			for _, candidate := range route.Candidates {
				s.Nil(candidate.CanRecord, modality+"/"+route.Id+" "+candidate.Provider)
			}
		}
	}
}

func (s *SpeechRecordersSuite) TestSpeechToTextRoutesSayNothingAboutRecording() {
	var resolved []Candidate
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/stt/routes/en-low-latency", nil, &resolved))

	s.Require().NotEmpty(resolved)
	for _, candidate := range resolved {
		s.Nil(candidate.CanRecord, candidate.Provider)
	}
}

// onlyStreams is what a recording of live is told, which names the model that records.
func (s *SpeechRecordersSuite) onlyStreams() string {
	return s.live + " can only stream speech, not record it; models that record: stub/stub-model"
}

// recordingRows is how many recordings, queued or not, this test's app has.
func (s *SpeechRecordersSuite) recordingRows() int {
	count, err := s.store.DB().NewSelect().Model((*store.Recording)(nil)).Where("customer_id = ?", s.customerID()).
		Count(context.Background())
	s.Require().NoError(err)
	return count
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
