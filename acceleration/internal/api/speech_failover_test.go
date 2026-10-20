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

// SpeechFailoverSuite records over a shortcut per test, named for the test, whose first
// model fails every recording at its vendor and whose second, stub-model, records.
type SpeechFailoverSuite struct {
	RouterSuite
	// down is each test's failing model, new each run: a model's failures make it
	// unavailable, which would put it behind stub-model in the next test or run.
	down map[string]string
}

func TestSpeechFailoverSuite(t *testing.T) {
	suite := &SpeechFailoverSuite{down: map[string]string{}}
	suite.ttsConfig = routableConfig()
	for _, shortcut := range []string{"inline", "job", "pinned"} {
		model := uuid.NewString()
		suite.down[shortcut] = "down/" + model
		suite.ttsConfig.Providers = append(suite.ttsConfig.Providers, routing.ProviderConfig{
			Provider: "down", Model: model, Languages: []string{"en"}, Terms: everyTerm,
		})
		suite.ttsConfig.Aliases[shortcut] = routing.Alias{
			Languages: []string{"en"}, Prefer: "down/" + model, Only: []string{"down/" + model, "stub/stub-model"},
		}
	}
	runSuite(t, suite)
}

func (s *SpeechFailoverSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SpeechFailoverSuite) TestATextSpokenInlineIsRecordedByTheNextModelWhenTheFirstFailsIt() {
	var spoken Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "options": map[string]any{"target": "inline"}}, &spoken))

	s.Require().Equal(RecordingStatusCompleted, spoken.Status, value(spoken.Error))
	s.Equal("stub", value(spoken.Provider))
	s.Equal("stub-model", value(spoken.Model))
	s.Require().Eventually(func() bool {
		return slices.ContainsFunc(s.speechStats(), func(row store.Request) bool {
			return row.Provider+"/"+row.Model == s.down["inline"] && row.ErrorCode == "record_failed"
		})
	}, settleFor, 25*time.Millisecond, "down was not tried first, or its failure was not counted")
}

func (s *SpeechFailoverSuite) TestASpeechJobIsRecordedByTheNextModelWhenTheFirstFailsIt() {
	var accepted Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"text": "hello", "options": map[string]any{"target": "job"}}, &accepted))

	finished := s.finished(accepted.Id)

	s.Require().Equal(RecordingStatusCompleted, finished.Status, value(finished.Error))
	s.Equal("stub", value(finished.Provider))
	s.Equal("stub-model", value(finished.Model))
	s.NotEmpty(value(finished.Audio))
}

// The control below takes its expected values from 1c5fa58db, the base: a pinned model
// has no next candidate, so the job fails as it always has.
func (s *SpeechFailoverSuite) TestASpeechJobPinnedToAModelThatFailsStillFailsAsBefore() {
	var accepted Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"text": "hello", "options": map[string]any{"target": s.down["pinned"]}}, &accepted))

	finished := s.finished(accepted.Id)

	s.Equal(RecordingStatusFailed, finished.Status)
	s.Equal("down: record: 503 Service Unavailable: overloaded", value(finished.Error))
	s.Equal("down", value(finished.Provider))
	s.Equal(s.down["pinned"][len("down/"):], value(finished.Model))
}

// finished polls a speech job until it has ended.
func (s *SpeechFailoverSuite) finished(id string) Speech {
	var job Speech
	s.Require().Eventually(func() bool {
		job = Speech{}
		s.serverClient.do(http.MethodGet, "/v1/tts/recordings/"+id, nil, &job)
		return job.Status != RecordingStatusQueued && job.Status != RecordingStatusRunning
	}, settleFor, 25*time.Millisecond, "the job never finished")
	return job
}

// speechStats are the text-to-speech rows written for this test's app.
func (s *SpeechFailoverSuite) speechStats() []store.Request {
	rows := []store.Request{}
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", s.customerID()).
		Where("modality = ?", "tts").Scan(context.Background()))
	return rows
}
