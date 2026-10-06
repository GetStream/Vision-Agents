//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type RecordingsSuite struct {
	RouterSuite
}

func TestRecordingsSuite(t *testing.T) {
	runSuite(t, new(RecordingsSuite))
}

func (s *RecordingsSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *RecordingsSuite) TestARecordingIsTranscribedAndReadBack() {
	accepted := s.transcribe(map[string]any{
		"source":  map[string]any{"url": "https://example.test/call.mp3"},
		"options": map[string]any{"diarize": true, "words": true, "language": "en"},
	})
	s.Require().NotEmpty(accepted.Id)
	s.Equal(RecordingStatusQueued, accepted.Status, "a job is a row before it is work")

	finished := s.finishedTranscription(accepted.Id)
	s.Equal("a call costs a penny", value(finished.Text))
	s.Len(value(finished.Words), 2, "word timings were asked for")
	s.Equal([]string{"speaker_0"}, value(finished.Speakers))
	s.Equal("stub", value(finished.Provider))
}

func (s *RecordingsSuite) TestSubtitlesAreRenderedFromTheTimingsWhoeverServedThem() {
	// Vendors offer subtitles inconsistently, and every one of them returns what it takes
	// to render them, so a caller asking for srt gets it from whichever one answered.
	accepted := s.transcribe(map[string]any{
		"source":  map[string]any{"url": "https://example.test/call.mp3"},
		"options": map[string]any{"output": "srt"},
	})

	finished := s.finishedTranscription(accepted.Id)
	s.Contains(value(finished.Subtitles), "00:00:00,000 --> 00:00:00,600")
	s.Contains(value(finished.Subtitles), "a call")
}

func (s *RecordingsSuite) TestARecordingWithNothingToTranscribeIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/stt/recordings",
		map[string]any{"source": map[string]any{}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "url or the audio")
}

func (s *RecordingsSuite) TestAnOutputFormatNobodyRendersIsRefusedBeforeTheJobRuns() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/stt/recordings", map[string]any{
		"source":  map[string]any{"url": "https://example.test/call.mp3"},
		"options": map[string]any{"output": "ass"},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "srt")
	s.Contains(failure, "vtt")
}

func (s *RecordingsSuite) TestATextIsSpokenIntoOneFileAndReadBack() {
	var accepted Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{
			"text":    "Chapter one. A call costs a penny.",
			"options": map[string]any{"format": "mp3_44100_128"},
		}, &accepted))
	s.Require().NotEmpty(accepted.Id)

	var finished Speech
	s.Require().Eventually(func() bool {
		finished = Speech{}
		s.serverClient.do(http.MethodGet, "/v1/tts/recordings/"+accepted.Id, nil, &finished)
		return finished.Status != RecordingStatusQueued && finished.Status != RecordingStatusRunning
	}, settleFor, 25*time.Millisecond, "the job never finished")

	s.Require().Equal(RecordingStatusCompleted, finished.Status, value(finished.Error))
	s.NotEmpty(value(finished.Audio))
	s.Equal("mp3_44100_128", value(finished.Format))
	s.EqualValues(34, value(finished.Characters))
}

func (s *RecordingsSuite) TestASpeechJobWithNothingToSayIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"text": "   "})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "nothing to say")
}

func (s *RecordingsSuite) TestAFinishedRecordingCallsBackWhoeverAskedToBeTold() {
	told := make(chan Transcription, 1)
	listener := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		var finished Transcription
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&finished))
		told <- finished
	}))
	defer listener.Close()

	s.transcribe(map[string]any{
		"source":   map[string]any{"url": "https://example.test/call.mp3"},
		"callback": listener.URL + "/done",
	})

	select {
	case finished := <-told:
		s.Equal(RecordingStatusCompleted, finished.Status)
		s.Equal("a call costs a penny", value(finished.Text))
	case <-time.After(settleFor):
		s.Fail("nobody was told the job had finished")
	}
}

func (s *RecordingsSuite) TestATextSpokenInlineComesBackInTheAnswer() {
	var spoken Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello"}, &spoken))

	s.Equal(RecordingStatusCompleted, spoken.Status, "there was no job to come back for")
	s.NotEmpty(value(spoken.Audio))
}

func (s *RecordingsSuite) TestARecordingTranscribedInlineComesBackInTheAnswer() {
	var transcript Transcription
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/stt/recordings",
		map[string]any{"inline": true, "source": map[string]any{"audio": "YXVkaW8="}}, &transcript))

	s.Equal(RecordingStatusCompleted, transcript.Status)
	s.NotEmpty(value(transcript.Text))
}

func (s *RecordingsSuite) TestNobodyIsCalledBackAboutSomethingAnsweredInline() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "callback": "https://example.test/done"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "callback")
}

func (s *RecordingsSuite) TestAudioAnsweredInlineHasToBeInTheRequest() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/stt/recordings", map[string]any{
		"inline": true, "source": map[string]any{"url": "https://example.test/call.mp3"},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "no URL")
}

func (s *RecordingsSuite) TestAnotherAppsRecordingIsNotFound() {
	accepted := s.transcribe(map[string]any{
		"source": map[string]any{"url": "https://example.test/call.mp3"},
	})

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/stt/recordings/"+accepted.Id, nil, nil)
	})
}

func (s *RecordingsSuite) TestOnlyTheAppsOwnBackendMayTranscribeARecording() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/stt/recordings",
			map[string]any{"source": map[string]any{"url": "https://example.test/call.mp3"}}, nil)
	})
}

// transcribe queues a job the router must accept.
func (s *RecordingsSuite) transcribe(body map[string]any) Transcription {
	var accepted Transcription
	s.Require().Equal(http.StatusAccepted,
		s.serverClient.do(http.MethodPost, "/v1/stt/recordings", body, &accepted))
	return accepted
}

// finishedTranscription polls a job until it is neither queued nor running.
func (s *RecordingsSuite) finishedTranscription(id string) Transcription {
	var finished Transcription
	s.Require().Eventually(func() bool {
		finished = Transcription{}
		s.serverClient.do(http.MethodGet, "/v1/stt/recordings/"+id, nil, &finished)
		return finished.Status != RecordingStatusQueued && finished.Status != RecordingStatusRunning
	}, settleFor, 25*time.Millisecond, "the job never finished")

	s.Require().Equal(RecordingStatusCompleted, finished.Status, value(finished.Error))
	return finished
}
