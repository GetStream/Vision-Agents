//go:build integration

package api

import (
	"encoding/json"
	"maps"
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
	s.Nil(finished.Voice, "no voice was asked and the stub has no default, so none is claimed")
}

func (s *RecordingsSuite) TestASpeechJobSaysWhichVoiceItWasSpokenIn() {
	var accepted Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"text": "hello", "options": map[string]any{"voice": "library-voice"}}, &accepted))

	var finished Speech
	s.Require().Eventually(func() bool {
		finished = Speech{}
		s.serverClient.do(http.MethodGet, "/v1/tts/recordings/"+accepted.Id, nil, &finished)
		return finished.Status == RecordingStatusCompleted
	}, settleFor, 25*time.Millisecond, "the job never finished")

	s.Equal("library-voice", value(finished.Voice))
}

func (s *RecordingsSuite) TestASpeechJobWithNothingToSayIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"text": "   "})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "nothing to say")
}

func (s *RecordingsSuite) TestARecordingNamingItsModelOutsideTheOptionsIsRefused() {
	// On 80ed8ac26 each of these was a 202 served by the default recorded model: the body
	// takes unknown fields, so a top-level model was dropped and the caller billed for a
	// model they never asked for.
	speech := map[string]any{"text": "hello"}
	transcription := map[string]any{"source": map[string]any{"url": "https://example.test/call.mp3"}}
	for path, body := range map[string]map[string]any{"/v1/tts/recordings": speech, "/v1/stt/recordings": transcription} {
		for _, model := range []any{"stub/stub-model", "", 5, map[string]any{}, []string{"stub/stub-model"}} {
			for _, inline := range []bool{false, true} {
				request := maps.Clone(body)
				request["model"], request["inline"] = model, inline
				if inline && path == "/v1/stt/recordings" {
					request["source"] = map[string]any{"audio": "YXVkaW8="}
				}
				s.assertTopLevelModelRefused(path, request)
			}
		}
		request := maps.Clone(body)
		request["model"] = "stub/stub-model"
		request["options"] = map[string]any{"target": "stub/stub-model"}
		s.assertTopLevelModelRefused(path, request)
	}
}

func (s *RecordingsSuite) TestARecordingWithANullModelIsServedAsOneWithNone() {
	// A null is how a generated client leaves a field out. Probed on 80ed8ac26: 202 and
	// completed, as now.
	var spoken Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "model": nil}, &spoken))
	s.Equal(RecordingStatusCompleted, spoken.Status, value(spoken.Error))

	var transcript Transcription
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/stt/recordings",
		map[string]any{"inline": true, "source": map[string]any{"audio": "YXVkaW8="}, "model": nil}, &transcript))
	s.Equal(RecordingStatusCompleted, transcript.Status, value(transcript.Error))
}

func (s *RecordingsSuite) TestARecordingNamingItsModelInOptionsTargetIsServedByIt() {
	// Probed on 80ed8ac26: each was served by the model named, as now.
	var spoken Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "options": map[string]any{"target": "stub/stub-model"}}, &spoken))
	s.Require().Equal(RecordingStatusCompleted, spoken.Status, value(spoken.Error))
	s.Equal("stub-model", value(spoken.Model))

	var transcript Transcription
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/stt/recordings", map[string]any{
		"inline": true, "source": map[string]any{"audio": "YXVkaW8="},
		"options": map[string]any{"target": "stub/stub-batch-es"},
	}, &transcript))
	s.Require().Equal(RecordingStatusCompleted, transcript.Status, value(transcript.Error))
	s.Equal("stub-batch-es", value(transcript.Model), "not stub-batch, which a recording naming no target gets")
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

func (s *RecordingsSuite) TestATextSpokenInlineSaysWhichVoiceItWasSpokenIn() {
	var spoken Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"inline": true, "text": "hello", "options": map[string]any{"voice": "library-voice"}}, &spoken))

	s.Equal("library-voice", value(spoken.Voice))
}

func (s *RecordingsSuite) TestARecordingTranscribedInlineComesBackInTheAnswer() {
	var transcript Transcription
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/stt/recordings",
		map[string]any{"inline": true, "source": map[string]any{"audio": "YXVkaW8="}}, &transcript))

	s.Equal(RecordingStatusCompleted, transcript.Status)
	s.NotEmpty(value(transcript.Text))
}

func (s *RecordingsSuite) TestARecordingThroughAConfigOnTheLiveShortcutIsServedByTheBatchModel() {
	// What the dashboard's router test sends: a config whose STT is the live shortcut.
	var config RouterConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/router/configs", map[string]any{
		"name": "live-" + s.utils.uuid(), "stt": map[string]any{"target": "en-low-latency"},
	}, &config))

	var transcript Transcription
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/stt/recordings", map[string]any{
		"config_id": config.Id, "inline": true, "source": map[string]any{"audio": "YXVkaW8="},
	}, &transcript))

	s.Equal(RecordingStatusCompleted, transcript.Status, value(transcript.Error))
	s.Equal("stub-batch", value(transcript.Model))
}

func (s *RecordingsSuite) TestARecordingPinnedToABatchModelIsServedByThatModel() {
	var transcript Transcription
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/stt/recordings", map[string]any{
		"inline": true, "source": map[string]any{"audio": "YXVkaW8="},
		"options": map[string]any{"target": "stub/stub-batch"},
	}, &transcript))

	s.Equal(RecordingStatusCompleted, transcript.Status, value(transcript.Error))
	s.Equal("stub-batch", value(transcript.Model))
}

func (s *RecordingsSuite) TestARecordingInAnotherLanguageWithNoTargetGoesToTheMultilingualRecordedShortcut() {
	var transcript Transcription
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/stt/recordings", map[string]any{
		"inline": true, "source": map[string]any{"audio": "YXVkaW8="},
		"options": map[string]any{"languages": []string{"es"}},
	}, &transcript))

	s.Equal(RecordingStatusCompleted, transcript.Status, value(transcript.Error))
	s.Equal("stub-batch-es", value(transcript.Model))
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

// assertTopLevelModelRefused sends a recording with a top-level model and expects the 400
// that names where the model goes.
func (s *RecordingsSuite) assertTopLevelModelRefused(path string, request map[string]any) {
	status, payload := s.serverClient.call(http.MethodPost, path, request)
	s.Require().Equal(http.StatusBadRequest, status, "%s %v: %s", path, request, payload)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered))
	s.Equal(ErrorTypeInvalidRequest, answered.Error.Type)
	s.Contains(answered.Error.Message, "options.target")
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
