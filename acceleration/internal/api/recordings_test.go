//go:build integration

package api

import (
	"context"
	"encoding/json"
	"io"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type RecordingsSuite struct {
	RouterSuite

	// called is every post the callback receiver took, by the path and query it was sent to.
	called chan receivedCallback
}

// receivedCallback is one post a caller's callback endpoint took.
type receivedCallback struct {
	uri  string
	body []byte
}

func TestRecordingsSuite(t *testing.T) {
	runSuite(t, new(RecordingsSuite))
}

// SetupSuite stands up a caller's callback endpoint. A callback must name a public address, so
// a job names one, and the router's callback client is handed a dial that reaches the receiver
// on loopback whatever the address, as a public endpoint would answer.
func (s *RecordingsSuite) SetupSuite() {
	s.called = make(chan receivedCallback, 16)
	receiver := httptest.NewTLSServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		// A post nobody reads for must not hold the handler, or receiver.Close waits on it.
		select {
		case s.called <- receivedCallback{uri: r.URL.RequestURI(), body: body}:
		default:
		}
	}))
	s.T().Cleanup(receiver.Close)
	transport := receiver.Client().Transport.(*http.Transport).Clone()
	transport.DialContext = func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, receiver.Listener.Addr().String())
	}
	// A name the receiver's certificate holds (net/http/internal/testcert), whatever host
	// the job named.
	transport.TLSClientConfig.ServerName = "example.com"
	s.callbackHTTP = &http.Client{Transport: transport}
	s.RouterSuite.SetupSuite()
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
	// The path and the query are the caller's own and arrive as they were given.
	uri := "/done/" + s.utils.uuid() + "?job=stt"
	s.transcribe(map[string]any{
		"source":   map[string]any{"url": "https://example.test/call.mp3"},
		"callback": "https://8.8.8.8" + uri, // public, Google DNS
	})

	var finished Transcription
	s.Require().NoError(json.Unmarshal(s.calledBack(uri), &finished))
	s.Equal(RecordingStatusCompleted, finished.Status)
	s.Equal("a call costs a penny", value(finished.Text))
}

func (s *RecordingsSuite) TestAFinishedSpeechJobCallsBackWhoeverAskedToBeTold() {
	uri := "/done/" + s.utils.uuid()
	var accepted Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings",
		map[string]any{"text": "hello", "callback": "https://[2001:4860:4860::8888]" + uri}, &accepted)) // public, Google DNS

	var finished Speech
	s.Require().NoError(json.Unmarshal(s.calledBack(uri), &finished))
	s.Equal(accepted.Id, finished.Id)
	s.Equal(RecordingStatusCompleted, finished.Status)
}

// privateCallbackURLs are callbacks the router would never post to, since it dials only
// public https addresses. A path and a query, the caller's own, are kept on some.
var privateCallbackURLs = []string{
	"https://10.1.2.3/done",            // Private-Use, RFC 1918
	"https://100.101.102.103/done",     // Shared Address Space, RFC 6598
	"https://[fd7a:115c:a1e0::1]/done", // Unique-Local, RFC 4193
	"https://127.0.0.1:8443/done?a=1",  // Loopback, RFC 1122 section 3.2.1.3
	"https://[::1]/done",               // Loopback, RFC 4291
	"https://169.254.169.254/done",     // Link Local, RFC 3927: the cloud metadata server
	"https://[fe80::1]/done",           // Link-Local Unicast, RFC 4291
	"https://localhost/done",           // resolves to loopback, RFC 6761 section 6.3
	"https://2130706433/done",          // 127.0.0.1 written as one number
	"https://hooks.acme.invalid/done",  // never resolves, RFC 6761 section 6.4
	"https://user:pass@8.8.8.8/done",   // userinfo, which egress never sends
	"https:///done",                    // no host
	"https://hooks acme.com/done",      // a host url.Parse refuses
	"http://8.8.8.8/done",              // plain http, which egress never sends
	"8.8.8.8/done",                     // no scheme
}

func (s *RecordingsSuite) TestATranscriptionWithAPrivateCallbackIsRefused() {
	for _, url := range privateCallbackURLs {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/stt/recordings", map[string]any{
			"source": map[string]any{"url": "https://example.test/call.mp3"}, "callback": url,
		})

		s.Equal(http.StatusBadRequest, status, url)
		s.Contains(failure, "callback must be a public https URL", url)
	}
}

func (s *RecordingsSuite) TestASpeechJobWithAPrivateCallbackIsRefused() {
	for _, url := range privateCallbackURLs {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
			map[string]any{"text": "hello", "callback": url})

		s.Equal(http.StatusBadRequest, status, url)
		s.Contains(failure, "callback must be a public https URL", url)
	}
}

func (s *RecordingsSuite) TestACallbackRefusalDoesNotTellAnUnresolvableNameFromAPrivateOne() {
	refusals := map[string]bool{}
	for _, url := range []string{"https://hooks.acme.invalid/done", "https://localhost/done", "https://10.1.2.3/done"} {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/tts/recordings",
			map[string]any{"text": "hello", "callback": url})
		s.Equal(http.StatusBadRequest, status, url)
		refusals[failure] = true
	}

	s.Len(refusals, 1, "every refusal reads the same")
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

// assertTopLevelModelRefused sends a request with a top-level model and expects the 400
// that names where the model goes.
func (s *RouterSuite) assertTopLevelModelRefused(path string, request map[string]any) {
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

// calledBack waits for the post the receiver took at uri and returns its body.
func (s *RecordingsSuite) calledBack(uri string) []byte {
	deadline := time.After(settleFor)
	for {
		select {
		case call := <-s.called:
			if call.uri == uri {
				return call.body
			}
		case <-deadline:
			s.FailNow("nobody was told the job had finished", uri)
			return nil
		}
	}
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
