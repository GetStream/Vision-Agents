package openaicompat

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type realtimeCapture struct {
	path      string
	query     string
	authority string
	update    realtimeSessionUpdate
	appends   int
	audio     []byte
	turns     [][]byte
	commits   int
	commitIDs []string
	serverErr error
}

type realtimeSessionUpdate struct {
	Type    string `json:"type"`
	Session struct {
		Type    string   `json:"type"`
		Include []string `json:"include"`
		Audio   struct {
			Input struct {
				Format struct {
					Type string `json:"type"`
					Rate int    `json:"rate"`
				} `json:"format"`
				Transcription struct {
					Model     string   `json:"model"`
					Language  string   `json:"language"`
					Languages []string `json:"languages"`
					Prompt    string   `json:"prompt"`
					Keywords  []string `json:"keywords"`
					Delay     string   `json:"delay"`
				} `json:"transcription"`
				TurnDetection  any `json:"turn_detection"`
				NoiseReduction any `json:"noise_reduction"`
			} `json:"input"`
		} `json:"audio"`
	} `json:"session"`
}

type realtimeFake struct {
	server *httptest.Server

	mu                         sync.Mutex
	busy                       bool
	holdFirstUntilSecondCommit bool
	precommitPartials          bool
	failFirst                  bool
	missingAck                 bool
	rejectFirstCommit          bool
	interleavePartials         bool
	delta                      string
	postCommitDelta            string
	deltaContentIndex          int
	mismatchCommitID           bool
	emptyFinal                 bool
	firstCommit                chan struct{}
	secondCommit               chan struct{}
	captures                   chan realtimeCapture
}

type RealtimeSocketSuite struct {
	suite.Suite
	fake *realtimeFake
}

func TestRealtimeSocketSuite(t *testing.T) {
	suite.Run(t, new(RealtimeSocketSuite))
}

func (s *RealtimeSocketSuite) SetupSuite() {
	s.fake = &realtimeFake{}
	s.fake.reset()
	s.fake.server = httptest.NewServer(http.HandlerFunc(s.fake.serve))
}

func (s *RealtimeSocketSuite) SetupTest() { s.fake.reset() }

func (f *realtimeFake) reset() {
	f.mu.Lock()
	f.busy = false
	f.holdFirstUntilSecondCommit = false
	f.precommitPartials = false
	f.failFirst = false
	f.missingAck = false
	f.rejectFirstCommit = false
	f.interleavePartials = false
	f.delta = ""
	f.postCommitDelta = ""
	f.deltaContentIndex = 0
	f.mismatchCommitID = false
	f.emptyFinal = false
	f.firstCommit = make(chan struct{})
	f.secondCommit = make(chan struct{})
	f.captures = make(chan realtimeCapture, 1)
	f.mu.Unlock()
}

func (f *realtimeFake) serve(w http.ResponseWriter, request *http.Request) {
	f.mu.Lock()
	busy, holdFirstUntilSecondCommit := f.busy, f.holdFirstUntilSecondCommit
	precommitPartials := f.precommitPartials
	failFirst, missingAck := f.failFirst, f.missingAck
	rejectFirstCommit, interleavePartials := f.rejectFirstCommit, f.interleavePartials
	delta, postCommitDelta := f.delta, f.postCommitDelta
	deltaContentIndex, mismatchCommitID := f.deltaContentIndex, f.mismatchCommitID
	emptyFinal := f.emptyFinal
	captures, firstCommit := f.captures, f.firstCommit
	f.mu.Unlock()

	capture := realtimeCapture{
		path:      request.URL.Path,
		query:     request.URL.RawQuery,
		authority: request.Header.Get("Authorization"),
	}
	defer func() { captures <- capture }()

	upgrader := websocket.Upgrader{}
	conn, err := upgrader.Upgrade(w, request, nil)
	if err != nil {
		capture.serverErr = err
		return
	}
	defer conn.Close()
	if err = conn.WriteJSON(map[string]string{"type": "session.created"}); err != nil {
		capture.serverErr = err
		return
	}
	_, raw, err := conn.ReadMessage()
	if err != nil {
		capture.serverErr = err
		return
	}
	if err = json.Unmarshal(raw, &capture.update); err != nil {
		capture.serverErr = err
		return
	}
	if err = conn.WriteJSON(map[string]string{"type": "session.updated"}); err != nil {
		capture.serverErr = err
		return
	}

	var pendingAudio []byte
	var itemIDs []string
	deltaSent := false
	precommitPartialsSent := false
	for {
		_, raw, err = conn.ReadMessage()
		if err != nil {
			if !missingAck && !rejectFirstCommit && websocket.IsUnexpectedCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				capture.serverErr = err
			}
			return
		}
		var message struct {
			Type    string `json:"type"`
			Audio   string `json:"audio"`
			EventID string `json:"event_id"`
		}
		if err = json.Unmarshal(raw, &message); err != nil {
			capture.serverErr = err
			return
		}
		switch message.Type {
		case "input_audio_buffer.append":
			capture.appends++
			chunk, decodeErr := base64.StdEncoding.DecodeString(message.Audio)
			if decodeErr != nil {
				capture.serverErr = decodeErr
				return
			}
			capture.audio = append(capture.audio, chunk...)
			pendingAudio = append(pendingAudio, chunk...)
			if delta != "" && !deltaSent {
				deltaSent = true
				err = conn.WriteJSON(map[string]any{
					"type":          "conversation.item.input_audio_transcription.delta",
					"item_id":       fmt.Sprintf("item-%d", capture.commits+1),
					"content_index": deltaContentIndex,
					"delta":         delta,
				})
				if err != nil {
					capture.serverErr = err
					return
				}
			}
			if precommitPartials && !precommitPartialsSent && capture.commits == 1 {
				firstID := itemIDs[0]
				err = conn.WriteJSON(map[string]any{
					"type": "conversation.item.input_audio_transcription.delta", "item_id": firstID, "content_index": 0, "delta": "first partial",
				})
				if err == nil {
					err = conn.WriteJSON(map[string]string{"type": "input_audio_buffer.committed", "item_id": firstID})
				}
				if err == nil {
					err = conn.WriteJSON(map[string]any{
						"type": "conversation.item.input_audio_transcription.delta", "item_id": fmt.Sprintf("item-%d", capture.commits+1), "content_index": 0, "delta": "second partial",
					})
				}
				if err != nil {
					capture.serverErr = err
					return
				}
				precommitPartialsSent = true
			}
		case "input_audio_buffer.commit":
			capture.commits++
			capture.turns = append(capture.turns, pendingAudio)
			pendingAudio = nil
			if capture.commits == 1 {
				close(firstCommit)
			} else if capture.commits == 2 {
				close(f.secondCommit)
			}
			itemID := fmt.Sprintf("item-%d", capture.commits)
			itemIDs = append(itemIDs, itemID)
			capture.commitIDs = append(capture.commitIDs, message.EventID)
			if mismatchCommitID {
				itemID += "-different"
			}
			if busy {
				err = conn.WriteJSON(map[string]any{"type": "error", "error": map[string]string{"code": "server_busy"}})
			} else if rejectFirstCommit {
				if capture.commits == 2 {
					err = conn.WriteJSON(map[string]any{
						"type":     "error",
						"event_id": "server-error-1",
						"error": map[string]string{
							"event_id": capture.commitIDs[0],
							"code":     "invalid_request_error",
							"message":  "commit rejected",
						},
					})
				}
			} else if missingAck || (precommitPartials && capture.commits == 1) {
				err = nil
			} else {
				err = conn.WriteJSON(map[string]string{"type": "input_audio_buffer.committed", "item_id": itemID})
				if err == nil && postCommitDelta != "" {
					err = conn.WriteJSON(map[string]any{
						"type":          "conversation.item.input_audio_transcription.delta",
						"item_id":       itemID,
						"content_index": 0,
						"delta":         postCommitDelta,
					})
				}
				switch {
				case err != nil:
				case capture.commits == 1 && holdFirstUntilSecondCommit:
				case capture.commits == 2 && holdFirstUntilSecondCommit:
					if interleavePartials && !precommitPartialsSent {
						err = conn.WriteJSON(map[string]any{
							"type": "conversation.item.input_audio_transcription.delta", "item_id": itemIDs[1], "content_index": 0, "delta": "second partial",
						})
						if err == nil {
							err = conn.WriteJSON(map[string]any{
								"type": "conversation.item.input_audio_transcription.delta", "item_id": itemIDs[0], "content_index": 0, "delta": "first partial",
							})
						}
					}
					if err == nil {
						err = writeFakeFinal(conn, itemIDs[1], "turn-2")
					}
					if err == nil && failFirst {
						err = writeFakeFailure(conn, itemIDs[0], "audio_unintelligible")
					} else if err == nil {
						err = writeFakeFinal(conn, itemIDs[0], "turn-1")
					}
				default:
					if failFirst && capture.commits == 1 {
						err = writeFakeFailure(conn, itemIDs[0], "audio_unintelligible")
					} else {
						transcript := fmt.Sprintf("turn-%d", capture.commits)
						if emptyFinal {
							transcript = ""
						}
						err = writeFakeFinal(conn, itemID, transcript)
					}
				}
			}
			deltaSent = false
			if err != nil {
				capture.serverErr = err
				return
			}
		}
	}
}

func writeFakeFinal(conn *websocket.Conn, itemID, transcript string) error {
	return conn.WriteJSON(map[string]string{
		"type":       "conversation.item.input_audio_transcription.completed",
		"item_id":    itemID,
		"transcript": transcript,
	})
}

func writeFakeFailure(conn *websocket.Conn, itemID, code string) error {
	return conn.WriteJSON(map[string]any{
		"type":          "conversation.item.input_audio_transcription.failed",
		"item_id":       itemID,
		"content_index": 0,
		"error":         map[string]string{"code": code, "message": "transcription failed"},
	})
}

func (s *RealtimeSocketSuite) build() *Realtime {
	provider, err := NewRealtime(Options{
		BaseURL:  s.fake.server.URL + "/v1",
		APIKey:   "test-key",
		Model:    "whisper-1",
		Language: "en",
	})
	s.Require().NoError(err)
	return provider.(*Realtime)
}

func (s *RealtimeSocketSuite) connect() *Realtime {
	provider := s.build()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
	return provider
}

func (s *RealtimeSocketSuite) TestExplicitKeylessEndpointDoesNotReceiveTheEnvironmentKey() {
	s.T().Setenv("OPENAI_STT_API_KEY", "ambient-secret")
	provider, err := NewRealtime(Options{BaseURL: s.fake.server.URL + "/v1", Model: "whisper-1"})
	s.Require().NoError(err)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
	s.NoError(provider.Close())

	capture := s.capture()
	s.NoError(capture.serverErr)
	s.Empty(capture.authority)
}

func (s *RealtimeSocketSuite) TestSessionUpdateUsesExplicitPluralLanguagesAndStandardHints() {
	provider, err := NewRealtime(Options{
		BaseURL:   s.fake.server.URL + "/v1",
		Model:     "gpt-live-transcribe",
		Language:  "en",
		Languages: []string{"en", "fr"},
		Settings: TranscriptionSettings{
			Prompt:    "A customer call about account AC-42.",
			Languages: []string{"en", "fr"},
			Keywords:  []string{"AC-42", "premium plan"},
			Delay:     "low",
			Include:   []string{"item.input_audio_transcription.logprobs"},
		},
	})
	s.Require().NoError(err)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
	s.NoError(provider.Close())

	capture := s.capture()
	s.NoError(capture.serverErr)
	transcription := capture.update.Session.Audio.Input.Transcription
	s.Equal("gpt-live-transcribe", transcription.Model)
	s.Empty(transcription.Language)
	s.Equal([]string{"en", "fr"}, transcription.Languages)
	s.Equal("A customer call about account AC-42.", transcription.Prompt)
	s.Equal([]string{"AC-42", "premium plan"}, transcription.Keywords)
	s.Equal("low", transcription.Delay)
	s.Equal([]string{"item.input_audio_transcription.logprobs"}, capture.update.Session.Include)
	s.Nil(capture.update.Session.Audio.Input.TurnDetection)
	s.Nil(capture.update.Session.Audio.Input.NoiseReduction)
}

func (s *RealtimeSocketSuite) TestRealtimeRejectsBatchOnlyTranscriptionSettings() {
	temperature := 0.2
	stream := false
	cases := []struct {
		name     string
		settings TranscriptionSettings
	}{
		{name: "temperature", settings: TranscriptionSettings{Temperature: &temperature}},
		{name: "response format", settings: TranscriptionSettings{ResponseFormat: "verbose_json"}},
		{name: "timestamp granularities", settings: TranscriptionSettings{TimestampGranularities: []string{"word"}}},
		{name: "chunking strategy", settings: TranscriptionSettings{ChunkingStrategy: []byte(`{"type":"auto"}`)}},
		{name: "known speaker names", settings: TranscriptionSettings{KnownSpeakerNames: []string{"speaker"}}},
		{name: "known speaker references", settings: TranscriptionSettings{KnownSpeakerReferences: []string{"AQID"}}},
		{name: "streaming", settings: TranscriptionSettings{Stream: &stream}},
	}
	for _, testCase := range cases {
		s.Run(testCase.name, func() {
			provider, err := NewRealtime(Options{
				BaseURL:  "https://api.openai.com/v1",
				Settings: testCase.settings,
			})
			s.Nil(provider)
			s.ErrorContains(err, "batch-only transcription settings")
		})
	}
}

func (s *RealtimeSocketSuite) TestRealtimeLeavesLanguageValidationToTheEndpoint() {
	provider, err := NewRealtime(Options{
		BaseURL:  "https://api.openai.com/v1",
		Language: "en-US",
	})
	s.Require().NoError(err)
	s.NotNil(provider)
}

func (s *RealtimeSocketSuite) process(provider *Realtime, samples []int16, participant stt.Participant) {
	s.Require().NoError(s.tryProcess(provider, samples, participant))
}

func (s *RealtimeSocketSuite) tryProcess(provider *Realtime, samples []int16, participant stt.Participant) error {
	// Odd-sized chunks exercise both frame assembly and the streaming resampler carry.
	for len(samples) > 0 {
		count := min(len(samples), 137)
		err := provider.ProcessAudio(stt.PcmData{
			Samples: samples[:count], SampleRate: stt.SampleRate, Channels: 1,
		}, participant)
		if err != nil {
			return err
		}
		samples = samples[count:]
	}
	return nil
}

func (s *RealtimeSocketSuite) capture() realtimeCapture {
	select {
	case capture := <-s.fake.captures:
		return capture
	case <-time.After(5 * time.Second):
		s.FailNow("the websocket server did not finish")
		return realtimeCapture{}
	}
}

func (s *RealtimeSocketSuite) final(provider *Realtime) stt.Transcript {
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				s.FailNow("the session closed before a final transcript")
				return stt.Transcript{}
			}
			if failure, ok := event.(stt.Error); ok {
				s.FailNow("the session reported an error before a final transcript: " + failure.Error())
				return stt.Transcript{}
			}
			if transcript, ok := event.(stt.Transcript); ok && transcript.Final() {
				return transcript
			}
		case <-deadline:
			s.FailNow("timed out waiting for the final transcript")
			return stt.Transcript{}
		}
	}
}

func (s *RealtimeSocketSuite) transcript(provider *Realtime, mode stt.Mode) stt.Transcript {
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				s.FailNow("the session closed before the expected transcript revision")
				return stt.Transcript{}
			}
			if failure, ok := event.(stt.Error); ok {
				s.FailNow("the session reported an error before the expected transcript revision: " + failure.Error())
				return stt.Transcript{}
			}
			if transcript, ok := event.(stt.Transcript); ok && transcript.Mode == mode {
				return transcript
			}
		case <-deadline:
			s.FailNow("timed out waiting for the expected transcript revision")
			return stt.Transcript{}
		}
	}
}

func (s *RealtimeSocketSuite) failure(provider *Realtime) stt.Error {
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event, open := <-provider.Events():
			if !open {
				s.FailNow("the session closed before reporting an error")
				return stt.Error{}
			}
			if failure, ok := event.(stt.Error); ok {
				return failure
			}
		case <-deadline:
			s.FailNow("timed out waiting for a session error")
			return stt.Error{}
		}
	}
}

func (s *RealtimeSocketSuite) TestTheOpenSessionCommitsAllAudioAndAttributesItsTranscript() {
	provider := s.connect()
	participant := stt.Participant{ID: "speaker-a", UserID: "user-a", Name: "A"}
	quiet := make([]int16, stt.SampleRate*240/1000)
	for i := range quiet {
		quiet[i] = 100
	}
	speech := make([]int16, stt.SampleRate*320/1000)
	for i := range speech {
		speech[i] = 2000
	}
	silence := make([]int16, stt.SampleRate*600/1000)
	s.process(provider, append(append(quiet, speech...), silence...), participant)

	transcript := s.final(provider)
	s.Equal("turn-1", transcript.Text)
	s.Equal(participant, transcript.Participant)
	s.Equal(int64(1), transcript.Utterance)
	s.InDelta(1120, transcript.AudioDurationMs, 0.1, "the final covers 200ms of pre-roll, speech, and the endpointing silence")
	s.NoError(provider.Close())

	capture := s.capture()
	s.NoError(capture.serverErr)
	s.Equal("/v1/realtime", capture.path)
	s.Equal("intent=transcription", capture.query)
	s.Equal("Bearer test-key", capture.authority)
	s.Equal("session.update", capture.update.Type)
	s.Equal("transcription", capture.update.Session.Type)
	s.Equal("audio/pcm", capture.update.Session.Audio.Input.Format.Type)
	s.Equal(24000, capture.update.Session.Audio.Input.Format.Rate)
	s.Equal("whisper-1", capture.update.Session.Audio.Input.Transcription.Model)
	s.Equal("en", capture.update.Session.Audio.Input.Transcription.Language)
	s.Greater(capture.appends, 1)
	s.Equal(1, capture.commits)
	s.Len(capture.audio, 2*stt.SampleRate*1120*3/2000, "all chunks and the pre-roll must reach the one commit")
	s.Equal(int16(100), int16(binary.LittleEndian.Uint16(capture.audio[:2])), "the quiet pre-roll should preserve a soft onset")
}

func (s *RealtimeSocketSuite) TestDeltasKeepTheirSpacesAndFinalTextReconcilesTheTurn() {
	s.fake.mu.Lock()
	s.fake.delta = " hello "
	s.fake.postCommitDelta = "world"
	s.fake.mu.Unlock()

	provider := s.connect()
	participant := stt.Participant{ID: "speaker-a", UserID: "user-a", Name: "A"}
	speech := make([]int16, stt.SampleRate*1200/1000)
	for i := range speech {
		speech[i] = 2000
	}
	s.process(provider, speech, participant)

	partial := s.transcript(provider, stt.ModeDelta)
	s.Equal(" hello ", partial.Text)
	s.Equal(participant, partial.Participant)
	s.Equal(int64(1), partial.Utterance)

	s.process(provider, make([]int16, stt.SampleRate*600/1000), participant)
	postCommit := s.transcript(provider, stt.ModeDelta)
	s.Equal("world", postCommit.Text)
	s.Equal(participant, postCommit.Participant)
	s.Equal(partial.Utterance, postCommit.Utterance)

	final := s.final(provider)
	s.Equal(stt.ModeFinal, final.Mode)
	s.Equal("turn-1", final.Text)
	s.Equal(participant, final.Participant)
	s.Equal(partial.Utterance, final.Utterance)
	s.NoError(provider.Close())
	capture := s.capture()
	s.NoError(capture.serverErr)
	s.Equal(1, capture.commits)
}

func (s *RealtimeSocketSuite) TestAnEmptyFinalClearsAPreviouslyPublishedDelta() {
	s.fake.mu.Lock()
	s.fake.delta = "speculative words"
	s.fake.emptyFinal = true
	s.fake.mu.Unlock()

	provider := s.connect()
	participant := stt.Participant{ID: "speaker-a"}
	speech := make([]int16, stt.SampleRate*1200/1000)
	for i := range speech {
		speech[i] = 2000
	}
	s.process(provider, speech, participant)
	partial := s.transcript(provider, stt.ModeDelta)
	s.Equal("speculative words", partial.Text)

	s.process(provider, make([]int16, stt.SampleRate*600/1000), participant)
	final := s.final(provider)
	s.Empty(final.Text)
	s.Equal(stt.ModeFinal, final.Mode)
	s.Equal(partial.Utterance, final.Utterance)
	s.NoError(provider.Close())
	_ = s.capture()
}

func (s *RealtimeSocketSuite) TestACommitMustKeepTheProvisionalDeltaItemID() {
	s.fake.mu.Lock()
	s.fake.delta = "hello"
	s.fake.mismatchCommitID = true
	s.fake.mu.Unlock()

	provider := s.connect()
	participant := stt.Participant{ID: "speaker-a"}
	speech := make([]int16, stt.SampleRate*1200/1000)
	for i := range speech {
		speech[i] = 2000
	}
	s.process(provider, speech, participant)
	_ = s.transcript(provider, stt.ModeDelta)
	s.process(provider, make([]int16, stt.SampleRate*600/1000), participant)

	failure := s.failure(provider)
	s.True(failure.Fatal)
	s.Contains(failure.Error(), "commit acknowledgement")
	_ = provider.Close()
	_ = s.capture()
}

func (s *RealtimeSocketSuite) TestAReatimeDeltaRejectsAnUnsupportedContentIndex() {
	s.fake.mu.Lock()
	s.fake.delta = "hello"
	s.fake.deltaContentIndex = 1
	s.fake.mu.Unlock()

	provider := s.connect()
	participant := stt.Participant{ID: "speaker-a"}
	speech := make([]int16, stt.SampleRate*1200/1000)
	for i := range speech {
		speech[i] = 2000
	}
	_ = s.tryProcess(provider, speech, participant) // The expected fatal protocol error may race with input processing.
	failure := s.failure(provider)
	s.True(failure.Fatal)
	s.Contains(failure.Error(), "content index")
	_ = provider.Close()
	_ = s.capture()
}

func (s *RealtimeSocketSuite) TestCloseCommitsTheUnfinishedTail() {
	provider := s.connect()
	participant := stt.Participant{ID: "speaker-a"}
	speech := make([]int16, stt.SampleRate*160/1000)
	for i := range speech {
		speech[i] = 2000
	}
	s.process(provider, speech, participant)
	s.NoError(provider.Close())

	transcript := s.final(provider)
	s.Equal(participant, transcript.Participant)
	s.InDelta(160, transcript.AudioDurationMs, 0.1)
	capture := s.capture()
	s.NoError(capture.serverErr)
	s.Equal(1, capture.commits)
	s.Len(capture.audio, 2*stt.SampleRate*160*3/2000)
}

func (s *RealtimeSocketSuite) TestMultipleTurnsAreSentBeforeEarlierFinalAndCloseWaitsForBoth() {
	s.fake.mu.Lock()
	s.fake.holdFirstUntilSecondCommit = true
	s.fake.interleavePartials = true
	s.fake.precommitPartials = true
	s.fake.mu.Unlock()
	provider := s.connect()
	firstParticipant := stt.Participant{ID: "speaker-a"}
	secondParticipant := stt.Participant{ID: "speaker-b"}
	first := make([]int16, stt.SampleRate*160/1000)
	second := make([]int16, stt.SampleRate*160/1000)
	for i := range first {
		first[i] = 2000
	}
	for i := range second {
		second[i] = 3000
	}
	s.process(provider, append(first, make([]int16, stt.SampleRate*600/1000)...), firstParticipant)
	select {
	case <-s.fake.firstCommit:
	case <-time.After(5 * time.Second):
		s.FailNow("the first turn was not committed")
	}
	s.process(provider, append(second, make([]int16, stt.SampleRate*600/1000)...), secondParticipant)
	select {
	case <-s.fake.secondCommit:
	case <-time.After(5 * time.Second):
		s.FailNow("the second turn was not committed while the first final was held")
	}

	closed := make(chan error, 1)
	go func() { closed <- provider.Close() }()
	select {
	case err := <-closed:
		s.NoError(err)
	case <-time.After(5 * time.Second):
		s.FailNow("Close did not wait for both transcriptions")
	}

	var finals []stt.Transcript
	var partials []stt.Transcript
	for event := range provider.Events() {
		if transcript, ok := event.(stt.Transcript); ok {
			if transcript.Final() {
				finals = append(finals, transcript)
			} else if transcript.Mode == stt.ModeDelta {
				partials = append(partials, transcript)
			}
		}
	}
	s.Require().Len(finals, 2)
	s.Require().Len(partials, 2)
	s.Equal([]string{"turn-2", "turn-1"}, []string{finals[0].Text, finals[1].Text})
	s.Equal([]stt.Participant{secondParticipant, firstParticipant}, []stt.Participant{
		finals[0].Participant, finals[1].Participant,
	})
	s.Equal([]stt.Participant{firstParticipant, secondParticipant}, []stt.Participant{
		partials[0].Participant, partials[1].Participant,
	})
	s.Equal([]int64{2, 1}, []int64{finals[0].Utterance, finals[1].Utterance})
	s.Equal([]int64{1, 2}, []int64{partials[0].Utterance, partials[1].Utterance})
	s.Equal([]string{"first partial", "second partial"}, []string{partials[0].Text, partials[1].Text})
	capture := s.capture()
	s.NoError(capture.serverErr)
	s.Equal(2, capture.commits)
	s.Equal(2, len(capture.turns))
	s.Len(capture.turns[0], 2*stt.SampleRate*760*3/2000)
	s.Len(capture.turns[1], 2*stt.SampleRate*760*3/2000)
}

func (s *RealtimeSocketSuite) TestAnItemFailureDoesNotStopOtherCommittedTurns() {
	s.fake.mu.Lock()
	s.fake.holdFirstUntilSecondCommit = true
	s.fake.failFirst = true
	s.fake.interleavePartials = true
	s.fake.mu.Unlock()
	provider := s.connect()
	firstParticipant := stt.Participant{ID: "speaker-a"}
	secondParticipant := stt.Participant{ID: "speaker-b"}
	first := make([]int16, stt.SampleRate*160/1000)
	second := make([]int16, stt.SampleRate*160/1000)
	for i := range first {
		first[i] = 2000
		second[i] = 3000
	}
	s.process(provider, append(first, make([]int16, stt.SampleRate*600/1000)...), firstParticipant)
	s.process(provider, append(second, make([]int16, stt.SampleRate*600/1000)...), secondParticipant)
	select {
	case <-s.fake.secondCommit:
	case <-time.After(5 * time.Second):
		s.FailNow("the second turn was not committed")
	}
	s.NoError(provider.Close())

	var finals []stt.Transcript
	var failures []stt.Error
	for event := range provider.Events() {
		switch observed := event.(type) {
		case stt.Transcript:
			if observed.Final() {
				finals = append(finals, observed)
			}
		case stt.Error:
			failures = append(failures, observed)
		}
	}
	s.Require().Len(finals, 2)
	s.Equal([]string{"turn-2", ""}, []string{finals[0].Text, finals[1].Text})
	s.Equal([]stt.Participant{secondParticipant, firstParticipant}, []stt.Participant{
		finals[0].Participant, finals[1].Participant,
	})
	s.InDelta(760, finals[1].AudioDurationMs, 0.1)
	s.Greater(finals[1].ProcessingTimeMs, float64(0))
	s.Require().Len(failures, 1)
	s.False(failures[0].Fatal)
	s.Contains(failures[0].Error(), "audio_unintelligible")
	capture := s.capture()
	s.NoError(capture.serverErr)
	s.Equal(2, capture.commits)
}

func (s *RealtimeSocketSuite) TestCommitRejectionUsesNestedClientEventIDAndFailsTheSession() {
	s.fake.mu.Lock()
	s.fake.rejectFirstCommit = true
	s.fake.mu.Unlock()
	provider := s.connect()
	participant := stt.Participant{ID: "speaker"}
	first := make([]int16, stt.SampleRate*160/1000)
	second := make([]int16, stt.SampleRate*160/1000)
	for i := range first {
		first[i] = 2000
		second[i] = 3000
	}
	s.process(provider, append(first, make([]int16, stt.SampleRate*600/1000)...), participant)
	select {
	case <-s.fake.firstCommit:
	case <-time.After(5 * time.Second):
		s.FailNow("the first turn was not committed")
	}
	_ = s.tryProcess(provider, append(second, make([]int16, stt.SampleRate*600/1000)...), participant)
	select {
	case <-s.fake.secondCommit:
	case <-time.After(5 * time.Second):
		s.FailNow("the second pipelined turn was not committed")
	}
	failure := s.failure(provider)
	s.True(failure.Fatal)
	s.Contains(failure.Error(), "realtime commit was rejected")
	s.Contains(failure.Error(), "invalid_request_error")
	_ = provider.Close()

	capture := s.capture()
	s.Equal(2, capture.commits)
	s.Equal([]string{"realtime-commit-1", "realtime-commit-2"}, capture.commitIDs)
}

func (s *RealtimeSocketSuite) TestMissingCommitAcknowledgementTimesOutTheOldestPendingTurn() {
	oldTimeout := realtimeInferenceTimeout
	realtimeInferenceTimeout = time.Second
	defer func() { realtimeInferenceTimeout = oldTimeout }()
	s.fake.mu.Lock()
	s.fake.missingAck = true
	s.fake.mu.Unlock()
	provider := s.connect()
	speech := make([]int16, stt.SampleRate*160/1000)
	for i := range speech {
		speech[i] = 2000
	}
	s.process(provider, append(speech, make([]int16, stt.SampleRate*600/1000)...), stt.Participant{ID: "speaker"})
	select {
	case <-s.fake.firstCommit:
	case <-time.After(5 * time.Second):
		s.FailNow("the first turn was not committed")
	}
	time.Sleep(600 * time.Millisecond)
	s.process(provider, append(speech, make([]int16, stt.SampleRate*600/1000)...), stt.Participant{ID: "speaker"})
	select {
	case <-s.fake.secondCommit:
	case <-time.After(5 * time.Second):
		s.FailNow("the second turn was not committed")
	}
	started := time.Now()
	failure := s.failure(provider)
	s.Less(time.Since(started), 700*time.Millisecond, "a later commit must not reset the first turn's deadline")
	s.True(failure.Fatal)
	s.Contains(failure.Error(), "inference timed out")
	capture := s.capture()
	s.NoError(capture.serverErr)
	s.Equal(2, capture.commits)
}

func (s *RealtimeSocketSuite) TestServerBusyIsAVisibleFatalError() {
	s.fake.mu.Lock()
	s.fake.busy = true
	s.fake.mu.Unlock()
	provider := s.connect()
	speech := make([]int16, stt.SampleRate*160/1000)
	for i := range speech {
		speech[i] = 2000
	}
	s.process(provider, speech, stt.Participant{ID: "speaker-a"})
	s.Require().NoError(provider.Close())

	var busy *stt.Error
	for event := range provider.Events() {
		if observed, ok := event.(stt.Error); ok {
			busy = &observed
			break
		}
	}
	s.Require().NotNil(busy)
	s.True(busy.Fatal)
	s.Contains(busy.Error(), "server is busy")
	_ = s.capture()
}

func (s *RealtimeSocketSuite) TestCloseDeadlineUnblocksAFullEventStream() {
	oldTimeout := realtimeCloseTimeout
	realtimeCloseTimeout = 50 * time.Millisecond
	defer func() { realtimeCloseTimeout = oldTimeout }()

	provider := s.build()
	for i := 0; i < 63; i++ {
		provider.emitter.Send(stt.Connected{Provider: ProviderName, Model: provider.Model()})
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	s.T().Cleanup(cancel)
	s.Require().NoError(provider.Start(ctx))
	speech := make([]int16, stt.SampleRate*160/1000)
	for i := range speech {
		speech[i] = 2000
	}
	s.process(provider, speech, stt.Participant{ID: "speaker-a"})

	started := time.Now()
	err := provider.Close()
	s.ErrorContains(err, "realtime close timed out")
	s.Less(time.Since(started), time.Second, "the deadline must include a blocked event send")
	_ = s.capture()
}
