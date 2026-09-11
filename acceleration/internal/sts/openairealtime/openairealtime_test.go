package openairealtime

import (
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type RealtimeSuite struct {
	suite.Suite
}

func TestRealtimeSuite(t *testing.T) {
	suite.Run(t, new(RealtimeSuite))
}

// newProvider returns a provider that is wired up but never connected, so the event
// mapping can be exercised without touching the network.
func (s *RealtimeSuite) newProvider(vendor Vendor) *STS {
	provider, err := New(Options{Vendor: vendor, APIKey: "test-key", Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	return provider
}

// drain collects the events emitted so far without blocking on an empty channel.
func (s *RealtimeSuite) drain(provider *STS) []sts.Event {
	var events []sts.Event
	for {
		select {
		case event := <-provider.Events():
			events = append(events, event)
		default:
			return events
		}
	}
}

func created(id string) serverEvent {
	return serverEvent{Type: eventResponseCreated, Response: &response{ID: id}}
}

func done(id, status string, cost *usage) serverEvent {
	return serverEvent{Type: eventResponseDone, Response: &response{ID: id, Status: status, Usage: cost}}
}

func audioDelta(id string, samples int) serverEvent {
	return serverEvent{
		Type:       eventAudioDelta,
		ResponseID: id,
		ItemID:     "item_1",
		Delta:      base64.StdEncoding.EncodeToString(make([]byte, samples*2)),
	}
}

func (s *RealtimeSuite) TestNewRequiresAnAPIKey() {
	s.T().Setenv("OPENAI_API_KEY", "")
	_, err := New(Options{Vendor: OpenAI})
	s.ErrorContains(err, "api key is required (set OPENAI_API_KEY)")
}

func (s *RealtimeSuite) TestNewFallsBackToTheVendorsOwnVariable() {
	s.T().Setenv("XAI_API_KEY", "key-from-env")
	provider, err := New(Options{Vendor: XAI})
	s.Require().NoError(err)
	s.Equal("key-from-env", provider.options.APIKey)
	s.Equal(XAI.Model, provider.Model(), "the vendor's default model stands in for none")
}

func (s *RealtimeSuite) TestNewRefusesWhatTheVendorCannotDo() {
	_, err := New(Options{Vendor: XAI, APIKey: "k", TurnDetection: options.TurnSemantic})
	s.ErrorContains(err, "no semantic turn detector")

	_, err = New(Options{Vendor: Qwen, APIKey: "k", Tools: []llm.Tool{{Name: "x"}}})
	s.ErrorContains(err, "does not call tools")

	_, err = New(Options{Vendor: OpenAI, APIKey: "k", TurnDetection: options.TurnManual})
	s.ErrorContains(err, "manual turns are not supported")
}

func (s *RealtimeSuite) TestCapabilitiesDifferPerVendor() {
	s.True(CapabilitiesFor(OpenAI, "gpt-realtime-2").SemanticTurns)
	s.True(CapabilitiesFor(OpenAI, "gpt-realtime-2").Accepts(options.ModalityImage))
	s.False(CapabilitiesFor(XAI, "grok-voice-think-fast-2.0").SemanticTurns)
	s.False(CapabilitiesFor(XAI, "grok-voice-think-fast-2.0").Accepts(options.ModalityImage))
	s.False(CapabilitiesFor(Qwen, "qwen3.5-omni-plus-realtime").Text, "Qwen takes no typed turns")
	s.False(CapabilitiesFor(Qwen, "qwen3.5-omni-plus-realtime").Tools, "Qwen calls no tools")
	s.Equal(60*time.Minute, CapabilitiesFor(OpenAI, "gpt-realtime-2").MaxDuration)
}

func (s *RealtimeSuite) TestTheEndpointCarriesTheModel() {
	provider := s.newProvider(OpenAI)
	s.Equal(OpenAI.URL+"?model=gpt-realtime-2", provider.endpoint())
	s.Equal(24_000, provider.SampleRate())
}

func (s *RealtimeSuite) TestWhatTheModelCannotDoIsRefusedNotDropped() {
	provider := s.newProvider(Qwen)
	s.ErrorIs(provider.SendText("hello", sts.Participant{}), sts.ErrNoText)
	s.ErrorIs(provider.Prompt("greet"), sts.ErrNoText)
	s.ErrorIs(provider.SetTools(nil), sts.ErrNoTools)
	s.ErrorIs(provider.Answer("c", "", nil), sts.ErrNoTools)
	s.ErrorIs(provider.SetInstructions("x"), sts.ErrInstructionsFixed)

	xai := s.newProvider(XAI)
	s.ErrorIs(xai.SendFrame(llm.ImagePart{Data: []byte{1}}), sts.ErrNoImages)
}

func (s *RealtimeSuite) TestTheReplyIsTimedFromWhenTheCallerStopped() {
	provider := s.newProvider(OpenAI)
	provider.participant = sts.Participant{ID: "alice"}

	provider.handleMessage(serverEvent{Type: eventSpeechStarted})
	provider.handleMessage(serverEvent{Type: eventSpeechStopped})
	time.Sleep(20 * time.Millisecond)
	provider.handleMessage(created("resp_1"))
	provider.handleMessage(audioDelta("resp_1", 2400))
	provider.handleMessage(done("resp_1", "completed", &usage{
		InputTokens:        100,
		OutputTokens:       50,
		InputTokenDetails:  &tokenDetails{CachedTokens: 10, AudioTokens: 80},
		OutputTokenDetails: &tokenDetails{AudioTokens: 45},
	}))

	events := s.drain(provider)
	s.Require().Len(events, 5)

	started, ok := events[0].(sts.SpeechStarted)
	s.Require().True(ok)
	s.Equal("alice", started.Participant.ID)
	_, ok = events[1].(sts.SpeechStopped)
	s.Require().True(ok)

	begun, ok := events[2].(sts.ResponseStarted)
	s.Require().True(ok)
	s.Equal("resp_1", begun.ResponseID)
	s.Equal(1, begun.Generation)

	chunk, ok := events[3].(sts.AudioChunk)
	s.Require().True(ok)
	s.Equal(24_000, chunk.Audio.SampleRate)
	s.Len(chunk.Audio.Samples, 2400)
	s.Equal(1, chunk.Generation)

	complete, ok := events[4].(sts.ResponseComplete)
	s.Require().True(ok)
	s.False(complete.Interrupted)
	s.GreaterOrEqual(complete.TimeToFirstByteMs, 20.0, "the wait began when the caller stopped")
	s.InDelta(100, complete.AudioDurationMs, 0.001)
	s.Equal(sts.Usage{InputTokens: 100, CachedInputTokens: 10, OutputTokens: 50, InputAudioTokens: 80, OutputAudioTokens: 45}, complete.Usage)
}

func (s *RealtimeSuite) TestACommitStandsInForSpeechStoppedUnderSemanticTurns() {
	provider := s.newProvider(OpenAI)

	// A semantic detector commits without ever saying speech stopped.
	provider.handleMessage(serverEvent{Type: eventCommitted})
	// Under a silence timer the stop comes first and the commit adds nothing.
	provider.handleMessage(created("r1"))
	provider.handleMessage(done("r1", "completed", nil))
	provider.handleMessage(serverEvent{Type: eventSpeechStopped})
	provider.handleMessage(serverEvent{Type: eventCommitted})

	var stopped int
	for _, event := range s.drain(provider) {
		if _, ok := event.(sts.SpeechStopped); ok {
			stopped++
		}
	}
	s.Equal(2, stopped, "one stop per turn, whichever event carried it")
}

func (s *RealtimeSuite) TestACancelledReplyIsInterrupted() {
	provider := s.newProvider(OpenAI)

	provider.handleMessage(created("r1"))
	provider.handleMessage(done("r1", "cancelled", nil))

	events := s.drain(provider)
	complete, ok := events[len(events)-1].(sts.ResponseComplete)
	s.Require().True(ok)
	s.True(complete.Interrupted, "a cancelled response is the one thing that means barge-in here")
}

func (s *RealtimeSuite) TestAFailedReplyReportsTheFailureAndStillSettles() {
	provider := s.newProvider(OpenAI)

	provider.handleMessage(created("r1"))
	provider.handleMessage(serverEvent{Type: eventResponseDone, Response: &response{
		ID: "r1", Status: "failed",
		StatusDetails: &statusDetails{Type: "failed", Error: &apiError{Code: "server_error", Message: "boom"}},
	}})

	events := s.drain(provider)
	s.Require().Len(events, 3)
	failure, ok := events[1].(sts.Error)
	s.Require().True(ok)
	s.Equal("r1", failure.ResponseID, "the failure names the reply so it stays one row")
	s.False(failure.Fatal)
	s.ErrorContains(failure.Err, "server_error: boom")
	_, ok = events[2].(sts.ResponseComplete)
	s.True(ok, "a failed reply still settles, so it is still billed once")
}

func (s *RealtimeSuite) TestTranscriptsArriveInBothSpellings() {
	provider := s.newProvider(Qwen)
	provider.participant = sts.Participant{ID: "alice"}

	provider.handleMessage(serverEvent{Type: eventInputCompleted, Transcript: " hello there "})
	provider.handleMessage(created("r1"))
	provider.handleMessage(serverEvent{Type: eventTranscriptDeltaV1, ResponseID: "r1", Delta: "Hi "})
	provider.handleMessage(serverEvent{Type: eventTranscriptDelta, ResponseID: "r1", Delta: "back."})
	provider.handleMessage(serverEvent{Type: eventTranscriptDoneV1, ResponseID: "r1", Transcript: "Hi back."})

	events := s.drain(provider)
	heard, ok := events[0].(sts.InputTranscript)
	s.Require().True(ok)
	s.Equal(stt.ModeFinal, heard.Mode)
	s.Equal("hello there", heard.Text)
	s.Equal("alice", heard.Participant.ID)

	said := []sts.OutputTranscript{}
	for _, event := range events {
		if typed, ok := event.(sts.OutputTranscript); ok {
			said = append(said, typed)
		}
	}
	s.Require().Len(said, 3)
	s.Equal(stt.ModeDelta, said[0].Mode)
	s.Equal(stt.ModeDelta, said[1].Mode)
	s.Equal(stt.ModeFinal, said[2].Mode)
	s.Equal("Hi back.", said[2].Text)
	s.Equal("r1", said[2].ResponseID)
}

func (s *RealtimeSuite) TestARestatedTranscriptSettlesOnlyOnceTheCallerHasStopped() {
	provider := s.newProvider(XAI)
	provider.participant = sts.Participant{ID: "alice"}

	// xAI writes the turn down as it grows, restating it from the beginning each time.
	provider.handleMessage(serverEvent{Type: eventSpeechStarted})
	provider.handleMessage(serverEvent{Type: eventInputCompleted, Transcript: "A quiet"})
	provider.handleMessage(serverEvent{Type: eventInputCompleted, Transcript: "A quiet village"})
	provider.handleMessage(created("r1"))
	provider.handleMessage(serverEvent{Type: eventSpeechStopped})
	provider.handleMessage(serverEvent{Type: eventInputCompleted, Transcript: "A quiet village."})
	provider.handleMessage(audioDelta("r1", 240))

	var heard []sts.InputTranscript
	for _, event := range s.drain(provider) {
		if typed, ok := event.(sts.InputTranscript); ok {
			heard = append(heard, typed)
		}
	}
	s.Require().Len(heard, 3)
	s.Equal(stt.ModeReplacement, heard[0].Mode, "a restatement supersedes the one before it")
	s.Equal(stt.ModeReplacement, heard[1].Mode)
	s.Equal(stt.ModeFinal, heard[2].Mode, "the report after the caller stopped is the settled one")
	s.Equal("A quiet village.", heard[2].Text)
}

func (s *RealtimeSuite) TestARestatementNothingFollowedIsSettledWhenTheReplyBegins() {
	provider := s.newProvider(XAI)

	provider.handleMessage(serverEvent{Type: eventSpeechStarted})
	provider.handleMessage(serverEvent{Type: eventInputCompleted, Transcript: "A quiet village."})
	provider.handleMessage(serverEvent{Type: eventSpeechStopped})
	provider.handleMessage(created("r1"))
	provider.handleMessage(audioDelta("r1", 240))
	provider.handleMessage(done("r1", "completed", nil))

	var heard []sts.InputTranscript
	var complete sts.ResponseComplete
	for _, event := range s.drain(provider) {
		switch typed := event.(type) {
		case sts.InputTranscript:
			heard = append(heard, typed)
		case sts.ResponseComplete:
			complete = typed
		}
	}
	s.Require().Len(heard, 2, "the restatement, then the same words settled once the model spoke")
	s.Equal(stt.ModeReplacement, heard[0].Mode)
	s.Equal(stt.ModeFinal, heard[1].Mode)
	s.Equal("A quiet village.", heard[1].Text)
	s.Greater(complete.TimeToFirstByteMs, 0.0, "a reply opened before the caller stopped is still timed from the stop")
}

func (s *RealtimeSuite) TestAToolCallCarriesItsArguments() {
	provider := s.newProvider(OpenAI)

	provider.handleMessage(created("r1"))
	provider.handleMessage(serverEvent{
		Type: eventArgumentsDone, ResponseID: "r1", CallID: "call_1", Name: "get_weather", Arguments: `{"city":"Paris"}`,
	})

	events := s.drain(provider)
	call, ok := events[1].(sts.ToolCall)
	s.Require().True(ok)
	s.Equal("r1", call.ResponseID)
	s.Equal("call_1", call.CallID)
	s.Equal("get_weather", call.Name)
	s.JSONEq(`{"city":"Paris"}`, call.Arguments)
}

func (s *RealtimeSuite) TestTheFramesXAISendsDecodeDespiteTheirStringDetails() {
	provider := s.newProvider(XAI)

	// xAI spells status_details as a bare string where OpenAI sends an object. Dropping
	// the frame over it would leave the reply open forever.
	var created, done serverEvent
	s.Require().NoError(json.Unmarshal([]byte(`{"type":"response.created","response":{"id":"r1","status":"in_progress","status_details":"in_progress"}}`), &created))
	s.Require().NoError(json.Unmarshal([]byte(`{"type":"response.done","response":{"id":"r1","status":"completed","status_details":"completed","usage":{"input_tokens":3,"output_tokens":4}}}`), &done))
	var detailed serverEvent
	s.Require().NoError(json.Unmarshal([]byte(`{"type":"response.done","response":{"id":"r2","status":"cancelled","status_details":{"type":"cancelled","reason":"turn_detected"}}}`), &detailed))

	provider.handleMessage(created)
	provider.handleMessage(done)
	provider.handleMessage(detailed)

	events := s.drain(provider)
	s.Require().Len(events, 2)
	complete, ok := events[1].(sts.ResponseComplete)
	s.Require().True(ok)
	s.Equal("r1", complete.ResponseID)
	s.False(complete.Interrupted)
	s.Equal(int64(3), complete.Usage.InputTokens)
	s.Equal("turn_detected", detailed.Response.StatusDetails.Reason, "the object form still reads")
}

func (s *RealtimeSuite) TestACancelThatFoundNothingIsNotAnError() {
	provider := s.newProvider(OpenAI)

	provider.handleMessage(serverEvent{Type: eventError, Error: &apiError{Code: cancelNotActive, Message: "nothing"}})
	provider.handleMessage(serverEvent{Type: eventError, Error: &apiError{Code: "invalid_request_error", Message: "bad"}})

	events := s.drain(provider)
	s.Require().Len(events, 1, "only the real error should be reported")
	failure, ok := events[0].(sts.Error)
	s.Require().True(ok)
	s.ErrorContains(failure.Err, "bad")
	s.False(failure.Fatal, "a rejected request does not end the session")
}

func (s *RealtimeSuite) TestAudioWithoutAnOpeningFrameStillOpensAReply() {
	provider := s.newProvider(OpenAI)

	provider.handleMessage(audioDelta("r1", 240))

	events := s.drain(provider)
	s.Require().Len(events, 2)
	_, ok := events[0].(sts.ResponseStarted)
	s.True(ok, "the reply is opened by its first audio when the server skipped the opening frame")
	_, ok = events[1].(sts.AudioChunk)
	s.True(ok)
}

func (s *RealtimeSuite) TestSessionConfigurationIsSpelledPerDialect() {
	eager := "high"
	threshold := 0.6
	silence := 400

	openai, err := New(Options{
		Vendor: OpenAI, APIKey: "k", Voice: "marin", Instructions: "Be brief.",
		TurnDetection: options.TurnSemantic, Eagerness: eager, InputTranscript: true,
		Tools: []llm.Tool{{Name: "get_weather", Description: "Weather.", Parameters: map[string]any{"type": "object"}}},
	})
	s.Require().NoError(err)
	configured := openai.session()
	s.Equal("realtime", configured.Type)
	s.Equal([]string{"audio"}, configured.OutputModalities)
	s.Equal("Be brief.", configured.Instructions)
	s.Require().NotNil(configured.Audio)
	s.Equal(24_000, configured.Audio.Input.Format.Rate)
	s.Equal("audio/pcm", configured.Audio.Input.Format.Type)
	s.Require().NotNil(configured.Audio.Input.Transcription, "asked for, so sent")
	s.Equal(openaiTranscriber, configured.Audio.Input.Transcription.Model,
		"OpenAI refuses a transcriber with no model named, so one is chosen for it")
	s.Equal("semantic_vad", configured.Audio.Input.TurnDetection.Type)
	s.Equal("high", configured.Audio.Input.TurnDetection.Eagerness)
	s.Equal("marin", configured.Audio.Output.Voice)
	s.Require().Len(configured.Tools, 1)
	s.Equal("function", configured.Tools[0].Type)
	s.Equal("get_weather", configured.Tools[0].Name)
	s.Empty(configured.Voice, "OpenAI's voice lives under audio.output")

	xai, err := New(Options{Vendor: XAI, APIKey: "k", Voice: "ara", Threshold: &threshold, SilenceMs: &silence})
	s.Require().NoError(err)
	configured = xai.session()
	s.Equal([]string{"text", "audio"}, configured.Modalities)
	s.Equal("ara", configured.Voice)
	s.Nil(configured.InputAudioTranscription, "not asked for, so left to the vendor")
	s.Require().NotNil(configured.TurnDetection, "a threshold on its own still needs a detector to hang off")
	s.Equal("server_vad", configured.TurnDetection.Type)
	s.Equal(0.6, *configured.TurnDetection.Threshold)
	s.Equal(400, *configured.TurnDetection.SilenceDurationMs)
	s.Nil(configured.TurnDetection.InterruptResponse, "nil leaves the vendor's default")
	s.True(*configured.TurnDetection.CreateResponse, "xAI answers on its own only when told to")
	s.Empty(configured.Type)

	quiet, err := New(Options{Vendor: XAI, APIKey: "k"})
	s.Require().NoError(err)
	s.Require().NotNil(quiet.session().TurnDetection, "even a session asking for nothing gets a detector that answers")
	s.Equal("server_vad", quiet.session().TurnDetection.Type)
	s.Nil(openai.session().Audio.Input.TurnDetection.CreateResponse, "OpenAI's own default already answers")

	qwen, err := New(Options{Vendor: Qwen, APIKey: "k", Voice: "Cherry"})
	s.Require().NoError(err)
	configured = qwen.session()
	s.Equal("pcm16", configured.InputAudioFormat)
	s.Equal("pcm24", configured.OutputAudioFormat)
	s.Require().NotNil(configured.InputAudioTranscription, "Qwen always transcribes")
	s.Equal(qwenTranscriber, configured.InputAudioTranscription.Model)
	s.Nil(configured.Audio)

	// Nothing about the wire shape may leak a field a dialect does not have.
	encoded, err := json.Marshal(clientEvent{Type: eventSessionUpdate, Session: configured})
	s.Require().NoError(err)
	s.NotContains(string(encoded), `"type":"realtime"`)
	s.NotContains(string(encoded), `"output_modalities"`)
}

func (s *RealtimeSuite) TestCloseSettlesAReplyInFlightAsInterrupted() {
	provider := s.newProvider(OpenAI)
	provider.handleMessage(created("r1"))

	s.Require().NoError(provider.Close())

	var events []sts.Event
	for event := range provider.Events() {
		events = append(events, event)
	}
	s.Require().Len(events, 2)
	complete, ok := events[1].(sts.ResponseComplete)
	s.Require().True(ok)
	s.True(complete.Interrupted, "a reply cut off by hanging up is still billed once")
}
