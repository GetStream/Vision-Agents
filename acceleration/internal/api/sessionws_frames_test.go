package api

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

func TestASettledTaskListsTheFilesItsCodeHandedBack(t *testing.T) {
	render := sandbox.Attachment{Name: "teapot.png", MIME: "image/png", URL: "https://cdn.example/teapot.png", Size: 3}

	sent, ok := frameOf(agent.TaskSettled{TaskID: "task-1", Skill: "render", Text: "A teapot.", Files: []sandbox.Attachment{render}})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.JSONEq(t, `[{"name":"teapot.png","mime_type":"image/png","url":"https://cdn.example/teapot.png","size":3}]`,
		string(mustField(t, raw, "files")))
}

func TestASettledTaskWithNoFilesSaysSoWithAnEmptyList(t *testing.T) {
	sent, ok := frameOf(agent.TaskSettled{TaskID: "task-1", Text: "Twelve."})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.JSONEq(t, `[]`, string(mustField(t, raw, "files")))
}

func TestAToolStartedWithoutPreSpeechIsTheFrameItAlwaysWas(t *testing.T) {
	// The bytes base sends, for a tool whose binding names no phrase or that is no
	// connector's at all.
	at := time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC)

	sent, ok := frameOf(agent.ToolStarted{ID: "call-1", TurnID: "turn-1", Tool: "crm__slow", StartedAt: at})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.Equal(t, `{"started_at":"2026-10-07T12:00:00Z","tool":"crm__slow","tool_call_id":"call-1","turn_id":"turn-1","type":"tool_started"}`,
		string(raw))
}

func TestAToolStartedCarriesItsBindingsPreSpeech(t *testing.T) {
	at := time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC)

	sent, ok := frameOf(agent.ToolStarted{ID: "call-1", TurnID: "turn-1", Tool: "crm__slow", StartedAt: at,
		PreSpeech: "Let me pull that up."})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.Equal(t, `"Let me pull that up."`, string(mustField(t, raw, "pre_speech")))
}

func mustField(t *testing.T, raw []byte, name string) json.RawMessage {
	t.Helper()
	var fields map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(raw, &fields))
	require.Contains(t, fields, name)
	return fields[name]
}

func TestATurnFrameCarriesWhenTheFirstFrameWasQueuedAndHeard(t *testing.T) {
	sent, ok := frameOf(agent.Turn{
		TurnID:               "turn-1",
		RoundtripMs:          1400,
		SpeechEndToAudioMs:   1520,
		FirstFrameQueuedMs:   880,
		FirstAudibleFrameMs:  940,
		SpeechEndToAudibleMs: 1060,
	})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.JSONEq(t, `1400`, string(mustField(t, raw, "roundtrip_ms")), "the figure taken at the return is kept")
	require.JSONEq(t, `880`, string(mustField(t, raw, "first_frame_queued_ms")))
	require.JSONEq(t, `940`, string(mustField(t, raw, "first_audible_frame_ms")))
	require.JSONEq(t, `1060`, string(mustField(t, raw, "speech_end_to_audible_ms")))
}

func TestATimelineEntryCarriesWhenTheFirstFrameWasQueuedAndHeard(t *testing.T) {
	queued, audible, speechEnd, roundtrip := 880.0, 940.0, 1060.0, 1400.0
	turns := []live.Turn{
		{TurnID: "turn-1", RoundtripMs: &roundtrip, FirstFrameQueuedMs: &queued,
			FirstAudibleFrameMs: &audible, SpeechEndToAudibleMs: &speechEnd},
		{TurnID: "turn-2", RoundtripMs: &roundtrip},
	}

	timeline := timelineOf(turns, nil, nil, nil)

	require.Len(t, timeline, 2)
	require.Equal(t, &queued, timeline[0].FirstFrameQueuedMs)
	require.Equal(t, &audible, timeline[0].FirstAudibleFrameMs)
	require.Equal(t, &speechEnd, timeline[0].SpeechEndToAudibleMs)
	require.Equal(t, &roundtrip, timeline[0].RoundtripMs, "the figure taken at the return is kept")
	require.Nil(t, timeline[1].FirstFrameQueuedMs, "an edge that does not report them leaves them out")
	require.Nil(t, timeline[1].FirstAudibleFrameMs)
}

func TestATimelineEntryDoesNotTakeAReplyTalkedOverAsSaid(t *testing.T) {
	start := time.Date(2026, 1, 2, 10, 0, 0, 0, time.UTC)
	turns := []live.Turn{{TurnID: "turn-1", StartedAt: start}}
	said := []chatlog.Spoken{
		{Speaker: "alice", Text: "Is it raining?", At: start.Add(-time.Second)},
		{Speaker: "agent", Agent: true, Text: "It is, and it will be all week.", At: start.Add(time.Second), Interrupted: true},
	}

	timeline := timelineOf(turns, said, nil, nil)

	require.Equal(t, "Is it raining?", *timeline[0].Heard)
	require.Nil(t, timeline[0].Said, "what it was going to say is not what it said")
}

func TestTheWordsOfALineNameItFromTheFirstOfThemToItsRuling(t *testing.T) {
	// The stored transcript keeps a participant's line under this id, so a watcher can tell
	// the words it is shown live and the line read back later apart from a second line.
	for _, event := range []agent.Event{
		agent.Hearing{Text: "where is", LineID: "line-1"},
		agent.Heard{Text: "Where is my order?", LineID: "line-1"},
		agent.Decided{Kind: string(agent.ActIgnore), LineID: "line-1"},
	} {
		sent, ok := frameOf(event)
		require.True(t, ok)
		require.Equal(t, "line-1", sent["line_id"])
	}
}

func TestAFrameAboutNoLineLeavesTheLineOut(t *testing.T) {
	sent, ok := frameOf(agent.Heard{Text: "typed into the call"})

	require.True(t, ok)
	require.NotContains(t, sent, "line_id")
}

func TestAScriptedLineIsShownAsSaid(t *testing.T) {
	sent, ok := frameOf(agent.Said{TurnID: "say-1", Text: "Hi, thanks for getting in touch."})

	require.True(t, ok)
	require.Equal(t, "said", sent["type"], "a client waiting on the model's answer would take it for one")
	require.Equal(t, "say-1", sent["turn_id"])
	require.Equal(t, "Hi, thanks for getting in touch.", sent["text"])
}

func TestALineSaidUnaskedSaysWhatItIsFor(t *testing.T) {
	greeting, ok := frameOf(agent.Said{TurnID: "say-1", Text: "Hi.", Purpose: agent.PurposeGreeting})
	require.True(t, ok)
	require.Equal(t, agent.PurposeGreeting, greeting["purpose"])

	checkIn, ok := frameOf(agent.Responded{TurnID: "back-1", Text: "Still there?", Purpose: agent.PurposeCheckIn})
	require.True(t, ok)
	require.Equal(t, agent.PurposeCheckIn, checkIn["purpose"])

	answer, ok := frameOf(agent.Responded{TurnID: "turn-1", Text: "Yes."})
	require.True(t, ok)
	require.NotContains(t, answer, "purpose")
}

// flowCall is a flow decision's model call as the router reports it.
func flowCall(success bool, errorCode string) llm.CallTiming {
	return llm.CallTiming{OperationID: "op-1", Purpose: "flow", TurnID: "turn-1", Provider: "stub", Model: "fast",
		TTFTMs: 80, DurationMs: 110, Success: success, ErrorCode: errorCode, InputTokens: 10, OutputTokens: 2}
}

func TestAServedModelCallIsTheFrameItAlwaysWas(t *testing.T) {
	// The bytes base 7b510a431 sends for this call.
	sent, ok := frameOf(agent.ModelCall{CallTiming: flowCall(true, "")})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.Equal(t, `{"duration_ms":110,"model":"fast","operation_id":"op-1","provider":"stub","purpose":"flow","success":true,"ttft_ms":80,"turn_id":"turn-1","type":"model_call"}`,
		string(raw))
}

func TestACancelledModelCallSaysItWasCancelled(t *testing.T) {
	sent, ok := frameOf(agent.ModelCall{CallTiming: flowCall(false, "cancelled")})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.JSONEq(t, `false`, string(mustField(t, raw, "success")))
	require.JSONEq(t, `"cancelled"`, string(mustField(t, raw, "error_code")))
	require.NotContains(t, sent, "error_message", "a cancel that said nothing adds no message")
}

func TestAFailedModelCallSaysWhatTheProviderAnswered(t *testing.T) {
	failed := flowCall(false, "create_failed")
	failed.ErrorMessage = `POST "https://gemma.example/v1/chat/completions": 502 Bad Gateway : failed to connect to local service`
	sent, ok := frameOf(agent.ModelCall{CallTiming: failed})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.JSONEq(t, `"create_failed"`, string(mustField(t, raw, "error_code")))
	require.JSONEq(t, `"POST \"https://gemma.example/v1/chat/completions\": 502 Bad Gateway : failed to connect to local service"`,
		string(mustField(t, raw, "error_message")))
}

func TestAServedModelCallCarriesNoMessageWhateverItsTimingHeld(t *testing.T) {
	served := flowCall(true, "")
	served.ErrorMessage = "a failure the provider recovered from"
	sent, ok := frameOf(agent.ModelCall{CallTiming: served})

	require.True(t, ok)
	require.NotContains(t, sent, "error_message")
	require.NotContains(t, sent, "error_code")
}

// timelineCall is the one model call a timeline lists for a turn that made only the request given.
func timelineCall(t *testing.T, request store.Request) []byte {
	t.Helper()
	latency := 80.0
	request.TurnID, request.OperationID, request.Purpose = "turn-1", "op-1", "flow"
	request.StartedAt = time.Date(2026, 10, 20, 12, 0, 0, 0, time.UTC)
	request.Provider, request.Model, request.LatencyMs = "stub", "fast", &latency
	request.InputTokens, request.OutputTokens = 10, 2
	timeline := timelineOf([]live.Turn{{TurnID: "turn-1"}}, nil, []store.Request{request}, nil)
	require.Len(t, timeline, 1)
	require.NotNil(t, timeline[0].ModelCalls)
	raw, err := json.Marshal(*timeline[0].ModelCalls)
	require.NoError(t, err)
	return raw
}

func TestAServedModelCallIsTheTimelineEntryItAlwaysWas(t *testing.T) {
	// The bytes base 7b510a431 lists for this request.
	require.Equal(t, `[{"input_tokens":10,"model":"fast","operation_id":"op-1","output_tokens":2,"provider":"stub","purpose":"flow","started_at":"2026-10-20T12:00:00Z","success":true,"ttft_ms":80}]`,
		string(timelineCall(t, store.Request{Success: true})))
}

func TestACancelledModelCallOnTheTimelineSaysWhyAndWhatItCost(t *testing.T) {
	raw := timelineCall(t, store.Request{Success: false, ErrorCode: store.ErrorCancelled})

	require.JSONEq(t, `[{"input_tokens":10,"model":"fast","operation_id":"op-1","output_tokens":2,"provider":"stub","purpose":"flow","started_at":"2026-10-20T12:00:00Z","success":false,"error_code":"cancelled","ttft_ms":80}]`,
		string(raw))
}

func TestAFailedModelCallOnTheTimelineSaysWhatTheProviderAnswered(t *testing.T) {
	raw := timelineCall(t, store.Request{Success: false, ErrorCode: "create_failed",
		ErrorMessage: "502 Bad Gateway : failed to connect to local service"})

	require.JSONEq(t, `[{"input_tokens":10,"model":"fast","operation_id":"op-1","output_tokens":2,"provider":"stub","purpose":"flow","started_at":"2026-10-20T12:00:00Z","success":false,"error_code":"create_failed","error_message":"502 Bad Gateway : failed to connect to local service","ttft_ms":80}]`,
		string(raw))
}

func TestASpeakingFrameSaysWhatStartedBeingHeardAndWhen(t *testing.T) {
	at := time.Date(2026, 10, 22, 9, 30, 0, 123456789, time.UTC)

	sent, ok := frameOf(agent.Speaking{TurnID: "turn-1", SynthesisID: "turn-1#0", Text: "Hello there.", At: at})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.Equal(t, `{"at":"2026-10-22T09:30:00.123456789Z","delay_ms":0,"synthesis_id":"turn-1#0","text":"Hello there.","turn_id":"turn-1","type":"speaking"}`,
		string(raw))
}

func TestASpeakingFrameReportedBeforeTheTrackGotToItSaysHowLongUntilItIsHeard(t *testing.T) {
	at := time.Date(2026, 10, 22, 9, 30, 0, 0, time.UTC)

	sent, ok := frameOf(agent.Speaking{TurnID: "turn-1", SynthesisID: "turn-1#1", Text: "Four.", At: at, DelayMs: 640})

	require.True(t, ok)
	require.Equal(t, 640.0, sent["delay_ms"])
}

func TestAnInterruptedFrameSaysHowMuchOfTheUtteranceCutWasHeard(t *testing.T) {
	sent, ok := frameOf(agent.Interrupted{TurnID: "turn-1", SynthesisID: "turn-1#2", HeardMs: 420})

	require.True(t, ok)
	require.Equal(t, "turn-1#2", sent["synthesis_id"])
	require.Equal(t, 420.0, sent["heard_ms"])

	unheard, ok := frameOf(agent.Interrupted{TurnID: "turn-2"})

	require.True(t, ok)
	require.NotContains(t, unheard, "heard_ms", "nothing heard is not zero heard, so a client keeps its own count")
	require.NotContains(t, unheard, "synthesis_id")
}

func TestASpokeFrameCarriesTheCharactersSynthesisedAndTheUtterance(t *testing.T) {
	sent, ok := frameOf(agent.Spoke{TurnID: "turn-1", SynthesisID: "turn-1#0", Characters: 12, AudioDurationMs: 900, TimeToFirstByteMs: 140})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.Equal(t, `{"audio_duration_ms":900,"characters":12,"synthesis_id":"turn-1#0","time_to_first_byte_ms":140,"turn_id":"turn-1","type":"spoke"}`,
		string(raw))
}

func TestASpeakingFrameSaysWhereItsUtteranceIsInTheReplysMessage(t *testing.T) {
	at := time.Date(2026, 10, 22, 9, 30, 0, 0, time.UTC)

	sent, ok := frameOf(agent.Speaking{TurnID: "turn-1", SynthesisID: "turn-1#1",
		Text: "Your card ends in last four digits, 1234.", Message: &agent.MessageSpan{Start: 6, End: 39}, At: at})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.Equal(t, `{"at":"2026-10-22T09:30:00Z","delay_ms":0,"message_end":39,"message_start":6,"synthesis_id":"turn-1#1","text":"Your card ends in last four digits, 1234.","turn_id":"turn-1","type":"speaking"}`,
		string(raw))
}

func TestASpokeFrameSaysWhereAllOfItsUtteranceIsInTheReplysMessage(t *testing.T) {
	sent, ok := frameOf(agent.Spoke{TurnID: "turn-1", SynthesisID: "turn-1", Characters: 48, AudioDurationMs: 2900,
		Message: &agent.MessageSpan{End: 41}})

	require.True(t, ok)
	require.Equal(t, 0, sent["message_start"])
	require.Equal(t, 41, sent["message_end"])
}

func TestAnInterruptedFrameSaysWhereTheUtteranceCutIsInTheReplysMessage(t *testing.T) {
	sent, ok := frameOf(agent.Interrupted{TurnID: "turn-1", SynthesisID: "turn-1#2", HeardMs: 420,
		Message: &agent.MessageSpan{Start: 39, End: 70}})

	require.True(t, ok)
	require.Equal(t, 39, sent["message_start"])
	require.Equal(t, 70, sent["message_end"])
}

func TestFramesOfAnUtteranceWithNoKnownPlaceInTheMessageLeaveItOut(t *testing.T) {
	// A client reading a router before message offsets counts the voice's characters, and
	// one with nothing to go by here does the same.
	for _, event := range []agent.Event{
		agent.Speaking{TurnID: "turn-1", SynthesisID: "turn-1", Text: "Hi."},
		agent.Spoke{TurnID: "turn-1", SynthesisID: "turn-1", Characters: 3},
		agent.Interrupted{TurnID: "turn-1"},
	} {
		sent, ok := frameOf(event)

		require.True(t, ok)
		require.NotContains(t, sent, "message_start")
		require.NotContains(t, sent, "message_end")
	}
}

func TestACallTheCallerAlreadyAllowedSaysSoOnBothSockets(t *testing.T) {
	asked, ok := frameOf(session.ToolCall{ID: "call-1", Name: "book_flight", Arguments: `{}`, Run: true, Approved: true})
	require.True(t, ok)
	require.Equal(t, true, asked["approved"])

	plain, ok := frameOf(session.ToolCall{ID: "call-2", Name: "book_flight", Arguments: `{}`, Run: true})
	require.True(t, ok)
	require.NotContains(t, plain, "approved", "a call nobody was asked about carries no approval")

	require.Equal(t, true, toolCallFrame(dispatch.ToolCall{ID: "call-1", Name: "book_flight", Approved: true})["approved"])
	require.NotContains(t, toolCallFrame(dispatch.ToolCall{ID: "call-2", Name: "book_flight"}), "approved")
}

func TestASequencedEventIsItsOwnFrameWithACursor(t *testing.T) {
	sent, ok := frameOf(session.Sequenced{Cursor: "a1b2.7", Event: agent.ResponseDelta{TurnID: "turn-1", Text: "Hel"}})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"response_delta","turn_id":"turn-1","text":"Hel","cursor":"a1b2.7"}`, string(raw))
}

func TestAResumedFrameCarriesTheReceiptOfTheCommandAskedAfter(t *testing.T) {
	receipt := conversation.CommandReceipt{CommandID: "command-1", UserMessageID: "command-1", AssistantMessageID: "reply-1", State: "interrupted"}

	sent, ok := frameOf(session.Resumed{Cursor: "a1b2.0", Complete: false, Command: &receipt})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"resumed","cursor":"a1b2.0","complete":false,"command":{"request_id":"command-1",
		"user_message_id":"command-1","assistant_message_id":"reply-1","state":"interrupted","duplicate":false}}`, string(raw))
}

func TestOnlyAResumableSocketIsToldTheRouterIsRestarting(t *testing.T) {
	halted := func() bool { return true }

	code, reason := endingOf(resuming{resumable: true}, halted)()
	plain, _ := endingOf(resuming{}, halted)()

	require.Equal(t, websocket.CloseServiceRestart, code)
	require.Equal(t, "router restarting", reason)
	require.Equal(t, websocket.CloseNormalClosure, plain, "a socket that never asked to resume is closed the way it always was")
}

func TestAResumableSocketWhoseSessionEndedIsToldItEnded(t *testing.T) {
	code, _ := endingOf(resuming{resumable: true}, func() bool { return false })()

	require.Equal(t, websocket.CloseNormalClosure, code)
}

// injection is the failure a response the prompt injection screen refused settles with.
var injection = fmt.Errorf("%w: instruction_override", policy.ErrPromptInjection)

// marshalled is a frame as the socket sends it.
func marshalled(t *testing.T, sent frame, ok bool) []byte {
	t.Helper()
	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	return raw
}

func TestAnErrorFrameForATurnTheScreenRefusedNamesTheRefusal(t *testing.T) {
	for name, err := range map[string]error{
		"as the screen gave it": injection,
		"wrapped on its way":    fmt.Errorf("agent: %w", injection),
	} {
		sent, ok := frameOf(agent.Error{Err: err, Context: "llm"})
		raw := marshalled(t, sent, ok)
		require.JSONEq(t, `"prompt_injection"`, string(mustField(t, raw, "code")), name)
		require.JSONEq(t, `"llm"`, string(mustField(t, raw, "context")), name)
	}
}

func TestAnErrorFrameForAnyOtherFailureIsTheFrameItAlwaysWas(t *testing.T) {
	// The bytes base cec7b425a sends for these.
	for err, want := range map[error]string{
		errors.New("503 Service Unavailable"):                       `{"context":"llm","error":"503 Service Unavailable","type":"error"}`,
		errors.New("policy: the input reads as a prompt injection"): `{"context":"llm","error":"policy: the input reads as a prompt injection","type":"error"}`,
		nil: `{"context":"llm","error":"","type":"error"}`,
	} {
		sent, ok := frameOf(agent.Error{Err: err, Context: "llm"})
		require.Equal(t, want, string(marshalled(t, sent, ok)))
	}
}

func TestAFailureFrameNeverCarriesACredentialAProviderEchoed(t *testing.T) {
	token := "sk-test-" + rand.Text()
	echoed := errors.New("502 Bad Gateway: no route\nAuthorization: Bearer " + token)
	for name, event := range map[string]session.Event{
		"error":        agent.Error{Err: echoed, Context: "llm"},
		"task settled": agent.TaskSettled{Err: echoed},
		"tool ran":     agent.ToolRan{Err: echoed},
	} {
		sent, ok := frameOf(event)
		raw := marshalled(t, sent, ok)
		require.NotContains(t, string(raw), token, name)
		require.Contains(t, string(raw), "502 Bad Gateway: no route", name)
	}
}

func TestAnLLMSocketErrorForARefusedResponseNamesTheRefusal(t *testing.T) {
	sent, ok := llmFrame(llm.ResponseFailed{ResponseID: "r1", Err: injection, Context: "stream"})
	raw := marshalled(t, sent, ok)

	require.JSONEq(t, `"prompt_injection"`, string(mustField(t, raw, "code")))
	require.JSONEq(t, `"r1"`, string(mustField(t, raw, "id")))
}

func TestAnLLMSocketErrorForAProviderFailureIsTheFrameItAlwaysWas(t *testing.T) {
	// The bytes base cec7b425a sends for it.
	sent, ok := llmFrame(llm.ResponseFailed{ResponseID: "r1", Err: errors.New("unauthorized"), Context: "stream"})

	require.Equal(t, `{"context":"stream","error":"unauthorized","fatal":false,"id":"r1","type":"error"}`,
		string(marshalled(t, sent, ok)))
}

func TestAUIDeltaFrameCarriesTheTurnFormatAndChunk(t *testing.T) {
	sent, ok := frameOf(agent.UIDelta{TurnID: "turn-1", Format: "openui", Mode: "replace", Delta: "root = Card([hdr])"})

	require.Equal(t, `{"delta":"root = Card([hdr])","format":"openui","turn_id":"turn-1","type":"ui_delta"}`,
		string(marshalled(t, sent, ok)))
}

func TestAUIPatchAndVersionAreIncludedWhenPresent(t *testing.T) {
	delta, ok := frameOf(agent.UIDelta{TurnID: "turn-2", Format: "openui", Version: "1.1", Mode: "patch", Delta: "$party = 4"})
	require.Equal(t, `{"delta":"$party = 4","format":"openui","mode":"patch","turn_id":"turn-2","type":"ui_delta","version":"1.1"}`,
		string(marshalled(t, delta, ok)))

	completed, ok := frameOf(agent.UICompleted{TurnID: "turn-2", Format: "openui", Version: "1.1", Mode: "patch", Content: "$party = 4"})
	require.Equal(t, `{"content":"$party = 4","format":"openui","mode":"patch","turn_id":"turn-2","type":"ui_completed","version":"1.1"}`,
		string(marshalled(t, completed, ok)))
}

// pauseFloorCheck and overAgentFloorCheck are chorus-asr checks as a call keeps them: one at a
// pause that ended the turn, with the tool the caller wants, and one over the agent, which has
// no pause and no point.
func pauseFloorCheck(at time.Time) live.FloorCheck {
	paused := 600.0
	return live.FloorCheck{
		At: at, Kind: live.FloorCheckPause, PausedMs: &paused,
		Options: []live.FloorOption{{Option: "respond", Probability: 0.91}, {Option: "wait", Probability: 0.07},
			{Option: "clarify", Probability: 0.000001}, {Option: "ignore", Probability: 0.02}},
		Ruling: live.FloorRuling{Disposition: "respond", Floor: "continue", Tool: "get_weather"},
		Acted:  true, LatencyMs: 38.2, Point: "preset:balanced",
	}
}

func overAgentFloorCheck(at time.Time) live.FloorCheck {
	return live.FloorCheck{
		At: at, Kind: live.FloorCheckOverAgent,
		Options: []live.FloorOption{{Option: "stop", Probability: 0.7}, {Option: "shorten", Probability: 0.2},
			{Option: "continue", Probability: 0.1}, {Option: "ignore", Probability: 0}},
		Ruling: live.FloorRuling{Disposition: "respond", Floor: "stop", Cannot: true},
		Acted:  true, LatencyMs: 44,
	}
}

// floorChecked is a check as the call's record keeps it, about turn.
func floorChecked(turn string, check live.FloorCheck) live.CallEvent {
	return live.CallEvent{At: check.At, Kind: "floor_check", TurnID: turn, FloorCheck: &check}
}

// A call's checks join the turn each names, a reply's late checks included, and a check that
// names none, or a turn with no entry, joins the first turn that began after it.
func TestATimelineEntryCarriesTheFloorChecksOfItsTurn(t *testing.T) {
	start := time.Date(2026, 1, 2, 10, 0, 0, 0, time.UTC)
	turns := []live.Turn{
		{TurnID: "turn-1", StartedAt: start.Add(time.Second)},
		{TurnID: "turn-2", StartedAt: start.Add(5 * time.Second)},
		{TurnID: "turn-3", StartedAt: start.Add(8 * time.Second)},
	}
	checks := []live.CallEvent{
		floorChecked("", pauseFloorCheck(start.Add(500*time.Millisecond))),
		floorChecked("superseded", pauseFloorCheck(start.Add(4*time.Second))),
		floorChecked("turn-1", overAgentFloorCheck(start.Add(6*time.Second))),
		floorChecked("", pauseFloorCheck(start.Add(9*time.Second))),
	}

	timeline := timelineOf(turns, nil, nil, checks)

	require.NotNil(t, timeline[0].FloorChecks)
	raw, err := json.Marshal(*timeline[0].FloorChecks)
	require.NoError(t, err)
	require.JSONEq(t, `[
		{"at":"2026-01-02T10:00:00.5Z","kind":"pause","paused_ms":600,
		 "options":[{"option":"respond","probability":0.91},{"option":"wait","probability":0.07},
		            {"option":"clarify","probability":0.000001},{"option":"ignore","probability":0.02}],
		 "ruling":{"disposition":"respond","floor":"continue","tool":"get_weather"},
		 "acted":true,"latency_ms":38.2,"point":"preset:balanced"},
		{"at":"2026-01-02T10:00:06Z","kind":"over_agent",
		 "options":[{"option":"stop","probability":0.7},{"option":"shorten","probability":0.2},
		            {"option":"continue","probability":0.1},{"option":"ignore","probability":0}],
		 "ruling":{"disposition":"respond","floor":"stop","cannot":true},
		 "acted":true,"latency_ms":44}
	]`, string(raw), "the pause that led to it and the check over its reply after turn-2 began, as the router got them")
	require.Len(t, *timeline[1].FloorChecks, 1, "words let go of lead to the next turn")
	require.Nil(t, timeline[2].FloorChecks, "a check after the last turn began leads to none yet")
	raw, err = json.Marshal(timeline[2])
	require.NoError(t, err)
	require.NotContains(t, string(raw), "floor_checks")
}

// A turn carries its latest 40 checks, oldest first.
func TestATimelineEntryCarriesItsLatestFortyFloorChecks(t *testing.T) {
	start := time.Date(2026, 1, 2, 10, 0, 0, 0, time.UTC)
	var checks []live.CallEvent
	for i := range 45 {
		checks = append(checks, floorChecked("turn-1", overAgentFloorCheck(start.Add(time.Duration(44-i)*time.Second))))
	}

	timeline := timelineOf([]live.Turn{{TurnID: "turn-1", StartedAt: start}}, nil, nil, checks)

	kept := *timeline[0].FloorChecks
	require.Len(t, kept, floorChecksPerTurn)
	require.Equal(t, start.Add(5*time.Second), kept[0].At)
	require.Equal(t, start.Add(44*time.Second), kept[floorChecksPerTurn-1].At)
}

// The live account of a turn's checks is its turn frame, as a timeline entry has them; an
// agent without chorus-asr floor control sends the frame it always did.
func TestATurnFrameCarriesItsFloorChecksAsTheTimelineDoes(t *testing.T) {
	at := time.Date(2026, 1, 2, 10, 0, 0, 0, time.UTC)
	sent, ok := frameOf(agent.Turn{TurnID: "turn-1", FloorChecks: []live.FloorCheck{overAgentFloorCheck(at)}})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	entry, err := json.Marshal([]FloorCheck{floorCheckOf(overAgentFloorCheck(at))})
	require.NoError(t, err)
	require.JSONEq(t, string(entry), string(mustField(t, raw, "floor_checks")))

	plain, ok := frameOf(agent.Turn{TurnID: "turn-2"})
	require.True(t, ok)
	raw, err = json.Marshal(plain)
	require.NoError(t, err)
	require.NotContains(t, string(raw), "floor_checks")
}
