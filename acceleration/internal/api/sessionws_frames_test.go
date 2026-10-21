package api

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
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

	timeline := timelineOf(turns, nil, nil)

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

	timeline := timelineOf(turns, said, nil)

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
}

// timelineCall is the one model call a timeline lists for a turn that made only the request given.
func timelineCall(t *testing.T, request store.Request) []byte {
	t.Helper()
	latency := 80.0
	request.TurnID, request.OperationID, request.Purpose = "turn-1", "op-1", "flow"
	request.StartedAt = time.Date(2026, 10, 20, 12, 0, 0, 0, time.UTC)
	request.Provider, request.Model, request.LatencyMs = "stub", "fast", &latency
	request.InputTokens, request.OutputTokens = 10, 2
	timeline := timelineOf([]live.Turn{{TurnID: "turn-1"}}, nil, []store.Request{request})
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
