package api

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
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

func TestATurnFrameCarriesHowLongTheReplyWasHeld(t *testing.T) {
	sent, ok := frameOf(agent.Turn{TurnID: "turn-1", TTSToAudioMs: 900, ReplyHoldMs: 620})

	require.True(t, ok)
	raw, err := json.Marshal(sent)
	require.NoError(t, err)
	require.JSONEq(t, `620`, string(mustField(t, raw, "reply_hold_ms")))
	require.JSONEq(t, `900`, string(mustField(t, raw, "tts_to_audio_ms")), "the leg that includes it is kept")
}

func TestATimelineEntryCarriesHowLongTheReplyWasHeld(t *testing.T) {
	held := 620.0
	turns := []store.Turn{{TurnID: "turn-1", ReplyHoldMs: &held}, {TurnID: "turn-2"}}

	timeline := timelineOf(turns, nil, nil)

	require.Len(t, timeline, 2)
	require.Equal(t, &held, timeline[0].ReplyHoldMs)
	require.Nil(t, timeline[1].ReplyHoldMs, "a reply that was not held leaves it out")
}

func TestATimelineEntryCarriesWhenTheFirstFrameWasQueuedAndHeard(t *testing.T) {
	queued, audible, speechEnd, roundtrip := 880.0, 940.0, 1060.0, 1400.0
	turns := []store.Turn{
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
