package api

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
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
