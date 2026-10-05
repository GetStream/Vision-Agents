package api

import (
	"encoding/json"
	"testing"

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

func mustField(t *testing.T, raw []byte, name string) json.RawMessage {
	t.Helper()
	var fields map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(raw, &fields))
	require.Contains(t, fields, name)
	return fields[name]
}
