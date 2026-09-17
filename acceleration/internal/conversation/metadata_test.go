package conversation

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/stretchr/testify/require"
)

func TestSchemaV1MetadataIsBoundedWhitelistedAndSafe(t *testing.T) {
	message := Message{
		State: "tools", Sequence: 7,
		Tools: []Tool{
			{ID: "call-1", Name: "athena_resource_metadata", Status: "completed", Summary: "provider secret"},
			{ID: "call-2", Name: "unapproved_tool", Status: "completed", Summary: "private reasoning"},
		},
		Sources: []Source{
			{ID: "source-1", Title: "Public reference", URL: "https://docs.example.com/reference", Citation: "API section"},
			{ID: "source-2", Title: "Local service", URL: "https://127.0.0.1/private"},
			{ID: "source-1", Title: "Duplicate", URL: "https://other.example.com/"},
		},
	}
	metadata, err := metadataOf(message)
	require.NoError(t, err)
	require.Equal(t, supportMessageVersion, metadata.SchemaVersion)
	require.Equal(t, 7, metadata.Sequence)
	require.Equal(t, []displayTool{{
		ID: "call-1", Name: "athena_resource_metadata", Status: "completed",
		Summary: "Conversation metadata checked.",
	}}, metadata.Tools)
	require.Equal(t, []displaySource{{
		ID: "source-1", Title: "Public reference", URL: "https://docs.example.com/reference",
		Citation: "API section",
	}}, metadata.Sources)
	encoded, err := json.Marshal(metadata)
	require.NoError(t, err)
	require.NotContains(t, string(encoded), "secret")
	require.NotContains(t, string(encoded), "reasoning")

	var raw map[string]any
	require.NoError(t, json.Unmarshal(encoded, &raw))
	raw["prompt"] = "not observable"
	_, err = decodeMetadata(raw)
	require.Error(t, err)
	raw = map[string]any{
		"schema_version": 1, "sequence": 1, "state": "writing",
		"sources": []any{map[string]any{
			"id": "source", "title": "Unsafe", "url": "https://user:pass@example.com/private",
		}},
	}
	_, err = decodeMetadata(raw)
	require.Error(t, err)
	message.State = "provider_reasoning"
	_, err = metadataOf(message)
	require.Error(t, err)
	message.State = "thinking"
	message.Sources = []Source{{
		ID: "source", Title: strings.Repeat("x", 201), URL: "https://docs.example.com/",
	}}
	metadata, err = metadataOf(message)
	require.NoError(t, err)
	require.Empty(t, metadata.Sources)
}

func TestActivityRevisionsIgnoreDuplicateLateAndCrossCommandEvents(t *testing.T) {
	_, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	conversation, _, _, err := service.OpenForCaller(
		context.Background(), "customer", "athena", "", "employee",
	)
	require.NoError(t, err)
	_, err = conversation.BeginCommand("command-a", "First question")
	require.NoError(t, err)
	conversation.BindTurn("command-a", "turn-a")

	conversation.Observe(agent.ToolStarted{
		ID: "call-a", TurnID: "turn-a", Tool: "athena_resource_metadata", StartedAt: time.Now().UTC(),
	})
	started := current(conversation)
	require.Equal(t, 1, started.Sequence)
	require.Equal(t, "turn-a", started.TurnID)
	conversation.Observe(agent.ToolStarted{
		ID: "call-a", TurnID: "turn-a", Tool: "athena_resource_metadata", StartedAt: time.Now().UTC(),
	})
	require.Equal(t, started.Sequence, current(conversation).Sequence)
	conversation.Observe(agent.ToolRan{
		ID: "call-a", TurnID: "turn-other", Tool: "athena_resource_metadata", Result: `{}`,
	})
	require.Equal(t, started.Sequence, current(conversation).Sequence)
	conversation.Observe(agent.ToolRan{
		ID: "call-a", TurnID: "turn-a", Tool: "athena_resource_metadata", Result: `{}`,
	})
	finishedTool := current(conversation)
	require.Equal(t, started.Sequence+1, finishedTool.Sequence)
	conversation.Observe(agent.ToolRan{
		ID: "call-a", TurnID: "turn-a", Tool: "athena_resource_metadata", Result: `{}`,
	})
	require.Equal(t, finishedTool.Sequence, current(conversation).Sequence)

	stopped, err := conversation.CancelCommand("command-a")
	require.NoError(t, err)
	require.Equal(t, "cancelled", stopped.State)
	cancelled := current(conversation)
	require.Greater(t, cancelled.Sequence, finishedTool.Sequence)
	conversation.Observe(agent.ResponseDelta{TurnID: "turn-a", Text: "late output"})
	require.Equal(t, cancelled, current(conversation))

	_, err = conversation.BeginCommand("command-b", "Second question")
	require.NoError(t, err)
	conversation.BindTurn("command-b", "turn-b")
	next := current(conversation)
	conversation.Observe(agent.ResponseDelta{TurnID: "turn-a", Text: "cross-command output"})
	require.Equal(t, next, current(conversation))
	conversation.Observe(agent.ResponseDelta{TurnID: "turn-b", Text: "Second answer"})
	require.Equal(t, "Second answer", current(conversation).Text)
}

func TestTrustedSearchSourcesRequireStrictSafeResults(t *testing.T) {
	result := `{"status":"answered","citations":[` +
		`{"id":"one","title":"Reference","url":"https://docs.example.com/reference","citation":"Section 2"},` +
		`{"id":"two","title":"Private","url":"https://localhost/internal"},` +
		`{"id":"one","title":"Duplicate","url":"https://other.example.com/"}]}`
	require.Equal(t, []Source{{
		ID: "one", Title: "Reference", URL: "https://docs.example.com/reference", Citation: "Section 2",
	}}, sourcesOf("search_docs", result))
	require.Empty(t, sourcesOf("athena_resource_metadata", result))
	require.Empty(t, sourcesOf("search_docs", `{"status":"answered","prompt":"leak","citations":[]}`))
	require.Empty(t, sourcesOf("search_docs", strings.Repeat("x", maxSupportMessageBytes+1)))
}

func TestStreamSnapshotsCarrySchemaV1AndStableCommandTurnIdentity(t *testing.T) {
	db, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	conversation, _, _, err := service.OpenForCaller(
		context.Background(), "customer", "athena", "", "employee",
	)
	require.NoError(t, err)
	receipt, err := conversation.BeginCommand("command-a", "Inspect this conversation")
	require.NoError(t, err)
	conversation.BindTurn("command-a", "turn-a")
	require.Eventually(t, func() bool {
		db.mu.Lock()
		defer db.mu.Unlock()
		return db.messages[receipt.AssistantMessageID] != nil
	}, 3*time.Second, 20*time.Millisecond)

	db.mu.Lock()
	created := db.messages[receipt.AssistantMessageID]
	initialCustom := created["custom"].(map[string]any)
	db.mu.Unlock()
	initial, err := decodeMetadata(initialCustom["support_message"])
	require.NoError(t, err)
	require.Equal(t, supportMessageVersion, initial.SchemaVersion)
	require.Equal(t, "thinking", initial.State)
	runtime, err := decodeRuntime(initialCustom["support_runtime"])
	require.NoError(t, err)
	require.Equal(t, "command-a", runtime.CommandID)

	conversation.Observe(agent.Responding{TurnID: "turn-a"})
	conversation.Observe(agent.ToolStarted{
		ID: "call-a", TurnID: "turn-a", Tool: "athena_resource_metadata", StartedAt: time.Now().UTC(),
	})
	require.Eventually(t, func() bool {
		db.mu.Lock()
		defer db.mu.Unlock()
		return len(db.patches) > 0
	}, 3*time.Second, 20*time.Millisecond)
	db.mu.Lock()
	patch := db.patches[len(db.patches)-1]
	db.mu.Unlock()
	transient, err := decodeMetadata(patch["support_message"])
	require.NoError(t, err)
	require.Greater(t, transient.Sequence, initial.Sequence)
	require.Equal(t, "tools", transient.State)
	require.Len(t, transient.Tools, 1)
	transientRuntime, err := decodeRuntime(patch["support_runtime"])
	require.NoError(t, err)
	require.Equal(t, "command-a", transientRuntime.CommandID)
	require.Equal(t, "turn-a", transientRuntime.TurnID)

	conversation.Observe(agent.ToolRan{
		ID: "call-a", TurnID: "turn-a", Tool: "athena_resource_metadata", Result: `{}`,
	})
	conversation.Observe(agent.ResponseDelta{TurnID: "turn-a", Text: "Done."})
	conversation.Observe(agent.Responded{TurnID: "turn-a"})
	saved(t, conversation)
	db.mu.Lock()
	terminalCustom := db.messages[receipt.AssistantMessageID]["custom"].(map[string]any)
	db.mu.Unlock()
	terminal, err := decodeMetadata(terminalCustom["support_message"])
	require.NoError(t, err)
	require.Greater(t, terminal.Sequence, transient.Sequence)
	require.Equal(t, "completed", terminal.State)
	require.Equal(t, "completed", terminal.Tools[0].Status)
	require.Equal(t, "Conversation metadata checked.", terminal.Tools[0].Summary)
	encoded, err := json.Marshal(terminalCustom["support_message"])
	require.NoError(t, err)
	require.NotContains(t, string(encoded), "arguments")
	require.NotContains(t, string(encoded), "Result")
}

func TestStoredArtifactReceiptsPublishNativeChatAttachments(t *testing.T) {
	canvas := `{"schema_version":1,"status":"stored","attachment":{"type":"athena_canvas","artifact_id":"canvas_01","revision":1,"title":"Analysis","sha256":"abc"},"publication":"pending"}`
	image := `{"schema_version":1,"status":"stored","attachment":{"type":"athena_image","artifact_id":"img_01","revision":2,"title":"Sketch","alt":"A sketch"},"publication":"pending"}`
	require.Equal(t, []ArtifactAttachment{{
		Type: "athena_canvas", ArtifactID: "canvas_01", Revision: 1, Title: "Analysis",
	}}, artifactsOf(canvas))
	require.Equal(t, []ArtifactAttachment{{
		Type: "athena_image", ArtifactID: "img_01", Revision: 2, Title: "Sketch", Alt: "A sketch",
	}}, artifactsOf(image))
	require.Empty(t, artifactsOf(`{"status":"answered","citations":[]}`))
	require.Empty(t, artifactsOf(`{"schema_version":1,"status":"stored","attachment":{"type":"athena_image","artifact_id":"img_01","revision":1,"title":"Sketch"},"publication":"pending"}`))
	require.Empty(t, artifactsOf(`{"schema_version":1,"status":"stored","attachment":{"type":"athena_pdf","artifact_id":"../x","revision":1,"title":"Report"},"publication":"pending"}`))
	require.Empty(t, artifactsOf(`{"schema_version":1,"status":"stored","attachment":{"type":"athena_canvas","artifact_id":"canvas_01","revision":1,"title":"Analysis"},"publication":"pending","secret":"no"}`))
	wire := streamAttachments(artifactsOf(canvas))
	encoded, err := json.Marshal(wire)
	require.NoError(t, err)
	require.Contains(t, string(encoded), `"type":"athena_canvas"`)
	require.NotContains(t, string(encoded), "sha256")
}
