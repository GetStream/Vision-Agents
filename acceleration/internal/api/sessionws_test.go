package api

import (
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/stretchr/testify/require"
)

func TestDurableToolFrameCarriesCommandAndTurn(t *testing.T) {
	encoded, ok := frameOf(session.ToolCall{
		ID: "call-a", CommandID: "command-a", TurnID: "turn-a",
		Name: "lookup_record", Arguments: `{"kind":"conversation"}`,
	})
	require.True(t, ok)
	require.Equal(t, "command-a", encoded["command_id"])
	require.Equal(t, "turn-a", encoded["turn_id"])
	require.Equal(t, "call-a", encoded["id"])
}
