//go:build integration

package api

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
)

// A tool a caller declares with an approval reaches the session's spec, which is what makes
// its calls wait for a person; one with no question asks nothing.
func TestADeclaredToolsApprovalReachesTheSpec(t *testing.T) {
	client := SessionToolExecutorClient
	message, reason := "Only your city is shared.", "purpose"
	tools := []SessionTool{
		{Name: "athena_device_location", Description: "Where they are", Executor: &client,
			Approval: &SessionToolApproval{Title: "Share your location?", Message: &message, ReasonArgument: &reason}},
		{Name: "athena_read_canvas", Description: "Read a canvas", Approval: &SessionToolApproval{}},
	}
	text := true
	spec := specOf(CreateSessionRequest{Text: &text, Tools: &tools}, "acme", nil)
	require.Len(t, spec.Tools, 2)
	require.True(t, spec.Tools[0].Client)
	require.Equal(t, &harness.ToolApproval{Title: "Share your location?", Message: "Only your city is shared.", ReasonArgument: "purpose"}, spec.Tools[0].Approval)
	require.Nil(t, spec.Tools[1].Approval)
}
