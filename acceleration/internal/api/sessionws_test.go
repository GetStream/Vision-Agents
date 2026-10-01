package api

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
)

func TestConnectorUnavailableHasAnExplicitSessionFrame(t *testing.T) {
	got, ok := frameOf(session.ConnectorUnavailable{
		Name: "linear", ConnectorID: "linear", Reason: "needs_reauthorization",
	})
	require.True(t, ok)
	require.Equal(t, frame{
		"type": "connector_unavailable", "name": "linear",
		"connector_id": "linear", "reason": "needs_reauthorization",
	}, got)
}
