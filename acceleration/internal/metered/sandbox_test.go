package metered

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

func TestASandboxIsOpenedOnlyForAnAppAndASession(t *testing.T) {
	sandboxes := NewSandboxes(SandboxesOptions{APIKey: "key-1", APIURL: "http://127.0.0.1:1"})
	for name, owner := range map[string]routing.Owner{
		"no app":     {Attribution: routing.Attribution{SessionID: "session-1"}},
		"no session": {CustomerID: "app-1"},
	} {
		t.Run(name, func(t *testing.T) {
			box, err := sandboxes.Open(t.Context(), owner, SandboxSpec{})
			require.Error(t, err)
			require.Nil(t, box)
		})
	}

	box, err := sandboxes.Open(t.Context(), routing.Owner{CustomerID: "app-1", Attribution: routing.Attribution{SessionID: "session-1"}}, SandboxSpec{})
	require.NoError(t, err, "nothing is asked of Daytona until code runs")
	require.NoError(t, box.Close())
}
