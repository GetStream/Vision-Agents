package server

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTheCommittedSpecIsRenderedFromTheOperations(t *testing.T) {
	// The spec is what staff tools are typed from, so one edited by hand or left behind by a
	// change to an operation is a tool out of step with the service.
	committed, err := os.ReadFile("openapi.yaml")
	require.NoError(t, err)
	rendered, err := Spec()
	require.NoError(t, err)

	require.Equal(t, string(rendered), string(committed), "run go run ./cmd/usage openapi in acceleration/")
}
