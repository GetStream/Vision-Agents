package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

func TestADeleteThatFailsOutsideTheMemoryStoreIsStillAnInternalError(t *testing.T) {
	failure := stack.Wrap(fmt.Errorf("store: delete session: %w", errors.New("dial tcp 10.0.0.7:5432: connection refused")))
	handler, logged := served(t, func() error { return memoryFailure(context.Background(), failure, errSessionKept) })

	response := postSync(handler, `{"name":"jean"}`)

	require.Equal(t, http.StatusInternalServerError, response.Code)
	require.Equal(t, "internal_error", answered(t, response.Body.Bytes()).Code)
	require.Contains(t, logged.String(), "connection refused")
}
