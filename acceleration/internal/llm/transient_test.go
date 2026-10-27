package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"syscall"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

func TestTransientReadsWhatTheProviderNamed(t *testing.T) {
	passing := &ProviderFailure{Err: errors.New("busy"), Code: "busy", Transient: true}
	refused := &ProviderFailure{Err: errors.New("bad request"), Code: "bad_request"}

	require.True(t, Transient(stack.Wrap(fmt.Errorf("reading: %w", passing))), "through every wrapping")
	require.False(t, Transient(stack.Wrap(refused)))
}

func TestTransientCountsATimeoutOrADroppedConnection(t *testing.T) {
	require.True(t, Transient(stack.Wrap(context.DeadlineExceeded)))
	require.True(t, Transient(fmt.Errorf("read: %w", io.ErrUnexpectedEOF)))
	require.True(t, Transient(&os.SyscallError{Syscall: "read", Err: syscall.ECONNRESET}))
}

func TestTransientLeavesOutWhatNothingClassifies(t *testing.T) {
	require.False(t, Transient(nil))
	require.False(t, Transient(context.Canceled), "the caller gave up, so there is nobody to ask for")
	require.False(t, Transient(errors.New("something went wrong")))
}

func TestTransientStatusIsTheServersSideAndRateLimits(t *testing.T) {
	for _, status := range []int{http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusServiceUnavailable, 529} {
		require.True(t, TransientStatus(status), status)
	}
	for _, status := range []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusPaymentRequired, http.StatusForbidden, http.StatusNotFound, http.StatusUnprocessableEntity} {
		require.False(t, TransientStatus(status), status)
	}
}
