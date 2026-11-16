package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"syscall"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/net/http2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

func TestTransientReadsWhatTheProviderNamed(t *testing.T) {
	passing := &ProviderFailure{Err: errors.New("busy"), Transient: true}
	refused := &ProviderFailure{Err: errors.New("bad request")}

	require.True(t, Transient(stack.Wrap(fmt.Errorf("reading: %w", passing))), "through every wrapping")
	require.False(t, Transient(stack.Wrap(refused)))
}

func TestTransientCountsATimeoutOrADroppedConnection(t *testing.T) {
	require.True(t, Transient(stack.Wrap(context.DeadlineExceeded)))
	require.True(t, Transient(fmt.Errorf("read: %w", io.ErrUnexpectedEOF)))
	require.True(t, Transient(&os.SyscallError{Syscall: "read", Err: syscall.ECONNRESET}))
	require.True(t, Transient(fmt.Errorf("read: %w", http2.StreamError{StreamID: 1, Code: http2.ErrCodeInternal})))
	require.True(t, Transient(fmt.Errorf("read: %w", http2.GoAwayError{LastStreamID: 1, ErrCode: http2.ErrCodeNo})))
}

// A server that resets a stream part way through a body reaches the reader as net/http's own
// copy of the HTTP/2 stream error, not x/net's.
func TestTransientCountsAStreamTheServerResetOverHTTP2(t *testing.T) {
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "part of an answer")
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler)
	}))
	server.EnableHTTP2 = true
	server.StartTLS()
	defer server.Close()

	response, err := server.Client().Get(server.URL)
	require.NoError(t, err)
	defer response.Body.Close()
	require.Equal(t, 2, response.ProtoMajor)
	_, err = io.ReadAll(response.Body)

	require.Error(t, err)
	require.True(t, Transient(err), err.Error())
}

func TestTransientLeavesOutWhatNothingClassifies(t *testing.T) {
	require.False(t, Transient(nil))
	require.False(t, Transient(context.Canceled), "the caller gave up, so there is nobody to ask for")
	require.False(t, Transient(errors.New("something went wrong")))
}

func TestTransientStatusIsTheServersSideAndRateLimits(t *testing.T) {
	for _, status := range []int{http.StatusRequestTimeout, http.StatusTooEarly, http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusServiceUnavailable, 529, 599} {
		require.True(t, TransientStatus(status), status)
	}
	for _, status := range []int{http.StatusOK, http.StatusBadRequest, http.StatusUnauthorized, http.StatusPaymentRequired, http.StatusForbidden, http.StatusNotFound, http.StatusUnprocessableEntity} {
		require.False(t, TransientStatus(status), status)
	}
	for _, status := range []int{600, 1301, 40001} {
		require.False(t, TransientStatus(status), status)
	}
}
