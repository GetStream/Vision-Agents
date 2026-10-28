package llm

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"syscall"

	"golang.org/x/net/http2"
)

// ProviderFailure is a failure a provider named in its own terms, with what those terms say
// about asking again. An adapter returns one for the failures its provider's protocol names,
// so that Transient can read them without knowing any provider.
type ProviderFailure struct {
	Err error
	// Transient says the provider named a condition of its own that passes: it was overloaded,
	// rate limited the request, or failed on its side.
	Transient bool
}

func (f *ProviderFailure) Error() string { return f.Err.Error() }

func (f *ProviderFailure) Unwrap() error { return f.Err }

// Transient reports whether the same request, asked again, may succeed where it failed: the
// provider said it was overloaded, rate limited or failed on its own side, or the connection
// to it timed out, dropped or reset the stream. A failure the request itself caused, such as
// credentials or credit refused, a malformed request or content the provider will not
// process, fails the same way every time. So does anything nothing classifies, since asking
// again is only worth its cost when there is a reason to expect a different answer.
func Transient(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) {
		return false
	}
	if named, ok := errors.AsType[*ProviderFailure](err); ok {
		return named.Transient
	}
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, io.ErrUnexpectedEOF) ||
		errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.EPIPE) {
		return true
	}
	if timeout, ok := errors.AsType[net.Error](err); ok && timeout.Timeout() {
		return true
	}
	// Over HTTP/2 the connection dropping under a stream, or the server resetting it, arrives
	// as these rather than as a reset socket. net/http's own copy of StreamError converts to
	// this one through its As method; its copy of GoAwayError has no such method, so only
	// x/net's is recognised here.
	if _, ok := errors.AsType[http2.StreamError](err); ok {
		return true
	}
	if _, ok := errors.AsType[http2.GoAwayError](err); ok {
		return true
	}
	return false
}

// TransientStatus reports whether an HTTP status says the request may succeed if asked again:
// a timeout, too many requests, or the server failing. Any other 4xx is the request's own,
// and a number outside the status range is not an HTTP status at all.
func TransientStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusTooEarly, http.StatusTooManyRequests:
		return true
	}
	return status >= http.StatusInternalServerError && status <= 599
}
