package stack_test

import (
	"fmt"
	"testing"

	"github.com/getsentry/sentry-go"
	"github.com/stretchr/testify/require"
)

// sentry-go puts the oldest call first, so the caller of Wrap is the last frame.
func TestSentryReadsTheStackOfWrap(t *testing.T) {
	trace := sentry.ExtractStacktrace(readConfig())

	require.NotNil(t, trace)
	require.NotEmpty(t, trace.Frames)
	last := trace.Frames[len(trace.Frames)-1]
	require.Equal(t, "readConfig", last.Function)
	require.Contains(t, last.Module, "internal/stack_test")
}

func TestSentryEventKeepsTheFirstWrapBelowOtherWrapping(t *testing.T) {
	client, err := sentry.NewClient(sentry.ClientOptions{})
	require.NoError(t, err)

	event := client.EventFromException(fmt.Errorf("agent: %w", loadAgent()), sentry.LevelError)

	var callers []string
	for _, exception := range event.Exception {
		if exception.Stacktrace == nil || len(exception.Stacktrace.Frames) == 0 {
			continue
		}
		frames := exception.Stacktrace.Frames
		callers = append(callers, frames[len(frames)-1].Function)
	}
	require.Contains(t, callers, "readConfig", "the stack of the first Wrap, not of loadAgent or of the capture")
	require.NotContains(t, callers, "loadAgent")
}
