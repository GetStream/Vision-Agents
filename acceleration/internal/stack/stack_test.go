package stack_test

import (
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

type refusal struct{ reason string }

func (r *refusal) Error() string { return r.reason }

func readConfig() error {
	return stack.Wrap(fmt.Errorf("config: read: %w", io.ErrUnexpectedEOF))
}

func loadAgent() error {
	return stack.Wrap(readConfig())
}

func TestWrapOfNilIsNil(t *testing.T) {
	require.NoError(t, stack.Wrap(nil))
}

func TestWrapKeepsTheMessageAndTheChain(t *testing.T) {
	err := readConfig()

	require.Equal(t, "config: read: unexpected EOF", err.Error())
	require.ErrorIs(t, err, io.ErrUnexpectedEOF)

	wrapped := stack.Wrap(&refusal{reason: "no"})
	found, ok := errors.AsType[*refusal](wrapped)
	require.True(t, ok)
	require.Equal(t, "no", found.reason)
}

func TestTraceStartsAtTheCallerOfWrap(t *testing.T) {
	trace := stack.Trace(readConfig())

	first, _, _ := strings.Cut(trace, "\n")
	require.True(t, strings.HasSuffix(first, "stack_test.readConfig"), "first frame is %q", first)
	require.Contains(t, trace, "stack_test.go:")
}

func TestWrappingAgainKeepsTheFirstStack(t *testing.T) {
	inner := readConfig()
	outer := loadAgent()

	require.Equal(t, inner, stack.Wrap(inner), "already carrying a stack, so returned as it is")

	first, _, _ := strings.Cut(stack.Trace(outer), "\n")
	require.True(t, strings.HasSuffix(first, "stack_test.readConfig"),
		"the stack is where the error entered, not where it was passed up: %q", first)
}

func TestAStackBelowOtherWrappingIsStillFound(t *testing.T) {
	err := fmt.Errorf("agent: %w", readConfig())

	require.Contains(t, stack.Trace(err), "stack_test.readConfig")
	require.Equal(t, err, stack.Wrap(err))
}

func TestTraceOfAnUnwrappedErrorIsEmpty(t *testing.T) {
	require.Empty(t, stack.Trace(errors.New("plain")))
	require.Empty(t, stack.Trace(nil))
}

func TestFormatPrintsTheStackOnlyWhenAskedFor(t *testing.T) {
	err := readConfig()

	require.Equal(t, "config: read: unexpected EOF", fmt.Sprintf("%v", err))
	require.Equal(t, "config: read: unexpected EOF", fmt.Sprintf("%s", err))
	require.Equal(t, `"config: read: unexpected EOF"`, fmt.Sprintf("%q", err))

	verbose := fmt.Sprintf("%+v", err)
	require.True(t, strings.HasPrefix(verbose, "config: read: unexpected EOF\n"))
	require.Contains(t, verbose, "stack_test.readConfig")
}
