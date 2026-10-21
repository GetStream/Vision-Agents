package api

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
)

func TestATurnThatSaidNothingIsNoReply(t *testing.T) {
	_, err := replyText(nil)

	var failure *scheduling.Failure
	require.True(t, errors.As(err, &failure), "%v", err)
	require.Equal(t, scheduling.ReasonNoReply, failure.Reason)
	require.False(t, failure.Final, "a later attempt may read the reply back")
}
