package api

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
)

func TestATurnThatSaidNothingIsNoReply(t *testing.T) {
	_, _, err := captured(nil, nil)

	var failure *scheduling.Failure
	require.True(t, errors.As(err, &failure), "%v", err)
	require.Equal(t, scheduling.ReasonNoReply, failure.Reason)
	require.False(t, failure.Final, "a later attempt may read the reply back")
}

func TestATurnThatOnlyMadeAFileIsCarried(t *testing.T) {
	chart := sandbox.Attachment{Name: "chart.png", MIME: "image/png", URL: "https://cdn.example/chart.png"}

	text, files, err := captured([]string{""}, []sandbox.Attachment{chart})

	require.NoError(t, err)
	require.Empty(t, text)
	require.Equal(t, []sandbox.Attachment{chart}, files)
}

func TestTheFilesCodeHandedBackAreThoseOfARunAndOfSettledWork(t *testing.T) {
	chart := sandbox.Attachment{Name: "chart.png", MIME: "image/png", URL: "https://cdn.example/chart.png"}

	require.Equal(t, []sandbox.Attachment{chart}, madeBy(agent.ToolRan{Files: []sandbox.Attachment{chart}}))
	require.Equal(t, []sandbox.Attachment{chart}, madeBy(agent.TaskSettled{Files: []sandbox.Attachment{chart}}))
	require.Nil(t, madeBy(agent.Responded{Text: "Here is the chart."}))
}
