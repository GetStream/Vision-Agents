package llmtest

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type ScriptSuite struct {
	suite.Suite
}

func TestScriptSuite(t *testing.T) {
	suite.Run(t, new(ScriptSuite))
}

func (s *ScriptSuite) TestACancelledScriptEndsAbandonedWithNoFailure() {
	script := New(llm.StreamOptions{ResponseID: "turn-1", Provider: "test", Model: "test-model"})
	stream := script.Stream()

	script.Cancel()

	for stream.Next() {
		_, failed := stream.Current().(llm.ResponseFailed)
		s.False(failed, "a cancelled reply reports no provider failure")
	}
	s.NoError(stream.Err())
	s.Equal(llm.StatusCancelled, stream.Response().Status)
}
