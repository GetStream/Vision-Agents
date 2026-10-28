package llmtest

import (
	"context"
	"errors"
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
	s.Empty(stream.Response().OutputText, "a cancelled reply writes no text")
}

func (s *ScriptSuite) TestAScriptCancelledBeforeItsCloseFailsWithTheProviderError() {
	script := New(llm.StreamOptions{ResponseID: "turn-1", Provider: "test", Model: "test-model"})
	broken := errors.New("connection reset by peer")
	ctx, cancel := context.WithCancel(context.Background())
	script.FailOnCancel(ctx, broken)

	cancel()
	s.Require().NoError(script.Stream().Close())

	stream := script.Stream()
	var failures []error
	for stream.Next() {
		if failed, ok := stream.Current().(llm.ResponseFailed); ok {
			failures = append(failures, failed.Err)
		}
	}
	s.Equal([]error{broken}, failures, "the failure is written before the close ends the response")
}

func (s *ScriptSuite) TestAScriptClosedWhileItsContextIsLiveDoesNotFail() {
	script := New(llm.StreamOptions{ResponseID: "turn-1", Provider: "test", Model: "test-model"})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	script.FailOnCancel(ctx, errors.New("connection reset by peer"))

	s.Require().NoError(script.Stream().Close())

	stream := script.Stream()
	for stream.Next() {
		_, failed := stream.Current().(llm.ResponseFailed)
		s.False(failed, "only a cancel breaks the response")
	}
	s.Equal(llm.StatusCancelled, stream.Response().Status)
}

func (s *ScriptSuite) TestAScriptRunsItsCloseHookBeforeTheResponseEnds() {
	script := New(llm.StreamOptions{ResponseID: "turn-1", Provider: "test", Model: "test-model"})
	ran, abandonedDuringHook := false, false
	script.OnClose(func() { ran, abandonedDuringHook = true, script.Abandoned() })

	s.Require().NoError(script.Stream().Close())

	s.True(ran, "the hook never ran")
	s.False(abandonedDuringHook, "the hook ran after the response was already abandoned")
	s.True(script.Abandoned())
}
