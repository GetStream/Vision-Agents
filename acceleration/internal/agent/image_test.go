package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// drawQueue stands in for the session's queue of pictures: it records what it was asked to
// queue, or refuses as the test says.
type drawQueue struct {
	err error

	mu     sync.Mutex
	queued []DrawRequest
}

func (q *drawQueue) queue(_ context.Context, request DrawRequest) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.err != nil {
		return q.err
	}
	q.queued = append(q.queued, request)
	return nil
}

func (q *drawQueue) requests() []DrawRequest {
	q.mu.Lock()
	defer q.mu.Unlock()
	return append([]DrawRequest(nil), q.queued...)
}

func drawCall(id, prompt string) llm.ToolCall {
	return llm.ToolCall{ID: id, Name: GenerateImageTool, Arguments: fmt.Sprintf(`{"prompt":%q,"aspect_ratio":"16:9"}`, prompt)}
}

func (s *AgentSuite) offersDrawing() bool {
	_, offered := s.agent.availableTools().Lookup(GenerateImageTool)
	return offered
}

func (s *AgentSuite) TestAPictureIsQueuedAndTheTurnIsAnsweredAtOnce() {
	// A picture takes a minute or two; the turn ends now and the picture is posted when ready.
	s.drawings = &drawQueue{}
	s.attribution = routing.Attribution{APIKey: "key-1", UserID: "ana", SessionID: "session-1"}
	s.joinText()
	s.Require().True(s.offersDrawing())
	s.NoError(s.agent.availableTools().Validate())

	ran := s.runsTool(drawCall("draw-1", "a yellow watering can"), 1)

	s.Require().NoError(ran.Err)
	s.Empty(ran.Files, "nothing is attached to this reply: the picture comes as its own message")
	s.Contains(ran.Result, "being drawn")
	s.Contains(ran.Result, "posted in this conversation")
	queued := s.drawings.requests()
	s.Require().Len(queued, 1)
	s.Equal(DrawRequest{
		ToolCallID: "draw-1", Prompt: "a yellow watering can", AspectRatio: "16:9", Target: "image-fast",
		AgentID: s.agentID, Attribution: routing.Attribution{APIKey: "key-1", UserID: "ana", SessionID: "session-1"},
	}, queued[0], "billed under the session that asked, once it is drawn")
}

func (s *AgentSuite) TestAPictureThatCouldNotBeQueuedIsToldInPlainWords() {
	// The person would wait for a picture nobody is drawing; the model is told so, and what broke
	// stays in the logs rather than in what the model may repeat.
	s.drawings = &drawQueue{err: errors.New("pq: connection refused to 10.0.0.7:5432")}
	s.joinText()

	ran := s.runsTool(drawCall("draw-1", "a field"), 1)

	s.NoError(ran.Err)
	s.Contains(ran.Result, "could not be started")
	s.NotContains(ran.Result, "being drawn")
	s.NotContains(ran.Result, "10.0.0.7")
	s.NotContains(ran.Result, "pq:")
}

func (s *AgentSuite) TestAShapeNoModelDrawsIsRefusedInTheTurn() {
	// Refused after the turn, the model could not pick another shape; it would only be told by a
	// failure posted minutes later.
	s.drawings = &drawQueue{}
	s.joinText()

	ran := s.runsTool(llm.ToolCall{ID: "draw-1", Name: GenerateImageTool, Arguments: `{"prompt":"a field","aspect_ratio":"5:2"}`}, 1)

	s.NoError(ran.Err)
	s.Contains(ran.Result, "16:9")
	s.Contains(ran.Result, `"5:2"`)
	s.Empty(s.drawings.requests(), "nothing is queued for a shape nobody draws")
	tool, _ := s.agent.availableTools().Lookup(GenerateImageTool)
	properties, _ := tool.Parameters["properties"].(map[string]any)
	shape, _ := properties["aspect_ratio"].(map[string]any)
	s.Equal(aspectRatios, shape["enum"], "the model is offered the shapes, not left to guess")
}

func (s *AgentSuite) TestAnAgentWithoutAnImageTargetIsNotOfferedDrawing() {
	s.joinText()

	s.False(s.offersDrawing())
}

func (s *AgentSuite) TestAnImageTargetReplacesACallersOwnGenerateImage() {
	// The caller's tool is the workaround customers wrote before drawing was built in: it asked
	// their own backend to draw, and needed a longer tool timeout to do it.
	s.drawings = &drawQueue{}
	s.ownsTools("drawn by the caller")
	s.tools.Tools = append(s.tools.Tools, harness.Tool{Name: GenerateImageTool, Description: "mine"})
	s.joinText()

	tools := s.agent.availableTools()
	s.NoError(tools.Validate(), "the name is offered once")
	tool, offered := tools.Lookup(GenerateImageTool)
	s.Require().True(offered)
	s.Equal(generateImageTool().Description, tool.Description)

	ran := s.runsTool(drawCall("draw-1", "a field"), 1)
	s.Require().NoError(ran.Err)
	s.Len(s.drawings.requests(), 1)
	s.Empty(s.runner.asked(), "the caller's own tool is not asked")
}

func (s *AgentSuite) TestWithoutAnImageTargetACallersOwnGenerateImageRunsUnchanged() {
	s.ownsTools("drawn by the caller")
	s.tools.Tools = append(s.tools.Tools, harness.Tool{Name: GenerateImageTool, Description: "mine"})
	s.joinText()

	tool, offered := s.agent.availableTools().Lookup(GenerateImageTool)
	s.Require().True(offered)
	s.Equal("mine", tool.Description)

	ran := s.runsTool(drawCall("draw-1", "a field"), 1)
	s.Equal("drawn by the caller", ran.Result)
	s.Len(s.runner.asked(), 1)
}

func (s *AgentSuite) TestOneReplyQueuesAtMostFourPictures() {
	s.drawings = &drawQueue{}
	s.joinText()

	for n := 1; n <= maxDrawsPerTurn+1; n++ {
		call := drawCall(fmt.Sprintf("draw-%d", n), "a field")
		s.agent.runTool(harness.ToolRequested{TurnID: "one-reply", Call: call})
		s.eventually(func() bool { return len(toolsRanIn(s.reported())) == n }, "the tool never settled")
	}

	ran := toolsRanIn(s.reported())
	s.Len(s.drawings.requests(), maxDrawsPerTurn)
	s.Contains(ran[maxDrawsPerTurn].Result, "at most 4 pictures")

	s.agent.runTool(harness.ToolRequested{TurnID: "next-reply", Call: drawCall("draw-next", "a field")})
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == maxDrawsPerTurn+2 }, "the tool never settled")
	s.Len(s.drawings.requests(), maxDrawsPerTurn+1, "the next reply may draw again")
}
