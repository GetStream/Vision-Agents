package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// stubDrawer is an image provider with no model behind it: it draws a fixed picture, or fails
// as the test says, and records what it was asked and how long it was given.
type stubDrawer struct {
	err error

	mu       sync.Mutex
	asked    []imagegen.Request
	deadline time.Duration
}

func (d *stubDrawer) Generate(ctx context.Context, request imagegen.Request) (imagegen.Result, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.asked = append(d.asked, request)
	if until, ok := ctx.Deadline(); ok {
		d.deadline = time.Until(until)
	}
	if d.err != nil {
		return imagegen.Result{}, d.err
	}
	return imagegen.Result{Images: []imagegen.Image{{Data: []byte("png"), MediaType: "image/png", Width: 1024, Height: 1024}}}, nil
}

func (d *stubDrawer) Start(context.Context) error { return nil }
func (d *stubDrawer) Close() error                { return nil }
func (d *stubDrawer) Provider() string            { return "stub" }
func (d *stubDrawer) Model() string               { return "draw" }

func (d *stubDrawer) requests() []imagegen.Request {
	d.mu.Lock()
	defer d.mu.Unlock()
	return append([]imagegen.Request(nil), d.asked...)
}

// imageRouter routes stub/draw to s.drawer.
func (s *AgentSuite) imageRouter() *imagerouter.Router {
	registry := imagerouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (imagegen.Provider, error) { return s.drawer, nil })
	router, err := imagerouter.New(imagerouter.Options{
		Config: routing.ModalityConfig{Providers: []routing.ProviderConfig{{
			Provider: "stub", Model: "draw", Languages: []string{"en"}, Tier: routing.LowLatency,
			Price: routing.Price{PerImage: 0.04},
		}}},
		Registry: registry,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

func drawCall(id, prompt string) llm.ToolCall {
	return llm.ToolCall{ID: id, Name: GenerateImageTool, Arguments: fmt.Sprintf(`{"prompt":%q}`, prompt)}
}

func (s *AgentSuite) offersDrawing() bool {
	_, offered := s.agent.availableTools().Lookup(GenerateImageTool)
	return offered
}

func (s *AgentSuite) TestAWrittenConversationWithAnImageTargetDrawsAndAttachesThePicture() {
	s.drawer = &stubDrawer{}
	s.shelf = &codeShelf{}
	s.joinText()
	s.Require().True(s.offersDrawing())
	s.NoError(s.agent.availableTools().Validate())

	ran := s.runsTool(drawCall("draw-1", "a yellow watering can"), 1)

	s.Require().NoError(ran.Err)
	s.Equal([]sandbox.Attachment{{Name: "picture-1.png", MIME: "image/png", URL: "https://cdn.example/picture-1.png", Size: 3}}, ran.Files)
	s.Contains(ran.Result, "attached to your reply")
	s.NotContains(ran.Result, "png", "the model is told about the picture, never handed it")
	asked := s.drawer.requests()
	s.Require().Len(asked, 1)
	s.Equal("a yellow watering can", asked[0].Prompt)
	s.Equal(1, asked[0].N, "one call is one picture")
}

func (s *AgentSuite) TestADrawIsGivenTheImageDeadlineNotTheCallersToolTimeout() {
	// A picture takes one to two minutes; the 30 s a caller's tool gets would fail most.
	s.drawer = &stubDrawer{}
	s.shelf = &codeShelf{}
	s.joinText()

	s.Require().NoError(s.runsTool(drawCall("draw-1", "a field"), 1).Err)

	s.drawer.mu.Lock()
	given := s.drawer.deadline
	s.drawer.mu.Unlock()
	s.Greater(given, imagerouter.Deadline-10*time.Second)
	s.LessOrEqual(given, imagerouter.Deadline)
}

func (s *AgentSuite) TestAnAgentWithoutAnImageTargetIsNotOfferedDrawing() {
	s.shelf = &codeShelf{}
	s.joinText()

	s.False(s.offersDrawing())
}

func (s *AgentSuite) TestAConversationWithNowhereToPutAPictureIsNotOfferedDrawing() {
	s.drawer = &stubDrawer{}
	s.joinText()

	s.False(s.offersDrawing(), "a picture nobody can see is paid for and lost")
}

func (s *AgentSuite) TestACallersOwnGenerateImageKeepsTheName() {
	// The tool customers wrote before drawing was built in keeps working, unchanged.
	s.drawer = &stubDrawer{}
	s.shelf = &codeShelf{}
	s.ownsTools("drawn by the caller")
	s.tools.Tools = append(s.tools.Tools, harness.Tool{Name: GenerateImageTool, Description: "mine"})
	s.joinText()

	tools := s.agent.availableTools()
	s.NoError(tools.Validate(), "the name is offered once")
	tool, offered := tools.Lookup(GenerateImageTool)
	s.Require().True(offered)
	s.Equal("mine", tool.Description)

	ran := s.runsTool(drawCall("draw-1", "a field"), 1)
	s.Equal("drawn by the caller", ran.Result)
	s.Empty(s.drawer.requests())
}

func (s *AgentSuite) TestAPictureTheSafetyFilterRefusedIsToldInWordsNotAsAFailure() {
	// An error tells the model the tool broke, and it rewords the prompt to get past the filter.
	s.drawer = &stubDrawer{err: imagegen.Fail(imagegen.ContentFiltered, true, errors.New("refused"))}
	s.shelf = &codeShelf{}
	s.joinText()

	ran := s.runsTool(drawCall("draw-1", "something refused"), 1)

	s.NoError(ran.Err)
	s.Empty(ran.Files)
	s.Contains(ran.Result, "safety filter refused")
	s.Contains(ran.Result, "do not try again")
	s.NotContains(ran.Result, didNotWork)
}

func (s *AgentSuite) TestAProviderThatBrokeIsAFailedCall() {
	s.drawer = &stubDrawer{err: imagegen.Fail(imagegen.ProviderFailed, true, errors.New("upstream 500"))}
	s.shelf = &codeShelf{}
	s.joinText()

	ran := s.runsTool(drawCall("draw-1", "a field"), 1)

	s.Error(ran.Err)
	s.Empty(ran.Files)
}

func (s *AgentSuite) TestOneReplyDrawsAtMostFourPictures() {
	s.drawer = &stubDrawer{}
	s.shelf = &codeShelf{}
	s.joinText()

	for n := 1; n <= maxDrawsPerTurn+1; n++ {
		call := drawCall(fmt.Sprintf("draw-%d", n), "a field")
		s.agent.runTool(harness.ToolRequested{TurnID: "one-reply", Call: call})
		s.eventually(func() bool { return len(toolsRanIn(s.reported())) == n }, "the tool never settled")
	}

	ran := toolsRanIn(s.reported())
	s.Len(s.drawer.requests(), maxDrawsPerTurn)
	s.Empty(ran[maxDrawsPerTurn].Files)
	s.Contains(ran[maxDrawsPerTurn].Result, "at most 4 pictures")

	s.agent.runTool(harness.ToolRequested{TurnID: "next-reply", Call: drawCall("draw-next", "a field")})
	s.eventually(func() bool { return len(toolsRanIn(s.reported())) == maxDrawsPerTurn+2 }, "the tool never settled")
	s.Len(s.drawer.requests(), maxDrawsPerTurn+1, "the next reply may draw again")
}
