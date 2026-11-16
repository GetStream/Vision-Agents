package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// GenerateImageTool is the built-in tool an agent whose config names an image target draws
// pictures with.
const GenerateImageTool = "generate_image"

// maxDrawsPerTurn caps the pictures one reply asks for. Each is paid for, so a model looping on
// "one more" is stopped well before a reply's worth of pictures piles up in the conversation.
const maxDrawsPerTurn = 4

// aspectRatios are the shapes generate_image offers: the ones every image model routed today
// draws (FAL's Qwen Image 3 takes these five, and Gemini these and more), so a shape the model
// asks for is refused in its turn, where it can pick another, not by the draw after it ended.
var aspectRatios = []string{"1:1", "4:3", "3:4", "16:9", "9:16"}

// DrawRequest is a picture generate_image asks for, as the session queues it.
type DrawRequest struct {
	// SessionID, TurnID and ToolCallID are the call that asked: a call run again in its turn
	// queues nothing more, and a call id a later turn reuses, such as ask-1, is another call.
	SessionID   string
	TurnID      string
	ToolCallID  string
	Prompt      string
	AspectRatio string
	// Target, AgentID, CallID, Region, Tags and Attribution are what the draw is routed and
	// billed under, as the turn's own work is.
	Target      string
	AgentID     string
	CallID      string
	Region      string
	Tags        routing.Tags
	Attribution routing.Attribution
}

// Drawer queues a picture to be drawn after the turn and posted into the conversation once it
// is ready, as a message of the agent's, and reports whether it queued one: a call run again
// queues nothing.
type Drawer func(ctx context.Context, request DrawRequest) (bool, error)

// generateImageTool is generate_image as the model is offered it.
func generateImageTool() harness.Tool {
	return harness.Tool{
		Name: GenerateImageTool,
		Description: "Draw one picture from a description. It is drawn in the background and posted " +
			"in this conversation when it is ready, a minute or two later, so tell the person it is " +
			"on its way and carry on. Use it when the person asks for a picture, an illustration or " +
			"a visual. Describe the subject, the style and what must not appear (such as text).",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"prompt": map[string]any{
					"type":        "string",
					"description": "What to draw, in detail: subject, setting, style, colours.",
				},
				"aspect_ratio": map[string]any{
					"type":        "string",
					"enum":        aspectRatios,
					"description": "The picture's shape as width:height. Leave it out for square.",
				},
			},
			"required": []string{"prompt"},
		},
		DisplayTitle: "Drawing a picture",
	}
}

// draws reports whether this agent is offered generate_image: a written conversation whose
// config names an image target, where the session can queue a picture to post. A voice model's
// caller would hear nothing of a picture posted minutes later. Unlike the other built-ins, it
// takes the name from a tool of the caller's own: that tool is the workaround customers wrote
// before drawing was built in, and a config naming an image target asks for this one.
func (a *Agent) draws() bool {
	return a.options.Text && a.options.Draws != nil && a.options.ImageTarget != ""
}

// generateImage queues one picture and answers at once, so the turn ends and the model replies
// while the picture is drawn. The model is told the picture is coming, never handed it.
func (a *Agent) generateImage(ctx context.Context, call llm.ToolCall) (string, error) {
	var arguments struct {
		Prompt      string `json:"prompt"`
		AspectRatio string `json:"aspect_ratio"`
	}
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		return "", fmt.Errorf("agent: could not read the generate_image arguments: %w", err)
	}
	prompt := strings.TrimSpace(arguments.Prompt)
	if prompt == "" {
		return "Nothing was drawn: describe the picture in prompt.", nil
	}
	shape := strings.TrimSpace(arguments.AspectRatio)
	if shape != "" && !slices.Contains(aspectRatios, shape) {
		return fmt.Sprintf("Nothing was drawn: aspect_ratio is one of %s, not %q. Ask again with one of them.",
			strings.Join(aspectRatios, ", "), shape), nil
	}
	if !a.drawAllowed(call.TurnID) {
		return fmt.Sprintf("Nothing was drawn: one reply may draw at most %d pictures. "+
			"Tell the person what is on its way so far.", maxDrawsPerTurn), nil
	}
	queued, err := a.options.Draws(ctx, DrawRequest{
		SessionID: a.options.SessionID, TurnID: call.TurnID, ToolCallID: call.ID, Prompt: prompt, AspectRatio: shape,
		Target: a.options.ImageTarget, AgentID: a.options.AgentID, CallID: a.options.CallID,
		Region: a.options.Region, Tags: a.options.Tags, Attribution: routing.AttributionFrom(ctx),
	})
	if err != nil || !queued {
		// Only a picture queued counts against the reply: one that failed, or a call run again
		// that queued nothing, gives back the place drawAllowed held for it.
		a.drawRefused(call.TurnID)
	}
	if err != nil {
		// The model is told in words it can pass on; what broke is for the logs.
		a.logger.Error("could not queue a picture", "call", call.ID, "error", err)
		return "Nothing was drawn: the picture could not be started. Tell the person to ask again in a moment.", nil
	}
	return "The picture is being drawn and will be posted in this conversation when it is ready, " +
		"in a minute or two. Tell the person it is on its way; do not describe it as done, link it " +
		"or embed it.", nil
}

// drawRefused gives back the place drawAllowed held for a draw of the reply turnID that queued
// nothing.
func (a *Agent) drawRefused(turnID string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.drawnTurn == turnID && a.drawnCount > 0 {
		a.drawnCount--
	}
}

// drawAllowed holds a place for a draw of its reply, and reports whether the reply had any left.
// The place is held before the draw is queued, so calls of one reply running at once never
// queue more than the cap; drawRefused gives it back when nothing was queued.
func (a *Agent) drawAllowed(turnID string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	// Only the latest reply is counted: a reply's calls run while it is the latest, and a
	// count kept per reply would grow for as long as a persistent conversation lives.
	if a.drawnTurn != turnID {
		a.drawnTurn, a.drawnCount = turnID, 0
	}
	if a.drawnCount >= maxDrawsPerTurn {
		return false
	}
	a.drawnCount++
	return true
}
