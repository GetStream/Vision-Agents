package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// GenerateImageTool is the built-in tool an agent whose config names an image target draws
// pictures with.
const GenerateImageTool = "generate_image"

// maxDrawsPerTurn caps the pictures one reply asks for. A reply carries at most eight files,
// and each picture is paid for, so a model looping on "one more" is stopped well before either.
const maxDrawsPerTurn = 4

// generateImageTool is generate_image as the model is offered it.
func generateImageTool() harness.Tool {
	return harness.Tool{
		Name: GenerateImageTool,
		Description: "Draw one picture from a description and attach it to your reply. Use it when " +
			"the person asks for a picture, an illustration or a visual. It takes a minute or two. " +
			"Describe the subject, the style and what must not appear (such as text). Do not put " +
			"the picture in your reply yourself: it is attached for you.",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"prompt": map[string]any{
					"type":        "string",
					"description": "What to draw, in detail: subject, setting, style, colours.",
				},
				"aspect_ratio": map[string]any{
					"type":        "string",
					"description": "The picture's shape as width:height, such as 1:1, 16:9 or 9:16. Leave it out for square.",
				},
			},
			"required": []string{"prompt"},
		},
		DisplayTitle: "Drawing a picture",
	}
}

// draws reports whether this agent is offered generate_image: a written conversation whose
// config names an image target, on a deployment that routes images, with a channel to put the
// picture in. A voice model would be answered minutes after it asked. Unlike the other
// built-ins, it takes the name from a tool of the caller's own: that tool is the workaround
// customers wrote before drawing was built in, and a config naming an image target asks for this one.
func (a *Agent) draws() bool {
	return a.options.Text && a.options.Image != nil && a.options.ImageTarget != "" && a.options.Publish != nil
}

// generateImage draws one picture and publishes it to the conversation, to be attached to the
// reply. The model is told what happened in words and never handed the picture: its bytes in
// the history would cost every later turn, and an image result is sent to the vision worker.
//
// A refusal is an answer, not an error. An error tells the model the tool broke, and it tries
// again with other words, which for a safety filter is shopping for a laxer one and for a spent
// limit is paying for nothing.
func (a *Agent) generateImage(ctx context.Context, call llm.ToolCall) (string, []sandbox.Attachment, error) {
	var arguments struct {
		Prompt      string `json:"prompt"`
		AspectRatio string `json:"aspect_ratio"`
	}
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		return "", nil, fmt.Errorf("agent: could not read the generate_image arguments: %w", err)
	}
	prompt := strings.TrimSpace(arguments.Prompt)
	if prompt == "" {
		return "Nothing was drawn: describe the picture in prompt.", nil, nil
	}
	if !a.drawAllowed(call.TurnID) {
		return fmt.Sprintf("Nothing was drawn: one reply may draw at most %d pictures. "+
			"Tell the person what you drew so far.", maxDrawsPerTurn), nil, nil
	}

	generation, err := a.options.Image.Generate(ctx, imagerouter.Request{
		CustomerID: a.options.CustomerID,
		AgentID:    a.options.AgentID,
		CallID:     a.options.CallID,
		Region:     a.options.Region,
		Tags:       a.options.Tags,
		Target:     a.options.ImageTarget,
		Image: imagegen.Request{
			Prompt:      prompt,
			AspectRatio: strings.TrimSpace(arguments.AspectRatio),
			N:           1,
		},
	})
	if err != nil {
		if said, refused := drawRefusal(err); refused {
			return said, nil, nil
		}
		return "", nil, fmt.Errorf("agent: draw: %w", err)
	}

	var published []sandbox.Attachment
	for index, picture := range generation.Images {
		file := sandbox.File{
			Name: fmt.Sprintf("picture-%d.%s", index+1, strings.TrimPrefix(picture.MediaType, "image/")),
			MIME: picture.MediaType,
			Data: picture.Data,
		}
		attachment, err := a.options.Publish(ctx, file)
		if err != nil {
			return "", nil, fmt.Errorf("agent: publish a drawn picture: %w", err)
		}
		published = append(published, attachment)
	}
	if len(published) == 0 {
		return "", nil, errors.New("agent: the image provider answered with no picture")
	}
	return "The picture is drawn and will be attached to your reply. Describe it in a sentence; " +
		"do not link or embed it.", published, nil
}

// drawAllowed counts a draw against its reply, and reports whether the reply had any left.
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

// drawRefusal is what the model is told about a draw that was refused rather than broken, and
// false for anything else.
func drawRefusal(err error) (string, bool) {
	for _, money := range []error{
		policy.ErrSuspended, policy.ErrNoCredits, policy.ErrTopUpRequired, policy.ErrSystemLimit,
		policy.ErrBudgetSpent, policy.ErrUserLimit,
	} {
		if errors.Is(err, money) {
			return "Nothing was drawn: the spending limit for pictures has been reached. " +
				"Tell the person you cannot draw right now; do not try again.", true
		}
	}
	if routing.ReservationRefused(err) {
		return "Nothing was drawn: there is no budget left for pictures. " +
			"Tell the person you cannot draw right now; do not try again.", true
	}
	switch imagegen.CodeOf(err) {
	case imagegen.ContentFiltered:
		return "Nothing was drawn: the safety filter refused this picture. Tell the person; " +
			"do not try again with reworded wording.", true
	case imagegen.UnsupportedOption:
		return "Nothing was drawn: that shape or size cannot be drawn. Try again without " +
			"aspect_ratio, or with 1:1, 16:9 or 9:16.", true
	case imagegen.Timeout:
		return "Nothing was drawn: the picture took too long. Tell the person it can be tried again.", true
	}
	return "", false
}
