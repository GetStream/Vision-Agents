package session

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// videoRunner marks the session as a video one when the caller hands back frames of the
// user's video.
type videoRunner struct {
	next    agent.ToolRunner
	session *Session
}

func (r *videoRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	parts, err := r.next.Run(ctx, call)
	if err == nil && call.Name == agent.VideoFramesTool {
		for _, part := range parts {
			if part.Image != nil {
				r.session.SawVideo()
				break
			}
		}
	}
	return parts, err
}
