package session

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/research"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/managed"
)

// ResearchProgress is an additive event; older clients can ignore it.
type ResearchProgress struct {
	ToolCallID        string
	Phase             string
	ElapsedMS         int64
	VerifiedCitations int
}

type researchRunner struct {
	workspace *managed.Workspace
	next      agent.ToolRunner
	emit      func(Event)
	progress  func(string, string)
}

func (r *researchRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	if call.Name != "investigate_sdk" {
		return r.next.Run(ctx, call)
	}
	var in research.Request
	if len(call.Arguments) > 8192 || json.Unmarshal([]byte(call.Arguments), &in) != nil {
		return nil, errors.New("research: invalid arguments")
	}
	result := r.workspace.Research(ctx, in, func(p research.Progress) {
		if r.progress != nil {
			r.progress(call.ID, p.Phase)
		}
		r.emit(ResearchProgress{ToolCallID: call.ID, Phase: p.Phase, ElapsedMS: p.ElapsedMS, VerifiedCitations: p.VerifiedCitations})
	})
	r.emit(ResearchProgress{ToolCallID: call.ID, Phase: result.Status, ElapsedMS: result.ElapsedMS, VerifiedCitations: len(result.Citations)})
	data, err := json.Marshal(result)
	return llm.TextParts(string(data)), err
}
func researchTool(profile research.Profile) harness.Tool {
	manifest, _ := json.Marshal(profile.Repositories)
	return harness.Tool{Name: "investigate_sdk", Description: "Read source using Cursor on the managed workspace. Use only these repositories and scopes: " + string(manifest) + ". Always preserve returned commit-linked citations. If research_failed or insufficient_evidence, explain the limitation; never infer an implementation. Only listed repositories are available. A repository may advertise additional scopes; keep the requested product and SDK when using it. Investigate a dependency only if it is explicitly listed, using its own supported scope.", Parameters: map[string]any{
		"type": "object", "additionalProperties": false, "required": []string{"product", "sdk", "question", "repository_ids"}, "properties": map[string]any{
			"product": map[string]any{"type": "string"}, "sdk": map[string]any{"type": "string"}, "question": map[string]any{"type": "string"}, "repository_ids": map[string]any{"type": "array", "items": map[string]string{"type": "string"}, "minItems": 1},
		}}}
}
