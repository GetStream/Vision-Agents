package session

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ProgressiveToolsSuite covers offering tools by a summary and handing their full
// description over on the first call.
type ProgressiveToolsSuite struct {
	suite.Suite
	ran    *ranTools
	update harness.Tool
	lookup harness.Tool
}

func TestProgressiveToolsSuite(t *testing.T) {
	suite.Run(t, new(ProgressiveToolsSuite))
}

type ranTools struct{ names []string }

func (r *ranTools) Run(_ context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	r.names = append(r.names, call.Name)
	return llm.TextParts("ran " + call.Name), nil
}

func (s *ProgressiveToolsSuite) SetupTest() {
	s.ran = &ranTools{}
	s.update = harness.Tool{
		Name: "sentry__update_issue",
		Description: "Update an issue's status or assignment in Sentry.\n\n" +
			"Use this tool when you need to resolve an issue.\n\n<examples>\nupdate_issue(...)\n</examples>",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"status": map[string]any{
					"type": "string", "enum": []any{"resolved", "ignored"},
					"description": "The new status of the issue.",
				},
				"description": map[string]any{"type": "string", "title": "A note", "examples": []any{"fixed"}},
				"assignees": map[string]any{
					"type":  "array",
					"items": map[string]any{"type": "string", "description": "A user or team."},
				},
			},
			"required": []any{"status"},
		},
	}
	s.lookup = harness.Tool{Name: "lookup", Description: "Look it up.\nIn the knowledge base."}
}

func (s *ProgressiveToolsSuite) offer() ([]harness.Tool, agent.ToolRunner) {
	return progressively([]harness.Tool{s.lookup, s.update}, []harness.Tool{s.update}, s.ran)
}

func (s *ProgressiveToolsSuite) call(runner agent.ToolRunner, name string) string {
	parts, err := runner.Run(context.Background(), llm.ToolCall{ID: "call", Name: name, Arguments: `{"status":"resolved"}`})
	s.Require().NoError(err)
	s.Require().Len(parts, 1)
	return parts[0].Text
}

func (s *ProgressiveToolsSuite) TestADeferredToolIsOfferedByItsFirstLineAndTheShapeOfItsArguments() {
	tools, _ := s.offer()

	s.Equal("Update an issue's status or assignment in Sentry. "+firstCallNote, tools[1].Description)
	s.Equal(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status":      map[string]any{"type": "string", "enum": []any{"resolved", "ignored"}},
			"description": map[string]any{"type": "string"},
			"assignees":   map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
		},
		"required": []any{"status"},
	}, tools[1].Parameters)
}

func (s *ProgressiveToolsSuite) TestTheFirstCallReturnsHowToUseTheToolAndRunsNothing() {
	_, runner := s.offer()

	answer := s.call(runner, "sentry__update_issue")

	s.Empty(s.ran.names)
	s.True(strings.HasPrefix(answer, "sentry__update_issue did not run."), answer)
	s.Contains(answer, "<examples>\nupdate_issue(...)\n</examples>")
	s.Contains(answer, `"description":"The new status of the issue."`)
}

func (s *ProgressiveToolsSuite) TestTheCallAfterThatRuns() {
	_, runner := s.offer()
	s.call(runner, "sentry__update_issue")

	answer := s.call(runner, "sentry__update_issue")

	s.Equal("ran sentry__update_issue", answer)
	s.Equal([]string{"sentry__update_issue"}, s.ran.names)
}

func (s *ProgressiveToolsSuite) TestAToolThatIsNotDeferredIsOfferedWholeAndRunsAtOnce() {
	tools, runner := s.offer()

	answer := s.call(runner, "lookup")

	s.Equal(s.lookup, tools[0])
	s.Equal("ran lookup", answer)
}

func (s *ProgressiveToolsSuite) TestNothingDeferredLeavesTheToolsAndTheRunnerAsTheyWere() {
	tools, runner := progressively([]harness.Tool{s.update}, nil, s.ran)

	s.Equal([]harness.Tool{s.update}, tools)
	s.Same(s.ran, runner)
}

func (s *ProgressiveToolsSuite) TestALongFirstLineIsCutAtAWord() {
	s.update.Description = strings.Repeat("word ", 60)

	tools, _ := s.offer()

	summary, _ := strings.CutSuffix(tools[1].Description, " "+firstCallNote)
	s.True(strings.HasSuffix(summary, "word…"), summary)
	s.LessOrEqual(len([]rune(summary)), summaryLimit+1)
}
