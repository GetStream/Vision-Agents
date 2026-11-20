package session

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ToolLoadingSuite covers how a session under an agent config offers its tools: only the
// functions a tool group names, each whole, by a summary whose full description the first
// call hands over, or held back for load_tool.
type ToolLoadingSuite struct {
	suite.Suite
	ran     *ranTools
	update  harness.Tool
	lookup  harness.Tool
	refund  harness.Tool
	loading toolLoading
}

func TestToolLoadingSuite(t *testing.T) {
	suite.Run(t, new(ToolLoadingSuite))
}

type ranTools struct{ names []string }

func (r *ranTools) Run(_ context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	r.names = append(r.names, call.Name)
	return llm.TextParts("ran " + call.Name), nil
}

func (s *ToolLoadingSuite) SetupTest() {
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
	s.refund = harness.Tool{Name: "refund_booking", Description: "Refund a booking."}
	s.loading = newToolLoading(Spec{
		ConfigID: "cfg-1",
		ToolGroups: []store.ToolGroup{
			{Name: "knowledge", Tools: []string{"lookup"}, OutputRetention: "1h"},
			{Name: "refunds", Loading: store.ToolLoadingDeferred, Tools: []string{"refund_booking"}},
		},
		ConnectorBindings: []store.ConnectorBinding{{Name: "sentry", Loading: store.ToolLoadingSummary}},
	})
}

func (s *ToolLoadingSuite) offer() (harness.Tools, agent.ToolRunner) {
	tools := []harness.Tool{s.lookup, s.update, s.refund}
	full := func(name string) (harness.Tool, bool) {
		at := slices.IndexFunc(tools, named(name))
		if at < 0 || s.loading.mode(name) != store.ToolLoadingSummary {
			return harness.Tool{}, false
		}
		return tools[at], true
	}
	return s.loading.arrange(tools), newProgressiveRunner(s.ran, full)
}

func (s *ToolLoadingSuite) call(runner agent.ToolRunner, name string) string {
	parts, err := runner.Run(context.Background(), llm.ToolCall{ID: "call", Name: name, Arguments: `{"status":"resolved"}`})
	s.Require().NoError(err)
	s.Require().Len(parts, 1)
	return parts[0].Text
}

func (s *ToolLoadingSuite) TestASummarisedToolIsOfferedByItsFirstLineAndTheShapeOfItsArguments() {
	tools, _ := s.offer()

	s.Equal("Update an issue's status or assignment in Sentry. "+firstCallNote, tools.Tools[1].Description)
	s.Equal(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status":      map[string]any{"type": "string", "enum": []any{"resolved", "ignored"}},
			"description": map[string]any{"type": "string"},
			"assignees":   map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
		},
		"required": []any{"status"},
	}, tools.Tools[1].Parameters)
}

func (s *ToolLoadingSuite) TestTheFirstCallReturnsHowToUseTheToolAndRunsNothing() {
	_, runner := s.offer()

	answer := s.call(runner, "sentry__update_issue")

	s.Empty(s.ran.names)
	s.True(strings.HasPrefix(answer, "sentry__update_issue did not run."), answer)
	s.Contains(answer, "<examples>\nupdate_issue(...)\n</examples>")
	s.Contains(answer, `"description":"The new status of the issue."`)
}

func (s *ToolLoadingSuite) TestTheCallAfterThatRuns() {
	_, runner := s.offer()
	s.call(runner, "sentry__update_issue")

	answer := s.call(runner, "sentry__update_issue")

	s.Equal("ran sentry__update_issue", answer)
	s.Equal([]string{"sentry__update_issue"}, s.ran.names)
}

func (s *ToolLoadingSuite) TestAFullToolIsOfferedWholeAndRunsAtOnce() {
	tools, runner := s.offer()

	answer := s.call(runner, "lookup")

	s.Equal(s.lookup, tools.Tools[0])
	s.Equal("ran lookup", answer)
}

func (s *ToolLoadingSuite) TestADeferredToolIsHeldBackWhole() {
	tools, _ := s.offer()

	s.Equal([]harness.Tool{s.refund}, tools.Deferred)
	s.False(slices.ContainsFunc(tools.Tools, named("refund_booking")), "offered before it was loaded")
}

func (s *ToolLoadingSuite) TestALongFirstLineIsCutAtAWord() {
	s.update.Description = strings.Repeat("word ", 60)

	tools, _ := s.offer()

	summary, _ := strings.CutSuffix(tools.Tools[1].Description, " "+firstCallNote)
	s.True(strings.HasSuffix(summary, "word…"), summary)
	s.Less(len([]rune(summary)), len([]rune(strings.Repeat("word ", 60))))
}

func (s *ToolLoadingSuite) TestAFunctionNoGroupNamesIsUngroupedUnderAConfig() {
	s.Equal([]string{"cancel_booking"},
		s.loading.ungrouped([]harness.Tool{s.lookup, {Name: "cancel_booking"}, {Name: agent.VideoFramesTool}}))
}

func (s *ToolLoadingSuite) TestAGroupedFunctionNothingOnTheSessionRunsIsMissing() {
	s.Equal([]string{"refund_booking"}, s.loading.missing([]harness.Tool{s.lookup, s.update}))
	s.Empty(s.loading.missing([]harness.Tool{s.refund, s.lookup}))
}

func (s *ToolLoadingSuite) TestASessionWithoutAConfigOffersWhatItsCallerDeclares() {
	loading := newToolLoading(Spec{CustomerID: "acme", AgentID: "browser"})

	s.Empty(loading.ungrouped([]harness.Tool{{Name: "cancel_booking"}}))
	s.Equal(store.ToolLoadingFull, loading.mode("cancel_booking"))
}

func (s *ToolLoadingSuite) TestAToolKeepsWhatItGaveBackAsLongAsItsGroupSays() {
	s.Equal(options.Retention("1h"), s.loading.retention("lookup"))
	s.Equal(options.Retention(""), s.loading.retention("refund_booking"))
}

func (s *ToolLoadingSuite) TestAConnectorsToolIsOfferedAsItsBindingSays() {
	s.Equal(store.ToolLoadingSummary, s.loading.mode("sentry__list_issues"))
	s.Equal(store.ToolLoadingFull, s.loading.mode("linear__list_issues"), "a connector the config does not bind")
	s.Equal(store.ToolLoadingFull, s.loading.mode("sentry"), "a function named like an alias")
}
