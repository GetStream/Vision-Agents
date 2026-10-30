package session

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// unstoredScheduler runs the scheduling tools with no store, so only a call refused before it
// reads one can be run.
func unstoredScheduler(spec Spec) *scheduleRunner {
	runner := &scheduleRunner{spec: spec, owned: map[string]bool{}}
	for _, tool := range scheduleTools() {
		runner.owned[tool.Name] = true
	}
	return runner
}

// refusedAs is the error code a scheduling tool's result refused with.
func refusedAs(t *testing.T, runner *scheduleRunner, ctx context.Context, name string) string {
	t.Helper()
	parts, err := runner.Run(ctx, llm.ToolCall{ID: "call-1", Name: name, Arguments: `{"id":"task-1","title":"Pay","instructions":"Pay it.","in":"PT2H"}`})
	require.NoError(t, err)
	return llm.TextOf(parts)
}

func TestATaskIsNeitherMadeNorChangedInATurnAScheduledTaskFired(t *testing.T) {
	runner := unstoredScheduler(Spec{})
	ctx := agent.WithProvenance(context.Background(), agent.Provenance{Scheduled: true})

	require.Contains(t, refusedAs(t, runner, ctx, scheduleTaskTool), `"error":"unattended"`)
	require.Contains(t, refusedAs(t, runner, ctx, updateScheduledTool), `"error":"unattended"`)
}

func TestATaskIsNotMadeInASessionAnEventOpened(t *testing.T) {
	for _, origin := range []string{routing.OriginHookPlugin, routing.OriginHookConnector} {
		runner := unstoredScheduler(Spec{Attribution: routing.Attribution{APIKey: origin}})

		require.Contains(t, refusedAs(t, runner, context.Background(), scheduleTaskTool), `"error":"unattended"`, origin)
	}
}

func TestATaskIsNotMadeAfterTheModelReadAnotherToolsResult(t *testing.T) {
	runner := unstoredScheduler(Spec{})
	ctx := agent.WithProvenance(context.Background(), agent.Provenance{ToolsRead: []string{"web_search"}})

	require.Contains(t, refusedAs(t, runner, ctx, scheduleTaskTool), `"error":"not_asked_by_user"`)
	require.Contains(t, refusedAs(t, runner, ctx, updateScheduledTool), `"error":"not_asked_by_user"`)
}

func TestATaskIsNotMadeInATurnDelegatedWorkStarted(t *testing.T) {
	runner := unstoredScheduler(Spec{})
	ctx := agent.WithProvenance(context.Background(), agent.Provenance{Delegated: true})

	require.Contains(t, refusedAs(t, runner, ctx, scheduleTaskTool), `"error":"not_asked_by_user"`)
}

func TestReadingTheUsersOwnTasksFirstStillLetsOneBeChanged(t *testing.T) {
	runner := unstoredScheduler(Spec{})

	read := agent.WithProvenance(context.Background(), agent.Provenance{ToolsRead: []string{listScheduledTool}})

	require.Nil(t, runner.unasked(read))
	require.Nil(t, runner.unasked(context.Background()), "a call answering the person's own words")
}
