package conversation

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/stretchr/testify/require"
	"testing"
	"unicode/utf8"
)

func TestPublicProgressBoundarySurvivesHistory(t *testing.T) {
	_, client := newChat(t)
	s, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	t.Cleanup(s.Close)
	c, _, _, err := s.Open(t.Context(), "customer", "agent", "")
	require.NoError(t, err)
	require.NoError(t, c.Begin("Check and answer"))
	c.Observe(agent.Responding{})
	c.Observe(agent.ResponseDelta{Text: "Checking 🌱."})
	c.Observe(agent.ToolStarted{ID: "tool-1", Tool: "athena_resource_metadata"})
	c.Observe(agent.Responded{PendingWork: true})
	c.Observe(agent.ToolRan{ID: "tool-1", Tool: "athena_resource_metadata", Result: `{}`})
	c.Observe(agent.Responding{})
	c.Observe(agent.ResponseDelta{Text: "The final answer.\n\nSecond paragraph."})
	c.Observe(agent.Responded{})
	saved(t, c)
	page, err := s.History(t.Context(), "customer", "agent", c.CID(), "")
	require.NoError(t, err)
	require.Len(t, page.Messages, 2)
	m := page.Messages[1]
	require.Equal(t, 1, m.TextLayout)
	require.Equal(t, utf8.RuneCountInString("Checking 🌱.\n\n"), m.AnswerStart)
	require.Equal(t, "The final answer.\n\nSecond paragraph.", string([]rune(m.Text)[m.AnswerStart:]))
	require.Contains(t, m.Text, "Checking 🌱.")
}
