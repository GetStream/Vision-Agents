package session

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

func TestAConfigsTeamTagIsItsSessionsTeam(t *testing.T) {
	spec := FromConfig(store.AgentConfig{ID: "c", Tags: map[string]string{TeamTag: "support", "env": "prod"}})

	require.Equal(t, "support", spec.Team)
	require.Equal(t, "support", spec.Tags[TeamTag], "it stays a cost label too")
}

// A tag saved before it was a team, which config writes now refuse, is a cost label alone
// rather than a reason every session of the agent fails.
func TestATeamTagThatIsNoTeamIsLeftOut(t *testing.T) {
	spec := FromConfig(store.AgentConfig{ID: "c", Tags: map[string]string{TeamTag: "customer support"}})

	require.Empty(t, spec.Team)
	require.Equal(t, "customer support", spec.Tags[TeamTag])
}
