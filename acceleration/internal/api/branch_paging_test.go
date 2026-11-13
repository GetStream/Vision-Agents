package api

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A branch is paged in the tree's order. Its first turn was sent again by an earlier
// activation, so its copies are newer than the turn after it, and the next page still starts
// right after the message the cursor names.
func TestABranchIsPagedInTheTreesOrderThoughAnAncestorWasSentAgain(t *testing.T) {
	at := time.Date(2026, 11, 13, 12, 0, 0, 0, time.UTC)
	shown := []store.AgentBranchMessage{
		{OriginalMessageID: "q1-again", ChatCreatedAt: at.Add(5 * time.Minute)},
		{OriginalMessageID: "a1-again", ChatCreatedAt: at.Add(6 * time.Minute)},
		{OriginalMessageID: "q2", ChatCreatedAt: at.Add(time.Minute)},
		{OriginalMessageID: "a2", ChatCreatedAt: at.Add(2 * time.Minute)},
	}

	var read []string
	var after *branchPosition
	for start := branchFrom(shown, after); start < len(shown); start = branchFrom(shown, after) {
		read = append(read, shown[start].OriginalMessageID)
		after = &branchPosition{At: shown[start].ChatCreatedAt, ID: shown[start].OriginalMessageID}
	}

	require.Equal(t, []string{"q1-again", "a1-again", "q2", "a2"}, read, "one message a page, each once")
	require.Equal(t, 0, branchFrom(shown, nil))
}
