package agent

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const echoReply = "Tomorrow in Lisbon it will be sunny and warm, with a high of twenty four degrees."

// The agent's words at the end of a transcript are its echo, as the model heard them: what
// is before them is the caller's, and an echo alone leaves nothing.
func TestTheAgentsOwnWordsAtTheEndAreItsEcho(t *testing.T) {
	for _, c := range []struct {
		heard, left string
		echo        int
	}{
		{"Will it rain in Porto this weekend? Tomorrow in Lisbon it will be sunny and warm.", "Will it rain in Porto this weekend?", 9},
		{"Tomorrow in Lisbon, it will be sunny and warm, with a high of.", "", 13},
		{"with a high of 24 degrees.", "", 6},
		{"Uh huh. sunny and warm", "Uh huh.", 3},
		{"Will it rain in Porto this weekend?", "Will it rain in Porto this weekend?", 0},
		{"and warm", "and warm", 0},
		{"Is it warm in Porto?", "Is it warm in Porto?", 0},
	} {
		left, echo := echoed(c.heard, echoReply)
		require.Equal(t, c.left, left, "%q", c.heard)
		require.Equal(t, c.echo, echo, "%q", c.heard)
	}
}

// Only chorus-asr's words are weighed, and only those heard while the agent spoke or just
// after: a caller who says the agent's words long after it stopped said them.
func TestOnlyWordsHeardOverTheAgentAreItsEcho(t *testing.T) {
	a := &Agent{history: []llm.Message{{Role: llm.Assistant, Content: echoReply}}}
	heard := stt.Transcript{Provider: audiofloor.ProviderName, Text: "Tomorrow in Lisbon it will be sunny", AudioDurationMs: 1500, ProcessingTimeMs: 300}

	_, echo := a.withoutEcho(heard)
	require.Zero(t, echo, "the agent has said nothing on this call")

	a.spokeFrom, a.lastSpokeAt = time.Now().Add(-3*time.Second), time.Now().Add(-500*time.Millisecond)
	left, echo := a.withoutEcho(heard)
	require.Equal(t, 7, echo)
	require.Empty(t, left)

	other := heard
	other.Provider = "deepgram"
	_, echo = a.withoutEcho(other)
	require.Zero(t, echo, "another transcriber's words are left alone")

	a.spokeFrom, a.lastSpokeAt = time.Now().Add(-20*time.Second), time.Now().Add(-10*time.Second)
	_, echo = a.withoutEcho(heard)
	require.Zero(t, echo, "long after the agent stopped")
}
