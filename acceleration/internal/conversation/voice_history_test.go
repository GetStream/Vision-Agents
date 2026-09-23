package conversation

import (
	"encoding/json"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/require"
)

func TestVoiceHistoryRestoresOnlySettledSpeechFromTheExpectedAuthor(t *testing.T) {
	for _, test := range []struct {
		name, author, source, role string
		generating, interrupted    bool
		accepted                   bool
	}{
		{name: "assistant", author: "media-agent", source: "agent", role: "assistant", accepted: true},
		{name: "caller", author: "alice", source: "speech", role: "user", accepted: true},
		{name: "forged assistant", author: "alice", source: "agent"},
		{name: "other agent", author: "other-agent", source: "agent"},
		{name: "agent posing as caller", author: "media-agent", source: "speech"},
		{name: "unfinished", author: "media-agent", source: "agent", generating: true},
		{name: "interrupted", author: "media-agent", source: "agent", interrupted: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			raw, err := json.Marshal(map[string]any{
				"id": "voice-message", "text": "Hello", "created_at": time.Date(2026, 9, 22, 12, 0, 0, 0, time.UTC).UnixNano(),
				"user":   map[string]any{"id": test.author},
				"custom": map[string]any{"source": test.source, "generating": test.generating, "interrupted": test.interrupted},
			})
			require.NoError(t, err)
			var wire getstream.MessageResponse
			require.NoError(t, json.Unmarshal(raw, &wire))
			message, ok := messageFromVoice(wire, "media-agent")
			require.Equal(t, test.accepted, ok)
			if ok {
				require.Equal(t, test.role, message.Role)
				require.Equal(t, "Hello", message.Text)
				require.Equal(t, test.author, message.authorID)
			}
		})
	}
}
