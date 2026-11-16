package api

import (
	"encoding/json"
	"maps"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

func TestAPeerIsGivenLongEnoughToRingAndHoldATasksCall(t *testing.T) {
	// The ring, the task sent while a reply is waited out, and the call held
	// to its limit: 90 s + 10 min + 10 min.
	require.Equal(t, 21*time.Minute+30*time.Second, scheduledPeerWindow())
	require.Greater(t, scheduledPeerWindow(), scheduledPhoneWindow+scheduledCallLimit+time.Minute,
		"a call answered at the last ring and held to its limit is not cut by the asker")
}

// A peer handed an occurrence rebuilds its delivery with taskDelivery from the task's row, so
// whatever the row holds reaches the call it may place from there, never from the payload.
func TestADeliveryCarriesWhatItsTasksRowHolds(t *testing.T) {
	due := time.Date(2026, 10, 20, 9, 0, 0, 0, time.UTC)
	task := store.ScheduledTask{
		ID: "task", CustomerID: "customer", SessionID: "session", ConversationID: "conversation",
		ConfigID: "config", UserID: "user", CallerKind: "authenticated", APIKey: "key",
		UsageCategories: []string{"project:reminders"}, Timezone: "Europe/Amsterdam",
		Channels: []string{"sms", "phone"},
	}

	delivery := taskDelivery(task, "run", "the prompt", due)

	require.Equal(t, scheduledDelivery{
		CustomerID: "customer", TaskID: "task", RunID: "run", SessionID: "session",
		ConversationID: "conversation", ConfigID: "config", UserID: "user", CallerKind: "authenticated",
		Timezone: "Europe/Amsterdam", CommandID: "st_run", Text: "the prompt",
		Channels: []string{"sms", "phone"}, Due: due,
		APIKey: "key", UsageCategories: []string{"project:reminders"},
	}, delivery)
}

// A peer reads only the ids it checks of what it is handed: any other field would be the
// asker's word on the conversation, zone, command, prompt or due time it delivers under. An
// older router hands over the whole delivery, and its ids still read.
func TestAHandOffReadsOnlyTheIdsAPeerChecks(t *testing.T) {
	sent, err := json.Marshal(scheduledDelivery{
		CustomerID: "customer", TaskID: "task", RunID: "run", SessionID: "session",
		ConversationID: "conversation", ConfigID: "config", UserID: "user", CallerKind: "authenticated",
		Timezone: "Europe/Amsterdam", CommandID: "st_run", Text: "the prompt", Channels: []string{"phone"},
		Due: time.Now(), APIKey: "key", UsageCategories: []string{"project:reminders"},
	})
	require.NoError(t, err)

	var handOff scheduledHandOff
	require.NoError(t, json.Unmarshal(sent, &handOff))

	require.Equal(t, scheduledHandOff{
		CustomerID: "customer", TaskID: "task", RunID: "run", SessionID: "session",
		UserID: "user", CallerKind: "authenticated",
	}, handOff)
	read, err := json.Marshal(handOff)
	require.NoError(t, err)
	var fields map[string]any
	require.NoError(t, json.Unmarshal(read, &fields))
	require.ElementsMatch(t, []string{"customer_id", "task_id", "run_id", "session_id", "user_id", "caller_kind"},
		slices.Collect(maps.Keys(fields)))
}
