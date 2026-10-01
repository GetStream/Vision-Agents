package conversation

import (
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/require"
)

// watcher applies one reasoning step's windows the way Athena's clients do: it appends
// what it does not have, ignores what it has and marks a gap it cannot fill. Windows of
// another step are another step's.
type watcher struct {
	id   string
	have int
	text string
}

func (w *watcher) apply(r reasoningWindow) {
	if w.id == "" {
		w.id = r.ID
	}
	if r.ID != w.id {
		return
	}
	if r.Length <= w.have {
		return
	}
	if r.Offset > w.have {
		w.text += "…"
	} else {
		r.Text = string([]rune(r.Text)[w.have-r.Offset:])
	}
	w.text += r.Text
	w.have = r.Length
}

// drain sends every pending window, as the live loop would over successive ticks.
func drain(t *testing.T, r *liveReasoning, now time.Time, w *watcher) []reasoningWindow {
	t.Helper()
	var sent []reasoningWindow
	for r.pending() {
		window, ok := r.window(now)
		require.True(t, ok)
		require.True(t, utf8.ValidString(window.Text), "a window split a character")
		require.LessOrEqual(t, len(window.Text), maxReasoningWindow)
		require.Equal(t, window.Offset+utf8.RuneCountInString(window.Text), window.Length)
		r.delivered(window, now)
		w.apply(window)
		sent = append(sent, window)
		require.Less(t, len(sent), 100, "windows stopped advancing")
	}
	return sent
}

func TestReasoningWindowsCarryOnlyNewThinking(t *testing.T) {
	r := liveReasoning{id: "r1"}
	var w watcher
	start := time.Now()
	_, ok := r.window(start)
	require.False(t, ok, "nothing thought yet")

	r.add("Weighing", start)
	first := drain(t, &r, start, &w)
	require.Equal(t, []reasoningWindow{{ID: "r1", Offset: 0, Text: "Weighing", Length: 8, key: true}}, first)

	r.add(" the options.", start.Add(1500*time.Millisecond))
	next := drain(t, &r, start.Add(1500*time.Millisecond), &w)
	require.Len(t, next, 1)
	require.Equal(t, 8, next[0].Offset)
	require.Equal(t, " the options.", next[0].Text, "an update repeated thinking already sent")
	require.Equal(t, "Weighing the options.", w.text)
	require.EqualValues(t, 1500, r.snapshot().durationMS)

	_, ok = r.window(start.Add(2 * time.Second))
	require.False(t, ok, "nothing new and no keyframe due")
}

func TestReasoningKeyframeLetsLateWatchersCatchUp(t *testing.T) {
	r := liveReasoning{id: "r1"}
	var inSync, late watcher
	start := time.Now()
	thinking := strings.Repeat("considering ", 300)
	r.add(thinking, start)
	drain(t, &r, start, &inSync)
	require.Equal(t, thinking, inSync.text)

	key, ok := r.window(start.Add(reasoningKeyframeEvery))
	require.True(t, ok, "a due keyframe rides on the next update")
	require.True(t, key.key)
	require.LessOrEqual(t, len(key.Text), reasoningKeyframe)
	require.Equal(t, r.total, key.Length)
	require.True(t, strings.HasSuffix(thinking, key.Text))
	r.delivered(key, start.Add(reasoningKeyframeEvery))

	inSync.apply(key)
	require.Equal(t, thinking, inSync.text, "a watcher in step ignores a keyframe")
	late.apply(key)
	require.Equal(t, "…"+key.Text, late.text)

	_, ok = r.window(start.Add(reasoningKeyframeEvery + time.Second))
	require.False(t, ok, "the keyframe was only due once")
}

func TestReasoningBacklogGoesOutInOrder(t *testing.T) {
	r := liveReasoning{id: "r1"}
	var w watcher
	now := time.Now()
	thinking := strings.Repeat("é🙂 naïve ", 400)
	r.add(thinking, now)
	windows := drain(t, &r, now, &w)
	require.Greater(t, len(windows), 1, "a long backlog is split")
	require.Equal(t, thinking, w.text)
}

func TestReasoningBacklogBeyondTheBufferSkipsAhead(t *testing.T) {
	r := liveReasoning{id: "r1"}
	var w watcher
	now := time.Now()
	r.add("start ", now)
	drain(t, &r, now, &w)
	r.add(strings.Repeat("x", maxReasoningBuffer+500), now)
	require.LessOrEqual(t, len(r.buf), maxReasoningBuffer)
	windows := drain(t, &r, now, &w)
	require.Greater(t, windows[0].Offset, len("start "), "an unsendable backlog is skipped, not sent")
	require.True(t, strings.HasPrefix(w.text, "start …xxx"))
	require.Equal(t, r.total, w.have)
}

func TestReasoningKeepsOnlyTheOpeningOfARound(t *testing.T) {
	r := liveReasoning{id: "r1"}
	now := time.Now()
	r.add(strings.Repeat("é", maxHead), now)
	require.LessOrEqual(t, len(r.head), maxHead)
	require.True(t, utf8.ValidString(r.head), "the opening split a character")
	other := liveReasoning{id: "r2"}
	other.add("Resumed.", now)
	old, _ := r.window(now)
	other.delivered(old, now)
	require.Zero(t, other.sent, "a window of another step does not count as delivered")
}
