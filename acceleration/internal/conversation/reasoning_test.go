package conversation

import (
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/stretchr/testify/suite"
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

// ReasoningSuite covers the windows a streaming reasoning step is sent in.
type ReasoningSuite struct {
	suite.Suite
	r     liveReasoning
	start time.Time
}

func TestReasoningSuite(t *testing.T) { suite.Run(t, new(ReasoningSuite)) }

func (s *ReasoningSuite) SetupTest() {
	s.r = liveReasoning{id: "r1"}
	s.start = time.Now()
}

// drain sends every pending window, as the live loop would over successive ticks.
func (s *ReasoningSuite) drain(now time.Time, w *watcher) []reasoningWindow {
	var sent []reasoningWindow
	for s.r.pending() {
		window, ok := s.r.window(now)
		s.Require().True(ok)
		s.Require().True(utf8.ValidString(window.Text), "a window split a character")
		s.Require().LessOrEqual(len(window.Text), maxReasoningWindow)
		s.Require().Equal(window.Offset+utf8.RuneCountInString(window.Text), window.Length)
		s.r.delivered(window, now)
		w.apply(window)
		sent = append(sent, window)
		s.Require().Less(len(sent), 100, "windows stopped advancing")
	}
	return sent
}

func (s *ReasoningSuite) TestWindowsCarryOnlyNewThinking() {
	var w watcher
	_, ok := s.r.window(s.start)
	s.False(ok, "nothing thought yet")

	s.r.add("Weighing", s.start)
	first := s.drain(s.start, &w)
	s.Equal([]reasoningWindow{{ID: "r1", Offset: 0, Text: "Weighing", Length: 8, key: true}}, first)

	later := s.start.Add(1500 * time.Millisecond)
	s.r.add(" the options.", later)
	next := s.drain(later, &w)
	s.Require().Len(next, 1)
	s.Equal(8, next[0].Offset)
	s.Equal(" the options.", next[0].Text, "an update repeated thinking already sent")
	s.Equal("Weighing the options.", w.text)
	s.EqualValues(1500, s.r.snapshot().durationMS)

	_, ok = s.r.window(s.start.Add(2 * time.Second))
	s.False(ok, "nothing new and no keyframe due")
}

func (s *ReasoningSuite) TestAKeyframeLetsLateWatchersCatchUp() {
	var inSync, late watcher
	thinking := strings.Repeat("considering ", 300)
	s.r.add(thinking, s.start)
	s.drain(s.start, &inSync)
	s.Equal(thinking, inSync.text)

	key, ok := s.r.window(s.start.Add(reasoningKeyframeEvery))
	s.Require().True(ok, "a due keyframe rides on the next update")
	s.True(key.key)
	s.LessOrEqual(len(key.Text), reasoningKeyframe)
	s.Equal(s.r.total, key.Length)
	s.True(strings.HasSuffix(thinking, key.Text))
	s.r.delivered(key, s.start.Add(reasoningKeyframeEvery))

	inSync.apply(key)
	s.Equal(thinking, inSync.text, "a watcher in step ignores a keyframe")
	late.apply(key)
	s.Equal("…"+key.Text, late.text)

	_, ok = s.r.window(s.start.Add(reasoningKeyframeEvery + time.Second))
	s.False(ok, "the keyframe was only due once")
}

func (s *ReasoningSuite) TestABacklogGoesOutInOrder() {
	var w watcher
	thinking := strings.Repeat("é🙂 naïve ", 400)
	s.r.add(thinking, s.start)
	windows := s.drain(s.start, &w)
	s.Greater(len(windows), 1, "a long backlog is split")
	s.Equal(thinking, w.text)
}

func (s *ReasoningSuite) TestABacklogBeyondTheBufferSkipsAhead() {
	var w watcher
	s.r.add("start ", s.start)
	s.drain(s.start, &w)
	s.r.add(strings.Repeat("x", maxReasoningBuffer+500), s.start)
	s.LessOrEqual(len(s.r.buf), maxReasoningBuffer)
	windows := s.drain(s.start, &w)
	s.Greater(windows[0].Offset, len("start "), "an unsendable backlog is skipped, not sent")
	s.True(strings.HasPrefix(w.text, "start …xxx"))
	s.Equal(s.r.total, w.have)
}

func (s *ReasoningSuite) TestOnlyTheOpeningOfARoundIsKept() {
	s.r.add(strings.Repeat("é", maxHead), s.start)
	s.LessOrEqual(len(s.r.head), maxHead)
	s.True(utf8.ValidString(s.r.head), "the opening split a character")
	other := liveReasoning{id: "r2"}
	other.add("Resumed.", s.start)
	old, _ := s.r.window(s.start)
	other.delivered(old, s.start)
	s.Zero(other.sent, "a window of another step does not count as delivered")
}
