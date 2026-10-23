package session

import (
	"strconv"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

// says puts a line on the session's stream the way the agent saying it would.
func says(running *Session, text string) {
	running.broadcast(agent.Said{Text: text})
}

// drain reads what a watcher has been sent so far.
func drain(events <-chan Event) []Event {
	var got []Event
	for {
		select {
		case event, open := <-events:
			if !open {
				return got
			}
			got = append(got, event)
		case <-time.After(50 * time.Millisecond):
			return got
		}
	}
}

// lines are the said lines among what a resumable watcher was sent, in order.
func lines(events []Event) []string {
	var said []string
	for _, event := range events {
		if sequenced, ok := event.(Sequenced); ok {
			if line, ok := sequenced.Event.(agent.Said); ok {
				said = append(said, line.Text)
			}
		}
	}
	return said
}

// cursorOf is the cursor the said line came with.
func cursorOf(events []Event, said string) string {
	for _, event := range events {
		if sequenced, ok := event.(Sequenced); ok && sequenced.Event == (agent.Said{Text: said}) {
			return sequenced.Cursor
		}
	}
	return ""
}

func (s *SessionSuite) resumable(running *Session, cursor string) (<-chan Event, func()) {
	events, detach, err := running.WatchFrom(cursor, "", false)
	s.Require().NoError(err)
	return events, detach
}

func (s *SessionSuite) TestAResumableWatcherIsFirstToldWhereItsStreamStarts() {
	s.manages()
	running := s.writes(Spec{})
	events, detach := s.resumable(running, "")
	defer detach()

	resumed, ok := (<-events).(Resumed)
	s.Require().True(ok, "the first thing sent is where the stream starts")
	s.True(resumed.Complete, "a watcher starting out has missed nothing")
	s.NotEmpty(resumed.Cursor)
}

func (s *SessionSuite) TestAWatcherThatComesBackIsSentWhatItMissedAndNothingTwice() {
	s.manages()
	running := s.writes(Spec{})
	events, detach := s.resumable(running, "")
	says(running, "one")
	last := cursorOf(drain(events), "one")
	detach()

	says(running, "two")
	says(running, "three")
	again, detachAgain := s.resumable(running, last)
	defer detachAgain()

	got := drain(again)
	s.Require().NotEmpty(got)
	s.True(got[0].(Resumed).Complete, "everything missed is still kept")
	s.Equal([]string{"two", "three"}, lines(got))
}

func (s *SessionSuite) TestACursorFromAnotherStreamIsToldItMissedEvents() {
	s.manages()
	before := s.writes(Spec{})
	events, detach := s.resumable(before, "")
	says(before, "said before the restart")
	last := cursorOf(drain(events), "said before the restart")
	detach()

	reopened := s.writes(Spec{})
	again, detachAgain := s.resumable(reopened, last)
	defer detachAgain()

	got := drain(again)
	s.False(got[0].(Resumed).Complete, "a cursor of another stream says nothing about this one")
	s.Empty(lines(got))
}

func (s *SessionSuite) TestAWatcherGoneForMoreThanTheSessionKeepsIsToldItMissedEvents() {
	s.manages()
	running := s.writes(Spec{})
	events, detach := s.resumable(running, "")
	first := drain(events)[0].(Resumed).Cursor
	detach()

	for range replayLength + 1 {
		says(running, "line")
	}
	again, detachAgain := s.resumable(running, first)
	defer detachAgain()

	got := drain(again)
	s.False(got[0].(Resumed).Complete, "one line fell out of what the session keeps")
	s.Len(lines(got), replayLength, "what is kept is still sent")
}

func (s *SessionSuite) TestAWatcherBackAfterTheOldestWereDroppedIsSentTheRestInOrder() {
	s.manages()
	running := s.writes(Spec{})
	events, detach := s.resumable(running, "")
	for i := range replayLength {
		says(running, strconv.Itoa(i))
	}
	at := cursorOf(drain(events), strconv.Itoa(replayLength-3))
	detach()

	for i := range 3 {
		says(running, "after "+strconv.Itoa(i))
	}
	again, detachAgain := s.resumable(running, at)
	defer detachAgain()

	got := drain(again)
	s.True(got[0].(Resumed).Complete)
	s.Equal([]string{strconv.Itoa(replayLength - 2), strconv.Itoa(replayLength - 1), "after 0", "after 1", "after 2"}, lines(got))
}

func (s *SessionSuite) TestACursorThisRouterNeverHandedOutIsRefused() {
	s.manages()
	running := s.writes(Spec{})

	_, _, err := running.WatchFrom("not a cursor", "", false)

	s.ErrorIs(err, ErrBadCursor)
}

func (s *SessionSuite) TestAPlainWatcherIsSentEventsAsTheyAlwaysWere() {
	s.manages()
	running := s.writes(Spec{})
	_, detachResumable := s.resumable(running, "")
	defer detachResumable()
	events, detach := running.Watch()
	defer detach()

	says(running, "hello")

	got := drain(events)
	s.Contains(got, agent.Said{Text: "hello"})
	for _, event := range got {
		s.IsNotType(Sequenced{}, event, "a watcher that did not ask to resume is sent no cursor")
		s.IsNotType(Resumed{}, event)
	}
}

func (s *SessionSuite) TestAWatcherComingBackIsToldHowTheCommandItWaitedOnEnded() {
	s.persists()
	running := s.commands()
	_, _, err := running.RespondCommand(s.ctx, "command-a", "Question", "", nil, options.LLM{})
	s.Require().NoError(err)
	s.Equal("Question", s.asked())
	_, err = running.InterruptCommand("command-a")
	s.Require().NoError(err)

	events, detach, err := running.WatchFrom("", "command-a", false)
	s.Require().NoError(err)
	defer detach()

	resumed := (<-events).(Resumed)
	s.Require().NotNil(resumed.Command)
	s.Equal("command-a", resumed.Command.CommandID)
	s.Equal("cancelled", resumed.Command.State)
}
