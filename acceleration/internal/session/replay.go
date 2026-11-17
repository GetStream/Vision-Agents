package session

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"strconv"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
)

// replayLength is how many events a session keeps for a resumable watcher that drops and
// comes back. It bounds what each session holds, whatever it says: a watcher that missed
// more than this is told so rather than sent all of it.
const replayLength = 256

// ErrBadCursor is a cursor this router never handed out.
var ErrBadCursor = errors.New("session: the cursor is not one this session handed out")

// Sequenced is an event as a resumable watcher is sent it, with its place in the
// session's stream. Cursor is what the watcher reconnects from.
type Sequenced struct {
	Cursor string
	Event  Event
}

// Resumed is the first thing a resumable watcher is sent. Cursor is where its stream
// carries on from. Complete says every event after the cursor it reconnected from is in
// what follows; false means some are not, because the router restarted, the session was
// reopened, or the watcher was gone for more than replayLength events, and the caller
// reads what it missed back from the conversation instead.
type Resumed struct {
	Cursor   string
	Complete bool
	// Command is the receipt of the request the watcher asked after, nil when it named
	// none or the conversation knows of none.
	Command *persistent.CommandReceipt
}

// replay numbers a session's events and keeps the latest replayLength of them, oldest at
// start once it is full. Every event numbered is kept, so the kept ones are the numbers up
// to last. epoch names this session object, so a cursor from before a restart or a reopen
// is never mistaken for one of this stream's.
type replay struct {
	epoch string
	last  uint64
	kept  []Sequenced
	start int
}

func newReplay() *replay {
	id := make([]byte, 8)
	_, _ = rand.Read(id)
	return &replay{epoch: hex.EncodeToString(id)}
}

func (r *replay) cursor(seq uint64) string {
	return r.epoch + "." + strconv.FormatUint(seq, 10)
}

// add numbers an event and keeps it in place of the oldest once replayLength are kept.
func (r *replay) add(event Event) Sequenced {
	r.last++
	sequenced := Sequenced{Cursor: r.cursor(r.last), Event: event}
	if len(r.kept) < replayLength {
		r.kept = append(r.kept, sequenced)
	} else {
		r.kept[r.start] = sequenced
		r.start = (r.start + 1) % replayLength
	}
	return sequenced
}

// after returns what is kept past cursor, oldest first, and whether that is everything
// past it. An empty cursor is a watcher starting out, which has missed nothing.
func (r *replay) after(cursor string) ([]Sequenced, bool) {
	if cursor == "" {
		return nil, true
	}
	epoch, seq, _ := parseCursor(cursor)
	if epoch != r.epoch || seq > r.last {
		return nil, false
	}
	missed := min(r.last-seq, uint64(len(r.kept)))
	replayed := make([]Sequenced, 0, missed)
	for i := uint64(len(r.kept)) - missed; i < uint64(len(r.kept)); i++ {
		replayed = append(replayed, r.kept[(r.start+int(i))%len(r.kept)])
	}
	return replayed, missed == r.last-seq
}

// parseCursor splits a cursor into the stream it is of and the event it is at.
func parseCursor(cursor string) (string, uint64, error) {
	epoch, number, ok := strings.Cut(cursor, ".")
	if !ok || epoch == "" || len(epoch) > 32 {
		return "", 0, ErrBadCursor
	}
	seq, err := strconv.ParseUint(number, 10, 64)
	if err != nil {
		return "", 0, ErrBadCursor
	}
	return epoch, seq, nil
}

// CheckCursor refuses a cursor that is not of the form a session hands out.
func CheckCursor(cursor string) error {
	_, _, err := parseCursor(cursor)
	return err
}

// sequences says whether an event takes a place in the stream. Interim transcripts arrive
// several times a second and are superseded by the line they become, so they are neither
// numbered nor kept.
func sequences(event Event) bool {
	_, interim := event.(agent.Hearing)
	return !interim
}

// WatchFrom attaches a resumable watcher in seat, as WatchAs does: every event it is sent carries a cursor, and it
// is first sent a Resumed, then what it missed after cursor while the session kept it.
// requestID names a command whose receipt the Resumed carries, for a caller that was
// waiting on its answer when it dropped.
//
// The caller must already be authorized for this session, as for Watch.
func (s *Session) WatchFrom(seat Seat, cursor, requestID string) (<-chan Event, func(), error) {
	if cursor != "" {
		if err := CheckCursor(cursor); err != nil {
			return nil, nil, err
		}
	}
	resumed := &Resumed{}
	if requestID != "" && s.persisted != nil {
		if receipt, err := s.Command(requestID); err == nil {
			resumed.Command = &receipt
		}
	}
	events, detach := s.watch(seat, cursor, resumed)
	return events, detach, nil
}

// sequence numbers an event for the resumable watchers and returns its cursor, reporting
// false when there is none to number it for. The caller holds s.mu.
func (s *Session) sequence(event Event) (string, bool) {
	if s.replay == nil || !sequences(event) {
		return "", false
	}
	return s.replay.add(event).Cursor, true
}

// form is the event as this watcher is sent it: event is already addressed to its seat.
func (w *watcher) form(event Event, cursor string, numbered bool) Event {
	if w.sequenced && numbered {
		return Sequenced{Cursor: cursor, Event: event}
	}
	return event
}
