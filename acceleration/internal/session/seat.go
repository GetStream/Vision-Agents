package session

import (
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
)

// Seat is who a watcher is and which of the session's tools it runs.
type Seat struct {
	Owner Owner
	// Tools is what it said it runs, when Declares. A watcher that said nothing runs what
	// the session's creator declared when it is on the creator's side, as before watchers
	// could say.
	Tools    []string
	Declares bool
	// ReplayPendingVoiceTools asks for the calls of a call still waiting on an answer.
	ReplayPendingVoiceTools bool
}

// seatRuns reports whether a watcher in seat is one to run a call of name.
//
// A tool a backend declared runs with the backend's credentials, so only a backend runs it,
// whatever a browser says it holds. A name nobody declared, such as the video frames a
// phone hands over, goes to every watcher that did not say otherwise.
func (s *Session) seatRuns(seat Seat, name string) bool {
	declared := slices.ContainsFunc(s.spec.Tools, func(tool harness.Tool) bool { return tool.Name == name })
	if declared && s.spec.CallerKind == auth.KindServer && seat.Owner.Kind != auth.KindServer {
		return false
	}
	if seat.Declares {
		return slices.Contains(seat.Tools, name)
	}
	if !declared {
		return true
	}
	return seat.Owner.Kind == s.spec.CallerKind &&
		(seat.Owner.Kind == auth.KindServer || seat.Owner.UserID == s.spec.Caller.UserID)
}

// Answers reports whether a watcher in seat may answer the call id. A call this session is
// not waiting on is left to the resolve, which drops it.
func (s *Session) Answers(seat Seat, id string) bool {
	name, waiting := s.tools.nameOf(id)
	return !waiting || s.seatRuns(seat, name)
}
