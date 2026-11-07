package session

import "time"

// EndReason is why a session ended: which wait timed out, or what the agent did. It is
// named in the log line and kept on the session's row.
type EndReason string

const (
	// EndAgentHangup ends a session whose agent hung up (end_call).
	EndAgentHangup EndReason = "agent_hangup"
	// EndTransferred is a session whose agent handed the caller to a human.
	EndTransferred EndReason = "transferred"
	// TimeoutHangup ends a phone session after the caller has left.
	TimeoutHangup EndReason = "hangup"
	// TimeoutStart ends a phone session nobody joined in time.
	TimeoutStart EndReason = "start"
	// TimeoutUnanswered ends a placed call's session once its vendor says nobody will answer.
	TimeoutUnanswered EndReason = "unanswered"
	// EndVoicemail ends a placed call an answering machine picked up, once the message is left.
	EndVoicemail EndReason = "voicemail"
)

// SetTimeout ends the session d from now. Calling it again moves the moment and replaces
// the reason.
func (s *Session) SetTimeout(d time.Duration, reason EndReason) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closing {
		return
	}
	s.stopTimeout()
	armed := s.timeoutArmed
	s.timeout = time.AfterFunc(d, func() { s.timedOut(armed, reason) })
}

// ClearTimeout removes the timeout: the session runs until it is stopped.
func (s *Session) ClearTimeout() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closing {
		return
	}
	s.stopTimeout()
}

// stopTimeout stops the timer and disowns it, since Stop does not hold back one already
// firing. Called holding s.mu.
func (s *Session) stopTimeout() {
	if s.timeout != nil {
		s.timeout.Stop()
		s.timeout = nil
	}
	s.timeoutArmed++
}

// timedOut ends the session, unless the timer that fired was moved, cleared or stopped by
// the session closing in the meantime. It waits out a voice start or stop under way, which
// may clear the timeout. The ending is called holding no lock: it closes the session, which
// takes commandMu and s.mu.
func (s *Session) timedOut(armed uint64, reason EndReason) {
	s.commandMu.Lock()
	s.mu.Lock()
	if s.closing || armed != s.timeoutArmed {
		s.mu.Unlock()
		s.commandMu.Unlock()
		return
	}
	s.timeout = nil
	end := s.end
	s.mu.Unlock()
	s.commandMu.Unlock()
	if end != nil {
		end(reason)
	}
}

// noteEnd records why the session is ending. The first reason wins: a transferred call
// still times out once the caller leaves, and that is not why it ended.
func (s *Session) noteEnd(reason EndReason) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ended == "" {
		s.ended = reason
	}
}

// endsFor notes reason as why the session ends, reporting whether it did: not once the
// session is closing or another reason was noted.
func (s *Session) endsFor(reason EndReason) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closing || s.ended != "" {
		return false
	}
	s.ended = reason
	return true
}

// closingNow reports whether the session has begun to close.
func (s *Session) closingNow() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closing
}

// EndReason is why the session ended, empty while nothing has said.
func (s *Session) EndReason() EndReason {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ended
}
