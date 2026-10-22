package session

import "time"

// TimeoutReason is which wait a timeout is, named in the log line when it ends a session.
type TimeoutReason string

const (
	// TimeoutHangup ends a phone session after the caller has left.
	TimeoutHangup TimeoutReason = "hangup"
	// TimeoutStart ends a phone session nobody joined in time.
	TimeoutStart TimeoutReason = "start"
)

// SetTimeout ends the session d from now. Calling it again moves the moment and replaces
// the reason.
func (s *Session) SetTimeout(d time.Duration, reason TimeoutReason) {
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
func (s *Session) timedOut(armed uint64, reason TimeoutReason) {
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
