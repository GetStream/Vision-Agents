package session

import "time"

// EndReason is why a session ended: which wait timed out, what the agent did, or what failed.
// It is named in the log line, kept on the session's row and counted in
// router_session_ends_total. Every value is in EndReasons, which says whose doing it was and
// at which stage. A value, once stored, keeps its name.
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
	// EndClosed is a session closed with no other reason: by its caller (Manager.Close), by the
	// agent's next session in the same call (supersede), by a reopened conversation nobody was
	// watching (takeOver), or by a chat left unwatched past its grace.
	EndClosed EndReason = "closed"
	// EndShutdown is a session the router ended as it went away (Manager.Shutdown).
	EndShutdown EndReason = "shutdown"
	// EndCreateFailed is a session Create made and then gave up on before it was held, for a
	// failure of the router's own, such as a call that refused the agent for a reason other
	// than EndCreateNoVideo's.
	EndCreateFailed EndReason = "create_failed"
	// EndCreateVendorFailed is a session Create gave up on because no vendor would start a model
	// it needs.
	EndCreateVendorFailed EndReason = "create_vendor_failed"
	// EndCreateConfigRefused is a session Create gave up on because a vendor refused a setting it
	// was asked for, or the customer's policies allow none of its models.
	EndCreateConfigRefused EndReason = "create_config_refused"
	// EndCreateNoCredit is a session Create gave up on because the customer's credit or spend
	// limits refused the work.
	EndCreateNoCredit EndReason = "create_no_credit"
	// EndCreateNoVideo is a session Create gave up on because its call refused the agent: the
	// Stream server the customer's app is on runs no video (agent.ErrNoVideo).
	EndCreateNoVideo EndReason = "create_no_video"
	// EndAgentFailed is a session whose agent left the call by itself, or answered no turn, for
	// a failure of the router's own, such as its playback queue filling.
	EndAgentFailed EndReason = "agent_failed"
	// EndModelFailed is a session whose model vendor failed it: its speech-to-speech session
	// ended and could not be replaced, or its model failed a turn and answered none. The agent
	// keeps the conversation going after a failed turn (agent.Error), so one turn lost on a call
	// that was answered is no failure of the session. Like every failure of the agent or its
	// model, it is counted when the session closes, whatever ends it then.
	EndModelFailed EndReason = "model_failed"
	// EndModelConfigRefused is EndModelFailed on the customer's side: the last model to fail ran
	// on the customer's own key or own model, or refused a setting it was asked for.
	EndModelConfigRefused EndReason = "model_config_refused"
	// EndModelNoCredit is a session whose model's work the customer's credit or spend limits
	// refused, so the agent answered no turn or left the call.
	EndModelNoCredit EndReason = "model_no_credit"
)

// Fault is whose doing a session's end was.
type Fault string

const (
	// FaultNone is an end nobody is to blame for: the agent or the customer's app ended it.
	FaultNone Fault = "none"
	// FaultCaller is the person on the call: the caller, or whoever a placed call rang.
	FaultCaller Fault = "caller"
	// FaultRouter is the router itself, or Stream's call it runs on.
	FaultRouter Fault = "router"
	// FaultVendor is a model vendor.
	FaultVendor Fault = "vendor"
	// FaultCarrier is the phone carrier that bridges a call into Stream's.
	FaultCarrier Fault = "carrier"
	// FaultCustomerConfig is the customer's configuration or key.
	FaultCustomerConfig Fault = "customer_config"
	// FaultCustomerCredit is the customer's credit or spend limits.
	FaultCustomerCredit Fault = "customer_credit"
)

// Stage is how far a session had got when it ended.
type Stage string

const (
	// StageStarting is before the session was held: Create gave up on it.
	StageStarting Stage = "starting"
	// StageRinging is before a person was on the line of a phone call.
	StageRinging Stage = "ringing"
	// StageConversation is once the session was held.
	StageConversation Stage = "conversation"
)

// EndInfo is what an EndReason means: whose doing it was, the stage it came at, and whether
// it counts as a failure of the session in router_session_ends_total.
type EndInfo struct {
	Reason  EndReason
	Meaning string
	Fault   Fault
	Stage   Stage
	Failed  bool
}

// EndReasons is every EndReason, the one list the metric, the API and the tests read. A failure
// is a session that never got going, whose agent dropped out, or whose model answered nothing;
// a callee who did not answer or a router shutting down is no failure of the session.
var EndReasons = []EndInfo{
	{EndAgentHangup, "the agent hung up", FaultNone, StageConversation, false},
	{EndTransferred, "the agent handed the caller to a person", FaultNone, StageConversation, false},
	{TimeoutHangup, "the caller left", FaultCaller, StageConversation, false},
	{TimeoutStart, "nobody joined the phone call in time", FaultCarrier, StageRinging, true},
	{TimeoutUnanswered, "the call placed was never answered", FaultCaller, StageRinging, false},
	{EndVoicemail, "an answering machine picked up the call placed, and the agent left its message or hung up",
		FaultNone, StageRinging, false},
	{EndClosed, "the session was closed through the API, or replaced by a newer one", FaultNone, StageConversation, false},
	{EndShutdown, "the router ended the session as it shut down", FaultRouter, StageConversation, false},
	{EndCreateFailed, "the session could not be opened for a failure of the router's own", FaultRouter, StageStarting, true},
	{EndCreateVendorFailed, "no vendor would start a model the session needs", FaultVendor, StageStarting, true},
	{EndCreateConfigRefused, "a setting of the session was refused, or no model it names is allowed",
		FaultCustomerConfig, StageStarting, true},
	{EndCreateNoCredit, "the credit or spend limits refused the session's work", FaultCustomerCredit, StageStarting, true},
	{EndCreateNoVideo, "the call refused the agent: the app's Stream server runs no video", FaultCustomerConfig,
		StageStarting, true},
	{EndAgentFailed, "the agent left the call or answered no turn for a failure of the router's own",
		FaultRouter, StageConversation, true},
	{EndModelFailed, "a model vendor failed: the agent left the call or answered no turn", FaultVendor, StageConversation, true},
	{EndModelConfigRefused, "the model failed on the customer's own key or model, or refused a setting, and the " +
		"agent answered no turn", FaultCustomerConfig, StageConversation, true},
	{EndModelNoCredit, "the credit or spend limits refused the model's work, and the agent answered no turn",
		FaultCustomerCredit, StageConversation, true},
}

// Info is what reason means, and false for a value outside EndReasons.
func (r EndReason) Info() (EndInfo, bool) {
	for _, info := range EndReasons {
		if info.Reason == r {
			return info, true
		}
	}
	return EndInfo{}, false
}

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
