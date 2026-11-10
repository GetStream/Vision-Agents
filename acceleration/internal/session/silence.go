package session

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

// The kinds router_silent_sessions_total counts a session under: phone for one created
// with a phone or that a phone caller joined (phoneSession), web for any other.
const (
	kindPhone = "phone"
	kindWeb   = "web"
)

// case is an agent.Silence, each defined in agent/silence.go.
var silentSessions = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "silent_sessions_total",
	Help:      "Voice sessions ended with audio missing one way, once per case they are in, by case (no_inbound_audio, silent_inbound_audio or no_words for the caller's side, no_audible_outbound for the agent's; one with no audio either way is in two) and kind, phone or web; only sessions somebody besides the agent joined are judged",
}, []string{"case", "kind"})

// A session counts here once, at the same moment and under the same kind as in silentSessions,
// whether or not it is in a case, so a case's count divided by this one is at most 1.
var judgedSessions = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "voice_sessions_judged_total",
	Help:      "Voice sessions ended that were judged for silence, once each, by kind, phone or web: sessions somebody besides the agent joined, the ones router_silent_sessions_total counts cases of, so a case's count divided by this one is at most 1",
}, []string{"kind"})

// The series exist from the start, so a rate over the first silent session reads it.
func init() {
	judgedSessions.WithLabelValues(kindPhone)
	judgedSessions.WithLabelValues(kindWeb)
	for _, silence := range agent.Silences {
		silentSessions.WithLabelValues(string(silence), kindPhone)
		silentSessions.WithLabelValues(string(silence), kindWeb)
	}
}

// heard adds what one agent run's call carried to the session's. Called once its events have
// closed; runs overlap while voice starts or stops, so the sum is locked.
func (s *Session) heard(held *agent.Agent) {
	carried := held.CallAudio()
	s.carriedMu.Lock()
	defer s.carriedMu.Unlock()
	s.carried = s.carried.Merge(carried)
}

// countSilences counts the session as judged and under each case its runs put it in, if
// somebody besides the agent joined. Called once, from counted, after every run has been
// added.
func (s *Session) countSilences() {
	kind := kindWeb
	if s.phoneSession(s.Spec()) {
		kind = kindPhone
	}
	s.carriedMu.Lock()
	judged := s.carried.Judged()
	silences := s.carried.Silences()
	s.carriedMu.Unlock()
	if !judged {
		return
	}
	judgedSessions.WithLabelValues(kind).Inc()
	for _, silence := range silences {
		silentSessions.WithLabelValues(string(silence), kind).Inc()
	}
}
