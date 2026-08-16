package sttrouter

import (
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Session is a live transcription attached to one customer. It forwards the provider's
// events untouched and records a stat row per completed turn on the way past.
type Session struct {
	provider stt.STT
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	owner    routing.Owner
	recorder *routing.Recorder

	events chan stt.Event

	mu sync.Mutex
	// turnStartedAt is when the current turn began, so a turn's stat covers the whole
	// turn rather than the last transcript in it.
	turnStartedAt time.Time

	closeOnce sync.Once
}

func newSession(
	provider stt.STT,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder *routing.Recorder,
) *Session {
	session := &Session{
		provider: provider,
		config:   config,
		owner:    owner,
		recorder: recorder,
		events:   make(chan stt.Event, 64),
	}
	go session.forward()
	return session
}

// ProcessAudio streams audio to the selected provider.
func (s *Session) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	return s.provider.ProcessAudio(pcm, participant)
}

// Events returns the provider's events. The channel closes when the session closes.
func (s *Session) Events() <-chan stt.Event { return s.events }

// Provider is the provider serving this session.
func (s *Session) Provider() string { return s.config.Provider }

// Model is the model serving this session.
func (s *Session) Model() string { return s.config.Model }

// STT exposes the underlying provider so callers can reach provider-specific features.
func (s *Session) STT() stt.STT { return s.provider }

// Close ends the session. The event channel closes once the provider's events are drained.
func (s *Session) Close() error {
	var err error
	s.closeOnce.Do(func() { err = s.provider.Close() })
	return err
}

// forward relays provider events and records statistics as they pass.
func (s *Session) forward() {
	defer close(s.events)

	for event := range s.provider.Events() {
		s.observe(event)
		s.events <- event
	}
}

func (s *Session) observe(event stt.Event) {
	switch typed := event.(type) {
	case stt.TurnStarted:
		s.mu.Lock()
		s.turnStartedAt = time.Now()
		s.mu.Unlock()

	case stt.Transcript:
		// Only settled transcripts are billable work; interim ones are revisions of a
		// turn that has not finished yet.
		if !typed.Final() {
			return
		}
		s.recorder.Record(s.config, routing.Stat{
			Owner:     s.owner,
			StartedAt: s.turnStart(),
			Usage:     routing.Usage{AudioMs: int64(typed.AudioDurationMs)},
			LatencyMs: typed.ProcessingTimeMs,
			Success:   true,
		})

	case stt.Error:
		s.recorder.Record(s.config, routing.Stat{
			Owner:     s.owner,
			StartedAt: s.turnStart(),
			Success:   false,
			ErrorCode: errorCode(typed),
		})
	}
}

// turnStart returns the start of the current turn, consuming it so the next turn gets its
// own timestamp. It falls back to now for providers that never signal a turn start.
func (s *Session) turnStart() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()

	started := s.turnStartedAt
	s.turnStartedAt = time.Time{}
	if started.IsZero() {
		return time.Now().UTC()
	}
	return started.UTC()
}

func errorCode(failure stt.Error) string {
	if failure.Fatal {
		return "provider_fatal"
	}
	if failure.Context != "" {
		return failure.Context
	}
	return "provider_error"
}
