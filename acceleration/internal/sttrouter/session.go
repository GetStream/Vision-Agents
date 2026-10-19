package sttrouter

import (
	"sync"
	"sync/atomic"
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
	// failed is set once the provider reports a fatal error. The caller has been told, and
	// the provider hears nothing more, so later audio is dropped rather than failing once
	// per frame.
	failed atomic.Bool

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

// ProcessAudio streams audio to the selected provider, and drops it once the provider has
// failed for good.
func (s *Session) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if s.failed.Load() {
		return nil
	}
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
	case stt.Transcript:
		// Only settled transcripts are billable work; interim ones are revisions of a
		// turn that has not finished yet.
		if !typed.Final() {
			return
		}
		s.recorder.Record(s.config, routing.Stat{
			Owner:     s.owner,
			StartedAt: time.Now().UTC(),
			Usage:     routing.Usage{AudioMs: int64(typed.AudioDurationMs)},
			LatencyMs: typed.ProcessingTimeMs,
			Success:   true,
		})

	case stt.Error:
		stat := routing.Stat{
			Owner:     s.owner,
			StartedAt: time.Now().UTC(),
			Success:   false,
			ErrorCode: errorCode(typed),
			// The vendor's own words are what tell an account it refuses from a dropped
			// socket.
			ErrorMessage: typed.Err.Error(),
		}
		if typed.Fatal {
			// Set before the error is passed on, so a caller who has read it sends no audio
			// that still reaches the provider.
			s.failed.Store(true)
			// Only a refusal the vendor sent marks the model unavailable. Providers report a
			// socket that dropped as a fatal error with the context read or write, and the
			// next session can still reach the model.
			stat.Fatal = !transportDrop(typed)
		}
		s.recorder.Record(s.config, stat)
	}
}

func transportDrop(failure stt.Error) bool {
	return failure.Context == "read" || failure.Context == "write"
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
