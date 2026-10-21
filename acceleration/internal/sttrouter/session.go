package sttrouter

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Session is a live transcription attached to one customer. It forwards the provider's
// events untouched and records a stat row per completed turn on the way past. A vendor
// that refuses the session part way through hands it to the next candidate.
type Session struct {
	// mu guards provider, config and closed: a failover replaces the provider while the
	// caller is still sending audio and may close the session.
	mu       sync.Mutex
	provider stt.STT
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	closed   bool
	owner    routing.Owner
	recorder *routing.Recorder
	// reopen starts the next candidate, none of those named in refused. Nil leaves a
	// refusal the end of the session.
	reopen func(refused []string) (stt.STT, routing.ProviderConfig, error)
	// cancel ends the context reopen starts the next candidate under. Nil when there is no
	// such context.
	cancel context.CancelFunc

	events chan stt.Event
	// failed is set once the provider reports a fatal error. The caller has been told, and
	// the provider hears nothing more, so later audio is dropped rather than failing once
	// per frame.
	failed atomic.Bool
	// readAhead asks forward to read what the provider sent before a write to it failed,
	// and closes the channel it is sent once it has. done closes when forward returns.
	readAhead chan chan struct{}
	done      chan struct{}

	closeOnce sync.Once
}

func newSession(
	provider stt.STT,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder *routing.Recorder,
	reopen func(refused []string) (stt.STT, routing.ProviderConfig, error),
	cancel context.CancelFunc,
) *Session {
	session := &Session{
		provider: provider,
		config:   config,
		owner:    owner,
		recorder: recorder,
		reopen:   reopen,
		cancel:   cancel,
		events:   make(chan stt.Event, 64),
		// Unbuffered, so only a forward still running takes a request, and it answers it.
		readAhead: make(chan chan struct{}),
		done:      make(chan struct{}),
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
	provider := s.STT()
	err := provider.ProcessAudio(pcm, participant)
	if err != nil && s.toldOf(provider) {
		// The provider failed while this frame was on its way to it. The caller hears of
		// that failure, or of the next model taking over, once, and not again for the frame.
		return nil
	}
	return err
}

// toldOf reports whether the caller hears of provider's failure, or of the next model
// taking over, from the session's events. The vendor's refusal can reach the provider just
// before a write fails, or wait behind events the caller has not read yet, so forward is
// first asked to read that far.
func (s *Session) toldOf(provider stt.STT) bool {
	told := func() bool { return s.failed.Load() || provider != s.STT() }
	// Asked first so a write that fails while the next model dials does not wait for it.
	if told() {
		return true
	}
	read := make(chan struct{})
	select {
	case s.readAhead <- read:
		<-read
	case <-s.done:
	}
	return told()
}

// Events returns the provider's events. The channel closes when the session closes.
func (s *Session) Events() <-chan stt.Event { return s.events }

// Provider is the provider serving this session.
func (s *Session) Provider() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.config.Provider
}

// Model is the model serving this session.
func (s *Session) Model() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.config.Model
}

// STT exposes the underlying provider so callers can reach provider-specific features.
func (s *Session) STT() stt.STT {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.provider
}

// Close ends the session, and the start of a next candidate still under way. The event
// channel closes once the provider's events are drained.
func (s *Session) Close() error {
	var err error
	s.closeOnce.Do(func() {
		s.mu.Lock()
		s.closed = true
		provider := s.provider
		s.mu.Unlock()
		err = provider.Close()
		// After the provider has closed: a provider started under this context may hold its
		// socket by it, and closing may still wait on the vendor for the last words.
		if s.cancel != nil {
			s.cancel()
		}
	})
	return err
}

// forward relays provider events and records statistics as they pass. A refusal the next
// candidate takes over from is not passed on: the caller hears of the Failover instead.
// When no candidate takes over, the caller hears the refusal, as if none had been asked.
func (s *Session) forward() {
	defer close(s.done)
	defer close(s.events)

	current := &relay{provider: s.provider, config: s.config}
	var refused []string
	for {
		event, open := s.next(current)
		if !open {
			return
		}
		if failure, ok := event.(stt.Error); ok && failure.Fatal && !transportDrop(failure) && s.reopen != nil {
			from := current.config.Name()
			refused = append(refused, from)
			if next, nextConfig, err := s.reopen(refused); err == nil && s.swap(current.provider, next, nextConfig) {
				// Before the Failover is passed on: a write that fails meanwhile went to next.
				current = &relay{provider: next, config: nextConfig}
				s.pass(current, stt.Failover{Provider: nextConfig.Provider, Model: nextConfig.Model, From: from, Err: failure.Err})
				continue
			}
		}
		s.pass(current, event)
	}
}

// relay is the provider forward reads, and what it has read of it ahead of the caller.
type relay struct {
	provider stt.STT
	config   routing.ProviderConfig
	// ahead is what was read to answer a failed write and not passed on yet. It stops at a
	// fatal error, so nothing the provider sends after one is read.
	ahead []stt.Event
}

// next returns the provider's next event, observed, and reads ahead for a failed write
// while it waits.
func (s *Session) next(current *relay) (stt.Event, bool) {
	for {
		if len(current.ahead) > 0 {
			event := current.ahead[0]
			current.ahead = current.ahead[1:]
			return event, true
		}
		select {
		case event, open := <-current.provider.Events():
			if !open {
				return nil, false
			}
			s.observe(current.config, event)
			return event, true
		case read := <-s.readAhead:
			s.readAheadOf(current)
			close(read)
		}
	}
}

// pass hands the caller an event, and reads ahead for a failed write while the caller has
// not taken it.
func (s *Session) pass(current *relay, event stt.Event) {
	for {
		select {
		case s.events <- event:
			return
		case read := <-s.readAhead:
			s.readAheadOf(current)
			close(read)
		}
	}
}

// readAheadOf observes what the provider has sent and forward has not read, up to a fatal
// error, which marks the session failed.
func (s *Session) readAheadOf(current *relay) {
	for !s.failed.Load() {
		select {
		case event, open := <-current.provider.Events():
			if !open {
				return
			}
			s.observe(current.config, event)
			current.ahead = append(current.ahead, event)
		default:
			return
		}
	}
}

// swap puts next in place of the provider that refused, and closes that one. A session
// closed while next was starting has closed the refusing provider already, so next is
// closed instead and the swap does not happen.
func (s *Session) swap(refusing, next stt.STT, config routing.ProviderConfig) bool {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		_ = next.Close()
		return false
	}
	s.provider, s.config = next, config
	s.mu.Unlock()
	s.failed.Store(false)

	// Off this goroutine, since closing may wait on the vendor for the last words of the
	// call: ElevenLabs's Close flushes for up to FlushTimeout (elevenlabs.go, flush).
	go func() { _ = refusing.Close() }()
	return true
}

func (s *Session) observe(config routing.ProviderConfig, event stt.Event) {
	switch typed := event.(type) {
	case stt.Transcript:
		// Only settled transcripts are billable work; interim ones are revisions of a
		// turn that has not finished yet.
		if !typed.Final() {
			return
		}
		s.recorder.Record(config, routing.Stat{
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
		s.recorder.Record(config, stat)
	}
}

// transportDrop is a read or write that failed without the vendor refusing the session. A
// close frame whose code says the vendor refused it, such as Muse's 1008 "Ingress below
// real-time", is not a drop. Any other close, and any network error, is: the next session
// can still reach the model.
//
// The refusal codes are RFC 6455 section 7.4.1 (IANA WebSocket Close Code registry): 1002
// protocol error, 1003 unsupported data, 1007 invalid payload, 1008 policy violation, 1009
// message too big, 1010 mandatory extension. 4000-4999 is private use, so the vendor
// defines it; the only one documented by our vendors is Together's 4503 no_healthy_workers
// (docs.together.ai/docs/inference/transcription/streaming.md:103, read 2026-10-20), a
// refusal. The codes left out are drops: 1005 and 1006 (no frame), 1011 to 1014 (Muse's
// "Max session duration reached" and "Rate limited", dev.meta.ai/docs/api-reference/voice/
// realtime.md:119-120) and AssemblyAI's 3006 inactivity and 3008 3-hour limit, which are
// the session ending, not the model refusing.
func transportDrop(failure stt.Error) bool {
	if failure.Context != "read" && failure.Context != "write" {
		return false
	}
	var closed *websocket.CloseError
	if !errors.As(failure.Err, &closed) {
		return true
	}
	switch code := closed.Code; {
	case code == websocket.CloseProtocolError, code == websocket.CloseUnsupportedData,
		code >= websocket.CloseInvalidFramePayloadData && code <= websocket.CloseMandatoryExtension,
		code >= 4000 && code <= 4999:
		return false
	}
	return true
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
