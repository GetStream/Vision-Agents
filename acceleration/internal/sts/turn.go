package sts

import (
	"fmt"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

// Turn tracks one reply in flight.
//
// Every provider needs the same bookkeeping to report a reply honestly: when the caller
// stopped talking, when the first audio came back, how much speech there was. Audio arrives
// on the provider's read goroutine while an interrupt arrives on the caller's, so this is
// safe for both.
type Turn struct {
	// ID correlates the events belonging to this reply. It is the vendor's response id
	// where the vendor has one and a number made up here where it does not.
	ID string
	// Generation counts replies within a session from one.
	Generation int

	mu          sync.Mutex
	heardAt     time.Time
	startedAt   time.Time
	firstByteAt time.Time
	audioMs     float64
	chunks      int
}

// NewTurn starts tracking a reply. heardAt is when the caller stopped talking, and zero
// for a reply that answers nobody's wait, such as one asked for by Prompt.
func NewTurn(id string, generation int, heardAt time.Time) *Turn {
	if id == "" {
		id = fmt.Sprintf("r-%d", generation)
	}
	return &Turn{ID: id, Generation: generation, heardAt: heardAt, startedAt: time.Now()}
}

// Chunk records a piece of audio and returns the event to emit for it.
func (t *Turn) Chunk(pcm audio.PcmData) AudioChunk {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.firstByteAt.IsZero() {
		t.firstByteAt = time.Now()
	}
	t.audioMs += pcm.DurationMs()
	index := t.chunks
	t.chunks++

	return AudioChunk{ResponseID: t.ID, Generation: t.Generation, Index: index, Audio: pcm}
}

// HeardAt records when the caller stopped, for a reply the model opened before they had.
// It is ignored once the reply has begun to speak, since a wait that ended cannot start.
func (t *Turn) HeardAt(at time.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.heardAt.IsZero() && t.firstByteAt.IsZero() {
		t.heardAt = at
	}
}

// SentMs is how much of the reply has been handed on so far, which is the best a provider
// can say about where the listener was cut off when nobody told it.
func (t *Turn) SentMs() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return int(t.audioMs)
}

// Complete returns the event that settles the reply. A reply the caller was not waiting
// for reports no time to first byte, since there was nobody waiting to measure.
func (t *Turn) Complete(provider, model string, interrupted bool, usage Usage) ResponseComplete {
	t.mu.Lock()
	defer t.mu.Unlock()

	var timeToFirstByte float64
	if !t.firstByteAt.IsZero() && !t.heardAt.IsZero() {
		timeToFirstByte = float64(t.firstByteAt.Sub(t.heardAt).Microseconds()) / 1000
	}

	return ResponseComplete{
		ResponseID:        t.ID,
		Generation:        t.Generation,
		Provider:          provider,
		Model:             model,
		Interrupted:       interrupted,
		AudioDurationMs:   t.audioMs,
		TimeToFirstByteMs: timeToFirstByte,
		ResponseTimeMs:    float64(time.Since(t.startedAt).Microseconds()) / 1000,
		Usage:             usage,
	}
}
