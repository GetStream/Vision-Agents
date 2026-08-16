package tts

import (
	"fmt"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

// Synthesis tracks one utterance in flight.
//
// Every provider needs the same bookkeeping to report a synthesis honestly: when the
// request went out, when the first audio came back, how many characters and how much
// speech. Text arrives on the caller's goroutine while audio arrives on the provider's
// read goroutine, so this is safe for both.
type Synthesis struct {
	// ID correlates the events belonging to this utterance.
	ID string

	mu          sync.Mutex
	startedAt   time.Time
	firstByteAt time.Time
	characters  int64
	audioMs     float64
	chunks      int
}

// NewSynthesis starts tracking an utterance, generating an ID when the caller has none.
func NewSynthesis(id string) *Synthesis {
	if id == "" {
		id = fmt.Sprintf("s-%d", time.Now().UnixNano())
	}
	return &Synthesis{ID: id, startedAt: time.Now()}
}

// AddText counts text sent upstream. A streaming provider calls it once per delta.
func (s *Synthesis) AddText(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.characters += int64(len([]rune(text)))
}

// Chunk records a piece of audio and returns the event to emit for it.
func (s *Synthesis) Chunk(pcm audio.PcmData) AudioChunk {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.firstByteAt.IsZero() {
		s.firstByteAt = time.Now()
	}
	s.audioMs += pcm.DurationMs()
	index := s.chunks
	s.chunks++

	return AudioChunk{SynthesisID: s.ID, Index: index, Audio: pcm}
}

// Complete returns the event that settles the utterance. An utterance that produced no
// audio reports a zero time to first byte rather than the whole synthesis time.
func (s *Synthesis) Complete(provider, model string, interrupted bool) SynthesisComplete {
	s.mu.Lock()
	defer s.mu.Unlock()

	var timeToFirstByte float64
	if !s.firstByteAt.IsZero() {
		timeToFirstByte = float64(s.firstByteAt.Sub(s.startedAt).Microseconds()) / 1000
	}

	return SynthesisComplete{
		SynthesisID:       s.ID,
		Provider:          provider,
		Model:             model,
		Characters:        s.characters,
		AudioDurationMs:   s.audioMs,
		TimeToFirstByteMs: timeToFirstByte,
		SynthesisTimeMs:   float64(time.Since(s.startedAt).Microseconds()) / 1000,
		Interrupted:       interrupted,
	}
}
