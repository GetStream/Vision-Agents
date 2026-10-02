// Package socketedge is a call carried over a websocket: the caller's PCM comes in, and the
// agent's speech goes back out at the rate it would have been heard on a call.
//
// It exists so a conversation can be held by something that is not a browser or a phone,
// such as a benchmark's simulated caller, while everything between hearing and answering
// stays the agent's own: the cadence, the flow controller, the speculation and barge-in.
// The transport is the only thing that changes.
package socketedge

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// chunkDuration is how much speech goes out per frame, which is what a call carries per
// packet.
const chunkDuration = 20 * time.Millisecond

// playoutAhead is how much of the agent's speech may be queued before publishing waits.
// A voice provider streams far faster than speech is spoken, and an edge that took a whole
// reply at once would have nothing left to drop when the caller cuts in.
const playoutAhead = 400 * time.Millisecond

// ErrLeft is what hearing more audio returns once the call is over.
var ErrLeft = errors.New("socketedge: the call has been left")

// Options configures an Edge.
type Options struct {
	// SampleRate is the rate the socket carries in both directions. Zero means 16 kHz.
	SampleRate int
	// Caller is who the inbound audio is from.
	Caller stt.Participant
	// Send writes one frame of the agent's speech, as PCM16 mono at SampleRate. It is called
	// from a single goroutine, at the pace the speech is heard.
	Send func(pcm []byte) error
	// Cleared, when set, is told that speech already sent was thrown away, so a client
	// holding a buffer of its own can drop it too.
	Cleared func()
}

// Edge is the call.
type Edge struct {
	rate    int
	caller  stt.Participant
	send    func([]byte) error
	cleared func()
	inbound chan agent.InboundAudio

	mu sync.Mutex
	// queue is speech published and not sent yet, at the socket's rate.
	queue []int16
	// drained is closed and replaced whenever the queue shrinks, which is what a publisher
	// waiting for room waits on.
	drained chan struct{}
	left    bool

	stop     chan struct{}
	stopOnce sync.Once
	done     chan struct{}
}

// New returns an edge that has not been joined.
func New(options Options) *Edge {
	if options.SampleRate <= 0 {
		options.SampleRate = stt.SampleRate
	}
	return &Edge{
		rate:    options.SampleRate,
		caller:  options.Caller,
		send:    options.Send,
		cleared: options.Cleared,
		inbound: make(chan agent.InboundAudio, 64),
		drained: make(chan struct{}),
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
	}
}

// Join starts sending the agent's speech.
func (e *Edge) Join(context.Context) error {
	go e.playout()
	return nil
}

func (e *Edge) Audio() <-chan agent.InboundAudio { return e.inbound }

// PublishAudio queues the agent's speech, and waits while more than playoutAhead of it is
// already waiting to be heard.
func (e *Edge) PublishAudio(pcm audio.PcmData) error {
	samples := audio.Resample(pcm, e.rate, 1).Samples
	ahead := e.rate * int(playoutAhead/time.Millisecond) / 1000

	e.mu.Lock()
	if e.left {
		e.mu.Unlock()
		return ErrLeft
	}
	e.queue = append(e.queue, samples...)
	for len(e.queue) > ahead {
		drained := e.drained
		e.mu.Unlock()
		select {
		case <-drained:
		case <-e.stop:
			return ErrLeft
		}
		e.mu.Lock()
	}
	e.mu.Unlock()
	return nil
}

// SpeechPending reports whether speech published is still waiting to be sent.
func (e *Edge) SpeechPending() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.queue) > 0
}

// DropSpeech throws away speech not sent yet, so a caller who cuts in stops the agent within
// a frame.
func (e *Edge) DropSpeech() {
	e.mu.Lock()
	dropped := len(e.queue) > 0
	e.queue = nil
	e.signalLocked()
	e.mu.Unlock()
	if dropped && e.cleared != nil {
		e.cleared()
	}
}

// Hear takes one frame of the caller's audio, PCM16 mono at the socket's rate.
func (e *Edge) Hear(raw []byte) error {
	pcm := audio.Resample(audio.FromBytes(raw, e.rate, 1), stt.SampleRate, 1)
	e.mu.Lock()
	left := e.left
	e.mu.Unlock()
	if left {
		return ErrLeft
	}
	select {
	case e.inbound <- agent.InboundAudio{Participant: e.caller, Audio: pcm}:
		return nil
	case <-e.stop:
		return ErrLeft
	}
}

// Leave hangs up. It waits for the last frame to go out, so nothing is sent after it
// returns, and closes the audio channel, which is what tells the agent the call is over.
func (e *Edge) Leave() error {
	e.mu.Lock()
	if e.left {
		e.mu.Unlock()
		return nil
	}
	e.left = true
	e.mu.Unlock()

	e.stopOnce.Do(func() { close(e.stop) })
	<-e.done
	close(e.inbound)
	return nil
}

// playout sends a frame of queued speech every chunkDuration, which is the pace a listener
// hears it at.
func (e *Edge) playout() {
	defer close(e.done)
	ticker := time.NewTicker(chunkDuration)
	defer ticker.Stop()
	size := e.rate * int(chunkDuration/time.Millisecond) / 1000

	for {
		select {
		case <-e.stop:
			return
		case <-ticker.C:
		}
		e.mu.Lock()
		if len(e.queue) == 0 {
			e.mu.Unlock()
			continue
		}
		n := min(size, len(e.queue))
		frame := audio.PcmData{Samples: e.queue[:n:n], SampleRate: e.rate, Channels: 1}
		e.queue = e.queue[n:]
		e.signalLocked()
		e.mu.Unlock()

		if err := e.send(frame.Bytes()); err != nil {
			e.stopOnce.Do(func() { close(e.stop) })
			return
		}
	}
}

func (e *Edge) signalLocked() {
	close(e.drained)
	e.drained = make(chan struct{})
}
