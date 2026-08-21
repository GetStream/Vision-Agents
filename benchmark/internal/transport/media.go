package transport

import (
	"context"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
)

// Frame is one 20 ms chunk from the far end.
type Frame struct {
	PCM []int16
}

// Media is a bidirectional PCM pipe at audio.Rate.
type Media interface {
	Send(pcm []int16) error
	Recv() <-chan Frame
	// WaitForAgent blocks until the agent is in the call and publishing audio.
	WaitForAgent(ctx context.Context) error
	// Dropped is how many inbound frames were discarded because the consumer was behind. A
	// dropped frame becomes zero-padded silence on the agent leg and can move a measured
	// onset, so a run has to be able to say this was zero.
	Dropped() int
	Close() error
}

// Chunk appends samples to pending and sends every whole audio.FrameSamples frame
// to out, dropping frames when out is full. It returns the leftover samples and how many
// frames it had to drop.
func Chunk(pending []int16, samples []int16, out chan<- Frame) ([]int16, int) {
	dropped := 0
	pending = append(pending, samples...)
	for len(pending) >= audio.FrameSamples {
		frame := Frame{PCM: append([]int16(nil), pending[:audio.FrameSamples]...)}
		pending = append(pending[:0], pending[audio.FrameSamples:]...)
		select {
		case out <- frame:
		default:
			dropped++
		}
	}
	return pending, dropped
}
