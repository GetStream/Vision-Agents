package transport

import (
	"context"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
)

// Loopback is an in-process duplex pipe used in tests.
type Loopback struct {
	toAgent   chan Frame
	toCaller  chan Frame
	mu        sync.Mutex
	closed    bool
	dropped   int
	AgentRecv <-chan Frame
}

// NewLoopback returns a pair of Media ends. Caller uses the Loopback itself;
// the fake agent reads AgentRecv and writes with SendAgent.
func NewLoopback() *Loopback {
	toAgent := make(chan Frame, 64)
	toCaller := make(chan Frame, 64)
	return &Loopback{toAgent: toAgent, toCaller: toCaller, AgentRecv: toAgent}
}

func (l *Loopback) Send(pcm []int16) error {
	return l.push(l.toAgent, pcm, false)
}

func (l *Loopback) Recv() <-chan Frame { return l.toCaller }

// Dropped counts frames the caller leg was too slow to take.
func (l *Loopback) Dropped() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.dropped
}

// WaitForAgent returns immediately: the fake agent is in the pipe from the start.
func (l *Loopback) WaitForAgent(context.Context) error { return nil }

func (l *Loopback) SendAgent(pcm []int16) error {
	return l.push(l.toCaller, pcm, true)
}

func (l *Loopback) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	l.closed = true
	close(l.toAgent)
	close(l.toCaller)
	return nil
}

func (l *Loopback) push(dst chan Frame, pcm []int16, pace bool) error {
	for i := 0; i < len(pcm); i += audio.FrameSamples {
		end := min(i+audio.FrameSamples, len(pcm))
		chunk := audio.PadRight(append([]int16(nil), pcm[i:end]...), audio.FrameSamples)
		l.mu.Lock()
		closed := l.closed
		if !closed {
			select {
			case dst <- Frame{PCM: chunk}:
			default:
				if pace {
					l.dropped++
				}
			}
		}
		l.mu.Unlock()
		if closed {
			return nil
		}
		if pace {
			time.Sleep(20 * time.Millisecond)
		}
	}
	return nil
}
