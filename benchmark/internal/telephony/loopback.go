package telephony

import (
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
)

// Loopback is an in-process duplex pipe used without Telnyx.
type Loopback struct {
	toAgent   chan Frame
	toCaller  chan Frame
	mu        sync.Mutex
	closed    bool
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
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	for i := 0; i < len(pcm); i += audio.FrameSamples {
		end := min(i+audio.FrameSamples, len(pcm))
		chunk := audio.PadRight(append([]int16(nil), pcm[i:end]...), audio.FrameSamples)
		select {
		case l.toAgent <- Frame{PCM: chunk}:
		default:
		}
	}
	return nil
}

func (l *Loopback) Recv() <-chan Frame { return l.toCaller }

func (l *Loopback) SendAgent(pcm []int16) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil
	}
	for i := 0; i < len(pcm); i += audio.FrameSamples {
		end := min(i+audio.FrameSamples, len(pcm))
		chunk := audio.PadRight(append([]int16(nil), pcm[i:end]...), audio.FrameSamples)
		select {
		case l.toCaller <- Frame{PCM: chunk}:
		default:
		}
		time.Sleep(20 * time.Millisecond)
	}
	return nil
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
