//go:build webrtc

package rtcaudio

import (
	"fmt"
	"sync"

	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	"github.com/pion/webrtc/v4"

	"github.com/GetStream/Vision-Agents/performance_benchmark/internal/rtc/audio"
	"github.com/GetStream/Vision-Agents/performance_benchmark/internal/rtc/transport"
)

// Inbound decodes remote opus tracks into 20 ms PCM frames the caller can read.
type Inbound struct {
	recv chan transport.Frame

	mu        sync.Mutex
	listening map[string]chan struct{}
	pending   []int16
	dropped   int
	closed    bool
}

func NewInbound() *Inbound {
	return &Inbound{
		recv:      make(chan transport.Frame, 64),
		listening: map[string]chan struct{}{},
	}
}

// Track decodes one remote audio track, replacing any earlier decoder for the same track.
func (i *Inbound) Track(track *webrtc.TrackRemote) error {
	reader, err := audiortc.NewTrackReader(track,
		audiortc.ReaderConfig{Opus: opus.Config{SampleRate: audio.Rate}})
	if err != nil {
		return fmt.Errorf("rtcaudio: decode remote audio: %w", err)
	}
	stop := make(chan struct{})
	i.mu.Lock()
	if i.closed {
		i.mu.Unlock()
		return nil
	}
	if previous, ok := i.listening[track.ID()]; ok {
		close(previous)
	}
	i.listening[track.ID()] = stop
	i.mu.Unlock()
	go i.hear(reader, stop)
	return nil
}

func (i *Inbound) Recv() <-chan transport.Frame { return i.recv }

// Dropped is how many decoded frames were discarded because the consumer was behind.
func (i *Inbound) Dropped() int {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.dropped
}

// Closed reports whether the pipe has been shut down.
func (i *Inbound) Closed() bool {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.closed
}

func (i *Inbound) Close() error {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed {
		return nil
	}
	i.closed = true
	for _, stop := range i.listening {
		close(stop)
	}
	i.listening = nil
	close(i.recv)
	return nil
}

// hear decodes one track until it ends or stop is closed. A read blocks until the next
// packet, so a stopped track is only let go once its next packet arrives or the call ends.
func (i *Inbound) hear(reader *audiortc.TrackReader, stop <-chan struct{}) {
	defer reader.Close()
	for pcm, err := range reader.Frames() {
		if err != nil {
			return
		}
		select {
		case <-stop:
			return
		default:
		}
		i.push(pcm.ToInt16().Int16())
	}
}

func (i *Inbound) push(samples []int16) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.closed {
		return
	}
	var dropped int
	i.pending, dropped = transport.Chunk(i.pending, samples, i.recv)
	i.dropped += dropped
}
