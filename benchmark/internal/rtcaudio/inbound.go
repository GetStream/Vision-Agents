//go:build cgo && webrtc

package rtcaudio

import (
	"fmt"
	"sync"

	"github.com/livekit/media-sdk"
	lkmedia "github.com/livekit/server-sdk-go/v2/pkg/media"
	"github.com/pion/webrtc/v4"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

// Inbound decodes remote opus tracks into 20 ms PCM frames the caller can read. Both the
// Stream and the LiveKit transport share it; only how they discover tracks differs.
type Inbound struct {
	recv chan transport.Frame

	mu       sync.Mutex
	decoders map[string]*lkmedia.PCMRemoteTrack
	pending  []int16
	dropped  int
	closed   bool
}

func NewInbound() *Inbound {
	return &Inbound{
		recv:     make(chan transport.Frame, 64),
		decoders: map[string]*lkmedia.PCMRemoteTrack{},
	}
}

// Track decodes one remote audio track, replacing any earlier decoder for the same track.
func (i *Inbound) Track(track *webrtc.TrackRemote) error {
	decoder, err := lkmedia.NewPCMRemoteTrack(track, &listener{inbound: i},
		lkmedia.WithTargetSampleRate(audio.Rate),
		lkmedia.WithTargetChannels(1),
	)
	if err != nil {
		return fmt.Errorf("rtcaudio: decode remote audio: %w", err)
	}
	i.mu.Lock()
	if i.closed {
		i.mu.Unlock()
		decoder.Close()
		return nil
	}
	if previous, ok := i.decoders[track.ID()]; ok {
		previous.Close()
	}
	i.decoders[track.ID()] = decoder
	i.mu.Unlock()
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
	if i.closed {
		i.mu.Unlock()
		return nil
	}
	i.closed = true
	decoders := make([]*lkmedia.PCMRemoteTrack, 0, len(i.decoders))
	for _, decoder := range i.decoders {
		decoders = append(decoders, decoder)
	}
	i.decoders = map[string]*lkmedia.PCMRemoteTrack{}
	i.mu.Unlock()

	close(i.recv)
	for _, decoder := range decoders {
		decoder.Close()
	}
	return nil
}

func (i *Inbound) push(sample media.PCM16Sample) error {
	i.mu.Lock()
	if i.closed {
		i.mu.Unlock()
		return nil
	}
	var dropped int
	i.pending, dropped = transport.Chunk(i.pending, sample, i.recv)
	i.dropped += dropped
	i.mu.Unlock()
	return nil
}

type listener struct {
	inbound *Inbound
}

func (l *listener) WriteSample(sample media.PCM16Sample) error { return l.inbound.push(sample) }

func (l *listener) Close() error { return nil }
