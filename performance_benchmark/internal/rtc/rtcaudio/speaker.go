//go:build webrtc

package rtcaudio

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	webrtcaudio "github.com/GetStream/getstream-go-webrtc/audio"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	webrtcmedia "github.com/pion/webrtc/v4/pkg/media"

	"github.com/GetStream/Vision-Agents/performance_benchmark/internal/rtc/audio"
)

const (
	// OpusSampleRate is the clock rate the encoder negotiates.
	OpusSampleRate = 48_000
	// OpusFrameDuration is the duration of one encoded frame.
	OpusFrameDuration = 20 * time.Millisecond

	playoutFrames      = 20
	audioLevelSilent   = 127
	audioLevelSpeaking = 20
)

var silenceFrame = []byte{0xf8, 0xff, 0xfe}

// Speaker encodes caller speech to opus and hands frames to the WebRTC SDK on demand.
type Speaker struct {
	mu       sync.Mutex
	drained  *sync.Cond
	frames   [][]byte
	encoder  *opus.Encoder
	speaking bool
	closed   bool
}

func NewSpeaker() *Speaker {
	talker := &Speaker{}
	talker.drained = sync.NewCond(&talker.mu)
	return talker
}

// Write encodes pcm and blocks while the playout queue is full.
func (s *Speaker) Write(pcm audio.PCM) error {
	if len(pcm.Samples) == 0 {
		return nil
	}
	if pcm.Rate <= 0 {
		return fmt.Errorf("rtcaudio: audio has no sample rate")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return errors.New("rtcaudio: the call has been left")
	}
	if s.encoder == nil {
		// The encoder resamples whatever rate it is written at, so one serves every write.
		encoder, err := opus.NewEncoder(opus.Config{
			SampleRate: OpusSampleRate, FrameDuration: OpusFrameDuration,
		})
		if err != nil {
			return fmt.Errorf("rtcaudio: build opus encoder: %w", err)
		}
		s.encoder = encoder
	}
	packets, err := s.encoder.Encode(webrtcaudio.FromInt16(pcm.Samples, pcm.Rate, 1))
	if err != nil {
		return fmt.Errorf("rtcaudio: encode speech: %w", err)
	}
	s.frames = append(s.frames, packets...)
	for len(s.frames) > playoutFrames && !s.closed {
		s.drained.Wait()
	}
	return nil
}

func (s *Speaker) NextSample(ctx context.Context) (webrtcmedia.Sample, error) {
	if err := ctx.Err(); err != nil {
		return webrtcmedia.Sample{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.frames) == 0 {
		s.speaking = false
		return webrtcmedia.Sample{Data: silenceFrame, Duration: OpusFrameDuration}, nil
	}
	frame := s.frames[0]
	s.frames = s.frames[1:]
	s.speaking = true
	s.drained.Signal()
	return webrtcmedia.Sample{Data: frame, Duration: OpusFrameDuration}, nil
}

func (s *Speaker) CurrentAudioLevel() uint8 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.speaking {
		return audioLevelSpeaking
	}
	return audioLevelSilent
}

func (s *Speaker) OnBind() error { return nil }

func (s *Speaker) OnUnbind() error { return nil }

func (s *Speaker) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	s.frames = nil
	s.encoder = nil
	s.drained.Broadcast()
	return nil
}
