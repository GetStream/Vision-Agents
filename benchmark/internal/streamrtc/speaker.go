//go:build cgo && webrtc

package streamrtc

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/GetStream/getstream-go-webrtc/track"
	"github.com/livekit/media-sdk"
	"github.com/livekit/media-sdk/opus"
	protoLogger "github.com/livekit/protocol/logger"
	webrtcmedia "github.com/pion/webrtc/v4/pkg/media"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
)

const (
	opusSampleRate         = 48_000
	opusFrameDuration      = 20 * time.Millisecond
	opusFrameSamples       = opusSampleRate / 50
	opusNegotiatedChannels = 2
	playoutFrames          = 20
	audioLevelSilent       = 127
	audioLevelSpeaking     = 20
)

var silenceFrame = []byte{0xf8, 0xff, 0xfe}

type speaker struct {
	track.BaseSampleProvider
	mu        sync.Mutex
	drained   *sync.Cond
	frames    [][]byte
	pipeline  media.PCM16Writer
	inputRate int
	speaking  bool
	closed    bool
}

func newSpeaker() *speaker {
	talker := &speaker{}
	talker.drained = sync.NewCond(&talker.mu)
	return talker
}

func (s *speaker) Write(pcm audio.PCM) error {
	if len(pcm.Samples) == 0 {
		return nil
	}
	if pcm.Rate <= 0 {
		return fmt.Errorf("streamrtc: audio has no sample rate")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return errors.New("streamrtc: the call has been left")
	}
	if s.pipeline == nil || s.inputRate != pcm.Rate {
		pipeline, err := s.encoderFor(pcm.Rate)
		if err != nil {
			return err
		}
		s.pipeline, s.inputRate = pipeline, pcm.Rate
	}
	if err := s.pipeline.WriteSample(media.PCM16Sample(pcm.Samples)); err != nil {
		return fmt.Errorf("streamrtc: encode speech: %w", err)
	}
	for len(s.frames) > playoutFrames && !s.closed {
		s.drained.Wait()
	}
	return nil
}

func (s *speaker) NextSample(ctx context.Context) (webrtcmedia.Sample, error) {
	if err := ctx.Err(); err != nil {
		return webrtcmedia.Sample{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.frames) == 0 {
		s.speaking = false
		return webrtcmedia.Sample{Data: silenceFrame, Duration: opusFrameDuration}, nil
	}
	frame := s.frames[0]
	s.frames = s.frames[1:]
	s.speaking = true
	s.drained.Signal()
	return webrtcmedia.Sample{Data: frame, Duration: opusFrameDuration}, nil
}

func (s *speaker) CurrentAudioLevel() uint8 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.speaking {
		return audioLevelSpeaking
	}
	return audioLevelSilent
}

func (s *speaker) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	s.frames = nil
	s.drained.Broadcast()
	if s.pipeline == nil {
		return nil
	}
	pipeline := s.pipeline
	s.pipeline = nil
	return pipeline.Close()
}

func (s *speaker) encoderFor(inputRate int) (media.PCM16Writer, error) {
	encoder, err := opus.Encode(&frameSink{speaker: s}, 1, protoLogger.GetLogger())
	if err != nil {
		return nil, fmt.Errorf("streamrtc: build opus encoder: %w", err)
	}
	return media.ResampleWriter(media.FullFrames(encoder, opusFrameSamples), inputRate), nil
}

type frameSink struct {
	speaker *speaker
}

func (f *frameSink) String() string  { return "streamrtc.speaker" }
func (f *frameSink) SampleRate() int { return opusSampleRate }
func (f *frameSink) Close() error    { return nil }

func (f *frameSink) WriteSample(frame opus.Sample) error {
	f.speaker.frames = append(f.speaker.frames, append([]byte(nil), frame...))
	return nil
}
