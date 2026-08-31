package streamedge

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/getstream-go-webrtc/track"
	"github.com/livekit/media-sdk"
	"github.com/livekit/media-sdk/opus"
	protoLogger "github.com/livekit/protocol/logger"
	webrtcmedia "github.com/pion/webrtc/v4/pkg/media"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

const (
	// opusSampleRate is the only rate WebRTC carries Opus at.
	opusSampleRate = 48_000
	// opusFrameDuration is the frame size every Opus encoder and decoder handles.
	opusFrameDuration = 20 * time.Millisecond
	// opusFrameSamples is one frame's worth of mono samples.
	opusFrameSamples = opusSampleRate / 50
	// opusNegotiatedChannels is the channel count the published track has to declare. Opus
	// is always offered as two-channel in SDP, and pion refuses to bind a track whose
	// channel count does not match what was negotiated. What goes out is still mono: the
	// payload is opaque to WebRTC.
	opusNegotiatedChannels = 2
	// playoutFrames is how much speech may wait to go out, in frames, and so how far
	// behind the queue the participants can be. A voice provider streams an utterance far
	// faster than it is spoken, so without a bound the whole reply would be queued in a
	// moment and barge-in would arrive too late to stop it. 400 ms is deep enough to
	// absorb a provider's jitter and short enough that stopping still sounds immediate.
	playoutFrames = 20
	// audioLevelSilent and audioLevelSpeaking are the WebRTC audio-level scale, where 0 is
	// loudest and 127 is silence.
	audioLevelSilent   = 127
	audioLevelSpeaking = 20
)

// silenceFrame is the canonical Opus silence packet. The track is published for the whole
// call, so something has to go out while the agent has nothing to say.
var silenceFrame = []byte{0xf8, 0xff, 0xfe}

// speaker is the agent's voice on the call: PCM in, 20 ms Opus frames out.
//
// track.Local paces its reads off each sample's duration, so NextSample must return
// promptly rather than wait for audio to arrive.
type speaker struct {
	track.BaseSampleProvider

	logger *slog.Logger

	mu sync.Mutex
	// drained wakes a writer once the queue is back under the playout bound.
	drained *sync.Cond
	// pulled is set once the track has asked for its first frame. Until it does, nothing
	// written here is going anywhere.
	pulled bool
	// frames are encoded and waiting to be sent, oldest first.
	frames [][]byte
	// pipeline resamples, frames and encodes PCM written at inputRate. It is rebuilt when
	// a provider changes rate, which is what happens when routing fails over mid-call.
	pipeline  media.PCM16Writer
	inputRate int
	// unflushed is set while the pipeline may still be holding the tail of an utterance,
	// so it is only pushed out once rather than on every quiet frame.
	unflushed bool
	speaking  bool
	closed    bool
}

func newSpeaker(logger *slog.Logger) *speaker {
	if logger == nil {
		logger = slog.Default()
	}
	talker := &speaker{logger: logger}
	talker.drained = sync.NewCond(&talker.mu)
	return talker
}

// Write queues a chunk of the agent's speech, blocking while the queue is full so the agent
// publishes at the rate the audio is heard rather than as fast as it is synthesised.
func (s *speaker) Write(pcm audio.PcmData) error {
	if len(pcm.Samples) == 0 {
		return nil
	}
	if pcm.Channels != 1 {
		return fmt.Errorf("streamedge: audio must be mono, got %d channels", pcm.Channels)
	}
	if pcm.SampleRate <= 0 {
		return fmt.Errorf("streamedge: audio has no sample rate")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return errors.New("streamedge: the call has been left")
	}
	if s.pipeline == nil || s.inputRate != pcm.SampleRate {
		pipeline, err := s.encoderFor(pcm.SampleRate)
		if err != nil {
			return err
		}
		s.pipeline, s.inputRate = pipeline, pcm.SampleRate
	}

	// The sink at the end of the pipeline appends to frames directly: it is only ever
	// reached from here, under this lock.
	if err := s.pipeline.WriteSample(media.PCM16Sample(pcm.Samples)); err != nil {
		return fmt.Errorf("streamedge: encode speech: %w", err)
	}
	s.unflushed = true

	if len(s.frames) <= playoutFrames || s.closed {
		return nil
	}
	// Waiting here is normal: it is what paces the agent to the speed of speech. Waiting
	// a long time is not, and means the track is not taking frames, in which case this
	// speech is sitting in the queue rather than going out.
	waited := time.Now()
	for len(s.frames) > playoutFrames && !s.closed {
		s.drained.Wait()
	}
	if elapsed := time.Since(waited); elapsed > time.Second {
		s.logger.Warn("speech waited to go out, the call was not taking audio",
			"waited", elapsed, "pulled", s.pulled)
	}
	return nil
}

// NextSample hands the track one frame, or silence when the agent has nothing to say.
func (s *speaker) NextSample(ctx context.Context) (webrtcmedia.Sample, error) {
	if err := ctx.Err(); err != nil {
		return webrtcmedia.Sample{}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.pulled {
		s.pulled = true
		s.logger.Debug("the call started taking the agent's audio")
	}
	if len(s.frames) == 0 {
		s.flush()
	}
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

// flush pushes the end of an utterance out of the pipeline. The caller must hold the lock.
//
// The resampler runs a chunk behind: what was written last is held until more audio comes
// along to take its place. That is fine mid-sentence, but at the end of a reply nothing
// more is coming, and without this the last chunk of every utterance stays in the pipeline
// and the caller hears the agent stop short of its final word.
//
// Silence is what displaces it, which costs nothing: it is what the track sends anyway
// once the queue is empty.
func (s *speaker) flush() {
	if s.pipeline == nil || !s.unflushed {
		return
	}
	s.unflushed = false

	quiet := make([]int16, s.inputRate/50)
	if err := s.pipeline.WriteSample(media.PCM16Sample(quiet)); err != nil {
		s.logger.Debug("could not flush the end of an utterance", "error", err)
	}
}

// CurrentAudioLevel is what the SDK puts in the audio-level RTP header extension, which is
// how the other participants' clients know the agent is the one talking.
func (s *speaker) CurrentAudioLevel() uint8 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.speaking {
		return audioLevelSpeaking
	}
	return audioLevelSilent
}

// Close releases the encoder and lets go of anyone waiting to publish.
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

// encoderFor builds the pipeline for one input rate: resample to 48 kHz, cut into whole
// frames because an Opus encoder only accepts those, then encode.
func (s *speaker) encoderFor(inputRate int) (media.PCM16Writer, error) {
	encoder, err := opus.Encode(&frameSink{speaker: s}, 1, protoLogger.GetLogger())
	if err != nil {
		return nil, fmt.Errorf("streamedge: build opus encoder: %w", err)
	}
	return media.ResampleWriter(media.FullFrames(encoder, opusFrameSamples), inputRate), nil
}

// frameSink is the end of the pipeline: it queues each encoded frame for the track.
type frameSink struct {
	speaker *speaker
}

func (f *frameSink) String() string { return "streamedge.speaker" }

func (f *frameSink) SampleRate() int { return opusSampleRate }

func (f *frameSink) WriteSample(frame opus.Sample) error {
	// Write holds the lock while the pipeline runs, so the queue is appended to directly.
	f.speaker.frames = append(f.speaker.frames, append([]byte(nil), frame...))
	return nil
}

func (f *frameSink) Close() error { return nil }
