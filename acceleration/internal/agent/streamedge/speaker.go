package streamedge

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	webrtcaudio "github.com/GetStream/getstream-go-webrtc/audio"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	"github.com/GetStream/getstream-go-webrtc/track"
	webrtcmedia "github.com/pion/webrtc/v4/pkg/media"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

const (
	// opusSampleRate is the only rate WebRTC carries Opus at.
	opusSampleRate = 48_000
	// opusFrameDuration is the frame size every Opus encoder and decoder handles.
	opusFrameDuration = 20 * time.Millisecond
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
	// encoder resamples, frames and encodes PCM at whatever rate it is written. A provider
	// changes rate when routing fails over mid-call, and the encoder follows it.
	encoder *opus.Encoder
	// unflushed is set while the encoder may still be holding the tail of an utterance,
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
	if s.encoder == nil {
		encoder, err := opus.NewEncoder(opus.Config{SampleRate: opusSampleRate, FrameDuration: opusFrameDuration})
		if err != nil {
			return fmt.Errorf("streamedge: build opus encoder: %w", err)
		}
		s.encoder = encoder
	}

	packets, err := s.encoder.Encode(webrtcaudio.FromInt16(pcm.Samples, pcm.SampleRate, 1))
	if err != nil {
		return fmt.Errorf("streamedge: encode speech: %w", err)
	}
	s.frames = append(s.frames, packets...)
	s.unflushed = true

	if len(s.frames) <= playoutFrames || s.closed {
		return nil
	}
	// Waiting here is normal: it is what paces the agent to the speed of speech, so a chunk
	// that is seconds long is seconds spent here. Waiting much longer than the queue was deep
	// is not, and means the track has stopped taking frames, in which case this speech is
	// sitting in the queue rather than going out.
	waited := time.Now()
	queued := len(s.frames)
	for len(s.frames) > playoutFrames && !s.closed {
		s.drained.Wait()
	}
	elapsed := time.Since(waited)
	if elapsed > time.Duration(queued)*opusFrameDuration+time.Second {
		s.logger.Warn("speech waited to go out, the call was not taking audio",
			"waited", elapsed, "queued", queued, "pulled", s.pulled)
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

// flush pushes the end of an utterance out of the encoder. The caller must hold the lock.
//
// The encoder holds on to what it cannot fill a frame with, waiting for more audio to take
// its place. That is fine mid-sentence, but at the end of a reply nothing more is coming,
// and without this the tail of every utterance stays in the encoder and the caller hears
// the agent stop short of its final word.
func (s *speaker) flush() {
	if s.encoder == nil || !s.unflushed {
		return
	}
	s.unflushed = false

	packets, err := s.encoder.Flush()
	if err != nil {
		s.logger.Debug("could not flush the end of an utterance", "error", err)
		return
	}
	s.frames = append(s.frames, packets...)
}

// drop throws away speech that has been published but not heard yet, so barge-in stops the
// agent within a frame rather than at the end of what is already queued.
//
// What the encoder is holding is discarded along with the rest: a tail left inside belongs
// to the reply being abandoned, and would otherwise be the first thing heard at the start
// of the next one.
func (s *speaker) drop() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.encoder != nil {
		s.encoder.Reset()
	}
	s.unflushed = false
	s.frames = nil
	s.speaking = false
	s.drained.Broadcast()
}

// pending reports whether any of the speech written here is still waiting to go out,
// whether queued as frames or held inside the encoder.
//
// A voice provider streams an utterance far faster than it is spoken, so when it reports an
// utterance finished it has only sent the last of it: up to playoutFrames of the reply is
// still queued here, and closing on that word would throw the tail away.
//
// A track that has never asked for a frame has nothing draining it, so what it holds is
// reported as settled rather than leaving a caller waiting for a queue that cannot move.
func (s *speaker) pending() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || !s.pulled {
		return false
	}
	return len(s.frames) > 0 || s.unflushed
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
	s.encoder = nil
	return nil
}
