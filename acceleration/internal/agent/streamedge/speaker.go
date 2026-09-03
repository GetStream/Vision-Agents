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
	// flushFrames is how much silence ends an utterance, in frames. Two stages hold on to
	// what they cannot fill a frame with -- the resampler, and the framing in front of the
	// encoder -- so it takes a frame each to displace what they are holding.
	flushFrames = 2
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

// flush pushes the end of an utterance out of the pipeline. The caller must hold the lock.
//
// Each stage of the pipeline holds on to what it cannot fill a frame with, waiting for more
// audio to take its place. That is fine mid-sentence, but at the end of a reply nothing more
// is coming, and without this the tail of every utterance stays in the pipeline and the
// caller hears the agent stop short of its final word.
//
// Silence is what displaces it, which costs nothing: it is what the track sends anyway once
// the queue is empty.
func (s *speaker) flush() {
	if s.pipeline == nil || !s.unflushed {
		return
	}
	s.unflushed = false

	quiet := media.PCM16Sample(make([]int16, s.inputRate/50))
	for range flushFrames {
		if err := s.pipeline.WriteSample(quiet); err != nil {
			s.logger.Debug("could not flush the end of an utterance", "error", err)
			return
		}
	}
}

// drop throws away speech that has been published but not heard yet, so barge-in stops the
// agent within a frame rather than at the end of what is already queued.
//
// The pipeline is flushed first and what comes out of it discarded along with the rest: a
// tail left inside belongs to the reply being abandoned, and would otherwise be the first
// thing heard at the start of the next one.
func (s *speaker) drop() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.flush()
	s.frames = nil
	s.speaking = false
	s.drained.Broadcast()
}

// pending reports whether any of the speech written here is still waiting to go out,
// whether queued as frames or held inside the pipeline.
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

	if s.pipeline == nil {
		return nil
	}
	pipeline := s.pipeline
	s.pipeline = nil
	return pipeline.Close()
}

// encoderFor builds the pipeline for one input rate: cut the input into frames, resample to
// 48 kHz, cut into whole frames because an Opus encoder only accepts those, then encode.
func (s *speaker) encoderFor(inputRate int) (media.PCM16Writer, error) {
	encoder, err := opus.Encode(&frameSink{speaker: s}, 1, protoLogger.GetLogger())
	if err != nil {
		return nil, fmt.Errorf("streamedge: build opus encoder: %w", err)
	}
	resampled := media.ResampleWriter(media.FullFrames(wholeFrames{encoder}, opusFrameSamples), inputRate)
	// A rate too low to fill a frame is not speech, but it must not be a pipeline that
	// cannot make progress either.
	return frameChunks{PCM16Writer: resampled, frame: max(inputRate/50, 1)}, nil
}

// frameChunks cuts every write down to one frame.
//
// The resampler behind it keeps back whatever is left over once it has emitted every whole
// frame it can, and it sizes that frame off the largest write it has ever been given. Handed
// a voice provider's chunks whole, it keeps back as much as the biggest of them -- a few
// hundred milliseconds -- and the end of a reply stays inside it until the next reply arrives
// to push it out, which the caller hears as the agent stopping a few words short and then
// saying the missing words at the start of its next sentence. A frame at a time bounds what
// it can hold to what flush is able to displace.
//
// It wraps the pipeline rather than being done at each call site because it is a property of
// how the pipeline was built, not a rule every writer has to remember.
type frameChunks struct {
	media.PCM16Writer
	frame int
}

func (f frameChunks) WriteSample(in media.PCM16Sample) error {
	for len(in) > 0 {
		size := min(f.frame, len(in))
		if err := f.PCM16Writer.WriteSample(in[:size]); err != nil {
			return err
		}
		in = in[size:]
	}
	return nil
}

// wholeFrames drops what is not a whole frame, which is only ever the residue the framing in
// front of the encoder force-flushes when it is closed. That residue is not audio anybody is
// going to hear, since closing throws the queue away, and an Opus encoder rejects a part
// frame -- so passing it on would end most calls reporting a failure that did not happen.
type wholeFrames struct {
	media.PCM16Writer
}

func (w wholeFrames) WriteSample(in media.PCM16Sample) error {
	if len(in) != opusFrameSamples {
		return nil
	}
	return w.PCM16Writer.WriteSample(in)
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
