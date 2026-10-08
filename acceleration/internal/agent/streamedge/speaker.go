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

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
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
	// audibleSample is the level a sample has to reach, in either direction, for the piece
	// holding it to count as speech: about -48 dBFS, above what a voice leaves in its pauses.
	audibleSample = 128
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
	// now is the clock the playout moments are read from.
	now func() time.Time

	mu sync.Mutex
	// drained wakes a writer once the queue is back under the playout bound.
	drained *sync.Cond
	// pulled is set once the track has asked for its first frame. Until it does, nothing
	// written here is going anywhere.
	pulled bool
	// frames are encoded and waiting to be sent, oldest first.
	frames [][]byte
	// head counts the frames that have left the queue, taken by the track or dropped, so a
	// frame keeps its number, head plus its place in the queue, as the ones ahead of it go.
	head uint64
	// armed are the replies waiting for the track to take the first frame of theirs that is
	// not silence, oldest first.
	armed []armedMark
	// encoder resamples, frames and encodes PCM at whatever rate it is written. A provider
	// changes rate when routing fails over mid-call, and the encoder follows it.
	encoder *opus.Encoder
	// unflushed is set while the encoder may still be holding the tail of an utterance,
	// so it is only pushed out once rather than on every quiet frame.
	unflushed bool
	speaking  bool
	closed    bool
}

// armedMark is a reply waiting on the track, with the number of the frame to report it at. The
// frame may not exist yet: the encoder holds back what does not fill a frame, so the first
// audible one can be the next to come out of it.
type armedMark struct {
	marks agent.PlayoutMarks
	seq   uint64
}

func newSpeaker(logger *slog.Logger) *speaker {
	if logger == nil {
		logger = slog.Default()
	}
	talker := &speaker{logger: logger, now: time.Now}
	talker.drained = sync.NewCond(&talker.mu)
	return talker
}

// Write queues a chunk of the agent's speech, blocking while the queue is full so the agent
// publishes at the rate the audio is heard rather than as fast as it is synthesised.
func (s *speaker) Write(pcm audio.PcmData) error {
	return s.WriteMarked(pcm, nil)
}

// WriteMarked is Write that also tells marks when the first frame of the chunk was queued and
// when the track took the first one that was not silence. Write returns once the queue is
// back under its bound, which for a chunk longer than the queue is as the track drains all
// but the end of it, so neither moment can be read from the return.
//
// Speech dropped before the track reached it is forgotten along with the report owed for it,
// so abandoned speech is never counted as the start of the reply that follows.
func (s *speaker) WriteMarked(pcm audio.PcmData, marks agent.PlayoutMarks) error {
	if len(pcm.Samples) == 0 {
		return nil
	}
	if pcm.Channels != 1 {
		return fmt.Errorf("streamedge: audio must be mono, got %d channels", pcm.Channels)
	}
	if pcm.SampleRate <= 0 {
		return fmt.Errorf("streamedge: audio has no sample rate")
	}

	queuedAt, err := s.queue(pcm, marks)
	if err != nil {
		return err
	}
	// Told with the lock released: the marks belong to the turn, and the track must never
	// wait on one.
	if marks != nil && !queuedAt.IsZero() {
		marks.FirstFrameQueued(queuedAt)
	}
	s.waitForRoom()
	return nil
}

// queue encodes a chunk and puts its frames in the queue. It returns when the first of them
// went in, or the zero time when the encoder held all of it back or marks is nil.
func (s *speaker) queue(pcm audio.PcmData, marks agent.PlayoutMarks) (time.Time, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return time.Time{}, errors.New("streamedge: the call has been left")
	}
	if s.encoder == nil {
		encoder, err := opus.NewEncoder(opus.Config{SampleRate: opusSampleRate, FrameDuration: opusFrameDuration})
		if err != nil {
			return time.Time{}, fmt.Errorf("streamedge: build opus encoder: %w", err)
		}
		s.encoder = encoder
	}

	// A chunk can open with silence, so the first frame to be heard is not always its first.
	// What comes before the first piece with speech in it is encoded on its own, which makes
	// the frame that follows the one to report. The encoder takes audio in whatever pieces it
	// is given, so this does not change what is sent.
	speech := pcm.Samples
	lead := -1
	if marks != nil {
		lead = firstAudible(pcm)
	}
	queued := len(s.frames)
	if lead > 0 {
		packets, err := s.encode(speech[:lead], pcm.SampleRate)
		if err != nil {
			return time.Time{}, err
		}
		s.frames = append(s.frames, packets...)
		s.unflushed = true
		speech = speech[lead:]
	}
	seq := s.head + uint64(len(s.frames))
	packets, err := s.encode(speech, pcm.SampleRate)
	if err != nil {
		return time.Time{}, err
	}
	s.frames = append(s.frames, packets...)
	s.unflushed = true
	if lead >= 0 {
		s.armed = append(s.armed, armedMark{marks: marks, seq: seq})
	}
	if marks == nil || len(s.frames) == queued {
		return time.Time{}, nil
	}
	return s.now(), nil
}

func (s *speaker) encode(samples []int16, sampleRate int) ([][]byte, error) {
	packets, err := s.encoder.Encode(webrtcaudio.FromInt16(samples, sampleRate, 1))
	if err != nil {
		return nil, fmt.Errorf("streamedge: encode speech: %w", err)
	}
	return packets, nil
}

// waitForRoom blocks while the queue is over its bound.
func (s *speaker) waitForRoom() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.frames) <= playoutFrames || s.closed {
		return
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
}

// firstAudible is where the first 20 ms piece of speech with anything in it above the noise a
// voice leaves in its pauses begins, in samples, or -1 when there is none.
func firstAudible(pcm audio.PcmData) int {
	piece := max(pcm.SampleRate/int(time.Second/opusFrameDuration), 1)
	for start := 0; start < len(pcm.Samples); start += piece {
		for _, sample := range pcm.Samples[start:min(start+piece, len(pcm.Samples))] {
			if sample >= audibleSample || sample <= -audibleSample {
				return start
			}
		}
	}
	return -1
}

// NextSample hands the track one frame, or silence when the agent has nothing to say.
//
// It runs on the track's clock, so it neither allocates nor waits on anything but the queue:
// the report of a reply's first audible frame is made once the lock is released.
func (s *speaker) NextSample(ctx context.Context) (webrtcmedia.Sample, error) {
	if err := ctx.Err(); err != nil {
		return webrtcmedia.Sample{}, err
	}

	sample, marks, at := s.next()
	if marks != nil {
		marks.FirstAudiblePulled(at)
	}
	return sample, nil
}

// next takes the frame to hand the track, and the reply to report if it is the first audible
// one of a reply whose speech was not dropped since.
func (s *speaker) next() (webrtcmedia.Sample, agent.PlayoutMarks, time.Time) {
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
		return webrtcmedia.Sample{Data: silenceFrame, Duration: opusFrameDuration}, nil, time.Time{}
	}

	frame := s.frames[0]
	s.frames = s.frames[1:]
	seq := s.head
	s.head++
	s.speaking = true
	s.drained.Signal()

	sample := webrtcmedia.Sample{Data: frame, Duration: opusFrameDuration}
	// A report whose frame never came, because the encoder had nothing to give for it, is
	// behind the track by now and is let go rather than left to hold up the ones after it.
	for len(s.armed) > 0 && s.armed[0].seq < seq {
		s.armed[0] = armedMark{}
		s.armed = s.armed[1:]
	}
	if len(s.armed) == 0 || s.armed[0].seq != seq {
		return sample, nil, time.Time{}
	}
	marked := s.armed[0]
	s.armed[0] = armedMark{}
	s.armed = s.armed[1:]
	return sample, marked.marks, s.now()
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
	s.forgetFramesLocked()
	s.speaking = false
	s.drained.Broadcast()
}

// forgetFramesLocked throws away the queue and the reports owed for what was in it. The
// caller holds the lock.
func (s *speaker) forgetFramesLocked() {
	s.head += uint64(len(s.frames))
	s.frames = nil
	clear(s.armed)
	s.armed = s.armed[:0]
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
	s.forgetFramesLocked()
	s.drained.Broadcast()
	s.encoder = nil
	return nil
}
