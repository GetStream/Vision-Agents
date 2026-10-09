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
	// audibleSample is the level a sample has to reach, in either direction, for the frame
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
	// now is the clock the playout milestones are read from.
	now func() time.Time
	// writeGate keeps provider chunks in order without blocking DropSpeech or the track.
	// It has one token and is also selectable on context cancellation and Close.
	writeGate  chan struct{}
	closedDone chan struct{}

	mu sync.Mutex
	// drained wakes a writer once the queue is back under the playout bound.
	drained *sync.Cond
	// pulled is set once the track has asked for its first frame. Until it does, nothing
	// written here is going anywhere.
	pulled bool
	// generation changes whenever queued speech is abandoned or the call closes. Writers
	// capture it before publishing, so a Background writer cannot refill after DropSpeech.
	generation uint64
	// cancelWake is cached for one publication context, since each TTS chunk in an epoch
	// shares its Done channel. It is installed only when the queue first blocks.
	cancelWakeDone <-chan struct{}
	cancelWakeStop func() bool
	// writers prevents NextSample from treating an internal PCM slice boundary as the
	// end of an utterance and flushing the encoder's tail.
	writers int
	// frames are encoded and waiting to be sent, oldest first.
	frames [][]byte
	// head counts the frames that have left the queue, taken by the track or dropped, so a
	// queued frame stays at the position head plus its index as the ones ahead of it go.
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

// armedMark is a reply waiting on the track, with where in the queue its frame is and the
// generation it was queued in, so a frame that outlives its reply is never reported for it.
type armedMark struct {
	marks      agent.PlayoutMarks
	seq        uint64
	generation uint64
}

func newSpeaker(logger *slog.Logger) *speaker {
	if logger == nil {
		logger = slog.Default()
	}
	talker := &speaker{
		logger:     logger,
		now:        time.Now,
		writeGate:  make(chan struct{}, 1),
		closedDone: make(chan struct{}),
	}
	talker.writeGate <- struct{}{}
	talker.drained = sync.NewCond(&talker.mu)
	return talker
}

// Write queues a chunk of the agent's speech, blocking while the queue is full so the agent
// publishes at the rate the audio is heard rather than as fast as it is synthesised.
func (s *speaker) Write(pcm audio.PcmData) error {
	return s.WriteContext(context.Background(), pcm)
}

// WriteContext queues speech until the track has room for it, or ctx is cancelled.
// Feeding the encoder in at most 20 ms source pieces bounds both the queue and the
// time spent encoding while holding the speaker lock.
func (s *speaker) WriteContext(ctx context.Context, pcm audio.PcmData) error {
	return s.WriteMarked(ctx, pcm, nil)
}

// WriteMarked is WriteContext that also tells marks when the first frame of the chunk was
// queued and when the track took the first one that was not silence. WriteContext returns
// with the whole chunk queued, which for one longer than the queue is as the track drains
// all but the end of it, so neither moment can be read from the return.
//
// Both belong to the generation the chunk was queued in: DropSpeech forgets the frames
// still waiting, and with them the report owed for them, so abandoned speech is never
// counted as the start of the reply that follows.
func (s *speaker) WriteMarked(ctx context.Context, pcm audio.PcmData, marks agent.PlayoutMarks) error {
	if err := ctx.Err(); err != nil {
		return err
	}
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
	if err := ctx.Err(); err != nil {
		s.mu.Unlock()
		return err
	}
	if s.closed {
		s.mu.Unlock()
		return errors.New("streamedge: the call has been left")
	}
	generation := s.generation
	s.writers++
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.writers--
		s.drained.Broadcast()
		s.mu.Unlock()
	}()

	select {
	case <-s.writeGate:
	case <-ctx.Done():
		return ctx.Err()
	case <-s.closedDone:
		return errors.New("streamedge: the call has been left")
	}
	defer func() { s.writeGate <- struct{}{} }()
	s.mu.Lock()
	if err := ctx.Err(); err != nil {
		s.mu.Unlock()
		return err
	}
	if s.closed {
		s.mu.Unlock()
		return errors.New("streamedge: the call has been left")
	}
	if generation != s.generation {
		s.mu.Unlock()
		return context.Canceled
	}
	s.mu.Unlock()

	// Integer division keeps each source piece at or below 20 ms; common provider rates
	// divide evenly by 50.
	samplesPerPiece := pcm.SampleRate / int(time.Second/opusFrameDuration)
	if samplesPerPiece < 1 {
		samplesPerPiece = 1
	}
	// Convert once so each encoded piece can be a view into the same sample buffer rather
	// than allocating a fresh PCM conversion for every 20 ms slice. Check cancellation at
	// the same boundaries used for encoding so even conversion of a large provider chunk
	// does not delay an abandoned reply.
	inputSamples := make([]float32, len(pcm.Samples))
	for start := 0; start < len(pcm.Samples); start += samplesPerPiece {
		if err := ctx.Err(); err != nil {
			s.mu.Lock()
			s.resetTailAfterCancellationLocked(ctx, generation, err)
			s.mu.Unlock()
			return err
		}
		end := min(start+samplesPerPiece, len(pcm.Samples))
		for i := start; i < end; i++ {
			inputSamples[i] = float32(pcm.Samples[i]) * (1.0 / 32768.0)
		}
	}

	s.mu.Lock()
	if err := ctx.Err(); err != nil {
		s.resetTailAfterCancellationLocked(ctx, generation, err)
		s.mu.Unlock()
		return err
	}
	if s.closed {
		s.mu.Unlock()
		return errors.New("streamedge: the call has been left")
	}
	if generation != s.generation {
		s.mu.Unlock()
		return context.Canceled
	}
	s.mu.Unlock()

	var waited time.Time
	var queued int
	// queuedAt is when the first frame went into the queue. It is reported as soon as the
	// piece holding it is done, or on the way out if the write ends first.
	var queuedAt time.Time
	var told, listening, armed bool
	defer func() {
		if marks != nil && !told && !queuedAt.IsZero() {
			marks.FirstFrameQueued(queuedAt)
		}
	}()
	for start := 0; start < len(pcm.Samples); {
		s.mu.Lock()
		if err := s.waitForRoomLocked(ctx, generation, &waited, &queued); err != nil {
			s.resetTailAfterCancellationLocked(ctx, generation, err)
			s.mu.Unlock()
			return err
		}
		if s.encoder == nil {
			encoder, err := opus.NewEncoder(opus.Config{SampleRate: opusSampleRate, FrameDuration: opusFrameDuration})
			if err != nil {
				s.mu.Unlock()
				return fmt.Errorf("streamedge: build opus encoder: %w", err)
			}
			s.encoder = encoder
		}

		end := min(start+samplesPerPiece, len(pcm.Samples))
		piece := inputSamples[start:end]
		if marks != nil && !armed && !listening {
			listening = audible(pcm.Samples[start:end])
		}
		packets, err := s.encoder.Encode(webrtcaudio.FromFloat32(piece, pcm.SampleRate, 1))
		if err != nil {
			s.mu.Unlock()
			return fmt.Errorf("streamedge: encode speech: %w", err)
		}
		if generation != s.generation {
			s.mu.Unlock()
			return context.Canceled
		}
		if err := ctx.Err(); err != nil {
			s.resetTailAfterCancellationLocked(ctx, generation, err)
			s.mu.Unlock()
			return err
		}
		s.unflushed = true
		for _, packet := range packets {
			if err := s.waitForRoomLocked(ctx, generation, &waited, &queued); err != nil {
				s.resetTailAfterCancellationLocked(ctx, generation, err)
				s.mu.Unlock()
				return err
			}
			// Check under mu immediately before enqueue. A cancelled writer that was waiting
			// when DropSpeech ran can never refill the cleared queue.
			if err := ctx.Err(); err != nil {
				s.resetTailAfterCancellationLocked(ctx, generation, err)
				s.mu.Unlock()
				return err
			}
			seq := s.head + uint64(len(s.frames))
			s.frames = append(s.frames, packet)
			if marks == nil {
				continue
			}
			if queuedAt.IsZero() {
				queuedAt = s.now()
			}
			if listening && !armed {
				s.armed = append(s.armed, armedMark{marks: marks, seq: seq, generation: generation})
				armed = true
			}
		}
		if err := ctx.Err(); err != nil {
			s.resetTailAfterCancellationLocked(ctx, generation, err)
			s.mu.Unlock()
			return err
		}
		if generation != s.generation {
			s.mu.Unlock()
			return context.Canceled
		}
		start = end
		s.mu.Unlock()
		if marks != nil && !told && !queuedAt.IsZero() {
			marks.FirstFrameQueued(queuedAt)
			told = true
		}
	}

	if !waited.IsZero() {
		elapsed := time.Since(waited)
		if elapsed > time.Duration(queued)*opusFrameDuration+time.Second {
			s.mu.Lock()
			pulled := s.pulled
			s.mu.Unlock()
			s.logger.Warn("speech waited to go out, the call was not taking audio",
				"waited", elapsed, "queued", queued, "pulled", pulled)
		}
	}
	return nil
}

// audible reports whether a piece of speech has anything in it above the noise a voice leaves
// in its pauses.
func audible(samples []int16) bool {
	for _, sample := range samples {
		if sample >= audibleSample || sample <= -audibleSample {
			return true
		}
	}
	return false
}

// waitForRoomLocked returns with s.mu held. DropSpeech and Close wake waiting writers;
// callers check cancellation and the generation under the lock before publishing.
func (s *speaker) waitForRoomLocked(ctx context.Context, generation uint64,
	waited *time.Time, queued *int) error {
	for len(s.frames) >= playoutFrames && !s.closed {
		if err := ctx.Err(); err != nil {
			return err
		}
		if waited.IsZero() {
			*waited = time.Now()
			*queued = len(s.frames)
			s.registerCancellationWakeLocked(ctx)
		}
		s.drained.Wait()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if s.closed {
		return errors.New("streamedge: the call has been left")
	}
	if generation != s.generation {
		return context.Canceled
	}
	return nil
}

// registerCancellationWakeLocked caches one waiter callback per context Done channel.
// Agent publications reuse their epoch context for each provider chunk, so the callback
// is not allocated again for every chunk in the same epoch.
func (s *speaker) registerCancellationWakeLocked(ctx context.Context) {
	done := ctx.Done()
	if done == s.cancelWakeDone {
		return
	}
	if s.cancelWakeStop != nil {
		s.cancelWakeStop()
	}
	s.cancelWakeDone = done
	s.cancelWakeStop = nil
	if done == nil {
		return
	}
	s.cancelWakeStop = context.AfterFunc(ctx, func() {
		s.mu.Lock()
		s.drained.Broadcast()
		s.mu.Unlock()
	})
}

// stopCancellationWakeLocked releases the cached context until the next blocked write.
func (s *speaker) stopCancellationWakeLocked() {
	if s.cancelWakeStop != nil {
		s.cancelWakeStop()
	}
	s.cancelWakeDone = nil
	s.cancelWakeStop = nil
}

// resetTailAfterCancellationLocked ensures a partially encoded cancelled write cannot be
// flushed by a later NextSample. DropSpeech clears complete queued frames as well.
func (s *speaker) resetTailAfterCancellationLocked(ctx context.Context, generation uint64, err error) {
	if generation == s.generation && (errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil) {
		if s.encoder != nil {
			s.encoder.Reset()
		}
		s.unflushed = false
	}
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
	if len(s.frames) == 0 && s.writers == 0 {
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
	if len(s.armed) == 0 || s.armed[0].seq != seq {
		return sample, nil, time.Time{}
	}
	marked := s.armed[0]
	s.armed[0] = armedMark{}
	s.armed = s.armed[1:]
	if marked.generation != s.generation {
		return sample, nil, time.Time{}
	}
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

	s.generation++
	s.stopCancellationWakeLocked()
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
	s.generation++
	close(s.closedDone)
	s.stopCancellationWakeLocked()
	s.forgetFramesLocked()
	s.drained.Broadcast()
	s.encoder = nil
	return nil
}
