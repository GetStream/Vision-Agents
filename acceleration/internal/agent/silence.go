package agent

import (
	"context"
	"log/slog"
	"math"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const (
	// defaultReplySilence is how long a caller must have been quiet, on their audio, before the
	// first sound of the reply to them is let out. The words a reply answers are settled on a
	// pause, and a pause in the middle of a turn is shorter than the end of one, so waiting for
	// the silence to be confirmed catches the replies that were started on a breath.
	defaultReplySilence = 700 * time.Millisecond
	// voicedFloor is the quietest level, as RMS on the 16-bit scale, that a chunk of audio can
	// have and still be taken for a voice: about -42 dBFS, above the hiss of a quiet line and
	// below a soft voice.
	voicedFloor = 250.0
	// voicedOverNoise is how many times louder than a participant's own background a chunk has
	// to be to count as a voice, so a room that is merely noisy is not heard as somebody
	// talking.
	voicedOverNoise = 3.0
	// noiseSettle and noiseDrift are how long the estimate of a participant's background takes
	// to move most of the way up to a level that has lasted, while it is quiet enough to be
	// background and while it is not. It falls at once, so a pause between words finds it.
	// Drifting up through a voice is slow on purpose, and is only there so that a background
	// that became louder for good is eventually taken for one.
	noiseSettle = 2 * time.Second
	noiseDrift  = 20 * time.Second
	// noiseCeiling is the loudest a background is taken to be, so a long turn cannot raise the
	// bar above the voice that is speaking over it: any voice louder than voicedOverNoise times
	// this is heard however the estimate has drifted.
	noiseCeiling = 2 * voicedFloor
	// chunkDuration is what a chunk of audio that does not say how long it is counts for.
	chunkDuration = 20 * time.Millisecond
)

// voiceActivity follows when each participant's audio last carried a voice.
//
// It is what tells the agent that a caller has gone quiet or has started again without
// reading their words: a transcript arrives after the audio it describes, and a transcriber
// that finalizes a turn has only decided the caller stopped for a moment.
type voiceActivity struct {
	mu       sync.Mutex
	speakers map[string]*voiceSpeaker
	// wake is nudged whenever a voice is heard, which is what ends a wait for silence early.
	wake chan struct{}

	// waiting holds the one timer a wait uses, so waiting allocates nothing once it exists,
	// and serialises the waits that share it.
	waiting sync.Mutex
	timer   *time.Timer
}

// voiceSpeaker is what is known of one participant's audio.
type voiceSpeaker struct {
	// noise is the running estimate of their background, as RMS.
	noise float64
	// last is when a chunk of their audio was last found to carry a voice. Zero until one was.
	last time.Time
}

func newVoiceActivity() *voiceActivity {
	return &voiceActivity{speakers: map[string]*voiceSpeaker{}, wake: make(chan struct{}, 1)}
}

// observe takes a chunk of a participant's audio, heard at the given time. It allocates only
// the first time it hears a participant.
func (v *voiceActivity) observe(participantID string, pcm audio.PcmData, at time.Time) {
	if participantID == "" || len(pcm.Samples) == 0 {
		return
	}
	var squares float64
	for _, sample := range pcm.Samples {
		squares += float64(sample) * float64(sample)
	}
	level := math.Sqrt(squares / float64(len(pcm.Samples)))
	length := time.Duration(pcm.DurationMs() * float64(time.Millisecond))
	if length <= 0 {
		length = chunkDuration
	}

	v.mu.Lock()
	speaker := v.speakers[participantID]
	if speaker == nil {
		// Somebody is taken to be as noisy as the first thing heard from them, so a call that
		// opens with speech is not mistaken for a noisy room.
		speaker = &voiceSpeaker{noise: min(level, noiseCeiling)}
		v.speakers[participantID] = speaker
	}
	voiced := level >= max(voicedFloor, voicedOverNoise*speaker.noise)
	switch {
	case level < speaker.noise:
		speaker.noise = level
	case voiced:
		speaker.noise += (level - speaker.noise) * min(1, float64(length)/float64(noiseDrift))
	default:
		speaker.noise += (level - speaker.noise) * min(1, float64(length)/float64(noiseSettle))
	}
	speaker.noise = min(speaker.noise, noiseCeiling)
	if voiced {
		speaker.last = at
	}
	v.mu.Unlock()

	if voiced {
		select {
		case v.wake <- struct{}{}:
		default:
		}
	}
}

// lastVoiced is when a participant's audio last carried a voice, or the zero time if it never
// has.
func (v *voiceActivity) lastVoiced(participantID string) time.Time {
	v.mu.Lock()
	defer v.mu.Unlock()
	if speaker := v.speakers[participantID]; speaker != nil {
		return speaker.last
	}
	return time.Time{}
}

// forget drops what is known of a participant who has left.
func (v *voiceActivity) forget(participantID string) {
	v.mu.Lock()
	delete(v.speakers, participantID)
	v.mu.Unlock()
}

// wait blocks for up to d. It returns early when a voice is heard, and when either context
// ends. The caller works out which of those it was.
func (v *voiceActivity) wait(d time.Duration, first, second context.Context) {
	v.waiting.Lock()
	defer v.waiting.Unlock()
	if v.timer == nil {
		v.timer = time.NewTimer(d)
	} else {
		v.timer.Reset(d)
	}
	defer v.timer.Stop()
	select {
	case <-v.timer.C:
	case <-v.wake:
	case <-first.Done():
	case <-second.Done():
	}
}

// heldReply is a reply to a caller's words, and who the caller is, whose first audio waits
// for that caller to have been quiet.
type heldReply struct {
	turn        string
	participant stt.Participant
}

// admitFirstFrame says whether the first frame of a turn's audio may be published now.
//
// A reply that answers a caller's words is only let out once the caller has been silent for
// the reply silence since they were last heard to voice anything, however ready it is, because
// the words it answers were settled on a pause that may turn out to be a breath. A caller who
// has been quiet for that long is not waited for. One who has not is waited on until they have,
// and one who starts again in the meantime has the reply dropped unheard, the way talking over
// it a moment later would have, except that nothing of it ever reached them.
//
// Only the speaking goroutine asks, and it is the one that waits: no lock is held while it
// does, the wait ends the moment the reply is abandoned or the pipeline stops, and the frames
// of the reply after the first are never asked about. A turn that is not a reply to a caller,
// such as a greeting, a murmur or a follow-up nobody asked for, is let straight through.
func (a *Agent) admitFirstFrame(p *pipeline, publishCtx context.Context, turnID string) bool {
	a.mu.Lock()
	held := a.gated
	if held.turn != turnID {
		a.mu.Unlock()
		return true
	}
	a.gated = heldReply{}
	window := a.replySilence
	a.mu.Unlock()

	voiced := a.voiced
	if voiced == nil || window <= 0 {
		return true
	}
	var heldSince time.Time
	last := voiced.lastVoiced(held.participant.ID)
	for {
		if publishCtx.Err() != nil || p.ctx.Err() != nil {
			return false
		}
		latest := voiced.lastVoiced(held.participant.ID)
		if latest.After(last) {
			a.dropHeldReply(p, held, heldSince)
			return false
		}
		quiet := time.Since(latest)
		if latest.IsZero() || quiet >= window {
			if !heldSince.IsZero() && a.logger.Enabled(publishCtx, slog.LevelDebug) {
				a.logger.Debug("the caller stayed quiet, so the reply was let out",
					"turn", turnID, "participant", held.participant.ID, "held", time.Since(heldSince))
			}
			return true
		}
		if heldSince.IsZero() {
			heldSince = time.Now()
			// Guarded, because formatting a line is an allocation on the path of every reply.
			if a.logger.Enabled(publishCtx, slog.LevelDebug) {
				a.logger.Debug("holding the reply until the caller has been quiet",
					"turn", turnID, "participant", held.participant.ID, "quiet", quiet, "window", window)
			}
		}
		voiced.wait(window-quiet, publishCtx, p.ctx)
	}
}

// dropHeldReply gives up a reply whose caller started talking again before any of it was
// let out. It is a caller taking the floor before the reply was heard, so it ends as one:
// the turn is abandoned and reported as interrupted.
func (a *Agent) dropHeldReply(p *pipeline, held heldReply, since time.Time) {
	var waited time.Duration
	if !since.IsZero() {
		waited = time.Since(since)
	}
	a.logger.Debug("the caller started again before the reply was heard, so it was dropped",
		"turn", held.turn, "participant", held.participant.ID, "held", waited)
	stopped, ok := a.stopPlayback(held.participant, held.turn, time.Time{}, "reply_gate", "audio")
	if !ok {
		return
	}
	// A reply that finished generating while it was held has the end of its speech queued
	// behind the frame that was held, which would close the turn as spoken. It is closed as
	// interrupted first.
	a.turns.interrupt(held.turn)
	// The rest of ending the interruption calls the voice, whose events are read by the
	// goroutine that is here, so it is finished beside it.
	p.running.Add(1)
	go func() {
		defer p.running.Done()
		a.finishInterruptedTurn(stopped)
	}()
}
