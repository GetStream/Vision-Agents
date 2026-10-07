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
	// defaultReplySilenceMax is the longest the first sound of a reply is held for that silence
	// once it is ready. A line that never goes quiet, because somebody else is talking in the
	// room or there is a steady babble, never confirms the silence, and a reply waiting for it
	// would wait for as long as that lasts. When this has passed the reply plays, and a caller
	// who really is talking over it is dealt with the way any interruption is.
	defaultReplySilenceMax = time.Second
	// voiceResumeQuiet is how much of a participant's audio has to stay below the level of a
	// voice for the next voice after it to be them starting again, as against the same sound
	// carrying on. A voice that comes back sooner than this is a breath or a gap between
	// words, which is also what a steady babble looks like to a level detector, so it does not
	// count as the caller resuming and does not drop a reply that is waiting.
	voiceResumeQuiet = 150 * time.Millisecond
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
	// wake is nudged whenever a participant starts talking after a quiet stretch, which is what
	// ends a wait for silence early.
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
	// onset is when a voice last began after at least voiceResumeQuiet of audio below the level
	// of one, which is the last time they started talking as against kept talking. Zero until
	// one did.
	onset time.Time
	// quiet is how much of their audio since a voice was last heard stayed below the level of
	// one, up to voiceResumeQuiet. It is counted in audio rather than in wall time, so a late
	// chunk is not mistaken for a pause. Somebody never heard from has had a long one.
	quiet time.Duration
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
		speaker = &voiceSpeaker{noise: min(level, noiseCeiling), quiet: voiceResumeQuiet}
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
	started := false
	if voiced {
		started = speaker.quiet >= voiceResumeQuiet
		if started {
			speaker.onset = at
		}
		speaker.last = at
		speaker.quiet = 0
	} else {
		speaker.quiet = min(speaker.quiet+length, voiceResumeQuiet)
	}
	v.mu.Unlock()

	if started {
		select {
		case v.wake <- struct{}{}:
		default:
		}
	}
}

// lastVoiced is when a participant's audio last carried a voice, or the zero time if it never
// has.
func (v *voiceActivity) lastVoiced(participantID string) time.Time {
	last, _ := v.heard(participantID)
	return last
}

// heard is when a participant's audio last carried a voice and when they last started talking
// after a quiet stretch, each the zero time if it never has.
func (v *voiceActivity) heard(participantID string) (last, onset time.Time) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if speaker := v.speakers[participantID]; speaker != nil {
		return speaker.last, speaker.onset
	}
	return time.Time{}, time.Time{}
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
// but never for longer than the longest hold after the reply was ready: a line that does not go
// quiet, because of a conversation in the room or a steady babble, would otherwise keep it for as
// long as that lasted, and the reply is let out when the hold runs out.
//
// A reply is dropped unheard only when the caller resumes while it is held, meaning a voice
// after a stretch of audio below the level of one, the way talking over it a moment later would
// have ended it, except that nothing of it ever reached them. Sound that carries on without
// having gone quiet is not a caller starting again, so it only holds the reply up to the cap.
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
	window, longest := a.replySilence, a.replySilenceMax
	a.mu.Unlock()

	voiced := a.voiced
	if voiced == nil || window <= 0 {
		return true
	}
	// The first frame is ready now, which is when the hold starts, and a caller who started
	// talking before it was ready is not one resuming while it is held.
	readyAt := time.Now()
	_, began := voiced.heard(held.participant.ID)
	waited := false
	for {
		if publishCtx.Err() != nil || p.ctx.Err() != nil {
			return false
		}
		latest, onset := voiced.heard(held.participant.ID)
		if onset.After(began) {
			a.dropHeldReply(p, held, readyAt)
			return false
		}
		quiet := time.Since(latest)
		if latest.IsZero() || quiet >= window {
			if waited && a.logger.Enabled(publishCtx, slog.LevelDebug) {
				a.logger.Debug("the caller stayed quiet, so the reply was let out",
					"turn", turnID, "participant", held.participant.ID, "held", time.Since(readyAt))
			}
			return true
		}
		remaining := longest - time.Since(readyAt)
		if remaining <= 0 {
			if a.logger.Enabled(publishCtx, slog.LevelDebug) {
				a.logger.Debug("the caller never went quiet, so the reply was let out after the longest hold",
					"turn", turnID, "participant", held.participant.ID, "quiet", quiet, "held", time.Since(readyAt))
			}
			return true
		}
		if !waited {
			waited = true
			// Guarded, because formatting a line is an allocation on the path of every reply.
			if a.logger.Enabled(publishCtx, slog.LevelDebug) {
				a.logger.Debug("holding the reply until the caller has been quiet",
					"turn", turnID, "participant", held.participant.ID, "quiet", quiet,
					"window", window, "longest", longest)
			}
		}
		voiced.wait(min(window-quiet, remaining), publishCtx, p.ctx)
	}
}

// dropHeldReply gives up a reply whose caller resumed talking before any of it was let out. It
// is a caller taking the floor before the reply was heard, so it ends as one: the turn is
// abandoned and reported as interrupted.
func (a *Agent) dropHeldReply(p *pipeline, held heldReply, since time.Time) {
	a.logger.Debug("the caller started again before the reply was heard, so it was dropped",
		"turn", held.turn, "participant", held.participant.ID, "held", time.Since(since))
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
