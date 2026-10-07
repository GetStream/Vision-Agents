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
	// defaultReplySilenceConfident is how long the caller must have been quiet instead, for a
	// reply to a turn that the acoustic end-of-turn model was sure had ended. The silence is there
	// for the endings that are in doubt, and a score that high is the model saying this one is not.
	defaultReplySilenceConfident = 300 * time.Millisecond
	// defaultReplyConfidentScore is the acoustic end-of-turn score from which it is taken to be
	// sure.
	defaultReplyConfidentScore = 0.9
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
	return &voiceActivity{speakers: map[string]*voiceSpeaker{}}
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
	defer v.mu.Unlock()
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

// quietFor is how long a participant's audio has been quiet at the time now, since it last
// carried a voice. It is for good if it never has.
func (v *voiceActivity) quietFor(participantID string, now time.Time) time.Duration {
	last := v.lastVoiced(participantID)
	if last.IsZero() {
		return time.Duration(math.MaxInt64)
	}
	return now.Sub(last)
}

// forget drops what is known of a participant who has left.
func (v *voiceActivity) forget(participantID string) {
	v.mu.Lock()
	delete(v.speakers, participantID)
	v.mu.Unlock()
}

// holdFor is how much longer a reply to a participant is to be held at the time now, which is
// zero once it may be let out, and whether it is let out because the longest hold has passed
// rather than because the caller has been quiet.
//
// A reply that became ready at readyAt waits until the participant's audio has been quiet for
// the window since it last carried a voice. A voice during the wait, whatever it is, only
// restarts that count: the wait never ends the reply, and never lasts beyond the longest hold.
// A participant never heard to voice anything has nothing to wait for.
func (v *voiceActivity) holdFor(participantID string, window, longest time.Duration, readyAt, now time.Time) (left time.Duration, capped bool) {
	last := v.lastVoiced(participantID)
	if last.IsZero() {
		return 0, false
	}
	quiet := now.Sub(last)
	if quiet >= window {
		return 0, false
	}
	remaining := longest - now.Sub(readyAt)
	if remaining <= 0 {
		return 0, true
	}
	return min(window-quiet, remaining), false
}

// wait blocks for up to d. It returns early when either context ends. The caller works out
// which of those it was.
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
	case <-first.Done():
	case <-second.Done():
	}
}

// heldReply is a reply to a caller's words, and who the caller is, whose first audio waits
// for that caller to have been quiet. It is the reply until the first of its audio has been
// let out, which is what makes it one nobody has heard any of.
type heldReply struct {
	turn        string
	participant stt.Participant
	// confident says the turn was decided by an acoustic end-of-turn score high enough to be
	// sure of, which shortens the silence the caller is waited for.
	confident bool
	// readyAt is when the first audio of the reply arrived and the hold began. Zero before.
	readyAt time.Time
	// committed is how long the history was once the reply's own entry had been added to it, so
	// that a reply nobody heard any of can take that entry back. Zero until it was added.
	committed int
}

// silenceFor is how long the caller must have been quiet for a held reply to be let out. A turn
// the acoustic end-of-turn model was sure had ended is waited on for the confident silence, if
// that is shorter, and any other for the reply silence.
func (a *Agent) silenceFor(held heldReply) time.Duration {
	if held.confident {
		return min(a.replySilence, a.replySilenceConfident)
	}
	return a.replySilence
}

// admitFirstFrame says whether the first frame of a turn's audio may be published now.
//
// A reply that answers a caller's words is only let out once the caller has been silent for
// the reply silence since they were last heard to voice anything, however ready it is, because
// the words it answers were settled on a pause that may turn out to be a breath. The silence is
// shorter for a turn that the acoustic end-of-turn model was sure had ended. A caller who
// has been quiet for that long is not waited for. One who has not is waited on until they have,
// but never for longer than the longest hold after the reply was ready: a line that does not go
// quiet, because of a conversation in the room or a steady babble, would otherwise keep it for as
// long as that lasted, and the reply is let out when the hold runs out.
//
// The hold only delays. A voice while it lasts, a cough or a word, only restarts the count of
// quiet, and the reply is never dropped for it: whether the caller really went on is told by
// their words, and those cancel the reply the way they cancel any other.
//
// Only the speaking goroutine asks, and it is the one that waits: no lock is held while it
// does, the wait ends the moment the reply is abandoned or the pipeline stops, and the frames
// of the reply after the first are never asked about. A turn that is not a reply to a caller,
// such as a greeting, a murmur or a follow-up nobody asked for, is let straight through.
func (a *Agent) admitFirstFrame(p *pipeline, publishCtx context.Context, turnID string) bool {
	a.mu.Lock()
	held := a.gated
	if held.turn == "" || held.turn != turnID {
		a.mu.Unlock()
		return true
	}
	readyAt := time.Now()
	a.gated.readyAt = readyAt
	window, longest := a.silenceFor(held), a.replySilenceMax
	a.mu.Unlock()

	voiced := a.voiced
	if voiced == nil || window <= 0 {
		return true
	}
	waited := false
	for {
		if publishCtx.Err() != nil || p.ctx.Err() != nil {
			return false
		}
		now := time.Now()
		left, capped := voiced.holdFor(held.participant.ID, window, longest, readyAt, now)
		if left <= 0 {
			// Guarded, because formatting a line is an allocation on the path of every reply.
			if capped && a.logger.Enabled(publishCtx, slog.LevelDebug) {
				a.logger.Debug("the caller never went quiet, so the reply was let out after the longest hold",
					"turn", turnID, "participant", held.participant.ID, "held", now.Sub(readyAt))
			} else if waited && a.logger.Enabled(publishCtx, slog.LevelDebug) {
				a.logger.Debug("the caller stayed quiet, so the reply was let out",
					"turn", turnID, "participant", held.participant.ID, "held", now.Sub(readyAt))
			}
			return true
		}
		if !waited {
			waited = true
			if a.logger.Enabled(publishCtx, slog.LevelDebug) {
				a.logger.Debug("holding the reply until the caller has been quiet",
					"turn", turnID, "participant", held.participant.ID, "window", window, "longest", longest,
					"confident", held.confident)
			}
		}
		voiced.wait(left, publishCtx, p.ctx)
	}
}
