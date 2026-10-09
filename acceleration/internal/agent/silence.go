package agent

import (
	"context"
	"log/slog"
	"math"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

const (
	// defaultReplySilence catches replies started on a breath rather than a completed turn.
	defaultReplySilence = 700 * time.Millisecond
	// defaultReplySilenceMax bounds the hold even when background noise never stops.
	defaultReplySilenceMax = time.Second
	// defaultReplySilenceConfident shortens the hold for a high-confidence acoustic ending.
	defaultReplySilenceConfident = 300 * time.Millisecond
	// defaultReplyConfidentScore is the threshold for the shorter hold.
	defaultReplyConfidentScore = 0.9
	// voicedFloor is the minimum voice RMS on the PCM16 scale, about -42 dBFS.
	voicedFloor = 250.0
	// voicedOverNoise rejects audio below three times the speaker's background RMS.
	voicedOverNoise = 3.0
	// noiseSettle follows quiet background changes; noiseDrift adapts slowly through
	// speech. The estimate falls immediately during a pause.
	noiseSettle = 2 * time.Second
	noiseDrift  = 20 * time.Second
	// noiseCeiling keeps sustained speech from raising the threshold above the voice.
	noiseCeiling = 2 * voicedFloor
	// chunkDuration is the fallback for audio with no duration.
	chunkDuration = 20 * time.Millisecond
)

// voiceActivity tracks each participant's last voiced audio, ahead of delayed transcripts.
type voiceActivity struct {
	mu       sync.Mutex
	speakers map[string]*voiceSpeaker
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

// levelOf is the RMS of a chunk of audio on the 16-bit scale.
func levelOf(samples []int16) float64 {
	if len(samples) == 0 {
		return 0
	}
	var squares float64
	for _, sample := range samples {
		squares += float64(sample) * float64(sample)
	}
	return math.Sqrt(squares / float64(len(samples)))
}

// lengthOf is how long a chunk of audio plays for, which is chunkDuration if it does not say.
func lengthOf(pcm audio.PcmData) time.Duration {
	length := time.Duration(pcm.DurationMs() * float64(time.Millisecond))
	if length <= 0 {
		return chunkDuration
	}
	return length
}

// observe takes a chunk of a participant's audio, heard at the given time. It allocates only
// the first time it hears a participant.
func (v *voiceActivity) observe(participantID string, pcm audio.PcmData, at time.Time) {
	if participantID == "" || len(pcm.Samples) == 0 {
		return
	}
	level := levelOf(pcm.Samples)
	length := lengthOf(pcm)

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

// holdFor waits for a quiet window, bounded by longest since readyAt. New voice
// restarts the quiet window without cancelling the reply. capped reports a timeout.
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

// heldReply tracks a reply until its first audio is published.
type heldReply struct {
	turn        string
	participant stt.Participant
	// confident selects the shorter silence window.
	confident bool
	// readyAt is the first audio arrival, or zero before the hold begins.
	readyAt time.Time
	// committed is the history length after adding the assistant entry, for rollback.
	committed int
	// asked is the history length after adding the user entry, for rollback.
	asked int
	// noted records whether this turn consumed an interruption note, for rollback.
	noted bool
}

// unansweredWords records the last user turn whose unheard reply was discarded.
// A continuation replaces that history entry instead of creating another turn.
type unansweredWords struct {
	// asked is the history length after adding the user entry, for rollback.
	asked int
	// text is the words, which the entry must still hold for them to be replaced.
	text string
	// noted records whether this turn consumed an interruption note, for rollback.
	noted bool
}

// heldTurn buffers audio and subsequent synthesis events until a silence hold ends.
// Events are replayed in order; other turns keep draining meanwhile.
type heldTurn struct {
	turn        string
	participant string
	// readyAt is the first audio arrival, or zero before the hold begins.
	readyAt time.Time
	window  time.Duration
	longest time.Duration
	// ctx ends when the reply is abandoned, and stop gives up what waits on it.
	ctx    context.Context
	stop   func() bool
	events []tts.Event
}

// silenceFor selects the shorter window for a confident acoustic ending.
func (a *Agent) silenceFor(held heldReply) time.Duration {
	if held.confident {
		return min(a.replySilence, a.replySilenceConfident)
	}
	return a.replySilence
}

// holdFirstFrame returns a hold or permission to publish the first audio. Only
// replies to a caller wait; greetings and murmurs pass through. Cancellation
// suppresses publication, while voice alone only delays it until the hold expires.
// The caller buffers the audio so other synthesis events can continue draining.
func (a *Agent) holdFirstFrame(p *pipeline, publishCtx context.Context, turnID string, now time.Time) (hold *heldTurn, publish bool) {
	a.mu.Lock()
	held := a.gated
	window := a.silenceFor(held)
	a.mu.Unlock()
	if held.turn == "" || held.turn != turnID {
		return nil, true
	}
	if publishCtx.Err() != nil || p.ctx.Err() != nil {
		return nil, false
	}
	voiced := a.voiced
	if voiced == nil || window <= 0 {
		return nil, true
	}
	if left, _ := voiced.holdFor(held.participant.ID, window, a.replySilenceMax, now, now); left <= 0 {
		return nil, true
	}
	a.mu.Lock()
	if a.gated.turn == turnID {
		a.gated.readyAt = now
	}
	a.mu.Unlock()
	if a.logger.Enabled(publishCtx, slog.LevelDebug) {
		a.logger.Debug("holding the reply until the caller has been quiet",
			"turn", turnID, "participant", held.participant.ID, "window", window,
			"longest", a.replySilenceMax, "confident", held.confident)
	}
	return &heldTurn{turn: turnID, participant: held.participant.ID, readyAt: now, window: window,
		longest: a.replySilenceMax, ctx: publishCtx}, false
}

// holdLeft returns the remaining wait and whether its time limit expired.
func (a *Agent) holdLeft(hold *heldTurn, now time.Time) (left time.Duration, capped bool) {
	return a.voiced.holdFor(hold.participant, hold.window, hold.longest, hold.readyAt, now)
}

// heldBy is the hold that an event of a synthesis belongs to, which is the one for its turn,
// or nil when the turn is not being held or the event is not about a synthesis.
func heldBy(holds []*heldTurn, event tts.Event) *heldTurn {
	if len(holds) == 0 {
		return nil
	}
	var synthesisID string
	switch typed := event.(type) {
	case tts.AudioChunk:
		synthesisID = typed.SynthesisID
	case tts.SynthesisComplete:
		synthesisID = typed.SynthesisID
	case tts.Error:
		synthesisID = typed.SynthesisID
	default:
		return nil
	}
	turnID := turnOf(synthesisID)
	for _, hold := range holds {
		if hold.turn == turnID {
			return hold
		}
	}
	return nil
}

// giveUpHold counts buffered audio as dropped but still processes completion
// and error events to settle the synthesis.
func (a *Agent) giveUpHold(hold *heldTurn, speak func(tts.Event)) {
	for _, event := range hold.events {
		if chunk, ok := event.(tts.AudioChunk); ok {
			a.turns.droppedFromHold(hold.turn, chunk.Audio.DurationMs())
			continue
		}
		speak(event)
	}
}

// logHoldEnded says why a held reply was let out.
func (a *Agent) logHoldEnded(hold *heldTurn, capped bool, now time.Time) {
	if !a.logger.Enabled(hold.ctx, slog.LevelDebug) {
		return
	}
	if capped {
		a.logger.Debug("the caller never went quiet, so the reply was let out after the longest hold",
			"turn", hold.turn, "participant", hold.participant, "held", now.Sub(hold.readyAt))
		return
	}
	a.logger.Debug("the caller stayed quiet, so the reply was let out",
		"turn", hold.turn, "participant", hold.participant, "held", now.Sub(hold.readyAt))
}
