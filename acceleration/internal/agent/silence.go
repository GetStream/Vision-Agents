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

// heldReply is a reply to a caller's words, and who the caller is, whose first audio waits
// for that caller to have been quiet. It is the reply until the first of its audio has been
// let out, which is what makes it one nobody has heard any of.
type heldReply struct {
	turn        string
	participant stt.Participant
	// confident says the turn was decided by an acoustic end-of-turn score high enough to be
	// sure of, which shortens the silence the caller is waited for.
	confident bool
	// readyAt is when the first audio of the reply arrived and the hold began, if it had to
	// wait. Zero before.
	readyAt time.Time
	// committed is how long the history was once the reply's own entry had been added to it, so
	// that a reply nobody heard any of can take that entry back. Zero until it was added.
	committed int
}

// heldTurn is the first audio of a reply that is waiting for the caller to have been quiet, with
// the events of its synthesis that arrived after it, which are only acted on once it has been let
// out, in the order they came.
type heldTurn struct {
	turn        string
	participant string
	// readyAt is when the first audio arrived and the hold began, and window how long the caller
	// has to have been quiet to end it.
	readyAt time.Time
	window  time.Duration
	// ctx ends when the reply is abandoned, and stop gives up what waits on it.
	ctx    context.Context
	stop   func() bool
	events []tts.Event
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

// holdFirstFrame says what is to be done with the first frame of a turn's audio, which arrived
// at the time now: the hold it is to wait in, or otherwise whether it may be published.
//
// A reply that answers a caller's words is only let out once the caller has been silent for
// the reply silence since they were last heard to voice anything, however ready it is, because
// the words it answers were settled on a pause that may turn out to be a breath. The silence is
// shorter for a turn that the acoustic end-of-turn model was sure had ended. A caller who has
// been quiet for that long is not waited for. One who has not is waited on until they have, but
// never for longer than the longest hold after the reply was ready: a line that does not go
// quiet, because of a conversation in the room or a steady babble, would otherwise keep it for
// as long as that lasted, and the reply is let out when the hold runs out.
//
// The hold only delays. A voice while it lasts, a cough or a word, only restarts the count of
// quiet, and the reply is never dropped for it: whether the caller really went on is told by
// their words, and those cancel the reply the way they cancel any other.
//
// Nothing waits here: the audio is held by whoever asked, and the rest of what the voice says
// carries on being read. The frames of the reply after the first are never asked about, and a
// turn that is not a reply to a caller, such as a greeting, a murmur or a follow-up nobody asked
// for, is let straight through. A reply that has been abandoned or whose pipeline has stopped is
// let out no more.
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
		ctx: publishCtx}, false
}

// holdLeft is how much longer a held reply is to be held at the time now, which is zero once it
// may be let out, and whether it is let out because the longest hold has passed rather than
// because the caller has been quiet.
func (a *Agent) holdLeft(hold *heldTurn, now time.Time) (left time.Duration, capped bool) {
	return a.voiced.holdFor(hold.participant, hold.window, a.replySilenceMax, hold.readyAt, now)
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

// giveUpHold deals with the events of a reply that will not be let out. The audio of it was
// synthesised and paid for and never published, which the turn says; what else the voice said
// of it is acted on as it would be for any turn, since a completion still settles the synthesis.
func (a *Agent) giveUpHold(hold *heldTurn, speak func(tts.Event)) {
	for _, event := range hold.events {
		if chunk, ok := event.(tts.AudioChunk); ok {
			a.turns.dropped(hold.turn, chunk.Audio.DurationMs())
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
