package audiofloor

import (
	"context"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// A caller in a noisy room can finish and never pause: the room keeps the speech gate voiced, so
// the pause the policy's checks count from the voice's last frame never begins, or begins again at
// every burst of the room, and the turn does not end until the caller speaks again. In τ³-Voice's
// regular rooms (a TV, a street, people talking) the gate was voiced on about 85% of the frames
// after a caller had finished, and a third of the callers' ends went more than 2 s without their
// next check; most of those never ended before the caller asked whether anybody was there. The
// level can't tell the caller from such a room, whose loudest tenth is as loud as the caller's
// middle, nor can the speech detector, which takes a TV and babble for speech. The words can: the
// server writes nothing of a room, and what it writes of the caller grows while they talk.
//
// So a pause the gate can't hear is measured by the words instead. Every request that writes the
// utterance's words with the floor free moves the clock: a pause's first check, an interim, and a
// held check, which the transcriber asks while the gate holds the line voiced once nothing has
// written the words for heldEvery of speech. A held check whose words are the same as they have
// been for stillFor, with the gate voiced for at least half of that time, ends the turn on its own
// ruling, as the policy's cap ends one: the model's disposition and floor, whatever p_eot says.
//
// The voiced half is what keeps a quiet line as it is. After a caller finishes on a quiet line the
// gate is quiet, the pause's checks rule as they always have, and none of this asks or ends
// anything: a cough, a breath or a slow caller's next digit is not voiced for half of stillFor. On
// a quiet line held checks are asked only in a voiced stretch with no pause and no interim for
// heldEvery, as a caller's first second and more before the utterance's language is known, and
// their words, which change while the caller talks, end nothing. Over the agent nothing is asked:
// what a caller voiced over it means is the checks over the agent's to say. Words that are none
// end nothing either: an utterance the server has written nothing of is the room alone, and
// ending it would cut off the first word of a caller who begins as it ends.
//
// A floor that rules alone (NewFloor) has the server write no words: they are the other
// transcriber's, which come as it writes them, so they move the clock on every voiced frame they
// are the utterance's, and nothing is asked to write them. Its held check asks for the decision
// alone, and a turn it ends has the other transcriber's words, cut at the caller's last new word,
// as a pause's check that ends one does.

// stillFor is how long the caller's words must have held still for a held check to end the turn,
// unless the agent's point lets a caller who has finished wait longer (stillBound). It is the safe
// preset's wait, the most a finished caller waits at p90 on a quiet line, where turns end 0.78 s
// after the caller at the median and 1.09 s at p90: a caller in a noisy room is given as long to go
// on as a quiet one, counted from their last new word rather than their voice's last frame. The
// words are written about once a second of speech, so on the regular rooms' timelines the turn
// would end 2.2 s after the caller had finished at the median and under 3 s at p90, where 112 of
// 386 such ends never came before the caller spoke again: a caller who has stopped and hears
// nothing for 5 s asks whether anybody is there, and the reply takes 1.9 s at the median and
// 2.5 s at p90 from the turn's end to its first sound.
const stillFor = 1100 * time.Millisecond

// heldEvery is the speech, voiced frames of the utterance, after which a held check writes the
// words when nothing has written them meanwhile. It is longer than interimEvery, so that where
// interims are written they are the words, and a held check is asked only where they are not: an
// utterance whose language is not known yet, interims off, or interims blanked as another
// language than the one the utterance was heard in. Like an interim's, it is longer past Window
// of utterance.
const heldEvery = 1100 * time.Millisecond

var stillEnds = promauto.NewCounter(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "chorus_asr_still_ends_total",
	Help:      "Turns ended with the floor free on the caller's words holding still while the speech gate held the line voiced (a noisy room).",
})

// stillWords is the utterance's words as the requests that wrote them with the floor free last
// read them, the turn-end clock of a pause the speech gate can't hear. Frames are the session's
// (Stream.Frames), talked frames are voiced frames of the utterance (Transcriber.talked).
type stillWords struct {
	utterance int64
	// words are the latest writing's, as plain words, and written says there has been one. since
	// is the frame the writing that first read them was asked at, and sinceTalked the voiced
	// frames by then; at and atTalked are the latest writing's. confirmed says a writing asked
	// after since read the same words.
	words              string
	written            bool
	since, sinceTalked int64
	at, atTalked       int64
	confirmed          bool
	// inFlight says a held check is out, asked at frame asked with askedTalked voiced frames by
	// then, and polled is the frame the agent was last asked whether it is speaking, which
	// speaking says. A held check that wrote nothing, failed or blanked, waits out heldEvery as
	// one that wrote the words does.
	inFlight    bool
	asked       int64
	askedTalked int64
	polled      int64
	speaking    bool
}

// heldCheck is one held check, asked off the session's goroutine: at frame at, with talked voiced
// frames of the utterance by then, and paused into the gate's pause, if one had begun.
type heldCheck struct {
	participant    stt.Participant
	utterance      int64
	at, talked     int64
	paused         time.Duration
	voiced, length time.Duration
	audio          harness.Audio
	asked          time.Time
}

// stillBound is how long the words must hold still under ending: stillFor, or the wait of the
// agent's point, when it chose one that lets a caller who has finished wait longer.
func stillBound(ending Ending) time.Duration {
	if ending.Point == nil {
		return stillFor
	}
	return max(stillFor, time.Duration(ending.Point.MaxWaitMs)*time.Millisecond)
}

// wroteLocked moves the clock on words of utterance written with the floor free, by a request
// asked at frame at with talked voiced frames by then. A writing asked before the latest one is
// older news, and moves nothing. The caller holds t.mu.
func (t *Transcriber) wroteLocked(utterance, at, talked int64, text string) {
	s := &t.still
	if s.utterance != utterance || at < s.at {
		return
	}
	words := strings.Join(plainWords(text), " ")
	switch {
	case !s.written || words != s.words:
		s.words, s.since, s.sinceTalked, s.confirmed = words, at, talked, false
	case at > s.since:
		s.confirmed = true
	}
	s.written, s.at, s.atTalked = true, at, talked
}

func (t *Transcriber) wrote(utterance, at, talked int64, text string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.wroteLocked(utterance, at, talked, text)
}

// stillLocked says why the words have not held still long enough, at frame at with talked voiced
// frames, for the turn to end on them, and nothing when they have: written and not none, the same
// since at least stillBound before, and the gate voiced for at least half of that time. The caller
// holds t.mu.
func (t *Transcriber) stillLocked(at, talked int64) string {
	s := &t.still
	held := at - s.since
	switch {
	case !s.written || s.words == "":
		return "no words"
	case time.Duration(held)*frame < t.stillBound:
		return "not still long enough"
	case 2*(talked-s.sinceTalked) < held:
		return "voiced under half"
	}
	return ""
}

// checksHeld asks a held check, with the floor free and the gate holding the line voiced, an
// agent of the app's asking (AskOnPauses): once nothing has written the utterance's words for
// heldEvery of speech, and as soon as the words a later writing read again have held still long
// enough to end the turn (stillLocked), no sooner than the policy's interval after the last held
// check. Nothing is asked while a pause's check, an interim or a held check is out, nor once the
// pause's checks have ended the turn or a ruling over the agent stands. It runs on the session's
// goroutine, on every frame of a voiced utterance.
func (t *Transcriber) checksHeld(participant stt.Participant, heard int64) {
	first, every, _ := t.checkTimes()
	length := heard - t.start
	if t.quiet >= first || length >= int64(maxUtterance/frame) || t.interimOut.Load() {
		return
	}
	scale := max(1, float64(length)*float64(frame)/float64(Window))
	t.mu.Lock()
	s, p := &t.still, &t.pause
	if t.words != nil {
		if text, ok := t.words.of(t.stillOf); ok {
			t.wroteLocked(t.stillOf, heard, t.talked, text)
		}
	}
	_, own := t.asker.(neutralAgent)
	if t.asker == nil || own || s.inFlight || p.inFlight || p.ended != nil || p.standing != nil || p.done {
		t.mu.Unlock()
		return
	}
	track := t.words == nil && time.Duration(t.talked-max(s.atTalked, s.askedTalked))*frame >= time.Duration(float64(heldEvery)*scale)
	end := s.confirmed && time.Duration(heard-s.asked)*frame >= every && t.stillLocked(heard, t.talked) == ""
	if !track && !end {
		t.mu.Unlock()
		return
	}
	// The agent is asked whether it is talking at most every overPoll, a lock of its own.
	if s.polled == 0 || time.Duration(heard-s.polled)*frame >= overPoll {
		asker := t.asker
		t.mu.Unlock()
		speaking := asker.Speaking()
		t.mu.Lock()
		s.polled, s.speaking = heard, speaking
	}
	if s.speaking {
		t.mu.Unlock()
		return
	}
	s.inFlight, s.asked, s.askedTalked = true, heard, t.talked
	asker := t.asker
	t.mu.Unlock()
	c := heldCheck{participant: participant, utterance: t.utterance + 1, at: heard, talked: t.talked, paused: t.quiet,
		voiced: time.Duration(t.talked) * frame, length: time.Duration(length) * frame, audio: t.heard.Last(time.Duration(length) * frame),
		asked: time.Now()}
	t.running.Add(1)
	go t.checkHeld(asker, c)
}

// newStill begins the clock of the utterance being heard. It runs on the session's goroutine, on
// the utterance's first voiced frame, before anything of it is asked for.
func (t *Transcriber) newStill() {
	t.stillOf = t.utterance + 1
	t.mu.Lock()
	t.still = stillWords{utterance: t.stillOf}
	t.mu.Unlock()
}

// checkHeld asks the server about the utterance whose words the gate held, the decision and the
// whole utterance's words in one request, as a pause's first check does (for a floor that rules
// alone, the decision, its words the other transcriber's so far), and ends the turn on its
// ruling when its words have held still long enough (stillLocked) and no pause's check has ruled
// on the pause going on: the pause's checks rule a pause the gate heard. Its words are waited for
// before anything is ended, as they say whether it may be. An agent talking by then has the
// check end nothing.
func (t *Transcriber) checkHeld(asker Asker, c heldCheck) {
	defer t.running.Done()
	defer func() {
		t.mu.Lock()
		if t.still.utterance == c.utterance {
			t.still.inFlight = false
		}
		t.mu.Unlock()
	}()
	turn := asker.Moment(c.participant)
	if turn.Speaking {
		return
	}
	turn.ParticipantID, turn.Audio, turn.Paused = c.participant.ID, c.audio, c.paused
	deadline := harness.HearDeadline(turn)
	ctx, cancel := context.WithTimeout(t.ctx, deadline)
	defer cancel()
	var said <-chan Words
	checked, err := func() (Check, error) {
		policy, err := t.client.Policy(ctx)
		if err != nil {
			return Check{}, err
		}
		agent := Agent{Instructions: turn.Instructions, Tools: turn.Tools, Languages: t.languages}
		var answer Answer
		if answer, said, err = t.decideWords(ctx, agent, turnOf(turn), turn.Audio); err != nil {
			return Check{}, err
		}
		return asker.Hearer().ruleStill(policy, answer)
	}()
	t.checked(err, deadline, c.participant)
	checks.WithLabelValues(outcome(err)).Inc()
	if err != nil {
		t.logHeld(c, checked, 0, "", err)
		return
	}
	checkSeconds.Observe(time.Since(c.asked).Seconds())
	var words Words
	select {
	case words = <-said:
	case <-t.ctx.Done():
		return
	}
	if words.Err != nil {
		transcribeFailures.WithLabelValues(failure(words.Err)).Inc()
		t.logHeld(c, checked, 0, "", words.Err)
		return
	}
	text := strings.TrimSpace(words.Text)
	t.mu.Lock()
	t.stillBound = stillBound(checked.Ending)
	s, p := &t.still, &t.pause
	var why string
	switch {
	case s.utterance != c.utterance:
		why = "utterance over"
	case words.Noise != "":
		why = "words blanked"
	}
	if why == "" {
		t.wroteLocked(c.utterance, c.at, c.talked, text)
		why = t.stillLocked(c.at, c.talked)
		switch {
		case s.since >= c.at:
			why = "words changed"
		case why == "" && (p.checks > 0 || p.inFlight || p.ended != nil || p.standing != nil || p.done):
			why = "a pause's check rules"
		}
	}
	held := time.Duration(c.at-s.since) * frame
	ends := why == ""
	epoch := p.epoch
	if ends {
		p.ended = &Ruled{Check: checked, Paused: c.paused, Utterance: c.utterance}
		p.endedAt, p.participant, p.utterance = c.at, c.participant, c.utterance
		if t.words != nil {
			// Cut at the frame before the words were first seen as they stand, so that a final
			// that brought them counts as settling them.
			cut := pauseCheck{epoch: epoch, utterance: c.utterance, participant: c.participant, length: c.length, asked: c.asked}
			t.cutLocked(cut, c.asked.Add(-held-frame))
		} else {
			p.words, p.noise = &text, words.Noise
			p.audioMs, p.wordsMs = float64(c.audio.Duration.Microseconds())/1000, float64(time.Since(c.asked).Microseconds())/1000
		}
	}
	t.mu.Unlock()
	t.logHeld(c, checked, held, why, nil)
	if ends {
		stillEnds.Inc()
		t.finish(epoch)
	}
}

// logHeld says what a held check asked and what came of it, as logCheck does a pause's, on the
// same line with held set: still_ms is how long the words had held still, and ended whether the
// turn ended on them, which is known once the words are in, or why_not why it did not.
func (t *Transcriber) logHeld(c heldCheck, checked Check, held time.Duration, why string, err error) {
	attributes := []any{"participant", c.participant.ID, "speaking", false, "held", true, "paused_ms", c.paused.Milliseconds(),
		"voiced_ms", c.voiced.Milliseconds(), "utterance_ms", c.length.Milliseconds(), "took_ms", time.Since(c.asked).Milliseconds()}
	if err != nil {
		t.logger.Debug("chorus-asr did not rule on a check", append(attributes, "error", err)...)
		return
	}
	var pEOT any
	if checked.Answer.PEOT != nil {
		pEOT = *checked.Answer.PEOT
	}
	t.logger.Debug("chorus-asr checked", append(attributes, "answer", checked.Answer.Answer, "p_eot", pEOT, "still_ms", held.Milliseconds(),
		"ended", why == "", "why_not", why, "disposition", checked.Ruling.Disposition, "floor", checked.Ruling.Floor)...)
}
