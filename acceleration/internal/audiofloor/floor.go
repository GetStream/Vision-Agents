package audiofloor

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// finalWait bounds how long an utterance the floor ended waits for the final of the run of
// speech the transcriber writing its words still had open: a final asked for (Finalizer)
// comes in a few hundred milliseconds, and one heard at the run's own end within a second in
// most calls. Past it the utterance ends on the words of that run as they stand.
const finalWait = time.Second

var floorWords = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "chorus_asr_floor_words_total",
	Help: "Utterances the chorus-asr floor ended on another transcriber's words, by how those words were settled: " +
		"settled (the transcriber had written its final since the voice stopped), final (its final came within the wait), " +
		"run (it began another run first), partial (no final within the wait: the words as they stood) or none (it wrote nothing).",
}, []string{"settled"})

// NewFloor returns an unstarted transcriber that rules on the floor alone, for a participant
// whose words another transcriber writes. It hears their audio and asks about each of their
// pauses as a Transcriber does, for the agent that has it ask (AskOnPauses), and over the
// agent as well, so that when a turn ends, and what is done about the floor over the agent,
// are chorus-asr's from the audio whatever writes the words. The server writes none: the
// decisions are asked alone, and the words are the other transcriber's, handed over as they
// come (Heard) and shown meanwhile as interims, whichever kind they are, so that nothing is
// read into how they hold still. A check that ends an utterance asks that transcriber for its
// final (finalize, where it can be told: stt.Finalizer), and the utterance's words are its
// finals since the last utterance ended, the one it was asked for or heard last included,
// waited for up to finalWait. A run of speech the other transcriber ends on its own is a run,
// not a turn: its words wait for the floor's end.
//
// A pause no check can rule on ends at the other transcriber's own end of a run, once the
// voice has paused, or at endAfter. A server that will never rule (Misconfigured), or that
// answers none of failoverAfter checks in a row over failoverWindow (Unreachable), is down for
// the call: the floor steps aside, says so on its events (stt.Disconnected), and takes no more
// words (Heard), which are then the listener's, as they would be with no floor.
func NewFloor(client *Client, languages []string, finalize func() (bool, error), logger *slog.Logger) *Transcriber {
	t := NewTranscriber(client, DefaultModel, languages, logger)
	t.words, t.finalize, t.interims = &spoken{utterance: 1, logger: logger}, finalize, false
	return t
}

// Heard takes a transcript of the participant's audio from the transcriber that writes their
// words, for a floor that rules alone (NewFloor), and says whether it took it: once the floor
// has stepped aside it takes none. What the floor makes of the words goes out on Events.
func (t *Transcriber) Heard(transcript stt.Transcript) bool {
	if t.words == nil || t.stepped.Load() {
		return false
	}
	interim, show, force := t.words.heard(transcript)
	if force {
		t.force()
	}
	if show && t.sent.over(interim.Utterance-1) {
		t.send(interim)
	}
	return true
}

// FailedOver tells a floor that rules alone that the transcriber writing the words has handed
// the session to another, which numbers its runs afresh: the run the last one had open is
// over, its words as they stand.
func (t *Transcriber) FailedOver() {
	if t.words != nil {
		t.words.failedOver()
	}
}

// force asks the transcriber writing the words for the final of the run it has open.
func (t *Transcriber) force() {
	if t.finalize == nil {
		return
	}
	asked, err := t.finalize()
	switch {
	case err != nil:
		t.logger.Debug("the transcriber writing the words was not told to settle them", "error", err)
	case !asked:
		t.logger.Debug("the transcriber writing the words can't be told to settle them, its final is waited for")
	}
}

// cutLocked ends the words of the utterance check c ended, for a floor that rules alone, at
// the voice's last frame, stopped: they go out as its final once the transcriber writing them
// has settled them (settle). t.mu is held.
func (t *Transcriber) cutLocked(c pauseCheck, stopped time.Time) {
	said, force := t.words.end(c.utterance, stopped)
	t.running.Add(1)
	go t.settle(c, said, force)
}

// settle keeps the words of the utterance check c ended, once the transcriber writing them
// has settled them, and sends them as its final with the check's ruling (finish).
func (t *Transcriber) settle(c pauseCheck, said <-chan Words, force bool) {
	defer t.running.Done()
	if force {
		t.force()
	}
	var words Words
	select {
	case words = <-said:
	case <-t.ctx.Done():
		return
	}
	text := strings.TrimSpace(words.Text)
	t.mu.Lock()
	if p := &t.pause; p.epoch == c.epoch && p.utterance == c.utterance {
		p.words, p.audioMs, p.wordsMs = &text, milliseconds(c.length), milliseconds(time.Since(c.asked))
	}
	t.mu.Unlock()
	t.finish(c.epoch)
}

// wordsEnded says the pause going on is the end of a run the transcriber writing the words
// settled, for a floor that rules alone whose checks can't rule on it: the voice has paused,
// and that transcriber has written its final since.
func (t *Transcriber) wordsEnded() bool {
	return t.words != nil && t.quiet >= writeAfter && t.words.settledSince(time.Now().Add(-t.quiet))
}

// endOnWords ends the utterance being heard with no check's ruling, for a floor that rules
// alone, and sends its words as its final once the transcriber writing them has settled them.
// It runs on the session's goroutine.
func (t *Transcriber) endOnWords(participant stt.Participant, heard int64) {
	utterance, length, asked := t.utterance+1, time.Duration(heard-t.start)*frame, time.Now()
	said, force := t.words.end(utterance, asked.Add(-t.quiet))
	t.running.Add(1)
	go func() {
		defer t.running.Done()
		if force {
			t.force()
		}
		var words Words
		select {
		case words = <-said:
		case <-t.ctx.Done():
			return
		}
		text := strings.TrimSpace(words.Text)
		if text == "" {
			t.sent.ended(utterance)
			return
		}
		provider, model := t.writer()
		t.send(stt.Transcript{
			Participant:      participant,
			Mode:             stt.ModeFinal,
			Utterance:        utterance,
			Text:             text,
			Language:         words.Language,
			Provider:         provider,
			Model:            model,
			ProcessingTimeMs: milliseconds(time.Since(asked)),
			AudioDurationMs:  milliseconds(length),
		})
	}()
}

// writer is the transcriber the session's words are written by: the server, or, for a floor
// that rules alone, the one that wrote the last of them.
func (t *Transcriber) writer() (provider, model string) {
	if t.words == nil {
		return ProviderName, t.model
	}
	return t.words.writer()
}

// decideWords asks for a decision about audio and its words in one request, or, for a floor
// that rules alone, for the decision alone, the words being the other transcriber's so far
// once it is made.
func (t *Transcriber) decideWords(ctx context.Context, agent Agent, turn Turn, audio harness.Audio) (Answer, <-chan Words, error) {
	if t.words == nil {
		return t.client.DecideWords(ctx, agent, turn, audio)
	}
	answer, err := t.client.Decide(ctx, agent, turn, audio)
	if err != nil {
		return Answer{}, nil, err
	}
	said := make(chan Words, 1)
	said <- t.words.soFar()
	return answer, said, nil
}

// checked counts a check of a floor that rules alone, which err says the server did not rule
// on, and has the floor step aside once the server is down for the call: it will never rule
// (Misconfigured), or it answered none of failoverAfter checks in a row, the last at least
// failoverWindow after the first. A check answered starts the count again.
func (t *Transcriber) checked(err error) {
	if t.words == nil || errors.Is(err, context.Canceled) {
		return
	}
	t.mu.Lock()
	switch {
	case err == nil, !Misconfigured(err) && !Unreachable(err):
		t.unruled = 0
		t.mu.Unlock()
		return
	case Unreachable(err):
		if t.unruled++; t.unruled == 1 {
			t.unruledSince = t.now()
		}
		if t.unruled < failoverAfter || t.now().Sub(t.unruledSince) < failoverWindow {
			t.mu.Unlock()
			return
		}
		err = fmt.Errorf("audiofloor: the server ruled on none of %d checks in %s, the last: %w",
			t.unruled, t.now().Sub(t.unruledSince).Round(time.Second), err)
	}
	t.mu.Unlock()
	if !t.stepped.CompareAndSwap(false, true) {
		return
	}
	t.logger.Warn("the chorus-asr server can't rule on this participant's floor, so the transcriber writing the words ends their turns for the rest of the session",
		"error", err)
	t.emitter.Send(stt.Disconnected{Provider: ProviderName, Model: t.model, Reason: stack.Wrap(err).Error(), At: time.Now()})
}

func milliseconds(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// spoken is what another transcriber has written of a participant's utterance being heard,
// for a floor that rules alone: its finals since the floor's last cut, each the words of a run
// of speech it settled, and its revision of the run it has not. It is written on the
// listener's goroutine (Heard) and read on the session's and the checks'.
type spoken struct {
	logger *slog.Logger

	mu sync.Mutex
	// utterance is the floor's utterance the words are of, numbered as its finals are, and
	// began when its first words came.
	utterance int64
	began     time.Time
	settled   []settledRun
	// open is the revision of the run not settled, numbered run (stt.Transcript.Utterance),
	// and finalAt when the last final came.
	open    string
	run     int64
	finalAt time.Time
	// language is the language the words were last said to be in, and provider and model the
	// transcriber that wrote them.
	language        string
	provider, model string
	// cut is the utterance the floor ended, waiting for its words, if one is.
	cut *cut
	// spent is a run whose final the wait gave up on: the utterance ended on its words as they
	// stood, and what more comes of it is that utterance's, which is over.
	spent int64
}

// settledRun is the words of a run the transcriber settled, and the run's number, none when
// zero.
type settledRun struct {
	run  int64
	text string
}

// cut is an utterance the floor ended, waiting for its words: the final of run, the run the
// transcriber had open when it ended (none: the first it writes after), or of any run it ends
// before it.
type cut struct {
	utterance int64
	run       int64
	at        time.Time
	forced    bool
	said      chan Words
	timer     *time.Timer
}

// heard keeps transcript, and says what to show of the utterance's words meanwhile, and
// whether the transcriber is to be asked for its final: an utterance the floor ended with
// nothing written of it, whose run has now begun.
func (w *spoken) heard(transcript stt.Transcript) (interim stt.Transcript, show, force bool) {
	text := strings.TrimSpace(transcript.Text)
	run := transcript.Utterance
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.spent != 0 {
		if run == w.spent {
			w.logger.Debug("words of a run the floor ended an utterance in, after it gave up waiting for its final, are dropped",
				"provider", transcript.Provider, "run", run, "mode", transcript.Mode, "text", text)
			return stt.Transcript{}, false, false
		}
		w.spent = 0
	}
	if c := w.cut; c != nil && c.run != 0 && run != 0 && run != c.run {
		// Another run began before the one the cut was in settled: that one is over, its
		// words as they stand, and these are the next utterance's.
		w.deliverLocked(c, "run")
	}
	if c := w.cut; c != nil && c.run == 0 && run != 0 {
		c.run = run
		force = !transcript.Final() && !c.forced
		c.forced = c.forced || force
	}
	w.provider, w.model = transcript.Provider, transcript.Model
	if transcript.Language != "" {
		w.language = transcript.Language
	}
	if w.began.IsZero() && text != "" {
		w.began = time.Now()
	}
	if last := len(w.settled) - 1; last >= 0 && run != 0 && w.settled[last].run == run && w.open == "" {
		// The run its last final settled carries on, restated whole, as a transcriber that took
		// the end of a run back does, or its final came again: these are its words now.
		w.settled = w.settled[:last]
	}
	switch transcript.Mode {
	case stt.ModeFinal:
		if text != "" {
			w.settled = append(w.settled, settledRun{run: run, text: text})
		}
		w.open, w.run, w.finalAt = "", 0, time.Now()
	case stt.ModeDelta:
		w.open, w.run = w.open+transcript.Text, run
	default:
		if w.open != "" && w.run != 0 && run != 0 && run != w.run {
			// A run that ended without a final: its words as they stood.
			w.settled = append(w.settled, settledRun{run: w.run, text: strings.TrimSpace(w.open)})
		}
		w.open, w.run = text, run
	}
	if c := w.cut; c != nil && transcript.Final() && (c.run == 0 || run == 0 || run == c.run) {
		w.deliverLocked(c, "final")
		return stt.Transcript{}, false, force
	}
	said := w.textLocked()
	if said == "" {
		return stt.Transcript{}, false, force
	}
	return stt.Transcript{
		Participant:     transcript.Participant,
		Mode:            stt.ModeReplacement,
		Interim:         true,
		Utterance:       w.utterance,
		Text:            said,
		Language:        w.language,
		Provider:        w.provider,
		Model:           w.model,
		AudioDurationMs: milliseconds(time.Since(w.began)),
	}, true, force
}

// end ends the words of utterance, which the floor ended at a pause the voice stopped for at
// stopped: at once if the transcriber has written its final since and has nothing open, and
// otherwise once it writes the final of the run it has open, or of the first it writes, up to
// finalWait. It says whether the transcriber is to be asked for that final: it has a run open.
func (w *spoken) end(utterance int64, stopped time.Time) (<-chan Words, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if c := w.cut; c != nil {
		// The floor ended the utterance before at endAfter, with no check's ruling, and this
		// one before that one's words came: the words are this one's, which goes on from it.
		c.timer.Stop()
		w.cut = nil
		c.said <- Words{}
	}
	c := &cut{utterance: utterance, at: time.Now(), said: make(chan Words, 1)}
	if strings.TrimSpace(w.open) == "" && w.finalAt.After(stopped) {
		w.deliverLocked(c, "settled")
		return c.said, false
	}
	if strings.TrimSpace(w.open) != "" {
		c.run, c.forced = w.run, true
	}
	w.cut = c
	c.timer = time.AfterFunc(finalWait, func() { w.giveUp(c) })
	return c.said, c.forced
}

// giveUp ends the words of an utterance whose final did not come within finalWait, on the
// words as they stand.
func (w *spoken) giveUp(c *cut) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.cut != c {
		return
	}
	settled := "none"
	if strings.TrimSpace(w.open) != "" {
		w.spent, settled = w.run, "partial"
	}
	w.deliverLocked(c, settled)
}

// deliverLocked hands c its words, the words gathered so far, and begins the next utterance's.
// w.mu is held.
func (w *spoken) deliverLocked(c *cut, settled string) {
	said := Words{Text: w.textLocked(), Language: w.language, Ms: milliseconds(time.Since(c.at))}
	if said.Text == "" && settled != "settled" {
		settled = "none"
	}
	floorWords.WithLabelValues(settled).Inc()
	w.logger.Debug("the floor's utterance has the words of the transcriber that writes them",
		"utterance", c.utterance, "provider", w.provider, "model", w.model, "settled", settled, "asked_final", c.forced,
		"waited_ms", int64(said.Ms))
	if c.timer != nil {
		c.timer.Stop()
	}
	w.settled, w.open, w.run, w.began = nil, "", 0, time.Time{}
	w.cut, w.utterance = nil, c.utterance+1
	c.said <- said
}

// textLocked is the words gathered so far. w.mu is held.
func (w *spoken) textLocked() string {
	texts := make([]string, 0, len(w.settled)+1)
	for _, settled := range w.settled {
		texts = append(texts, settled.text)
	}
	return strings.TrimSpace(strings.Join(append(texts, w.open), " "))
}

// soFar is the words of the utterance being heard as they stand.
func (w *spoken) soFar() Words {
	w.mu.Lock()
	defer w.mu.Unlock()
	return Words{Text: w.textLocked(), Language: w.language}
}

// of is the words so far of utterance, and whether they are its: none are while an utterance
// the floor ended waits for its words.
func (w *spoken) of(utterance int64) (string, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.cut != nil || w.utterance != utterance {
		return "", false
	}
	return w.textLocked(), true
}

// settledSince says the transcriber has written its final since at, and has nothing open.
func (w *spoken) settledSince(at time.Time) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return strings.TrimSpace(w.open) == "" && len(w.settled) > 0 && w.finalAt.After(at)
}

// writer is the transcriber that wrote the last words, the server until one has.
func (w *spoken) writer() (provider, model string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.provider == "" {
		return ProviderName, DefaultModel
	}
	return w.provider, w.model
}

// failedOver ends the run the transcriber that handed the session over had open, its words as
// they stand: the next numbers its runs afresh.
func (w *spoken) failedOver() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if open := strings.TrimSpace(w.open); open != "" {
		w.settled = append(w.settled, settledRun{text: open})
	}
	w.open, w.run, w.spent = "", 0, 0
	if c := w.cut; c != nil && c.run != 0 {
		w.deliverLocked(c, "run")
	}
}
