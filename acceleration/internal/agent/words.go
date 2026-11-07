package agent

import (
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// wordFate is what became of words a caller said: answered, or set aside for a reason.
type wordFate string

// The fates a caller's words meet (router_caller_words_total). Each is one place in the loop
// that answers words or sets them aside, and lost is words that met none of them.
const (
	// fateAnswered went to the reply model in a turn, or in the note of one (keepTurnLocked,
	// forgetQueuedAnsweredOver).
	fateAnswered wordFate = "answered"
	// fateEcho is the agent's own voice heard back on the caller's line (withoutEcho).
	fateEcho wordFate = "echo"
	// fateIgnored is the flow controller's ignore: background speech, or meant for somebody
	// else.
	fateIgnored wordFate = "ignored"
	// fateBackchannel is the flow controller's ignore of a murmur or of assent said over the
	// agent ("uh-huh", "okay"), which takes no floor (takesFloor).
	fateBackchannel wordFate = "backchannel"
	// fateNoise is a cough or other non-speech heard over the reply (overlapNoise).
	fateNoise wordFate = "noise"
	// fateStop is a stop's words, spent on the stop that silenced the agent (stopSpent).
	fateStop wordFate = "stop"
	// fateLanguage is words a check over the agent wrote of an utterance the transcriber
	// then heard in a language the callers don't speak, which no final carries.
	fateLanguage wordFate = "language"
	// fateRepeated is a final saying again what the agent already dealt with, which the
	// transcriber sends as it settles (cadence.restating).
	fateRepeated wordFate = "repeated"
	// fateReplaced is a turn queued behind a reply that a later one took the place of
	// (converse.hold): only the last is kept.
	fateReplaced wordFate = "replaced"
	// fateUnwritten is words a check over the agent wrote of an utterance whose final the
	// transcriber never wrote: its server was behind or failed, or it closed first.
	fateUnwritten wordFate = "unwritten"
	// fateCallEnded is words still on their way through the loop when the call ended.
	fateCallEnded wordFate = "call_ended"
	// fateLost is words that were neither answered nor set aside for a reason. It should
	// never happen, and is warned about when it does.
	fateLost wordFate = "lost"
)

// wordsFated is what became of some of a caller's words: those of one utterance, or of the
// part of it the fate is about.
type wordsFated struct {
	Participant string
	Utterance   int64
	Text        string
	Words       int
	Fate        wordFate
	Reason      string
}

// wordLedger holds the loop to its account of a caller's words. Every word of a caller's that
// the transcriber delivered in a final reaches the reply model in a turn, or is set aside for
// a reason that is named, logged and counted. Words that a check over the agent wrote, and
// the agent acted on, are held to it too, until the utterance's final carries them: an
// utterance that ended with no final would otherwise take them out of the loop unseen.
//
// The loop decides; the ledger only counts what was decided. Each place that answers words or
// sets them aside says so, with the words it acted on, and the ledger closes the caller's
// words those cover. Two reasons are checked against what happened, as they were given on
// words that should have been answered: a stop's words are spent only on a stop that silenced
// the agent, begun once the caller had heard what it stopped, and saying nothing more than
// it; and echo is only in audio the agent's voice could be in. Words with no reason, or with
// one that does not hold, are lost: counted, and warned about.
//
// It never calls into the agent, the conversation or the cadence: they call it, some under
// their own locks.
type wordLedger struct {
	logger *slog.Logger
	// observe is told every fate as it is met, for a test to hold a call to the account
	// (wordTrail). Nil in production.
	observe func(wordsFated)

	mu sync.Mutex
	// pending is each participant's words not yet answered or set aside, oldest first.
	pending map[string][]pendingWords
	// stops is the stop over the agent each participant's utterance silenced the agent on.
	stops map[stopKey]stopEvidence
	// last is the utterance of each participant's whose words were last dealt with, which
	// a transcriber may send again or grow.
	last map[string]pendingWords
	// ending says the call is ending: words that arrive from now on are past answering.
	ending bool
}

// pendingWords are words of a participant's utterance not yet answered or set aside.
type pendingWords struct {
	utterance int64
	// text is what of the utterance is owed: its final, without the start of it the agent
	// already dealt with (answeredPrefix). full is the final whole.
	text, full string
	// overheard says the words are what a check over the agent wrote, and no final has
	// come for the utterance yet.
	overheard bool
	// late says they came once the call was ending.
	late bool
}

type stopKey struct {
	participant string
	utterance   int64
}

// stopEvidence is a stop over the agent as it was carried out: the words it was ruled on,
// whether the caller had heard any of the agent's answer when they began them
// (answerHeardBefore), and whether it left the agent saying nothing.
type stopEvidence struct {
	words    string
	heard    bool
	silenced bool
}

func newWordLedger(logger *slog.Logger) *wordLedger {
	return &wordLedger{
		logger:  logger,
		pending: map[string][]pendingWords{},
		stops:   map[stopKey]stopEvidence{},
		last:    map[string]pendingWords{},
	}
}

// final opens the words of a participant's final, without the agent's echo. A final of an
// utterance that already has words open replaces them, as the cadence does: a numbered one
// by its number, and one with no number when it is a revision of them.
func (l *wordLedger) final(participant string, utterance int64, text string) {
	if l == nil {
		return
	}
	if len(wordList(text)) == 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	owed := text
	if last := l.last[participant]; utterance != 0 && last.utterance == utterance {
		// The agent dealt with the start of this utterance, and the rest is what is owed,
		// as the turn on it says (candidate.Said).
		if answered := answeredPrefix(last.full, text); answered != "" {
			owed = (candidate{Text: text, Answered: answered}).Said()
		}
	}
	words := pendingWords{utterance: utterance, text: owed, full: text, late: l.ending}
	open := l.pending[participant]
	if i := slices.IndexFunc(open, func(p pendingWords) bool {
		if utterance != 0 {
			return p.utterance == utterance
		}
		return p.utterance == 0 && !p.overheard && revisesTranscript(p.full, text)
	}); i >= 0 {
		open[i] = words
		return
	}
	l.pending[participant] = append(open, words)
}

// overheard keeps what a check over the agent wrote of an utterance, once the agent acted on
// a check of it, until the utterance's final, and the stop it was carried out as, if it was
// one. The first stop to silence the agent in an utterance is the one its words are spent on:
// checks after it, with nothing left to stop, are no stop of theirs.
func (l *wordLedger) overheard(participant string, utterance int64, text string, acted bool, stop *stopEvidence) {
	if l == nil || utterance == 0 || len(wordList(text)) == 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if key := (stopKey{participant, utterance}); stop != nil {
		if _, known := l.stops[key]; !known {
			l.stops[key] = *stop
		}
	}
	open := l.pending[participant]
	i := slices.IndexFunc(open, func(p pendingWords) bool { return p.utterance == utterance })
	switch {
	case i < 0 && acted:
		l.pending[participant] = append(open, pendingWords{utterance: utterance, text: text, full: text, overheard: true, late: l.ending})
	case i < 0:
	case open[i].overheard:
		open[i].text, open[i].full = text, text
	}
}

// echoed counts words of a participant's final taken for the agent's own voice heard back:
// n of them, of heard, matched in line, what the agent was saying and last said. Echo is the
// agent's voice in the caller's microphone, so it can only be in audio that came while the
// voice was going out, from to to, or within echoDelay of it: words taken for it from audio
// from began to ended outside that are the caller's, and lost.
func (l *wordLedger) echoed(participant string, utterance int64, heard string, n int, line string, began, ended, from, to time.Time) {
	if l == nil || n <= 0 {
		return
	}
	fated := wordsFated{Participant: participant, Utterance: utterance, Text: heard, Words: n, Fate: fateEcho,
		Reason: fmt.Sprintf("the agent's own voice heard back on the caller's line, matched in %q", preview(line))}
	switch {
	case from.IsZero():
		fated.Fate, fated.Reason = fateLost, "taken for the agent's echo, though none of its voice had gone out"
	case ended.Before(from):
		fated.Fate, fated.Reason = fateLost, fmt.Sprintf("taken for the agent's echo from audio that ended %s before its voice went out",
			from.Sub(ended).Round(time.Millisecond))
	case began.After(to.Add(echoDelay)):
		fated.Fate, fated.Reason = fateLost, fmt.Sprintf("taken for the agent's echo from audio that began %s after its voice last went out, past the %s an echo takes",
			began.Sub(to).Round(time.Millisecond), echoDelay)
	}
	l.settle([]wordsFated{fated})
}

// answered closes the participant's words that text, a turn the reply model was given, says.
func (l *wordLedger) answered(participant, text string) {
	l.setAside(participant, text, fateAnswered, "")
}

// setAside closes the participant's words that text says, for reason.
func (l *wordLedger) setAside(participant, text string, fate wordFate, reason string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	fates := l.closeLocked(participant, func(p pendingWords) bool { return !p.overheard && containsWords(text, p.text) }, fate, reason)
	l.mu.Unlock()
	l.settle(fates)
}

// spent closes a settled turn set aside as the words of a stop over the agent (stopSpent):
// spent on the stop, if one silenced the agent in that utterance, begun once the caller had
// heard some of what it stopped, and text says nothing more than it was ruled on. Otherwise
// the words were the caller's turn, and are lost.
func (l *wordLedger) spent(participant string, utterance int64, text string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	key := stopKey{participant, utterance}
	stop, known := l.stops[key]
	delete(l.stops, key)
	fate, reason := spentOn(stop, known, text)
	fates := l.closeLocked(participant, func(p pendingWords) bool { return !p.overheard && containsWords(text, p.text) }, fate, reason)
	l.mu.Unlock()
	l.settle(fates)
}

// spentOn is the fate of an utterance's words, text, set aside as spent on stop.
func spentOn(stop stopEvidence, known bool, text string) (wordFate, string) {
	switch {
	case !known:
		return fateLost, "set aside as a stop's words, though no stop over the agent silenced it in this utterance"
	case !stop.heard:
		return fateLost, "set aside as a stop's words, though the caller began them before any of the agent's answer went out: they went on with their own turn"
	case !stop.silenced:
		return fateLost, "set aside as a stop's words, though the stop left the agent talking"
	case saysMore(text, stop.words):
		return fateLost, fmt.Sprintf("set aside as a stop's words, though they say more than the stop that silenced the agent: %q", stop.words)
	}
	return fateStop, "a stop's words, spent on silencing the agent"
}

// repeated closes a final that says again what the agent already dealt with.
func (l *wordLedger) repeated(participant string, utterance int64, text string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	fates := l.closeLocked(participant, func(p pendingWords) bool {
		return !p.overheard && p.utterance == utterance && sameWords(p.full, text)
	}, fateRepeated, "the transcriber sent words the agent already dealt with again")
	l.mu.Unlock()
	l.settle(fates)
}

// unwritten closes what a check over the agent wrote of an utterance the transcriber ended
// with no final, why being its reason (audiofloor.Unwritten): "language" for words in a
// language the callers don't speak, "behind" or "failed" for a final the server never wrote,
// and empty for none written. A stop's words are spent on the stop, as its final's would be;
// any other words are lost.
func (l *wordLedger) unwritten(participant string, utterance int64, why string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	key := stopKey{participant, utterance}
	stop, known := l.stops[key]
	delete(l.stops, key)
	written := func(p pendingWords) bool { return p.overheard && p.utterance == utterance }
	i := slices.IndexFunc(l.pending[participant], written)
	if i < 0 {
		l.mu.Unlock()
		return
	}
	fate, reason := spentOn(stop, known, l.pending[participant][i].text)
	switch {
	case why == "language":
		fate, reason = fateLanguage, "the transcriber heard the utterance in a language the callers don't speak, and wrote no final"
	case fate == fateStop:
	case why != "":
		fate, reason = fateUnwritten, "the transcriber wrote no final of the utterance: "+why
	default:
		fate, reason = fateLost, "the utterance ended with no words, and no final carried those a check over the agent wrote"
	}
	fates := l.closeLocked(participant, written, fate, reason)
	l.mu.Unlock()
	l.settle(fates)
}

// closed lets go of what checks over the agent wrote of a participant's utterances once the
// transcriber that heard them has closed or been replaced, and of the stops ruled in them: the
// next numbers its utterances afresh, and no final will carry those words.
func (l *wordLedger) closed(participant string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	for key := range l.stops {
		if key.participant == participant {
			delete(l.stops, key)
		}
	}
	fates := l.closeLocked(participant, func(p pendingWords) bool { return p.overheard }, fateUnwritten,
		"the transcriber that heard them closed before writing the utterance's final")
	l.mu.Unlock()
	l.settle(fates)
}

// closing marks the call as ending: words from now on are past answering.
func (l *wordLedger) closing() {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.ending = true
}

// end closes every word still open as the call ends. Words the loop holds, settling,
// before the flow controller or queued behind a reply (holds), and words that came once the
// call was ending, are the call's to have ended on. Words the loop holds nowhere were dropped
// on the way with no reason, and are lost.
func (l *wordLedger) end(holds func(participant, text string) bool) {
	if l == nil {
		return
	}
	type open struct {
		participant string
		words       pendingWords
	}
	var still []open
	l.mu.Lock()
	for participant, pending := range l.pending {
		for _, p := range pending {
			still = append(still, open{participant, p})
		}
	}
	l.pending = map[string][]pendingWords{}
	clear(l.stops)
	l.mu.Unlock()
	slices.SortFunc(still, func(a, b open) int { return strings.Compare(a.participant, b.participant) })

	fates := make([]wordsFated, 0, len(still))
	for _, o := range still {
		fated := wordsFated{Participant: o.participant, Utterance: o.words.utterance, Text: o.words.text,
			Words: len(wordList(o.words.text)), Fate: fateCallEnded, Reason: "the call ended while they were on their way"}
		switch {
		case o.words.late:
			fated.Reason = "they came as the call was ending"
		case o.words.overheard:
			fated.Reason = "the call ended before the utterance's final"
		case holds == nil || !holds(o.participant, o.words.text):
			fated.Fate, fated.Reason = fateLost, "the loop held them nowhere when the call ended: they were dropped with no reason"
		}
		fates = append(fates, fated)
	}
	l.settle(fates)
}

// closeLocked closes the participant's open words that match, as fate for reason, and says
// what it closed. The caller holds the lock.
func (l *wordLedger) closeLocked(participant string, match func(pendingWords) bool, fate wordFate, reason string) []wordsFated {
	var fates []wordsFated
	l.pending[participant] = slices.DeleteFunc(l.pending[participant], func(p pendingWords) bool {
		if !match(p) {
			return false
		}
		fates = append(fates, wordsFated{Participant: participant, Utterance: p.utterance, Text: p.text,
			Words: len(wordList(p.text)), Fate: fate, Reason: reason})
		if p.utterance != 0 && !p.overheard {
			// The utterance's final is dealt with, and with it any stop ruled in it.
			l.last[participant] = p
			delete(l.stops, stopKey{participant, p.utterance})
		}
		return true
	})
	if len(l.pending[participant]) == 0 {
		delete(l.pending, participant)
	}
	return fates
}

// settle counts and reports fates met, outside the lock.
func (l *wordLedger) settle(fates []wordsFated) {
	for _, fated := range fates {
		if fated.Words == 0 {
			continue
		}
		callerWords.WithLabelValues(string(fated.Fate)).Add(float64(fated.Words))
		switch fated.Fate {
		case fateAnswered:
		case fateLost:
			l.logger.Warn("a caller's words were lost, neither answered nor set aside for a reason",
				"participant", fated.Participant, "utterance", fated.Utterance, "words", fated.Words,
				"preview", preview(fated.Text), "why", fated.Reason)
		default:
			l.logger.Debug("a caller's words were set aside",
				"fate", fated.Fate, "reason", fated.Reason, "participant", fated.Participant,
				"utterance", fated.Utterance, "words", fated.Words, "preview", preview(fated.Text))
		}
		if l.observe != nil {
			l.observe(fated)
		}
	}
}

// containsWords reports whether text says all of part's words, in order and together.
func containsWords(text, part string) bool {
	said, owed := wordList(text), wordList(part)
	if len(owed) == 0 {
		return true
	}
	for i := 0; i+len(owed) <= len(said); i++ {
		if slices.Equal(said[i:i+len(owed)], owed) {
			return true
		}
	}
	return false
}

// previewWords is how many of a caller's words a log line shows of them.
const previewWords = 8

// preview is the first few words of text, for a log line.
func preview(text string) string {
	fields := strings.Fields(text)
	if len(fields) <= previewWords {
		return text
	}
	return strings.Join(fields[:previewWords], " ") + " …"
}

// ignoredAs is the fate of words the loop ignored: noise over the reply (overlapNoise); a murmur or assent, which takes no floor; or anything else ruled background
// or meant for somebody else. It names the fate for the count, and decides nothing.
func ignoredAs(text string, noise bool) wordFate {
	switch {
	case noise:
		return fateNoise
	case !takesFloor(text):
		return fateBackchannel
	}
	return fateIgnored
}

// echoedWords counts echo words of a final taken for the agent's own voice heard back
// (droppingEcho), with what the agent was saying and when its voice was going out, which say
// whether they could be its echo.
func (a *Agent) echoedWords(transcript stt.Transcript, echo int) {
	a.mu.Lock()
	from, to := a.voiceOut.from, a.voiceOut.to
	line := strings.TrimSpace(a.saying + " " + lastAssistantSaid(a.history))
	a.mu.Unlock()
	// As withoutEcho times them: the audio ended as the transcriber began writing the words,
	// and began as long before as it lasts.
	ended := time.Now().Add(-time.Duration(transcript.ProcessingTimeMs * float64(time.Millisecond)))
	began := ended.Add(-time.Duration(transcript.AudioDurationMs * float64(time.Millisecond)))
	a.words.echoed(transcript.Participant.ID, transcript.Utterance, transcript.Text, echo, line, began, ended, from, to)
}

// overheardWords keeps what a check over the agent wrote of an utterance, once the agent
// acted on it, until its final, and the stop it was, if it silenced the agent: heard says the
// caller had heard the agent's answer when they began, and abandoned that acting on it
// abandoned a reply. A stop silences the agent when nothing it is saying is left playing but
// a murmur.
func (a *Agent) overheardWords(participant string, utterance int64, text string, actions []Action, heard, abandoned bool) {
	acted := len(actions) > 0
	var stop *stopEvidence
	if abandoned && slices.ContainsFunc(actions, func(action Action) bool { return action.Kind == ActInterrupt }) {
		stop = &stopEvidence{words: text, heard: heard, silenced: !a.stillSaying()}
	}
	a.words.overheard(participant, utterance, text, acted, stop)
}

// stillSaying says some turn the agent has not abandoned is still being said, other than a
// murmur.
func (a *Agent) stillSaying() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, heard := range a.heard {
		if _, gone := a.abandoned[heard.turnID]; !gone && !strings.HasPrefix(heard.turnID, backchannelPrefix) {
			return true
		}
	}
	return false
}

// holds says whether the participant's words, text, are still in the loop: being settled,
// before the flow controller, or queued behind a reply.
func (c *converse) holds(participant, text string) bool {
	c.mu.Lock()
	var kept []string
	for _, ready := range c.candidates {
		if ready.Participant.ID == participant {
			kept = append(kept, ready.Text)
		}
	}
	if c.queued != nil && c.queued.candidate.Participant.ID == participant {
		kept = append(kept, c.queued.candidate.Text)
	}
	c.mu.Unlock()
	kept = append(kept, c.cadence.gathering(participant)...)
	return slices.ContainsFunc(kept, func(words string) bool { return containsWords(words, text) })
}

// gathering is what the cadence holds of a participant's words: those being settled, and a
// turn held for a queue.
func (c *cadence) gathering(participant string) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	current, ok := c.speakers[participant]
	if !ok {
		return nil
	}
	kept := []string{current.text}
	if current.held != nil {
		kept = append(kept, current.held.candidate.Text)
	}
	return kept
}
