package audiofloor

import (
	"context"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// checkGrace is how long past the policy's cap a pause's checks have to end it before the
// pause ends as it would with no agent: on its words alone.
const checkGrace = time.Second

// Asker is an agent whose transcriber asks the chorus-asr server about each of the caller's
// pauses on its behalf (Pauses), timed from the audio: the first check of a pause asks for
// the decision and the words in one pass, so a pause costs one request, and the turn ends
// when the model says so, counted from the voice's last frame rather than from the words'
// last change. The words of an utterance that ended go out as its one final, and there are
// no others: words of a pause the model waited on are kept for its end.
type Asker interface {
	// Moment is the call as participant's words would be put to the flow controller now:
	// what the agent was told, its customer's tools, the conversation and what the agent is
	// saying. The audio and the pause are the transcriber's to fill in.
	Moment(participant stt.Participant) harness.FlowTurn
	// Ruled is a check's ruling on an utterance that ended, with its words, handed over just
	// before they go out as the utterance's final.
	Ruled(participant stt.Participant, ruled Ruled)
	// Hearer is the agent's, whose choice of how to end a turn (EndOfTurn) reads each check:
	// the point's bars, and how many checks in a row must reach them.
	Hearer() *Hearer
	// Speaking says the agent is talking now: a caller voiced over it is asked about without
	// waiting for a pause (checksOver), as a pause may never come over the agent's own echo
	// or the room.
	Speaking() bool
	// Overheard is a check's ruling over the agent to stop or shorten, asked while the
	// caller talked on, with the words so far: the agent acts on it at once, unless the
	// words are its own voice heard back, which is no reason to stop. The utterance goes on,
	// and ends by its pause checks.
	Overheard(participant stt.Participant, ruled Ruled)
	// Hearing says the caller is in an utterance, voiced or in a pause no check has ended,
	// whether or not any words of it have been written: the line is not silent. It is said
	// every interval of the policy while the utterance lasts, on the session's goroutine,
	// and must not block.
	Hearing(participant stt.Participant)
}

// Ruled is what a check ruled on an utterance, and the words it was about: a pause check's
// on an utterance that ended, or a check over the agent's (Overheard) on one going on.
type Ruled struct {
	Check
	Words string
	// AudioMs and WordsMs are how much audio the words are of and how long they took, as the
	// transcript of them says, by which the agent tells its own echo from the caller.
	AudioMs, WordsMs float64
	// Speaking says the agent was talking when the check was asked, and Paused how far into
	// the pause it was, from the voice's last frame.
	Speaking bool
	Paused   time.Duration
	// Overheard says the check was asked over the agent without a pause, to be acted on at
	// once.
	Overheard bool
	// Utterance is the utterance the words are of, which its final names.
	Utterance int64
}

// Pauses is a transcriber an agent can have ask about each pause.
type Pauses interface {
	// AskOnPauses has the transcriber ask about the pauses from now on for agent.
	AskOnPauses(agent Asker)
}

// neutralInstructions is what the transcriber's own agent is told, for a session no agent
// asks on behalf of (AskOnPauses): transcription alone, or an agent that acts on nothing the
// model says. The point of the model is that it ends turns, and it does so under a prompt,
// as it was trained to; with no app agent to supply one the transcriber brings its own, so
// that an utterance still ends when the model says so rather than at the silence cap. The
// prompt is a plain voice assistant with no tools and no conversation: it names no business
// and asks for no behaviour, so what it adds to the decision is as near nothing as a prompt
// can be, and p_eot is the audio's. It goes through POST /v1/agents like any other agent,
// once per server (the client remembers it by content), and the turn ends by the server's
// default preset, the point it publishes for an agent that chooses none. Its ruling goes
// nowhere: what to do with the words is the listener's to decide, as with any transcriber.
const neutralInstructions = "You are a voice assistant on a phone call. Listen to what the caller says and answer them."

// neutralAgent is the Asker of a session no agent asks for: the transcriber's own.
type neutralAgent struct{ hearer *Hearer }

// Moment is the call with nothing said yet: the words are the audio's, and the agent is
// saying nothing.
func (n neutralAgent) Moment(stt.Participant) harness.FlowTurn {
	return harness.FlowTurn{Instructions: neutralInstructions}
}

// Ruled hands the ruling to nobody.
func (neutralAgent) Ruled(stt.Participant, Ruled) {}

// Hearer ends turns by the server's default preset.
func (n neutralAgent) Hearer() *Hearer { return n.hearer }

// Speaking is never: the transcriber's own agent says nothing, so nobody talks over it.
func (neutralAgent) Speaking() bool { return false }

// Overheard is never asked, as nobody talks over an agent that says nothing.
func (neutralAgent) Overheard(stt.Participant, Ruled) {}

// Hearing tells nobody: the transcriber's own agent fills no silence.
func (neutralAgent) Hearing(stt.Participant) {}

// pause is the pause going on in the utterance being heard, and what its checks said.
type pause struct {
	// epoch tells this pause from the next: speech resuming, or the utterance ending,
	// begins another, and a check of an earlier one answers no one.
	epoch       int64
	checks      int
	asked       time.Duration
	inFlight    bool
	participant stt.Participant
	utterance   int64
	// failed says the pause is the words' alone to end, as with no agent: a check of it
	// failed, its words did, or none ended it by the cap and checkGrace.
	failed bool
	// resumed says speech came while a check was out (resumed): the pause is over unless
	// that check ended the turn.
	resumed bool
	// words are the first check's, once written, and audioMs and wordsMs how much audio
	// they are of and how long they took.
	words            *string
	audioMs, wordsMs float64
	// ended is the check that ended the turn, and endedAt where its audio ended, in frames.
	// done says the words have gone out, and the session is to begin the next utterance.
	ended   *Ruled
	endedAt int64
	done    bool
}

// pauseCheck is one question about a pause, asked off the session's goroutine. voiced is how
// much of the utterance so far was speech, and length how long it is, the pause included.
type pauseCheck struct {
	epoch          int64
	utterance      int64
	first          bool
	firstPause     bool
	paused         time.Duration
	voiced, length time.Duration
	at             int64
	participant    stt.Participant
	audio          harness.Audio
	asked          time.Time
}

// AskOnPauses has the transcriber ask about each pause for agent, from the next on, in place
// of its own (neutralAgent). Nil has it ask about none, and end utterances on the words alone.
func (t *Transcriber) AskOnPauses(agent Asker) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.asker = agent
}

// checkTimes are the server's policy's: the first check into a pause, the interval after it,
// and the pause that ends a turn whatever the model says. Until the server has said, they are
// what the transcriber writes and ends utterances at.
func (t *Transcriber) checkTimes() (first, every, cap time.Duration) {
	t.client.mu.Lock()
	defer t.client.mu.Unlock()
	if t.client.policy == nil {
		return writeAfter, 300 * time.Millisecond, endAfter
	}
	p := t.client.policy
	return time.Duration(p.FirstCheckMs) * time.Millisecond, p.Interval(), p.Cap()
}

// checksPause asks about the pause going on, when an agent has the transcriber do so: at the
// policy's first check into it, then every interval, each once the one before has answered.
// It says whether the pause is the checks' to end, which it is until one fails, or none has
// ended it by the policy's cap and checkGrace, or the utterance has run to maxUtterance. A
// check that ended it is waited on for its words, which a long utterance takes seconds to
// write.
func (t *Transcriber) checksPause(participant stt.Participant, heard int64) bool {
	first, every, cap := t.checkTimes()
	t.mu.Lock()
	defer t.mu.Unlock()
	p := &t.pause
	switch {
	case t.asker == nil || p.failed || heard-t.start >= int64(maxUtterance/frame):
		return false
	case t.quiet >= cap+checkGrace && p.ended == nil:
		t.logger.Warn("no check of a pause ended it, its words are asked for alone", "participant", participant.ID)
		p.failed = true
		return false
	}
	if p.inFlight || p.ended != nil || t.quiet < first || (p.checks > 0 && t.quiet < p.asked+every) {
		return true
	}
	p.checks++
	p.asked, p.inFlight, p.participant, p.utterance = t.quiet, true, participant, t.utterance+1
	if p.checks == 1 {
		// The first check writes the words so far, so no interim is due for a while.
		t.pauses++
		t.spoken = 0
	}
	c := pauseCheck{epoch: p.epoch, utterance: p.utterance, first: p.checks == 1, firstPause: t.pauses == 1, paused: t.quiet, at: heard,
		voiced: time.Duration(t.talked) * frame, length: time.Duration(heard-t.start) * frame,
		participant: participant, asked: time.Now()}
	// The check that asks for words sends the whole utterance: the model decides on its
	// last 16 s and writes all of it. The others send what a decision hears.
	length := c.length
	if !c.first {
		length = min(length, Window)
	}
	c.audio = t.heard.Last(length)
	t.running.Add(1)
	go t.check(t.asker, c)
	return true
}

// resumed is speech in the utterance: a pause that had been asked about is over, unless a
// check has ended the turn on it. That turn is over whatever comes next, the caller going on
// or somebody else in the room: its words go out as its final when they are written, and
// the speech is the next utterance's (endedOnCheck). A check still out is waited for: the
// model was asked about the pause as it was, and its answer stands (check). A pause whose
// words failed falls to the words alone, and the next pause is asked about afresh.
func (t *Transcriber) resumed() {
	t.mu.Lock()
	defer t.mu.Unlock()
	p := &t.pause
	switch {
	case p.checks == 0 || p.done:
		return
	case p.ended != nil && !p.failed:
		return
	case p.inFlight:
		p.resumed = true
		return
	}
	t.pause = pause{epoch: p.epoch + 1}
}

// endedOnCheck says a check ended the utterance and its words have gone out, and where its
// audio ended, which the next utterance begins after.
func (t *Transcriber) endedOnCheck() (int64, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.pause.done {
		return 0, false
	}
	at := t.pause.endedAt
	t.newUtteranceLocked()
	return at, true
}

func (t *Transcriber) newUtteranceLocked() {
	t.pause = pause{epoch: t.pause.epoch + 1}
	t.pauses = 0
	t.over = overChecks{}
}

// overAfter is how long the caller is voiced in an utterance before they are first asked
// about over the agent, without a pause. A murmur over the agent, "uh-huh", is voiced for
// about 400 ms, so the first check hears most of it and the model rules on a murmur, which
// it was trained to let the agent talk through, rather than on the first syllable of
// anything; a word that stops the agent, "no" or "stop", is said inside it. The agent's own
// echo is voiced for as long as it talks, and is told from the caller by the words, which
// the check asks for with the decision.
const overAfter = 350 * time.Millisecond

// overChecks are the checks of the utterance going on asked over the agent, without a pause
// (checksOver): whether one is out, when the last was asked, in frames, when the agent was
// last asked whether it is speaking and whether it was, which says whether interims are the
// checks' to write meanwhile.
type overChecks struct {
	inFlight bool
	at       int64
	polled   int64
	speaking bool
	// utterance is the utterance the checks are of: a new one starts them again.
	utterance int64
}

// overPoll is how often the agent is asked whether it is speaking while the caller is
// voiced and no check is due: a lock of the agent's, a few times a second.
const overPoll = 100 * time.Millisecond

// overCheck is one question about the caller talking over the agent, asked off the
// session's goroutine: the utterance so far, up to Window of it, and whether that is all of
// it, which its words are an interim of.
type overCheck struct {
	participant    stt.Participant
	utterance      int64
	audio          harness.Audio
	whole          bool
	voiced, length time.Duration
	asked          time.Time
}

// checksOver asks about the caller talking over the agent, without waiting for a pause,
// when an agent has the transcriber ask (AskOnPauses) and says it is speaking: once the
// caller has been voiced for overAfter in the utterance, then every interval of the policy,
// each once the one before has answered. A pause may never come over the agent: its echo on
// the caller's line or the room keeps the speech gate open, and a caller who talks straight
// over it leaves none the policy's first check would catch, so the pause checks alone left
// the agent talking until its reply ended. On its own echo the model rules ignore and
// writes the echo's words, which the agent drops as its own, so a line with no echo
// cancellation costs a decision and the words an interval while the agent talks and
// nothing is done. It says whether the checks are the words' to write meanwhile, as each
// asks for them: interims are not asked for besides. It runs on the session's goroutine, on
// a frame of speech.
func (t *Transcriber) checksOver(participant stt.Participant, heard int64) bool {
	if time.Duration(t.talked)*frame < overAfter || heard-t.start >= int64(maxUtterance/frame) {
		return false
	}
	_, every, _ := t.checkTimes()
	t.mu.Lock()
	asker, o := t.asker, t.over
	t.mu.Unlock()
	if asker == nil {
		return false
	}
	if o.inFlight || (o.at > 0 && time.Duration(heard-o.at)*frame < every) {
		return o.speaking
	}
	if !o.speaking && o.polled > 0 && time.Duration(heard-o.polled)*frame < overPoll {
		return false
	}
	speaking := asker.Speaking()
	t.mu.Lock()
	defer t.mu.Unlock()
	t.over.speaking, t.over.polled = speaking, heard
	if !speaking {
		return false
	}
	t.over.inFlight, t.over.at, t.over.utterance = true, heard, t.utterance+1
	length := time.Duration(heard-t.start) * frame
	c := overCheck{participant: participant, utterance: t.utterance + 1, whole: length <= Window,
		voiced: time.Duration(t.talked) * frame, length: length, asked: time.Now()}
	c.audio = t.heard.Last(min(length, Window))
	// The check writes the words so far, so no interim is due for a while.
	t.spoken = 0
	t.running.Add(1)
	go t.checkOver(asker, c)
	return true
}

// checkOver asks the server about the caller talking over the agent, the decision and the
// words in one request, and hands a ruling to stop or shorten to the agent at once with the
// words (Overheard), which it tells its own echo by: a ruling with no words is not acted on.
// The words are shown meanwhile as an interim when they are of the whole utterance. The
// utterance goes on: when it ends is the pause checks' to say, with the floor free once the
// agent has stopped.
func (t *Transcriber) checkOver(asker Asker, c overCheck) {
	defer t.running.Done()
	defer func() {
		t.mu.Lock()
		t.over.inFlight = false
		t.mu.Unlock()
	}()
	turn := asker.Moment(c.participant)
	if !turn.Speaking {
		// The agent has stopped since: when the turn ends is the pause checks' to say.
		return
	}
	turn.ParticipantID, turn.Audio, turn.Unfinished = c.participant.ID, c.audio, true
	ctx, cancel := context.WithTimeout(t.ctx, harness.HearDeadline(turn))
	defer cancel()
	var said <-chan Words
	checked, err := func() (Check, error) {
		policy, err := t.client.Policy(ctx)
		if err != nil {
			return Check{}, err
		}
		agent := Agent{Instructions: turn.Instructions, Tools: turn.Tools, Languages: t.languages}
		var answer Answer
		answer, said, err = t.client.DecideWords(ctx, agent, turnOf(turn), turn.Audio)
		if err != nil {
			return Check{}, err
		}
		return asker.Hearer().rule(policy, turn, answer)
	}()
	checks.WithLabelValues(outcome(err)).Inc()
	if err == nil {
		checkSeconds.Observe(time.Since(c.asked).Seconds())
	}
	t.logCheck(turn, pauseCheck{first: true, voiced: c.voiced, length: c.length, participant: c.participant, asked: c.asked}, checked, err)
	if err != nil {
		return
	}
	var words Words
	select {
	case words = <-said:
	case <-t.ctx.Done():
		return
	}
	if words.Err != nil {
		transcribeFailures.WithLabelValues(failure(words.Err)).Inc()
		t.logger.Debug("the chorus-asr server did not write the words of a check over the agent, whose ruling is not acted on",
			"participant", c.participant.ID, "error", words.Err)
		return
	}
	text := strings.TrimSpace(words.Text)
	if text == "" {
		return
	}
	audioMs, wordsMs := float64(c.audio.Duration.Microseconds())/1000, float64(time.Since(c.asked).Microseconds())/1000
	spoken := t.sent.establish(c.utterance, words.Language)
	if c.whole && t.interims && (words.Language == "" || words.Language == spoken) && t.sent.over(c.utterance-1) {
		t.send(stt.Transcript{
			Participant:      c.participant,
			Mode:             stt.ModeReplacement,
			Interim:          true,
			Utterance:        c.utterance,
			Text:             text,
			Provider:         ProviderName,
			Model:            t.model,
			ProcessingTimeMs: wordsMs,
			AudioDurationMs:  audioMs,
		})
	}
	if checked.Ruling.Disposition == harness.Ignore || checked.Ruling.Floor == harness.Continue {
		return
	}
	if t.overtaken(c) {
		t.logger.Debug("a check over the agent answered after its utterance ended, which is not acted on",
			"participant", c.participant.ID, "utterance", c.utterance, "floor", checked.Ruling.Floor)
		return
	}
	asker.Overheard(c.participant, Ruled{Check: checked, Words: text, Speaking: true, Overheard: true, AudioMs: audioMs, WordsMs: wordsMs, Utterance: c.utterance})
}

// overtaken says the utterance a check over the agent asked about has ended since, a pause
// check having ended it with a ruling of its own: the check is about an utterance that is
// over, and that ruling is the one that stands.
func (t *Transcriber) overtaken(c overCheck) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.over.utterance != c.utterance || t.pause.ended != nil
}

// check asks the server about a pause, and keeps what it said if the pause is still going on.
func (t *Transcriber) check(asker Asker, c pauseCheck) {
	defer t.running.Done()
	turn := asker.Moment(c.participant)
	// The checks in a row a point asks for are counted by participant, by the ID that is
	// theirs alone, from each pause's first.
	turn.ParticipantID = c.participant.ID
	turn.Audio, turn.Paused, turn.FirstPause, turn.Waited = c.audio, c.paused, c.firstPause, !c.first
	ctx, cancel := context.WithTimeout(t.ctx, harness.HearDeadline(turn))
	defer cancel()
	var said <-chan Words
	checked, err := func() (Check, error) {
		policy, err := t.client.Policy(ctx)
		if err != nil {
			return Check{}, err
		}
		agent := Agent{Instructions: turn.Instructions, Tools: turn.Tools, Languages: t.languages}
		var answer Answer
		if c.first {
			answer, said, err = t.client.DecideWords(ctx, agent, turnOf(turn), turn.Audio)
		} else {
			answer, err = t.client.Decide(ctx, agent, turnOf(turn), turn.Audio)
		}
		if err != nil {
			return Check{}, err
		}
		return asker.Hearer().rule(policy, turn, answer)
	}()
	checks.WithLabelValues(outcome(err)).Inc()
	if err == nil {
		checkSeconds.Observe(time.Since(c.asked).Seconds())
	}
	t.logCheck(turn, c, checked, err)
	t.mu.Lock()
	p := &t.pause
	if p.epoch != c.epoch {
		// The caller spoke again, or the utterance ended: nobody waits for this.
		t.mu.Unlock()
		return
	}
	p.inFlight = false
	if err != nil {
		p.failed = true
		if p.resumed {
			t.pause = pause{epoch: p.epoch + 1}
		}
		// A server that will never rule, with no policy for its model or refusing the
		// request, is asked no more on the transcriber's own behalf: every pause would
		// otherwise cost a request that fails. An agent asking keeps asking, as it chose to.
		_, own := t.asker.(neutralAgent)
		if own && Misconfigured(err) {
			t.asker = nil
			t.mu.Unlock()
			t.logger.Warn("the chorus-asr server can't rule on pauses, so this session's utterances end on the words alone",
				"participant", c.participant.ID, "error", err)
			return
		}
		t.mu.Unlock()
		t.logger.Debug("the chorus-asr server did not rule on a pause, its words are asked for alone",
			"participant", c.participant.ID, "error", err)
		return
	}
	if checked.Ended && p.ended == nil {
		p.ended, p.endedAt = &Ruled{Check: checked, Speaking: turn.Speaking, Paused: c.paused, Utterance: c.utterance}, c.at
	} else if !checked.Ended && p.resumed {
		// Speech came while the check was out, and it did not end the turn: the pause is
		// over. Its words, if asked for, are still the utterance's so far (await).
		t.pause = pause{epoch: p.epoch + 1}
	}
	t.mu.Unlock()
	if said != nil {
		t.running.Add(1)
		go t.await(c, said)
	}
	t.finish(c.epoch)
}

// logCheck says what a check asked and what came of it, for a call's debug log: no audio
// and no words.
func (t *Transcriber) logCheck(turn harness.FlowTurn, c pauseCheck, checked Check, err error) {
	attributes := []any{"participant", c.participant.ID, "speaking", turn.Speaking, "paused_ms", c.paused.Milliseconds(),
		"voiced_ms", c.voiced.Milliseconds(), "utterance_ms", c.length.Milliseconds(), "first", c.first, "first_pause", c.firstPause,
		"took_ms", time.Since(c.asked).Milliseconds()}
	if err != nil {
		t.logger.Debug("chorus-asr did not rule on a check", append(attributes, "error", err)...)
		return
	}
	var pEOT any
	if checked.Answer.PEOT != nil {
		pEOT = *checked.Answer.PEOT
	}
	t.logger.Debug("chorus-asr checked", append(attributes, "answer", checked.Answer.Answer, "p_eot", pEOT, "bar", checked.Bar,
		"in_a_row", checked.InARow, "ended", checked.Ended, "disposition", checked.Ruling.Disposition, "floor", checked.Ruling.Floor)...)
}

// await keeps the words of a pause's first check, once written, and shows them meanwhile
// (Interim) when the check did not end the turn: they are the words so far, written anyway,
// and the final is as it would be without them.
func (t *Transcriber) await(c pauseCheck, said <-chan Words) {
	defer t.running.Done()
	var words Words
	select {
	case words = <-said:
	case <-t.ctx.Done():
		return
	}
	audioMs, wordsMs := float64(c.audio.Duration.Microseconds())/1000, float64(time.Since(c.asked).Microseconds())/1000
	t.mu.Lock()
	p := &t.pause
	// The pause may be over, the caller having gone on: its words are still the utterance's
	// so far, to show and to say what language it is in, though no final is made of them.
	same := p.epoch == c.epoch && p.utterance == c.utterance
	if words.Err != nil {
		if same {
			p.failed = true
		}
		t.mu.Unlock()
		transcribeFailures.WithLabelValues(failure(words.Err)).Inc()
		t.logger.Warn("the chorus-asr server did not write a pause's words, they are asked for alone",
			"participant", c.participant.ID, "error", words.Err)
		return
	}
	text := strings.TrimSpace(words.Text)
	waiting := true
	if same {
		p.words, p.audioMs, p.wordsMs = &text, audioMs, wordsMs
		waiting = p.ended == nil
	}
	t.mu.Unlock()
	if text == "" {
		t.finish(c.epoch)
		return
	}
	// The utterance's first words at a pause say what language it is spoken in, which its
	// interims are held to: a pause's words in another are kept for the final, not shown.
	// Nor are they shown before the utterance before has had its final.
	spoken := t.sent.establish(c.utterance, words.Language)
	if waiting && t.interims && (words.Language == "" || words.Language == spoken) && t.sent.over(c.utterance-1) {
		t.send(stt.Transcript{
			Participant:      c.participant,
			Mode:             stt.ModeReplacement,
			Interim:          true,
			Utterance:        c.utterance,
			Text:             text,
			Provider:         ProviderName,
			Model:            t.model,
			ProcessingTimeMs: wordsMs,
			AudioDurationMs:  audioMs,
		})
	}
	t.finish(c.epoch)
}

// finish sends the words of a pause a check ended, once both are in: the ruling to the
// agent, then the words as the utterance's final. Words that came back empty are nothing
// heard, and end the utterance with nothing sent.
func (t *Transcriber) finish(epoch int64) {
	t.mu.Lock()
	p := &t.pause
	if p.epoch != epoch || p.done || p.ended == nil || p.words == nil {
		t.mu.Unlock()
		return
	}
	p.done = true
	ruled, asker, participant, utterance := *p.ended, t.asker, p.participant, p.utterance
	ruled.Words, ruled.AudioMs, ruled.WordsMs = *p.words, p.audioMs, p.wordsMs
	t.mu.Unlock()
	if ruled.Words == "" {
		t.sent.ended(utterance)
		return
	}
	asker.Ruled(participant, ruled)
	t.send(stt.Transcript{
		Participant:      participant,
		Mode:             stt.ModeFinal,
		Utterance:        utterance,
		Text:             ruled.Words,
		Provider:         ProviderName,
		Model:            t.model,
		ProcessingTimeMs: ruled.WordsMs,
		AudioDurationMs:  ruled.AudioMs,
	})
}
