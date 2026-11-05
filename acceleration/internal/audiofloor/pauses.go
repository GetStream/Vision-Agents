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
}

// Ruled is what a pause check ruled on an utterance that ended, and the words it was about.
type Ruled struct {
	Check
	Words string
	// Speaking says the agent was talking when the check was asked, and Paused how far into
	// the pause it was, from the voice's last frame.
	Speaking bool
	Paused   time.Duration
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
	// failed says a check could not be made, and the pause ends as it would with no agent.
	failed bool
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
	c := pauseCheck{epoch: p.epoch, first: p.checks == 1, firstPause: t.pauses == 1, paused: t.quiet, at: heard,
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

// resumed is speech in the utterance: a pause that had been asked about is over.
func (t *Transcriber) resumed() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.pause.checks > 0 && !t.pause.done {
		t.pause = pause{epoch: t.pause.epoch + 1}
	}
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
	if checked.Ended {
		p.ended, p.endedAt = &Ruled{Check: checked, Speaking: turn.Speaking, Paused: c.paused}, c.at
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
	t.mu.Lock()
	p := &t.pause
	if p.epoch != c.epoch {
		t.mu.Unlock()
		return
	}
	if words.Err != nil {
		p.failed = true
		t.mu.Unlock()
		transcribeFailures.WithLabelValues(failure(words.Err)).Inc()
		t.logger.Warn("the chorus-asr server did not write a pause's words, they are asked for alone",
			"participant", c.participant.ID, "error", words.Err)
		return
	}
	text := strings.TrimSpace(words.Text)
	p.words, p.audioMs, p.wordsMs = &text, float64(c.audio.Duration.Microseconds())/1000, float64(time.Since(c.asked).Microseconds())/1000
	waiting, utterance, audioMs, wordsMs := p.ended == nil, p.utterance, p.audioMs, p.wordsMs
	t.mu.Unlock()
	if text == "" {
		t.finish(c.epoch)
		return
	}
	// The utterance's first words at a pause say what language it is spoken in, which its
	// interims are held to: a pause's words in another are kept for the final, not shown.
	spoken := t.sent.establish(utterance, words.Language)
	if waiting && t.interims && (words.Language == "" || words.Language == spoken) {
		t.send(stt.Transcript{
			Participant:      c.participant,
			Mode:             stt.ModeReplacement,
			Interim:          true,
			Utterance:        utterance,
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
	ruled.Words = *p.words
	audioMs, wordsMs := p.audioMs, p.wordsMs
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
		ProcessingTimeMs: wordsMs,
		AudioDurationMs:  audioMs,
	})
}
