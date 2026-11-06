package agent

import (
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// wordTrail is every fate a call's words met (wordLedger), for a test to hold the call to the
// account: no word of a caller's that the transcriber delivered in a final is lost.
type wordTrail struct {
	mu    sync.Mutex
	fates []wordsFated
	// known is words a test shows being lost, as a gap the loop still has.
	known []string
}

// tracksWords has a trail told of every fate the words of agent's callers meet. It is set
// before the agent joins, so before any of them.
func tracksWords(agent *Agent) *wordTrail {
	trail := &wordTrail{}
	agent.words.observe = trail.add
	return trail
}

func (w *wordTrail) add(fated wordsFated) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.fates = append(w.fates, fated)
}

// all is the trail so far, in the order the fates were met.
func (w *wordTrail) all() []wordsFated {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]wordsFated(nil), w.fates...)
}

// met is the fates in the trail of words that say text.
func (w *wordTrail) met(text string) []wordFate {
	var fates []wordFate
	for _, fated := range w.all() {
		if sameWords(fated.Text, text) {
			fates = append(fates, fated.Fate)
		}
	}
	return fates
}

// losing says the test shows text being lost, a gap the loop still has, which noWordLost
// lets pass.
func (w *wordTrail) losing(text string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.known = append(w.known, text)
}

// lost is the words in the trail that were lost, but for those the test shows being lost.
func (w *wordTrail) lost() []wordsFated {
	w.mu.Lock()
	known := append([]string(nil), w.known...)
	w.mu.Unlock()
	var lost []wordsFated
	for _, fated := range w.all() {
		if fated.Fate == fateLost && !slices.ContainsFunc(known, func(text string) bool { return sameWords(text, fated.Text) }) {
			lost = append(lost, fated)
		}
	}
	return lost
}

func (w *wordTrail) String() string {
	var trail strings.Builder
	for _, fated := range w.all() {
		fmt.Fprintf(&trail, "\n  %s utterance %d %-11s %q (%d words) %s", fated.Participant, fated.Utterance, fated.Fate,
			fated.Text, fated.Words, fated.Reason)
	}
	return trail.String()
}

// noWordLost fails t if any word of a caller's final in the trail's call was lost: every
// one is answered, or set aside for a named reason. It is checked once the call has ended,
// when the ledger has closed every word still open.
func noWordLost(t testing.TB, trail *wordTrail) {
	t.Helper()
	if trail == nil {
		return
	}
	if lost := trail.lost(); len(lost) > 0 {
		t.Errorf("%d of the callers' words were lost, neither answered nor set aside for a reason; the call's words:%s",
			len(lost), trail)
	}
}

// trackedLedger is a ledger whose every fate a trail is told of.
func trackedLedger() (*wordLedger, *wordTrail) {
	ledger, trail := newWordLedger(slog.New(slog.DiscardHandler)), &wordTrail{}
	ledger.observe = trail.add
	return ledger, trail
}

// The loop's bugs that cost a caller words the transcriber wrote, each replayed through the
// ledger as its log has it, the way the loop told the ledger of it then: each is lost. Three
// were set aside for a reason, which did not hold, and one with none.
func TestTheWordsEachLoopBugDroppedAreLost(t *testing.T) {
	const caller = "caller"
	silencing := func(words string) *stopEvidence {
		return &stopEvidence{words: words, heard: true, silenced: true}
	}
	for _, c := range []struct {
		name string
		call func(l *wordLedger)
		// lost is the words lost; kept are the words answered or set aside with a reason
		// that holds.
		lost string
		kept map[string]wordFate
	}{
		{
			// #1825, restaurant.interrupt at f1a3c3f84:
			//   17:02:44.274 the agent was interrupted turn=tool-1793984562609855000 cause=talking ruler=heard words="Wait. Make."
			//   17:02:44.581 decision=interrupt reason="the caller is still talking…" turn=""  (and every ~300 ms to 46.39)
			//   17:02:46.673 decision=ignore reason="the words a stop over the agent was ruled on…" text="Wait. Make it six, and one of us is gluten free instead."
			// Only the first check silenced the agent; the rest found no reply to stop.
			name: "#1825 a correction after a stop, set aside as the stop's words",
			call: func(l *wordLedger) {
				l.overheard(caller, 2, "Wait. Make.", true, silencing("Wait. Make."))
				for _, more := range []string{"Wait. Make it six,", "Wait. Make it six, and one of", "Wait. Make it six, and one of us is",
					"Wait. Make it six, and one of us is gluten free", "Wait. Make it six, and one of us is gluten free instead."} {
					l.overheard(caller, 2, more, true, nil)
				}
				l.final(caller, 2, "Wait. Make it six, and one of us is gluten free instead.")
				l.spent(caller, 2, "Wait. Make it six, and one of us is gluten free instead.")
			},
			lost: "Wait. Make it six, and one of us is gluten free instead.",
		},
		{
			// #1826, the replay's C2 line at f7f535b40: "wait" over the forecast list once its
			// tool's follow-up has begun; only the follow-up was stopped, and the list was heard
			// for 5 s after. On the call:
			//   17:02:44.274 the agent was interrupted turn=tool-1793984562609855000 cause=talking ruler=heard words="Wait. Make." played_ms=0
			//   17:02:50.371 finished speaking turn=turn-1793984560211969000 audio_ms=9110 synthesis_cut=false turn_abandoned=false
			name: "#1826 a stop's words spent while the reply it stopped played on",
			call: func(l *wordLedger) {
				l.overheard(caller, 2, "Wait.", true, &stopEvidence{words: "Wait.", heard: true, silenced: false})
				l.final(caller, 2, "Wait.")
				l.spent(caller, 2, "Wait.")
			},
			lost: "Wait.",
		},
		{
			// #1827, restaurant.monologue at d3bb9f600: the caller began at 13.29 s, and the
			// reply was first heard at 14.02 s.
			//   18:49:44.575 the agent was interrupted cause=talking ruler=heard words="The name for the booking is Alvarez. That's A L V A R E Z." played_ms=3112
			//   18:49:45.800 decision=ignore reason="the words a stop over the agent was ruled on…" text="The name for the booking is Alvarez. That's A L V A R E Z."
			name: "#1827 a continuation begun before the reply was heard, set aside as the stop's words",
			call: func(l *wordLedger) {
				said := "The name for the booking is Alvarez. That's A L V A R E Z."
				l.overheard(caller, 2, said, true, &stopEvidence{words: said, heard: false, silenced: true})
				l.final(caller, 2, said)
				l.spent(caller, 2, said)
			},
			lost: "The name for the booking is Alvarez. That's A L V A R E Z.",
		},
		{
			// #1829, telecom.interrupt t2 at d3bb9f600: a check over the agent wrote the digits
			// and shortened the reply, the pause after them ended the utterance with its words
			// empty, and no final carried them.
			//   19:07:37.498 the agent was interrupted cause=shortened words="Last four: ninety-eight twenty-one."
			//   (no line for the utterance's end: finish() sends nothing when its words are empty)
			//   decision=answer text="Fourteen Cedar Lane."
			name: "#1829 an utterance ended with no final, after a check over the agent wrote its words",
			call: func(l *wordLedger) {
				l.overheard(caller, 1, "Last four: ninety-eight twenty-one.", true, nil)
				l.unwritten(caller, 1, "")
				l.final(caller, 2, "Fourteen Cedar Lane.")
				l.answered(caller, "Fourteen Cedar Lane.")
			},
			lost: "Last four: ninety-eight twenty-one.",
			kept: map[string]wordFate{"Fourteen Cedar Lane.": fateAnswered},
		},
		{
			// #1830, healthcare.interrupt t3 at d3bb9f600: the agent's last audible sample went
			// out 1,240 ms before the caller's audio began, and the old gate, 1.5 s from the
			// last chunk queued, took "move it to" for its echo.
			//   18:56:50.452 finished speaking turn=turn-1793991399943391000 audio_ms=8980 frames_queued=449 frames_sent=429 queued_ms=400
			//   18:56:56.183 the conversation decided decision=answer text="Keep Maya's Friday ten a.m. with Doctor Chen. Just Tuesday morning."
			name: "#1830 the caller's words taken for the agent's echo after its voice had stopped",
			call: func(l *wordLedger) {
				to := time.Now().Add(-10 * time.Second)
				began := to.Add(1240 * time.Millisecond)
				l.echoed(caller, 3, keptAndMoved, 3, movedLine, began, began.Add(4460*time.Millisecond), to.Add(-8480*time.Millisecond), to)
				left := "Keep Maya's Friday ten a.m. with Doctor Chen. Just Tuesday morning."
				l.final(caller, 3, left)
				l.answered(caller, left)
			},
			lost: keptAndMoved,
			kept: map[string]wordFate{"Keep Maya's Friday ten a.m. with Doctor Chen. Just Tuesday morning.": fateAnswered},
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			ledger, trail := trackedLedger()
			c.call(ledger)
			ledger.end(nil)
			lost := trail.lost()
			require.Len(t, lost, 1, "the call's words:%s", trail)
			require.Equal(t, c.lost, lost[0].Text)
			require.NotEmpty(t, lost[0].Reason, "why they are lost")
			for text, fate := range c.kept {
				require.Equal(t, []wordFate{fate}, trail.met(text), "the call's words:%s", trail)
			}
		})
	}
}

// Every word is answered or set aside for a reason that holds, and each reason is its own
// fate. A stop's words are spent on a stop that silenced the agent, begun once the caller had
// heard it, and the agent's echo is in audio its voice could be in.
func TestEveryWordIsAnsweredOrSetAsideForAReason(t *testing.T) {
	const caller = "caller"
	to := time.Now().Add(-10 * time.Second)
	for _, c := range []struct {
		name  string
		call  func(l *wordLedger)
		text  string
		fates []wordFate
	}{
		{"answered", func(l *wordLedger) {
			l.final(caller, 1, "Two tickets to Bilbao, please.")
			l.answered(caller, "Two tickets to Bilbao, please.")
		}, "Two tickets to Bilbao, please.", []wordFate{fateAnswered}},
		{"answered across utterances the cadence carried into one turn", func(l *wordLedger) {
			l.final(caller, 1, "6:00.")
			l.final(caller, 2, "fifteen")
			l.answered(caller, "6:00. fifteen")
		}, "fifteen", []wordFate{fateAnswered}},
		{"the rest of an utterance the agent answered the start of", func(l *wordLedger) {
			l.final(caller, 1, "Book a table")
			l.answered(caller, "Book a table")
			l.final(caller, 1, "Book a table for four on Friday")
			l.answered(caller, "for four on Friday")
		}, "for four on Friday", []wordFate{fateAnswered}},
		{"a bare stop spent on silencing the agent", func(l *wordLedger) {
			l.overheard(caller, 2, "Stop.", true, &stopEvidence{words: "Stop.", heard: true, silenced: true})
			l.final(caller, 2, "Stop.")
			l.spent(caller, 2, "Stop.")
		}, "Stop.", []wordFate{fateStop}},
		{"a bare stop whose utterance came back with no words", func(l *wordLedger) {
			l.overheard(caller, 2, "Hold on.", true, &stopEvidence{words: "Hold on.", heard: true, silenced: true})
			l.unwritten(caller, 2, "")
		}, "Hold on.", []wordFate{fateStop}},
		{"the agent's echo", func(l *wordLedger) {
			l.echoed(caller, 1, "the day you'd like to move it to", 8, movedLine, to.Add(-1800*time.Millisecond), to.Add(400*time.Millisecond),
				to.Add(-8480*time.Millisecond), to)
		}, "the day you'd like to move it to", []wordFate{fateEcho}},
		{"the flow controller's ignore", func(l *wordLedger) {
			l.final(caller, 1, "Did you feed the dog?")
			l.setAside(caller, "Did you feed the dog?", ignoredAs("Did you feed the dog?", false), "background")
		}, "Did you feed the dog?", []wordFate{fateIgnored}},
		{"assent over the agent", func(l *wordLedger) {
			l.final(caller, 1, "Uh-huh.")
			l.setAside(caller, "Uh-huh.", ignoredAs("Uh-huh.", false), "background")
		}, "Uh-huh.", []wordFate{fateBackchannel}},
		{"a cough over the agent", func(l *wordLedger) {
			l.final(caller, 1, "[cough]")
			l.setAside(caller, "[cough]", ignoredAs("[cough]", true), "noise")
		}, "[cough]", []wordFate{fateNoise}},
		{"another language", func(l *wordLedger) {
			l.overheard(caller, 1, "No. No.", true, nil)
			l.unwritten(caller, 1, "language")
		}, "No. No.", []wordFate{fateLanguage}},
		{"a final the server never wrote", func(l *wordLedger) {
			l.overheard(caller, 1, "My member ID is", true, nil)
			l.unwritten(caller, 1, "behind")
		}, "My member ID is", []wordFate{fateUnwritten}},
		{"a transcriber that closed first", func(l *wordLedger) {
			l.overheard(caller, 1, "My member ID is", true, nil)
			l.closed(caller)
		}, "My member ID is", []wordFate{fateUnwritten}},
		{"the transcriber restating what was answered", func(l *wordLedger) {
			l.final(caller, 1, "one two three")
			l.answered(caller, "one two three")
			l.final(caller, 1, "1 2 3")
			l.repeated(caller, 1, "1 2 3")
		}, "1 2 3", []wordFate{fateRepeated}},
		{"a queued turn another took the place of", func(l *wordLedger) {
			l.final(caller, 1, "And in Porto.")
			l.setAside(caller, "And in Porto.", fateReplaced, "replaced")
		}, "And in Porto.", []wordFate{fateReplaced}},
		{"words the loop still held when the call ended", func(l *wordLedger) {
			l.final(caller, 1, "Is the ferry running")
			l.end(func(string, string) bool { return true })
		}, "Is the ferry running", []wordFate{fateCallEnded}},
		{"words that came as the call ended", func(l *wordLedger) {
			l.closing()
			l.final(caller, 1, "Bye.")
			l.end(func(string, string) bool { return false })
		}, "Bye.", []wordFate{fateCallEnded}},
		{"words the loop held nowhere when the call ended", func(l *wordLedger) {
			l.final(caller, 1, "Is the ferry running")
			l.end(func(string, string) bool { return false })
		}, "Is the ferry running", []wordFate{fateLost}},
		{"a stop's words that say more than the stop", func(l *wordLedger) {
			l.overheard(caller, 2, "No.", true, &stopEvidence{words: "No.", heard: true, silenced: true})
			l.final(caller, 2, "No, Porto.")
			l.spent(caller, 2, "No, Porto.")
		}, "No, Porto.", []wordFate{fateLost}},
		{"a stop's words where no stop silenced the agent", func(l *wordLedger) {
			l.final(caller, 2, "Stop.")
			l.spent(caller, 2, "Stop.")
		}, "Stop.", []wordFate{fateLost}},
	} {
		t.Run(c.name, func(t *testing.T) {
			ledger, trail := trackedLedger()
			c.call(ledger)
			ledger.end(nil)
			require.Equal(t, c.fates, trail.met(c.text), "the call's words:%s", trail)
		})
	}
}

// overheardStop has a check over the agent rule stop on words of utterance 2, the caller
// having begun it at began, as chorus-asr's transcriber hands one over.
func (s *AgentSuite) overheardStop(participant stt.Participant, words string, began time.Time) {
	pauseAsker{s.agent}.Overheard(participant, audiofloor.Ruled{Check: audiofloor.Check{Ruling: harness.Decided{Disposition: harness.Wait, Floor: harness.Stop}},
		Words: words, Speaking: true, Overheard: true, Began: began, AudioMs: 700, WordsMs: 100, Utterance: 2})
}

// finalOf has chorus-asr's transcriber send the final of a caller's utterance.
func (s *AgentSuite) finalOf(participant stt.Participant, utterance int64, text string) {
	s.ears.emitter.Send(stt.Transcript{Participant: participant, Mode: stt.ModeFinal, Utterance: utterance, Text: text,
		Provider: audiofloor.ProviderName, Language: "en", Confidence: 1})
}

// metFate waits for text's words to meet a fate.
func (s *AgentSuite) metFate(text string) []wordFate {
	s.eventually(func() bool { return len(s.words.met(text)) > 0 }, "the words met no fate: "+text)
	return s.words.met(text)
}

// What a caller says over the agent and is ruled stop on is answered, or spent on a stop that
// silenced the agent, and none of it is lost: the three calls whose words were set aside as a
// stop's, as their logs have them. A bare stop begun over the reply is spent (C2 stoponly). A
// correction stopped on its first words is answered whole: on restaurant.interrupt (#1825)
// "Wait. Make it six, and one of us is gluten free instead." was stopped on "Wait. Make.", and
// set aside as a stop's words once the checks after it, with nothing left to stop, had ruled
// stop on all of it. And words the caller went on with before any of the reply went out are
// answered: on restaurant.monologue (#1827) the name was set aside.
func (s *AgentSuite) TestNoWordOfAStopOrWhatTheCallerSaidWithItIsLost() {
	cases := map[string]struct {
		checks []string
		// before says the caller began before any of the reply went out.
		before bool
		fate   wordFate
	}{
		"Stop.": {checks: []string{"Stop."}, fate: fateStop},
		"Wait. Make it six, and one of us is gluten free instead.": {checks: []string{"Wait. Make.", "Wait. Make it six,",
			"Wait. Make it six, and one of", "Wait. Make it six, and one of us is gluten free instead."}, fate: fateAnswered},
		"The name for the booking is Alvarez. That's A L V A R E Z.": {checks: []string{"The name for the booking is Alvarez. That's A L V A R E Z."},
			before: true, fate: fateAnswered},
	}
	for said, c := range cases {
		s.Run(said, func() {
			s.join(true)
			participant := stt.Participant{ID: "alice"}
			s.voice.silent = true
			s.model.reply = []string{"I have a table for four on Saturday at seven thirty, on the patio. "}
			s.speak(participant)
			s.says(participant, "A table for four, Saturday at seven thirty. Patio, peanut allergy.")
			s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 && len(s.model.requests()) == 1 }, "the reply never started")
			began := time.Now()
			s.heardSomeOf(s.model.requests()[0].ID)
			s.replyWentOut()
			if !c.before {
				began = time.Now()
			}
			for i, words := range c.checks {
				s.overheardStop(participant, words, began)
				if i == 0 {
					s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 }, "the stop was not acted on")
				}
			}

			s.model.reply = []string{"Of course."}
			s.finalOf(participant, 2, said)
			s.Equal([]wordFate{c.fate}, s.metFate(said))
			s.Equal([]wordFate{fateAnswered}, s.words.met("A table for four, Saturday at seven thirty. Patio, peanut allergy."))
		})
	}
}

// A stop's words are spent on the stop only once it has silenced the agent, everything it was
// saying. On restaurant.interrupt (#1826) the agent had taken its next turn, to say what a tool
// came back with, while the reply that called the tool still played: a stop abandoned the
// follow-up alone, and the reply played on for 6 s over the caller. Acting on chorus-asr, the
// stop silences both, and "Wait." is spent on it. An agent that doesn't act on chorus-asr still
// abandons the follow-up alone: a stop over it, as a check over the agent hands one over, leaves
// the reply playing, and "Wait." set aside as its words is lost.
func (s *AgentSuite) TestTheWordsOfAStopThatLeftTheReplyPlayingAreLost() {
	for _, chorusASR := range []bool{true, false} {
		s.Run(map[bool]string{true: "acting on chorus-asr", false: "acting on another floor control"}[chorusASR], func() {
			s.floor, s.floorMode, s.earsPause = nil, "", false
			reply, toolTurn := s.replyPlayingBehindAToolFollowUp(chorusASR, nil)
			participant := stt.Participant{ID: "alice"}
			s.overheardStop(participant, "Wait.", time.Now())
			s.eventually(func() bool { return s.agent.abandonedTurn(toolTurn) }, "the stop was not acted on")
			s.Equal(chorusASR, s.agent.abandonedTurn(reply))
			if !chorusASR {
				s.words.losing("Wait.")
			}

			s.finalOf(participant, 2, "Wait.")
			s.Equal([]wordFate{map[bool]wordFate{true: fateStop, false: fateLost}[chorusASR]}, s.metFate("Wait."))
		})
	}
}

// A caller who says the agent's words once its voice has stopped said them, and they are
// answered: on healthcare.interrupt (#1830) the caller began 1.24 s after the agent's last
// audible sample went out, and "move it to" was taken for its echo. The agent's words heard
// back while its voice can be in the caller's audio are its echo.
func (s *AgentSuite) TestNoWordTheCallerRepeatedOfTheAgentIsLostToItsEcho() {
	for _, c := range []struct {
		name string
		// after is when the caller's audio began, from the agent's last audible sample going
		// out, and lasts how long it lasts.
		after, lasts time.Duration
		said         string
		fate         wordFate
	}{
		{"the caller, 1.24 s after", ms(1240), ms(4460), keptAndMoved, fateAnswered},
		{"the echo, as late as it can begin", echoDelay - ms(100), ms(1500), "you'd like to move it to", fateEcho},
	} {
		s.Run(c.name, func() {
			s.join(true)
			participant := stt.Participant{ID: "alice"}
			s.model.reply = []string{movedLine}
			s.speak(participant)
			s.says(participant, "I need to move my son's appointment.")
			s.eventually(func() bool { return s.spokenText("move it to?") && !s.agent.Busy() }, "the agent never asked")

			// The line goes out as it did on the call, its last audible sample c.after before the
			// caller's audio began.
			const processing = 120
			began := time.Now().Add(-ms(processing) - c.lasts)
			s.agent.mu.Lock()
			s.agent.voiceOut = voiceOut{}
			s.agent.mu.Unlock()
			publishLine(s.agent, began.Add(-c.after-ms(8500)), spokenLine(ms(8980), ms(20), ms(8500)), ms(400))
			s.model.reply = []string{"Tuesday morning it is."}
			s.ears.emitter.Send(stt.Transcript{Participant: participant, Mode: stt.ModeFinal, Utterance: 2, Text: c.said,
				Provider: audiofloor.ProviderName, Language: "en", Confidence: 1,
				AudioDurationMs: float64(c.lasts.Milliseconds()), ProcessingTimeMs: processing})
			s.Equal([]wordFate{c.fate}, s.metFate(c.said))
		})
	}
}

// What a check over the agent wrote of an utterance that then ends with no final is lost. On
// telecom.interrupt (#1829, t2 and t3) a check over the agent wrote "Last four: ninety-eight
// twenty-one." and shortened the reply, and the pause after the digits ended the utterance
// with its words empty: chorus-asr's transcriber sends no final then (pauses.go, finish), and
// tells the agent the utterance went unwritten. The digits never reached the model, and are
// counted lost, with a warning, until a final carries them.
func (s *AgentSuite) TestTheWordsOfAnUtteranceThatEndedWithNoFinalAreLost() {
	s.join(true)
	participant := stt.Participant{ID: "alice"}
	s.voice.silent = true
	s.model.reply = []string{"I can help with your internet. Can you confirm the last four digits of the account holder's number"}
	s.speak(participant)
	s.says(participant, "Internet down since six. Pin forty-four seventy-one.")
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 && len(s.model.requests()) == 1 }, "the reply never started")
	s.heardSomeOf(s.model.requests()[0].ID)
	s.replyWentOut()

	digits := "Last four: ninety-eight twenty-one."
	s.words.losing(digits)
	pauseAsker{s.agent}.Overheard(participant, audiofloor.Ruled{Check: audiofloor.Check{Ruling: harness.Decided{Disposition: harness.Wait, Floor: harness.Shorten}},
		Words: digits, Speaking: true, Overheard: true, Began: time.Now(), AudioMs: 1800, WordsMs: 100, Utterance: 1})
	s.eventually(func() bool {
		return slices.ContainsFunc(s.reported(), func(event Event) bool {
			d, ok := event.(Decided)
			return ok && d.Kind == string(ActShorten)
		})
	}, "the shortening was not acted on")
	pauseAsker{s.agent}.Unwritten(participant, 1, "")
	s.Equal([]wordFate{fateLost}, s.metFate(digits))

	s.model.reply = []string{"Thanks."}
	s.finalOf(participant, 2, "Fourteen Cedar Lane.")
	s.Equal([]wordFate{fateAnswered}, s.metFate("Fourteen Cedar Lane."))
	s.False(slices.ContainsFunc(s.model.requests(), func(request llm.ResponseParams) bool {
		return slices.ContainsFunc(request.Input, func(message llm.Message) bool { return strings.Contains(message.Content, "ninety-eight") })
	}), "the digits reached the model")
}
