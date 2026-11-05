//go:build integration

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
)

// The scenarios past the acceptance checks: what real calls do that TestAudioFloorReplay
// does not play. Each is a call of its own against the server ROUTER_CHORUS_ASR_URL names,
// as the replay plays them, and logs a row of what the loop did and how long it took: from
// the end of the caller's speech to the server's ruling, the turn answered and the reply
// heard, and over the agent from the caller's first word to the agent stopping.
//
//	ROUTER_CHORUS_ASR_URL=http://127.0.0.1:8080 \
//	  go test -tags integration -count=1 -timeout 30m -run 'TestAudioFloorScenarios/S1' -v ./internal/agent/
func TestAudioFloorScenarios(t *testing.T) {
	url := os.Getenv("ROUTER_CHORUS_ASR_URL")
	if url == "" {
		t.Skip("needs ROUTER_CHORUS_ASR_URL, a chorus-asr server")
	}
	h := newFloorHarness(t, url)
	h.warm(t)
	s := &scenarios{floorHarness: h, clips: map[string][]int16{}}
	for _, name := range []string{"pauses", "pauses-long", "noporto", "nomeant", "second", "andporto", "thinking", "more", "porto-pt"} {
		s.clips[name] = fixture(t, name)
	}
	h.lines = map[string][]int16{listReply: fixture(t, "list"), questionReply: fixture(t, "question"), greetingLine: fixture(t, "greeting")}

	t.Run("S1a two pauses of 800 ms inside a turn", func(t *testing.T) { s.pausesInside(t, "pauses") })
	t.Run("S1b two pauses of 1300 ms inside a turn", func(t *testing.T) { s.pausesInside(t, "pauses-long") })
	t.Run("S2a a correction as the reply begins", func(t *testing.T) { s.overTheReply(t, "noporto", 500*time.Millisecond, replyText) })
	t.Run("S2b a correction with a pause in it as the reply begins", func(t *testing.T) { s.overTheReply(t, "nomeant", 500*time.Millisecond, replyText) })
	t.Run("S3 a question while the agent reads a list", func(t *testing.T) { s.overTheReply(t, "second", 4*time.Second, listReply) })
	t.Run("S4a a second turn before the reply is heard", s.secondTurnBeforeTheReply)
	t.Run("S4b a second turn over the reply", func(t *testing.T) { s.overTheReply(t, "andporto", 2*time.Second, replyText) })
	t.Run("S5 silence after the agent asks a question", s.silenceAfterAQuestion)
	t.Run("S6 a thinking murmur inside a long pause", func(t *testing.T) { s.pausesInside(t, "thinking") })
	t.Run("S7 a cough over the agent", s.coughOverTheAgent)
	t.Run("S8 the caller goes on after being answered", func(t *testing.T) { s.overTheReply(t, "more", time.Second, replyText) })
	t.Run("S9a the server slow, a decision over the interval", s.serverSlow)
	t.Run("S9b the server answering 503 to decisions", s.server503Decide)
	t.Run("S9c the server answering 503 to the words alone", s.server503Transcribe)
	t.Run("S10a the agent speaks first, the caller waits", func(t *testing.T) { s.agentSpeaksFirst(t, 0) })
	t.Run("S10b the agent speaks first, the caller talks over the greeting", func(t *testing.T) { s.agentSpeaksFirst(t, time.Second) })
	t.Run("S11 a monologue of over 60 s", s.monologue)
	t.Run("S12a the caller switches to Portuguese, callers speak en", func(t *testing.T) { s.languageSwitch(t, []string{"en"}) })
	t.Run("S12b the caller switches to Portuguese, no languages", func(t *testing.T) { s.languageSwitch(t, nil) })
}

// listReply is a reply long enough to ask a question into, which the model is told the agent
// is saying when the caller does.
const listReply = "Here are the forecasts you asked for. One, Lisbon on Friday: sunny and warm, twenty four degrees. " +
	"Two, Porto on Saturday: cloudy in the morning, clearing by noon, nineteen degrees. " +
	"Three, Coimbra on Sunday: a chance of showers in the afternoon, twenty one degrees. " +
	"Four, Aveiro on Monday: windy by the lagoon, with sun in the afternoon."

// questionReply is a reply that ends on a question to the caller.
const questionReply = "Tomorrow in Lisbon it will be sunny and warm, with a high of twenty four degrees. Would you like the forecast for Porto as well?"

// greetingLine is what the agent opens a call with when it speaks first.
const greetingLine = "Hello, you have reached the weather line. How can I help you today?"

type scenarios struct {
	*floorHarness
	clips map[string][]int16
}

func (s *scenarios) clip(name string) []int16 { return s.clips[name] }

// keyword is a word of each clip that what the conversation heard of it must carry.
var keyword = map[string]string{"noporto": "porto", "nomeant": "porto", "second": "second", "andporto": "porto", "more": "umbrella"}

// row logs what a scenario measured, one line a reader can collect into a table.
func row(t *testing.T, format string, args ...any) {
	t.Helper()
	t.Logf("row: %s", fmt.Sprintf(format, args...))
}

// stoppedAfter is how long after said began the agent stopped or shortened its reply, if it
// did.
func (c *call) stoppedAfter(said *cue) (time.Duration, string, bool) {
	for _, i := range all[Interrupted](c) {
		if !i.At.Before(said.began()) {
			return i.At.Sub(said.began()), "stop", true
		}
	}
	for _, o := range all[OverlapDecided](c) {
		if o.E.Action == "shorten" && !o.At.Before(said.began()) {
			return o.At.Sub(said.began()), "shorten", true
		}
	}
	return 0, "", false
}

// sharesRun says text has n words in a row that reference has in a row too: the agent's
// voice written into the caller's words, word for word. A caller's own words share two or
// three with a reply about the same thing ("in Lisbon tomorrow"), which echoed, lenient by
// design, would take for an echo.
func sharesRun(text, reference string, n int) bool {
	heard, said := strings.Fields(wordsOfText(text)), wordsOfText(reference)
	for at := 0; at+n <= len(heard); at++ {
		if strings.Contains(" "+said+" ", " "+strings.Join(heard[at:at+n], " ")+" ") {
			return true
		}
	}
	return false
}

// firstRulingOver is when the server first ruled stop or shorten over the agent on words
// that were not the agent's own echo of reply alone, its decision's first line, at or after
// since: the first ruling the agent had anything of the caller's to act on.
func (c *call) firstRulingOver(since time.Time, reply string) time.Time {
	for _, r := range c.requests() {
		if r.path != "/v1/decide" || r.firstLine.IsZero() || r.firstLine.Before(since) {
			continue
		}
		var answer struct {
			Ruling *struct {
				Floor string `json:"floor"`
			} `json:"ruling"`
		}
		var words struct {
			Transcript *struct {
				Text string `json:"text"`
			} `json:"transcript"`
		}
		first, rest, _ := strings.Cut(r.body, "\n")
		if json.Unmarshal([]byte(first), &answer) != nil || answer.Ruling == nil || (answer.Ruling.Floor != "stop" && answer.Ruling.Floor != "shorten") {
			continue
		}
		if json.Unmarshal([]byte(strings.TrimSpace(rest)), &words) != nil || words.Transcript == nil {
			continue
		}
		if left, _ := echoed(words.Transcript.Text, reply); strings.TrimSpace(left) != "" {
			return r.firstLine
		}
	}
	return time.Time{}
}

// answers are the turns answered, in order.
func (c *call) answers() []stampedEvent[Decided] {
	return c.decisionsBetween(c.began, time.Now().Add(time.Hour), string(ActAnswer))
}

// S1, S6. A turn with pauses inside it: one answer, after the caller has finished, to the
// whole of it. The 1300 ms pauses and the thinking murmur are past the safe point's 1100 ms
// bar drop, where the policy was measured to cut in on 4.5% of mid-turn pauses: what it does
// here is recorded, and only a cut-in on the 800 ms pauses fails.
func (s *scenarios) pausesInside(t *testing.T, name string) {
	c := s.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	said := c.edge.say(name, s.clip(name))
	c.edge.pause(5 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	early := c.decisionsBetween(said.began(), said.ended(), string(ActAnswer))
	for _, d := range early {
		t.Logf("answered at %s, before the caller finished: %q", c.since(d.At), d.E.Text)
	}
	answered := c.answered(said)
	answers := c.answers()
	finals := c.finals()
	var texts []string
	for _, f := range finals {
		texts = append(texts, f.T.Text)
	}
	row(t, "%s: %d answers (%d before the caller finished), %d finals %q; ruling %s, answered %s, reply %s after the voice",
		name, len(answers), len(early), len(finals), texts,
		since(answered.speechEnd, answered.ruled), since(answered.speechEnd, answered.decided), since(answered.speechEnd, answered.replied))
	assert.Empty(t, all[Error](c), "no failure")
	assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer")
	assert.True(t, answered.ok, "the turn was answered")
	if name == "pauses" {
		assert.Empty(t, early, "no cut-in on an 800 ms pause")
		assert.Len(t, answers, 1, "one answer")
		if assert.Len(t, finals, 1, "one final") {
			assert.Contains(t, wordsOfText(finals[0].T.Text), "lisbon")
			assert.Contains(t, wordsOfText(finals[0].T.Text), "raining", "the final has the turn's end")
		}
	}
	s.measured(t, name, answered)
}

// S2, S3, S4b, S8. The caller speaks over the reply, after into it, with the reply echoed at
// -30 dB as a phone without echo cancellation sends it: the agent stops or shortens within
// stopWithin of their first word, on the model's ruling, and what they said is answered once,
// with none of the agent's voice in it.
func (s *scenarios) overTheReply(t *testing.T, name string, after time.Duration, reply string) {
	c := s.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.model.mu.Lock()
	c.model.reply = []string{reply}
	c.model.mu.Unlock()
	c.edge.background(nil, echoAt(-30))
	c.edge.pause(time.Second)
	c.edge.say("weather", s.weather)
	c.until(func() bool { return len(c.edge.replyStarts()) > 0 }, 20*time.Second, "the reply began")
	time.Sleep(after)
	over := c.edge.say(name, s.clip(name))
	c.until(func() bool { return c.answered(over).ok }, 30*time.Second, "what the caller said over the agent was answered")
	c.settle(4*time.Second, time.Minute)
	defer c.report()

	took, how, stopped := c.stoppedAfter(over)
	ruledOver := c.firstRulingOver(over.began(), reply)
	answered := c.answered(over)
	answers := c.decisionsBetween(over.began(), time.Now().Add(time.Hour), string(ActAnswer))
	var heard []string
	for _, h := range all[Heard](c) {
		if !h.At.Before(over.began()) {
			heard = append(heard, h.E.Text)
		}
	}
	row(t, "%s %s into the reply: the model ruled against the reply %s and the agent gave way (%s) %s after the first word; %d answers to it, heard %q; ruling %s, answered %s, reply %s after the voice",
		name, after, since(over.began(), ruledOver), how, took.Round(time.Millisecond), len(answers), heard,
		since(answered.speechEnd, answered.ruled), since(answered.speechEnd, answered.decided), since(answered.speechEnd, answered.replied))
	assert.Empty(t, all[Error](c), "no failure")
	if assert.True(t, stopped, "the agent stopped or shortened") && assert.False(t, ruledOver.IsZero(), "the model ruled against the reply") {
		// The loop's own part: the model's first ruling against the reply whose words carry
		// any of the caller's is acted on within the policy's interval. One on the echo alone
		// stops nothing, and how soon the model rules is its own.
		acted := over.began().Add(took).Sub(ruledOver)
		assert.LessOrEqual(t, acted, time.Duration(s.policy.IntervalMs)*time.Millisecond, "the agent gave way %s after the model's first ruling against the reply", acted.Round(time.Millisecond))
	}
	assert.True(t, answered.ok, "what the caller said over the agent was answered")
	assert.Contains(t, wordsOfText(strings.Join(heard, " ")), keyword[name], "what the caller said reached the conversation")
	for _, line := range heard {
		assert.False(t, sharesRun(line, reply, 4), "the agent's voice in what was heard over it: %q", line)
	}
	assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer")
	s.measured(t, name, answered)
}

// S4a. A second question 400 ms after the first, before any of the reply has been heard: the
// reply to the first is dropped as unheard, and the caller is answered about both.
func (s *scenarios) secondTurnBeforeTheReply(t *testing.T) {
	c := s.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	first := c.edge.say("weather", s.weather)
	c.edge.pause(400 * time.Millisecond)
	second := c.edge.say("andporto", s.clip("andporto"))
	c.edge.pause(6 * time.Second)
	c.settle(3*time.Second, time.Minute)
	defer c.report()

	answers := c.answers()
	replies := all[Responding](c)
	var prompts []string
	for _, r := range replies {
		prompts = append(prompts, r.E.Prompt)
	}
	answered := c.answered(second)
	row(t, "two turns %s apart: %d answers, %d replies started %q, %d replies heard, %d interrupted; the second ruled %s, answered %s, reply %s after its voice",
		second.began().Sub(first.ended()).Round(time.Millisecond), len(answers), len(replies), prompts, len(c.edge.replyStarts()), len(all[Interrupted](c)),
		since(answered.speechEnd, answered.ruled), since(answered.speechEnd, answered.decided), since(answered.speechEnd, answered.replied))
	assert.Empty(t, all[Error](c), "no failure")
	assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer")
	assert.Contains(t, wordsOfText(strings.Join(prompts, " ")), "porto", "the second question was answered")
	assert.LessOrEqual(t, len(c.edge.replyStarts()), 2, "at most two replies heard")
	s.measured(t, "S4a", answered)
}

// S5. The agent's reply ends on a question, echoed back at -30 dB, and the caller says
// nothing for 8 s: nothing is written of the silence, no request goes to the server once the
// echo has been written off, and the caller is not answered again.
func (s *scenarios) silenceAfterAQuestion(t *testing.T) {
	c := s.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.model.mu.Lock()
	c.model.reply = []string{questionReply}
	c.model.mu.Unlock()
	c.edge.background(nil, echoAt(-30))
	c.edge.pause(time.Second)
	said := c.edge.say("weather", s.weather)
	c.until(func() bool { return len(c.edge.replyStarts()) > 0 }, 20*time.Second, "the reply began")
	c.until(func() bool { return !c.edge.SpeechPending() && !c.agent.Busy() }, 30*time.Second, "the reply ended")
	replyEnded := time.Now()
	c.edge.pause(8 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	quietFrom := replyEnded.Add(3 * time.Second)
	var late []string
	for _, r := range c.requests() {
		if r.start.After(quietFrom) {
			late = append(late, fmt.Sprintf("%s %s at %s", r.method, r.path, c.since(r.start)))
		}
	}
	written := c.wordsBetween(quietFrom, time.Now())
	answers := c.answers()
	row(t, "after the question: %d answers, %d transcripts and %d requests more than 3 s into the silence %v, %d caller lines in the history",
		len(answers), len(written), len(late), late, len(c.callerLines()))
	assert.Empty(t, all[Error](c), "no failure")
	assert.Len(t, answers, 1, "the caller was answered once, and not again for the silence")
	assert.Empty(t, late, "no request to the server once the line fell silent")
	assert.Empty(t, written, "nothing written of the silence")
	assert.Len(t, c.callerLines(), 1, "one line of the caller's in the history")
	s.measured(t, "S5", c.answered(said))
}

// S7. A cough over the reply: the agent is not stopped, and nothing is answered.
func (s *scenarios) coughOverTheAgent(t *testing.T) {
	c := s.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.background(nil, echoAt(-30))
	c.edge.pause(time.Second)
	c.edge.say("weather", s.weather)
	c.until(func() bool { return len(c.edge.replyStarts()) > 0 }, 20*time.Second, "the reply began")
	time.Sleep(2 * time.Second)
	c.edge.say("cough", cough())
	c.settle(4*time.Second, time.Minute)
	defer c.report()

	row(t, "a cough over the reply: %d answers, %d interrupted, %d dropped, heard %s of %s",
		len(c.answers()), len(all[Interrupted](c)), c.edge.dropped(), c.edge.heard().Round(time.Millisecond), c.voice.replyLength())
	assert.Len(t, c.answers(), 1, "one answer, none to the cough")
	assert.Empty(t, all[Interrupted](c), "the reply was not interrupted")
	assert.Zero(t, c.edge.dropped(), "none of the reply was thrown away")
	assert.InDelta(t, float64(c.voice.replyLength()), float64(c.edge.heard()), float64(300*time.Millisecond), "the whole reply was heard")
}

// S9a. The server slow: every decision delayed by 400 ms, over the policy's 300 ms interval
// and inside the pause's 1 s deadline. The checks come late and still rule: the turn is
// answered on the model's ruling, later.
func (s *scenarios) serverSlow(t *testing.T) {
	s.faults.set("/v1/decide", 400*time.Millisecond, 0, 0)
	defer s.faults.clear()
	c := s.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	said := c.edge.say("rain", s.rain)
	c.edge.pause(5 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	answered := c.answered(said)
	row(t, "decisions 400 ms late: %d answers; ruling %s, answered %s, reply %s after the voice; Hearer missed %d",
		len(c.answers()), since(answered.speechEnd, answered.ruled), since(answered.speechEnd, answered.decided), since(answered.speechEnd, answered.replied), len(c.hearerMissed()))
	assert.Empty(t, all[Error](c), "no failure")
	assert.True(t, answered.ok, "the turn was answered")
	assert.Len(t, c.answers(), 1, "one answer")
	assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer, late")
	s.measured(t, "S9a", answered)
}

// S9b. The server shedding load: the first two decisions are answered 503. The pause falls
// to the words alone and the flow controller, which still answers the caller once, and the
// next turn is ruled by the model again.
func (s *scenarios) server503Decide(t *testing.T) {
	s.faults.set("/v1/decide", 0, http.StatusServiceUnavailable, 2)
	c := s.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	first := c.edge.say("weather", s.weather)
	c.edge.pause(6 * time.Second)
	c.settle(2*time.Second, time.Minute)
	s.faults.clear()
	second := c.edge.say("rain", s.rain)
	c.edge.pause(4 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	answers := c.answers()
	a1, a2 := c.answered(first), c.answered(second)
	row(t, "503 to two decisions: %d answers; the first answered %s after the voice (Hearer missed %d), the second ruled %s, answered %s",
		len(answers), since(a1.speechEnd, a1.decided), len(c.hearerMissed()), since(a2.speechEnd, a2.ruled), since(a2.speechEnd, a2.decided))
	assert.Empty(t, all[Error](c), "no failure")
	assert.Len(t, answers, 2, "each turn answered once")
	assert.True(t, a1.ok, "the first turn was answered, by the flow controller")
	assert.True(t, a2.ok && !a2.ruled.IsZero(), "the second turn was ruled by the model")
	s.measured(t, "S9b", a2)
}

// S9c. The server refusing the words alone: every POST /v1/transcribe is 503, as a server
// short of transcription lanes answers. Interims are refused and not asked again; the turn's
// words come with the pause's decision, and the turn is answered as ever.
func (s *scenarios) server503Transcribe(t *testing.T) {
	s.faults.set("/v1/transcribe", 0, http.StatusServiceUnavailable, 1<<20)
	defer s.faults.clear()
	c := s.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	said := c.edge.say("long", s.long)
	c.edge.pause(5 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	var interims int
	for _, w := range c.words.all() {
		if w.T.Interim {
			interims++
		}
	}
	answered := c.answered(said)
	finals := c.finals()
	row(t, "503 to the words alone: %d transcribe requests refused, %d interims shown, %d finals, ruling %s, answered %s after the voice",
		len(c.requestsTo("/v1/transcribe")), interims, len(finals), since(answered.speechEnd, answered.ruled), since(answered.speechEnd, answered.decided))
	assert.Empty(t, all[Error](c), "no failure")
	assert.True(t, answered.ok, "the turn was answered")
	if assert.Len(t, finals, 1, "one final") {
		assert.Contains(t, wordsOfText(finals[0].T.Text), "here is my situation", "the final has the turn's beginning")
	}
	assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer")
	s.measured(t, "S9c", answered)
}

// S10. The agent speaks first, its greeting echoed at -30 dB. The caller waits for it to end
// (0), or talks over it after into it: either way the question is answered once, the greeting
// never answered as the caller's words.
func (s *scenarios) agentSpeaksFirst(t *testing.T, into time.Duration) {
	c := s.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.background(nil, echoAt(-30))
	require.NoError(t, c.agent.Greet(context.Background(), greetingLine, false))
	c.until(func() bool { return len(c.edge.replyStarts()) > 0 }, 20*time.Second, "the greeting began")
	if into == 0 {
		c.until(func() bool { return !c.edge.SpeechPending() && !c.agent.Busy() }, 30*time.Second, "the greeting ended")
		c.edge.pause(time.Second)
	} else {
		time.Sleep(into)
	}
	said := c.edge.say("weather", s.weather)
	c.edge.pause(6 * time.Second)
	c.settle(3*time.Second, time.Minute)
	defer c.report()

	answered := c.answered(said)
	took, how, stopped := c.stoppedAfter(said)
	row(t, "greeting first, the caller %s: %d answers, %d caller lines %q, greeting %s %s; ruling %s, answered %s, reply %s after the voice",
		map[bool]string{true: "over it at " + into.String(), false: "after it"}[into > 0], len(c.answers()), len(c.callerLines()), c.callerLines(),
		map[bool]string{true: how + " after", false: "left going"}[stopped], took.Round(time.Millisecond),
		since(answered.speechEnd, answered.ruled), since(answered.speechEnd, answered.decided), since(answered.speechEnd, answered.replied))
	assert.Empty(t, all[Error](c), "no failure")
	assert.True(t, answered.ok, "the question was answered")
	assert.Len(t, c.answers(), 1, "one answer")
	for _, line := range c.callerLines() {
		assert.False(t, sharesRun(line, greetingLine, 4), "the greeting in the caller's line of the history: %q", line)
	}
	assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer")
	s.measured(t, "S10", answered)
}

// S11. A monologue of over 60 s without a pause the policy would check: the long turn four
// times over with 150 ms between, 80 s. The transcriber ends an utterance at 60 s as it is,
// so the caller is answered then, while still talking (Limits), and the rest is another
// turn: what is recorded is that nothing is lost and nothing fails.
func (s *scenarios) monologue(t *testing.T) {
	c := s.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	gap := make([]int16, replayRate*150/1000)
	said := c.edge.say("monologue", slices.Concat(s.long, gap, s.long, gap, s.long, gap, s.long))
	c.edge.pause(6 * time.Second)
	c.settle(3*time.Second, 2*time.Minute)
	defer c.report()

	finals := c.finals()
	var lengths []string
	for _, f := range finals {
		lengths = append(lengths, fmt.Sprintf("#%d %.1fs/%d words", f.T.Utterance, f.T.AudioDurationMs/1000, len(strings.Fields(f.T.Text))))
	}
	answers := c.answers()
	var at []string
	for _, a := range answers {
		at = append(at, a.At.Sub(said.began()).Round(time.Millisecond).String())
	}
	row(t, "a %s monologue: %d finals %v, %d answers at %v after the first word, %d interrupted, %d errors",
		said.ended().Sub(said.began()).Round(time.Second), len(finals), lengths, len(answers), at, len(all[Interrupted](c)), len(all[Error](c)))
	assert.Empty(t, all[Error](c), "no failure")
	assert.NotEmpty(t, finals, "the monologue was written")
	assert.GreaterOrEqual(t, len(answers), 1, "the caller was answered")
	if len(finals) > 0 {
		assert.Contains(t, wordsOfText(finals[0].T.Text), "here is my situation", "the first final has the beginning")
		assert.Contains(t, wordsOfText(finals[len(finals)-1].T.Text), "whole weekend", "the last final has the end")
	}
}

// S12. The caller asks in English, is answered, and asks the next question in Portuguese:
// with en declared the server writes nothing of a language the callers don't speak, or
// writes it; with none it writes the Portuguese and the caller is answered. Recorded either
// way; what fails is a failure, a turn in another script, or with no languages the question
// unanswered.
func (s *scenarios) languageSwitch(t *testing.T, languages []string) {
	c := s.start(t, audiofloor.Act, replayInstructions, languages)
	c.edge.pause(time.Second)
	c.edge.say("weather", s.weather)
	c.edge.pause(4 * time.Second)
	c.settle(2*time.Second, time.Minute)
	said := c.edge.say("porto-pt", s.clip("porto-pt"))
	c.edge.pause(5 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	var written []string
	for _, w := range c.wordsBetween(said.began(), time.Now()) {
		written = append(written, fmt.Sprintf("%s %q (%s)", map[bool]string{true: "interim", false: string(w.T.Mode)}[w.T.Interim], w.T.Text, w.T.Language))
	}
	answered := c.answered(said)
	row(t, "Portuguese after English, languages %v: written %v; %d answers in all; ruling %s, answered %s after the voice",
		languages, written, len(c.answers()), since(answered.speechEnd, answered.ruled), since(answered.speechEnd, answered.decided))
	assert.Empty(t, all[Error](c), "no failure")
	for _, heard := range all[Heard](c) {
		assert.False(t, foreign(heard.E.Text), "a turn in another script: %q", heard.E.Text)
	}
	if languages == nil {
		assert.True(t, answered.ok, "the Portuguese question was answered")
		assert.Contains(t, wordsOfText(strings.Join(c.callerLines(), " ")), "porto", "the Portuguese heard: %q", c.callerLines())
	}
	s.measured(t, "S12", answered)
}

// faults sits between the router and the server: requests to path are delayed by delay, and
// the first failing of them are answered status without reaching the server, as a server
// shedding load answers.
type faults struct {
	base http.RoundTripper
	// prefix is the server's own path, which path leaves out.
	prefix string

	mu      sync.Mutex
	path    string
	delay   time.Duration
	status  int
	failing int
}

func (f *faults) set(path string, delay time.Duration, status, failing int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.path, f.delay, f.status, f.failing = path, delay, status, failing
}

func (f *faults) clear() { f.set("", 0, 0, 0) }

func (f *faults) RoundTrip(req *http.Request) (*http.Response, error) {
	f.mu.Lock()
	hit := strings.TrimPrefix(req.URL.Path, f.prefix) == f.path
	delay := f.delay
	fail := hit && f.failing > 0
	if fail {
		f.failing--
	}
	status := f.status
	f.mu.Unlock()
	if fail {
		return &http.Response{
			StatusCode: status, Status: fmt.Sprintf("%d %s", status, http.StatusText(status)), Proto: "HTTP/1.1", ProtoMajor: 1, ProtoMinor: 1,
			Header: http.Header{"Content-Type": []string{"application/json"}}, Request: req,
			Body: io.NopCloser(strings.NewReader(`{"error":"every lane is busy"}`)),
		}, nil
	}
	if hit && delay > 0 {
		select {
		case <-time.After(delay):
		case <-req.Context().Done():
			return nil, req.Context().Err()
		}
	}
	return f.base.RoundTrip(req)
}
