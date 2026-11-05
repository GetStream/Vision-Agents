//go:build integration

package agent

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
)

// The loop's contract (deploy/chorus-asr/README.md, "The loop's contract"), a check a line,
// each a call of its own against the server ROUTER_CHORUS_ASR_URL names, played as the replay
// plays its calls. What each check expects is what the contract says a caller can count on;
// where the model rules otherwise the check fails, and its timeline says what the model said.
//
//	ROUTER_CHORUS_ASR_URL=http://127.0.0.1:8080 GOEXPERIMENT=simd \
//	  go test -tags integration -count=1 -timeout 30m -run 'TestAudioFloorContract/C2' -v ./internal/agent/
func TestAudioFloorContract(t *testing.T) {
	url := os.Getenv("ROUTER_CHORUS_ASR_URL")
	if url == "" {
		t.Skip("needs ROUTER_CHORUS_ASR_URL, a chorus-asr server")
	}
	h := newFloorHarness(t, url)
	h.warm(t)
	k := &contract{floorHarness: h, clips: map[string][]int16{}}
	for _, name := range []string{"mmhmm", "hmhm", "okay", "right", "yeah", "laugh", "stoponly", "wait", "holdon", "andlisbon",
		"stopporto", "weatherporto", "dontstop", "memberid", "andporto", "noporto"} {
		k.clips[name] = fixture(t, name)
	}
	h.lines = map[string][]int16{listReply: fixture(t, "list"), questionReply: fixture(t, "question")}

	for _, name := range []string{"mmhmm", "hmhm", "okay", "right", "yeah", "laugh"} {
		t.Run("C1 "+name+" over the agent is ignored", func(t *testing.T) { k.ignored(t, name, k.clips[name], nil, 0) })
	}
	t.Run("C1 the agent's own echo at -20 dB is ignored", func(t *testing.T) { k.ignored(t, "", nil, nil, echoAt(-20)) })
	t.Run("C1 the room in another language at -30 dBFS is ignored", func(t *testing.T) {
		k.ignored(t, "", nil, bedAt(h.portuguese, -30), 0)
	})
	for _, name := range []string{"stoponly", "wait", "holdon"} {
		t.Run("C2 "+name+" stops the agent, which stays quiet", func(t *testing.T) { k.stopsAndQuiet(t, name) })
	}
	t.Run("C3 and in Lisbon over a forecast is answered", func(t *testing.T) { k.stopsAndAnswered(t, "andlisbon", "lisbon", false) })
	t.Run("C3 stop, what about Porto in one breath is answered once, about Porto", func(t *testing.T) {
		k.stopsAndAnswered(t, "stopporto", "porto", true)
	})
	t.Run("C3 stop, then a question 1 s later, the question answered", k.stopThenQuestion)
	t.Run("C4 a shortened reply stops within 250 ms of the ruling", func(t *testing.T) { k.shortened(t, "andporto") })
	t.Run("C4 a check-in talked over stops", k.checkInTalkedOver)
	t.Run("C5 don't stop over the agent: one ruling acted on, no second wait", func(t *testing.T) { k.oneWait(t, "dontstop") })
	t.Run("C5 no, Porto over the agent: answered at the point, no second wait", func(t *testing.T) { k.oneWait(t, "noporto") })
	t.Run("C6 no check-in while the caller counts slowly, interims off", k.noCheckInMidTurn)
	t.Run("C6 a member ID dictated with pauses is one turn", k.dictation)
	t.Run("C7 one final, one answer, at once, ending at the first check", k.oneFinal)
}

// shortenTail is how much of the agent's speech a shortening lets finish (README, "The loop's
// contract"): a stop has it quiet within stopWithin of the caller's first word.
const shortenTail = 250 * time.Millisecond

// frameSlack is the frame the edge was playing when the agent was stopped, and the next.
const frameSlack = 2 * replayFrameTime

type contract struct {
	*floorHarness
	clips map[string][]int16
}

// overTheList starts a call asked about the weather and answered with the forecast list, 22 s
// long, and says what was given after into it.
func (k *contract) overTheList(t *testing.T, after time.Duration) *call {
	c := k.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.model.mu.Lock()
	c.model.reply = []string{listReply}
	c.model.mu.Unlock()
	c.edge.pause(time.Second)
	c.edge.say("weather", k.weather)
	c.until(func() bool { return len(c.edge.replyStarts()) > 0 }, 20*time.Second, "the reply began")
	time.Sleep(after)
	return c
}

// C1. Said over the agent, or heard on the line while it speaks, and not for it: the reply
// plays whole, nothing is stopped or shortened, and nothing is answered but the question.
func (k *contract) ignored(t *testing.T, name string, said []int16, bed []float64, echo float64) {
	c := k.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	c.edge.say("weather", k.weather)
	c.until(func() bool { return len(c.edge.replyStarts()) > 0 }, 20*time.Second, "the reply began")
	c.edge.background(bed, echo)
	time.Sleep(2 * time.Second)
	if said != nil {
		c.edge.say(name, said)
	}
	c.until(func() bool { return !c.edge.SpeechPending() && !c.agent.Busy() }, 30*time.Second, "the reply ended")
	c.edge.background(nil, 0)
	c.settle(3*time.Second, time.Minute)
	defer c.report()

	var shortened int
	for _, o := range all[OverlapDecided](c) {
		if o.E.Action == "shorten" {
			shortened++
		}
	}
	assert.Empty(t, all[Error](c), "no failure")
	assert.Len(t, c.answers(), 1, "one answer, the question's")
	assert.Empty(t, all[Interrupted](c), "the reply was not stopped")
	assert.Zero(t, shortened, "the reply was not shortened")
	assert.Zero(t, c.edge.dropped(), "none of the reply was thrown away")
	assert.InDelta(t, float64(c.voice.replyLength()), float64(c.edge.heard()), float64(300*time.Millisecond),
		"the whole reply was heard: %s of %s", c.edge.heard(), c.voice.replyLength())
}

// interrupted is the first interruption from at on.
func (c *call) interrupted(at time.Time) (stampedEvent[Interrupted], bool) {
	for _, i := range all[Interrupted](c) {
		if !i.At.Before(at) {
			return i, true
		}
	}
	return stampedEvent[Interrupted]{}, false
}

// C2. "Stop", "wait", "hold on" 2 s into the list: the agent is quiet within stopWithin of the
// first word, none of what it was saying is heard after, and nothing is answered for 5 s.
func (k *contract) stopsAndQuiet(t *testing.T, name string) {
	c := k.overTheList(t, 2*time.Second)
	over := c.edge.say(name, k.clips[name])
	c.edge.pause(6 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	assert.Empty(t, all[Error](c), "no failure")
	stopped, ok := c.interrupted(over.began())
	if !assert.True(t, ok, "the agent was stopped") {
		return
	}
	took := stopped.At.Sub(over.began())
	assert.LessOrEqual(t, took, stopWithin, "the agent stopped %s after the caller's first word", took.Round(time.Millisecond))
	quietFrom := stopped.At.Add(frameSlack)
	assert.Empty(t, c.edge.heardBetween(quietFrom, quietFrom.Add(5*time.Second)), "the agent was heard after it stopped")
	assert.Empty(t, c.decisionsAfter(over.began(), string(ActAnswer)), "nothing answered after the stop")
	assert.Len(t, all[Responding](c), 1, "one reply, the question's")
}

// C3. A correction or a request over the forecast list: the agent gives way, and the caller is
// answered once, on what they said after the words the stop was ruled on. oneBreath says the
// line is said with no gap the policy's first check would catch.
func (k *contract) stopsAndAnswered(t *testing.T, name, keyword string, oneBreath bool) {
	c := k.overTheList(t, 4*time.Second)
	said := k.clips[name]
	if oneBreath {
		said = withGapsUnder(said, 150*time.Millisecond)
	}
	over := c.edge.say(name, said)
	c.until(func() bool { return c.answered(over).ok }, 30*time.Second, "what the caller said over the agent was answered")
	c.settle(4*time.Second, time.Minute)
	defer c.report()

	assert.Empty(t, all[Error](c), "no failure")
	took, how, gaveWay := c.stoppedAfter(over)
	if assert.True(t, gaveWay, "the agent stopped or shortened") {
		assert.LessOrEqual(t, took, stopWithin, "the agent gave way (%s) %s after the caller's first word", how, took.Round(time.Millisecond))
	}
	answers := c.decisionsAfter(over.began(), string(ActAnswer))
	if assert.Len(t, answers, 1, "answered once") {
		assert.Contains(t, wordsOfText(answers[0].E.Text), keyword, "the answer is to what the caller asked")
	}
	replies := all[Responding](c)
	if assert.Len(t, replies, 2, "the question's reply, and one to what was said over it") {
		assert.Contains(t, wordsOfText(replies[1].E.Prompt), keyword)
	}
	assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer")
	k.measured(t, "C3 "+name, c.answered(over))
}

// C3. "Stop." over the list, then 1 s later "What's the weather in Porto?": the agent stops,
// and the question is answered once; "stop" is not.
func (k *contract) stopThenQuestion(t *testing.T) {
	c := k.overTheList(t, 2*time.Second)
	stop := c.edge.say("stoponly", k.clips["stoponly"])
	c.edge.pause(time.Second)
	question := c.edge.say("weatherporto", k.clips["weatherporto"])
	c.edge.pause(5 * time.Second)
	c.settle(3*time.Second, time.Minute)
	defer c.report()

	assert.Empty(t, all[Error](c), "no failure")
	stopped, ok := c.interrupted(stop.began())
	if assert.True(t, ok, "the agent was stopped") {
		assert.LessOrEqual(t, stopped.At.Sub(stop.began()), stopWithin, "stopped within the budget")
	}
	answers := c.decisionsAfter(stop.began(), string(ActAnswer))
	if assert.Len(t, answers, 1, "one answer after the stop") {
		assert.Contains(t, wordsOfText(answers[0].E.Text), "porto", "the question was answered")
	}
	answered := c.answered(question)
	assert.True(t, answered.ok, "the question was answered")
	assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer")
	k.measured(t, "C3 stop then question", answered)
}

// C4. One more item over the forecast list: the agent stops speaking within shortenTail of
// the model's shortening, or within stopWithin of the first word if the model stops it, and
// the item is answered.
func (k *contract) shortened(t *testing.T, name string) {
	c := k.overTheList(t, 4*time.Second)
	over := c.edge.say(name, k.clips[name])
	c.until(func() bool { return c.answered(over).ok }, 30*time.Second, "the item was answered")
	c.settle(4*time.Second, time.Minute)
	defer c.report()

	assert.Empty(t, all[Error](c), "no failure")
	took, how, gaveWay := c.stoppedAfter(over)
	if !assert.True(t, gaveWay, "the agent stopped or shortened") {
		return
	}
	gave := over.began().Add(took)
	answered := c.answered(over)
	last := c.edge.heardBetween(gave, answered.replied)
	t.Logf("the model %s the reply %s after the first word; the agent was last heard %d frames after", how, took.Round(time.Millisecond), len(last))
	bound := shortenTail + frameSlack
	if how == "stop" {
		bound = frameSlack
	}
	if len(last) > 0 {
		assert.LessOrEqual(t, last[len(last)-1].Sub(gave), bound, "the agent went on %s after it gave way (%s)", last[len(last)-1].Sub(gave).Round(time.Millisecond), how)
	}
	assert.True(t, answered.ok, "the item was answered")
	assert.Len(t, c.decisionsAfter(over.began(), string(ActAnswer)), 1, "answered once")
}

// C4. The agent asks a question, the caller says nothing, and the agent checks in after the
// silence with a line of its own. The caller talks over it 1.5 s in: the line stops within
// stopWithin of their first word, whether the model stops or shortens it, and the caller is
// answered.
func (k *contract) checkInTalkedOver(t *testing.T) {
	c := k.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.model.mu.Lock()
	c.model.reply = []string{questionReply}
	c.model.then = []string{replyText}
	c.model.mu.Unlock()
	c.edge.pause(time.Second)
	c.edge.say("weather", k.weather)
	c.until(func() bool { return len(c.edge.replyStarts()) > 0 }, 20*time.Second, "the reply began")
	c.until(func() bool {
		return len(all[Responded](c)) > 0 && all[Responded](c)[len(all[Responded](c))-1].E.Purpose == PurposeCheckIn
	},
		30*time.Second, "the agent checked in")
	c.until(func() bool { return len(c.edge.replyStarts()) > 1 }, 10*time.Second, "the check-in was heard")
	time.Sleep(1500 * time.Millisecond)
	over := c.edge.say("andporto", k.clips["andporto"])
	c.edge.pause(5 * time.Second)
	c.settle(3*time.Second, time.Minute)
	defer c.report()

	assert.Empty(t, all[Error](c), "no failure")
	took, how, gaveWay := c.stoppedAfter(over)
	if assert.True(t, gaveWay, "the check-in gave way") {
		gave := over.began().Add(took)
		t.Logf("the model %s the check-in %s after the first word", how, took.Round(time.Millisecond))
		next := time.Now()
		if answered := c.answered(over); answered.ok && !answered.replied.IsZero() {
			next = answered.replied
		}
		last := c.edge.heardBetween(gave, next)
		if len(last) > 0 {
			assert.LessOrEqual(t, last[len(last)-1].Sub(over.began()), stopWithin+shortenTail,
				"the check-in went on %s after the caller's first word", last[len(last)-1].Sub(over.began()).Round(time.Millisecond))
		}
	}
	assert.True(t, c.answered(over).ok, "the caller was answered")
}

// C5. Words over the agent that the model rules on at a check: whatever it rules, the loop
// acts on that ruling once. Answered, the turn is answered within a check and a second of the
// caller's voice, with no ruling put again ("giving the caller longer").
func (k *contract) oneWait(t *testing.T, name string) {
	c := k.overTheList(t, 3*time.Second)
	over := c.edge.say(name, k.clips[name])
	c.edge.pause(6 * time.Second)
	c.settle(3*time.Second, time.Minute)
	defer c.report()

	assert.Empty(t, all[Error](c), "no failure")
	_, how, gaveWay := c.stoppedAfter(over)
	answered := c.answered(over)
	t.Logf("gave way: %v (%s); answered: %v %q", gaveWay, how, answered.ok, answered.text)
	var again []logLine
	for _, l := range c.logs.find("giving the caller longer to finish") {
		if !l.At.Before(over.began()) {
			again = append(again, l)
		}
	}
	assert.Empty(t, again, "a ruling put again after the caller spoke over the agent")
	if answered.ok {
		assert.LessOrEqual(t, answered.decided.Sub(answered.speechEnd), k.soon(), "answered %s after the voice", answered.decided.Sub(answered.speechEnd).Round(time.Millisecond))
		k.measured(t, "C5 "+name, answered)
	}
	assert.LessOrEqual(t, len(c.decisionsAfter(over.began(), string(ActAnswer))), 1, "answered at most once")
}

// C6. The agent asks a question, and the caller, after 3 s, counts one to ten a second
// apart, with interims off as a deployment short of transcription lanes has them: the agent
// says nothing of its own accord while the model waits on the count, and answers it once,
// after "ten".
func (k *contract) noCheckInMidTurn(t *testing.T) {
	client, _ := audiofloor.Open(config.Agent{AudioFloorURL: k.url, AudioFloorAPIKey: os.Getenv("ROUTER_CHORUS_ASR_API_KEY"),
		AudioFloorInterims: "off", AudioFloorMode: string(audiofloor.Act)})
	require.NotNil(t, client)
	c := k.startWith(t, callSetup{mode: audiofloor.Act, instructions: replayInstructions, languages: []string{"en"}, client: client})
	c.model.mu.Lock()
	c.model.reply = []string{questionReply}
	c.model.mu.Unlock()
	c.edge.pause(time.Second)
	c.edge.say("weather", k.weather)
	c.until(func() bool { return len(c.edge.replyStarts()) > 0 }, 20*time.Second, "the reply began")
	c.until(func() bool { return !c.edge.SpeechPending() && !c.agent.Busy() }, 30*time.Second, "the reply ended")
	c.edge.pause(3 * time.Second)
	gap := make([]int16, replayRate)
	var counted []int16
	for i, n := range k.count {
		if i > 0 {
			counted = append(counted, gap...)
		}
		counted = append(counted, n...)
	}
	count := c.edge.say("count", counted)
	c.edge.pause(5 * time.Second)
	c.settle(3*time.Second, time.Minute)
	defer c.report()

	assert.Empty(t, all[Error](c), "no failure")
	assert.Empty(t, c.decisionsBetween(count.began(), count.ended(), string(ActCheckIn)), "the agent checked in while the caller counted")
	assert.Empty(t, c.decisionsBetween(count.began(), count.ended(), string(ActAnswer)), "the count was answered before it ended")
	answers := c.decisionsAfter(count.began(), string(ActAnswer))
	if assert.Len(t, answers, 1, "the count answered once") {
		assert.Contains(t, wordsOfText(answers[0].E.Text), "ten", "the answer has the count's end")
	}
}

// C6. "My member ID is K T R [800 ms] 5 8 3 [800 ms] 0 2 7.": one turn, answered after its end.
func (k *contract) dictation(t *testing.T) {
	c := k.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	said := c.edge.say("memberid", k.clips["memberid"])
	c.edge.pause(5 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	assert.Empty(t, all[Error](c), "no failure")
	assert.Empty(t, c.decisionsBetween(said.began(), said.ended(), string(ActAnswer)), "answered before the caller finished")
	answers := c.answers()
	if assert.Len(t, answers, 1, "one answer") {
		assert.Contains(t, strings.ReplaceAll(wordsOfText(answers[0].E.Text), " ", ""), "027", "the answer has the ID's end")
	}
	k.measured(t, "C6 member ID", c.answered(said))
}

// C7. A short question, under bars that end it at the first check ([0 ms: 0.5, 500 ms: 0.3],
// as a live call chose), then silence: one final, one answer, one reply heard, and the answer
// decided as the final arrives.
func (k *contract) oneFinal(t *testing.T) {
	eager := audiofloor.EndOfTurn{Bars: []audiofloor.PolicyBar{{FromMs: 0, Bar: 0.5}, {FromMs: 500, Bar: 0.3}}}
	c := k.startWith(t, callSetup{mode: audiofloor.Act, instructions: replayInstructions, languages: []string{"en"}, endOfTurn: &eager})
	c.edge.pause(time.Second)
	said := c.edge.say("weather", k.weather)
	c.edge.pause(8 * time.Second)
	c.settle(3*time.Second, time.Minute)
	defer c.report()

	assert.Empty(t, all[Error](c), "no failure")
	finals := c.finals()
	assert.Len(t, finals, 1, "one final")
	assert.Len(t, c.answers(), 1, "one answer")
	assert.Len(t, all[Responding](c), 1, "one reply")
	assert.Len(t, c.edge.replyStarts(), 1, "one reply heard")
	answered := c.answered(said)
	if len(finals) == 1 && answered.ok {
		settled := answered.decided.Sub(finals[0].At)
		t.Logf("the final came %s after the voice, the answer %s after the final", finals[0].At.Sub(said.ended()).Round(time.Millisecond), settled.Round(time.Millisecond))
		assert.Less(t, settled, 30*time.Millisecond, "the final waited %s before it was answered", settled.Round(time.Millisecond))
	}
	k.measured(t, "C7", answered)
}
