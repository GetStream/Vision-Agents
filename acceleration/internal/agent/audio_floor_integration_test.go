//go:build integration

package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/thesyncim/gopus"
	"github.com/thesyncim/gopus/container/ogg"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// The chorus-asr acceptance replay: recorded calls played into an agent in real time, as a
// call's edge hands them over, with the chorus-asr model as both the transcriber
// (stream/chorus-asr) and, acting, the flow controller's Hearer, against the real server
// ROUTER_CHORUS_ASR_URL names. Only the conversation model and the voice are stand-ins:
// both answer at once, and the voice speaks a recorded reply long enough to talk over.
//
//	ROUTER_CHORUS_ASR_URL=http://127.0.0.1:8080 \
//	  go test -tags integration -count=1 -timeout 30m -run TestAudioFloorReplay -v ./internal/agent/
//
// CHORUS_ASR_REPLAY_LOGS names a directory for each call's debug log (default: a new one in
// the system's temporary directory), and CHORUS_ASR_END_OF_TURN the end_of_turn every call
// chooses, as an agent config writes it ({"preset":"fast"}, {"max_wait_ms":1400}; default:
// the server's). A failed check is what the call did, with its timeline in the test's output.
// Check 4, a slow count, fails under a point that cuts slow counts, as the server measured.
const (
	replayRate  = audiofloor.SampleRate
	replayFrame = replayRate / 50
	// replayFrameTime is how often the edge hands the agent a frame of the caller.
	replayFrameTime = 20 * time.Millisecond
	// replayPlayout is how much of the agent's speech the edge holds before PublishAudio
	// blocks, as the Stream edge's queue does (streamedge: playoutFrames).
	replayPlayout = 20 * replayFrame
	// audibleLevel is the level, as a fraction of full scale, from which a frame of the
	// agent's speech counts as heard: about -50 dBFS.
	audibleLevel = 0.003
	// voicedLevel is where a clip's speech starts and ends: about -40 dBFS.
	voicedLevel = 0.01

	// replyText is what the conversation model answers every turn with, and the voice
	// speaks testdata/audiofloor/reply.ogg for it: one sentence of about 9 s.
	replyText = "Tomorrow in Lisbon it will be sunny and warm, with a high of twenty four degrees, " +
		"a light breeze from the north in the afternoon, and a clear and mild evening by the river."
	// replayInstructions are the agent's, shared by every call but the one that registers
	// a new agent.
	replayInstructions = "You answer the phone for a weather service. Keep each answer short."
	// fallbackRuling is the flow controller's language model's answer, when the Hearer did
	// not rule: one that is wrong for every check here, so that a check passes only on the
	// Hearer's rulings.
	fallbackRuling = `{"disposition":"respond","floor":"stop"}`
)

var replayTools = harness.Tools{Tools: []harness.Tool{{
	Name:        "get_weather",
	Description: "Get the forecast for a city.",
	Parameters:  map[string]any{"type": "object", "required": []any{"city"}, "properties": map[string]any{"city": map[string]any{"type": "string"}}},
}}}

// TestAudioFloorReplay is the acceptance replay: each check a call of its own, in order.
func TestAudioFloorReplay(t *testing.T) {
	url := os.Getenv("ROUTER_CHORUS_ASR_URL")
	if url == "" {
		t.Skip("needs ROUTER_CHORUS_ASR_URL, a chorus-asr server")
	}
	h := newFloorHarness(t, url)
	h.warm(t)

	t.Run("7 a new agent registers within the deadline", h.newAgentRegisters)
	t.Run("1 a backchannel leaves the reply going", h.backchannel)
	t.Run("2a one final, then silence", func(t *testing.T) { h.oneFinal(t, audiofloor.Act, nil, 0) })
	t.Run("2b one final, then room tone", func(t *testing.T) { h.oneFinal(t, audiofloor.Act, roomTone(30*time.Second, -60), 0) })
	t.Run("2c one final, then line noise at -50 dBFS", func(t *testing.T) { h.oneFinal(t, audiofloor.Act, lineNoise(30*time.Second, -50), 0) })
	t.Run("2d one final, with the reply echoed at -20 dB", func(t *testing.T) { h.oneFinal(t, audiofloor.Act, nil, echoAt(-20)) })
	t.Run("2e one final, with the reply echoed at -30 dB", func(t *testing.T) { h.oneFinal(t, audiofloor.Act, nil, echoAt(-30)) })
	t.Run("2f one final, echoed at -20 dB, transcription alone", func(t *testing.T) { h.oneFinal(t, "", nil, echoAt(-20)) })
	t.Run("2g one final, echoed at -20 dB, in shadow", func(t *testing.T) { h.oneFinal(t, audiofloor.Shadow, nil, echoAt(-20)) })
	t.Run("3a trailing breath and noise, callers speak en", func(t *testing.T) { h.trailingNoise(t, []string{"en"}) })
	t.Run("3b trailing breath and noise, no languages", func(t *testing.T) { h.trailingNoise(t, nil) })
	t.Run("4 a slow count", h.slowCount)
	t.Run("5 a long turn", h.longTurn)
	t.Run("6 a cough alone", h.coughAlone)
	t.Run("9 transcription alone", h.transcriptionAlone)
	t.Run("10a live words over another language in the room, no languages", func(t *testing.T) { h.liveWordsOverAnotherLanguage(t, nil) })
	t.Run("10b live words over another language in the room, callers speak en", func(t *testing.T) { h.liveWordsOverAnotherLanguage(t, []string{"en"}) })
	t.Run("11 a caller talking over the agent without a pause", h.talkedOver)
	t.Run("8 latency", h.latency)
}

// floorHarness is what every call shares: the server, one client of it as a router has, and
// what each call measured.
type floorHarness struct {
	url      string
	client   *audiofloor.Client
	requests *exchanges
	policy   audiofloor.Policy
	logDir   string
	// endOfTurn is what every call chooses (CHORUS_ASR_END_OF_TURN), and ending what it is
	// under the server's policy.
	endOfTurn audiofloor.EndOfTurn
	ending    audiofloor.Ending

	weather, uhhuh, rain, pharmacy, long, stop, reply []int16
	count                                             [][]int16
	// portuguese is somebody else in the room, talking in another language.
	portuguese []int16

	mu        sync.Mutex
	latencies []turnLatency
}

func newFloorHarness(t *testing.T, url string) *floorHarness {
	requests := &exchanges{base: http.DefaultTransport}
	h := &floorHarness{
		url:      url,
		requests: requests,
		client:   audiofloor.New(url, audiofloor.Opus, &http.Client{Transport: requests}),
		logDir:   os.Getenv("CHORUS_ASR_REPLAY_LOGS"),
	}
	if h.logDir == "" {
		dir, err := os.MkdirTemp("", "audio-floor-replay-")
		require.NoError(t, err)
		h.logDir = dir
	}
	require.NoError(t, os.MkdirAll(h.logDir, 0o755))
	t.Logf("each call's debug log is in %s", h.logDir)
	h.endOfTurn = endOfTurnOf(t, os.Getenv("CHORUS_ASR_END_OF_TURN"))
	h.weather, h.uhhuh, h.rain = fixture(t, "weather"), fixture(t, "uhhuh"), fixture(t, "rain")
	h.pharmacy, h.long, h.reply = fixture(t, "pharmacy"), fixture(t, "long"), fixture(t, "reply")
	h.stop, h.portuguese = fixture(t, "stop"), fixture(t, "portuguese")
	for _, n := range []string{"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"} {
		h.count = append(h.count, trimmed(fixture(t, "count-"+n)))
	}
	return h
}

// endOfTurnOf is an end_of_turn as an agent config writes it, such as {"preset":"fast"} or
// {"max_wait_ms":1400}: empty is the server's default.
func endOfTurnOf(t *testing.T, raw string) audiofloor.EndOfTurn {
	if raw == "" {
		return audiofloor.EndOfTurn{}
	}
	var choice struct {
		MaxWaitMs int                    `json:"max_wait_ms"`
		Preset    string                 `json:"preset"`
		Bars      []audiofloor.PolicyBar `json:"bars"`
	}
	require.NoError(t, json.Unmarshal([]byte(raw), &choice), "CHORUS_ASR_END_OF_TURN")
	return audiofloor.EndOfTurn{MaxWaitMs: choice.MaxWaitMs, Preset: choice.Preset, Bars: choice.Bars}
}

// warm has the server load what it maps and evaluates lazily, so that no check pays for it:
// a transcript and a decision about another agent.
func (h *floorHarness) warm(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	policy, err := h.client.Policy(ctx)
	require.NoError(t, err, "GET /v1/info")
	h.policy = policy
	ending, err := policy.Resolve(h.endOfTurn)
	require.NoError(t, err, "the end_of_turn the calls choose, %+v", h.endOfTurn)
	h.ending = ending
	t.Logf("the calls end turns by %s: bars %+v, %d checks in a row, dictation guard %s; checked %d ms into a pause, every %d ms, cap %d ms",
		ending.Name, ending.Bars, ending.ChecksInARow, ending.Guard(), policy.FirstCheckMs, policy.IntervalMs, policy.CapMs)
	stream, err := h.client.NewStream(audiofloor.Window)
	require.NoError(t, err)
	require.NoError(t, stream.Write(append(slices.Clone(h.weather), make([]int16, replayRate/2)...)))
	said, err := h.client.Transcribe(ctx, stream.Last(audiofloor.Window), []string{"en"})
	require.NoError(t, err)
	_, err = audiofloor.NewHearer(h.client, audiofloor.EndOfTurn{}, slog.New(slog.DiscardHandler)).Check(ctx, harness.FlowTurn{
		Instructions: "A warm-up agent.", Audio: stream.Last(audiofloor.Window), Paused: 500 * time.Millisecond, FirstPause: true,
	})
	require.NoError(t, err)
	t.Logf("warm: %q in %s", said.Text, said.Language)
}

// 7. A new agent's first turn registers it (POST /v1/agents) inside the Hearer's deadline,
// and the call has no failure.
func (h *floorHarness) newAgentRegisters(t *testing.T) {
	instructions := fmt.Sprintf("%s This line is %d.", replayInstructions, time.Now().UnixNano())
	c := h.start(t, audiofloor.Act, instructions, []string{"en"})
	c.edge.pause(time.Second)
	said := c.edge.say("weather", h.weather)
	c.edge.pause(4 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	registrations := c.requestsTo("/v1/agents")
	if assert.Len(t, registrations, 1, "one registration of the new agent") {
		r := registrations[0]
		assert.Equal(t, http.StatusOK, r.status, "registered: %s", r.body)
		took := r.done.Sub(r.start)
		assert.Less(t, took, harness.FlowDeadline/3, "registered within the Hearer's deadline")
		t.Logf("POST /v1/agents: %d in %s: %s", r.status, took.Round(time.Millisecond), r.body)
	}
	assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer")
	assert.Empty(t, all[Error](c), "no failure")
	answered := c.answered(said)
	assert.True(t, answered.ok, "the question was answered")
	h.measured(t, "7", answered)
}

// 1. "Uh-huh" over the agent's reply leaves it playing to the end, and earns no new answer.
func (h *floorHarness) backchannel(t *testing.T) {
	c := h.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	said := c.edge.say("weather", h.weather)
	c.until(func() bool { return len(c.edge.replyStarts()) > 0 }, 20*time.Second, "the reply began")
	time.Sleep(2 * time.Second)
	murmur := c.edge.say("uh-huh", h.uhhuh)
	c.settle(4*time.Second, time.Minute)
	defer c.report()

	answered := c.answered(said)
	h.measured(t, "1", answered)
	assert.True(t, murmur.ended().Before(answered.replyEnd()), "the murmur came while the reply played")
	assert.Len(t, all[Responding](c), 1, "one answer, none to the murmur")
	assert.Empty(t, all[Interrupted](c), "the reply was not interrupted")
	assert.Zero(t, c.edge.dropped(), "none of the reply was thrown away")
	assert.Zero(t, c.voice.interrupted(), "the voice was not stopped")
	assert.InDelta(t, float64(c.voice.replyLength()), float64(c.edge.heard()), float64(300*time.Millisecond),
		"the whole reply was heard: %s of %s", c.edge.heard(), c.voice.replyLength())
	assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer")
}

// 2. A finished question then 10 s of nothing: one final transcript, nothing written of the
// silence, and the reply played whole. bed is what the line carries under it all, and echo
// the agent's own voice heard back on the caller's line, as a phone with no echo
// cancellation sends it. What the transcriber wrote and what the conversation heard of it are
// checked apart: [transcriber] is every transcript, [conversation] the turns the agent took.
//
// With the reply echoed, the transcriber hears the agent's voice on the caller's line and
// writes it, as any transcriber would. In an agent's call the agent drops it (withoutEcho)
// before the conversation, the model or anyone watching sees it, so what the transcriber wrote
// of it is logged as an observation, and the contract is what they see: one turn of the
// caller's, none of the agent's voice. Transcribing alone (mode off), the router does not know
// the app's voice, and cancelling the echo is the edge's to do; the contract is the
// transcriber's: the model ends the caller's turn under the transcriber's own agent, so their
// words are one final of their own, written within a check or two of their voice rather than
// at the silence cap, which the agent's reply came back inside of, and the agent's voice is a
// final of its own rather than part of theirs.
func (h *floorHarness) oneFinal(t *testing.T, mode audiofloor.Mode, bed []float64, echo float64) {
	c := h.start(t, mode, replayInstructions, []string{"en"})
	c.edge.background(bed, echo)
	c.edge.pause(time.Second)
	said := c.edge.say("rain", h.rain)
	c.edge.pause(10 * time.Second)
	c.settle(3*time.Second, time.Minute)
	defer c.report()

	finals := c.finals()
	utterances := map[int64]bool{}
	for _, w := range c.words.all() {
		utterances[w.T.Utterance] = true
	}
	if echo == 0 {
		assert.Len(t, finals, 1, "[transcriber] one final transcript")
		if len(finals) > 0 {
			after := c.wordsBetween(finals[0].At.Add(time.Nanosecond), time.Now())
			assert.Empty(t, after, "[transcriber] nothing written after the final")
		}
		for _, f := range finals {
			_, echo := echoed(f.T.Text, replyText)
			assert.Zero(t, echo, "[transcriber] the agent's voice written as the caller's: %q", f.T.Text)
		}
		assert.Len(t, utterances, 1, "[transcriber] one utterance")
		// Interims asked while the caller talked are before this; the turn's words are the
		// pause's one request.
		assert.Len(t, c.wordRequestsSince(said.ended().Add(h.firstCheck()/2)), 1, "one request for the words of the turn once the caller stopped")
	} else {
		callers, voice, both := byEcho(finals)
		t.Logf("[transcriber] with the agent's voice echoed, %d finals in %d utterances: the caller's %q, the agent's voice alone %q, the caller's with the agent's voice on the end %q",
			len(finals), len(utterances), finalTexts(callers), finalTexts(voice), finalTexts(both))
		if mode == "" {
			if assert.Len(t, callers, 1, "[transcriber] one final of the caller's words") {
				wait := callers[0].At.Sub(said.ended())
				assert.Less(t, wait, h.soon(), "[transcriber] the caller's final came %s after their voice: a turn the model ends is written within a check or two of it", wait.Round(time.Millisecond))
			}
			assert.NotEmpty(t, voice, "[transcriber] the agent's voice, heard back on the line, written as a final of its own")
			assert.Empty(t, both, "[transcriber] the agent's voice written into the caller's final rather than as a final of its own: %q", finalTexts(both))
		}
	}

	// After the agent has dropped its own voice: what arrives (Hearing), what was heard
	// (Heard) and what the model is told (the history) hold one turn of the caller's and none
	// of the agent's voice.
	for _, hearing := range all[Hearing](c) {
		_, echo := echoed(hearing.E.Text, replyText)
		assert.Zero(t, echo, "[conversation] the agent's voice arriving as the caller's: %q", hearing.E.Text)
	}
	for _, heard := range all[Heard](c) {
		_, echo := echoed(heard.E.Text, replyText)
		assert.Zero(t, echo, "[conversation] the agent's voice heard as the caller's: %q", heard.E.Text)
	}
	if lines := c.callerLines(); assert.Len(t, lines, 1, "[conversation] one line of the caller's in the history: %q", lines) {
		_, echo := echoed(lines[0], replyText)
		assert.Zero(t, echo, "[conversation] the agent's voice in the caller's line of the history: %q", lines[0])
	}
	answered := c.answered(said)
	h.measured(t, "2", answered)
	assert.True(t, answered.ok, "[conversation] the question was answered")
	assert.Len(t, all[Heard](c), 1, "[conversation] one turn heard")
	assert.Len(t, all[Responding](c), 1, "[conversation] one answer")
	assert.Empty(t, all[Interrupted](c), "[conversation] the reply was not interrupted")
	assert.Empty(t, c.decisionsAfter(answered.decided, "supersede", "interrupt", "answer"), "[conversation] the answer was not superseded")
	assert.Zero(t, c.edge.dropped(), "[conversation] none of the reply was thrown away")
	assert.InDelta(t, float64(c.voice.replyLength()), float64(c.edge.heard()), float64(300*time.Millisecond),
		"[conversation] the whole reply was heard: %s of %s", c.edge.heard(), c.voice.replyLength())
	if mode == audiofloor.Act {
		assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer")
	}
}

func echoAt(db float64) float64 { return math.Pow(10, db/20) }

// byEcho sorts finals by what they are of: the caller's words, the agent's voice heard back
// alone, or the caller's words with the agent's voice on the end of them, as the agent tells
// them apart (echoed).
func byEcho(finals []heardWords) (callers, voice, both []heardWords) {
	for _, f := range finals {
		left, echo := echoed(f.T.Text, replyText)
		switch {
		case echo == 0:
			callers = append(callers, f)
		case strings.TrimSpace(left) == "":
			voice = append(voice, f)
		default:
			both = append(both, f)
		}
	}
	return callers, voice, both
}

func finalTexts(finals []heardWords) []string {
	out := []string{}
	for _, f := range finals {
		out = append(out, f.T.Text)
	}
	return out
}

// callerLines are the caller's lines of the agent's history, as the model is told them.
func (c *call) callerLines() []string {
	c.agent.mu.Lock()
	history := harness.WithoutNotes(slices.Clone(c.agent.history))
	c.agent.mu.Unlock()
	var out []string
	for _, message := range history {
		if message.Role == llm.User {
			out = append(out, message.Content)
		}
	}
	return out
}

// 3. A question followed by a breath, and another by a burst of noise: nothing written in a
// script the caller doesn't speak.
func (h *floorHarness) trailingNoise(t *testing.T, languages []string) {
	c := h.start(t, audiofloor.Act, replayInstructions, languages)
	c.edge.pause(time.Second)
	first := c.edge.say("pharmacy", h.pharmacy)
	c.edge.pause(300 * time.Millisecond)
	c.edge.say("breath", breath())
	c.edge.pause(5 * time.Second)
	c.settle(2*time.Second, time.Minute)
	second := c.edge.say("pharmacy", h.pharmacy)
	c.edge.pause(300 * time.Millisecond)
	c.edge.say("noise", rustle())
	c.edge.pause(5 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	for _, w := range c.words.all() {
		assert.False(t, foreign(w.T.Text), "a transcript in another script: %q", w.T.Text)
	}
	for _, heard := range all[Heard](c) {
		assert.False(t, foreign(heard.E.Text), "a turn in another script: %q", heard.E.Text)
	}
	h.measured(t, "3", c.answered(first))
	h.measured(t, "3", c.answered(second))
}

// 4. One to ten with a second between each, under the server's default preset: no turn ends
// before "ten".
func (h *floorHarness) slowCount(t *testing.T) {
	c := h.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	var first, last *cue
	for i, n := range h.count {
		if i > 0 {
			c.edge.pause(time.Second)
		}
		said := c.edge.say(fmt.Sprintf("count %d", i+1), n)
		if i == 0 {
			first = said
		}
		last = said
	}
	c.edge.pause(5 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	t.Logf("end_of_turn %s, %d checks in a row, slow counts cut %s", h.ending.Name, h.ending.ChecksInARow, slowCounts(h.ending))
	early := c.decisionsBetween(first.began(), last.ended(), "answer")
	for _, d := range early {
		t.Logf("answered at %s, before ten ended: %q (%s)", c.since(d.At), d.E.Text, d.E.Reason)
	}
	assert.Empty(t, early, "no turn ended before ten")
	answered := c.answered(last)
	assert.True(t, answered.ok, "the count was answered once it was done")
	h.measured(t, "4", answered)
}

func slowCounts(ending audiofloor.Ending) string {
	if ending.Point == nil {
		return "unmeasured"
	}
	return fmt.Sprintf("%d of %d", ending.Point.Measured.SlowCountsCut, ending.Point.Measured.SlowCounts)
}

// 5. A turn of more than 16 s: one transcript, and it has the turn's beginning.
func (h *floorHarness) longTurn(t *testing.T) {
	c := h.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	said := c.edge.say("long", h.long)
	c.edge.pause(6 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	t.Logf("spoke for %s", said.ended().Sub(said.began()).Round(time.Millisecond))
	finals := c.finals()
	assert.Len(t, finals, 1, "one final transcript")
	for _, f := range finals {
		assert.Contains(t, plainWords(f.T.Text), "here is my situation", "the final has the turn's beginning: %q", f.T.Text)
	}
	replies := all[Responding](c)
	assert.Len(t, replies, 1, "one answer")
	for _, r := range replies {
		assert.Contains(t, plainWords(r.E.Prompt), "here is my situation", "the answer was to the whole turn: %q", r.E.Prompt)
	}
	h.measured(t, "5", c.answered(said))
}

// 6. A cough alone: nothing written, and no answer.
func (h *floorHarness) coughAlone(t *testing.T) {
	c := h.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	c.edge.say("cough", cough())
	c.edge.pause(6 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	assert.Empty(t, c.words.all(), "nothing written")
	assert.Empty(t, all[Responding](c), "no answer")
}

// 9. With the mode off, the chorus-asr transcriber writes a turn by itself: the model ends
// it under the transcriber's own agent, the decision and the words in one request, within a
// check or two of the voice, and no words are asked for alone once the caller has stopped.
func (h *floorHarness) transcriptionAlone(t *testing.T) {
	c := h.start(t, "", replayInstructions, []string{"en"})
	c.edge.pause(time.Second)
	said := c.edge.say("weather", h.weather)
	c.edge.pause(4 * time.Second)
	c.settle(2*time.Second, time.Minute)
	defer c.report()

	finals := c.finals()
	if assert.Len(t, finals, 1, "one final transcript") {
		assert.Contains(t, plainWords(finals[0].T.Text), "lisbon", "the turn written: %q", finals[0].T.Text)
		wait := finals[0].At.Sub(said.ended())
		assert.Less(t, wait, h.soon(), "the final came %s after the voice: the model ended the turn", wait.Round(time.Millisecond))
	}
	heard := all[Heard](c)
	if assert.NotEmpty(t, heard, "the turn reached the conversation") {
		assert.Contains(t, plainWords(heard[0].E.Text), "lisbon")
	}
	registered := c.requestsTo("/v1/agents")
	for _, r := range registered {
		assert.NotContains(t, r.body, "get_weather", "the transcriber's own agent has no tools: %s", r.body)
	}
	stopped := c.wordRequestsSince(said.ended().Add(h.firstCheck() / 2))
	if assert.Len(t, stopped, 1, "one request for the words once the caller stopped") {
		assert.Equal(t, "/v1/decide", stopped[0].path, "the decision and the words in one pass: %s %s?%s", stopped[0].method, stopped[0].path, stopped[0].query)
	}
	for _, r := range c.requestsTo("/v1/transcribe") {
		// The words alone are asked for only while the caller talks, as interims.
		assert.True(t, r.start.Before(stopped[0].start), "words asked for alone once the caller stopped: %s %s at %s", r.method, r.path, c.since(r.start))
	}
}

// roomLevel is how far below the caller somebody else talking in the room is heard, in dB:
// a voice across the room, which the model writes when it is told no language.
const roomLevel = 18

// roomWords are the words of the room's Portuguese that English speech is never written
// as, lowercase and without accents as plainWords strips them, to tell the room's words
// from the caller's in what the agent heard.
var roomWords = map[string]bool{"nao": true, "não": true, "desculpa": true, "buscar": true, "jogo": true, "tenho": true, "aquilo": true,
	"agora": true, "passava": true, "acho": true, "olha": true, "deixa": true, "depois": true, "trato": true, "disso": true, "onde": true,
	"tirar": true, "tarde": true, "disse": true, "vem": true, "está": true, "esta": true, "deixar": true, "estar": true}

// fromTheRoom are text's words that are the room's.
func fromTheRoom(text string) []string {
	var out []string
	for _, w := range strings.Fields(plainWords(text)) {
		if roomWords[w] {
			out = append(out, w)
		}
	}
	return out
}

// 10. The caller's long turn with somebody else talking in the room, in Portuguese, 18 dB
// below them: with no languages declared (10a), and with en (10b). The words so far are
// shown while the caller talks, and none of them, nor any final, is in another script or
// carries the room's words: with no language declared, an interim is asked for in the
// language of the turn's first words at a pause rather than the server's guess at the audio
// so far, which over the room's voice flipped language and script from one to the next.
func (h *floorHarness) liveWordsOverAnotherLanguage(t *testing.T, languages []string) {
	c := h.start(t, audiofloor.Act, replayInstructions, languages)
	c.edge.background(bedBelow(h.portuguese, h.long, roomLevel), 0)
	c.edge.pause(time.Second)
	said := c.edge.say("long", h.long)
	c.edge.pause(5 * time.Second)
	c.settle(3*time.Second, 2*time.Minute)
	defer c.report()

	var interims, finals int
	for _, w := range c.words.all() {
		switch {
		case w.T.Interim:
			interims++
		case w.T.Final():
			finals++
			assert.False(t, foreign(w.T.Text), "[transcriber] a final in another script: %q", w.T.Text)
			assert.Empty(t, fromTheRoom(w.T.Text), "[transcriber] the room's words in a final: %q", w.T.Text)
		}
	}
	t.Logf("[transcriber] %d interims shown and %d finals written of a %s turn", interims, finals, said.ended().Sub(said.began()).Round(time.Second))
	assert.NotZero(t, interims, "[transcriber] the words so far were shown while the caller talked")
	assert.NotZero(t, finals, "[transcriber] the turn was written")
	for _, hearing := range all[Hearing](c) {
		assert.False(t, foreign(hearing.E.Text), "[conversation] words in another script shown: %q", hearing.E.Text)
		assert.Empty(t, fromTheRoom(hearing.E.Text), "[conversation] the room's words shown as the caller's: %q", hearing.E.Text)
	}
	var lines []string
	for _, heard := range all[Heard](c) {
		lines = append(lines, heard.E.Text)
		assert.False(t, foreign(heard.E.Text), "[conversation] a turn in another script: %q", heard.E.Text)
		assert.Empty(t, fromTheRoom(heard.E.Text), "[conversation] the room's words heard as the caller's: %q", heard.E.Text)
	}
	assert.Contains(t, plainWords(strings.Join(lines, " ")), "lisbon", "[conversation] the caller's English heard: %q", lines)
}

// stopWithin is how soon after the caller's first word over the agent the agent is to have
// stopped: the time they are voiced before they are asked about, the decision, the words
// and the interruption.
const stopWithin = 700 * time.Millisecond

// 11. A caller who talks over the agent without pausing: "No, no, stop, that's not what I
// asked, I wanted Lisbon", its gaps cut to under 200 ms so that no pause reaches the
// policy's first check, 1.5 s into a 9 s reply echoed at -30 dB with somebody talking in
// the room at -40 dBFS. The agent stops within stopWithin of the caller's first word, on
// the model's ruling, and answers what they said. A murmur over the agent, check 1, still
// leaves the reply going. The room falls quiet once the caller has been answered: talking
// on, it keeps the speech gate open and the call never settles (Limits).
func (h *floorHarness) talkedOver(t *testing.T) {
	c := h.start(t, audiofloor.Act, replayInstructions, []string{"en"})
	c.edge.background(bedAt(h.portuguese, -40), echoAt(-30))
	c.edge.pause(time.Second)
	c.edge.say("weather", h.weather)
	c.until(func() bool { return len(c.edge.replyStarts()) > 0 }, 20*time.Second, "the reply began")
	time.Sleep(1500 * time.Millisecond)
	over := c.edge.say("stop", withGapsUnder(h.stop, 150*time.Millisecond))
	c.until(func() bool { return c.answered(over).ok }, 30*time.Second, "what the caller said over the agent was answered")
	c.edge.background(nil, echoAt(-30))
	c.settle(4*time.Second, time.Minute)
	defer c.report()

	interrupted := all[Interrupted](c)
	if assert.NotEmpty(t, interrupted, "the agent was stopped") {
		took := interrupted[0].At.Sub(over.began())
		t.Logf("the agent stopped %s after the caller's first word", took.Round(time.Millisecond))
		assert.False(t, interrupted[0].At.Before(over.began()), "the agent stopped before the caller spoke")
		assert.LessOrEqual(t, took, stopWithin, "the agent stopped %s after the caller's first word", took.Round(time.Millisecond))
	}
	for _, at := range c.edge.drops() {
		assert.False(t, at.Before(over.began()), "the reply was thrown away %s before the caller spoke: the room or the echo stopped it", over.began().Sub(at).Round(time.Millisecond))
	}
	answered := c.answered(over)
	h.measured(t, "11", answered)
	assert.True(t, answered.ok, "what the caller said over the agent was answered")
	assert.Empty(t, c.hearerMissed(), "every turn ruled by the Hearer")
}

// firstCheck is how far into a pause the server's policy checks it first.
func (h *floorHarness) firstCheck() time.Duration {
	return time.Duration(h.policy.FirstCheckMs) * time.Millisecond
}

// soon is how long after the voice a turn the model ends is written, with room for a second
// check and the words: the policy's first check and its interval, and a second. The silence
// cap the transcriber ended utterances at before, 2.5 s and the words, is well past it.
func (h *floorHarness) soon() time.Duration {
	return h.firstCheck() + time.Duration(h.policy.IntervalMs)*time.Millisecond + time.Second
}

// 8. From the end of the caller's speech to the turn's ruling, and to the reply being heard,
// for every answered turn above.
func (h *floorHarness) latency(t *testing.T) {
	h.mu.Lock()
	defer h.mu.Unlock()
	var ruled, decided, replied []time.Duration
	t.Logf("%-6s %14s %14s %12s", "check", "server ruled", "turn answered", "reply heard")
	for _, l := range h.latencies {
		d, r := l.decided.Sub(l.speechEnd), l.replied.Sub(l.speechEnd)
		decided, replied = append(decided, d), append(replied, r)
		if !l.ruled.IsZero() {
			ruled = append(ruled, l.ruled.Sub(l.speechEnd))
		}
		t.Logf("%-6s %14s %14s %12s", l.check, since(l.speechEnd, l.ruled), d.Round(time.Millisecond), r.Round(time.Millisecond))
	}
	if !assert.NotEmpty(t, decided, "turns measured") {
		return
	}
	if len(ruled) > 0 {
		t.Logf("end of speech to the server's ruling: p50 %s, max %s, over %d turns", median(ruled), slices.Max(ruled).Round(time.Millisecond), len(ruled))
	}
	t.Logf("end of speech to the turn answered:   p50 %s, max %s, over %d turns", median(decided), slices.Max(decided).Round(time.Millisecond), len(decided))
	t.Logf("end of speech to the reply heard:     p50 %s, max %s", median(replied), slices.Max(replied).Round(time.Millisecond))
}

type turnLatency struct {
	check                              string
	speechEnd, ruled, decided, replied time.Time
}

func (h *floorHarness) measured(t *testing.T, check string, answered answer) {
	// A turn answered before the caller had finished is a cut-in, which its check reports.
	if !answered.ok || answered.replied.IsZero() || answered.decided.Before(answered.speechEnd) {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.latencies = append(h.latencies, turnLatency{check: check, speechEnd: answered.speechEnd, ruled: answered.ruled, decided: answered.decided, replied: answered.replied})
	t.Logf("latency: end of speech to the server's ruling %s, to the turn answered %s, to the reply %s",
		since(answered.speechEnd, answered.ruled), answered.decided.Sub(answered.speechEnd).Round(time.Millisecond),
		answered.replied.Sub(answered.speechEnd).Round(time.Millisecond))
}

func since(from, to time.Time) string {
	if to.IsZero() {
		return "-"
	}
	return to.Sub(from).Round(time.Millisecond).String()
}

func median(values []time.Duration) time.Duration {
	sorted := slices.Sorted(slices.Values(values))
	return sorted[len(sorted)/2].Round(time.Millisecond)
}

// call is one replayed call: an agent on an edge playing the caller in real time.
type call struct {
	t      *testing.T
	h      *floorHarness
	began  time.Time
	agent  *Agent
	edge   *replayEdge
	voice  *replayVoice
	flow   *stubLLM
	events *timeline
	words  *wordLog
	logs   *logBook
}

// start joins an agent to a new call: the chorus-asr model transcribing, and asked as mode
// says, with the conversation model and the voice stand-ins.
func (h *floorHarness) start(t *testing.T, mode audiofloor.Mode, instructions string, languages []string) *call {
	logs := newLogBook(t, filepath.Join(h.logDir, strings.NewReplacer(" ", "-", "/", "-", ",", "").Replace(t.Name())+".jsonl"))
	logger := slog.New(&logHandler{book: logs, json: slog.NewJSONHandler(logs.file, &slog.HandlerOptions{Level: slog.LevelDebug})})
	c := &call{t: t, h: h, began: time.Now(), words: &wordLog{}, logs: logs}

	// The router's own factory for the transcriber, with what it writes kept as it goes.
	real := sttrouter.NewRegistry()
	sttrouter.RegisterAudioFloor(real, h.client)
	transcribers := sttrouter.NewRegistry()
	transcribers.Register(audiofloor.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		inner, err := real.Build(audiofloor.ProviderName, spec)
		if err != nil {
			return nil, err
		}
		return teeWords(inner, c.words), nil
	})
	config, err := routing.DefaultConfig()
	require.NoError(t, err)
	transcriber, err := sttrouter.New(sttrouter.Options{Config: config[routing.STT], Registry: transcribers, Logger: logger})
	require.NoError(t, err)
	t.Cleanup(transcriber.Close)

	model := newStubLLM()
	model.reply = []string{replyText}
	c.flow = newStubLLM()
	c.flow.reply = []string{fallbackRuling}
	models := []*stubLLM{model, c.flow}
	var opened int
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		provider := models[min(opened, len(models)-1)]
		opened++
		return provider, nil
	})
	reasoner, err := llmrouter.New(llmrouter.Options{Config: stubConfig(), Registry: reasoning, Logger: logger})
	require.NoError(t, err)
	t.Cleanup(reasoner.Close)

	c.voice = newReplayVoice(h.reply)
	voices := ttsrouter.NewRegistry()
	voices.Register("stub", func(routing.Spec) (tts.TTS, error) { return c.voice, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{Config: stubConfig(), Registry: voices, Logger: logger})
	require.NoError(t, err)
	t.Cleanup(speaker.Close)

	c.edge = newReplayEdge(stt.Participant{ID: "caller", UserID: "caller"})
	agent, err := New(Options{
		Edge:           c.edge,
		Instructions:   instructions,
		Tools:          replayTools,
		CustomerID:     "acme",
		AgentID:        "audio-floor-replay",
		SessionID:      "replay",
		LLM:            reasoner,
		LLMTarget:      "en-low-latency",
		STT:            transcriber,
		STTTarget:      audiofloor.ProviderName + "/" + audiofloor.DefaultModel,
		TTS:            speaker,
		TTSTarget:      "en-low-latency",
		AudioFloor:     h.client,
		AudioFloorMode: mode,
		EndOfTurn:      h.endOfTurn,
		LanguageHints:  languages,
		Logger:         logger,
	})
	require.NoError(t, err)
	c.agent = agent
	require.NoError(t, agent.Join(context.Background()))
	c.events = record(agent)
	t.Cleanup(func() {
		closed := make(chan struct{})
		go func() {
			_ = agent.Close()
			<-c.events.done
			close(closed)
		}()
		select {
		case <-closed:
		case <-time.After(30 * time.Second):
			t.Errorf("the agent did not close")
		}
		logs.close()
	})
	return c
}

// settle waits for the caller's lines to be played and the agent to have been quiet for
// quiet, the edge going on with the line's background all the while.
func (c *call) settle(quiet, limit time.Duration) {
	deadline := time.Now().Add(limit)
	var since time.Time
	for time.Now().Before(deadline) {
		if c.edge.scripted() || c.edge.SpeechPending() || c.agent.Busy() || c.asking() {
			since = time.Time{}
		} else if since.IsZero() {
			since = time.Now()
		} else if time.Since(since) >= quiet {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	c.t.Errorf("the call did not settle within %s", limit)
}

func (c *call) until(condition func() bool, limit time.Duration, what string) {
	deadline := time.Now().Add(limit)
	for !condition() {
		if time.Now().After(deadline) {
			c.t.Errorf("not within %s: %s", limit, what)
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func (c *call) since(at time.Time) string {
	return fmt.Sprintf("%7.3fs", at.Sub(c.began).Seconds())
}

// answer is a turn's answer, timed from the end of what the caller said.
type answer struct {
	ok                 bool
	speechEnd, decided time.Time
	// ruled is when the server's decision that ended the turn came back, its first line.
	ruled        time.Time
	replied      time.Time
	replyLength  time.Duration
	text, reason string
	turnID       string
}

func (a answer) replyEnd() time.Time { return a.replied.Add(a.replyLength) }

// answered is the first answer ruled once the line said began, and when its reply was first
// heard.
func (c *call) answered(said *cue) answer {
	out := answer{speechEnd: said.ended(), replyLength: c.voice.replyLength()}
	for _, d := range all[Decided](c) {
		if d.E.Kind == string(ActAnswer) && !d.At.Before(said.began()) {
			out.ok, out.decided, out.text, out.reason, out.turnID = true, d.At, d.E.Text, d.E.Reason, d.E.TurnID
			break
		}
	}
	if !out.ok {
		return out
	}
	for _, r := range c.requests() {
		if r.path == "/v1/decide" && !r.firstLine.IsZero() && !r.firstLine.After(out.decided) && r.firstLine.After(out.speechEnd) {
			out.ruled = r.firstLine
		}
	}
	for _, at := range c.edge.replyStarts() {
		if at.After(out.decided) {
			out.replied = at
			break
		}
	}
	return out
}

func (c *call) finals() []heardWords {
	var out []heardWords
	for _, w := range c.words.all() {
		if w.T.Final() {
			out = append(out, w)
		}
	}
	return out
}

func (c *call) wordsBetween(from, to time.Time) []heardWords {
	var out []heardWords
	for _, w := range c.words.all() {
		if !w.At.Before(from) && w.At.Before(to) {
			out = append(out, w)
		}
	}
	return out
}

func (c *call) decisionsAfter(at time.Time, kinds ...string) []stampedEvent[Decided] {
	return c.decisionsBetween(at.Add(time.Nanosecond), time.Now().Add(time.Hour), kinds...)
}

func (c *call) decisionsBetween(from, to time.Time, kinds ...string) []stampedEvent[Decided] {
	var out []stampedEvent[Decided]
	for _, d := range all[Decided](c) {
		if !d.At.Before(from) && d.At.Before(to) && slices.Contains(kinds, d.E.Kind) {
			out = append(out, d)
		}
	}
	return out
}

// hearerMissed are the turns the Hearer did not rule on, which the language model did.
func (c *call) hearerMissed() []logLine {
	return c.logs.find("the hearer did not rule, the flow controller will")
}

// requests are the server's, since the call began.
func (c *call) requests() []exchange {
	var out []exchange
	for _, r := range c.h.requests.all() {
		if !r.start.Before(c.began) {
			out = append(out, r)
		}
	}
	return out
}

// asking says a request to the server is still out.
func (c *call) asking() bool {
	for _, r := range c.requests() {
		if r.done.IsZero() {
			return true
		}
	}
	return false
}

// wordRequestsSince are the requests for words started at or after at: transcripts, and
// decisions that wrote words too.
func (c *call) wordRequestsSince(at time.Time) []exchange {
	var out []exchange
	for _, r := range c.requests() {
		if r.start.Before(at) {
			continue
		}
		if r.path == "/v1/transcribe" || (r.path == "/v1/decide" && strings.Contains(r.query, "transcript=true")) {
			out = append(out, r)
		}
	}
	return out
}

func (c *call) requestsTo(path string) []exchange {
	var out []exchange
	for _, r := range c.requests() {
		if r.path == path {
			out = append(out, r)
		}
	}
	return out
}

// report logs the call's timeline: the caller's lines, what was written and decided, the
// reply, and each request to the server.
func (c *call) report() {
	type line struct {
		at   time.Time
		text string
	}
	var lines []line
	add := func(at time.Time, format string, args ...any) {
		lines = append(lines, line{at, fmt.Sprintf(format, args...)})
	}
	for _, cue := range c.edge.cues() {
		if !cue.start.IsZero() {
			add(cue.start, "caller  %s begins", cue.name)
		}
		if !cue.end.IsZero() {
			add(cue.end, "caller  %s ends", cue.name)
		}
	}
	for _, w := range c.words.all() {
		mode := string(w.T.Mode)
		if w.T.Interim {
			mode = "interim"
		}
		add(w.At, "words   %s #%d %q (%.0f ms of audio in %.0f ms)", mode, w.T.Utterance, w.T.Text, w.T.AudioDurationMs, w.T.ProcessingTimeMs)
	}
	for _, e := range c.events.all() {
		switch typed := e.E.(type) {
		case Decided:
			add(e.At, "decided %s: %s %q took %.0f ms", typed.Kind, typed.Reason, typed.Text, typed.LatencyMs)
		case Responding:
			add(e.At, "reply   %s to %q", typed.TurnID, typed.Prompt)
		case Interrupted:
			add(e.At, "INTERRUPTED %s after %.0f ms heard", typed.TurnID, typed.HeardMs)
		case OverlapDecided:
			add(e.At, "overlap %s %s", typed.Action, typed.TurnID)
		case Error:
			add(e.At, "ERROR   %s: %v", typed.Context, typed.Err)
		case Backchannel:
			add(e.At, "murmur  %q", typed.Text)
		}
	}
	for _, at := range c.edge.replyStarts() {
		add(at, "heard   the agent's voice begins")
	}
	for _, at := range c.edge.drops() {
		add(at, "DROPPED the agent's speech")
	}
	for _, r := range c.requests() {
		path := r.path
		if strings.Contains(r.query, "transcript=true") {
			path += "+words"
		}
		first := ""
		if !r.firstLine.IsZero() && strings.Contains(r.query, "transcript=true") {
			first = fmt.Sprintf(", decision in %.0f ms", float64(r.firstLine.Sub(r.start).Microseconds())/1000)
		}
		add(r.start, "server  %s %s: %d in %.0f ms%s %s", r.method, path, r.status, float64(r.done.Sub(r.start).Microseconds())/1000, first, brief(r.body))
	}
	for _, l := range c.logs.find("the transcriber heard the agent's own voice on the caller's line") {
		add(l.At, "ECHO    %v words of the agent's dropped from %q, left %q", l.Attrs["echo_words"], l.Attrs["text"], l.Attrs["left"])
	}
	for _, l := range c.logs.find("the hearer did not rule, the flow controller will") {
		add(l.At, "MISSED  the Hearer did not rule on %v: %v", l.Attrs["candidate"], l.Attrs["error"])
	}
	for _, asked := range c.flow.requests() {
		add(time.Time{}, "FALLBACK the language model ruled on %s", asked.TurnID)
	}
	slices.SortStableFunc(lines, func(a, b line) int { return a.at.Compare(b.at) })
	var out strings.Builder
	for _, l := range lines {
		at := "      -"
		if !l.at.IsZero() {
			at = c.since(l.at)
		}
		fmt.Fprintf(&out, "%s %s\n", at, l.text)
	}
	c.t.Logf("timeline:\n%s", out.String())
}

// brief is a server's answer without its options, which are long and say little here.
func brief(body string) string {
	var answer struct {
		Answer     string   `json:"answer"`
		P          float64  `json:"answer_probability"`
		PEOT       *float64 `json:"p_eot"`
		DecideMs   float64  `json:"decide_ms"`
		Ruling     any      `json:"ruling"`
		Transcript *struct {
			Text  string `json:"text"`
			Noise string `json:"noise"`
		} `json:"transcript"`
		Agent string `json:"agent"`
		Error string `json:"error"`
	}
	var parts []string
	for _, raw := range strings.Split(strings.TrimSpace(body), "\n") {
		if json.Unmarshal([]byte(raw), &answer) != nil {
			parts = append(parts, raw)
			continue
		}
		switch {
		case answer.Ruling != nil:
			peot := -1.0
			if answer.PEOT != nil {
				peot = *answer.PEOT
			}
			parts = append(parts, fmt.Sprintf("%s %.3f p_eot %.4f ruling %v (%.0f ms)", answer.Answer, answer.P, peot, answer.Ruling, answer.DecideMs))
		case answer.Transcript != nil:
			parts = append(parts, fmt.Sprintf("%q %s", answer.Transcript.Text, answer.Transcript.Noise))
		case answer.Error != "":
			parts = append(parts, "error: "+answer.Error)
		case answer.Agent != "":
			parts = append(parts, "agent "+answer.Agent)
		}
		answer.Ruling, answer.Transcript, answer.PEOT = nil, nil, nil
	}
	return strings.Join(parts, " | ")
}

// plainWords is text as words: lowercase, without punctuation.
func plainWords(text string) string {
	return strings.Join(strings.Fields(strings.Map(func(r rune) rune {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r), r == '\'':
			return unicode.ToLower(r)
		}
		return ' '
	}, text)), " ")
}

// foreign says text has a letter in a script other than Latin.
func foreign(text string) bool {
	for _, r := range text {
		if unicode.IsLetter(r) && !unicode.Is(unicode.Latin, r) {
			return true
		}
	}
	return false
}

// replayEdge is a call's edge in real time: every 20 ms it plays a frame of the agent's speech
// to the caller and hands the agent a frame of the caller's line, which is what the caller is
// saying, the line's background, and as much of the agent's own voice as the line echoes back.
type replayEdge struct {
	caller  stt.Participant
	inbound chan InboundAudio
	done    chan struct{}

	mu      sync.Mutex
	drained *sync.Cond
	joined  bool
	left    bool
	// playout is the agent's speech waiting to be heard.
	playout []int16
	// script is what the caller has still to say, its first sample the at'th of the line.
	script  []int16
	at, fed int64
	lines   []*cue
	bed     []float64
	bedAt   int
	echo    float64
	// What the caller heard of the agent: when its voice began, after 200 ms with none of
	// its speech waiting, how much of its speech was played, and when some was thrown away.
	idle   int
	starts []time.Time
	played int
	thrown []time.Time
}

// cue is a line the caller says: when its speech began and ended on the line.
type cue struct {
	name       string
	from, to   int64
	start, end time.Time
}

func newReplayEdge(caller stt.Participant) *replayEdge {
	e := &replayEdge{caller: caller, inbound: make(chan InboundAudio, 64), done: make(chan struct{}), idle: 10}
	e.drained = sync.NewCond(&e.mu)
	return e
}

func (e *replayEdge) Join(context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.joined {
		e.joined = true
		go e.run()
	}
	return nil
}

func (e *replayEdge) Audio() <-chan InboundAudio { return e.inbound }

// PublishAudio queues the agent's speech, and blocks while more than replayPlayout of it
// waits, so that the agent speaks at the pace it is heard.
func (e *replayEdge) PublishAudio(pcm audio.PcmData) error {
	if pcm.SampleRate != replayRate || pcm.Channels != 1 {
		return fmt.Errorf("replay: the agent spoke at %d Hz in %d channels", pcm.SampleRate, pcm.Channels)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.left {
		return ErrCallLeft
	}
	e.playout = append(e.playout, pcm.Samples...)
	for len(e.playout) > replayPlayout && !e.left {
		e.drained.Wait()
	}
	return nil
}

func (e *replayEdge) SpeechPending() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.playout) > 0
}

func (e *replayEdge) DropSpeech() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.playout) > 0 {
		e.thrown = append(e.thrown, time.Now())
	}
	e.playout = nil
	e.drained.Broadcast()
}

func (e *replayEdge) Leave() error {
	e.mu.Lock()
	e.left = true
	joined := e.joined
	e.drained.Broadcast()
	e.mu.Unlock()
	if joined {
		<-e.done
	} else {
		close(e.inbound)
	}
	return nil
}

// background has the line carry bed under everything, looped, and echo the agent's voice
// back at that gain.
func (e *replayEdge) background(bed []float64, echo float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.bed, e.bedAt, e.echo = bed, 0, echo
}

// say has the caller say samples once what they have still to say is said.
func (e *replayEdge) say(name string, samples []int16) *cue {
	e.mu.Lock()
	defer e.mu.Unlock()
	at := e.fed
	if len(e.script) > 0 {
		at = e.at + int64(len(e.script))
	} else {
		e.at = at
	}
	from, to := voiced(samples)
	line := &cue{name: name, from: at + int64(from), to: at + int64(to)}
	e.lines = append(e.lines, line)
	e.script = append(e.script, samples...)
	return line
}

// pause has the caller say nothing for d, once what they have still to say is said.
func (e *replayEdge) pause(d time.Duration) {
	e.say("", make([]int16, int(d.Seconds()*replayRate)))
}

// scripted says the caller has something still to say.
func (e *replayEdge) scripted() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.script) > 0
}

func (e *replayEdge) run() {
	defer close(e.done)
	defer close(e.inbound)
	next := time.Now()
	line := make([]float64, replayFrame)
	for {
		next = next.Add(replayFrameTime)
		time.Sleep(time.Until(next))
		e.mu.Lock()
		if e.left {
			e.mu.Unlock()
			return
		}
		clear(line)
		said := e.playout[:min(len(e.playout), replayFrame)]
		e.playout = e.playout[len(said):]
		var energy float64
		for i, s := range said {
			energy += float64(s) * float64(s)
			line[i] += e.echo * float64(s)
		}
		e.drained.Broadcast()
		now := time.Now()
		e.played += len(said)
		switch {
		case len(said) == 0:
			e.idle++
		case math.Sqrt(energy/replayFrame)/math.MaxInt16 >= audibleLevel:
			if e.idle >= 10 {
				e.starts = append(e.starts, now)
			}
			e.idle = 0
		}
		n := min(len(e.script), replayFrame)
		for i, s := range e.script[:n] {
			line[i] += float64(s)
		}
		e.script = e.script[n:]
		e.at += int64(n)
		for i := range line {
			if len(e.bed) > 0 {
				line[i] += e.bed[e.bedAt]
				e.bedAt = (e.bedAt + 1) % len(e.bed)
			}
		}
		e.fed += replayFrame
		for _, l := range e.lines {
			if l.start.IsZero() && l.from < e.fed && l.to > l.from {
				l.start = now
			}
			if l.end.IsZero() && l.to <= e.fed && l.to > l.from {
				l.end = now
			}
		}
		e.mu.Unlock()
		samples := make([]int16, replayFrame)
		for i, v := range line {
			samples[i] = int16(max(math.MinInt16, min(math.MaxInt16, math.Round(v))))
		}
		e.inbound <- InboundAudio{Participant: e.caller, Audio: audio.PcmData{Samples: samples, SampleRate: replayRate, Channels: 1}}
	}
}

func (e *replayEdge) cues() []cue {
	e.mu.Lock()
	defer e.mu.Unlock()
	var out []cue
	for _, l := range e.lines {
		if l.name != "" {
			out = append(out, *l)
		}
	}
	return out
}

func (e *replayEdge) replyStarts() []time.Time {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.starts)
}

func (e *replayEdge) drops() []time.Time {
	e.mu.Lock()
	defer e.mu.Unlock()
	return slices.Clone(e.thrown)
}

func (e *replayEdge) dropped() int { return len(e.drops()) }

// heard is how much of the agent's speech the caller heard, its pauses too.
func (e *replayEdge) heard() time.Duration {
	e.mu.Lock()
	defer e.mu.Unlock()
	return time.Duration(e.played) * time.Second / replayRate
}

func (c *cue) began() time.Time { return c.start }
func (c *cue) ended() time.Time { return c.end }

// replayVoice speaks the recorded reply for the agent: all of it at once, as the reply's
// length in characters is of the recording's.
type replayVoice struct {
	reply   []int16
	emitter *tts.Emitter
	ctx     context.Context
	stop    context.CancelFunc

	mu         sync.Mutex
	pending    map[string]string
	inFlight   map[string]context.CancelFunc
	interrupts int
	speaking   sync.WaitGroup
}

func newReplayVoice(reply []int16) *replayVoice {
	return &replayVoice{reply: trimmed(reply), emitter: tts.NewEmitter(64), pending: map[string]string{}, inFlight: map[string]context.CancelFunc{}}
}

func (v *replayVoice) Start(ctx context.Context) error {
	v.ctx, v.stop = context.WithCancel(context.WithoutCancel(ctx))
	v.emitter.Send(tts.Connected{Provider: "stub", Model: "replay", At: time.Now()})
	return nil
}

func (v *replayVoice) Synthesize(request tts.Request) error {
	if v.ctx == nil {
		return errors.New("replay: the voice is not open")
	}
	v.mu.Lock()
	said := v.pending[request.ID] + request.Text
	if !request.Final {
		v.pending[request.ID] = said
		v.mu.Unlock()
		return nil
	}
	delete(v.pending, request.ID)
	ctx, cancel := context.WithCancel(v.ctx)
	v.inFlight[request.ID] = cancel
	v.speaking.Add(1)
	v.mu.Unlock()
	go func() {
		defer v.speaking.Done()
		defer cancel()
		v.speak(ctx, request.ID, said)
	}()
	return nil
}

func (v *replayVoice) speak(ctx context.Context, id, said string) {
	started := time.Now()
	v.emitter.Send(tts.SynthesisStarted{SynthesisID: id, Provider: "stub", Model: "replay", At: started})
	length := len(v.reply) * utf8.RuneCountInString(said) / utf8.RuneCountInString(replyText)
	length -= length % replayFrame
	sent := 0
	for index := 0; sent < length && ctx.Err() == nil; index++ {
		piece := make([]int16, min(10*replayFrame, length-sent))
		for i := range piece {
			piece[i] = v.reply[(sent+i)%len(v.reply)]
		}
		v.emitter.Send(tts.AudioChunk{SynthesisID: id, Index: index, Audio: audio.PcmData{Samples: piece, SampleRate: replayRate, Channels: 1}})
		sent += len(piece)
	}
	v.mu.Lock()
	delete(v.inFlight, id)
	v.mu.Unlock()
	v.emitter.Send(tts.SynthesisComplete{
		SynthesisID: id, Provider: "stub", Model: "replay", Characters: int64(utf8.RuneCountInString(said)),
		AudioDurationMs: float64(sent) * 1000 / replayRate, SynthesisTimeMs: float64(time.Since(started).Microseconds()) / 1000,
		Interrupted: sent < length,
	})
}

func (v *replayVoice) Interrupt() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.interrupts++
	for _, cancel := range v.inFlight {
		cancel()
	}
	return nil
}

func (v *replayVoice) interrupted() int {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.interrupts
}

// replyLength is how long the whole reply takes to say.
func (v *replayVoice) replyLength() time.Duration {
	return time.Duration(len(v.reply)/replayFrame) * replayFrameTime
}

func (v *replayVoice) Events() <-chan tts.Event { return v.emitter.Events() }

func (v *replayVoice) Close() error {
	if v.stop != nil {
		v.stop()
	}
	v.speaking.Wait()
	v.emitter.Send(tts.Disconnected{Provider: "stub", Model: "replay", Clean: true, At: time.Now()})
	v.emitter.Close()
	return nil
}

func (v *replayVoice) Provider() string { return "stub" }
func (v *replayVoice) Model() string    { return "replay" }
func (v *replayVoice) Streaming() bool  { return true }
func (v *replayVoice) Prompt() string   { return "" }
func (v *replayVoice) Performs() bool   { return false }

// wordLog is what the transcriber wrote, as it wrote it.
type wordLog struct {
	mu    sync.Mutex
	heard []heardWords
}

type heardWords struct {
	At time.Time
	T  stt.Transcript
}

func (w *wordLog) add(transcript stt.Transcript) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.heard = append(w.heard, heardWords{At: time.Now(), T: transcript})
}

func (w *wordLog) all() []heardWords {
	w.mu.Lock()
	defer w.mu.Unlock()
	return slices.Clone(w.heard)
}

// teeSTT is a transcriber whose transcripts are also kept in a wordLog.
type teeSTT struct {
	stt.STT
	out chan stt.Event
}

func teeWords(inner stt.STT, log *wordLog) *teeSTT {
	t := &teeSTT{STT: inner, out: make(chan stt.Event, 64)}
	go func() {
		defer close(t.out)
		for event := range inner.Events() {
			if transcript, ok := event.(stt.Transcript); ok {
				log.add(transcript)
			}
			t.out <- event
		}
	}()
	return t
}

func (t *teeSTT) Events() <-chan stt.Event { return t.out }

// AskOnPauses passes the agent on to the transcriber, as the router's session does.
func (t *teeSTT) AskOnPauses(agent audiofloor.Asker) {
	if pauses, ok := t.STT.(audiofloor.Pauses); ok {
		pauses.AskOnPauses(agent)
	}
}

// timeline is an agent's events, each stamped as it arrived.
type timeline struct {
	mu     sync.Mutex
	events []stampedEvent[Event]
	done   chan struct{}
}

type stampedEvent[E any] struct {
	At time.Time
	E  E
}

func record(agent *Agent) *timeline {
	t := &timeline{done: make(chan struct{})}
	go func() {
		defer close(t.done)
		for event := range agent.Events() {
			t.mu.Lock()
			t.events = append(t.events, stampedEvent[Event]{At: time.Now(), E: event})
			t.mu.Unlock()
		}
	}()
	return t
}

func (t *timeline) all() []stampedEvent[Event] {
	t.mu.Lock()
	defer t.mu.Unlock()
	return slices.Clone(t.events)
}

// all are a call's events of one type, in the order they came.
func all[E Event](c *call) []stampedEvent[E] {
	var out []stampedEvent[E]
	for _, e := range c.events.all() {
		if typed, ok := e.E.(E); ok {
			out = append(out, stampedEvent[E]{At: e.At, E: typed})
		}
	}
	return out
}

// exchanges is a RoundTripper keeping each request to the server: when it went, when its
// answer ended, and the answer.
type exchanges struct {
	base http.RoundTripper
	mu   sync.Mutex
	kept []*exchange
}

type exchange struct {
	method, path, query string
	status              int
	// firstLine is when the answer's first line was read, which for a decision that writes
	// its words is the decision.
	start, firstLine, done time.Time
	body                   string
	err                    error
}

func (x *exchanges) RoundTrip(req *http.Request) (*http.Response, error) {
	kept := &exchange{method: req.Method, path: req.URL.Path, query: req.URL.RawQuery, start: time.Now()}
	x.mu.Lock()
	x.kept = append(x.kept, kept)
	x.mu.Unlock()
	resp, err := x.base.RoundTrip(req)
	if err != nil {
		x.mu.Lock()
		kept.err, kept.done = err, time.Now()
		x.mu.Unlock()
		return nil, err
	}
	x.mu.Lock()
	kept.status = resp.StatusCode
	x.mu.Unlock()
	resp.Body = &keptBody{ReadCloser: resp.Body, x: x, kept: kept}
	return resp, nil
}

func (x *exchanges) all() []exchange {
	x.mu.Lock()
	defer x.mu.Unlock()
	out := make([]exchange, 0, len(x.kept))
	for _, k := range x.kept {
		out = append(out, *k)
	}
	return out
}

type keptBody struct {
	io.ReadCloser
	x    *exchanges
	kept *exchange
	buf  bytes.Buffer
	once sync.Once
}

func (b *keptBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	b.buf.Write(p[:n])
	if bytes.IndexByte(p[:n], '\n') >= 0 {
		b.x.mu.Lock()
		if b.kept.firstLine.IsZero() {
			b.kept.firstLine = time.Now()
		}
		b.x.mu.Unlock()
	}
	return n, err
}

func (b *keptBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(func() {
		b.x.mu.Lock()
		b.kept.done, b.kept.body = time.Now(), b.buf.String()
		b.x.mu.Unlock()
	})
	return err
}

// logBook keeps an agent's log lines, and writes them to a file as JSON.
type logBook struct {
	file  *os.File
	mu    sync.Mutex
	lines []logLine
}

type logLine struct {
	At    time.Time
	Msg   string
	Attrs map[string]any
}

func newLogBook(t *testing.T, path string) *logBook {
	file, err := os.Create(path)
	require.NoError(t, err)
	return &logBook{file: file}
}

func (b *logBook) close() { _ = b.file.Close() }

func (b *logBook) find(msg string) []logLine {
	b.mu.Lock()
	defer b.mu.Unlock()
	var out []logLine
	for _, l := range b.lines {
		if l.Msg == msg {
			out = append(out, l)
		}
	}
	return out
}

type logHandler struct {
	book  *logBook
	attrs []slog.Attr
	json  slog.Handler
}

func (h *logHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *logHandler) Handle(ctx context.Context, r slog.Record) error {
	attrs := map[string]any{}
	for _, a := range h.attrs {
		attrs[a.Key] = a.Value.Any()
	}
	r.Attrs(func(a slog.Attr) bool {
		attrs[a.Key] = a.Value.Any()
		return true
	})
	h.book.mu.Lock()
	h.book.lines = append(h.book.lines, logLine{At: r.Time, Msg: r.Message, Attrs: attrs})
	h.book.mu.Unlock()
	return h.json.Handle(ctx, r)
}

func (h *logHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &logHandler{book: h.book, attrs: append(slices.Clone(h.attrs), attrs...), json: h.json.WithAttrs(attrs)}
}

func (h *logHandler) WithGroup(name string) slog.Handler {
	return &logHandler{book: h.book, attrs: h.attrs, json: h.json.WithGroup(name)}
}

// fixture is testdata/audiofloor/name.ogg, 16 kHz mono.
func fixture(t *testing.T, name string) []int16 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "audiofloor", name+".ogg"))
	require.NoError(t, err)
	reader, err := ogg.NewReader(bytes.NewReader(raw))
	require.NoError(t, err)
	decoder, err := gopus.NewDecoder(gopus.DefaultDecoderConfig(replayRate, 1))
	require.NoError(t, err)
	pcm := make([]int16, 5760)
	var out []int16
	var granule uint64
	for {
		packet, position, err := reader.ReadPacket()
		if errors.Is(err, io.EOF) {
			break
		}
		require.NoError(t, err)
		if bytes.HasPrefix(packet, []byte("OpusHead")) || bytes.HasPrefix(packet, []byte("OpusTags")) {
			continue
		}
		n, err := decoder.DecodeInt16(packet, pcm)
		require.NoError(t, err)
		out = append(out, pcm[:n]...)
		granule = max(granule, position)
	}
	skip := int(reader.PreSkip()) / 3
	end := min(len(out), int(granule)/3)
	require.Greater(t, end, skip, "%s holds no audio", name)
	return out[skip:end]
}

// voiced is where a clip's speech begins and ends: its first and last 20 ms at -40 dBFS or
// louder.
func voiced(samples []int16) (int, int) {
	from, to := -1, 0
	for at := 0; at+replayFrame <= len(samples); at += replayFrame {
		if rms(samples[at:at+replayFrame]) >= voicedLevel {
			if from < 0 {
				from = at
			}
			to = at + replayFrame
		}
	}
	if from < 0 {
		return 0, 0
	}
	return from, to
}

func trimmed(samples []int16) []int16 {
	from, to := voiced(samples)
	return samples[from:to]
}

func rms(samples []int16) float64 {
	var sum float64
	for _, s := range samples {
		sum += float64(s) * float64(s)
	}
	return math.Sqrt(sum/float64(len(samples))) / math.MaxInt16
}

// The line's sounds, made here: the same every run.

func level(dbfs float64) float64 { return math.Pow(10, dbfs/20) * math.MaxInt16 }

// scaled is samples at an RMS of dbfs.
func scaled(samples []float64, dbfs float64) []float64 {
	var sum float64
	for _, s := range samples {
		sum += s * s
	}
	gain := level(dbfs) / math.Sqrt(sum/float64(len(samples)))
	for i := range samples {
		samples[i] *= gain
	}
	return samples
}

// activeLevel is the RMS of samples' voiced frames, as a fraction of full scale: the level
// of a voice, its pauses left out.
func activeLevel(samples []int16) float64 {
	var sum float64
	var n int
	for at := 0; at+replayFrame <= len(samples); at += replayFrame {
		if r := rms(samples[at : at+replayFrame]); r >= voicedLevel {
			sum, n = sum+r*r, n+1
		}
	}
	if n == 0 {
		return 0
	}
	return math.Sqrt(sum / float64(n))
}

// bedAt is speech as a bed under the line, its voice at dbfs.
func bedAt(speech []int16, dbfs float64) []float64 {
	gain := level(dbfs) / (activeLevel(speech) * math.MaxInt16)
	out := make([]float64, len(speech))
	for i, s := range speech {
		out[i] = float64(s) * gain
	}
	return out
}

// bedBelow is speech as a bed under the line, its voice dbBelow the caller's in reference.
func bedBelow(speech, reference []int16, dbBelow float64) []float64 {
	return bedAt(speech, 20*math.Log10(activeLevel(reference))-dbBelow)
}

// withGapsUnder is speech with every pause in it cut to gap: a caller who runs their words
// together, so that no pause reaches a policy's first check.
func withGapsUnder(samples []int16, gap time.Duration) []int16 {
	keep := int(gap / replayFrameTime)
	var out []int16
	var quiet int
	for at := 0; at+replayFrame <= len(samples); at += replayFrame {
		frame := samples[at : at+replayFrame]
		if rms(frame) >= voicedLevel {
			quiet = 0
		} else if quiet++; quiet > keep {
			continue
		}
		out = append(out, frame...)
	}
	return trimmed(out)
}

// lineNoise is a phone line's hiss: white noise at dbfs.
func lineNoise(d time.Duration, dbfs float64) []float64 {
	rng := rand.New(rand.NewPCG(1, 2))
	out := make([]float64, int(d.Seconds()*replayRate))
	for i := range out {
		out[i] = rng.NormFloat64()
	}
	return scaled(out, dbfs)
}

// roomTone is a quiet room's rumble, a low hum and air: brown noise and 50 Hz at dbfs.
func roomTone(d time.Duration, dbfs float64) []float64 {
	rng := rand.New(rand.NewPCG(3, 4))
	out := make([]float64, int(d.Seconds()*replayRate))
	var brown float64
	for i := range out {
		brown = 0.995*brown + rng.NormFloat64()
		out[i] = brown + 2*math.Sin(2*math.Pi*50*float64(i)/replayRate) + 0.5*rng.NormFloat64()
	}
	return scaled(out, dbfs)
}

// burst is noise, low-passed by smooth, under envelope, at an RMS of dbfs over its loudest
// 20 ms.
func burst(seed uint64, d time.Duration, smooth float64, dbfs float64, envelope func(t float64) float64) []int16 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	n := int(d.Seconds() * replayRate)
	shaped := make([]float64, n)
	var low float64
	for i := range shaped {
		low = smooth*low + (1-smooth)*rng.NormFloat64()
		shaped[i] = low * envelope(float64(i)/float64(n))
	}
	var loudest float64
	for at := 0; at+replayFrame <= n; at += replayFrame {
		var sum float64
		for _, s := range shaped[at : at+replayFrame] {
			sum += s * s
		}
		loudest = max(loudest, math.Sqrt(sum/replayFrame))
	}
	out := make([]int16, n)
	for i, s := range shaped {
		out[i] = int16(max(math.MinInt16, min(math.MaxInt16, s*level(dbfs)/loudest)))
	}
	return out
}

// breath is an exhalation near the microphone: 600 ms of soft, airy noise.
func breath() []int16 {
	return burst(5, 600*time.Millisecond, 0.3, -34, func(t float64) float64 { return math.Pow(math.Sin(math.Pi*t), 2) })
}

// rustle is the handset brushing something: 350 ms of broadband noise, struck and fading.
func rustle() []int16 {
	return burst(7, 350*time.Millisecond, 0, -26, func(t float64) float64 { return math.Min(1, t*40) * math.Exp(-4*t) })
}

// cough is two short, hard bursts.
func cough() []int16 {
	hit := func(t float64) float64 { return math.Min(1, t*30) * math.Exp(-5*t) }
	first := burst(9, 220*time.Millisecond, 0.6, -14, hit)
	second := burst(11, 160*time.Millisecond, 0.6, -18, hit)
	return slices.Concat(make([]int16, replayRate/10), first, make([]int16, replayRate*12/100), second, make([]int16, replayRate/10))
}
