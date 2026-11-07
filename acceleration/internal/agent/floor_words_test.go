package agent

import (
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// endsAtOnce is a chorus-asr server's ruling that the turn has ended, at the first check.
const endsAtOnce = `{"answer":"respond","answer_probability":0.99,"p_eot":0.999,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"continue"}}`

// floorRules has the agent joined next act on chorus-asr at a server ruling every check as
// decision, while the stub transcriber, another than chorus-asr's, writes its callers' words. It
// says how many decisions the server has answered.
func (s *AgentSuite) floorRules(decision string) *atomic.Int64 {
	var decided atomic.Int64
	ruling := floorServer(s.T(), decision)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ruling.Config.Handler.ServeHTTP(w, r)
		if r.URL.Path == "/v1/decide" {
			decided.Add(1)
		}
	}))
	s.T().Cleanup(srv.Close)
	s.floor, s.floorMode = audiofloor.New(srv.URL, audiofloor.Opus, nil), audiofloor.Act
	return &decided
}

// callerLine is a caller's line as the edge hears it: a voice the speech detector takes for
// speech, or quiet, over a line's faint noise, a 20 ms frame at a time, four times as fast as
// it is said.
type callerLine struct {
	s           *AgentSuite
	participant stt.Participant
	phase       float64
	at          int
	noise       uint32
}

func (l *callerLine) say(ms int, voiced bool) {
	const rate = 16000
	for range ms / 20 {
		frame := make([]int16, rate/50)
		for i := range frame {
			l.noise = l.noise*1664525 + 1013904223
			frame[i] = int16((float64(l.noise>>16)/65535*2 - 1) * 16)
		}
		if voiced {
			// A wandering pitch, its harmonics and a syllable's rhythm.
			for i := range frame {
				second := float64(l.at) / rate
				pitch := 150 + 30*math.Sin(2*math.Pi*0.5*second) + 12*math.Sin(2*math.Pi*1.3*second)
				l.phase += 2 * math.Pi * pitch / rate
				value := 0.6*math.Sin(l.phase) + 0.25*math.Sin(2*l.phase) + 0.15*math.Sin(3*l.phase)
				envelope := 0.5 + 0.5*math.Pow(0.5+0.5*math.Sin(2*math.Pi*4.2*second), 1.5)
				frame[i] += int16(24000 * value * envelope)
				l.at++
			}
		}
		l.s.edge.inbound <- InboundAudio{Participant: l.participant, Audio: audio.PcmData{Samples: frame, SampleRate: rate, Channels: 1}}
		time.Sleep(5 * time.Millisecond)
	}
}

// transcriberWrites has the stub transcriber write words of run, as one that numbers its runs does.
func (s *AgentSuite) transcriberWrites(participant stt.Participant, run int64, mode stt.Mode, text string) {
	s.ears.emitter.Send(stt.Transcript{Participant: participant, Mode: mode, Utterance: run, Text: text, Language: "en",
		Confidence: 1, Provider: "stub", Model: "stub-model"})
}

// asksOverTheFloor has participant say text, the stub transcriber writing it, and pause, for a
// floor of chorus-asr's own to end the turn.
func (s *AgentSuite) asksOverTheFloor(participant stt.Participant, text string) {
	line := &callerLine{s: s, participant: participant}
	line.say(300, false)
	line.say(600, true)
	s.transcriberWrites(participant, 1, stt.ModeFinal, text)
	line.say(600, false)
}

// heardOnly waits for the conversation to have heard lines, and only them, then closes the agent
// and requires that it heard nothing more and answered each once.
func (s *AgentSuite) heardOnly(lines ...string) {
	s.T().Helper()
	s.eventually(func() bool { return slices.Equal(s.heardLines(), lines) }, "the lines heard were not "+`"`+lines[0]+`"`)
	s.eventually(func() bool { return len(s.model.turns()) == len(lines) }, "the lines heard were never answered")
	s.Require().NoError(s.agent.Close())
	s.Equal(lines, s.heardLines(), "heard more")
	s.Len(s.model.turns(), len(lines), "answered more than once")
}

// heardLines are the lines the conversation heard, in order.
func (s *AgentSuite) heardLines() []string {
	var lines []string
	for _, event := range s.reported() {
		if heard, ok := event.(Heard); ok {
			lines = append(lines, heard.Text)
		}
	}
	return lines
}

// Acting on chorus-asr with another transcriber, the turn ends where chorus-asr rules it did,
// at the caller's pause: the run of speech the transcriber ended at a breath mid-sentence is
// no turn, and the turn's words are all it wrote of the utterance, its final of the last run
// waited for.
func (s *AgentSuite) TestActingWithAnotherTranscriberARunItEndsMidSentenceIsNoTurn() {
	decided := s.floorRules(endsAtOnce)
	s.join(true)
	s.model.reply = []string{"Booked."}
	caller := stt.Participant{ID: "alice"}
	line := &callerLine{s: s, participant: caller}
	line.say(300, false)
	line.say(1200, true)
	s.transcriberWrites(caller, 1, stt.ModeReplacement, "A table for two at 7:30")
	s.transcriberWrites(caller, 1, stt.ModeFinal, "A table for two at 7:30, patio.")
	line.say(800, true)
	s.Empty(s.model.turns(), "answered while the caller was talking")
	s.transcriberWrites(caller, 2, stt.ModeReplacement, "peanut allergy")
	line.say(600, false)
	s.eventually(func() bool { return decided.Load() > 0 }, "chorus-asr was never asked about the pause")
	s.Empty(s.model.turns(), "answered before the transcriber settled the words")
	s.True(s.agent.actsOnChorusASR(), "the loop's contract is kept")
	s.transcriberWrites(caller, 2, stt.ModeFinal, "Peanut allergy.")

	s.heardOnly("A table for two at 7:30, patio. Peanut allergy.")
}

// The transcriber's words holding still while the caller talks on are no pause: the turn is
// answered after chorus-asr rules on the caller's pause, on the words as they stood when no
// final came in finalWait.
func (s *AgentSuite) TestActingWithAnotherTranscriberWordsHoldingStillAreNoPause() {
	s.floorRules(endsAtOnce)
	s.join(true)
	caller := stt.Participant{ID: "alice"}
	line := &callerLine{s: s, participant: caller}
	line.say(300, false)
	line.say(600, true)
	s.transcriberWrites(caller, 1, stt.ModeReplacement, "I'd like to book a table")
	line.say(2400, true)
	s.Empty(s.model.turns(), "answered while the caller was talking")
	line.say(600, false)

	s.heardOnly("I'd like to book a table")
}

// A final that comes after chorus-asr ruled the turn over, correcting what the transcriber had
// written of it, is waited for: the turn is answered on the final.
func (s *AgentSuite) TestActingWithAnotherTranscriberALateFinalIsWaitedFor() {
	decided := s.floorRules(endsAtOnce)
	s.join(true)
	caller := stt.Participant{ID: "alice"}
	line := &callerLine{s: s, participant: caller}
	line.say(300, false)
	line.say(1200, true)
	s.transcriberWrites(caller, 1, stt.ModeReplacement, "a table at seven thirty")
	line.say(600, false)
	s.eventually(func() bool { return decided.Load() > 0 }, "chorus-asr was never asked about the pause")
	s.Empty(s.model.turns(), "answered before the transcriber's final")
	s.transcriberWrites(caller, 1, stt.ModeFinal, "A table at 7:13.")

	s.heardOnly("A table at 7:13.")
}

// A chorus-asr server that can't rule has its floor step aside: the transcriber writing the
// words ends the turns, as with no floor, and the loop's contract goes with it.
func (s *AgentSuite) TestActingWithAnotherTranscriberAServerThatCantRuleLeavesTheTurnsToIt() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotImplemented)
		_, _ = w.Write([]byte(`{"error":"no policy for this model"}`))
	}))
	s.T().Cleanup(srv.Close)
	s.floor, s.floorMode = audiofloor.New(srv.URL, audiofloor.Opus, nil), audiofloor.Act
	s.join(true)
	caller := stt.Participant{ID: "alice"}
	line := &callerLine{s: s, participant: caller}
	line.say(300, false)
	s.eventually(s.agent.actsOnChorusASR, "chorus-asr never ruled on the caller's floor")
	line.say(600, true)
	line.say(400, false)
	s.eventually(func() bool { return !s.agent.actsOnChorusASR() }, "the contract was kept with no floor to keep it")

	s.transcriberWrites(caller, 2, stt.ModeFinal, "What time do you open?")
	s.heardOnly("What time do you open?")
}

// Non-chorus agents hear the transcriber as they always have: no floor of chorus-asr's, and its
// words holding still are put to the flow controller.
func (s *AgentSuite) TestWithoutChorusASRTheTranscribersWordsHoldingStillAreATurn() {
	s.join(true)
	caller := stt.Participant{ID: "alice"}
	s.speak(caller)
	s.transcriberWrites(caller, 1, stt.ModeReplacement, "what time do you open")
	s.eventually(func() bool { return len(s.model.turns()) == 1 }, "the words that held still were never answered")
	s.Nil(s.agent.audioFloor)
	s.False(s.agent.actsOnChorusASR())
}
