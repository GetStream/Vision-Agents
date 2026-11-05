package audiofloor

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"github.com/thesyncim/gopus"
	"github.com/thesyncim/gopus/container/ogg"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// testInfo is /v1/info as a server publishes it: the measured frontier of a checkpoint, and
// presets naming its points.
const testInfo = `{"template":"t","end_of_turn":{"first_check_ms":200,"first_pause_logit":0.5,"interval_ms":300,"cap_ms":2500,` +
	`"default":"safe","presets":{"safe":1100,"fast":500},"frontier":[` + testFrontier + `]}}`

// testFrontier is that checkpoint's frontier: its 500 ms point fails the dictation guard, and
// its 1400 ms point ends a turn on two checks in a row.
const testFrontier = `{"max_wait_ms":500,"bars":[{"from_ms":0,"bar":0.8},{"from_ms":500,"bar":0.3}],"checks_in_a_row":1,` +
	`"measured":{"cut_ins":0.1451,"answer_ms_median":500,"answer_ms_p90":800,"dictation_pauses_cut":0.16,"slow_counts_cut":9,"slow_counts":9,"dictation_guard":false}},` +
	`{"max_wait_ms":1100,"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}],"checks_in_a_row":1,` +
	`"measured":{"cut_ins":0.0454,"answer_ms_median":1100,"answer_ms_p90":1100,"dictation_pauses_cut":0.1257,"slow_counts_cut":0,"slow_counts":9,"dictation_guard":true}},` +
	`{"max_wait_ms":1400,"bars":[{"from_ms":0,"bar":0.99},{"from_ms":1100,"bar":0.2}],"checks_in_a_row":2,` +
	`"measured":{"cut_ins":0.0307,"answer_ms_median":1400,"answer_ms_p90":1400,"dictation_pauses_cut":0.1057,"slow_counts_cut":0,"slow_counts":9,"dictation_guard":true}}`

// fakeServer answers as an audio floor server does, with what a test sets.
type fakeServer struct {
	mu         sync.Mutex
	registered []agentJSON
	known      map[string]bool
	infos      int
	turns      []Turn
	audio      [][]byte
	types      []string
	languages  []string
	// refuse is the status /v1/agents answers an agent with tools, fail has /v1/decide
	// answer 500, and info is /v1/info's body.
	refuse   int
	fail     bool
	info     string
	decision string
	words    string
}

// requests is how many decisions the server was asked for.
func (f *fakeServer) requests() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.audio)
}

func (f *fakeServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch r.URL.Path {
	case "/v1/info":
		f.infos++
		_, _ = w.Write([]byte(f.info))
	case "/v1/agents":
		var spec agentJSON
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &spec)
		f.registered = append(f.registered, spec)
		if f.refuse != 0 && len(spec.Tools) > 0 {
			w.WriteHeader(f.refuse)
			_, _ = w.Write([]byte(`{"error":"floor: 22 tools, and an agent can have 21"}`))
			return
		}
		id := fmt.Sprintf("agent-%d", len(f.registered))
		f.known[id] = true
		_ = json.NewEncoder(w).Encode(map[string]string{"agent": id})
	case "/v1/transcribe":
		if f.fail {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		raw, _ := io.ReadAll(r.Body)
		f.audio, f.types = append(f.audio, raw), append(f.types, r.Header.Get("Content-Type"))
		f.languages = append(f.languages, r.URL.Query().Get("languages"))
		_, _ = fmt.Fprintf(w, `{"transcript":{"text":%q}}`+"\n", f.words)
	case "/v1/decide":
		if f.fail {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		if !f.known[r.URL.Query().Get("agent")] {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, params, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
		parts := multipart.NewReader(r.Body, params["boundary"])
		var turn Turn
		for _, name := range []string{"turn", "audio"} {
			part, err := parts.NextPart()
			if err != nil || part.FormName() != name {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			raw, _ := io.ReadAll(part)
			if name == "turn" {
				_ = json.Unmarshal(raw, &turn)
				continue
			}
			f.audio = append(f.audio, raw)
			f.types = append(f.types, part.Header.Get("Content-Type"))
		}
		f.turns = append(f.turns, turn)
		_, _ = w.Write([]byte(f.decision + "\n"))
	}
}

// serverSuite is a Client of a fake server.
type serverSuite struct {
	suite.Suite
	server *fakeServer
	client *Client
}

func (s *serverSuite) SetupTest() {
	s.server = &fakeServer{known: map[string]bool{}, info: testInfo,
		decision: `{"answer":"book","answer_probability":0.9,"p_eot":0.95,"options":[{"name":"book","p":0.9},{"name":"wait","p":0.05}],"decide_ms":12,"ruling":{"disposition":"respond","floor":"continue","tool":"book"}}`}
	srv := httptest.NewServer(s.server)
	s.T().Cleanup(srv.Close)
	s.client = New(srv.URL+"/", PCM, srv.Client())
}

// ClientSuite asks a fake server.
type ClientSuite struct{ serverSuite }

func TestClientSuite(t *testing.T) {
	suite.Run(t, new(ClientSuite))
}

var booking = Agent{Instructions: "Take bookings.", Tools: []llm.Tool{{Name: "book", Description: "Book it.", Parameters: map[string]any{"required": []any{"day"}}}}}

// some is a moment of audio.
var some = harness.Audio{Type: pcmType, Data: []byte{1, 0, 0xff, 0xff}, Duration: time.Millisecond / 8}

func (s *ClientSuite) TestAnAgentIsRegisteredOnceAndTheMomentGoesInTheBody() {
	for range 2 {
		got, err := s.client.Decide(context.Background(), booking, Turn{Speaker: caller, Speaking: true, Said: "Votre traversée à 8 h"}, some)
		s.Require().NoError(err)
		s.Equal(Ruling{Disposition: "respond", Floor: "continue", Tool: "book"}, *got.Ruling)
		s.InDelta(0.95, *got.PEOT, 1e-9)
	}
	s.Len(s.server.registered, 1)
	s.Equal([]any{"day"}, s.server.registered[0].Tools[0].Parameters["required"])
	s.Equal("Votre traversée à 8 h", s.server.turns[0].Said)
	s.Equal([]byte{1, 0, 0xff, 0xff}, s.server.audio[0])
	s.Equal("audio/pcm;rate=16000;channels=1;format=s16le", s.server.types[0])
}

func (s *ClientSuite) TestAChangedAgentIsRegisteredAgain() {
	for _, instructions := range []string{"One.", "One.", "Two."} {
		_, err := s.client.Decide(context.Background(), Agent{Instructions: instructions}, Turn{}, some)
		s.Require().NoError(err)
	}
	s.Len(s.server.registered, 2)
}

// A restarted server has forgotten its agents, and may serve another model and policy.
func (s *ClientSuite) TestARestartedServerIsAskedAgain() {
	_, err := s.client.Policy(context.Background())
	s.Require().NoError(err)
	_, err = s.client.Decide(context.Background(), booking, Turn{}, some)
	s.Require().NoError(err)
	s.server.mu.Lock()
	s.server.known = map[string]bool{}
	s.server.mu.Unlock()
	_, err = s.client.Decide(context.Background(), booking, Turn{}, some)
	s.Require().NoError(err)
	s.Len(s.server.registered, 2)
	_, err = s.client.Policy(context.Background())
	s.Require().NoError(err)
	s.Equal(2, s.server.infos)
}

// An agent with more tools than the model has answers for is refused, and isn't offered
// again until it changes.
func (s *ClientSuite) TestARefusedAgentIsRefusedUntilItChanges() {
	s.server.refuse = http.StatusUnprocessableEntity
	for range 2 {
		_, err := s.client.Decide(context.Background(), booking, Turn{}, some)
		s.ErrorIs(err, ErrRefused)
		s.ErrorContains(err, "22 tools")
	}
	s.Len(s.server.registered, 1)
	_, err := s.client.Decide(context.Background(), Agent{Instructions: booking.Instructions}, Turn{}, some)
	s.Require().NoError(err)
}

func (s *ClientSuite) TestAServerWithoutAPolicyIsUnsupported() {
	s.server.refuse = http.StatusNotImplemented
	_, err := s.client.Decide(context.Background(), booking, Turn{}, some)
	s.ErrorIs(err, ErrUnsupported)
	const times = `"first_check_ms":200,"interval_ms":300,"cap_ms":2500`
	point := func(bars string) string {
		return `{"max_wait_ms":1100,"bars":` + bars + `,"checks_in_a_row":1,"measured":{"cut_ins":0.05,"answer_ms_median":1100,"answer_ms_p90":1100}}`
	}
	for _, info := range []string{
		`{"end_of_turn":{}}`,
		`{"end_of_turn":{` + times + `,"default":"a","presets":{"a":1100}}}`,
		`{"end_of_turn":{` + times + `,"default":"b","presets":{"a":1100},"frontier":[` + point(`[{"from_ms":0,"bar":0.9}]`) + `]}}`,
		`{"end_of_turn":{` + times + `,"default":"a","presets":{"a":1100,"b":900},"frontier":[` + point(`[{"from_ms":0,"bar":0.9}]`) + `]}}`,
		`{"end_of_turn":{` + times + `,"default":"a","presets":{"a":1100},"frontier":[` + point(`[{"from_ms":500,"bar":0.9}]`) + `]}}`,
		`{"end_of_turn":{` + times + `,"default":"a","presets":{"a":1100},"frontier":[` + point(`[{"from_ms":0,"bar":0.9},{"from_ms":0,"bar":0.2}]`) + `]}}`,
		// Presets of bars, as servers published them before the frontier.
		`{"end_of_turn":{` + times + `,"default":"a","presets":{"a":{"bars":[{"from_ms":0,"bar":0.9}]}}}}`,
		`{"end_of_turn":{` + times + `,"first_bar":0.7,"later_bar":0.2}}`,
	} {
		s.server.info = info
		_, err = s.client.Policy(context.Background())
		s.ErrorIs(err, ErrUnsupported, info)
	}
}

// The words alone need no agent and no policy, and name the callers' languages.
func (s *ClientSuite) TestTheWordsAloneAreAskedForAlone() {
	s.server.words = "What time is the last ferry?"
	got, err := s.client.Transcribe(context.Background(), some, []string{"en", "pt-BR"})
	s.Require().NoError(err)
	s.Equal("What time is the last ferry?", got)
	s.Equal([]string{"en,pt-BR"}, s.server.languages)
	s.Equal([][]byte{some.Data}, s.server.audio)
	s.Equal([]string{pcmType}, s.server.types)
	s.Empty(s.server.registered)
	s.Zero(s.server.infos)
}

// The body carries no more than a prompt can show: the reply's last 800 characters, and the
// lines of the last six turns, a run of one side's lines being one turn.
func (s *ClientSuite) TestTheBodyCarriesWhatAPromptCanShow() {
	var lines []Line
	for i := range 9 {
		lines = append(lines, Line{Agent: i%2 == 1, Text: fmt.Sprintf("turn %d", i)})
	}
	lines = append(lines[:6], append([]Line{{Agent: true, Text: "turn 5, more"}}, lines[6:]...)...)
	lines[8].Text = "early " + strings.Repeat("late ", maxLineWords)
	_, err := s.client.Decide(context.Background(), Agent{}, Turn{Speaking: true, Said: "x" + strings.Repeat("é", maxSaid), Conversation: lines}, some)
	s.Require().NoError(err)
	got := s.server.turns[0]
	s.Equal(strings.Repeat("é", maxSaid), got.Said)
	want := slices.Clone(lines[3:])
	want[5].Text = strings.TrimSpace(strings.Repeat("late ", maxLineWords))
	s.Equal(want, got.Conversation)
	s.Equal("early "+strings.Repeat("late ", maxLineWords), lines[8].Text, "the caller's lines are theirs")
}

func (s *ClientSuite) TestNoAudioIsNotAsked() {
	_, err := s.client.Decide(context.Background(), booking, Turn{}, harness.Audio{})
	s.Require().Error(err)
	s.Zero(s.server.requests())
}

// The bar is the last of the bars to have begun by the check's time into the pause,
// raised on the turn's first pause.
func TestTheBarIsTheOneBegunByTheCheck(t *testing.T) {
	var info struct {
		EndOfTurn Policy `json:"end_of_turn"`
	}
	require.NoError(t, json.Unmarshal([]byte(testInfo), &info))
	p := info.EndOfTurn
	for _, c := range []struct {
		paused time.Duration
		want   float64
	}{{200 * time.Millisecond, 0.995}, {800 * time.Millisecond, 0.995}, {1099 * time.Millisecond, 0.995},
		{1100 * time.Millisecond, 0.20}, {2 * time.Second, 0.20}} {
		require.InDelta(t, c.want, p.Bar(p.DefaultEnding().Bars, c.paused, false), 1e-9, "%v into the pause", c.paused)
	}
	require.InDelta(t, 0.292, p.Bar(p.DefaultEnding().Bars, 1100*time.Millisecond, true), 1e-3)
	require.InDelta(t, 0.997, p.Bar(p.DefaultEnding().Bars, 800*time.Millisecond, true), 1e-3)
}

// HearerSuite reads the model's answers as the flow controller's rulings.
type HearerSuite struct {
	serverSuite
	hearer *Hearer
}

func TestHearerSuite(t *testing.T) {
	suite.Run(t, new(HearerSuite))
}

func (s *HearerSuite) SetupTest() {
	s.serverSuite.SetupTest()
	s.hearer = NewHearer(s.client, EndOfTurn{}, slog.New(slog.DiscardHandler))
}

func (s *HearerSuite) answering(pEOT float64, ruling string) {
	s.server.decision = fmt.Sprintf(`{"answer":"wait","answer_probability":%v,"p_eot":%v,"options":[],"decide_ms":1,"ruling":%s}`, 1-pEOT, pEOT, ruling)
}

// p_eot of 0.6 doesn't end a turn at a check 800 ms into the pause, where the bar is 0.995,
// and ends it at one 1.1 s in, where the bar is 0.2, as a ruling the server read past wait.
func (s *HearerSuite) TestATurnHasEndedWhenPEndReachesTheBar() {
	s.answering(0.6, `{"disposition":"respond","floor":"continue","tool":"book"}`)
	first, err := s.hearer.Check(context.Background(), harness.FlowTurn{Audio: some, Waited: true, Paused: 800 * time.Millisecond})
	s.Require().NoError(err)
	s.False(first.Ended)
	s.InDelta(0.995, first.Bar, 1e-9)
	s.Equal("preset:safe", first.Ending.Name)
	s.Equal(harness.Decided{Disposition: harness.Wait, Floor: harness.Continue, Again: 300 * time.Millisecond}, first.Ruling)

	later, err := s.hearer.Check(context.Background(), harness.FlowTurn{Audio: some, Waited: true, Paused: 1100 * time.Millisecond})
	s.Require().NoError(err)
	s.True(later.Ended)
	s.Equal(harness.Decided{Disposition: harness.Respond, Floor: harness.Continue, Tool: "book"}, later.Ruling)
}

func (s *HearerSuite) TestATurnHasEndedAtTheCapWhateverTheModelSays() {
	s.answering(0.01, `{"disposition":"respond","floor":"continue"}`)
	got, err := s.hearer.Check(context.Background(), harness.FlowTurn{Audio: some, Waited: true, Paused: 2500 * time.Millisecond})
	s.Require().NoError(err)
	s.Equal(harness.Respond, got.Ruling.Disposition)
}

// Over the agent there is no wait to weigh, and the ruling is the model's.
func (s *HearerSuite) TestOverTheAgentTheRulingIsTheModels() {
	s.server.decision = `{"answer":"stop","answer_probability":0.8,"options":[],"decide_ms":1,"ruling":{"disposition":"wait","floor":"stop"}}`
	got, err := s.hearer.Hear(context.Background(), harness.FlowTurn{Audio: some, Speaking: true, Unfinished: true, Reply: "Your crossing is"})
	s.Require().NoError(err)
	s.Equal(harness.Decided{Disposition: harness.Wait, Floor: harness.Stop}, got)
	s.Equal(Turn{Speaker: caller, Speaking: true, Said: "Your crossing is", Unfinished: true}, s.server.turns[0])
}

// A participant's name never reaches the model, the agent's reply only matters over it, and
// the conversation is the caller's and the agent's words.
func (s *HearerSuite) TestATurnIsSaidInTheModelsTerms() {
	history := []llm.Message{{Role: llm.System, Content: "x"}, {Role: llm.User, Content: "hi"}, {Role: llm.Assistant, Content: "hello"}, {Role: llm.User, Content: " "}}
	_, err := s.hearer.Hear(context.Background(), harness.FlowTurn{Audio: some, Participant: "Alice", AnotherVoice: true, Reply: "hello", Unfinished: true, History: history})
	s.Require().NoError(err)
	s.Equal(Turn{Speaker: anotherVoice, Conversation: []Line{{Text: "hi"}, {Agent: true, Text: "hello"}}}, s.server.turns[0])
}

// TranscriberSuite runs the transcriber on frames of voice, of noise and of quiet.
type TranscriberSuite struct {
	serverSuite
	transcriber *Transcriber
	// phase and at keep the voice going from one frame to the next, and noise the line's
	// noise, floor loud.
	phase float64
	at    int
	noise uint32
	floor float64
}

func TestTranscriberSuite(t *testing.T) {
	suite.Run(t, new(TranscriberSuite))
}

func (s *TranscriberSuite) SetupTest() {
	s.serverSuite.SetupTest()
	s.server.words = "Two tickets to Bilbao, please."
	s.client.format = Opus
	s.floor = 16 // about -66 dBFS
	s.open()
}

// open starts the transcriber, and sends the speech detector's warm-up, as a call's first
// moments.
func (s *TranscriberSuite) open() {
	s.transcriber = NewTranscriber(s.client, "", []string{"en", "pt-BR"}, slog.New(slog.DiscardHandler))
	s.Require().NoError(s.transcriber.Start(context.Background()))
	transcriber := s.transcriber
	s.T().Cleanup(func() { s.Require().NoError(transcriber.Close()) })
	s.say(300, false)
}

// pause is long enough to have the words written: the detector's window ends 10 ms before
// the newest sample, so a pause is heard a frame late.
const pause = int((writeAfter + 2*frame) / time.Millisecond)

// requests are the server's transcription requests, by how much audio each carried.
func (s *TranscriberSuite) requests() []int {
	s.server.mu.Lock()
	defer s.server.mu.Unlock()
	var out []int
	for _, body := range s.server.audio {
		out = append(out, decoded(s.T(), body))
	}
	return out
}

// say sends ms of 20 ms frames of the line's noise, voiced or quiet.
func (s *TranscriberSuite) say(ms int, voiced bool) {
	for range ms / 20 {
		frame := make([]int16, frameSamples)
		for i := range frame {
			s.noise = s.noise*1664525 + 1013904223
			frame[i] = int16((float64(s.noise>>16)/65535*2 - 1) * s.floor)
		}
		if voiced {
			// The transcription service's voiced fixture: a wandering pitch, its harmonics
			// and a syllable's rhythm, which the speech detector scores as speech.
			for i := range frame {
				second := float64(s.at) / SampleRate
				pitch := 150 + 30*math.Sin(2*math.Pi*0.5*second) + 12*math.Sin(2*math.Pi*1.3*second)
				s.phase += 2 * math.Pi * pitch / SampleRate
				value := 0.6*math.Sin(s.phase) + 0.25*math.Sin(2*s.phase) + 0.15*math.Sin(3*s.phase)
				envelope := 0.5 + 0.5*math.Pow(0.5+0.5*math.Sin(2*math.Pi*4.2*second), 1.5)
				frame[i] += int16(24000 * value * envelope)
				s.at++
			}
		}
		s.send(frame)
	}
}

func (s *TranscriberSuite) send(frame []int16) {
	s.Require().NoError(s.transcriber.ProcessAudio(audio.PcmData{Samples: frame, SampleRate: SampleRate, Channels: 1}, stt.Participant{ID: "p"}))
}

// next is the next event of type E.
func next[E stt.Event](s *TranscriberSuite) E {
	s.T().Helper()
	for {
		select {
		case event := <-s.transcriber.Events():
			if typed, ok := event.(E); ok {
				return typed
			}
		case <-time.After(5 * time.Second):
			s.FailNow("no event")
		}
	}
}

// none says no transcript comes.
func (s *TranscriberSuite) none() {
	s.T().Helper()
	for {
		select {
		case event := <-s.transcriber.Events():
			_, ok := event.(stt.Transcript)
			s.Require().False(ok, "a transcript: %+v", event)
		case <-time.After(100 * time.Millisecond):
			return
		}
	}
}

// An utterance's words are written at its pauses, the whole of it each time, and it ends
// at a pause of endAfter with one final on the words last written. The callers' languages
// go with each request.
func (s *TranscriberSuite) TestAnUtteranceIsWrittenAtItsPausesAndEndsOnce() {
	s.say(1700, false)
	s.say(1000, true)
	s.say(180, false)
	s.Empty(s.requests())
	s.say(pause-180, false)
	said := next[stt.Transcript](s)
	s.Equal(stt.ModeReplacement, said.Mode)
	s.Equal(int64(1), said.Utterance)
	s.Equal("Two tickets to Bilbao, please.", said.Text)
	s.InDelta(float64(300+1000+writeAfter.Milliseconds()), said.AudioDurationMs, 40, "300 ms before the voice, then the voice and the pause")
	s.Equal("audio/ogg", s.server.types[0])
	s.InDelta(said.AudioDurationMs*SampleRate/1000-104, float64(s.requests()[0]), 1, "less the pre-skip")
	s.Equal("en,pt-BR", s.server.languages[0])

	s.say(500, true)
	s.say(pause, false)
	more := next[stt.Transcript](s)
	s.Equal(int64(1), more.Utterance)
	s.InDelta(said.AudioDurationMs+float64(500+writeAfter.Milliseconds()), more.AudioDurationMs, 60)

	s.say(int(endAfter/time.Millisecond), false)
	end := next[stt.Transcript](s)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance)
	s.Len(s.requests(), 2, "a final with no speech since the last words asks for none")
}

// Silence after a final writes nothing more: the utterance is over, and the next waits for
// speech.
func (s *TranscriberSuite) TestSilenceAfterAFinalWritesNothing() {
	s.say(1000, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	s.Equal(stt.ModeReplacement, next[stt.Transcript](s).Mode)
	s.Equal(stt.ModeFinal, next[stt.Transcript](s).Mode)
	s.say(15000, false)
	s.none()
	s.Len(s.requests(), 1)
}

// A line whose noise hovers about -48 dBFS, as loud as quiet speech, writes nothing of its
// own, however long the silence: only speech asks for words. Taken by its level alone, as
// the transcriber once did, the noise asked for the words again at every crossing.
func (s *TranscriberSuite) TestANoisyLineWritesNothingOfItsOwn() {
	s.Require().NoError(s.transcriber.Close())
	s.floor = 230 // a frame's root mean square about voiceLevel
	s.open()
	s.say(1000, true)
	// Over noise as loud as voiceLevel the pause is heard once the detector's score has let
	// go of the voice, about 200 ms later.
	s.say(int(endAfter/time.Millisecond)+400, false)
	s.Equal(stt.ModeReplacement, next[stt.Transcript](s).Mode)
	s.Equal(stt.ModeFinal, next[stt.Transcript](s).Mode)
	s.say(15000, false)
	s.none()
	s.Len(s.requests(), 1)
}

// Audio is never written twice: once an utterance has ended, the next one, however soon it
// begins, carries none of its audio.
func (s *TranscriberSuite) TestNoAudioIsWrittenTwice() {
	s.say(1000, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	next[stt.Transcript](s)
	s.True(next[stt.Transcript](s).Final())
	s.say(600, true)
	s.say(pause, false)
	again := next[stt.Transcript](s)
	s.Equal(int64(2), again.Utterance)
	s.InDelta(float64(100+600+writeAfter.Milliseconds()), again.AudioDurationMs, 40, "the 100 ms since the final, the new speech and its pause")
	s.Len(s.requests(), 2, "one pass over each utterance")
}

// A caller who says more after a final starts a new utterance: a final is the last word on
// its own, and nothing about its audio follows it.
func (s *TranscriberSuite) TestSpeechAfterAFinalIsANewUtterance() {
	s.say(1000, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	next[stt.Transcript](s)
	first := next[stt.Transcript](s)
	s.True(first.Final())
	s.say(800, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	for _, mode := range []stt.Mode{stt.ModeReplacement, stt.ModeFinal} {
		said := next[stt.Transcript](s)
		s.Equal(mode, said.Mode)
		s.Equal(int64(2), said.Utterance)
	}
	s.none()
}

// The transcriber asks for the words alone: no agent, no decision and no policy, so a
// server that serves no policy transcribes as well.
func (s *TranscriberSuite) TestItNeedsNoAgentNorPolicy() {
	s.server.mu.Lock()
	s.server.info = ""
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(pause, false)
	s.Equal("Two tickets to Bilbao, please.", next[stt.Transcript](s).Text)
	s.Empty(s.server.registered)
	s.Empty(s.server.turns)
	s.Zero(s.server.infos)
}

// Empty words are nothing heard: what the server dropped as noise, or what had no words.
func (s *TranscriberSuite) TestEmptyWordsAreNothingHeard() {
	s.server.mu.Lock()
	s.server.words = ""
	s.server.mu.Unlock()
	s.say(200, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	s.none()
	s.Len(s.requests(), 1, "the pause's words, and none at the end, which has nothing more")
}

// Words that come back empty at a later pause don't take back the words before them, and
// the utterance ends on those.
func (s *TranscriberSuite) TestAnUtteranceEndsOnItsLastWords() {
	s.say(1000, true)
	s.say(pause, false)
	s.Equal("Two tickets to Bilbao, please.", next[stt.Transcript](s).Text)
	s.server.mu.Lock()
	s.server.words = ""
	s.server.mu.Unlock()
	s.say(300, true)
	s.say(int(endAfter/time.Millisecond)+100, false)
	end := next[stt.Transcript](s)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal("Two tickets to Bilbao, please.", end.Text)
}

// A server that can't write the words hands the call to the next transcriber.
func (s *TranscriberSuite) TestAServerThatFailsIsFatal() {
	s.server.mu.Lock()
	s.server.fail = true
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(pause, false)
	s.True(next[stt.Error](s).Fatal)
}

// speech is a voice-like test signal: a falling pitch with its harmonics, swelling and
// fading as syllables do.
func speech(n int) []int16 {
	out := make([]int16, n)
	phase := 0.0
	for i := range out {
		t := float64(i) / SampleRate
		pitch := 140 + 40*math.Sin(2*math.Pi*0.7*t)
		phase += 2 * math.Pi * pitch / SampleRate
		var v float64
		for h := 1.0; h <= 12; h++ {
			v += math.Sin(h*phase) / h
		}
		out[i] = int16(5000 * v * (0.6 + 0.4*math.Sin(2*math.Pi*3*t)))
	}
	return out
}

// demux reads an Ogg Opus body back: its packets and granule positions, and its pre-skip.
func demux(t *testing.T, body []byte) ([][]byte, []uint64, int) {
	t.Helper()
	r, err := ogg.NewReader(bytes.NewReader(body))
	require.NoError(t, err)
	require.EqualValues(t, 1, r.Channels())
	var packets [][]byte
	var granules []uint64
	for {
		packet, granule, err := r.ReadPacket()
		if errors.Is(err, io.EOF) {
			break
		}
		require.NoError(t, err)
		packets, granules = append(packets, slices.Clone(packet)), append(granules, granule)
	}
	return packets, granules, int(r.PreSkip())
}

// decode is what a decoder that starts on body's first packet hears: the audio at 16 kHz,
// less the pre-skip, to the last page's granule position.
func decode(t *testing.T, body []byte) []int16 {
	t.Helper()
	packets, granules, preSkip := demux(t, body)
	decoder, err := gopus.NewDecoder(gopus.DefaultDecoderConfig(SampleRate, 1))
	require.NoError(t, err)
	var pcm []int16
	buf := make([]int16, 5760)
	for _, packet := range packets {
		n, err := decoder.DecodeInt16(packet, buf)
		require.NoError(t, err)
		pcm = append(pcm, buf[:n]...)
	}
	return pcm[preSkip/3 : granules[len(granules)-1]/3]
}

func decoded(t *testing.T, body []byte) int { return len(decode(t, body)) }

// What the stream writes is the packets it encoded, as each arrived, in pieces of whatever
// size, and they decode back to the audio written: the frames of its last 16 s, from the
// first frame's start.
func TestAStreamDecodesBackToWhatItWasWritten(t *testing.T) {
	stream, err := newStream(Opus, Window)
	require.NoError(t, err)
	written := speech(30 * SampleRate)
	for at, sizes := 0, []int{160, 333, 960, 320, 7}; at < len(written); {
		n := min(sizes[at%len(sizes)], len(written)-at)
		require.NoError(t, stream.Write(written[at:at+n]))
		at += n
	}
	frames := len(written) / frameSamples
	require.EqualValues(t, frames, stream.Frames())

	got := stream.Last(Window)
	require.Equal(t, "audio/ogg", got.Type)
	require.Equal(t, Window, got.Duration)
	packets, granules, preSkip := demux(t, got.Data)
	held := int(Window / frame)
	require.Len(t, packets, held)
	require.LessOrEqual(t, bytes.Count(got.Data, []byte("OggS")), 2+held/255+1, "packets laced many to a page")
	for i, packet := range packets {
		require.Equal(t, stream.frames[(stream.next+i)%held], packet, "packet %d", i)
	}
	require.EqualValues(t, 312, preSkip, "the encoder's lookahead, 6.5 ms, at 48 kHz")
	require.EqualValues(t, held*960, granules[len(granules)-1])

	heard := decode(t, got.Data)
	require.Len(t, heard, held*frameSamples-104)
	// The audio heard is the audio written, in time: an encoder codes a sound, not its
	// waveform, so the two are compared by their loudness frame by frame, and by where
	// they line up best. A decoder that starts mid-stream converges within a few frames.
	written = written[(frames-held)*frameSamples:]
	require.Greater(t, correlation(loudness(written[960:len(heard)]), loudness(heard[960:])), 0.95)
	lag, best := 0, math.Inf(-1)
	for shift := -frameSamples; shift <= frameSamples; shift++ {
		if c := correlation(toFloat(written[2*frameSamples+shift:][:SampleRate]), toFloat(heard[2*frameSamples:][:SampleRate])); c > best {
			lag, best = shift, c
		}
	}
	require.InDelta(t, 0, lag, 4, "the pre-skip lines the audio up with what was written")
}

// A packet of 255 bytes or more takes several lacing values, and a page holds at most 255.
func TestOggLacesEveryPacketWhole(t *testing.T) {
	var packets [][]byte
	for i, size := range append([]int{1, 254, 255, 256, 510, 600}, slices.Repeat([]int{3}, 300)...) {
		packets = append(packets, bytes.Repeat([]byte{byte(i)}, size))
	}
	got, granules, _ := demux(t, oggOpus(packets, 312))
	require.Equal(t, packets, got)
	require.EqualValues(t, len(packets)*960, granules[len(granules)-1])
}

// loudness is each frame's root mean square.
func loudness(pcm []int16) []float64 {
	var out []float64
	for f := range slices.Chunk(pcm, frameSamples) {
		var sum float64
		for _, v := range f {
			sum += float64(v) * float64(v)
		}
		out = append(out, math.Sqrt(sum/float64(len(f))))
	}
	return out
}

func toFloat(pcm []int16) []float64 {
	out := make([]float64, len(pcm))
	for i, v := range pcm {
		out[i] = float64(v)
	}
	return out
}

// correlation is Pearson's of two series of the same length.
func correlation(a, b []float64) float64 {
	n := float64(min(len(a), len(b)))
	var sa, sb, saa, sbb, sab float64
	for i := range int(n) {
		sa, sb = sa+a[i], sb+b[i]
		saa, sbb, sab = saa+a[i]*a[i], sbb+b[i]*b[i], sab+a[i]*b[i]
	}
	return (sab - sa*sb/n) / math.Sqrt((saa-sa*sa/n)*(sbb-sb*sb/n))
}

// Less than the window is what was heard, and nothing is nothing.
func TestAStreamSendsWhatItHas(t *testing.T) {
	stream, err := newStream(Opus, Window)
	require.NoError(t, err)
	require.Empty(t, stream.Last(Window).Data)
	require.NoError(t, stream.Write(speech(frameSamples*5/2)))
	got := stream.Last(Window)
	require.Equal(t, 2*frame, got.Duration, "half a frame waits for the rest")
	require.Equal(t, 2*frameSamples-104, decoded(t, got.Data))
	require.Equal(t, frame, stream.Last(frame).Duration)
}

// PCM is the frames' samples as they were written.
func TestAPCMStreamIsTheSamples(t *testing.T) {
	stream, err := newStream(PCM, time.Second)
	require.NoError(t, err)
	written := speech(3 * SampleRate)
	require.NoError(t, stream.Write(written))
	got := stream.Last(time.Second)
	require.Equal(t, pcmType, got.Type)
	require.Equal(t, audio.PcmData{Samples: written[2*SampleRate:], SampleRate: SampleRate, Channels: 1}.Bytes(), got.Data)
}

// Writing a frame allocates nothing once its slot in the ring has room for the packet.
func TestAStreamWritesWithoutAllocating(t *testing.T) {
	stream, err := newStream(Opus, time.Second)
	require.NoError(t, err)
	quiet := make([]int16, frameSamples)
	for range 2 * 50 {
		require.NoError(t, stream.Write(quiet))
	}
	require.Zero(t, testing.AllocsPerRun(100, func() { _ = stream.Write(quiet) }))
}

// benchSpeech is AUDIO_FLOOR_WAV's audio, a 16 kHz mono 16-bit WAV, or the test signal.
func benchSpeech(b *testing.B, n int) []int16 {
	path := os.Getenv("AUDIO_FLOOR_WAV")
	if path == "" {
		return speech(n)
	}
	raw, err := os.ReadFile(path)
	require.NoError(b, err)
	at := bytes.Index(raw, []byte("data"))
	pcm := audio.FromBytes(raw[at+8:at+8+int(binary.LittleEndian.Uint32(raw[at+4:]))], SampleRate, 1).Samples
	for len(pcm) < n {
		pcm = append(pcm, pcm...)
	}
	return pcm[:n]
}

// What a participant's audio costs as it arrives: one frame encoded.
func BenchmarkAFrame(b *testing.B) {
	for _, format := range []Format{Opus, PCM} {
		b.Run(string(format), func(b *testing.B) {
			pcm := benchSpeech(b, 60*SampleRate)
			stream, err := newStream(format, Window)
			require.NoError(b, err)
			b.ResetTimer()
			for i := range b.N {
				at := i % (len(pcm) / frameSamples) * frameSamples
				_ = stream.Write(pcm[at : at+frameSamples])
			}
		})
	}
}

// What a decision costs the router: the body of a full window.
func BenchmarkADecisionsBody(b *testing.B) {
	for _, format := range []Format{Opus, PCM} {
		b.Run(string(format), func(b *testing.B) {
			stream, err := newStream(format, Window)
			require.NoError(b, err)
			require.NoError(b, stream.Write(benchSpeech(b, int(Window/time.Second)*SampleRate)))
			turn := Turn{Speaker: caller, Conversation: []Line{{Agent: true, Text: "Weather line, how can I help?"}}}
			var size int
			b.ResetTimer()
			for range b.N {
				body, _, err := decisionBody(turn, stream.Last(Window))
				require.NoError(b, err)
				size = len(body)
			}
			b.ReportMetric(float64(size), "bytes")
		})
	}
}
