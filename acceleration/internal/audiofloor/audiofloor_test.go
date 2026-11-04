package audiofloor

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const testInfo = `{"template":"t","end_of_turn":{"first_check_ms":200,"first_bar":0.7,"later_bar":0.2,"first_pause_logit":0.5,"interval_ms":300,"cap_ms":2500}}`

// fakeServer answers as an audio floor server does, with what a test sets.
type fakeServer struct {
	mu         sync.Mutex
	registered []agentJSON
	known      map[string]bool
	infos      int
	turns      []Turn
	audio      [][]byte
	types      []string
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
		if r.URL.Query().Get("transcript") == "true" {
			_, _ = fmt.Fprintf(w, `{"transcript":{"text":%q}}`+"\n", f.words)
		}
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
	s.client = New(srv.URL+"/", srv.Client())
}

// ClientSuite asks a fake server.
type ClientSuite struct{ serverSuite }

func TestClientSuite(t *testing.T) {
	suite.Run(t, new(ClientSuite))
}

var booking = Agent{Instructions: "Take bookings.", Tools: []llm.Tool{{Name: "book", Description: "Book it.", Parameters: map[string]any{"required": []any{"day"}}}}}

func (s *ClientSuite) TestAnAgentIsRegisteredOnceAndTheMomentGoesInTheBody() {
	for range 2 {
		got, err := s.client.Decide(context.Background(), booking, Turn{Speaker: Caller, Speaking: true, Said: "Votre traversée à 8 h"}, []int16{1, -1}, false)
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
		_, err := s.client.Decide(context.Background(), Agent{Instructions: instructions}, Turn{}, []int16{1}, false)
		s.Require().NoError(err)
	}
	s.Len(s.server.registered, 2)
}

// A restarted server has forgotten its agents, and may serve another model and policy.
func (s *ClientSuite) TestARestartedServerIsAskedAgain() {
	_, err := s.client.Info(context.Background())
	s.Require().NoError(err)
	_, err = s.client.Decide(context.Background(), booking, Turn{}, []int16{1}, false)
	s.Require().NoError(err)
	s.server.mu.Lock()
	s.server.known = map[string]bool{}
	s.server.mu.Unlock()
	_, err = s.client.Decide(context.Background(), booking, Turn{}, []int16{1}, false)
	s.Require().NoError(err)
	s.Len(s.server.registered, 2)
	_, err = s.client.Info(context.Background())
	s.Require().NoError(err)
	s.Equal(2, s.server.infos)
}

// An agent with more tools than the model has answers for is refused, and isn't offered
// again until it changes.
func (s *ClientSuite) TestARefusedAgentIsRefusedUntilItChanges() {
	s.server.refuse = http.StatusUnprocessableEntity
	for range 2 {
		_, err := s.client.Decide(context.Background(), booking, Turn{}, []int16{1}, false)
		s.ErrorIs(err, ErrRefused)
		s.ErrorContains(err, "22 tools")
	}
	s.Len(s.server.registered, 1)
	_, err := s.client.Decide(context.Background(), Agent{Instructions: booking.Instructions}, Turn{}, []int16{1}, false)
	s.Require().NoError(err)
}

func (s *ClientSuite) TestAServerWithoutAPolicyIsUnsupported() {
	s.server.refuse = http.StatusNotImplemented
	_, err := s.client.Decide(context.Background(), booking, Turn{}, []int16{1}, false)
	s.ErrorIs(err, ErrUnsupported)
	s.server.info = `{"template":"t","end_of_turn":{}}`
	_, err = s.client.Info(context.Background())
	s.ErrorIs(err, ErrUnsupported)
}

func (s *ClientSuite) TestTheTranscriptIsTheAnswersSecondLine() {
	s.server.words = "What time is the last ferry?"
	got, err := s.client.Decide(context.Background(), Agent{}, Turn{}, []int16{1}, true)
	s.Require().NoError(err)
	s.Equal("What time is the last ferry?", got.Transcript)
}

// The body carries no more than a prompt can show: the reply's last 800 characters, and the
// lines of the last six turns, a run of one side's lines being one turn.
func (s *ClientSuite) TestTheBodyCarriesWhatAPromptCanShow() {
	var lines []Line
	for i := range 9 {
		lines = append(lines, Line{Agent: i%2 == 1, Text: fmt.Sprintf("turn %d", i)})
	}
	lines = append(lines[:6], append([]Line{{Agent: true, Text: "turn 5, more"}}, lines[6:]...)...)
	_, err := s.client.Decide(context.Background(), Agent{}, Turn{Speaking: true, Said: "x" + strings.Repeat("é", maxSaid), Conversation: lines}, []int16{1}, false)
	s.Require().NoError(err)
	got := s.server.turns[0]
	s.Equal(strings.Repeat("é", maxSaid), got.Said)
	s.Equal(lines[3:], got.Conversation)
}

func TestTheBarIsRaisedOnATurnsFirstPause(t *testing.T) {
	var info Info
	require.NoError(t, json.Unmarshal([]byte(testInfo), &info))
	p := info.EndOfTurn
	require.InDelta(t, 0.70, p.Bar(true, false), 1e-9)
	require.InDelta(t, 0.20, p.Bar(false, false), 1e-9)
	require.InDelta(t, 0.794, p.Bar(true, true), 1e-3)
	require.InDelta(t, 0.292, p.Bar(false, true), 1e-3)
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
	s.hearer = NewHearer(s.client)
}

func (s *HearerSuite) answering(pEOT float64, ruling string) {
	s.server.decision = fmt.Sprintf(`{"answer":"wait","answer_probability":%v,"p_eot":%v,"options":[],"decide_ms":1,"ruling":%s}`, 1-pEOT, pEOT, ruling)
}

// p_eot of 0.6 doesn't end a turn at a pause's first check, where the bar is 0.7, and ends
// it at the next, where the bar is 0.2, as a ruling the server read past wait.
func (s *HearerSuite) TestATurnHasEndedWhenPEndReachesTheBar() {
	s.answering(0.6, `{"disposition":"respond","floor":"continue","tool":"book"}`)
	first, err := s.hearer.Check(context.Background(), harness.FlowTurn{Paused: 350 * time.Millisecond})
	s.Require().NoError(err)
	s.False(first.Ended)
	s.Equal(harness.Decided{Disposition: harness.Wait, Floor: harness.Continue, Again: 300 * time.Millisecond}, first.Ruling)

	later, err := s.hearer.Check(context.Background(), harness.FlowTurn{Waited: true, Paused: time.Second})
	s.Require().NoError(err)
	s.True(later.Ended)
	s.Equal(harness.Decided{Disposition: harness.Respond, Floor: harness.Continue, Tool: "book"}, later.Ruling)
}

func (s *HearerSuite) TestATurnHasEndedAtTheCapWhateverTheModelSays() {
	s.answering(0.01, `{"disposition":"respond","floor":"continue"}`)
	got, err := s.hearer.Check(context.Background(), harness.FlowTurn{Waited: true, Paused: 2500 * time.Millisecond})
	s.Require().NoError(err)
	s.Equal(harness.Respond, got.Ruling.Disposition)
}

// Over the agent there is no wait to weigh, and the ruling is the model's.
func (s *HearerSuite) TestOverTheAgentTheRulingIsTheModels() {
	s.server.decision = `{"answer":"stop","answer_probability":0.8,"options":[],"decide_ms":1,"ruling":{"disposition":"wait","floor":"stop"}}`
	got, err := s.hearer.Hear(context.Background(), harness.FlowTurn{Speaking: true, Unfinished: true, Reply: "Your crossing is"})
	s.Require().NoError(err)
	s.Equal(harness.Decided{Disposition: harness.Wait, Floor: harness.Stop}, got)
	s.Equal(Turn{Speaker: Caller, Speaking: true, Said: "Your crossing is", Unfinished: true}, s.server.turns[0])
}

// A participant's name never reaches the model, the agent's reply only matters over it, and
// the conversation is the caller's and the agent's words.
func (s *HearerSuite) TestATurnIsSaidInTheModelsTerms() {
	history := []llm.Message{{Role: llm.System, Content: "x"}, {Role: llm.User, Content: "hi"}, {Role: llm.Assistant, Content: "hello"}, {Role: llm.User, Content: " "}}
	_, err := s.hearer.Hear(context.Background(), harness.FlowTurn{Participant: "Alice", AnotherVoice: true, Reply: "hello", Unfinished: true, History: history})
	s.Require().NoError(err)
	s.Equal(Turn{Speaker: AnotherVoice, Conversation: []Line{{Text: "hi"}, {Agent: true, Text: "hello"}}}, s.server.turns[0])
}

// TranscriberSuite runs the transcriber on frames of voice and of quiet.
type TranscriberSuite struct {
	serverSuite
	transcriber *Transcriber
}

func TestTranscriberSuite(t *testing.T) {
	suite.Run(t, new(TranscriberSuite))
}

func (s *TranscriberSuite) SetupTest() {
	s.serverSuite.SetupTest()
	s.server.words = "Two tickets to Bilbao, please."
	s.transcriber = NewTranscriber(s.client, "", slog.New(slog.DiscardHandler))
	s.Require().NoError(s.transcriber.Start(context.Background()))
	s.T().Cleanup(func() { s.Require().NoError(s.transcriber.Close()) })
}

// say sends ms of 20 ms frames, voiced or quiet.
func (s *TranscriberSuite) say(ms int, voiced bool) {
	frame := make([]int16, SampleRate/50)
	if voiced {
		for i := range frame {
			frame[i] = int16(3000 * (i%2*2 - 1))
		}
	}
	for range ms / 20 {
		s.Require().NoError(s.transcriber.ProcessAudio(audio.PcmData{Samples: frame, SampleRate: SampleRate, Channels: 1}, stt.Participant{ID: "p"}))
	}
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

// An utterance's words are written at its pauses, the whole of it each time, and it ends
// at the policy's cap on the words last written.
func (s *TranscriberSuite) TestAnUtteranceIsWrittenAtItsPausesAndEndsAtTheCap() {
	s.say(2000, false)
	s.say(1000, true)
	s.say(180, false)
	s.Zero(s.server.requests())
	s.say(20, false)
	said := next[stt.Transcript](s)
	s.Equal(stt.ModeReplacement, said.Mode)
	s.Equal(int64(1), said.Utterance)
	s.Equal("Two tickets to Bilbao, please.", said.Text)
	s.InDelta(1500, said.AudioDurationMs, 1)

	s.say(500, true)
	s.say(200, false)
	more := next[stt.Transcript](s)
	s.Equal(int64(1), more.Utterance)
	s.InDelta(2200, more.AudioDurationMs, 1)

	s.say(2300, false)
	end := next[stt.Transcript](s)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal(int64(1), end.Utterance)
	s.Equal(2, s.server.requests())

	s.say(1000, true)
	s.say(200, false)
	s.Equal(int64(2), next[stt.Transcript](s).Utterance)
}

func (s *TranscriberSuite) TestAClickIsNotWritten() {
	s.say(100, true)
	s.say(400, false)
	s.say(1000, true)
	s.say(200, false)
	next[stt.Transcript](s)
	s.Equal(1, s.server.requests())
	s.InDelta(1500, float64(len(s.server.audio[0]))/2*1000/SampleRate, 1)
}

// A server that can't write the words hands the call to the next transcriber.
func (s *TranscriberSuite) TestAServerThatFailsIsFatal() {
	s.server.mu.Lock()
	s.server.fail = true
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(200, false)
	s.True(next[stt.Error](s).Fatal)
}

func TestRingKeepsTheLastWindow(t *testing.T) {
	var r Ring
	require.Empty(t, r.Last())
	r.Write([]int16{1, 2, 3})
	require.Equal(t, []int16{1, 2, 3}, r.Last())

	frame := make([]int16, 320)
	next := int16(0)
	for range 2000 { // 40 s of 20 ms frames
		for i := range frame {
			frame[i] = next
			next++
		}
		r.Write(frame)
	}
	got := r.Last()
	require.Len(t, got, windowSamples)
	for i := 1; i < len(got); i++ {
		require.Equal(t, got[i-1]+1, got[i])
	}
	require.Equal(t, next-1, got[len(got)-1])

	long := make([]int16, windowSamples+5)
	for i := range long {
		long[i] = int16(i)
	}
	r.Write(long)
	got = r.Last()
	require.Len(t, got, windowSamples)
	require.Equal(t, int16(5), got[0])
}

func TestRingWriteAllocatesNothingOnceItHasItsBuffer(t *testing.T) {
	var r Ring
	frame := make([]int16, 320)
	r.Write(frame)
	require.Zero(t, testing.AllocsPerRun(100, func() { r.Write(frame) }))
}
