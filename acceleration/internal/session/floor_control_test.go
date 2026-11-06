package session

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// chorusASRPolicy is a chorus-asr server's policy, its default point ending a turn at a p_eot
// of 0.995 from the first check.
const chorusASRPolicy = `{"template":"t","end_of_turn":{"first_check_ms":200,"first_pause_logit":0,"interval_ms":300,"cap_ms":2500,` +
	`"default":"safe","presets":{"safe":1100},"frontier":[` +
	`{"max_wait_ms":1100,"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}],"checks_in_a_row":1,` +
	`"measured":{"cut_ins":0.0454,"answer_ms_median":1100,"answer_ms_p90":1100,"dictation_pauses_cut":0.1257,"slow_counts_cut":0,"slow_counts":9,"dictation_guard":true}}]}}`

// chorusASR is a chorus-asr server that rules every turn finished, to answer, and keeps what
// each agent put to it was told.
type chorusASR struct {
	mu sync.Mutex
	// told is each agent registered's instructions, and decided how many turns each was
	// asked about, by the id the server gave it.
	told    map[string]string
	decided map[string]int
}

func newChorusASR(s *SessionSuite) (*chorusASR, *audiofloor.Client) {
	server := &chorusASR{told: map[string]string{}, decided: map[string]int{}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		server.mu.Lock()
		defer server.mu.Unlock()
		switch r.URL.Path {
		case "/v1/info":
			_, _ = w.Write([]byte(chorusASRPolicy))
		case "/v1/agents":
			var told struct {
				Instructions string `json:"instructions"`
			}
			_ = json.NewDecoder(r.Body).Decode(&told)
			id := "agent-" + string(rune('a'+len(server.told)))
			server.told[id] = told.Instructions
			_, _ = w.Write([]byte(`{"agent":"` + id + `"}`))
		case "/v1/decide":
			server.decided[r.URL.Query().Get("agent")]++
			_, _ = w.Write([]byte(`{"answer":"respond","answer_probability":0.99,"p_eot":0.9999,"options":[],"decide_ms":1,` +
				`"ruling":{"disposition":"respond","floor":"continue"}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	s.T().Cleanup(srv.Close)
	return server, audiofloor.New(srv.URL, audiofloor.PCM, srv.Client())
}

// asked is how many turns the server was asked about for an agent told instructions.
func (c *chorusASR) asked(instructions string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	var turns int
	for id, told := range c.told {
		if strings.Contains(told, instructions) {
			turns += c.decided[id]
		}
	}
	return turns
}

// toldAbout says whether any agent told instructions was put to the server.
func (c *chorusASR) toldAbout(instructions string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, told := range c.told {
		if strings.Contains(told, instructions) {
			return true
		}
	}
	return false
}

// floorManager is a manager over the suite's stub models with client as its chorus-asr
// server. Each listener has a
// transcriber of its own, handed to the test as it is opened, and each agent a voice of its
// own, so that sessions running at once each hear and speak only on their own call.
func (s *SessionSuite) floorManager(client *audiofloor.Client) (*Manager, chan *pausingEars) {
	s.manages()
	opened := make(chan *pausingEars, 8)
	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) {
		ears := &pausingEars{stubSTT: &stubSTT{emitter: stt.NewEmitter(64)}}
		opened <- ears
		return ears, nil
	})
	transcriber, err := sttrouter.New(sttrouter.Options{Config: stubConfig(), Registry: transcription, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)
	voices := ttsrouter.NewRegistry()
	voices.Register("stub", func(routing.Spec) (tts.TTS, error) { return &stubTTS{emitter: tts.NewEmitter(64)}, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{Config: stubConfig(), Registry: voices, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)
	manager, err := NewManager(ManagerOptions{
		LLM: s.manager.options.LLM, STT: transcriber, TTS: speaker, Edge: s.manager.options.Edge,
		AudioFloor: client, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = manager.Shutdown() })
	return manager, opened
}

// pausingEars stands in for chorus-asr's transcriber, which an agent acting on chorus-asr has
// ask about its callers' pauses (audiofloor.Pauses), and says whether one did.
type pausingEars struct {
	*stubSTT
	asked atomic.Bool
}

func (p *pausingEars) AskOnPauses(audiofloor.Asker) { p.asked.Store(true) }

// floorCall is a voice session on a call of its own, and the listener its agent opened for
// the caller once they spoke.
type floorCall struct {
	session *Session
	edge    *quietEdge
	ears    *pausingEars
}

// floorSession opens a voice session told instructions, choosing floor control.
func (s *SessionSuite) floorSession(manager *Manager, instructions string, chosen store.FloorControl) *floorCall {
	edges := len(s.edges)
	created, err := manager.Create(s.ctx, Spec{
		CallID: "call-" + instructions, CustomerID: "acme", Instructions: instructions, FloorControl: chosen,
		LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency",
	})
	s.Require().NoError(err)
	s.Require().Eventually(func() bool { return len(s.edges) > edges }, settleFor, 10*time.Millisecond)
	return &floorCall{session: created, edge: s.edges[len(s.edges)-1]}
}

// callerAsks has the caller say words on the call, heard by the listener the agent opens for
// them, and waits for the session to answer.
func (s *SessionSuite) callerAsks(call *floorCall, opened chan *pausingEars, words string) {
	events, detach := call.session.Watch()
	defer detach()
	caller := stt.Participant{ID: "caller", Name: "Caller"}
	call.edge.inbound <- agent.InboundAudio{Participant: caller, Audio: audio.PcmData{Samples: make([]int16, 3200), SampleRate: 16_000, Channels: 1}}
	if call.ears == nil {
		select {
		case call.ears = <-opened:
		case <-time.After(settleFor):
			s.FailNow("the agent opened no listener for the caller")
		}
	}
	call.ears.emitter.Send(stt.Transcript{Participant: caller, Mode: stt.ModeFinal, Text: words, Language: "en", Confidence: 1})
	s.NotEmpty(awaitReply(events), "the caller was never answered")
	s.Require().Eventually(func() bool { return !call.session.Busy() }, settleFor, 10*time.Millisecond)
}

func chorusASRAt(mode string) store.FloorControl {
	return store.FloorControl{Model: store.FloorModelChorusASR, Mode: mode}
}

// Two sessions on one router at once, one acting on chorus-asr and one with it off, each run
// by its own floor control: the acting one's turns are put to the server and its transcriber
// asks about its caller's pauses, and nothing of the other's reaches the server, as on a
// router with no chorus-asr, its turns answered by its flow controller all the same.
func (s *SessionSuite) TestEachSessionOnARouterRunsByItsOwnFloorControl() {
	server, client := newChorusASR(s)
	manager, opened := s.floorManager(client)

	acting := s.floorSession(manager, "You sell ferry tickets.", chorusASRAt(store.FloorModeAct))
	off := s.floorSession(manager, "You book clinic visits.", chorusASRAt(store.FloorModeOff))

	s.callerAsks(acting, opened, "Two tickets to Bilbao, please.")
	s.callerAsks(off, opened, "Can I see a doctor on Friday?")
	s.callerAsks(acting, opened, "And a cabin for the night.")

	s.Equal(2, server.asked("You sell ferry tickets."), "each of the acting session's turns was put to chorus-asr")
	s.True(acting.ears.asked.Load(), "the acting session's transcriber asks about its caller's pauses")
	s.False(server.toldAbout("You book clinic visits."), "nothing of the session with it off reached chorus-asr")
	s.False(off.ears.asked.Load(), "the other's transcriber asks on behalf of no agent")
}

// A session whose agent chose no floor control asks chorus-asr nothing on a router that has
// it, as before there was floor control: an agent opts in for itself.
func (s *SessionSuite) TestASessionThatChoseNoFloorControlIsOff() {
	server, client := newChorusASR(s)
	manager, opened := s.floorManager(client)

	unset := s.floorSession(manager, "You sell ferry tickets.", store.FloorControl{})
	shadow := s.floorSession(manager, "You book clinic visits.", chorusASRAt(store.FloorModeShadow))
	s.callerAsks(unset, opened, "Two tickets to Bilbao, please.")
	s.callerAsks(shadow, opened, "Can I see a doctor on Friday?")

	s.False(server.toldAbout("You sell ferry tickets."), "none chosen is off")
	s.False(unset.ears.asked.Load())
	s.Eventually(func() bool { return server.asked("You book clinic visits.") == 1 }, settleFor, 10*time.Millisecond,
		"one that opts in to shadow is asked about its turn beside its flow controller")
	s.False(shadow.ears.asked.Load(), "and acts on nothing: its transcriber asks on behalf of no agent")
}

// A session that chose to act on chorus-asr on a router with no chorus-asr server, as a config
// written on another router may have, asks nothing, and is answered by its flow controller.
func (s *SessionSuite) TestAFloorControlWithNoServerIsTheFlowControllers() {
	manager, opened := s.floorManager(nil)
	acting := s.floorSession(manager, "You sell ferry tickets.", chorusASRAt(store.FloorModeAct))
	s.callerAsks(acting, opened, "Two tickets to Bilbao, please.")
	s.False(acting.ears.asked.Load())
}
