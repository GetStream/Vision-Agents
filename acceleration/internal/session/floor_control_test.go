package session

import (
	"bytes"
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
// server, and routerDefault as what an agent that chose no floor control gets. Each listener
// has a transcriber of its own, handed to the test as it is opened, and each agent a voice of
// its own, so that sessions running at once each hear and speak only on their own call.
func (s *SessionSuite) floorManager(client *audiofloor.Client, routerDefault string) (*Manager, chan *pausingEars) {
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
		AudioFloor: client, FloorControlDefault: routerDefault, Logger: slog.New(slog.DiscardHandler),
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

// Two sessions on one router at once, one whose floor chorus-asr controls and one with the
// default flow controller, each run by its own floor control: the first's turns are put to the
// server and its transcriber asks about its caller's pauses, and nothing of the other's
// reaches the server, as on a router with no chorus-asr, its turns answered by its flow
// controller all the same.
func (s *SessionSuite) TestEachSessionOnARouterRunsByItsOwnFloorControl() {
	server, client := newChorusASR(s)
	manager, opened := s.floorManager(client, "")

	acting := s.floorSession(manager, "You sell ferry tickets.", store.FloorControl{Model: store.FloorControlChorusASR, MaxWaitMs: 1400})
	unset := s.floorSession(manager, "You book clinic visits.", store.FloorControl{})

	s.callerAsks(acting, opened, "Two tickets to Bilbao, please.")
	s.callerAsks(unset, opened, "Can I see a doctor on Friday?")
	s.callerAsks(acting, opened, "And a cabin for the night.")

	s.Equal(2, server.asked("You sell ferry tickets."), "each of the chorus-asr session's turns was put to chorus-asr")
	s.True(acting.ears.asked.Load(), "the chorus-asr session's transcriber asks about its caller's pauses")
	s.False(server.toldAbout("You book clinic visits."), "nothing of the session with none chosen reached chorus-asr")
	s.False(unset.ears.asked.Load(), "the other's transcriber asks on behalf of no agent")
}

// A session that chose chorus-asr on a router with no chorus-asr server, as a config written
// on another router may have, asks nothing, and is answered by its flow controller.
func (s *SessionSuite) TestAFloorControlWithNoServerIsTheFlowControllers() {
	manager, opened := s.floorManager(nil, "")
	acting := s.floorSession(manager, "You sell ferry tickets.", store.FloorControl{Model: store.FloorControlChorusASR})
	s.callerAsks(acting, opened, "Two tickets to Bilbao, please.")
	s.False(acting.ears.asked.Load())
}

// What controls a session's floor is its agent's own choice, whatever the router's default,
// and the router's default where it chose none: chorus-asr at the server's default point on a
// router with a server, and the flow controller on one without, as with no default. The
// opt-out is the flow controller everywhere, and asks nothing; an empty choice is none.
func (s *SessionSuite) TestTheFloorIsTheAgentsChoiceElseTheRoutersDefault() {
	none, flow := store.FloorControl{}, store.FloorControl{Model: store.FloorControlFlowController}
	chorus, waiting := store.FloorControl{Model: store.FloorControlChorusASR}, store.FloorControl{Model: store.FloorControlChorusASR, MaxWaitMs: 1400}
	for _, row := range []struct {
		name          string
		chosen        store.FloorControl
		routerDefault string
		served        bool
		ruling        store.FloorControl
	}{
		{"none, no default", none, "", true, none},
		{"none, no default, no server", none, "", false, none},
		{"none, flow-controller default", none, store.FloorControlFlowController, true, none},
		{"none, chorus-asr default", none, store.FloorControlChorusASR, true, chorus},
		{"none, chorus-asr default, no server", none, store.FloorControlChorusASR, false, none},
		{"opt-out, no default", flow, "", true, none},
		{"opt-out, chorus-asr default", flow, store.FloorControlChorusASR, true, none},
		{"opt-out, chorus-asr default, no server", flow, store.FloorControlChorusASR, false, none},
		{"chorus-asr, no default", chorus, "", true, chorus},
		{"chorus-asr, chorus-asr default", chorus, store.FloorControlChorusASR, true, chorus},
		{"chorus-asr with a wait, chorus-asr default", waiting, store.FloorControlChorusASR, true, waiting},
		{"chorus-asr with a wait, no default", waiting, "", true, waiting},
		{"chorus-asr, no server", chorus, store.FloorControlChorusASR, false, chorus},
	} {
		s.Equal(row.ruling, FloorControlOf(row.chosen, row.routerDefault, row.served), row.name)
	}
	s.Equal(audiofloor.Off, FloorModeOf(flow), "the opt-out asks nothing")
	s.Equal(audiofloor.Off, FloorModeOf(none))
	s.Equal(audiofloor.Act, FloorModeOf(FloorControlOf(none, store.FloorControlChorusASR, true)))
	s.Equal(audiofloor.EndOfTurn{}, EndOfTurnOf(FloorControlOf(none, store.FloorControlChorusASR, true)), "the default is the server's default point")
}

// A router that defaults every agent to chorus-asr puts the turns of a session that chose
// nothing to the server, at the server's default point, beside one that chose chorus-asr with
// a wait of its own; one that opted out is the flow controller's, and nothing of it reaches
// the server.
func (s *SessionSuite) TestTheRoutersDefaultRulesEveryFloorThatChoseNone() {
	server, client := newChorusASR(s)
	manager, opened := s.floorManager(client, store.FloorControlChorusASR)

	unset := s.floorSession(manager, "You book clinic visits.", store.FloorControl{})
	optedOut := s.floorSession(manager, "You sell garden tools.", store.FloorControl{Model: store.FloorControlFlowController})
	acting := s.floorSession(manager, "You sell ferry tickets.", store.FloorControl{Model: store.FloorControlChorusASR, MaxWaitMs: 1400})

	s.callerAsks(unset, opened, "Can I see a doctor on Friday?")
	s.callerAsks(optedOut, opened, "Do you have a spade?")
	s.callerAsks(acting, opened, "Two tickets to Bilbao, please.")

	s.Equal(1, server.asked("You book clinic visits."), "the session that chose nothing is the router's default, chorus-asr")
	s.True(unset.ears.asked.Load(), "and its transcriber asks about its caller's pauses")
	s.Equal(1, server.asked("You sell ferry tickets."), "the session that chose chorus-asr is as it was")
	s.False(server.toldAbout("You sell garden tools."), "nothing of the session that opted out reached chorus-asr")
	s.False(optedOut.ears.asked.Load(), "its transcriber asks on behalf of no agent")
}

// The default asks nothing of a router with no chorus-asr server: the router warns once as
// the manager is built, not per session, and a session that chose nothing is the flow
// controller's.
func (s *SessionSuite) TestTheRoutersDefaultAsksNothingWithoutAServer() {
	manager, opened := s.floorManager(nil, store.FloorControlChorusASR)
	unset := s.floorSession(manager, "You book clinic visits.", store.FloorControl{})
	s.callerAsks(unset, opened, "Can I see a doctor on Friday?")
	s.False(unset.ears.asked.Load())

	var said bytes.Buffer
	warned, err := NewManager(ManagerOptions{
		LLM: s.manager.options.LLM, Edge: s.manager.options.Edge,
		FloorControlDefault: store.FloorControlChorusASR, Logger: slog.New(slog.NewTextHandler(&said, nil)),
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = warned.Shutdown() })
	s.Equal(1, strings.Count(said.String(), "ROUTER_FLOOR_CONTROL_DEFAULT"), said.String())
	s.Contains(said.String(), "ROUTER_CHORUS_ASR_URL")

	said.Reset()
	quiet, err := NewManager(ManagerOptions{
		LLM: s.manager.options.LLM, Edge: s.manager.options.Edge,
		FloorControlDefault: store.FloorControlChorusASR, AudioFloor: audiofloor.New("http://floor:8080", audiofloor.Opus, nil),
		Logger: slog.New(slog.NewTextHandler(&said, nil)),
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = quiet.Shutdown() })
	s.NotContains(said.String(), "ROUTER_FLOOR_CONTROL_DEFAULT", "with a server there is nothing to warn of")
}
