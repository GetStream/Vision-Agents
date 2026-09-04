package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// settleFor is how long a test waits for something crossing a socket to arrive.
const settleFor = 5 * time.Second

// silentEdge is a call with no network in it.
type silentEdge struct{ inbound chan agent.InboundAudio }

func (e *silentEdge) Join(context.Context) error       { return nil }
func (e *silentEdge) Audio() <-chan agent.InboundAudio { return e.inbound }
func (e *silentEdge) PublishAudio(audio.PcmData) error { return nil }

func (e *silentEdge) Leave() error {
	close(e.inbound)
	return nil
}

// scriptedLLM answers with a fixed reply and, on the first turn, whatever tool the test
// wants the model to reach for.
type scriptedLLM struct {
	mu    sync.Mutex
	turns int
	reply string
	calls []llm.ToolCall
}

func (s *scriptedLLM) Start(context.Context) error { return nil }

func (s *scriptedLLM) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	s.mu.Lock()
	s.turns++
	first := s.turns == 1
	reply := s.reply
	var calls []llm.ToolCall
	if first {
		calls = append([]llm.ToolCall(nil), s.calls...)
	}
	s.mu.Unlock()

	script := llmtest.New(llm.StreamOptions{
		ResponseID: params.ID,
		Provider:   s.Provider(),
		Model:      s.Model(),
	})
	script.OutputText(reply)
	if len(calls) > 0 {
		script.ToolCalls(calls...)
	}
	script.Done()
	return script.Stream(), nil
}

func (s *scriptedLLM) Provider() string               { return "stub" }
func (s *scriptedLLM) Model() string                  { return "stub-llm" }
func (s *scriptedLLM) Capabilities() llm.Capabilities { return llm.Capabilities{} }
func (s *scriptedLLM) Close() error                   { return nil }

// quietSTT hears nothing, since these tests drive the session over HTTP rather than
// through a microphone.
type quietSTT struct{ emitter *stt.Emitter }

func (s *quietSTT) Start(context.Context) error                     { return nil }
func (s *quietSTT) ProcessAudio(stt.PcmData, stt.Participant) error { return nil }
func (s *quietSTT) Events() <-chan stt.Event                        { return s.emitter.Events() }
func (s *quietSTT) Provider() string                                { return "stub" }
func (s *quietSTT) Model() string                                   { return "stub-stt" }

func (s *quietSTT) Close() error {
	s.emitter.Close()
	return nil
}

// recordingTTS keeps what it was asked to say.
type recordingTTS struct {
	emitter *tts.Emitter

	mu   sync.Mutex
	said []string
}

func (s *recordingTTS) Start(context.Context) error { return nil }

func (s *recordingTTS) Synthesize(request tts.Request) error {
	s.mu.Lock()
	s.said = append(s.said, request.Text)
	s.mu.Unlock()

	if request.Final {
		s.emitter.Send(tts.SynthesisComplete{SynthesisID: request.ID})
	}
	return nil
}

func (s *recordingTTS) Interrupt() error         { return nil }
func (s *recordingTTS) Events() <-chan tts.Event { return s.emitter.Events() }
func (s *recordingTTS) Provider() string         { return "stub" }
func (s *recordingTTS) Model() string            { return "stub-tts" }
func (s *recordingTTS) Streaming() bool          { return false }
func (s *recordingTTS) Performs() bool           { return false }
func (s *recordingTTS) Prompt() string           { return "" }

func (s *recordingTTS) Close() error {
	s.emitter.Close()
	return nil
}

func (s *recordingTTS) spoken() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.said...)
}

func routableConfig() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{
			Provider:  "stub",
			Model:     "stub-model",
			Languages: []string{"en"},
			Realtime:  true,
		}},
		Aliases: map[string]routing.Alias{
			"en-low-latency": {Languages: []string{"en"}, RequireRealtime: true},
		},
	}
}

type SessionAPISuite struct {
	suite.Suite

	server *httptest.Server
	model  *scriptedLLM
	voice  *recordingTTS
}

func TestSessionAPISuite(t *testing.T) {
	suite.Run(t, new(SessionAPISuite))
}

func (s *SessionAPISuite) SetupTest() {
	logger := slog.New(slog.DiscardHandler)

	ears := &quietSTT{emitter: stt.NewEmitter(64)}
	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) { return ears, nil })
	transcriber, err := sttrouter.New(sttrouter.Options{
		Config: routableConfig(), Registry: transcription, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)

	// The agent opens a voice model and a flow controller, each needing its own emitter:
	// two sessions sharing one channel would each consume the other's events.
	s.model = &scriptedLLM{reply: "Hello."}
	var opened int
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		defer func() { opened++ }()
		if opened == 0 {
			return s.model, nil
		}
		return &scriptedLLM{}, nil
	})
	reasoner, err := llmrouter.New(llmrouter.Options{
		Config: routableConfig(), Registry: reasoning, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)

	s.voice = &recordingTTS{emitter: tts.NewEmitter(64)}
	speech := ttsrouter.NewRegistry()
	speech.Register("stub", func(routing.Spec) (tts.TTS, error) { return s.voice, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{
		Config: routableConfig(), Registry: speech, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	sessions, err := session.NewManager(session.ManagerOptions{
		LLM:    reasoner,
		STT:    transcriber,
		TTS:    speaker,
		Logger: logger,
		Edge: func(session.Spec, *slog.Logger) (agent.Edge, error) {
			return &silentEdge{inbound: make(chan agent.InboundAudio, 4)}, nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { sessions.Shutdown() })

	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{
			routing.STT: transcriber,
			routing.TTS: speaker,
			routing.LLM: reasoner,
		},
		Sessions: sessions,
		Streams:  &Streams{STT: transcriber, TTS: speaker, LLM: reasoner},
		Logger:   logger,
	})
	s.Require().NoError(err)

	s.server = httptest.NewServer(server.Handler())
	s.T().Cleanup(s.server.Close)
}

// send issues a request against the test server, with the customer header unless it is
// empty.
func (s *SessionAPISuite) send(method, path, customerID string, body any) *http.Response {
	var payload *bytes.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		s.Require().NoError(err)
		payload = bytes.NewReader(encoded)
	} else {
		payload = bytes.NewReader(nil)
	}

	request, err := http.NewRequest(method, s.server.URL+path, payload)
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	if customerID != "" {
		request.Header.Set(CustomerHeader, customerID)
	}

	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	s.T().Cleanup(func() { response.Body.Close() })
	return response
}

func (s *SessionAPISuite) decodeBody(response *http.Response, target any) {
	s.Require().NoError(json.NewDecoder(response.Body).Decode(target))
}

// creates joins a call and returns the session as the API described it. The targets are
// filled in because the stub config declares one shortcut rather than the whole catalogue.
func (s *SessionAPISuite) creates(request CreateSessionRequest) Session {
	target := "en-low-latency"
	if request.Llm == nil {
		request.Llm = &target
	}
	if request.Stt == nil {
		request.Stt = &target
	}
	if request.Tts == nil {
		request.Tts = &target
	}

	response := s.send(http.MethodPost, "/v1/agents/sessions", "acme", request)
	s.Require().Equal(http.StatusCreated, response.StatusCode)

	var created Session
	s.decodeBody(response, &created)
	return created
}

// callID is the call to join, which the wire carries as optional because a text session
// joins none.
func callID(id string) *string { return &id }

// watches opens the events socket for a session.
func (s *SessionAPISuite) watches(id, customerID string) *websocket.Conn {
	address := "ws" + strings.TrimPrefix(s.server.URL, "http") + "/v1/agents/sessions/" + id + "/events"
	header := http.Header{}
	if customerID != "" {
		header.Set(CustomerHeader, customerID)
	}

	connection, _, err := websocket.DefaultDialer.Dial(address, header)
	s.Require().NoError(err)
	s.T().Cleanup(func() { connection.Close() })
	return connection
}

// await reads frames until one of the wanted type arrives.
func (s *SessionAPISuite) await(connection *websocket.Conn, wanted string) map[string]any {
	connection.SetReadDeadline(time.Now().Add(settleFor))
	for {
		var received map[string]any
		if err := connection.ReadJSON(&received); err != nil {
			s.Require().FailNow("the socket closed before " + wanted + " arrived: " + err.Error())
		}
		if received["type"] == wanted {
			return received
		}
	}
}

func (s *SessionAPISuite) TestCreatingASessionJoinsTheCallAndDescribesIt() {
	created := s.creates(CreateSessionRequest{CallId: callID("call-7")})

	s.NotEmpty(created.Id)
	s.Equal("call-7", created.CallId)
	s.Equal("agent", created.CallType, "a call type defaults rather than being required")
	s.Equal("call-7", created.AgentId)
	s.Equal(SessionState("live"), created.State)
	s.Require().NotNil(created.Llm)
	s.Equal("stub/stub-model", *created.Llm)
	s.Require().NotNil(created.Tts)
	s.Equal("stub/stub-model", *created.Tts)
	s.Nil(created.Stt, "transcription starts when somebody is heard, not on joining")
}

func (s *SessionAPISuite) TestASessionNeedsACallToJoin() {
	response := s.send(http.MethodPost, "/v1/agents/sessions", "acme", CreateSessionRequest{})

	s.Equal(http.StatusBadRequest, response.StatusCode)

	var failure Error
	s.decodeBody(response, &failure)
	s.Contains(failure.Error, "call id is required")
}

func (s *SessionAPISuite) TestATextSessionNeedsNoCallAndNoSpeechTargets() {
	target := "en-low-latency"
	text := true

	response := s.send(http.MethodPost, "/v1/agents/sessions", "acme",
		CreateSessionRequest{Text: &text, Llm: &target})

	s.Require().Equal(http.StatusCreated, response.StatusCode)
	var created Session
	s.decodeBody(response, &created)
	s.Empty(created.CallId, "a conversation in writing joins no call")
	s.NotEmpty(created.AgentId, "it is still keyed by an agent id of its own")
	s.Require().NotNil(created.Text)
	s.True(*created.Text)
	s.Nil(created.Tts, "nothing speaks for it")
}

func (s *SessionAPISuite) TestSessionsRequireTheCustomerHeader() {
	response := s.send(http.MethodPost, "/v1/agents/sessions", "", CreateSessionRequest{CallId: callID("call-1")})

	s.Equal(http.StatusUnauthorized, response.StatusCode)
}

func (s *SessionAPISuite) TestAnotherCustomersSessionIsNotFound() {
	// Reporting it as forbidden would confirm the id was real to somebody who guessed it.
	created := s.creates(CreateSessionRequest{CallId: callID("call-1")})

	response := s.send(http.MethodGet, "/v1/agents/sessions/"+created.Id, "other", nil)

	s.Equal(http.StatusNotFound, response.StatusCode)
	s.Empty(s.listed("other"))
}

func (s *SessionAPISuite) TestASessionIsListedForTheCustomerRunningIt() {
	created := s.creates(CreateSessionRequest{CallId: callID("call-1")})

	listed := s.listed("acme")

	s.Require().Len(listed, 1)
	s.Equal(created.Id, listed[0].Id)
}

func (s *SessionAPISuite) TestClosingASessionEndsIt() {
	created := s.creates(CreateSessionRequest{CallId: callID("call-1")})

	response := s.send(http.MethodDelete, "/v1/agents/sessions/"+created.Id, "acme", nil)

	s.Equal(http.StatusNoContent, response.StatusCode)
	s.Empty(s.listed("acme"))

	again := s.send(http.MethodDelete, "/v1/agents/sessions/"+created.Id, "acme", nil)
	s.Equal(http.StatusNotFound, again.StatusCode, "a session was closed twice")
}

func (s *SessionAPISuite) TestSayingSomethingSpeaksItWithoutTheModel() {
	created := s.creates(CreateSessionRequest{CallId: callID("call-1")})

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/say", "acme",
		SayRequest{Text: "Hi, I'm listening."})

	s.Equal(http.StatusNoContent, response.StatusCode)
	s.Require().Eventually(func() bool {
		return len(s.voice.spoken()) > 0
	}, settleFor, 5*time.Millisecond, "nothing was said")
	s.Contains(s.voice.spoken(), "Hi, I'm listening.")
}

func (s *SessionAPISuite) TestSayingNothingIsRefused() {
	created := s.creates(CreateSessionRequest{CallId: callID("call-1")})

	response := s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/say", "acme",
		SayRequest{Text: ""})

	s.Equal(http.StatusBadRequest, response.StatusCode)
}

func (s *SessionAPISuite) TestTheEventsSocketCarriesWhatTheConversationDid() {
	created := s.creates(CreateSessionRequest{CallId: callID("call-1")})
	connection := s.watches(created.Id, "acme")

	s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/respond", "acme",
		SayRequest{Text: "hello"})

	answered := s.await(connection, "responded")

	s.Equal("Hello.", answered["text"])
	s.NotEmpty(answered["turn_id"])
}

func (s *SessionAPISuite) TestTheSocketAsksTheCallerToRunItsOwnToolsAndUsesTheAnswer() {
	s.model.calls = []llm.ToolCall{{
		ID: "call-1", Name: "lookup_order", Arguments: `{"order":"12"}`,
	}}
	parameters := map[string]any{"type": "object"}
	created := s.creates(CreateSessionRequest{
		CallId: callID("call-1"),
		Tools: &[]SessionTool{{
			Name:        "lookup_order",
			Description: "find an order by its number",
			Parameters:  &parameters,
		}},
	})
	connection := s.watches(created.Id, "acme")

	s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/respond", "acme",
		SayRequest{Text: "where is my order"})

	asked := s.await(connection, "tool_call")
	s.Equal("lookup_order", asked["name"])
	s.Equal(`{"order":"12"}`, asked["arguments"])

	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type":         "tool_result",
		"tool_call_id": asked["id"],
		"output":       "it ships tomorrow",
	}))

	ran := s.await(connection, "tool_ran")
	s.Equal("lookup_order", ran["tool"])
	s.Equal("it ships tomorrow", ran["result"])
	s.Empty(ran["error"])
}

func (s *SessionAPISuite) TestAToolTheCallerCouldNotRunIsToldToTheModelInWords() {
	s.model.calls = []llm.ToolCall{{ID: "call-1", Name: "lookup_order", Arguments: "{}"}}
	created := s.creates(CreateSessionRequest{
		CallId: callID("call-1"),
		Tools:  &[]SessionTool{{Name: "lookup_order", Description: "find an order"}},
	})
	connection := s.watches(created.Id, "acme")

	s.send(http.MethodPost, "/v1/agents/sessions/"+created.Id+"/respond", "acme",
		SayRequest{Text: "where is my order"})

	asked := s.await(connection, "tool_call")
	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type":         "tool_result",
		"tool_call_id": asked["id"],
		"error":        "the orders service is down",
	}))

	ran := s.await(connection, "tool_ran")
	s.Contains(ran["error"], "the orders service is down")
	s.Contains(ran["result"], "did not work",
		"the model has to be told in words it can repeat to the caller")
}

func (s *SessionAPISuite) TestTheSocketCanSpeakAndEndTheCall() {
	created := s.creates(CreateSessionRequest{CallId: callID("call-1")})
	connection := s.watches(created.Id, "acme")

	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type": "say", "text": "one moment",
	}))
	s.Require().Eventually(func() bool {
		for _, said := range s.voice.spoken() {
			if said == "one moment" {
				return true
			}
		}
		return false
	}, settleFor, 5*time.Millisecond, "the socket could not make the agent talk")

	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "close"}))

	s.Require().Eventually(func() bool {
		return len(s.listed("acme")) == 0 ||
			s.listed("acme")[0].State == SessionState("ended")
	}, settleFor, 5*time.Millisecond, "the call outlived the socket that closed it")
}

func (s *SessionAPISuite) TestAnotherCustomerCannotWatchASession() {
	created := s.creates(CreateSessionRequest{CallId: callID("call-1")})
	address := "ws" + strings.TrimPrefix(s.server.URL, "http") +
		"/v1/agents/sessions/" + created.Id + "/events"

	header := http.Header{}
	header.Set(CustomerHeader, "other")
	_, response, err := websocket.DefaultDialer.Dial(address, header)

	s.Require().Error(err)
	s.Require().NotNil(response)
	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *SessionAPISuite) TestAModalityStreamNeedsATargetBeforeItRoutesAnything() {
	address := "ws" + strings.TrimPrefix(s.server.URL, "http") + "/v1/tts/stream"
	header := http.Header{}
	header.Set(CustomerHeader, "acme")

	connection, _, err := websocket.DefaultDialer.Dial(address, header)
	s.Require().NoError(err)
	defer connection.Close()

	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "start"}))

	connection.SetReadDeadline(time.Now().Add(settleFor))
	var failure map[string]any
	s.Require().NoError(connection.ReadJSON(&failure))
	s.Equal("error", failure["type"])
	s.Contains(failure["error"], "needs a target")
}

func (s *SessionAPISuite) TestAModalityStreamSpeaksWhatItIsSent() {
	address := "ws" + strings.TrimPrefix(s.server.URL, "http") + "/v1/tts/stream"
	header := http.Header{}
	header.Set(CustomerHeader, "acme")

	connection, _, err := websocket.DefaultDialer.Dial(address, header)
	s.Require().NoError(err)
	defer connection.Close()

	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type": "start", "target": "en-low-latency",
	}))

	connection.SetReadDeadline(time.Now().Add(settleFor))
	var started map[string]any
	s.Require().NoError(connection.ReadJSON(&started))
	s.Equal("started", started["type"])
	s.Equal("stub", started["provider"])

	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type": "speak", "id": "utterance-1", "text": "hello there",
	}))

	connection.SetReadDeadline(time.Now().Add(settleFor))
	var settled map[string]any
	s.Require().NoError(connection.ReadJSON(&settled))
	s.Equal("synthesis_complete", settled["type"])
	s.Contains(s.voice.spoken(), "hello there")
}

func (s *SessionAPISuite) TestAnUnroutedModalityHasNoStream() {
	address := "ws" + strings.TrimPrefix(s.server.URL, "http") + "/v1/memory/stream"
	header := http.Header{}
	header.Set(CustomerHeader, "acme")

	_, response, err := websocket.DefaultDialer.Dial(address, header)

	s.Require().Error(err)
	s.Require().NotNil(response)
	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *SessionAPISuite) TestAConfigFillsInWhatTheRequestLeftOut() {
	config := &store.AgentConfig{
		ID:                 "config-1",
		STT:                "config-stt",
		TTS:                "config-tts",
		Voice:              "config-voice",
		LLM:                "config-llm",
		Subagent:           "config-subagent",
		Instructions:       "be brief",
		Greeting:           "hello there",
		Skills:             []string{"refund"},
		KnowledgeNamespace: "handbook",
		Tags:               map[string]string{"agent": "support"},
	}

	spec := specOf(CreateSessionRequest{CallId: callID("call-1")}, "acme", config)

	s.Equal("config-1", spec.ConfigID)
	s.Equal("config-stt", spec.STTTarget)
	s.Equal("config-llm", spec.LLMTarget)
	s.Equal("config-subagent", spec.SubagentTarget)
	s.Equal("be brief", spec.Instructions)
	s.Equal("hello there", spec.Greeting)
	s.Equal([]string{"refund"}, spec.SkillNames)
	s.Equal("handbook", spec.KnowledgeNamespace)
	s.Equal("support", spec.Tags["agent"])
}

func (s *SessionAPISuite) TestTheRequestWinsOverTheConfigItNamed() {
	config := &store.AgentConfig{
		ID:           "config-1",
		LLM:          "config-llm",
		Instructions: "be brief",
		Tags:         map[string]string{"agent": "support", "tier": "gold"},
	}
	instructions := "be thorough"

	spec := specOf(CreateSessionRequest{
		CallId:       callID("call-1"),
		Instructions: &instructions,
		Tags:         &map[string]string{"tier": "silver", "call": "42"},
	}, "acme", config)

	s.Equal("be thorough", spec.Instructions)
	s.Equal("config-llm", spec.LLMTarget, "a field the request left out stays the config's")
	s.Equal("silver", spec.Tags["tier"], "a label the call names wins over the config's")
	s.Equal("support", spec.Tags["agent"], "the config's other labels are still billed on")
	s.Equal("42", spec.Tags["call"])
}

func (s *SessionAPISuite) TestATextSessionTakesItsSkillsAndKnowledgeFromItsConfig() {
	// Delegating and looking things up are the reason to hold a conversation in writing at
	// all, and both are configured on the agent rather than repeated on every request.
	config := &store.AgentConfig{
		ID:                 "config-1",
		LLM:                "config-llm",
		Subagent:           "config-subagent",
		STT:                "config-stt",
		TTS:                "config-tts",
		Skills:             []string{"explain"},
		KnowledgeNamespace: "docs",
	}
	text := true

	spec := specOf(CreateSessionRequest{Text: &text}, "acme", config)
	s.Require().NoError(spec.Normalize())

	s.Empty(spec.CallID, "a conversation in writing joins no call")
	s.Equal([]string{"explain"}, spec.SkillNames)
	s.Equal("docs", spec.KnowledgeNamespace)
	s.Equal("config-subagent", spec.SubagentTarget, "and there is somebody to hand work to")
}

func (s *SessionAPISuite) TestNamingAConfigWithoutADatabaseIsRefused() {
	// The deployment under test has no store, so a config could only ever be ignored,
	// and a session that quietly ran on the wrong model is worse than one that failed.
	configID := "config-1"
	response := s.send(http.MethodPost, "/v1/agents/sessions", "acme",
		CreateSessionRequest{CallId: callID("call-1"), ConfigId: &configID})

	s.Equal(http.StatusBadRequest, response.StatusCode)

	var failure Error
	s.decodeBody(response, &failure)
	s.Contains(failure.Error, "no database configured")
}

// listed is the customer's sessions as the API reports them.
func (s *SessionAPISuite) listed(customerID string) []Session {
	response := s.send(http.MethodGet, "/v1/agents/sessions", customerID, nil)
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var sessions []Session
	s.decodeBody(response, &sessions)
	return sessions
}
