package api

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/openairealtime"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
)

// stubConversation stands in for a speech-to-speech model, so the socket can be driven
// without a vendor. Whatever the socket sends it is recorded; whatever the test emits on it
// comes back down the socket.
type stubConversation struct {
	emitter      *sts.Emitter
	capabilities sts.Capabilities
	heard        chan sts.PcmData
	typed        chan string
	answers      chan string
	interrupts   chan int
	closed       chan struct{}
}

func newStubConversation(capabilities sts.Capabilities) *stubConversation {
	return &stubConversation{
		emitter:      sts.NewEmitter(sts.EmitterBuffer),
		capabilities: capabilities,
		heard:        make(chan sts.PcmData, 16),
		typed:        make(chan string, 16),
		answers:      make(chan string, 16),
		interrupts:   make(chan int, 16),
		closed:       make(chan struct{}),
	}
}

func (s *stubConversation) Start(context.Context) error { return nil }
func (s *stubConversation) ProcessAudio(pcm sts.PcmData, _ sts.Participant) error {
	s.heard <- pcm
	return nil
}
func (s *stubConversation) SendText(text string, _ sts.Participant) error { s.typed <- text; return nil }
func (s *stubConversation) SendFrame(llm.ImagePart) error                 { return sts.ErrNoImages }
func (s *stubConversation) SetInstructions(string) error                  { return nil }
func (s *stubConversation) SetTools([]llm.Tool) error                     { return nil }
func (s *stubConversation) Answer(callID, output string, _ error) error {
	s.answers <- callID + "=" + output
	return nil
}
func (s *stubConversation) Prompt(string) error          { return nil }
func (s *stubConversation) Interrupt(playedMs int) error { s.interrupts <- playedMs; return nil }
func (s *stubConversation) Events() <-chan sts.Event     { return s.emitter.Events() }
func (s *stubConversation) Close() error {
	select {
	case <-s.closed:
	default:
		close(s.closed)
	}
	s.emitter.Close()
	return nil
}
func (s *stubConversation) Provider() string                { return "stub" }
func (s *stubConversation) Model() string                   { return "stub" }
func (s *stubConversation) SampleRate() int                 { return 24_000 }
func (s *stubConversation) Capabilities() sts.Capabilities { return s.capabilities }

// STSStreamSuite drives the speech-to-speech socket end to end against a stub model.
type STSStreamSuite struct {
	suite.Suite
	server *httptest.Server
	// built receives every stub the registry makes, so a test can reach the one its
	// socket landed on.
	built chan *stubConversation
}

func TestSTSStreamSuite(t *testing.T) {
	suite.Run(t, new(STSStreamSuite))
}

func (s *STSStreamSuite) SetupTest() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	s.built = make(chan *stubConversation, 4)

	registry := stsrouter.NewRegistry()
	for _, provider := range config[routing.STS].Providers {
		capabilities := capabilitiesOf(provider)
		registry.Register(provider.Provider, func(routing.Spec) (sts.STS, error) {
			stub := newStubConversation(capabilities)
			s.built <- stub
			return stub, nil
		})
	}
	conversing, err := stsrouter.New(stsrouter.Options{Config: config[routing.STS], Registry: registry})
	s.Require().NoError(err)
	s.T().Cleanup(conversing.Close)

	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.STS: conversing},
		Streams: &Streams{STS: conversing},
	})
	s.Require().NoError(err)
	s.server = httptest.NewServer(server.Handler())
	s.T().Cleanup(s.server.Close)
}

// capabilitiesOf is what the real provider behind a config entry would report, so the
// router's own check of the config passes for the stub as it does for the vendor.
func capabilitiesOf(provider routing.ProviderConfig) sts.Capabilities {
	switch provider.Provider {
	case openairealtime.OpenAI.Provider:
		return openairealtime.CapabilitiesFor(openairealtime.OpenAI, provider.Model)
	case openairealtime.XAI.Provider:
		return openairealtime.CapabilitiesFor(openairealtime.XAI, provider.Model)
	case openairealtime.Qwen.Provider:
		return openairealtime.CapabilitiesFor(openairealtime.Qwen, provider.Model)
	default:
		return gemini.CapabilitiesFor(provider.Model)
	}
}

// open dials the socket and sends the start frame.
func (s *STSStreamSuite) open(opening map[string]any) *websocket.Conn {
	address := "ws" + strings.TrimPrefix(s.server.URL, "http") + "/v1/sts/stream"
	header := http.Header{}
	header.Set(CustomerHeader, "acme")

	connection, response, err := websocket.DefaultDialer.Dial(address, header)
	s.Require().NoError(err)
	s.T().Cleanup(func() { connection.Close() })
	s.Require().Equal(http.StatusSwitchingProtocols, response.StatusCode)

	opening["type"] = "start"
	s.Require().NoError(connection.WriteJSON(opening))
	return connection
}

// next reads the next frame: a JSON frame decoded, or a binary payload.
func (s *STSStreamSuite) next(connection *websocket.Conn) (map[string]any, []byte) {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(5 * time.Second)))
	kind, payload, err := connection.ReadMessage()
	s.Require().NoError(err)
	if kind == websocket.BinaryMessage {
		return nil, payload
	}
	var decoded map[string]any
	s.Require().NoError(json.Unmarshal(payload, &decoded))
	return decoded, nil
}

// nextJSON reads the next JSON frame, skipping audio.
func (s *STSStreamSuite) nextJSON(connection *websocket.Conn) map[string]any {
	for {
		decoded, payload := s.next(connection)
		if payload == nil {
			return decoded
		}
	}
}

func (s *STSStreamSuite) stub() *stubConversation {
	select {
	case stub := <-s.built:
		return stub
	case <-time.After(5 * time.Second):
		s.FailNow("no model was built")
		return nil
	}
}

func (s *STSStreamSuite) TestAConversationFlowsBothWays() {
	connection := s.open(map[string]any{
		"target": "openai/gpt-realtime-2",
		"sts":    map[string]any{"instructions": "Be brief."},
	})
	stub := s.stub()

	started := s.nextJSON(connection)
	s.Equal("started", started["type"])
	s.Equal("openai", started["provider"])
	s.Equal("gpt-realtime-2", started["model"])
	s.EqualValues(24_000, started["sample_rate"], "the rate the model speaks at, so playback can be readied")

	// The caller's audio goes up as it is.
	pcm := audio.PcmData{Samples: make([]int16, 160), SampleRate: 16_000, Channels: 1}
	s.Require().NoError(connection.WriteMessage(websocket.BinaryMessage, pcm.Bytes()))
	select {
	case heard := <-stub.heard:
		s.Equal(16_000, heard.SampleRate)
		s.Len(heard.Samples, 160)
	case <-time.After(5 * time.Second):
		s.FailNow("the model never heard the audio")
	}

	// The model's reply comes down: a frame saying it began, then its voice under a
	// header that says which reply the chunk belongs to.
	stub.emitter.Send(sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()})
	stub.emitter.Send(sts.AudioChunk{
		ResponseID: "r1", Generation: 1, Index: 0,
		Audio: audio.PcmData{Samples: []int16{1, 2, 3}, SampleRate: 24_000, Channels: 1},
	})
	begun := s.nextJSON(connection)
	s.Equal("response_started", begun["type"])
	s.Equal("r1", begun["id"])
	s.EqualValues(1, begun["generation"])

	_, payload := s.next(connection)
	s.Require().Len(payload, stsAudioHeader+6)
	s.EqualValues(24_000, binary.LittleEndian.Uint32(payload[0:4]))
	s.EqualValues(1, binary.LittleEndian.Uint16(payload[4:6]))
	s.EqualValues(stsAudioVersion, binary.LittleEndian.Uint16(payload[6:8]))
	s.EqualValues(1, binary.LittleEndian.Uint32(payload[8:12]), "the generation is what lets a client drop a cut-off reply's tail")
	s.EqualValues(0, binary.LittleEndian.Uint32(payload[12:16]))
	s.Equal([]int16{1, 2, 3}, audio.FromBytes(payload[stsAudioHeader:], 24_000, 1).Samples)

	// What the caller sends mid-stream reaches the model.
	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "text", "text": "hello"}))
	s.Equal("hello", <-stub.typed)
	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "tool_result", "tool_call_id": "c1", "output": "sunny"}))
	s.Equal("c1=sunny", <-stub.answers)
	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "interrupt", "played_ms": 120}))
	s.Equal(120, <-stub.interrupts)

	// A reply the caller cut off says so, and nothing more of it gets through.
	stub.emitter.Send(sts.ResponseComplete{ResponseID: "r1", Generation: 1, Interrupted: true, AudioDurationMs: 0.125})
	stub.emitter.Send(sts.AudioChunk{ResponseID: "r1", Generation: 1, Index: 1,
		Audio: audio.PcmData{Samples: []int16{9}, SampleRate: 24_000, Channels: 1}})
	stub.emitter.Send(sts.ToolCall{ResponseID: "r1", CallID: "c2", Name: "get_weather", Arguments: "{}"})
	complete := s.nextJSON(connection)
	s.Equal("response_complete", complete["type"])
	s.Equal(true, complete["interrupted"])
	decoded, payload := s.next(connection)
	s.Nil(payload, "audio from the interrupted reply must not follow its completion")
	s.Equal("tool_call", decoded["type"])
	s.Equal("get_weather", decoded["name"])

	// What the model cannot do is refused, not dropped.
	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "frame", "image_url": "data:image/png;base64,iVBORw0KGgo="}))
	refused := s.nextJSON(connection)
	s.Equal("error", refused["type"])
	s.Contains(refused["error"], "does not accept images")

	connection.Close()
	select {
	case <-stub.closed:
	case <-time.After(5 * time.Second):
		s.FailNow("hanging up should close the model's session")
	}
}

func (s *STSStreamSuite) TestATermNothingCanServeIsRefusedBeforeAnythingIsBilled() {
	connection := s.open(map[string]any{
		"target": "sts-fast",
		"sts":    map[string]any{"turn_detection": "semantic"},
	})

	refused := s.nextJSON(connection)
	s.Equal("error", refused["type"])
	s.Contains(refused["error"], "semantic_turns")
	s.Equal("closed", s.nextJSON(connection)["type"])
	s.Empty(s.built, "no model should have been started for a request that could never be served")
}

func (s *STSStreamSuite) TestToolsOnTheStartFrameRouteToAModelThatCallsThem() {
	connection := s.open(map[string]any{
		"target": "qwen/qwen3.5-omni-plus-realtime",
		"tools":  []map[string]any{{"name": "get_weather", "description": "Weather."}},
	})

	refused := s.nextJSON(connection)
	s.Equal("error", refused["type"])
	s.Contains(refused["error"], "tools", "handing a model tools it will never call is the thing terms exist to prevent")
}
