//go:build integration

package stsrouter

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/openailive"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/openairealtime"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// MeteringSuite holds conversations with the real adapters, against a vendor's socket that
// plays back what the vendor sent, through a session over the built-in price list into a
// database of its own, and reads back the rows they stored.
type MeteringSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	config routing.Config
	prices *pricing.List
}

func TestMeteringSuite(t *testing.T) {
	suite.Run(t, new(MeteringSuite))
}

func (s *MeteringSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "stsmeter"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })

	s.config, err = routing.DefaultConfig()
	s.Require().NoError(err)
	s.prices, err = routing.Prices(s.config, slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
}

// builtIn is the built-in config's speech-to-speech provider of that name.
func (s *MeteringSuite) builtIn(name string) routing.ProviderConfig {
	for _, provider := range s.config[routing.STS].Providers {
		if provider.Name() == name {
			return provider
		}
	}
	s.FailNow("the built-in config has no " + name)
	return routing.ProviderConfig{}
}

// vendorSocket is a vendor's socket for one session. It sends hello once connected, answers
// each client frame named in replies, and sends whatever the test hands it, in order.
type vendorSocket struct {
	server  *httptest.Server
	hello   string
	replies map[string]string

	mu   sync.Mutex
	conn *websocket.Conn
	open chan struct{}
}

func (s *MeteringSuite) vendorSocket(hello string, replies map[string]string) *vendorSocket {
	socket := &vendorSocket{hello: hello, replies: replies, open: make(chan struct{})}
	upgrader := websocket.Upgrader{}
	socket.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		socket.mu.Lock()
		socket.conn = conn
		socket.mu.Unlock()
		if socket.hello != "" {
			socket.send(socket.hello)
		}
		close(socket.open)
		for {
			_, raw, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var frame struct {
				Type string `json:"type"`
			}
			_ = json.Unmarshal(raw, &frame)
			if reply, ok := socket.replies[frame.Type]; ok {
				socket.send(reply)
			}
		}
	}))
	s.T().Cleanup(socket.server.Close)
	return socket
}

func (v *vendorSocket) url() string { return "ws://" + strings.TrimPrefix(v.server.URL, "http://") }

// send writes one frame to the client, and returns once it has gone.
func (v *vendorSocket) send(frame string) {
	v.mu.Lock()
	defer v.mu.Unlock()
	_ = v.conn.WriteMessage(websocket.TextMessage, []byte(frame))
}

// rows is what the customer's session stored, in the order it was written.
func (s *MeteringSuite) rows(customer string) []store.Request {
	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", customer).Order("id").Scan(s.ctx))
	return rows
}

// session opens a session of the built-in model name over the started provider.
func (s *MeteringSuite) session(provider sts.STS, name string) (*Session, *routing.Recorder, string) {
	ctx, cancel := context.WithTimeout(s.ctx, 5*time.Second)
	defer cancel()
	s.Require().NoError(provider.Start(ctx))
	customer := uuid.NewString()
	recorder := routing.NewRecorder(routing.STS, s.store, nil, s.prices, slog.New(slog.NewTextHandler(io.Discard, nil)))
	owner := routing.Owner{CustomerID: customer, Attribution: attributed}
	return newSession(provider, s.builtIn(name), owner, recorder), recorder, customer
}

// attributed is who every session here is billed to and asked for by, which every row it
// writes, its side rows included, names.
var attributed = routing.Attribution{OrganizationID: "org-1", APIKey: "key-1", UserID: "user-1"}

// requireAttributed checks a row names who the session was billed to and asked for by.
func (s *MeteringSuite) requireAttributed(row store.Request) {
	s.Equal(attributed.OrganizationID, row.OrganizationID)
	s.Equal(attributed.APIKey, row.APIKey)
	s.Equal(attributed.UserID, row.UserID)
}

// closed closes the session and waits for every row it wrote.
func (s *MeteringSuite) closed(session *Session, recorder *routing.Recorder) {
	s.Require().NoError(session.Close())
	for range session.Events() {
	}
	recorder.Close()
}

// audible is that much of the model saying something, as GPT-Live sends it.
func audible(ms int) string {
	samples := make([]byte, openailive.SampleRate*ms/1000*2)
	for i := 0; i < len(samples); i += 2 {
		binary.LittleEndian.PutUint16(samples[i:], uint16(int16(8_000)))
	}
	return base64.StdEncoding.EncodeToString(samples)
}

// TestGPTLiveBillsSessionSecondsAndTheBackend plays back the spike's 30-second session on a
// gpt-5.6-luna backend: the seconds at 13 and 28 and 29 as it closed, and two backend
// responses, the first sent twice.
func (s *MeteringSuite) TestGPTLiveBillsSessionSecondsAndTheBackend() {
	backend := func(id string, input, output, reasoning int) string {
		raw, _ := json.Marshal(map[string]any{"type": "response.event", "event": map[string]any{
			"type": "response.completed", "response": map[string]any{
				"id": id, "model": "gpt-5.6-luna", "status": "completed", "service_tier": "default",
				"usage": map[string]any{
					"input_tokens": input, "input_tokens_details": map[string]any{"cached_tokens": 0, "cache_write_tokens": 0},
					"output_tokens": output, "output_tokens_details": map[string]any{"reasoning_tokens": reasoning},
					"total_tokens": input + output,
				},
			},
		}})
		return string(raw)
	}
	socket := s.vendorSocket("", map[string]string{
		"session.start": `{"type":"session.started","session":{"id":"sess_1","model":"gpt-live-1"}}`,
		"session.close": `{"type":"session.closed","reason":"close_requested","usage":{"seconds":29.0}}`,
	})
	provider, err := openailive.New(openailive.Options{
		APIKey: "k", URL: socket.url(), Backend: "gpt-5.6-luna", HandshakeTimeout: 5 * time.Second,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	s.Require().NoError(err)
	session, recorder, customer := s.session(provider, "openai/gpt-live-1")
	<-socket.open

	socket.send(`{"type":"session.output_audio.delta","delta":"` + audible(200) + `"}`)
	socket.send(`{"type":"session.usage.updated","usage":{"seconds":13.0}}`)
	socket.send(backend("resp_1", 890, 31, 10))
	socket.send(backend("resp_2", 946, 15, 0))
	socket.send(backend("resp_1", 890, 31, 10))
	socket.send(`{"type":"session.usage.updated","usage":{"seconds":28.0}}`)
	s.closed(session, recorder)

	var seconds, backends, replies []store.Request
	for _, row := range s.rows(customer) {
		switch {
		case row.Modality == string(routing.LLM):
			backends = append(backends, row)
		case row.Meters[pricing.SessionSeconds] != 0:
			seconds = append(seconds, row)
		default:
			replies = append(replies, row)
		}
	}

	s.Require().Len(seconds, 3)
	var billed []int64
	for _, row := range seconds {
		billed = append(billed, row.Meters[pricing.SessionSeconds])
		s.Equal(session.id, row.OperationID)
		s.True(row.Billable)
		s.requireAttributed(row)
	}
	s.Equal([]int64{13, 15, 1}, billed, "the seconds since each report, adding up to the 29 it closed at")
	s.Equal([]int64{10_833, 12_500, 833}, []int64{seconds[0].CostMicros, seconds[1].CostMicros, seconds[2].CostMicros},
		"$0.05 a minute")

	s.Require().Len(backends, 2, "a backend response sent twice is billed once")
	s.Equal("openai", backends[0].Provider)
	s.Equal("gpt-5.6-luna", backends[0].Model)
	s.EqualValues(890, backends[0].InputTokens)
	s.EqualValues(31, backends[0].OutputTokens)
	s.Equal(map[string]string{parentSession: session.id, "service_tier": "default", "reasoning_tokens": "10"}, backends[0].Attributes)
	s.EqualValues(215, backends[0].CostMicros, "priced as gpt-5.6-luna")
	s.EqualValues(207, backends[1].CostMicros)
	s.Equal(session.id, backends[1].Attributes[parentSession])
	s.requireAttributed(backends[0])

	s.Require().Len(replies, 1, "the reply is still recorded, for its timing")
	s.False(replies[0].Billable, "the seconds are the bill")
	s.Zero(replies[0].CostMicros)
}

// TestXAIVoiceBillsTheAudioXAICountsAndTypedTurns plays back the spike's push-to-talk session
// on grok-voice-3.0: billable_audio_seconds at 2, 7, 9, 11 and 13 over five replies, a typed
// turn before the first, third and fifth, and a tool's result before the fourth. The caller
// speaks once more after the last reply and hangs up.
func (s *MeteringSuite) TestXAIVoiceBillsTheAudioXAICountsAndTypedTurns() {
	socket := s.vendorSocket(`{"type":"session.created","session":{"id":"sess_1"}}`, map[string]string{
		"session.update": `{"type":"session.updated","session":{"id":"sess_1"}}`,
	})
	provider, err := openairealtime.New(openairealtime.Options{
		Vendor: openairealtime.XAI, APIKey: "k", URL: socket.url(), Model: "grok-voice-3.0", HandshakeTimeout: 5 * time.Second,
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	s.Require().NoError(err)
	session, recorder, customer := s.session(provider, "xai/grok-voice-3.0")
	<-socket.open

	reply := func(id string, billable int) {
		socket.send(`{"type":"response.created","response":{"id":"` + id + `"}}`)
		socket.send(`{"type":"response.done","response":{"id":"` + id + `","status":"completed","usage":{}},` +
			`"usage":{"input_tokens":7,"output_tokens":119,"output_token_details":{"text_tokens":7,"audio_tokens":112},` +
			`"billable_audio_seconds":` + strconv.Itoa(billable) + `}}`)
		for event := range session.Events() {
			if complete, ok := event.(sts.ResponseComplete); ok && complete.ResponseID == id {
				return
			}
		}
		s.FailNow("the session closed before " + id + " settled")
	}
	caller := sts.PcmData{Samples: make([]int16, sts.InputSampleRate), SampleRate: sts.InputSampleRate, Channels: 1}

	s.Require().NoError(session.SendText("What is the capital of France?", sts.Participant{}))
	reply("resp_1", 2)
	s.Require().NoError(session.ProcessAudio(caller, sts.Participant{}))
	reply("resp_2", 7)
	s.Require().NoError(session.SendText("And of Italy?", sts.Participant{}))
	reply("resp_3", 9)
	s.Require().NoError(session.Answer("call_1", `{"temperature_c":17}`, nil))
	s.Require().NoError(session.ProcessAudio(caller, sts.Participant{}))
	reply("resp_4", 11)
	s.Require().NoError(session.SendText("Thanks.", sts.Participant{}))
	reply("resp_5", 13)
	s.Require().NoError(session.ProcessAudio(sts.PcmData{Samples: make([]int16, sts.InputSampleRate*3/2), SampleRate: sts.InputSampleRate, Channels: 1}, sts.Participant{}))
	s.closed(session, recorder)

	rows := s.rows(customer)
	s.Require().Len(rows, 6)
	var audio, messages, costs []int64
	var total int64
	for _, row := range rows {
		audio = append(audio, row.AudioMs)
		messages = append(messages, row.Meters[pricing.Messages])
		costs = append(costs, row.CostMicros)
		total += row.CostMicros
		s.Equal(session.id, row.OperationID)
	}
	s.Equal([]int64{2_000, 5_000, 2_000, 2_000, 2_000, 1_500}, audio,
		"the growth in what xAI bills, and the caller's audio after the last reply")
	s.Equal([]int64{1, 0, 1, 0, 1, 0}, messages, "the typed turns, and not the tool's result")
	s.Equal([]int64{6_667, 6_667, 6_667, 2_667, 6_667, 2_000}, costs, "$4.80 an hour and $0.004 a typed turn")
	s.EqualValues(31_335, total, "where the caller's audio before each reply priced 2,667")
	s.Equal(map[string]string{routing.AttributeMeasuredBy: routing.MeasuredByRouter}, rows[5].Attributes, "no event reported the last row")
}

// TestRealtimeTranscriptionIsItsOwnRow plays back OpenAI writing down two caller turns, the
// first reported twice, each 33 audio tokens in and 16 text out.
func (s *MeteringSuite) TestRealtimeTranscriptionIsItsOwnRow() {
	socket := s.vendorSocket(`{"type":"session.created","session":{"id":"sess_1"}}`, map[string]string{
		"session.update": `{"type":"session.updated","session":{"id":"sess_1"}}`,
	})
	provider, err := openairealtime.New(openairealtime.Options{
		Vendor: openairealtime.OpenAI, APIKey: "k", URL: socket.url(), Model: "gpt-realtime-2.1", InputTranscript: true,
		HandshakeTimeout: 5 * time.Second, Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	s.Require().NoError(err)
	session, recorder, customer := s.session(provider, "openai/gpt-realtime-2.1")
	<-socket.open

	transcribed := func(item, text string) {
		socket.send(`{"type":"conversation.item.input_audio_transcription.completed","item_id":"` + item +
			`","transcript":"` + text + `","usage":{"type":"tokens","input_tokens":33,` +
			`"input_token_details":{"audio_tokens":33,"text_tokens":0},"output_tokens":16,"total_tokens":49}}`)
		for event := range session.Events() {
			if heard, ok := event.(sts.InputTranscript); ok && heard.Text == text {
				return
			}
		}
		s.FailNow("the session closed before the caller was heard")
	}
	transcribed("item_1", "What is the weather in Turin?")
	transcribed("item_1", "What is the weather in Turin?")
	transcribed("item_2", "And tomorrow?")
	s.closed(session, recorder)

	rows := s.rows(customer)
	s.Require().Len(rows, 2, "one row per caller turn, however often it is reported")
	for _, row := range rows {
		s.Equal("gpt-4o-mini-transcribe", row.Model)
		s.Equal(string(routing.STS), row.Modality)
		s.Equal(pricing.Quantities{pricing.InputAudioTokens: 33, pricing.OutputTextTokens: 16}, row.Meters)
		s.EqualValues(121, row.CostMicros, "$1.25 a million in and $5 out")
		s.Equal(map[string]string{parentSession: session.id}, row.Attributes)
		s.requireAttributed(row)
	}
}
