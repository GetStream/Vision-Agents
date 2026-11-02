//go:build integration

package session

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/telnyx"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// TransferSuite hands a phone caller to a human: the number is in Postgres, its lines are
// made at an in-memory Stream, and Telnyx is a server that takes every call it is asked to
// place.
type TransferSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	customer string
	number   string
	model    *stubLLM
	manager  *Manager
	// stream is the Stream the number's lines are made at and its calls ended at.
	stream *chattest.Server
}

func TestTransferSuite(t *testing.T) {
	suite.Run(t, new(TransferSuite))
}

func (s *TransferSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	s.ctx = context.Background()
	db, err := store.Open(testenv.Database(dsn, "session_transfer"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *TransferSuite) SetupTest() {
	s.customer = "transfer-" + uuid.NewString()
	s.number = fmt.Sprintf("+1512%07d", rand.IntN(10_000_000))
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: s.number, Vendor: "telnyx", Country: "US", CustomerID: s.customer, PurchasedAt: time.Now().UTC(),
	}))
	s.manager = s.manages()
}

// manages builds a manager whose sessions transfer through Telnyx, with a model that says
// goodbye and transfers to support on its first turn.
func (s *TransferSuite) manages() *Manager {
	logger := slog.New(slog.DiscardHandler)
	vendor := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"call_sid":"v3:human","status":"queued"}`))
	}))
	s.T().Cleanup(vendor.Close)
	config, err := phone.DefaultConfig()
	s.Require().NoError(err)
	for i := range config.Vendors {
		if config.Vendors[i].Vendor == "telnyx" {
			config.Vendors[i].Credentials = nil
		}
	}
	registry := phone.NewRegistry(config)
	registry.Register("telnyx", func() (phone.Provider, error) {
		return telnyx.New(telnyx.Options{APIKey: "telnyx-key", ConnectionID: "texml-app", BaseURL: vendor.URL})
	})
	s.stream = chattest.NewServer(s.T())
	telephony, err := phone.NewService(phone.ServiceOptions{
		Registry: registry, Store: s.store, Stream: phone.NewStreamFromClient(s.stream.Client), Logger: logger,
	})
	s.Require().NoError(err)

	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) { return &stubSTT{emitter: stt.NewEmitter(64)}, nil })
	transcriber, err := sttrouter.New(sttrouter.Options{Config: stubConfig(), Registry: transcription, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)
	s.model = &stubLLM{reply: "Putting you through.", calls: []llm.ToolCall{{ID: "call-1", Name: "transfer", Arguments: `{"destination":"support"}`}}}
	reasoning := llmrouter.NewRegistry()
	var opened atomic.Int32
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		if opened.Add(1) == 1 {
			return s.model, nil
		}
		return &stubLLM{reply: "Sure."}, nil
	})
	reasoner, err := llmrouter.New(llmrouter.Options{Config: stubConfig(), Registry: reasoning, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)
	speech := ttsrouter.NewRegistry()
	speech.Register("stub", func(routing.Spec) (tts.TTS, error) { return &stubTTS{emitter: tts.NewEmitter(64)}, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{Config: stubConfig(), Registry: speech, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	m, err := NewManager(ManagerOptions{
		LLM: reasoner, STT: transcriber, TTS: speaker, Phone: telephony, Logger: logger,
		Edge: func(context.Context, Spec, streamapp.Bound, *slog.Logger) (agent.Edge, error) {
			return attendedEdge{newQuietEdge(caller)}, nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(m.Shutdown()) })
	return m
}

// answers opens the session of an agent answering a call to the suite's number, on a
// config that transfers to support and may hang up, so the session has a way to end the call.
func (s *TransferSuite) answers() *Session {
	spec := FromConfig(store.AgentConfig{ID: "cfg-1", Calls: store.AgentCalls{
		Transfer: &store.TransferSettings{
			Enabled: true, Destinations: []store.TransferDestination{{Name: "support", Number: "+15125550199"}},
		},
		EndCall: &store.EndCallSettings{Enabled: true},
	}})
	spec.CustomerID, spec.CallType, spec.CallID = s.customer, "default", "phone-"+uuid.NewString()
	spec.LLMTarget, spec.STTTarget, spec.TTSTarget = "en-low-latency", "en-low-latency", "en-low-latency"
	spec.Phone = &PhoneSpec{Number: s.number}
	created, err := s.manager.Create(s.ctx, spec)
	s.Require().NoError(err)
	return created
}

func (s *TransferSuite) TestACallTransferredColdEndsItsSessionAsTransferred() {
	created := s.answers()
	events, detach := created.Watch()
	defer detach()

	_, err := created.Respond(s.ctx, "can I talk to support please", nil, options.LLM{})
	s.Require().NoError(err)

	s.transferred(events)
	s.Require().Eventually(func() bool {
		return created.State() == Ended && !s.manager.Running(created.ID())
	}, settleFor, 20*time.Millisecond, "the session outlived the agent that left on the transfer")
	s.Equal(EndTransferred, created.endReason())
	s.NotEmpty(s.asked("/sip/inbound_trunks"), "the transfer made no line at the suite's Stream")
	s.Empty(s.asked("/mark_ended"), "the call the caller was handed over on was ended")
}

// asked is the requests the suite's Stream was sent whose path ends in suffix.
func (s *TransferSuite) asked(suffix string) []string {
	var paths []string
	for _, request := range s.stream.Requests("test") {
		if strings.HasSuffix(request.Path, suffix) {
			paths = append(paths, request.Method+" "+request.Path)
		}
	}
	return paths
}

// transferred waits for the agent to have handed the caller over.
func (s *TransferSuite) transferred(events <-chan Event) {
	deadline := time.After(settleFor)
	for {
		select {
		case event := <-events:
			if _, ok := event.(agent.Transferred); ok {
				return
			}
		case <-deadline:
			s.FailNow("the agent never transferred the caller")
		}
	}
}
