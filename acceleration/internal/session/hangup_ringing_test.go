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
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
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

// ringingHangUp is the request Telnyx is sent to hang up the suite's placed call.
const ringingHangUp = "POST /v2/calls/v3:ringing/actions/hangup"

// RingingSuite ends sessions for calls the router placed: the number is in Postgres, the
// call's row is written by placing it through a Telnyx that records what it is asked.
type RingingSuite struct {
	suite.Suite
	ctx       context.Context
	store     *store.Store
	customer  string
	number    string
	telephony *phone.Service
	edge      *quietEdge
	manager   *Manager

	mu    sync.Mutex
	asked []string
	// stall, once set, holds every hangup until the test ends.
	stall chan struct{}
	// joining, once set, holds the next session to join its call until it is closed.
	joining chan struct{}
}

func TestRingingSuite(t *testing.T) {
	suite.Run(t, new(RingingSuite))
}

func (s *RingingSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	s.ctx = context.Background()
	db, err := store.Open(testenv.Database(dsn, "session_ringing"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *RingingSuite) SetupTest() {
	s.customer = "ringing-" + uuid.NewString()
	s.number = fmt.Sprintf("+1512%07d", rand.IntN(10_000_000))
	s.asked, s.stall, s.joining = nil, nil, nil
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: s.number, Vendor: "telnyx", Country: "US", CustomerID: s.customer, PurchasedAt: time.Now().UTC(),
	}))
	s.manager = s.manages()
}

func (s *RingingSuite) manages() *Manager {
	logger := slog.New(slog.DiscardHandler)
	vendor := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		request := r.Method + " " + r.URL.Path
		s.mu.Lock()
		s.asked = append(s.asked, request)
		stall := s.stall
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if request != ringingHangUp {
			_, _ = w.Write([]byte(`{"call_sid":"v3:ringing","status":"queued"}`))
			return
		}
		if stall != nil {
			<-stall
		}
		_, _ = w.Write([]byte(`{"data":{"result":"ok"}}`))
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
	telephony, err := phone.NewService(phone.ServiceOptions{
		Registry: registry, Store: s.store, Stream: phone.NewStreamFromClient(chattest.NewServer(s.T()).Client), Logger: logger,
	})
	s.Require().NoError(err)
	s.telephony = telephony

	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) { return &stubSTT{emitter: stt.NewEmitter(64)}, nil })
	transcriber, err := sttrouter.New(sttrouter.Options{Config: stubConfig(), Registry: transcription, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) { return &stubLLM{reply: "Hello."}, nil })
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
			s.mu.Lock()
			joining := s.joining
			s.joining = nil
			s.edge = newQuietEdge()
			edge := s.edge
			s.mu.Unlock()
			if joining != nil {
				joining <- struct{}{}
				<-joining
			}
			return attendedEdge{edge}, nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(m.Shutdown()) })
	return m
}

// rings places a call from the suite's number and opens the session that waits in it for
// the person to answer.
func (s *RingingSuite) rings() *Session {
	return s.opens(s.places())
}

// places places a call from the suite's number, returning the spec of a session for it.
func (s *RingingSuite) places() Spec {
	placed, err := s.telephony.Call(s.ctx, phone.CallRequest{
		Owner: routing.Owner{CustomerID: s.customer}, From: s.number, To: "+15125550100",
	})
	s.Require().NoError(err)
	spec := FromConfig(store.AgentConfig{ID: "cfg-1"})
	spec.CustomerID, spec.CallType, spec.CallID = s.customer, placed.CallType, placed.SessionID
	spec.LLMTarget, spec.STTTarget, spec.TTSTarget = "en-low-latency", "en-low-latency", "en-low-latency"
	spec.Phone = &PhoneSpec{Number: s.number, Vendor: placed.Vendor, VendorCallID: placed.VendorCallID, To: "+15125550100"}
	return spec
}

func (s *RingingSuite) opens(spec Spec) *Session {
	created, err := s.manager.Create(s.ctx, spec)
	s.Require().NoError(err)
	return created
}

// hungUp reports whether Telnyx was asked to hang the placed call up.
func (s *RingingSuite) hungUp() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Contains(s.asked, ringingHangUp)
}

// outcome is how the session's placed call is stored as having ended.
func (s *RingingSuite) outcome(created *Session) outcome.Outcome {
	call, err := s.telephony.Outcomes().Read(s.ctx, s.customer, created.Spec().CallID)
	s.Require().NoError(err)
	return call.Outcome
}

func (s *RingingSuite) TestClosingASessionWhoseCallStillRingsHangsItUp() {
	created := s.rings()

	closed, err := s.manager.Close(created.ID(), OwnerOf(created.Spec()))
	s.Require().NoError(err)
	s.Require().True(closed)

	s.Require().Eventually(s.hungUp, settleFor, 10*time.Millisecond, "the ringing call was never hung up")
	s.Eventually(func() bool { return s.outcome(created) == outcome.Canceled },
		settleFor, 10*time.Millisecond, "the call was not stored as canceled")
}

func (s *RingingSuite) TestShuttingDownHangsUpACallStillRinging() {
	created := s.rings()

	s.Require().NoError(s.manager.Shutdown())

	s.True(s.hungUp(), "shutdown left the call ringing")
	s.Equal(outcome.Canceled, s.outcome(created))
}

func (s *RingingSuite) TestASessionWhoseCallerJoinedDoesNotHangUp() {
	created := s.rings()
	s.edge.arrives(caller)
	s.Require().Eventually(created.CallerJoined, settleFor, 5*time.Millisecond, "the caller never joined")

	_, err := s.manager.Close(created.ID(), OwnerOf(created.Spec()))
	s.Require().NoError(err)
	s.Require().NoError(s.manager.Shutdown())

	s.Equal(Ended, created.State())
	s.False(s.hungUp(), "the answered call was hung up")
	s.Empty(s.outcome(created))
}

func (s *RingingSuite) TestAHangUpThatHangsDoesNotHoldUpTheClose() {
	stall := make(chan struct{})
	s.mu.Lock()
	s.stall = stall
	s.mu.Unlock()
	s.T().Cleanup(func() { close(stall) })
	created := s.rings()

	started := time.Now()
	_, err := s.manager.Close(created.ID(), OwnerOf(created.Spec()))
	s.Require().NoError(err)

	s.Less(time.Since(started), 2*time.Second, "the close waited on the vendor hanging up")
	s.Require().Eventually(s.hungUp, settleFor, 10*time.Millisecond, "the ringing call was never hung up")
}

// settled waits for every hangup a session that ended has started.
func (s *RingingSuite) settled() {
	s.manager.ended.Wait()
}

func (s *RingingSuite) TestASessionSupersededWhileItsCallRingsLeavesItToTheNewOne() {
	spec := s.places()
	first := s.opens(spec)

	second := s.opens(spec)
	s.settled()

	s.Equal(Ended, first.State())
	s.False(s.hungUp(), "the superseded session hung up the call the new one serves")
	s.Empty(s.outcome(second))

	_, err := s.manager.Close(second.ID(), OwnerOf(second.Spec()))
	s.Require().NoError(err)
	s.Require().Eventually(s.hungUp, settleFor, 10*time.Millisecond, "the ringing call was never hung up")
	s.Eventually(func() bool { return s.outcome(second) == outcome.Canceled },
		settleFor, 10*time.Millisecond, "the call was not stored as canceled")
}

func (s *RingingSuite) TestASessionTurnedAwayForAnIdAlreadyHeldLeavesTheCallRinging() {
	spec := s.places()
	spec.ID = uuid.NewString()
	other := spec
	other.UserID = "another-instance"
	joining := make(chan struct{})
	s.mu.Lock()
	s.joining = joining
	s.mu.Unlock()

	refused := make(chan error, 1)
	go func() {
		_, err := s.manager.Create(s.ctx, spec)
		refused <- err
	}()
	<-joining
	held := s.opens(other)
	close(joining)
	s.Require().ErrorIs(<-refused, ErrSessionExists)
	s.settled()

	s.True(s.manager.Running(held.ID()), "the session holding the id was ended")
	s.False(s.hungUp(), "the session turned away hung up the call another one holds")
	s.Empty(s.outcome(held))
}

func (s *RingingSuite) TestASessionWithNoPhoneInTheCallDoesNotHangUp() {
	spec := s.places()
	spec.Phone = nil
	created := s.opens(spec)

	_, err := s.manager.Close(created.ID(), OwnerOf(created.Spec()))
	s.Require().NoError(err)
	s.settled()

	s.False(s.hungUp(), "a session with no phone in its call hung one up")
}

func (s *RingingSuite) TestASessionTurnedAwayByShutdownLeavesItsCallRinging() {
	spec := s.places()
	joining := make(chan struct{})
	s.mu.Lock()
	s.joining = joining
	s.mu.Unlock()

	refused := make(chan error, 1)
	go func() {
		_, err := s.manager.Create(s.ctx, spec)
		refused <- err
	}()
	<-joining
	s.Require().NoError(s.manager.Shutdown())
	close(joining)
	s.Require().ErrorContains(<-refused, "shut down")
	s.settled()

	s.False(s.hungUp(), "a session turned away by shutdown hung up its call")
}

func (s *RingingSuite) TestShutdownWaitsNoLongerThanTheHangUpBound() {
	stall := make(chan struct{})
	s.mu.Lock()
	s.stall = stall
	s.mu.Unlock()
	s.T().Cleanup(func() { close(stall) })
	s.rings()

	shut := make(chan error, 1)
	go func() { shut <- s.manager.Shutdown() }()

	select {
	case err := <-shut:
		s.NoError(err)
	case <-time.After(hangUpTimeout + 3*time.Second):
		s.Fail("shutdown waited on a vendor that never answered the hangup")
	}
	s.True(s.hungUp(), "the ringing call was never hung up")
}
