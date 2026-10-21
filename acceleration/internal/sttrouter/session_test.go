//go:build integration

package sttrouter

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/muse"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// refusal is what ElevenLabs sends an account that has not accepted its terms, partway
// through a call (probed 2026-10-19).
const refusal = "You have not accepted the terms required to use this feature."

// ingressRefusal is the close frame Muse ends a session with when audio arrives slower
// than real time (the same frame internal/stt/muse/socket_test.go sends).
var ingressRefusal = &websocket.CloseError{Code: websocket.ClosePolicyViolation, Text: "Ingress below real-time"}

// SessionSuite covers what a session does once its provider fails, and what that failure
// leaves in the stats and in the provider's health.
type SessionSuite struct {
	suite.Suite
	ctx   context.Context
	store *store.Store
	live  *live.Client
	// customerID, refusing and working are unique to each test, since health is shared by
	// every run reading the same Redis.
	customerID string
	refusing   string
	working    string
	// holdClose, if set, holds the refusing provider's Close until it is closed, as a
	// vendor waiting for the last words does.
	holdClose chan struct{}
}

func TestSessionSuite(t *testing.T) {
	suite.Run(t, new(SessionSuite))
}

func (s *SessionSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || address == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}
	s.ctx = context.Background()

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(s.ctx))
	s.store = pgStore

	liveClient, err := live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.live = liveClient
}

func (s *SessionSuite) SetupTest() {
	unique := time.Now().Format("150405.000000000")
	s.customerID = "customer-" + unique
	s.refusing = "refusing-" + unique
	s.working = "working-" + unique
	s.holdClose = nil
}

// router serves a provider that settles turns and then fails with failure, and one that
// settles a turn for every frame.
func (s *SessionSuite) router(turns int, failure stt.Error) *Router {
	return s.routerWith(turns, failure, func(routing.Spec) (stt.STT, error) {
		return &refusingSTT{emitter: stt.NewEmitter(64), turns: math.MaxInt}, nil
	})
}

// routerWith is router with working built by the factory given.
func (s *SessionSuite) routerWith(turns int, failure stt.Error, working func(routing.Spec) (stt.STT, error)) *Router {
	registry := NewRegistry()
	registry.Register(s.refusing, func(routing.Spec) (stt.STT, error) {
		return &refusingSTT{emitter: stt.NewEmitter(64), turns: turns, failure: failure, hold: s.holdClose}, nil
	})
	registry.Register(s.working, working)
	router, err := New(Options{
		Config: routing.ModalityConfig{
			Providers: []routing.ProviderConfig{
				{Provider: s.refusing, Model: "live", Languages: []string{"en"}, Realtime: true},
				{Provider: s.working, Model: "live", Languages: []string{"en"}, Realtime: true},
			},
		},
		Registry: registry,
		Store:    s.store,
		Live:     s.live,
	})
	s.Require().NoError(err)
	return router
}

// start opens a session on the first of the two that is available, the refusing one
// when both are, as a caller who listed them in that order asked.
func (s *SessionSuite) start(router *Router) *Session {
	return s.startOn(router, s.refusing+"/live", s.working+"/live")
}

// startOn opens a session on the first of providers that is available.
func (s *SessionSuite) startOn(router *Router, providers ...string) *Session {
	session, err := router.Start(s.ctx, Request{CustomerID: s.customerID, Options: options.STT{
		Providers: providers,
	}})
	s.Require().NoError(err)
	return session
}

// failed starts a session on the refusing provider alone, so no other model takes it over,
// and speaks into it until it fails.
func (s *SessionSuite) failed(router *Router) *Session {
	session := s.startOn(router, s.refusing+"/live")
	s.Require().Equal(s.refusing, session.Provider())
	for {
		s.Require().NoError(session.ProcessAudio(frame(), stt.Participant{ID: "caller"}))
		if _, failed := (<-session.Events()).(stt.Error); failed {
			return session
		}
	}
}

func (s *SessionSuite) TestARefusedSessionIsCarriedOnByTheNextModel() {
	router := s.router(1, refused())
	session := s.start(router)

	s.IsType(stt.Transcript{}, s.speakInto(session))
	failover, ok := s.speakInto(session).(stt.Failover)
	s.Require().True(ok, "the caller heard of the refusal rather than of the model taking over")
	s.Equal(s.working, failover.Provider)
	s.Equal("live", failover.Model)
	s.Equal(s.refusing+"/live", failover.From)
	s.EqualError(failover.Err, refusal)
	s.Equal(s.working, session.Provider())
	s.IsType(stt.Transcript{}, s.speakInto(session), "the next model does not hear what the caller says after the refusal")
	s.Require().NoError(session.Close())
	s.Empty(s.drain(session))
	router.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).
		Where("customer_id = ?", s.customerID).Scan(s.ctx))
	var recorded []string
	for _, row := range rows {
		recorded = append(recorded, row.Provider+" "+row.ErrorCode)
	}
	s.ElementsMatch([]string{s.refusing + " ", s.refusing + " provider_fatal", s.working + " "}, recorded)
}

func (s *SessionSuite) TestARefusalNoOtherModelCanTakeOverEndsTheSession() {
	router := s.router(1, refused())
	defer router.Close()
	session := s.startOn(router, s.refusing+"/live")

	s.IsType(stt.Transcript{}, s.speakInto(session))
	failure, ok := s.speakInto(session).(stt.Error)
	s.Require().True(ok)
	s.True(failure.Fatal)
	s.EqualError(failure, refusal)
	s.Require().NoError(session.Close())
	s.Empty(s.drain(session))
}

func (s *SessionSuite) TestASocketThatDroppedIsNotHandedToTheNextModel() {
	drop := stt.Error{Err: errors.New("websocket: close 1006 (abnormal closure)"), Context: "read", Fatal: true}
	router := s.router(1, drop)
	defer router.Close()
	session := s.start(router)
	defer session.Close()

	s.IsType(stt.Transcript{}, s.speakInto(session))
	s.Equal(drop, s.speakInto(session))
	s.Equal(s.refusing, session.Provider())
}

func (s *SessionSuite) TestAnErrorTheProviderOutlivesIsNotHandedToTheNextModel() {
	failure := refused()
	failure.Fatal = false
	router := s.router(1, failure)
	defer router.Close()
	session := s.start(router)
	defer session.Close()

	s.IsType(stt.Transcript{}, s.speakInto(session))
	s.Equal(failure, s.speakInto(session))
	s.Equal(s.refusing, session.Provider())
}

func (s *SessionSuite) TestEveryModelThatRefusedIsLeftOut() {
	// Without health, nothing but what the session excludes keeps the first model out.
	again := s.refusing + "-again"
	registry := NewRegistry()
	for _, name := range []string{s.refusing, again} {
		registry.Register(name, func(routing.Spec) (stt.STT, error) {
			return &refusingSTT{emitter: stt.NewEmitter(64), failure: refused()}, nil
		})
	}
	registry.Register(s.working, func(routing.Spec) (stt.STT, error) {
		return &refusingSTT{emitter: stt.NewEmitter(64), turns: math.MaxInt}, nil
	})
	var providers []routing.ProviderConfig
	for _, name := range []string{s.refusing, again, s.working} {
		providers = append(providers, routing.ProviderConfig{Provider: name, Model: "live", Languages: []string{"en"}, Realtime: true})
	}
	router, err := New(Options{Config: routing.ModalityConfig{Providers: providers}, Registry: registry})
	s.Require().NoError(err)
	defer router.Close()
	session := s.startOn(router, s.refusing+"/live", again+"/live", s.working+"/live")
	defer session.Close()

	s.Equal(again, s.speakInto(session).(stt.Failover).Provider)
	second := s.speakInto(session).(stt.Failover)
	s.Equal(again+"/live", second.From)
	s.Equal(s.working, second.Provider, "the model that refused first was asked again")
}

func (s *SessionSuite) TestASessionClosedWhileTheNextModelStartsClosesThatModel() {
	next := &startingSTT{quietSTT: quietSTT{emitter: stt.NewEmitter(64)},
		starting: make(chan struct{}), release: make(chan struct{})}
	router := s.routerWith(0, refused(), func(routing.Spec) (stt.STT, error) { return next, nil })
	defer router.Close()
	session := s.start(router)
	s.Require().NoError(session.ProcessAudio(frame(), stt.Participant{ID: "caller"}))
	select {
	case <-next.starting:
	case <-time.After(3 * time.Second):
		s.FailNow("the next model was never started")
	}

	s.Require().NoError(session.Close())
	close(next.release)

	events := s.drain(session)
	s.True(next.closed.Load(), "the model that finished starting after the session closed was left open")
	s.Require().Len(events, 1)
	s.IsType(stt.Error{}, events[0], "a session closed before another model took over ends on the refusal")
}

func (s *SessionSuite) TestTheNextModelServesWhileTheRefusingOneIsStillClosing() {
	s.holdClose = make(chan struct{})
	router := s.router(0, refused())
	defer router.Close()
	session := s.start(router)
	defer session.Close()
	defer close(s.holdClose)

	s.IsType(stt.Failover{}, s.speakInto(session))
	s.IsType(stt.Transcript{}, s.speakInto(session))
}

func (s *SessionSuite) TestAudioAfterAFatalErrorIsDropped() {
	session := s.failed(s.router(2, refused()))

	for range 5 {
		s.NoError(session.ProcessAudio(frame(), stt.Participant{ID: "caller"}),
			"the caller was told once; every frame after it would say the same again")
	}
}

func (s *SessionSuite) TestAudioStopsReachingTheProviderBeforeTheCallerHearsOfAFatalError() {
	provider := &countingSTT{emitter: stt.NewEmitter(128)}
	recorder := routing.NewRecorder(routing.STT, nil, nil, slog.Default())
	defer recorder.Close()
	session := newSession(provider, routing.ProviderConfig{Provider: "counting", Model: "live"},
		routing.Owner{CustomerID: s.customerID}, recorder, nil)
	defer session.Close()
	// Revisions nobody reads fill the session's buffer, so the fatal error is held until the
	// caller reads that far.
	for range cap(session.events) {
		provider.emitter.Send(stt.Transcript{Text: "hel", Mode: stt.ModeReplacement})
	}
	provider.emitter.Send(refused())

	s.Require().Eventually(func() bool {
		before := provider.heard.Load()
		err := session.ProcessAudio(frame(), stt.Participant{ID: "caller"})
		return err == nil && provider.heard.Load() == before
	}, 3*time.Second, 5*time.Millisecond, "audio still reaches a provider whose fatal error is on its way to the caller")

	for range cap(session.events) {
		<-session.Events()
	}
	s.IsType(stt.Error{}, <-session.Events())
	heard := provider.heard.Load()
	s.Require().NoError(session.ProcessAudio(frame(), stt.Participant{ID: "caller"}))
	s.Equal(heard, provider.heard.Load(), "a caller told of the fatal error still reached the provider")
}

func (s *SessionSuite) TestAudioAfterAnErrorTheProviderOutlivesStillFails() {
	failure := refused()
	failure.Fatal = false
	session := s.failed(s.router(2, failure))

	s.Error(session.ProcessAudio(frame(), stt.Participant{ID: "caller"}))
}

func (s *SessionSuite) TestAFatalErrorIsRecordedWithTheVendorsReason() {
	router := s.router(0, refused())
	s.Require().NoError(s.failed(router).Close())
	router.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).
		Where("customer_id = ?", s.customerID).Scan(s.ctx))
	s.Require().Len(rows, 1)
	s.Equal("provider_fatal", rows[0].ErrorCode)
	s.Equal(refusal, rows[0].ErrorMessage)
}

func (s *SessionSuite) TestAnErrorThatIsNotFatalIsRecordedWithItsMessage() {
	failure := refused()
	failure.Fatal = false
	failure.Context = "commit_throttled"
	router := s.router(0, failure)
	s.Require().NoError(s.failed(router).Close())
	router.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).
		Where("customer_id = ?", s.customerID).Scan(s.ctx))
	s.Require().Len(rows, 1)
	s.Equal("commit_throttled", rows[0].ErrorCode)
	s.Equal(refusal, rows[0].ErrorMessage)
}

func (s *SessionSuite) TestAVendorsCloseCodeMarksTheModelUnavailable() {
	s.leftUnavailable(stt.Error{Err: ingressRefusal, Context: "read", Fatal: true})
}

func (s *SessionSuite) TestAVendorsCloseCodeWrappedByTheProviderMarksTheModelUnavailable() {
	s.leftUnavailable(stt.Error{Err: fmt.Errorf("muse: read: %w", ingressRefusal), Context: "read", Fatal: true})
}

func (s *SessionSuite) TestASocketThatClosedWithoutAFrameLeavesTheModelAvailable() {
	// What gorilla reports for a connection that ended with no close frame (conn.go,
	// errUnexpectedEOF).
	closed := &websocket.CloseError{Code: websocket.CloseAbnormalClosure, Text: "unexpected EOF"}
	s.leftAvailable(stt.Error{Err: closed, Context: "read", Fatal: true})
}

func (s *SessionSuite) TestOnlyARefusalCloseCodeMarksTheModelUnavailable() {
	refusals := []int{1002, 1003, 1007, 1008, 1009, 1010, 4000, 4503, 4999}
	drops := []int{1005, 1011, 1012, 1013, 1014, 3006}
	for _, code := range refusals {
		s.Run(fmt.Sprintf("%d", code), func() {
			s.False(transportDrop(stt.Error{Err: &websocket.CloseError{Code: code}, Context: "read", Fatal: true}))
		})
	}
	for _, code := range drops {
		s.Run(fmt.Sprintf("%d", code), func() {
			s.True(transportDrop(stt.Error{Err: &websocket.CloseError{Code: code}, Context: "read", Fatal: true}))
		})
	}
}

func (s *SessionSuite) TestAConnectionResetLeavesTheModelAvailable() {
	reset := &net.OpError{Op: "read", Net: "tcp", Err: syscall.ECONNRESET}
	s.leftAvailable(stt.Error{Err: reset, Context: "read", Fatal: true})
}

func (s *SessionSuite) TestAMuseSocketTheVendorClosedWithACodeMarksTheModelUnavailable() {
	router := s.museRouter(func(conn *websocket.Conn) {
		_ = conn.WriteMessage(websocket.CloseMessage,
			websocket.FormatCloseMessage(ingressRefusal.Code, ingressRefusal.Text))
	})
	defer router.Close()
	s.Require().NoError(s.failedSilently(router).Close())
	s.awaitHealth(4)

	health, err := s.live.Health(s.ctx, "stt", s.refusing, "live")
	s.Require().NoError(err)
	s.False(health.Available)
	next := s.start(router)
	s.Equal(s.working, next.Provider())
	// Drained before the router closes: a next session on Muse would still be recording
	// the turns the socket sends it.
	s.Require().NoError(next.Close())
	for range next.Events() {
	}
}

func (s *SessionSuite) TestAMuseSocketThatDroppedLeavesTheModelAvailable() {
	router := s.museRouter(func(conn *websocket.Conn) {
		_ = conn.NetConn().Close()
	})
	defer router.Close()
	s.Require().NoError(s.failedSilently(router).Close())
	s.awaitHealth(4)

	health, err := s.live.Health(s.ctx, "stt", s.refusing, "live")
	s.Require().NoError(err)
	s.True(health.Available)
}

func (s *SessionSuite) TestADroppedSocketLeavesTheModelAvailable() {
	for _, context := range []string{"read", "write"} {
		s.Run(context, func() {
			s.SetupTest()
			failure := stt.Error{Err: errors.New("websocket: close 1006 (abnormal closure)"), Context: context, Fatal: true}
			router := s.router(3, failure)
			defer router.Close()
			session := s.failed(router)
			s.NoError(session.ProcessAudio(frame(), stt.Participant{ID: "caller"}),
				"the socket is gone, so audio is still dropped")
			s.Require().NoError(session.Close())
			s.awaitHealth(4)

			health, err := s.live.Health(s.ctx, "stt", s.refusing, "live")
			s.Require().NoError(err)
			s.True(health.Available)
			next := s.start(router)
			defer next.Close()
			s.Equal(s.refusing, next.Provider())
		})
	}
}

func (s *SessionSuite) TestOneFatalErrorSendsTheNextSessionToAnotherProvider() {
	router := s.router(3, refused())
	defer router.Close()
	s.Require().NoError(s.failed(router).Close())
	s.awaitHealth(4)

	session := s.start(router)
	defer session.Close()
	s.Equal(s.working, session.Provider(), "three turns served do not outweigh an account refused")
}

func (s *SessionSuite) TestAnErrorTheProviderOutlivesLeavesItFirst() {
	failure := refused()
	failure.Fatal = false
	router := s.router(3, failure)
	defer router.Close()
	s.Require().NoError(s.failed(router).Close())
	s.awaitHealth(4)

	session := s.start(router)
	defer session.Close()
	s.Equal(s.refusing, session.Provider(), "one error in four is under the threshold")
}

func (s *SessionSuite) TestAFatalErrorFromAVoiceLeavesItAvailable() {
	recorder := routing.NewRecorder(routing.TTS, nil, s.live, slog.Default())
	config := routing.ProviderConfig{Provider: s.refusing, Model: "voice"}
	owner := routing.Owner{CustomerID: s.customerID}
	for range 3 {
		recorder.Record(config, routing.Stat{Owner: owner, Success: true})
	}
	// What ttsrouter records for a fatal error.
	recorder.Record(config, routing.Stat{Owner: owner, ErrorCode: "provider_fatal"})
	recorder.Close()

	health, err := s.live.Health(s.ctx, "tts", s.refusing, "voice")
	s.Require().NoError(err)
	s.EqualValues(4, health.Requests)
	s.True(health.Available)
}

// leftUnavailable serves three turns, fails with failure, and checks that the next
// session goes to the working provider.
func (s *SessionSuite) leftUnavailable(failure stt.Error) {
	router := s.router(3, failure)
	defer router.Close()
	s.Require().NoError(s.failed(router).Close())
	s.awaitHealth(4)

	health, err := s.live.Health(s.ctx, "stt", s.refusing, "live")
	s.Require().NoError(err)
	s.False(health.Available)
	next := s.start(router)
	defer next.Close()
	s.Equal(s.working, next.Provider())
}

// leftAvailable serves three turns, fails with failure, and checks that the next session
// still goes to the same provider.
func (s *SessionSuite) leftAvailable(failure stt.Error) {
	router := s.router(3, failure)
	defer router.Close()
	s.Require().NoError(s.failed(router).Close())
	s.awaitHealth(4)

	health, err := s.live.Health(s.ctx, "stt", s.refusing, "live")
	s.Require().NoError(err)
	s.True(health.Available)
	next := s.start(router)
	defer next.Close()
	s.Equal(s.refusing, next.Provider())
}

// museRouter serves the real Muse client against a socket that takes its setup frame,
// settles three turns and then hangs up the way hangUp does, and one provider that works.
// The three turns keep one failure under the error rate, so only a fatal stat can mark the
// model unavailable.
func (s *SessionSuite) museRouter(hangUp func(*websocket.Conn)) *Router {
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
		for turn := 1; turn <= 3; turn++ {
			for _, message := range []map[string]any{
				{"type": "speechStart", "turnId": turn},
				{"type": "speechComplete", "turnId": turn, "transcript": "hello"},
			} {
				if err := conn.WriteJSON(message); err != nil {
					return
				}
			}
		}
		hangUp(conn)
	}))
	s.T().Cleanup(server.Close)

	registry := NewRegistry()
	registry.Register(s.refusing, func(routing.Spec) (stt.STT, error) {
		return muse.New(muse.Options{
			APIKey: "test-key",
			URL:    "ws://" + strings.TrimPrefix(server.URL, "http://"),
		})
	})
	registry.Register(s.working, func(routing.Spec) (stt.STT, error) {
		return &quietSTT{emitter: stt.NewEmitter(64)}, nil
	})
	router, err := New(Options{
		Config: routing.ModalityConfig{
			Providers: []routing.ProviderConfig{
				{Provider: s.refusing, Model: "live", Languages: []string{"en"}, Realtime: true},
				{Provider: s.working, Model: "live", Languages: []string{"en"}, Realtime: true},
			},
		},
		Registry: registry,
		Store:    s.store,
		Live:     s.live,
	})
	s.Require().NoError(err)
	return router
}

// failedSilently starts a session on the refusing provider alone and waits, sending
// nothing, until it fails.
func (s *SessionSuite) failedSilently(router *Router) *Session {
	session := s.startOn(router, s.refusing+"/live")
	s.Require().Equal(s.refusing, session.Provider())
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event := <-session.Events():
			if _, failed := event.(stt.Error); failed {
				return session
			}
		case <-deadline:
			s.FailNow("the provider never reported the socket closing")
			return nil
		}
	}
}

// speakInto sends the session one frame and returns the event it answers with.
func (s *SessionSuite) speakInto(session *Session) stt.Event {
	s.Require().NoError(session.ProcessAudio(frame(), stt.Participant{ID: "caller"}))
	select {
	case event := <-session.Events():
		return event
	case <-time.After(3 * time.Second):
		s.FailNow("the session answered nothing")
		return nil
	}
}

// drain returns what a closed session still had to say, failing if its events never end.
func (s *SessionSuite) drain(session *Session) []stt.Event {
	var events []stt.Event
	deadline := time.After(3 * time.Second)
	for {
		select {
		case event, open := <-session.Events():
			if !open {
				return events
			}
			events = append(events, event)
		case <-deadline:
			s.FailNow("the session's events never ended")
			return nil
		}
	}
}

// awaitHealth waits for the refusing provider's stats to reach its health, which the
// recorder writes off the request path.
func (s *SessionSuite) awaitHealth(requests int64) {
	s.Require().Eventually(func() bool {
		health, err := s.live.Health(s.ctx, "stt", s.refusing, "live")
		return err == nil && health.Requests == requests
	}, 5*time.Second, 20*time.Millisecond)
}

// refusingSTT settles one turn per frame of audio until turns run out, then fails the
// next frame the way ElevenLabs refuses an account, and every write after it the way a
// socket the vendor closed does.
type refusingSTT struct {
	emitter *stt.Emitter
	turns   int
	failure stt.Error
	heard   int
	hold    chan struct{}
}

func (r *refusingSTT) Start(context.Context) error { return nil }
func (r *refusingSTT) Events() <-chan stt.Event    { return r.emitter.Events() }
func (r *refusingSTT) Provider() string            { return "refusing" }
func (r *refusingSTT) Model() string               { return "live" }

func (r *refusingSTT) Close() error {
	if r.hold != nil {
		<-r.hold
	}
	r.emitter.Close()
	return nil
}

func (r *refusingSTT) ProcessAudio(stt.PcmData, stt.Participant) error {
	r.heard++
	switch {
	case r.heard <= r.turns:
		r.emitter.Send(stt.Transcript{Text: "hello", Mode: stt.ModeFinal, AudioDurationMs: 20})
	case r.heard == r.turns+1:
		r.emitter.Send(r.failure)
	default:
		return errors.New("refusing: write audio: websocket: close sent")
	}
	return nil
}

// countingSTT counts the audio it is sent and emits only what the test sends through its
// emitter.
type countingSTT struct {
	emitter *stt.Emitter
	heard   atomic.Int64
}

func (c *countingSTT) Start(context.Context) error { return nil }
func (c *countingSTT) Events() <-chan stt.Event    { return c.emitter.Events() }
func (c *countingSTT) Close() error                { c.emitter.Close(); return nil }
func (c *countingSTT) Provider() string            { return "counting" }
func (c *countingSTT) Model() string               { return "live" }

func (c *countingSTT) ProcessAudio(stt.PcmData, stt.Participant) error {
	c.heard.Add(1)
	return nil
}

// startingSTT is a model still dialling: its Start says it began, then waits for release.
type startingSTT struct {
	quietSTT
	starting chan struct{}
	release  chan struct{}
	closed   atomic.Bool
}

func (p *startingSTT) Start(context.Context) error {
	close(p.starting)
	<-p.release
	return nil
}

func (p *startingSTT) Close() error {
	p.closed.Store(true)
	return p.quietSTT.Close()
}

func refused() stt.Error {
	return stt.Error{Err: errors.New(refusal), Context: "unaccepted_terms", Fatal: true}
}

func frame() stt.PcmData {
	return audio.PcmData{Samples: make([]int16, 320), SampleRate: stt.SampleRate, Channels: 1}
}
