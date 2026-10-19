//go:build integration

package sttrouter

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// refusal is what ElevenLabs sends an account that has not accepted its terms, partway
// through a call (probed 2026-10-19).
const refusal = "You have not accepted the terms required to use this feature."

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
}

// router serves a provider that settles turns and then fails with failure, and one that
// works.
func (s *SessionSuite) router(turns int, failure stt.Error) *Router {
	registry := NewRegistry()
	registry.Register(s.refusing, func(routing.Spec) (stt.STT, error) {
		return &refusingSTT{emitter: stt.NewEmitter(64), turns: turns, failure: failure}, nil
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

// start opens a session on the first of the two that is available, the refusing one
// when both are, as a caller who listed them in that order asked.
func (s *SessionSuite) start(router *Router) *Session {
	session, err := router.Start(s.ctx, Request{CustomerID: s.customerID, Options: options.STT{
		Providers: []string{s.refusing + "/live", s.working + "/live"},
	}})
	s.Require().NoError(err)
	return session
}

// failed starts a session on the refusing provider and speaks into it until it fails.
func (s *SessionSuite) failed(router *Router) *Session {
	session := s.start(router)
	s.Require().Equal(s.refusing, session.Provider())
	for {
		s.Require().NoError(session.ProcessAudio(frame(), stt.Participant{ID: "caller"}))
		if _, failed := (<-session.Events()).(stt.Error); failed {
			return session
		}
	}
}

func (s *SessionSuite) TestAudioAfterAFatalErrorIsDropped() {
	session := s.failed(s.router(2, refused()))

	for range 5 {
		s.NoError(session.ProcessAudio(frame(), stt.Participant{ID: "caller"}),
			"the caller was told once; every frame after it would say the same again")
	}
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
}

func (r *refusingSTT) Start(context.Context) error { return nil }
func (r *refusingSTT) Events() <-chan stt.Event    { return r.emitter.Events() }
func (r *refusingSTT) Close() error                { r.emitter.Close(); return nil }
func (r *refusingSTT) Provider() string            { return "refusing" }
func (r *refusingSTT) Model() string               { return "live" }

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

func refused() stt.Error {
	return stt.Error{Err: errors.New(refusal), Context: "unaccepted_terms", Fatal: true}
}

func frame() stt.PcmData {
	return audio.PcmData{Samples: make([]int16, 320), SampleRate: stt.SampleRate, Channels: 1}
}
