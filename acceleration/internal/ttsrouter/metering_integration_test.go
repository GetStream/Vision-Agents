//go:build integration

package ttsrouter

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// MeteringSuite records syntheses through a session over the built-in price list into a
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
	opened, err := store.Open(testenv.Database(dsn, "ttsmeter"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })

	s.config, err = routing.DefaultConfig()
	s.Require().NoError(err)
	s.prices, err = routing.Prices(s.config, slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
}

// inworld is the built-in config's Inworld TTS 2 Flash, which is priced by the character.
func (s *MeteringSuite) inworld() routing.ProviderConfig {
	for _, provider := range s.config[routing.TTS].Providers {
		if provider.Name() == "inworld/inworld-tts-2-flash" {
			return provider
		}
	}
	s.FailNow("the built-in config has no inworld/inworld-tts-2-flash")
	return routing.ProviderConfig{}
}

// completed waits for the session to pass on the next settled synthesis.
func (s *MeteringSuite) completed(session *Session) {
	for event := range session.Events() {
		if _, ok := event.(tts.SynthesisComplete); ok {
			return
		}
	}
	s.FailNow("the session closed before the synthesis settled")
}

func (s *MeteringSuite) TestCharactersAreWhatWasSent() {
	var logs bytes.Buffer
	customer := uuid.NewString()
	recorder := routing.NewRecorder(routing.TTS, s.store, nil, s.prices, slog.New(slog.NewTextHandler(io.Discard, nil)))
	provider := newStubTTS()
	session := newSession(provider, s.inworld(), routing.Owner{CustomerID: customer}, recorder, slog.New(slog.NewTextHandler(&logs, nil)))

	s.Require().NoError(session.Synthesize(tts.Request{ID: "u1", Text: "Grüß dich, "}))
	s.Require().NoError(session.Synthesize(tts.Request{ID: "u1", Text: "Zoë.", Final: true}))
	provider.emitter.Send(tts.SynthesisComplete{SynthesisID: "u1", Characters: 17})
	s.completed(session)
	// Text for an utterance already settled, as after a barge-in, which the voice drops.
	s.Require().NoError(session.Synthesize(tts.Request{ID: "u1", Text: "and the rest", Final: true}))
	s.Require().NoError(session.Synthesize(tts.Request{Text: "Bye.", Final: true}))
	s.Require().Len(provider.said, 4)
	named := provider.said[3].ID
	s.NotEmpty(named, "the session names a request that came without an id, not the vendor")
	provider.emitter.Send(tts.SynthesisComplete{SynthesisID: named})
	s.completed(session)
	s.Require().NoError(session.Close())
	recorder.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", customer).Order("id").Scan(s.ctx))
	s.Require().Len(rows, 2)
	s.EqualValues(15, rows[0].Characters, "the runes handed to the voice, not the 17 the adapter counted")
	s.Equal("u1", rows[0].OperationID)
	s.EqualValues(75, rows[0].CostMicros, "15 characters at $5 a million")
	s.EqualValues(4, rows[1].Characters, "an adapter that counted nothing is billed what it was sent")
	s.Equal(named, rows[1].OperationID)
	s.Contains(logs.String(), "level=WARN msg=\"the voice counted characters other than those it was sent")
	session.mu.Lock()
	defer session.mu.Unlock()
	s.Empty(session.sent, "text for a settled utterance is not counted, so nothing is left waiting for a completion")
}
