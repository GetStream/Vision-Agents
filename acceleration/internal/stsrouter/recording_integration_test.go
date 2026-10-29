//go:build integration

package stsrouter

import (
	"context"
	"io"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing/pricingtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// RecordingSuite settles replies through a session over the built-in price list into
// Postgres, and reads back the row each one stored.
type RecordingSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	config routing.Config
	prices *pricing.List
}

func TestRecordingSuite(t *testing.T) {
	suite.Run(t, new(RecordingSuite))
}

func (s *RecordingSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })

	s.config, err = routing.DefaultConfig()
	s.Require().NoError(err)
	listed, err := routing.Prices(s.config, slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
	// At the vendors' rates, so each cost below is what a vendor bills for the turn.
	s.prices, err = pricing.NewList("vendor", pricing.AllAtCost(listed.Entries()), slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
}

// builtIn is the built-in config's speech-to-speech provider of that name.
func (s *RecordingSuite) builtIn(name string) routing.ProviderConfig {
	for _, provider := range s.config[routing.STS].Providers {
		if provider.Name() == name {
			return provider
		}
	}
	s.FailNow("the built-in config has no " + name)
	return routing.ProviderConfig{}
}

// recorded settles one reply that cost usage through a session of the built-in config's
// model, and returns the row it stored.
func (s *RecordingSuite) recorded(name string, usage sts.Usage) store.Request {
	customer := uuid.NewString()
	recorder := routing.NewRecorder(routing.STS, s.store, nil, s.prices, slog.New(slog.NewTextHandler(io.Discard, nil)))
	provider := newStubSTS()
	session := newSession(provider, s.builtIn(name), routing.Owner{CustomerID: customer}, recorder, recorder.For(routing.LLM))

	provider.emitter.Send(sts.ResponseStarted{ResponseID: "r1", Generation: 1, At: time.Now()})
	provider.emitter.Send(sts.ResponseComplete{ResponseID: "r1", Generation: 1, Usage: usage})
	s.Require().NoError(session.Close())
	for range session.Events() {
	}
	recorder.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", customer).Scan(s.ctx))
	s.Require().Len(rows, 1)
	return rows[0]
}

// TestARealtimeTurnIsPricedBySplit uses a spoken turn as gpt-realtime-2.1 reported it: 795
// text tokens in, 768 of them cached, 33 of audio, and 95 text and 176 audio out.
func (s *RecordingSuite) TestARealtimeTurnIsPricedBySplit() {
	row := s.recorded("openai/gpt-realtime-2.1", sts.Usage{
		InputTokens:           828,
		CachedInputTokens:     768,
		OutputTokens:          271,
		InputTextTokens:       795,
		InputAudioTokens:      33,
		CachedInputTextTokens: 768,
		OutputTextTokens:      95,
		OutputAudioTokens:     176,
		ReasoningTokens:       43,
	})

	s.Equal(pricing.Quantities{
		pricing.InputTextTokens:       27,
		pricing.InputAudioTokens:      33,
		pricing.CachedInputTextTokens: 768,
		pricing.OutputTextTokens:      95,
		pricing.OutputAudioTokens:     176,
	}, row.Meters)
	s.Equal(map[string]string{"reasoning_tokens": "43"}, row.Attributes)
	s.EqualValues(828, row.InputTokens, "the totals stay in their columns")
	s.EqualValues(768, row.CachedInputTokens)
	s.EqualValues(271, row.OutputTokens)
	s.EqualValues(15_015, row.CostMicros, "what OpenAI bills, where every token at the audio rate was 19,571")
}

// TestAGeminiLiveTurnIsPricedByModality uses a typed turn as
// gemini-3.8-live-extended-thinking reported it: a 3,162-token prompt whose details are
// 2,829 text and 222 audio, 54 audio out, and 60 of thinking.
func (s *RecordingSuite) TestAGeminiLiveTurnIsPricedByModality() {
	row := s.recorded("gemini/gemini-3.8-live-extended-thinking", sts.Usage{
		InputTokens:       3_162,
		OutputTokens:      114,
		InputTextTokens:   2_829,
		InputAudioTokens:  222,
		OutputTextTokens:  60,
		OutputAudioTokens: 54,
		ReasoningTokens:   60,
	})

	s.Equal(pricing.Quantities{
		pricing.InputTextTokens:   2_829,
		pricing.InputAudioTokens:  222,
		pricing.OutputTextTokens:  60,
		pricing.OutputAudioTokens: 54,
	}, row.Meters)
	s.Equal(map[string]string{"reasoning_tokens": "60", "prompt_tokens_not_in_details": "111"}, row.Attributes)
	s.EqualValues(3_706, row.CostMicros, "each modality at Google's rate, where $3 and $12 on the totals was 10,134")
}

func (s *RecordingSuite) TestAReplyWithNoSplitIsPricedOnItsTotals() {
	row := s.recorded("openai/gpt-realtime-2.1", sts.Usage{InputTokens: 1_000, CachedInputTokens: 500, OutputTokens: 100})

	s.Empty(row.Meters)
	s.Empty(row.Attributes)
	realtime := pricing.Item{Modality: string(routing.STS), Provider: "openai", Model: "gpt-realtime-2.1", Region: pricing.RegionUS}
	s.Equal(pricingtest.Micros(s.T(), s.prices, realtime, pricing.Quantities{pricing.InputTokens: 500, pricing.CachedInputTokens: 500,
		pricing.OutputTokens: 100}), row.CostMicros, "500 fresh, 500 cached and 100 generated at the audio rates, as before")
}
