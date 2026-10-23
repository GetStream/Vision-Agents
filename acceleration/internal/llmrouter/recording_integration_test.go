//go:build integration

package llmrouter

import (
	"context"
	"io"
	"log/slog"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// replayLLM answers every request with a response that has already happened.
type replayLLM struct {
	*stubLLM
	response llm.Response
}

func (r *replayLLM) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	response := r.response
	response.ID = params.ID
	return llm.Replay(llm.ResponseCompleted{Response: response}), nil
}

// RecordingSuite records responses through a session over the built-in price list into
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
	s.prices, err = routing.Prices(s.config, slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
}

// builtIn is the built-in config's LLM provider of that name.
func (s *RecordingSuite) builtIn(name string) routing.ProviderConfig {
	for _, provider := range s.config[routing.LLM].Providers {
		if provider.Name() == name {
			return provider
		}
	}
	s.FailNow("the built-in config has no " + name)
	return routing.ProviderConfig{}
}

// recorded asks the provider for one response through a session of the built-in config's
// model, and returns the row it stored and the cost the caller was told.
func (s *RecordingSuite) recorded(name string, response llm.Response) (store.Request, int64) {
	customer := uuid.NewString()
	recorder := routing.NewRecorder(routing.LLM, s.store, nil, s.prices, slog.New(slog.NewTextHandler(io.Discard, nil)))
	session := newSession(&replayLLM{stubLLM: newStubLLM(), response: response}, s.builtIn(name),
		routing.Owner{CustomerID: customer}, recorder, nil)

	var told int64
	stream, err := session.Create(s.ctx, llm.ResponseParams{
		ID:       "r1",
		Input:    prompt(),
		OnTiming: func(timing llm.CallTiming) { told = timing.CostMicros },
	})
	s.Require().NoError(err)
	_, err = llm.Collect(stream)
	s.Require().NoError(err)
	recorder.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", customer).Scan(s.ctx))
	s.Require().Len(rows, 1)
	return rows[0], told
}

// TestCacheWritesAndServiceTierReachTheRow uses the first turn of a cached conversation as
// gpt-5.6-terra answered it, served at the flex tier.
func (s *RecordingSuite) TestCacheWritesAndServiceTierReachTheRow() {
	row, told := s.recorded("openai/gpt-5.6-terra", llm.Response{
		Status:      llm.StatusCompleted,
		ServiceTier: "flex",
		Usage: llm.Usage{
			InputTokens:         2_389,
			InputTokensDetails:  llm.InputTokensDetails{CacheWriteTokens: 2_386},
			OutputTokens:        22,
			OutputTokensDetails: llm.OutputTokensDetails{ReasoningTokens: 7},
		},
	})

	s.Equal(pricing.Quantities{pricing.CacheWriteInputTokens: 2_386}, row.Meters)
	s.Equal(map[string]string{"service_tier": "flex", "reasoning_tokens": "7"}, row.Attributes)
	s.EqualValues(2_389, row.InputTokens, "the input stays the whole prompt")
	// Half of OpenAI's 6,235 µ$ at the default tier: 3 fresh tokens, 2,386 written at 1.25
	// times input and 22 generated.
	s.EqualValues(3_118, row.CostMicros)
	s.Equal(row.CostMicros, told, "the caller is told what the row costs")
	repriced := s.prices.Quote(routing.ItemOf(routing.LLM, s.builtIn("openai/gpt-5.6-terra"), row.Attributes), row.Quantities())
	s.Equal(row.CostMicros, repriced.CostMicros, "the row alone prices to what it was charged")
}

func (s *RecordingSuite) TestAResponseWithNothingBeyondItsTokensStoresNoMeters() {
	row, _ := s.recorded("openai/gpt-6-luna", llm.Response{
		Status: llm.StatusCompleted,
		Usage:  llm.Usage{InputTokens: 9, OutputTokens: 5},
	})

	s.Empty(row.Meters)
	s.Empty(row.Attributes)
	s.EqualValues(3, row.CostMicros)
}
