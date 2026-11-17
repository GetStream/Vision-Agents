//go:build integration

package routing_test

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// CustomModelRecorderSuite records rows for a customer's own model, configured as the router
// configures one, and reads back what was stored.
type CustomModelRecorderSuite struct {
	suite.Suite
	ctx   context.Context
	store *store.Store
	// logs is what the price list and the recorder's writer logged.
	logs     *bytes.Buffer
	recorder *routing.Recorder
}

func TestCustomModelRecorderSuite(t *testing.T) {
	suite.Run(t, new(CustomModelRecorderSuite))
}

func (s *CustomModelRecorderSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "routing"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

func (s *CustomModelRecorderSuite) SetupTest() {
	s.logs = &bytes.Buffer{}
	logger := slog.New(slog.NewTextHandler(s.logs, nil))
	prices, err := pricing.NewList("config", nil, logger)
	s.Require().NoError(err)
	s.recorder = routing.NewRecorder(routing.LLM, s.store, nil, prices, logger)
}

// recorded records the stat and returns the row it stored, written once the recorder has
// drained.
func (s *CustomModelRecorderSuite) recorded(config routing.ProviderConfig, stat routing.Stat) store.Request {
	stat.CustomerID = uuid.NewString()
	s.recorder.Record(config, stat)
	s.recorder.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", stat.CustomerID).Scan(s.ctx))
	s.Require().Len(rows, 1)
	return rows[0]
}

func (s *CustomModelRecorderSuite) TestACustomModelsRowKeepsItsCostAndIsNotBillable() {
	config := llmrouter.CustomModelConfig(store.CustomModel{Name: "mine", PerMillionInputTokens: 1, PerMillionOutputTokens: 2})
	usage := routing.Usage{InputTokens: 1_000_000, OutputTokens: 500_000}

	quote := s.recorder.Quote(config, usage)
	row := s.recorded(config, routing.Stat{Usage: usage, Success: true})

	s.Equal("custom", row.Provider)
	s.EqualValues(2_000_000, row.CostMicros, "the customer still sees what their host bills")
	s.EqualValues(1_000_000, row.OutputCostMicros)
	s.False(row.Billable, "the customer pays their own host")
	s.False(quote.Unpriced)
	s.NotContains(s.logs.String(), "billable work has no price")
}

func (s *CustomModelRecorderSuite) TestACustomModelWithNoRatesIsFree() {
	config := llmrouter.CustomModelConfig(store.CustomModel{Name: "unrated"})
	usage := routing.Usage{InputTokens: 1_000, OutputTokens: 1_000}

	quote := s.recorder.Quote(config, usage)
	row := s.recorded(config, routing.Stat{Usage: usage, Success: true})

	s.Zero(row.CostMicros)
	s.False(row.Billable)
	s.False(quote.Unpriced)
	s.NotContains(s.logs.String(), "billable work has no price")
}
