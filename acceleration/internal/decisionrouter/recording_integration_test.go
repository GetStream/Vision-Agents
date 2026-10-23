//go:build integration

package decisionrouter

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// RecordingSuite classifies through a router over Postgres, with a stub classifier standing in
// for the vendor, and reads back the row each request stored.
type RecordingSuite struct {
	suite.Suite
	ctx   context.Context
	store *store.Store
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
	opened, err := store.Open(testenv.Database(dsn, "decisionrouter"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

// classified asks provider one question for a customer of its own, and returns the row the
// request stored once the router has drained.
func (s *RecordingSuite) classified(provider *stubClassifier) store.Request {
	registry := NewRegistry()
	registry.Register("vendor", func(routing.Spec) (decisionmodel.Provider, error) { return provider, nil })
	router, err := New(Options{
		Config: routing.ModalityConfig{Providers: []routing.ProviderConfig{{
			Provider: "vendor", Model: "judge", Languages: []string{"en"}, Realtime: true, Tier: routing.LowLatency,
			Price: routing.Price{PerMillionInputTokens: 40},
		}}},
		Registry: registry,
		Store:    s.store,
		Logger:   slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	s.Require().NoError(err)
	customer := uuid.NewString()
	session, err := router.Start(s.ctx, Request{CustomerID: customer, Target: "vendor/judge"})
	s.Require().NoError(err)
	_, _ = session.Classify(s.ctx, decisionmodel.Request{
		State:     "how do I make a pizza",
		Questions: map[string]decisionmodel.Question{"violates": decisionmodel.Noul("Is this off topic?", "", "")},
	})
	router.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", customer).Scan(s.ctx))
	s.Require().Len(rows, 1, "one request is one row")
	return rows[0]
}

func (s *RecordingSuite) TestAnAnswerIsBilledWithTheVendorsOwnCostBesideIt() {
	answered := allowed(0.1)
	answered.Usage = decisionmodel.Usage{InputTokens: 150, OutputTokens: 2, VendorCostMicros: 6}

	row := s.classified(&stubClassifier{model: "judge", answered: answered})

	s.True(row.Success)
	s.EqualValues(150, row.InputTokens)
	s.EqualValues(2, row.OutputTokens)
	s.EqualValues(6_000, row.CostMicros, "priced by the list, not by what the vendor said")
	s.Equal(map[string]string{store.VendorCostAttribute: "6"}, row.Attributes)
}

func (s *RecordingSuite) TestAFailedRequestIsNotBilled() {
	row := s.classified(&stubClassifier{model: "judge", err: errors.New("vendor down")})

	s.False(row.Success)
	s.Equal("classify_failed", row.ErrorCode)
	s.Zero(row.InputTokens)
	s.Zero(row.CostMicros)
}
