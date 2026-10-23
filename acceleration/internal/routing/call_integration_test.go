//go:build integration

package routing

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// used is work that says what it used, as every result a vendor returns does.
type used pricing.Quantities

func (u used) Used() pricing.Quantities { return pricing.Quantities(u) }

// costed is work whose vendor also said what it cost.
type costed struct {
	used
	micros int64
}

func (c costed) VendorCostMicros() int64 { return c.micros }

// CallSuite calls stand-in vendors through routing.Call and Settle into Postgres, and reads
// back the rows they left.
type CallSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	prices *pricing.List
}

func TestCallSuite(t *testing.T) {
	suite.Run(t, new(CallSuite))
}

func (s *CallSuite) SetupSuite() {
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
	s.prices, err = Prices(Config{LLM: {Providers: []ProviderConfig{priced, gifted, number}}}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
}

// meter is a meter of config for a customer of its own, recording through a recorder the
// test closes with rows.
func (s *CallSuite) meter(config ProviderConfig) Meter {
	return Meter{
		Recorder: NewRecorder(LLM, s.store, nil, s.prices, slog.New(slog.NewTextHandler(io.Discard, nil))),
		Config:   config,
		Owner:    Owner{CustomerID: uuid.NewString()},
	}
}

// rows drains the meter's recorder and returns what its customer's calls stored.
func (s *CallSuite) rows(meter Meter) []store.Request {
	meter.Recorder.Close()
	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", meter.Owner.CustomerID).Scan(s.ctx))
	return rows
}

// only is the one row the meter's calls stored.
func (s *CallSuite) only(meter Meter) store.Request {
	rows := s.rows(meter)
	s.Require().Len(rows, 1)
	return rows[0]
}

func (s *CallSuite) TestARequestRecordsOneRowWhateverHappens() {
	answered, failed, panicked := s.meter(priced), s.meter(priced), s.meter(priced)

	_, err := Call(s.ctx, answered, func(context.Context) (used, error) {
		return used{pricing.InputTokens: 1_000}, nil
	})
	s.Require().NoError(err)
	_, err = Call(s.ctx, failed, func(context.Context) (used, error) { return nil, errors.New("vendor down") })
	s.Require().Error(err)
	s.PanicsWithValue("vendor fell over", func() {
		_, _ = Call(s.ctx, panicked, func(context.Context) (used, error) { panic("vendor fell over") })
	}, "the panic carries on")

	s.True(s.only(answered).Success)
	s.Equal("provider_error", s.only(failed).ErrorCode)
	row := s.only(panicked)
	s.False(row.Success)
	s.Equal("panic", row.ErrorCode)
}

func (s *CallSuite) TestAFailureIsNotBilled() {
	meter := s.meter(priced)
	meter.Code = func(error) string { return "vendor_refused" }
	meter.CostMicros = 1_000_000

	_, err := Call(s.ctx, meter, func(context.Context) (used, error) {
		return used{pricing.InputTokens: 1_000}, errors.New("no such model")
	})

	s.Require().Error(err)
	row := s.only(meter)
	s.Equal("vendor_refused", row.ErrorCode)
	s.Equal("no such model", row.ErrorMessage)
	s.Zero(row.InputTokens, "what a failure says it used is not kept")
	s.Zero(row.CostMicros, "nor is the fee it would have cost")
}

func (s *CallSuite) TestAUsageIDBillsTwoCallsOnce() {
	meter := s.meter(priced)
	meter.UsageID = uuid.Must(uuid.NewV7())

	for range 2 {
		_, err := Call(s.ctx, meter, func(context.Context) (used, error) { return used{pricing.InputTokens: 1_000}, nil })
		s.Require().NoError(err)
	}

	row := s.only(meter)
	s.Equal(meter.UsageID, row.UsageID)
	s.EqualValues(2_000, row.CostMicros)
}

func (s *CallSuite) TestQuantitiesLandInColumnsAndMeters() {
	meter := s.meter(priced)

	_, err := Call(s.ctx, meter, func(context.Context) (used, error) {
		return used{
			pricing.InputTokens:           3,
			pricing.CachedInputTokens:     11,
			pricing.CacheWriteInputTokens: 2_386,
			pricing.OutputTokens:          22,
			pricing.Images:                1,
			pricing.Pixels:                1_000_000,
			pricing.InputAudioTokens:      0,
			pricing.Seconds:               7,
		}, nil
	})
	s.Require().NoError(err)

	row := s.only(meter)
	s.EqualValues(2_400, row.InputTokens, "the input column is the whole prompt")
	s.EqualValues(11, row.CachedInputTokens)
	s.EqualValues(22, row.OutputTokens)
	s.EqualValues(1, row.Images)
	s.Equal(pricing.Quantities{pricing.CacheWriteInputTokens: 2_386, pricing.Pixels: 1_000_000, pricing.Seconds: 7}, row.Meters,
		"the rest are meters, and a meter that used nothing is left out")
	// 3 fresh, 11 cached, 2,386 written and 22 generated tokens and a megapixel.
	s.EqualValues(26_237, row.CostMicros)
	s.Equal(row.CostMicros, s.prices.Quote(ItemOf(LLM, priced, row.Attributes), row.Quantities()).CostMicros,
		"the row alone prices to what it was charged")
	s.NotNil(row.LatencyMs)
}

func (s *CallSuite) TestAVendorsOwnCostIsKeptBesideThePrice() {
	meter := s.meter(priced)

	_, err := Call(s.ctx, meter, func(context.Context) (costed, error) {
		return costed{used: used{pricing.InputTokens: 1_000}, micros: 6}, nil
	})
	s.Require().NoError(err)

	row := s.only(meter)
	s.Equal(map[string]string{store.VendorCostAttribute: "6"}, row.Attributes)
	s.EqualValues(2_000, row.CostMicros, "the vendor's figure never prices the row")
}

func (s *CallSuite) TestFreeWorkIsPricedButNotBilled() {
	meter := s.meter(priced)
	meter.Free = true

	_, err := Call(s.ctx, meter, func(context.Context) (used, error) { return used{pricing.InputTokens: 1_000}, nil })
	s.Require().NoError(err)

	row := s.only(meter)
	s.EqualValues(2_000, row.CostMicros)
	s.False(row.Billable)
}

func (s *CallSuite) TestSettledWorkIsBilledOnceByItsUsageID() {
	meter := s.meter(number)
	meter.UsageID = UsageID("number-month", "number-"+uuid.NewString(), "2026-11")
	meter.CostMicros = 1_150_000

	s.Require().NoError(Settle(meter, used{}))
	s.Require().NoError(Settle(meter, used{}))

	row := s.only(meter)
	s.True(row.Success)
	s.Equal(meter.UsageID, row.UsageID)
	s.EqualValues(1_150_000, row.CostMicros)
	s.True(row.Billable)
}

func (s *CallSuite) TestSettlingWithoutAUsageIDIsRefusedLoudly() {
	var logs bytes.Buffer
	meter := s.meter(priced)
	meter.Recorder = NewRecorder(LLM, s.store, nil, s.prices, slog.New(slog.NewTextHandler(&logs, nil)))
	before := alerts.Total(usageLost.WithLabelValues(lostInvalid))

	s.Require().ErrorIs(Settle(meter, used{pricing.InputTokens: 1_000}), errNoUsageID)

	s.Empty(s.rows(meter), "work that could be billed twice is not billed at all")
	s.Equal(before+1, alerts.Total(usageLost.WithLabelValues(lostInvalid)), "it is a lost row, which the lost alert reads")
	s.Contains(logs.String(), `level=ERROR msg="usage row lost" reason=invalid`)
	s.Contains(logs.String(), `\"InputTokens\":1000`, "logged in full, with what it used")
}

func (s *CallSuite) TestAMeterNamesTheRowsAttributes() {
	meter := s.meter(priced)
	meter.UsageID = uuid.Must(uuid.NewV7())
	meter.Attributes = map[string]string{"parent_session": "session-1"}

	s.Require().NoError(Settle(meter, used{pricing.InputTokens: 1_000}))

	s.Equal(map[string]string{"parent_session": "session-1"}, s.only(meter).Attributes)
}
