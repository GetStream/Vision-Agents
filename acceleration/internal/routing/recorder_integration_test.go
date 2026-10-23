//go:build integration

package routing

import (
	"context"
	"io"
	"log/slog"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// RecorderSuite records rows into Postgres and reads back what was stored.
type RecorderSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	prices *pricing.List
}

func TestRecorderSuite(t *testing.T) {
	suite.Run(t, new(RecorderSuite))
}

// priced bills cache writes at their own rate, pixels by the megapixel, and half price for
// work the vendor served at its flex tier. gifted is free. number is priced outright when it
// is bought, as a phone number is.
var (
	priced = ProviderConfig{Provider: "vendor", Model: "priced", Price: Price{
		PerMillionInputTokens:           2,
		PerMillionCachedInputTokens:     0.2,
		PerMillionCacheWriteInputTokens: 2.5,
		PerMillionOutputTokens:          12,
		PerMegapixel:                    0.02,
		ServiceTiers:                    map[string]float64{"flex": 0.5},
	}}
	gifted = ProviderConfig{Provider: "vendor", Model: "gifted", Price: Price{Free: true}}
	number = ProviderConfig{Provider: "vendor", Model: "number", Price: Price{Free: true}}
)

func (s *RecorderSuite) SetupSuite() {
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

// recorded records the stat and returns the row it stored, written once the recorder has
// drained.
func (s *RecorderSuite) recorded(config ProviderConfig, stat Stat) store.Request {
	return s.recordedAt(s.prices, config, stat)
}

// recordedAt is recorded with the rows priced by prices.
func (s *RecorderSuite) recordedAt(prices pricing.Pricer, config ProviderConfig, stat Stat) store.Request {
	stat.CustomerID = uuid.NewString()
	recorder := NewRecorder(LLM, s.store, nil, prices, slog.New(slog.NewTextHandler(io.Discard, nil)))
	recorder.Record(config, stat)
	recorder.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", stat.CustomerID).Scan(s.ctx))
	s.Require().Len(rows, 1)
	return rows[0]
}

func (s *RecorderSuite) TestTheRowStoresWhatPricedIt() {
	row := s.recorded(priced, Stat{
		Usage:      Usage{InputTokens: 2_400, CachedInputTokens: 11, OutputTokens: 22, Pixels: 1_000_000},
		Meters:     pricing.Quantities{pricing.CacheWriteInputTokens: 2_386},
		Attributes: map[string]string{"service_tier": "flex"},
		Success:    true,
	})

	s.Equal(pricing.Quantities{pricing.CacheWriteInputTokens: 2_386, pricing.Pixels: 1_000_000}, row.Meters)
	s.Equal(map[string]string{"service_tier": "flex"}, row.Attributes)
	// 3 fresh, 11 cached, 2,386 written and 22 generated tokens and a megapixel, at half
	// price: (6 + 2.2 + 5,965 + 264 + 20,000) / 2.
	s.EqualValues(13_119, row.CostMicros)
	s.EqualValues(132, row.OutputCostMicros)
	repriced := s.prices.Quote(ItemOf(LLM, priced, row.Attributes), row.Quantities())
	s.Equal(row.CostMicros, repriced.CostMicros, "the row alone prices to what it was charged")
}

func (s *RecorderSuite) TestAFreeRowIsNotBillable() {
	free := s.recorded(gifted, Stat{Usage: Usage{InputTokens: 1_000}, Success: true})
	paid := s.recorded(priced, Stat{Usage: Usage{InputTokens: 1_000}, Success: true})

	s.False(free.Billable)
	s.Zero(free.CostMicros)
	s.True(paid.Billable)
	s.EqualValues(2_000, paid.CostMicros)
}

func (s *RecorderSuite) TestARowWithNothingBeyondItsColumnsStoresNoMeters() {
	row := s.recorded(priced, Stat{Usage: Usage{InputTokens: 1_000}, Success: true})

	s.Empty(row.Meters)
	s.Empty(row.Attributes)
}

func (s *RecorderSuite) TestACostTheVendorQuotedOutrightIsBillable() {
	row := s.recorded(number, Stat{CostMicros: 1_150_000, Success: true})

	s.EqualValues(1_150_000, row.CostMicros)
	s.True(row.Billable)
}

func (s *RecorderSuite) TestAPurchaseThatFailedIsNotBillable() {
	row := s.recorded(number, Stat{ErrorCode: "vendor_failed"})

	s.Zero(row.CostMicros)
	s.False(row.Billable)
}

func (s *RecorderSuite) TestACostQuotedOutrightKeepsTheOutputShare() {
	small := s.recorded(priced, Stat{Usage: Usage{InputTokens: 1_000, OutputTokens: 100}, CostMicros: 5_000, Success: true})
	large := s.recorded(priced, Stat{Usage: Usage{InputTokens: 1_000, OutputTokens: 1_000}, CostMicros: 5_000, Success: true})

	s.EqualValues(5_000, small.CostMicros)
	s.EqualValues(1_200, small.OutputCostMicros, "100 generated tokens at their rate")
	s.EqualValues(5_000, large.CostMicros)
	s.EqualValues(5_000, large.OutputCostMicros, "the output share is never more than the cost")
	s.True(small.Billable)
}

// markedUp is a list of priced and number, written by change_a, with a markup of 2500 bps.
func (s *RecorderSuite) markedUp() *pricing.List {
	var entries []pricing.Entry
	for _, entry := range PriceEntries(Config{LLM: {Providers: []ProviderConfig{priced, number}}}) {
		entry.Version = "change_a"
		entries = append(entries, entry)
	}
	list, err := pricing.NewList(pricing.ActiveVersion, entries, 2500, nil)
	s.Require().NoError(err)
	return list
}

func (s *RecorderSuite) TestTheRowStoresItsVersionAndItsCostAtTheVendorsRates() {
	row := s.recordedAt(s.markedUp(), priced, Stat{Usage: Usage{InputTokens: 1_000, OutputTokens: 100}, Success: true})

	s.Equal("change_a", row.PriceVersion)
	s.EqualValues(3_200, row.BaseCostMicros, "1,000 input tokens at $2 and 100 output at $12 per million")
	s.EqualValues(4_000, row.CostMicros)
	s.EqualValues(1_500, row.OutputCostMicros)
}

func (s *RecorderSuite) TestARowWithNoPriceNamesNone() {
	unknown := ProviderConfig{Provider: "vendor", Model: "unknown"}

	row := s.recordedAt(s.markedUp(), unknown, Stat{Usage: Usage{InputTokens: 1_000}, Success: true})

	s.Empty(row.PriceVersion)
	s.Zero(row.BaseCostMicros)
}

func (s *RecorderSuite) TestACostTheVendorQuotedOutrightIsResoldWithTheMarkup() {
	row := s.recordedAt(s.markedUp(), number, Stat{CostMicros: 1_000_000, Success: true})

	s.EqualValues(1_250_000, row.CostMicros)
	s.EqualValues(1_000_000, row.BaseCostMicros)
	s.Equal(pricing.QuotedVersion, row.PriceVersion)
	s.True(row.Billable)
}
