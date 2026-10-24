//go:build integration

package routing

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/getsentry/sentry-go"
	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
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
	// logs is what the recorders of the test logged.
	logs *bytes.Buffer
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
	// realtime is speech to speech priced by its tokens alone, and hosted runs on the
	// deployment's own machines with no price.
	realtime = ProviderConfig{Provider: "vendor", Model: "realtime", Price: Price{Rates: map[pricing.Meter]pricing.Rate{
		pricing.InputAudioTokens:  {USD: 32, Per: 1_000_000},
		pricing.OutputAudioTokens: {USD: 64, Per: 1_000_000},
	}}}
	hosted = ProviderConfig{Provider: "vendor", Model: "hosted", Local: true}
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
	s.prices, err = Prices(Config{
		LLM: {Providers: []ProviderConfig{priced, gifted, number, hosted}},
		STS: {Providers: []ProviderConfig{realtime}},
	}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	s.Require().NoError(err)
}

func (s *RecorderSuite) SetupTest() {
	s.logs = &bytes.Buffer{}
}

// recorded records the stat and returns the row it stored, written once the recorder has
// drained.
func (s *RecorderSuite) recorded(config ProviderConfig, stat Stat) store.Request {
	return s.recordedAt(s.prices, config, stat)
}

// recordedAt is recorded with the rows priced by prices.
func (s *RecorderSuite) recordedAt(prices pricing.Pricer, config ProviderConfig, stat Stat) store.Request {
	return s.recordedAs(LLM, prices, config, stat)[0]
}

// recordedAs records the stats through one recorder of the modality, logging to s.logs and
// pricing by prices, and returns the rows they stored once it has drained.
func (s *RecorderSuite) recordedAs(modality Modality, prices pricing.Pricer, config ProviderConfig, stats ...Stat) []store.Request {
	customer := uuid.NewString()
	recorder := NewRecorder(modality, s.store, nil, prices, slog.New(slog.NewTextHandler(s.logs, nil)))
	for _, stat := range stats {
		// A key, as every owner site names one: a row without one is logged as an error.
		stat.CustomerID, stat.APIKey = customer, "key-a"
		recorder.Record(config, stat)
	}
	recorder.Close()

	var rows []store.Request
	s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", customer).Scan(s.ctx))
	s.Require().Len(rows, len(stats))
	return rows
}

// counted is how many rows a metering counter has counted for the item of config.
func counted(counter *prometheus.CounterVec, modality Modality, config ProviderConfig, labels ...string) float64 {
	return alerts.Total(counter.WithLabelValues(append([]string{string(modality), config.Provider, config.Model}, labels...)...))
}

func (s *RecorderSuite) TestBillableWorkWithNoQuantityRaisesTheAlarm() {
	before := counted(usageMissing, LLM, priced)

	s.recordedAs(LLM, s.prices, priced, Stat{Success: true}, Stat{Success: true})

	s.Equal(before+2, counted(usageMissing, LLM, priced), "every row is counted")
	s.Equal(1, strings.Count(s.logs.String(), `msg="billable work recorded no usage" modality=llm provider=vendor model=priced`),
		"and the item is logged once")
}

func (s *RecorderSuite) TestRealtimeAudioWithoutTokensRaisesTheAlarm() {
	before := counted(usageMissing, STS, realtime)

	s.recordedAs(STS, s.prices, realtime, Stat{Usage: Usage{AudioMs: 5_000}, Success: true})

	s.Equal(before+1, counted(usageMissing, STS, realtime), "the audio heard is not what the model is priced by")
}

func (s *RecorderSuite) TestWorkThatReportedWhatItIsPricedByIsQuiet() {
	before := counted(usageMissing, LLM, priced)

	s.recorded(priced, Stat{Usage: Usage{InputTokens: 1}, Success: true})

	s.Equal(before, counted(usageMissing, LLM, priced), "a token too cheap to cost a micro is still usage")
	s.NotContains(s.logs.String(), "level=ERROR")
}

func (s *RecorderSuite) TestFreeWorkWithNoQuantityIsQuiet() {
	before := counted(usageMissing, LLM, gifted)

	s.recorded(gifted, Stat{Success: true})

	s.Equal(before, counted(usageMissing, LLM, gifted))
	s.NotContains(s.logs.String(), "level=ERROR")
}

func (s *RecorderSuite) TestWorkTheRouterMeasuredAtNothingIsQuiet() {
	before := counted(usageMissing, LLM, priced)
	stat := Stat{Success: true}
	stat.MeasuredByRouter()

	row := s.recorded(priced, stat)

	s.True(row.Billable)
	s.Equal(before, counted(usageMissing, LLM, priced), "the router's own count is the count, even at zero")
	s.NotContains(s.logs.String(), "level=ERROR")
}

func (s *RecorderSuite) TestAFailureWithNoQuantityIsQuiet() {
	before := counted(usageMissing, LLM, priced)

	s.recorded(priced, Stat{ErrorCode: "provider_error"})

	s.Equal(before, counted(usageMissing, LLM, priced), "a failure is not billed, so it reports nothing")
}

func (s *RecorderSuite) TestBillableWorkWithNoPriceIsCounted() {
	unpriced := ProviderConfig{Provider: "vendor", Model: "unpriced"}
	before, missing := counted(usageUnpriced, LLM, unpriced), counted(usageMissing, LLM, unpriced)

	s.recordedAs(LLM, s.prices, unpriced, Stat{Usage: Usage{InputTokens: 1_000}, Success: true}, Stat{Success: true})

	s.Equal(before+2, counted(usageUnpriced, LLM, unpriced), "every row is counted")
	s.Equal(missing, counted(usageMissing, LLM, unpriced), "a row with no price raises one alarm, not two")
}

func (s *RecorderSuite) TestLocalWorkWithNoPriceIsQuiet() {
	before, missing := counted(usageUnpriced, LLM, hosted), counted(usageMissing, LLM, hosted)

	s.recorded(hosted, Stat{Success: true})

	s.Equal(before, counted(usageUnpriced, LLM, hosted))
	s.Equal(missing, counted(usageMissing, LLM, hosted))
}

func (s *RecorderSuite) TestAnEstimateIsBilledCountedAndLogged() {
	before, missing := counted(usageEstimated, LLM, priced, estimatedMissing), counted(usageMissing, LLM, priced)

	row := s.recorded(priced, Stat{Usage: Usage{InputTokens: 1_000}, Success: true,
		Attributes: map[string]string{AttributeUsageEstimated: "true"}})

	s.EqualValues(2_000, row.CostMicros)
	s.True(row.Billable)
	s.Equal(before+1, counted(usageEstimated, LLM, priced, estimatedMissing))
	s.Equal(missing, counted(usageMissing, LLM, priced))
	s.Contains(s.logs.String(), `msg="billable work recorded an estimate" modality=llm provider=vendor model=priced`)
}

func (s *RecorderSuite) TestAnEstimateOfWorkCutShortIsCountedQuietly() {
	before := counted(usageEstimated, LLM, priced, estimatedCancelled)

	row := s.recorded(priced, Stat{Usage: Usage{InputTokens: 1_000}, ErrorCode: ErrorCancelled,
		Attributes: map[string]string{AttributeUsageEstimated: "true"}})

	s.EqualValues(2_000, row.CostMicros, "what the vendor read before the cut is billed")
	s.Equal(before+1, counted(usageEstimated, LLM, priced, estimatedCancelled))
	s.NotContains(s.logs.String(), "level=ERROR")
}

func (s *RecorderSuite) TestAnEstimateOfWorkADeadlineCutShortIsCountedQuietly() {
	before, missing := counted(usageEstimated, LLM, priced, estimatedTimeout), counted(usageEstimated, LLM, priced, estimatedMissing)

	row := s.recorded(priced, Stat{Usage: Usage{InputTokens: 1_000}, ErrorCode: ErrorTimeout,
		Attributes: map[string]string{AttributeUsageEstimated: "true"}})

	s.EqualValues(2_000, row.CostMicros)
	s.Equal(before+1, counted(usageEstimated, LLM, priced, estimatedTimeout))
	s.Equal(missing, counted(usageEstimated, LLM, priced, estimatedMissing))
	s.NotContains(s.logs.String(), "level=ERROR")
}

func (s *RecorderSuite) TestTokensThatDoNotAddUpToTheVendorsTotalAreCounted() {
	before := counted(usageMismatch, LLM, priced)

	s.recorded(priced, Stat{Usage: Usage{InputTokens: 1_301, OutputTokens: 21}, VendorTotalTokens: 2_747, Success: true})
	s.recorded(priced, Stat{Usage: Usage{InputTokens: 1_301, OutputTokens: 1_446}, VendorTotalTokens: 2_747, Success: true})

	s.Equal(before+1, counted(usageMismatch, LLM, priced), "only the row that misses part of the total")
	s.Contains(s.logs.String(), "vendor_total_tokens=2747 input_tokens=1301 output_tokens=21")
}

func (s *RecorderSuite) TestEachItemTheRecorderCountedIsAnAlertOfItsOwn() {
	alarmed := ProviderConfig{Provider: "vendor", Model: "alarmed", Price: Price{PerMillionInputTokens: 2}}
	cut := ProviderConfig{Provider: "vendor", Model: "cut", Price: Price{PerMillionInputTokens: 2}}
	prices, err := Prices(Config{LLM: {Providers: []ProviderConfig{alarmed, cut}}}, nil)
	s.Require().NoError(err)
	s.recordedAs(LLM, prices, alarmed, Stat{Success: true}, Stat{Success: true})
	s.recordedAs(LLM, prices, cut, Stat{Usage: Usage{InputTokens: 10}, ErrorCode: ErrorCancelled,
		Attributes: map[string]string{AttributeUsageEstimated: "true"}})

	var mu sync.Mutex
	var sent []*sentry.Event
	client, err := sentry.NewClient(sentry.ClientOptions{BeforeSend: func(event *sentry.Event, _ *sentry.EventHint) *sentry.Event {
		mu.Lock()
		defer mu.Unlock()
		sent = append(sent, event)
		return nil
	}})
	s.Require().NoError(err)
	loop := alerts.New(alerts.Options{Hub: sentry.NewHub(client, sentry.NewScope()), Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
		Every: time.Hour, Limit: time.Hour}, MeteringAlerts(time.Hour)...)
	ctx, cancel := context.WithTimeout(s.ctx, 50*time.Millisecond)
	defer cancel()
	loop.Run(ctx)

	mu.Lock()
	defer mu.Unlock()
	var fingerprints []string
	for _, event := range sent {
		if item := event.Tags["item"]; item == "llm/vendor/alarmed" || item == "llm/vendor/cut" {
			fingerprints = append(fingerprints, event.Fingerprint...)
		}
	}
	s.Equal([]string{"usage_missing:llm/vendor/alarmed"}, fingerprints,
		"two rows of one item are one alert, and a response cut short is none")
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

// spending is a gate that keeps what it was told each unit of work cost, by usage id.
type spending struct {
	mu    sync.Mutex
	spent map[uuid.UUID]int64
	owner Owner
	// grants are what it answers every unit of work is drawn from.
	grants []int64
}

var _ Spender = (*spending)(nil)

func (g *spending) Admit(context.Context, Owner) (Admission, error) { return Admission{}, nil }

func (g *spending) Spent(owner Owner, usageID uuid.UUID, micros int64) []int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.spent == nil {
		g.spent = map[uuid.UUID]int64{}
	}
	g.spent[usageID] += micros
	g.owner = owner
	return g.grants
}

func (g *spending) total() int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	var sum int64
	for _, micros := range g.spent {
		sum += micros
	}
	return sum
}

func (s *RecorderSuite) TestRecordTellsASpenderWhatItCost() {
	gate := &spending{}
	writer := NewWriter(WriterOptions{Store: s.store, Logger: slog.New(slog.DiscardHandler)})
	defer writer.Close(s.ctx)
	recorder := newRecorder(LLM, nil, nil, gate, writer, s.prices, slog.New(slog.DiscardHandler))
	usageID := uuid.Must(uuid.NewV7())

	recorder.Record(priced, Stat{Owner: Owner{CustomerID: uuid.NewString(), Attribution: Attribution{APIKey: "key-a"}},
		UsageID: usageID, Usage: Usage{InputTokens: 1_000, OutputTokens: 100}, Success: true})

	s.EqualValues(3_200, gate.total(), "told as Record returns, before the writer has written anything")
	s.Contains(gate.spent, usageID, "under the row's usage id")
	s.Equal("key-a", gate.owner.APIKey)
}

func (s *RecorderSuite) TestARowNamesTheGrantsItsWorkIsDrawnFrom() {
	gate := &spending{grants: []int64{7, 9}}
	writer := NewWriter(WriterOptions{Store: s.store, Logger: slog.New(slog.DiscardHandler)})
	recorder := newRecorder(LLM, nil, nil, gate, writer, s.prices, slog.New(slog.DiscardHandler))
	customer := uuid.NewString()

	recorder.Record(priced, Stat{Owner: Owner{CustomerID: customer, Attribution: Attribution{APIKey: "key-a"}},
		Usage: Usage{InputTokens: 1_000}, Success: true})
	writer.Close(s.ctx)

	var row store.Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&row).Where("customer_id = ?", customer).Scan(s.ctx))
	s.Equal([]int64{7, 9}, row.LimitGrants)
}

func (s *RecorderSuite) TestWorkThatCostsNothingHoldsNothing() {
	gate := &spending{}
	writer := NewWriter(WriterOptions{Store: s.store, Logger: slog.New(slog.DiscardHandler)})
	defer writer.Close(s.ctx)
	recorder := newRecorder(LLM, nil, nil, gate, writer, s.prices, slog.New(slog.DiscardHandler))
	owner := Owner{CustomerID: uuid.NewString(), Attribution: Attribution{APIKey: "key-a"}}

	recorder.Record(gifted, Stat{Owner: owner, Usage: Usage{InputTokens: 1_000}, Success: true})
	recorder.Record(hosted, Stat{Owner: owner, Usage: Usage{InputTokens: 1_000}, Success: true})
	recorder.Record(priced, Stat{Owner: owner, Usage: Usage{InputTokens: 1_000}, Free: true, Success: true})

	s.Zero(gate.total(), "free, local and given-away work is no spend")
}

func (s *RecorderSuite) TestTheWritersGateIsTheSpenderOfARecorderWithNone() {
	gate := &spending{}
	writer := NewWriter(WriterOptions{Store: s.store, Gate: gate, Logger: slog.New(slog.DiscardHandler)})
	defer writer.Close(s.ctx)
	recorder := NewRecorderOn(LLM, writer, s.prices, slog.New(slog.DiscardHandler))

	recorder.Record(priced, Stat{Owner: Owner{CustomerID: uuid.NewString(), Attribution: Attribution{APIKey: "key-a"}},
		Usage: Usage{InputTokens: 1_000}, Success: true})

	s.EqualValues(2_000, gate.total())
}

func (s *RecorderSuite) TestTheWriterSaysWhichRowsTheShardHas() {
	var mu sync.Mutex
	told := map[uuid.UUID]int{}
	writer := NewWriter(WriterOptions{Store: s.store, Logger: slog.New(slog.DiscardHandler), Written: func(written []Acked) {
		mu.Lock()
		defer mu.Unlock()
		for _, row := range written {
			told[row.UsageID]++
		}
	}})
	recorder := NewRecorderOn(LLM, writer, s.prices, slog.New(slog.DiscardHandler))
	owner := Owner{CustomerID: uuid.NewString(), Attribution: Attribution{APIKey: "key-a"}}
	first, again := uuid.Must(uuid.NewV7()), uuid.Must(uuid.NewV7())

	recorder.Record(priced, Stat{Owner: owner, UsageID: first, Usage: Usage{InputTokens: 1_000}, Success: true})
	recorder.Record(priced, Stat{Owner: owner, UsageID: again, Usage: Usage{InputTokens: 1_000}, Success: true})
	s.Require().Eventually(func() bool {
		mu.Lock()
		defer mu.Unlock()
		return told[first] == 1 && told[again] == 1
	}, 5*time.Second, 10*time.Millisecond)
	recorder.Record(priced, Stat{Owner: owner, UsageID: again, Usage: Usage{InputTokens: 1_000}, Success: true})
	writer.Close(s.ctx)

	mu.Lock()
	defer mu.Unlock()
	s.Equal(2, told[again], "a row the shard already had is one it has")
	s.Equal(1, told[first])
}
