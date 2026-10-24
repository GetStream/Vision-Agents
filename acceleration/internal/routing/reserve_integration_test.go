//go:build integration

package routing_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// reservedModel is billed $2 per million input tokens.
var reservedModel = routing.ProviderConfig{Provider: "vendor", Model: "reserved", Price: routing.Price{PerMillionInputTokens: 2}}

// worked is what a vendor says its work used.
type worked pricing.Quantities

func (w worked) Used() pricing.Quantities { return pricing.Quantities(w) }

// ReserveSuite calls vendors through routing.Call with the real enforcer as the gate, reading
// money from the real usage service, and the real writer into Postgres.
type ReserveSuite struct {
	suite.Suite
	ctx     context.Context
	store   *store.Store
	service *usagetest.Service
	prices  *pricing.List

	enforcer *policy.Enforcer
	recorder *routing.Recorder
	app, org string
}

func TestReserveSuite(t *testing.T) {
	suite.Run(t, new(ReserveSuite))
}

func (s *ReserveSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "routing_reserve"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
	discard := slog.New(slog.NewTextHandler(io.Discard, nil))
	s.service = usagetest.Start(s.T(), usagetest.Options{
		DSN: usagetest.Database(s.T(), "usage_routing_reserve"), ManageKey: "manage-" + uuid.NewString(), Logger: discard,
	})
	s.prices, err = routing.Prices(routing.Config{routing.LLM: {Providers: []routing.ProviderConfig{reservedModel}}}, discard)
	s.Require().NoError(err)
}

func (s *ReserveSuite) SetupTest() {
	stamp := time.Now().UnixNano()
	s.app, s.org = fmt.Sprintf("app-%d", stamp), fmt.Sprintf("org-%d", stamp)
	s.Require().NoError(s.store.JoinOrganization(s.ctx, s.app, s.org))
	configs, err := appconfig.New(appconfig.Options{Store: s.store})
	s.Require().NoError(err)
	client, err := usage.NewClient(usage.ClientOptions{URL: s.service.URL, Token: s.service.Token})
	s.Require().NoError(err)
	discard := slog.New(slog.NewTextHandler(io.Discard, nil))
	s.enforcer, err = policy.New(configs, policy.Settings{
		Usage: client, Fresh: 50 * time.Millisecond, SpendEvery: 20 * time.Millisecond, SumEvery: 50 * time.Millisecond,
		MaxStale: time.Minute, Idle: time.Minute, PendingTTL: time.Minute,
	}, discard)
	s.Require().NoError(err)
	s.T().Cleanup(s.enforcer.Close)
	writer := routing.NewWriter(routing.WriterOptions{Store: s.store, Logger: discard, Gate: s.enforcer})
	s.T().Cleanup(func() { writer.Close(context.Background()) })
	s.recorder = routing.NewRecorderOn(routing.LLM, writer, s.prices, discard)
}

// owner is the app's work in its organization.
func (s *ReserveSuite) owner() routing.Owner {
	return routing.Owner{CustomerID: s.app, Attribution: routing.Attribution{OrganizationID: s.org}}
}

// credit gives the organization micros and waits until the enforcer has read them.
func (s *ReserveSuite) credit(micros int64) {
	if micros > 0 {
		s.service.Credit(s.T(), s.org, micros)
	}
	s.enforcer.Join(s.app, s.org, usage.DefaultTier)
	// An account never read admits; once read, holding all of its credit leaves none.
	probe := uuid.Must(uuid.NewV7())
	s.Eventually(func() bool {
		if micros > 0 {
			s.Require().NoError(s.enforcer.Reserve(s.owner(), probe, micros, time.Minute))
			defer s.enforcer.Release(probe)
		}
		admitted, err := s.enforcer.Admit(s.ctx, s.owner())
		return err == nil && errors.Is(admitted.Spend, policy.ErrNoCredits)
	}, 5*time.Second, 5*time.Millisecond, "the balance is read")
}

func (s *ReserveSuite) meter() routing.Meter {
	return routing.Meter{Recorder: s.recorder, Config: reservedModel, Owner: s.owner()}
}

// spend is why the organization's billable work is refused now, nil when it is not.
func (s *ReserveSuite) spend() error {
	admitted, err := s.enforcer.Admit(s.ctx, s.owner())
	s.Require().NoError(err)
	return admitted.Spend
}

// rows is what the app's calls stored, once the writer has them.
func (s *ReserveSuite) rows(want int) []store.Request {
	var rows []store.Request
	s.Eventually(func() bool {
		rows = nil
		s.Require().NoError(s.store.DB().NewSelect().Model(&rows).Where("customer_id = ?", s.app).Scan(s.ctx))
		return len(rows) == want
	}, 5*time.Second, 10*time.Millisecond)
	return rows
}

func (s *ReserveSuite) TestCallReservesBeforeTheVendorAndSettlesAfter() {
	s.credit(2_000_000)
	meter := s.meter()
	meter.Estimate = pricing.Quantities{pricing.InputTokens: 1_000_000}
	var estimate routing.Stat
	estimate.Use(meter.Estimate)
	s.Require().True(s.recorder.Billable(reservedModel))
	s.Require().EqualValues(2_000_000, s.recorder.QuoteStat(reservedModel, estimate).CostMicros)
	var during error

	_, err := routing.Call(s.ctx, meter, func(context.Context) (worked, error) {
		during = s.spend()
		return worked{pricing.InputTokens: 250_000}, nil
	})

	s.Require().NoError(err)
	s.ErrorIs(during, policy.ErrNoCredits, "the vendor works with its $2 estimate already held")
	s.NoError(s.spend(), "settled at the $0.50 it cost")
	row := s.rows(1)[0]
	s.EqualValues(500_000, row.CostMicros)
}

func (s *ReserveSuite) TestARefusedReservationCallsNoVendor() {
	s.credit(0)
	meter := s.meter()
	meter.Estimate = pricing.Quantities{pricing.InputTokens: 1_000}
	asked := false

	_, err := routing.Call(s.ctx, meter, func(context.Context) (worked, error) {
		asked = true
		return worked{pricing.InputTokens: 1_000}, nil
	})

	s.ErrorIs(err, policy.ErrNoCredits)
	s.True(routing.ReservationRefused(err), "no other vendor is to be asked")
	s.False(asked, "no vendor was asked")
	time.Sleep(100 * time.Millisecond)
	s.Empty(s.rows(0), "and no row recorded")
}

func (s *ReserveSuite) TestAFailedCallReleasesItsReservation() {
	s.credit(2_000_000)
	meter := s.meter()
	meter.Estimate = pricing.Quantities{pricing.InputTokens: 1_000_000}

	_, err := routing.Call(s.ctx, meter, func(context.Context) (worked, error) {
		return nil, errors.New("vendor down")
	})

	s.Require().Error(err)
	s.NoError(s.spend(), "a failure bills nothing, so nothing is held for it")
	s.False(s.rows(1)[0].Success)
}

func (s *ReserveSuite) TestWorkThatNamesItsOwnIDReleasesTheReservation() {
	s.credit(2_000_000)
	meter := s.meter()
	meter.Estimate = pricing.Quantities{pricing.InputTokens: 1_000_000}
	own := uuid.Must(uuid.NewV7())

	_, err := routing.Call(s.ctx, meter, func(context.Context) (identified, error) {
		return identified{worked: worked{pricing.InputTokens: 250_000}, id: own}, nil
	})

	s.Require().NoError(err)
	s.NoError(s.spend(), "held once, under the work's own id")
	s.Equal(own, s.rows(1)[0].UsageID)
}

func (s *ReserveSuite) TestARecorderThatRecordsNothingReservesNothing() {
	s.credit(0)
	discard := slog.New(slog.NewTextHandler(io.Discard, nil))
	// The gate, and no store or live counters: no row it records would ever settle a hold.
	writer := routing.NewWriter(routing.WriterOptions{Logger: discard, Gate: s.enforcer})
	s.T().Cleanup(func() { writer.Close(context.Background()) })
	recorder := routing.NewRecorderOn(routing.LLM, writer, s.prices, discard)

	err := recorder.Reserve(reservedModel, s.owner(), uuid.Must(uuid.NewV7()), pricing.Quantities{pricing.InputTokens: 1_000}, time.Minute)

	s.NoError(err, "nothing was asked of the gate")
	s.Nil(routing.StartLease(recorder, reservedModel, s.owner(), "test", uuid.NewString(), pricing.Quantities{pricing.InputTokens: 1_000}))
}

func (s *ReserveSuite) TestALeaseHoldsAMinuteUntilItStops() {
	s.credit(2_000_000)

	lease := routing.StartLease(s.recorder, reservedModel, s.owner(), "test", uuid.NewString(), pricing.Quantities{pricing.InputTokens: 1_000_000})

	s.Require().NotNil(lease)
	s.NoError(lease.Err())
	s.ErrorIs(s.spend(), policy.ErrNoCredits, "the stream's minute is held")
	lease.Stop()
	s.NoError(s.spend(), "and let go when it ends")
}

// identified is work that names its own usage id.
type identified struct {
	worked
	id uuid.UUID
}

func (i identified) UsageID() uuid.UUID { return i.id }
