//go:build integration

package phone

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// MeteringSuite is what the phone service bills, written to real Postgres through a real
// recorder priced from the vendor list: a placed call and a transfer as free dial rows and as
// legs billed by the minute when they end, a leg nothing ended, and each month of a number.
// The vendor is a stub named telnyx, and the service's clock is the suite's.
type MeteringSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	redis    rueidis.Client
	recorder *routing.Recorder
	service  *Service
	vendor   *stub
	clock    *meteringClock
	customer string
	e164     string
}

func TestMeteringSuite(t *testing.T) { suite.Run(t, new(MeteringSuite)) }

func (s *MeteringSuite) SetupSuite() {
	dsn, address := os.Getenv("ROUTER_POSTGRES_DSN"), os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || address == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "phone_metering"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
	s.redis, err = rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(s.redis.Close)
}

func (s *MeteringSuite) SetupTest() {
	s.withSandbox(dlc.Sandbox{})
}

// withSandbox builds the service over a fresh customer, a gate holding them to sandbox, and a
// recorder written out before the test reads.
func (s *MeteringSuite) withSandbox(sandbox dlc.Sandbox) {
	list, err := pricing.NewList("vendor", prices.Vendor().Entries, 0, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	s.recorder = routing.NewRecorder(routing.Phone, s.store, nil, list, slog.New(slog.DiscardHandler))
	s.T().Cleanup(s.recorder.Close)
	s.clock = &meteringClock{at: time.Now().UTC().Truncate(time.Second)}
	s.customer = "customer-" + uuid.NewString()
	s.e164 = "+1512" + uuid.NewString()[:7]
	s.vendor = &stub{vendor: "telnyx", monthly: 1_000_000}
	config, err := DefaultConfig()
	s.Require().NoError(err)
	registry := NewRegistry(config)
	registry.Register(s.vendor.vendor, func() (Provider, error) { return s.vendor, nil })
	declared, _ := config.Lookup(s.vendor.vendor)
	for _, name := range declared.Credentials {
		s.T().Setenv(name, "set")
	}
	deployment := chattest.NewServer(s.T())
	s.service, err = NewService(ServiceOptions{
		Registry: registry, Store: s.store, Recorder: s.recorder, Now: s.clock.now,
		Apps:   &testApps{deployment: deployment, own: map[string]*chattest.Server{}, pins: map[string]int64{}, blocked: map[string]bool{}},
		Gate:   dlc.NewGate(s.store, s.redis, sandbox, nil),
		Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.Require().NoError(s.store.RecordNumber(s.ctx, &store.PhoneNumber{
		E164: s.e164, Vendor: s.vendor.vendor, Country: "US", CustomerID: s.customer, PurchasedAt: s.clock.now(),
	}))
}

func (s *MeteringSuite) owner() routing.Owner {
	return routing.Owner{CustomerID: s.customer, Attribution: routing.Attribution{APIKey: "key-a", UserID: "u1"}}
}

func (s *MeteringSuite) TestAPlacedCallIsBilledForItsMinutesWhenItEnds() {
	placed, err := s.service.Call(s.ctx, CallRequest{Owner: s.owner(), From: s.e164, To: "+15125550123"})
	s.Require().NoError(err)
	s.clock.advance(61 * time.Second)

	s.Require().NoError(s.service.EndLegs(s.ctx, store.AppScope{Unpinned: true}, placed.CallType, placed.CallID))

	dial := s.row(ModelDial)
	s.True(dial.Success)
	s.False(dial.Billable, "placing a call is free: the vendor bills its minutes")
	minutes := s.row(ModelCallMinutes)
	s.Equal(pricing.Quantities{pricing.CallSeconds: 61}, minutes.Meters)
	s.Equal(int64(14_000), minutes.CostMicros, "two started minutes at $0.007")
	s.Equal("key-a", minutes.APIKey, "the minutes keep the key that placed the call")
	s.Equal("u1", minutes.UserID)
	s.Equal(map[string]string{"country": "US", "direction": "outbound", "measured_by": "router"}, minutes.Attributes)
}

func (s *MeteringSuite) TestAFailedDialIsARowThatIsNotBilledAndOpensNoLeg() {
	s.vendor.err = errors.New("the vendor refused")

	_, err := s.service.Call(s.ctx, CallRequest{Owner: s.owner(), From: s.e164, To: "+15125550123"})

	s.Require().Error(err)
	dial := s.row(ModelDial)
	s.False(dial.Success)
	s.Equal("vendor_failed", dial.ErrorCode)
	s.Empty(s.openLegs())
}

func (s *MeteringSuite) TestATransferIsALegOfTheCallItJoins() {
	callID := "call-" + uuid.NewString()
	_, err := s.service.Transfer(s.ctx, TransferRequest{Owner: s.owner(), From: s.e164, To: "+15125550124", CallID: callID})
	s.Require().NoError(err)
	s.clock.advance(3 * time.Minute)

	s.Require().NoError(s.service.EndLegs(s.ctx, store.AppScope{Unpinned: true}, defaultCallType, callID))

	minutes := s.row(ModelCallMinutes)
	s.Equal(pricing.Quantities{pricing.CallSeconds: 180}, minutes.Meters)
	s.Equal(int64(21_000), minutes.CostMicros)
}

func (s *MeteringSuite) TestALegLeftOpenIsClosedAtFourHours() {
	placed, err := s.service.Call(s.ctx, CallRequest{Owner: s.owner(), From: s.e164, To: "+15125550123"})
	s.Require().NoError(err)
	s.clock.advance(MaxCallLength + time.Minute)

	_, err = s.service.SweepLegs(s.ctx)
	s.Require().NoError(err)
	s.Require().NoError(s.service.EndLegs(s.ctx, store.AppScope{Unpinned: true}, placed.CallType, placed.CallID),
		"the session_ended that comes late ends nothing")

	minutes := s.row(ModelCallMinutes)
	s.Equal(pricing.Quantities{pricing.CallSeconds: int64(MaxCallLength / time.Second)}, minutes.Meters)
	s.Equal(int64(240*7_000), minutes.CostMicros)
	s.Len(s.rows(ModelCallMinutes), 1)
}

func (s *MeteringSuite) TestRecordingAnEndedLegAgainBillsNothing() {
	placed, err := s.service.Call(s.ctx, CallRequest{Owner: s.owner(), From: s.e164, To: "+15125550123"})
	s.Require().NoError(err)
	s.clock.advance(time.Minute)
	ended, err := s.store.EndCallLegs(s.ctx, store.AppScope{Unpinned: true}, placed.CallType, placed.CallID, s.clock.now())
	s.Require().NoError(err)
	s.Require().Len(ended, 1)

	s.service.settleLeg(s.ctx, ended[0])
	s.service.settleLeg(s.ctx, ended[0])

	s.row(ModelCallMinutes)
	s.Len(s.rows(ModelCallMinutes), 1, "the leg's usage id names it")
}

func (s *MeteringSuite) TestEveryLegOfAnEndedCallCountsTowardTheSandboxsMinutes() {
	s.withSandbox(dlc.Sandbox{Enabled: true})
	s.Require().NoError(s.service.gate.SetRecipients(s.ctx, s.customer, []string{"+15125550123", "+15125550124"}))
	placed, err := s.service.Call(s.ctx, CallRequest{Owner: s.owner(), From: s.e164, To: "+15125550123"})
	s.Require().NoError(err)
	s.clock.advance(4 * time.Minute)
	_, err = s.service.Transfer(s.ctx, TransferRequest{Owner: s.owner(), From: s.e164, To: "+15125550124", CallID: placed.CallID})
	s.Require().NoError(err)
	s.clock.advance(6 * time.Minute)

	s.Require().NoError(s.service.EndLegs(s.ctx, store.AppScope{Unpinned: true}, placed.CallType, placed.CallID))

	usage, err := s.service.gate.Usage(s.ctx, s.customer)
	s.Require().NoError(err)
	s.Equal(int64(16*60), usage.AudioSeconds, "both legs, not only the one that began first")
	for _, minutes := range s.rowsWhen(ModelCallMinutes, 2) {
		s.False(minutes.Billable, "the sandbox gives its minutes away")
		s.Equal("true", minutes.Attributes[AttributeDLCSandbox])
		s.NotEmpty(minutes.Attributes[routing.AttributeFreeCostMicros])
	}
}

func (s *MeteringSuite) TestABoughtNumberIsBilledItsFirstMonth() {
	bought, err := s.service.Buy(s.ctx, Purchase{Vendor: s.vendor.vendor, E164: "+1512" + uuid.NewString()[:7], Owner: s.owner()})
	s.Require().NoError(err)

	month := s.row(ModelNumber)
	s.Equal(int64(1_000_000), month.CostMicros, "the price the vendor quoted")
	s.True(month.Billable)
	s.Equal("key-a", month.APIKey, "the month it was bought in is the buyer's")
	s.Equal(bought.PurchasedAt, month.StartedAt.UTC())
}

func (s *MeteringSuite) TestANumberIsBilledEachMonthItIsHeld() {
	bought, err := s.service.Buy(s.ctx, Purchase{Vendor: s.vendor.vendor, E164: "+1512" + uuid.NewString()[:7], Owner: s.owner()})
	s.Require().NoError(err)
	s.row(ModelNumber)
	s.clock.set(monthAfter(bought.PurchasedAt, 1).Add(time.Hour))

	_, err = s.service.BillRentals(s.ctx)
	s.Require().NoError(err)
	_, err = s.service.BillRentals(s.ctx)
	s.Require().NoError(err)

	months := s.rowsWhen(ModelNumber, 2)
	s.Len(months, 2, "the month it was bought in and the next, each once")
	s.Equal(OriginNumberRental, months[1].APIKey)
	s.Equal(monthAfter(bought.PurchasedAt, 1), months[1].StartedAt.UTC())

	s.Require().NoError(s.store.ReleaseNumber(s.ctx, s.customer, bought.E164, monthAfter(bought.PurchasedAt, 1).Add(2*time.Hour)))
	s.clock.set(monthAfter(bought.PurchasedAt, 2).Add(time.Hour))
	_, err = s.service.BillRentals(s.ctx)
	s.Require().NoError(err)
	time.Sleep(500 * time.Millisecond)
	s.Len(s.rows(ModelNumber), 2, "a number released is not billed again")
}

// row waits for the customer's one row of a model and reads it.
func (s *MeteringSuite) row(model string) store.Request {
	return s.rowsWhen(model, 1)[0]
}

// rowsWhen waits until the customer has want rows of a model, the recorder writing them off the
// caller, and reads them oldest first.
func (s *MeteringSuite) rowsWhen(model string, want int) []store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = s.rows(model)
		return len(rows) >= want
	}, 5*time.Second, 25*time.Millisecond, "no %d %s rows", want, model)
	return rows
}

func (s *MeteringSuite) rows(model string) []store.Request {
	var rows []store.Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).
		Where("customer_id = ?", s.customer).Where("modality = ?", routing.Phone).Where("model = ?", model).
		Order("started_at").Scan(s.ctx))
	return rows
}

func (s *MeteringSuite) openLegs() []store.CallLeg {
	var legs []store.CallLeg
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&legs).
		Where("customer_id = ?", s.customer).Where("ended_at IS NULL").Scan(s.ctx))
	return legs
}

// meteringClock is the service's clock, moved by the test.
type meteringClock struct {
	mu sync.Mutex
	at time.Time
}

func (c *meteringClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.at
}

func (c *meteringClock) advance(by time.Duration) { c.set(c.now().Add(by)) }

func (c *meteringClock) set(at time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.at = at
}
