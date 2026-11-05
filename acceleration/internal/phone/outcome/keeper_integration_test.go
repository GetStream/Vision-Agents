//go:build integration

package outcome

import (
	"bytes"
	"context"
	"crypto/rand"
	"errors"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// KeeperSuite keeps calls in a real Postgres. Each test places its calls under a customer of
// its own, so tests never see each other's rows, at a vendor of its own, so it reads only the
// calls it counted.
type KeeperSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	keeper *Keeper
	now    time.Time
	call   Call
	vendor string
	// logs is what the keeper logged in the test.
	logs *bytes.Buffer
	// before is what each of the vendor's series held when the test began.
	before map[[2]string]float64
}

func TestKeeperSuite(t *testing.T) { suite.Run(t, new(KeeperSuite)) }

func (s *KeeperSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "phone_outcome"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

func (s *KeeperSuite) SetupTest() {
	s.vendor = "vendor-" + rand.Text()
	s.logs = &bytes.Buffer{}
	s.keeper = NewKeeper(s.store, func() time.Time { return s.now }, []string{s.vendor}, slog.New(slog.NewTextHandler(s.logs, nil)))
	s.before = map[[2]string]float64{}
	for _, direction := range []string{inbound, outbound} {
		for _, label := range allLabels {
			s.before[[2]string{direction, label}] = testutil.ToFloat64(callOutcomes.WithLabelValues(s.vendor, direction, label))
		}
	}
	s.now = time.Now().UTC().Truncate(time.Second)
	s.call = Call{
		Key: Key{
			CustomerID: "customer-" + uuid.NewString(),
			CallType:   "default",
			CallID:     "call-" + uuid.NewString(),
			Local:      "+15125550100",
			Remote:     "+15125550199",
		},
		Vendor:  s.vendor,
		Reports: true,
	}
}

// allLabels is every outcome a call is counted under.
var allLabels = []string{string(Completed), string(Busy), string(NoAnswer), string(Failed), string(Canceled),
	string(Unknown), dropped}

// counted is every series of the test's vendor that grew in the test, by direction and outcome.
func (s *KeeperSuite) counted() map[[2]string]float64 {
	grew := map[[2]string]float64{}
	for series, before := range s.before {
		if now := testutil.ToFloat64(callOutcomes.WithLabelValues(s.vendor, series[0], series[1])); now != before {
			grew[series] = now - before
		}
	}
	return grew
}

// countedOnce checks the test counted one call, under direction and label.
func (s *KeeperSuite) countedOnce(direction, label string) {
	s.Equal(map[[2]string]float64{{direction, label}: 1}, s.counted())
}

// at is the suite's clock and n seconds.
func (s *KeeperSuite) at(n int) time.Time { return s.now.Add(time.Duration(n) * time.Second) }

// placedAnew places the suite's call into a Stream call of its own and returns its token.
func (s *KeeperSuite) placedAnew() string {
	s.call.CallID = "call-" + uuid.NewString()
	return s.placed()
}

func (s *KeeperSuite) reported(token string, reported Outcome) {
	_, err := s.keeper.Reported(s.ctx, token, reported, string(reported))
	s.Require().NoError(err)
}

func (s *KeeperSuite) answered(at time.Time) {
	s.Require().NoError(s.keeper.Answered(s.ctx, s.call.Key, at))
}

// ended ends the placed call's leg at at.
func (s *KeeperSuite) ended(at time.Time) {
	s.Require().NoError(s.keeper.Ended(s.ctx, s.call.Key, at, Leg{Vendor: s.vendor}))
}

// sessionClosed keeps the router's session of the suite's call, closed at at for reason.
func (s *KeeperSuite) sessionClosed(at time.Time, reason string) {
	s.Require().NoError(s.store.SaveSession(s.ctx, &store.AgentSession{
		ID: s.call.CallID, CustomerID: s.call.CustomerID, CallType: s.call.CallType, CallID: s.call.CallID,
	}))
	s.Require().NoError(s.store.CloseSession(s.ctx, s.call.CallID, at, reason))
}

func (s *KeeperSuite) placed() string {
	token, err := s.keeper.Placed(s.ctx, s.call)
	s.Require().NoError(err)
	return token
}

func (s *KeeperSuite) TestAPlacedCallIsReadBack() {
	token := s.placed()
	s.Require().NoError(s.keeper.Dialed(s.ctx, token, "vendor-call"))

	call, err := s.keeper.Read(s.ctx, s.call.CustomerID, s.call.CallID)
	s.Require().NoError(err)
	s.Equal(s.call.Key, call.Key)
	s.Equal(s.vendor, call.Vendor)
	s.True(call.Reports)
	s.Equal("vendor-call", call.VendorCallID)
	s.Equal(s.now, call.PlacedAt.UTC())
	s.Empty(call.Outcome)
}

func (s *KeeperSuite) TestAFailedCallLeavesNothing() {
	token := s.placed()

	s.Require().NoError(s.keeper.Failed(s.ctx, token))

	_, err := s.keeper.Read(s.ctx, s.call.CustomerID, s.call.CallID)
	s.ErrorIs(err, ErrUnknownCall)
}

func (s *KeeperSuite) TestAReportOnAnUnknownTokenIsUnknown() {
	_, err := s.keeper.Reported(s.ctx, uuid.NewString(), Busy, "busy")

	s.ErrorIs(err, ErrUnknownCall)
}

func (s *KeeperSuite) TestAJoinOnACallWithNoRowIsNothing() {
	s.NoError(s.keeper.Answered(s.ctx, s.call.Key, s.now))
	s.NoError(s.keeper.Ended(s.ctx, s.call.Key, s.now, Leg{Vendor: s.vendor}))
}

func (s *KeeperSuite) TestAKeyMatchesByItsOwnNumbersOnly() {
	s.placed()
	transfer := s.call.Key
	transfer.Remote = "+15125550142"

	s.Require().NoError(s.keeper.Answered(s.ctx, transfer, s.now))

	call, err := s.keeper.Read(s.ctx, s.call.CustomerID, s.call.CallID)
	s.Require().NoError(err)
	s.Nil(call.AnsweredAt)
}

func (s *KeeperSuite) TestASweptTokenIsUnknownAndTheOutcomeStays() {
	token := s.placed()
	_, err := s.keeper.Reported(s.ctx, token, Busy, "busy")
	s.Require().NoError(err)

	swept, err := s.keeper.Sweep(s.ctx, s.now.Add(time.Second))
	s.Require().NoError(err)
	s.Positive(swept)

	_, err = s.keeper.ByToken(s.ctx, token)
	s.ErrorIs(err, ErrUnknownCall)
	call, err := s.keeper.Read(s.ctx, s.call.CustomerID, s.call.CallID)
	s.Require().NoError(err)
	s.Equal(Busy, call.Outcome)
}

func (s *KeeperSuite) TestReadManyOmitsCallsWithNoRow() {
	s.placed()

	calls, err := s.keeper.ReadMany(s.ctx, s.call.CustomerID, []string{s.call.CallID, "never placed"})
	s.Require().NoError(err)
	s.Len(calls, 1)
	s.Equal(s.call.Key, calls[s.call.CallID].Key)
}

func (s *KeeperSuite) TestABusyCallIsCountedOnceWhenItsLegEnds() {
	token := s.placed()
	s.reported(token, Busy)
	s.Empty(s.counted(), "the phone may still join")

	s.ended(s.at(1))
	s.ended(s.at(2))

	s.countedOnce(outbound, string(Busy))
}

func (s *KeeperSuite) TestAReportAfterTheEndCountsNothingMore() {
	token := s.placed()
	s.ended(s.at(1))

	s.reported(token, Busy)

	s.countedOnce(outbound, string(NoAnswer))
}

func (s *KeeperSuite) TestAnAnsweredCallIsCountedOnceAsCompletedWhateverTheOrderOfItsEvents() {
	orders := map[string]func(token string){
		"reported, answered, ended": func(token string) { s.reported(token, Completed); s.answered(s.at(1)); s.ended(s.at(2)) },
		"answered, reported, ended": func(token string) { s.answered(s.at(1)); s.reported(token, NoAnswer); s.ended(s.at(2)) },
		"busy, answered, ended":     func(token string) { s.reported(token, Busy); s.answered(s.at(1)); s.ended(s.at(2)) },
		"answered, ended, reported": func(token string) { s.answered(s.at(1)); s.ended(s.at(2)); s.reported(token, Busy) },
		"answered, ended twice":     func(string) { s.answered(s.at(1)); s.ended(s.at(2)); s.ended(s.at(3)) },
	}
	for name, order := range orders {
		order(s.placedAnew())
		call, err := s.keeper.Read(s.ctx, s.call.CustomerID, s.call.CallID)
		s.Require().NoError(err)
		s.Equal(Completed, call.Outcome, name)
	}

	s.Equal(map[[2]string]float64{{outbound, string(Completed)}: float64(len(orders))}, s.counted())
}

func (s *KeeperSuite) TestALegEndingTwiceAtOnceIsCountedOnce() {
	s.placed()
	s.answered(s.at(1))

	var ending sync.WaitGroup
	errs := make([]error, 8)
	for i := range errs {
		ending.Go(func() { errs[i] = s.keeper.Ended(s.ctx, s.call.Key, s.at(2), Leg{Vendor: s.vendor}) })
	}
	ending.Wait()
	s.Require().NoError(errors.Join(errs...))

	s.countedOnce(outbound, string(Completed))
}

func (s *KeeperSuite) TestACallWhoseAgentLeftItOnTheLineIsCountedDroppedAndStoredCompleted() {
	s.placed()
	s.answered(s.at(1))
	s.sessionClosed(s.at(5), "")

	s.ended(s.at(9))

	s.countedOnce(outbound, dropped)
	call, err := s.keeper.Read(s.ctx, s.call.CustomerID, s.call.CallID)
	s.Require().NoError(err)
	s.Equal(Completed, call.Outcome, "what is stored stays as it was")
}

func (s *KeeperSuite) TestACallItsAgentHungUpIsCountedCompleted() {
	s.placed()
	s.answered(s.at(1))
	s.sessionClosed(s.at(5), "agent_hangup")

	s.ended(s.at(6))

	s.countedOnce(outbound, string(Completed))
}

func (s *KeeperSuite) TestACallWhoseSessionClosedAfterThePhoneLeftIsCountedCompleted() {
	s.placed()
	s.answered(s.at(1))
	s.sessionClosed(s.at(9), "")

	s.ended(s.at(5))

	s.countedOnce(outbound, string(Completed))
}

func (s *KeeperSuite) TestAnUnansweredCallIsNeverCountedDropped() {
	s.placed()
	s.sessionClosed(s.at(1), "")

	s.ended(s.at(5))

	s.countedOnce(outbound, string(NoAnswer))
}

func (s *KeeperSuite) TestAVendorTheKeeperDoesNotNameIsCountedAsOther() {
	other := testutil.ToFloat64(callOutcomes.WithLabelValues(otherVendor, outbound, string(NoAnswer)))
	s.call.Vendor = "unnamed-" + rand.Text()
	s.placed()

	s.ended(s.at(1))

	s.Equal(1.0, testutil.ToFloat64(callOutcomes.WithLabelValues(otherVendor, outbound, string(NoAnswer)))-other)
	s.Zero(testutil.ToFloat64(callOutcomes.WithLabelValues(s.call.Vendor, outbound, string(NoAnswer))))
}

// inboundEnded ends the leg of a call that arrived on the suite's numbers, answered at answered.
func (s *KeeperSuite) inboundEnded(answered *time.Time, at time.Time) {
	leg := Leg{Vendor: s.vendor, Inbound: true, AnsweredAt: answered}
	s.Require().NoError(s.keeper.Ended(s.ctx, s.call.Key, at, leg))
}

func (s *KeeperSuite) TestAnInboundCallIsCountedWhenItsLegEnds() {
	s.inboundEnded(new(s.at(1)), s.at(5))

	s.countedOnce(inbound, string(Completed))
	s.Empty(s.logs.String(), "a call with no router session, as a worker answers, is no failure")
}

func (s *KeeperSuite) TestAnInboundCallItsAgentLeftOnTheLineIsCountedDropped() {
	s.sessionClosed(s.at(3), "")

	s.inboundEnded(new(s.at(1)), s.at(5))

	s.countedOnce(inbound, dropped)
}

// total is every call counted, at any vendor, read off the registry /metrics serves. No other
// test in this package counts calls while a suite runs.
func (s *KeeperSuite) total() float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	var total float64
	for _, family := range families {
		if family.GetName() == "router_phone_call_outcomes_total" {
			for _, metric := range family.GetMetric() {
				total += metric.GetCounter().GetValue()
			}
		}
	}
	return total
}

func (s *KeeperSuite) TestATransfersLegIsNotCounted() {
	before := s.total()
	s.answered(s.at(1))

	s.ended(s.at(5))

	s.Zero(s.total() - before)
}

func (s *KeeperSuite) TestAStoreThatFailsFailsTheEndAndCountsNothing() {
	broken, err := store.Open(testenv.Database(os.Getenv("ROUTER_POSTGRES_DSN"), "phone_outcome"))
	s.Require().NoError(err)
	s.Require().NoError(broken.Close())
	keeper := NewKeeper(broken, func() time.Time { return s.now }, []string{s.vendor}, slog.New(slog.DiscardHandler))
	before := s.total()

	err = keeper.Ended(s.ctx, s.call.Key, s.at(1), Leg{Vendor: s.vendor})

	s.Error(err)
	s.NotErrorIs(err, store.ErrNoPhoneCall)
	s.Zero(s.total() - before)
}

// TestAnInboundCallTheRouterRefusedIsCountedFailed catches a refused arriving call, which no
// agent answered, counted completed.
func (s *KeeperSuite) TestAnInboundCallTheRouterRefusedIsCountedFailed() {
	leg := Leg{Vendor: s.vendor, Inbound: true, AnsweredAt: new(s.at(1)), Refused: true}

	s.Require().NoError(s.keeper.Ended(s.ctx, s.call.Key, s.at(5), leg))

	s.countedOnce(inbound, string(Failed))
}

func (s *KeeperSuite) TestASessionThatCannotBeReadLeavesTheCallCountedAsStored() {
	s.call.CustomerID = ""

	s.inboundEnded(new(s.at(1)), s.at(5))

	s.countedOnce(inbound, string(Completed))
	s.Contains(s.logs.String(), "could not tell whether a phone call was dropped")
}
