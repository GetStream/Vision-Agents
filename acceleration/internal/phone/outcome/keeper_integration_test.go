//go:build integration

package outcome

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// KeeperSuite keeps calls in a real Postgres. Each test places its calls under a customer of
// its own, so tests never see each other's rows.
type KeeperSuite struct {
	suite.Suite
	ctx    context.Context
	store  *store.Store
	keeper *Keeper
	now    time.Time
	call   Call
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
	s.keeper = NewKeeper(opened, func() time.Time { return s.now })
}

func (s *KeeperSuite) SetupTest() {
	s.now = time.Now().UTC().Truncate(time.Second)
	s.call = Call{
		Key: Key{
			CustomerID: "customer-" + uuid.NewString(),
			CallType:   "default",
			CallID:     "call-" + uuid.NewString(),
			Local:      "+15125550100",
			Remote:     "+15125550199",
		},
		Vendor:  "telnyx",
		Reports: true,
	}
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
	s.Equal("telnyx", call.Vendor)
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
	s.NoError(s.keeper.Ended(s.ctx, s.call.Key, s.now))
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
