//go:build integration

package concurrency

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// AddressEnvVar is where the tests look for a Redis to run against.
const AddressEnvVar = "ROUTER_REDIS_ADDR"

type SlotsSuite struct {
	suite.Suite
	redis rueidis.Client
	ctx   context.Context
	// key is unique per test so parallel runs cannot collide, and no test cleans up.
	key string
	// clock is what the suite's Slots read as now.
	clock time.Time
	slots *Slots
}

func TestSlotsSuite(t *testing.T) {
	suite.Run(t, new(SlotsSuite))
}

func (s *SlotsSuite) SetupSuite() {
	address := os.Getenv(AddressEnvVar)
	if address == "" {
		s.T().Skipf("%s not set", AddressEnvVar)
	}
	client, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.redis = client
	s.ctx = context.Background()
	s.Require().NoError(client.Do(s.ctx, client.B().Ping().Build()).Error())
}

func (s *SlotsSuite) TearDownSuite() {
	if s.redis != nil {
		s.redis.Close()
	}
}

func (s *SlotsSuite) SetupTest() {
	s.key = "test:slots:" + uuid.NewString()
	s.clock = time.Now()
	s.slots = New(s.redis, func() time.Time { return s.clock })
}

func (s *SlotsSuite) TestSlotsUpToTheLimitAreTaken() {
	_, first := s.slots.Take(s.ctx, s.key, 2, time.Minute)
	_, second := s.slots.Take(s.ctx, s.key, 2, time.Minute)

	s.True(first)
	s.True(second)
}

func (s *SlotsSuite) TestASlotOverTheLimitIsRefused() {
	s.slots.Take(s.ctx, s.key, 1, time.Minute)

	_, taken := s.slots.Take(s.ctx, s.key, 1, time.Minute)

	s.False(taken)
}

func (s *SlotsSuite) TestAReleasedSlotIsFreeAgain() {
	release, _ := s.slots.Take(s.ctx, s.key, 1, time.Minute)
	release()

	_, taken := s.slots.Take(s.ctx, s.key, 1, time.Minute)

	s.True(taken)
}

func (s *SlotsSuite) TestASlotWhoseLeaseEndedIsFreeAgain() {
	s.slots.Take(s.ctx, s.key, 1, time.Minute)
	s.clock = s.clock.Add(time.Minute + time.Second)

	_, taken := s.slots.Take(s.ctx, s.key, 1, time.Minute)

	s.True(taken, "a router that died holding the slot does not hold it forever")
}

func (s *SlotsSuite) TestAnotherKeysSlotsDoNotCount() {
	s.slots.Take(s.ctx, s.key, 1, time.Minute)

	_, taken := s.slots.Take(s.ctx, s.key+":other", 1, time.Minute)

	s.True(taken)
}

func (s *SlotsSuite) TestNoLimitTakesEverySlot() {
	s.slots.Take(s.ctx, s.key, 0, time.Minute)

	_, taken := s.slots.Take(s.ctx, s.key, 0, time.Minute)

	s.True(taken)
}

func (s *SlotsSuite) TestAHeldSlotOutlivesItsLease() {
	slots := New(s.redis, nil)
	held, taken := slots.Hold(s.ctx, s.key, 1, 300*time.Millisecond)
	s.Require().True(taken)
	defer held.Release()

	s.Never(func() bool {
		release, taken := slots.Take(s.ctx, s.key, 1, time.Minute)
		release()
		return taken
	}, time.Second, 50*time.Millisecond, "a slot is held for as long as its work runs, however long its lease")
}

func (s *SlotsSuite) TestAReleasedHeldSlotIsFreeAgain() {
	held, _ := s.slots.Hold(s.ctx, s.key, 1, time.Minute)
	held.Release()
	held.Release()

	_, taken := s.slots.Take(s.ctx, s.key, 1, time.Minute)

	s.True(taken)
}

func (s *SlotsSuite) TestAHeldSlotOverTheLimitIsRefused() {
	s.slots.Take(s.ctx, s.key, 1, time.Minute)

	held, taken := s.slots.Hold(s.ctx, s.key, 1, time.Minute)

	s.False(taken)
	held.Release()
}
