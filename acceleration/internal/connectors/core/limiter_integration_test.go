//go:build integration

package core_test

import (
	"context"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// LimiterSuite runs the Limiter against Redis: how long a provider's 429 holds a key, and
// that every router on the same Redis holds it.
type LimiterSuite struct {
	suite.Suite
	ctx   context.Context
	redis rueidis.Client
	clock *limiterClock
	// key is unique per test, so no test reads another's block.
	key string
}

func TestLimiterSuite(t *testing.T) {
	suite.Run(t, new(LimiterSuite))
}

func (s *LimiterSuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR is not set")
	}
	s.ctx = context.Background()
	s.redis = s.connect(address)
}

func (s *LimiterSuite) SetupTest() {
	// Whole milliseconds, which is what a block keeps.
	s.clock = &limiterClock{now: time.Now().Truncate(time.Millisecond)}
	s.key = "test:" + uuid.NewString()
}

// connect is a client of its own, as each router has.
func (s *LimiterSuite) connect(address string) rueidis.Client {
	client, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	return client
}

func (s *LimiterSuite) TestAKeyIsHeldUntilTheRetryAfterPasses() {
	limiter := core.NewLimiter(s.redis, s.clock.Now)

	s.Equal(30*time.Second, limiter.Block(s.ctx, s.key, 30*time.Second))

	s.Equal(30*time.Second, limiter.Wait(s.ctx, s.key))
	s.clock.Add(29 * time.Second)
	s.Equal(time.Second, limiter.Wait(s.ctx, s.key))
	s.clock.Add(time.Second)
	s.Zero(limiter.Wait(s.ctx, s.key))
}

func (s *LimiterSuite) TestAnotherKeyIsNotHeld() {
	limiter := core.NewLimiter(s.redis, s.clock.Now)

	s.Equal(30*time.Second, limiter.Block(s.ctx, s.key, 30*time.Second))

	s.Zero(limiter.Wait(s.ctx, s.key+":other"))
}

// TestTwoRoutersOnOneRedisHoldTheSameKey: the block is in Redis, not in the router that saw
// the 429.
func (s *LimiterSuite) TestTwoRoutersOnOneRedisHoldTheSameKey() {
	first := core.NewLimiter(s.redis, s.clock.Now)
	second := core.NewLimiter(s.connect(os.Getenv("ROUTER_REDIS_ADDR")), s.clock.Now)

	s.Equal(30*time.Second, first.Block(s.ctx, s.key, 30*time.Second))

	s.Equal(30*time.Second, second.Wait(s.ctx, s.key))
}

// TestAShorterWaitNeverShortensABlock: two routers that see 429s at once keep the longer
// wait, whichever writes last.
func (s *LimiterSuite) TestAShorterWaitNeverShortensABlock() {
	limiter := core.NewLimiter(s.redis, s.clock.Now)

	s.Equal(30*time.Second, limiter.Block(s.ctx, s.key, 30*time.Second))
	s.Equal(10*time.Second, limiter.Block(s.ctx, s.key, 10*time.Second))

	s.Equal(30*time.Second, limiter.Wait(s.ctx, s.key))
}

func (s *LimiterSuite) TestALongerWaitLengthensABlock() {
	limiter := core.NewLimiter(s.redis, s.clock.Now)

	s.Equal(10*time.Second, limiter.Block(s.ctx, s.key, 10*time.Second))
	s.Equal(30*time.Second, limiter.Block(s.ctx, s.key, 30*time.Second))

	s.Equal(30*time.Second, limiter.Wait(s.ctx, s.key))
}

// TestTheBlockLeavesRedisWhenItEnds: the key lives no longer than the wait.
func (s *LimiterSuite) TestTheBlockLeavesRedisWhenItEnds() {
	limiter := core.NewLimiter(s.redis, nil)

	s.Equal(30*time.Second, limiter.Block(s.ctx, s.key, 30*time.Second))

	ttl, err := s.redis.Do(s.ctx, s.redis.B().Pttl().Key(s.key).Build()).AsInt64()
	s.Require().NoError(err)
	s.InDelta(30000, ttl, 1000)
}

// TestAWaitPastMaxBlockIsHeldForMaxBlock: a Retry-After of 2^32-1 seconds, the most the schemes
// parse, holds the key for MaxBlock, in the block and in Redis.
func (s *LimiterSuite) TestAWaitPastMaxBlockIsHeldForMaxBlock() {
	limiter := core.NewLimiter(s.redis, s.clock.Now)

	s.Equal(core.MaxBlock, limiter.Block(s.ctx, s.key, (1<<32-1)*time.Second))

	s.Equal(core.MaxBlock, limiter.Wait(s.ctx, s.key))
	ttl, err := s.redis.Do(s.ctx, s.redis.B().Pttl().Key(s.key).Build()).AsInt64()
	s.Require().NoError(err)
	s.InDelta(core.MaxBlock.Milliseconds(), ttl, 1000)
}

func (s *LimiterSuite) TestNoWaitHoldsNothing() {
	limiter := core.NewLimiter(s.redis, s.clock.Now)

	s.Zero(limiter.Block(s.ctx, s.key, 0))

	s.Zero(limiter.Wait(s.ctx, s.key))
}

func (s *LimiterSuite) TestAnEmptyKeyIsNeverHeld() {
	limiter := core.NewLimiter(s.redis, s.clock.Now)

	s.Zero(limiter.Block(s.ctx, "", 30*time.Second))

	s.Zero(limiter.Wait(s.ctx, ""))
}

func (s *LimiterSuite) TestANilLimiterHoldsNothing() {
	var limiter *core.Limiter

	s.Zero(limiter.Block(s.ctx, s.key, 30*time.Second))

	s.Zero(limiter.Wait(s.ctx, s.key))
}

// TestAnUnreachableRedisHoldsNothing: the limiter fails open, as the daily quota does.
func (s *LimiterSuite) TestAnUnreachableRedisHoldsNothing() {
	held := core.NewLimiter(s.redis, s.clock.Now)
	s.Require().Equal(30*time.Second, held.Block(s.ctx, s.key, 30*time.Second))
	gone := s.connect(os.Getenv("ROUTER_REDIS_ADDR"))
	gone.Close()
	limiter := core.NewLimiter(gone, s.clock.Now)

	s.Zero(limiter.Block(s.ctx, s.key, 30*time.Second))
	s.Zero(limiter.Wait(s.ctx, s.key))
}

// TestTheCallsUpToTheLimitGoAndTheNextWaitsForTheWindowToEnd: limit 3 a minute, from 12:00:35
// (AI-958).
func (s *LimiterSuite) TestTheCallsUpToTheLimitGoAndTheNextWaitsForTheWindowToEnd() {
	s.clock.now = s.clock.now.Truncate(time.Minute).Add(35 * time.Second)
	limiter := core.NewLimiter(s.redis, s.clock.Now)

	for range 3 {
		s.Zero(limiter.Take(s.ctx, s.key, 3, time.Minute))
	}

	s.Equal(25*time.Second, limiter.Take(s.ctx, s.key, 3, time.Minute))
	s.clock.Add(20 * time.Second)
	s.Equal(5*time.Second, limiter.Take(s.ctx, s.key, 3, time.Minute), "a refused call does not move the window")
	s.clock.Add(5 * time.Second)
	s.Zero(limiter.Take(s.ctx, s.key, 3, time.Minute), "the next window starts a new count")
}

// TestTwoRoutersOnOneRedisCountTheSameCalls: the count is in Redis, so a customer spreading
// calls over routers gets the limit once.
func (s *LimiterSuite) TestTwoRoutersOnOneRedisCountTheSameCalls() {
	first := core.NewLimiter(s.redis, s.clock.Now)
	second := core.NewLimiter(s.connect(os.Getenv("ROUTER_REDIS_ADDR")), s.clock.Now)
	s.clock.now = s.clock.now.Truncate(time.Minute)

	s.Zero(first.Take(s.ctx, s.key, 2, time.Minute))
	s.Zero(second.Take(s.ctx, s.key, 2, time.Minute))

	s.Positive(first.Take(s.ctx, s.key, 2, time.Minute))
}

func (s *LimiterSuite) TestAnotherKeyIsNotCounted() {
	limiter := core.NewLimiter(s.redis, s.clock.Now)
	s.Zero(limiter.Take(s.ctx, s.key, 1, time.Minute))
	s.Require().Positive(limiter.Take(s.ctx, s.key, 1, time.Minute))

	s.Zero(limiter.Take(s.ctx, s.key+":other", 1, time.Minute))
}

// TestACountLeavesRedisAfterTwoWindows: no count outlives the window after its own.
func (s *LimiterSuite) TestACountLeavesRedisAfterTwoWindows() {
	s.clock.now = s.clock.now.Truncate(time.Minute)
	limiter := core.NewLimiter(s.redis, s.clock.Now)

	s.Zero(limiter.Take(s.ctx, s.key, 1, time.Minute))

	ttl, err := s.redis.Do(s.ctx, s.redis.B().Pttl().Key(s.key+":"+strconv.FormatInt(s.clock.now.UnixMilli(), 10)).Build()).AsInt64()
	s.Require().NoError(err)
	s.InDelta((2 * time.Minute).Milliseconds(), ttl, 1000)
}

// TestNoLimitNoKeyOrNoLimiterCountsNothing: each refuses no call, however many are sent.
func (s *LimiterSuite) TestNoLimitNoKeyOrNoLimiterCountsNothing() {
	limiter := core.NewLimiter(s.redis, s.clock.Now)
	var absent *core.Limiter
	for range 3 {
		s.Zero(limiter.Take(s.ctx, s.key, 0, time.Minute))
		s.Zero(limiter.Take(s.ctx, "", 1, time.Minute))
		s.Zero(limiter.Take(s.ctx, s.key+":window", 1, 0))
		s.Zero(absent.Take(s.ctx, s.key, 1, time.Minute))
	}
}

// TestAnUnreachableRedisCountsNothing: Take fails open, as Wait does.
func (s *LimiterSuite) TestAnUnreachableRedisCountsNothing() {
	gone := s.connect(os.Getenv("ROUTER_REDIS_ADDR"))
	gone.Close()
	limiter := core.NewLimiter(gone, s.clock.Now)

	for range 3 {
		s.Zero(limiter.Take(s.ctx, s.key, 1, time.Minute))
	}
}

// limiterClock is a clock a test moves.
type limiterClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *limiterClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *limiterClock) Add(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}
