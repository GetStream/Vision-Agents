//go:build integration

package quota

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// AddressEnvVar is where the tests look for a Redis to run against.
const AddressEnvVar = "ROUTER_REDIS_ADDR"

type QuotaSuite struct {
	suite.Suite
	redis rueidis.Client
	ctx   context.Context
	// customer and caller are unique per test so parallel runs cannot collide on keys, and
	// so no test has to clean up after itself.
	customer string
	caller   routing.Caller
}

func TestQuotaSuite(t *testing.T) {
	suite.Run(t, new(QuotaSuite))
}

func (s *QuotaSuite) SetupSuite() {
	address := os.Getenv(AddressEnvVar)
	if address == "" {
		s.T().Skipf("%s not set", AddressEnvVar)
	}

	client, err := rueidis.NewClient(rueidis.ClientOption{
		InitAddress:  []string{address},
		DisableCache: true,
	})
	s.Require().NoError(err)
	s.redis = client
	s.ctx = context.Background()
	s.Require().NoError(client.Do(s.ctx, client.B().Ping().Build()).Error())
}

func (s *QuotaSuite) TearDownSuite() {
	if s.redis != nil {
		s.redis.Close()
	}
}

func (s *QuotaSuite) SetupTest() {
	unique := time.Now().UnixNano()
	s.customer = fmt.Sprintf("customer-%d", unique)
	s.caller = routing.Caller{
		UserID: fmt.Sprintf("user-%d", unique),
		IP:     fmt.Sprintf("192.0.2.%d", unique%256),
	}
}

// limiter returns a Limiter over the shared Redis with the given allowance.
func (s *QuotaSuite) limiter(messages, tokens int64) *Limiter {
	limiter, err := New(s.redis, Limits{MessagesPerDay: messages, TokensPerDay: tokens}, slog.Default())
	s.Require().NoError(err)
	return limiter
}

func (s *QuotaSuite) TestACallerUnderTheirAllowanceIsAllowed() {
	limiter := s.limiter(3, 0)
	limiter.Debit(s.ctx, s.customer, s.caller, 10)
	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	s.NoError(limiter.Allow(s.ctx, s.customer, s.caller))
}

func (s *QuotaSuite) TestTheMessageAfterTheLastOneIsRefused() {
	limiter := s.limiter(3, 0)
	for range 3 {
		s.Require().NoError(limiter.Allow(s.ctx, s.customer, s.caller))
		limiter.Debit(s.ctx, s.customer, s.caller, 10)
	}

	s.ErrorIs(limiter.Allow(s.ctx, s.customer, s.caller), ErrExhausted)
}

func (s *QuotaSuite) TestOneEnormousRequestSpendsTheDayBeforeTheMessagesRunOut() {
	// The point of the token limit: a caller with 200 messages left and no tokens is done.
	limiter := s.limiter(200, 1_000)
	limiter.Debit(s.ctx, s.customer, s.caller, 1_000)

	err := limiter.Allow(s.ctx, s.customer, s.caller)

	s.ErrorIs(err, ErrExhausted)
	s.Contains(err.Error(), "tokens", "the caller should be told which limit they reached")
}

func (s *QuotaSuite) TestTwoUsersOfOneCustomerDoNotShareAnAllowance() {
	limiter := s.limiter(1, 0)
	spent := routing.Caller{UserID: s.caller.UserID}
	limiter.Debit(s.ctx, s.customer, spent, 10)
	s.Require().ErrorIs(limiter.Allow(s.ctx, s.customer, spent), ErrExhausted)

	other := routing.Caller{UserID: s.caller.UserID + "-other"}

	s.NoError(limiter.Allow(s.ctx, s.customer, other))
}

func (s *QuotaSuite) TestTwoCustomersMayNameTheSameUser() {
	// A user id belongs to the customer who minted it, so the same name under two of them
	// is two people and gets two allowances.
	limiter := s.limiter(1, 0)
	spent := routing.Caller{UserID: s.caller.UserID}
	limiter.Debit(s.ctx, s.customer, spent, 10)
	s.Require().ErrorIs(limiter.Allow(s.ctx, s.customer, spent), ErrExhausted)

	s.NoError(limiter.Allow(s.ctx, s.customer+"-other", spent))
}

func (s *QuotaSuite) TestTheAddressIsCountedWhenTheUserIsFresh() {
	// A backend minting a new user id per request still has one address to answer for.
	limiter := s.limiter(1, 0)
	limiter.Debit(s.ctx, s.customer, routing.Caller{UserID: "first", IP: s.caller.IP}, 10)

	err := limiter.Allow(s.ctx, s.customer, routing.Caller{UserID: "second", IP: s.caller.IP})

	s.ErrorIs(err, ErrExhausted)
}

func (s *QuotaSuite) TestNobodyToCountAgainstIsNotCounted() {
	// A customer's own backend, which is trusted with its own spend.
	limiter := s.limiter(1, 1)
	backend := routing.Caller{}
	limiter.Debit(s.ctx, s.customer, backend, 10_000)
	limiter.Debit(s.ctx, s.customer, backend, 10_000)

	s.NoError(limiter.Allow(s.ctx, s.customer, backend))
}

func (s *QuotaSuite) TestALimitOfZeroIsNotEnforced() {
	// Messages off, tokens on: the count still climbs, and only the token limit refuses.
	limiter := s.limiter(0, 100)
	for range 5 {
		limiter.Debit(s.ctx, s.customer, s.caller, 10)
	}
	s.Require().NoError(limiter.Allow(s.ctx, s.customer, s.caller))

	limiter.Debit(s.ctx, s.customer, s.caller, 50)

	s.ErrorIs(limiter.Allow(s.ctx, s.customer, s.caller), ErrExhausted)
}

func (s *QuotaSuite) TestADebitCountsBothMessagesAndTokens() {
	limiter := s.limiter(10, 1_000)
	limiter.Debit(s.ctx, s.customer, s.caller, 250)
	limiter.Debit(s.ctx, s.customer, s.caller, 250)

	spent, err := limiter.spent(s.ctx, limiter.keys(s.customer, s.caller)[0])

	s.Require().NoError(err)
	s.EqualValues(2, spent.messages)
	s.EqualValues(500, spent.tokens)
}

func (s *QuotaSuite) TestADaysCountersExpireOnTheirOwn() {
	limiter := s.limiter(10, 0)
	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	for _, key := range limiter.keys(s.customer, s.caller) {
		ttl, err := s.redis.Do(s.ctx, s.redis.B().Ttl().Key(key).Build()).AsInt64()
		s.Require().NoError(err)
		s.Positive(ttl, "a day's counters should not outlive their retention")
		s.LessOrEqual(ttl, int64(retention.Seconds()))
	}
}

func (s *QuotaSuite) TestALimiterNeedsSomewhereToCountAndSomethingToEnforce() {
	_, err := New(nil, Limits{MessagesPerDay: 1}, slog.Default())
	s.Require().Error(err)
	s.Contains(err.Error(), "redis")

	_, err = New(s.redis, Limits{}, slog.Default())
	s.Require().Error(err)
	s.Contains(err.Error(), "limits")
}

func (s *QuotaSuite) TestANilLimiterAllowsEverything() {
	// What a deployment with no Redis to count in runs with.
	var limiter *Limiter

	limiter.Debit(s.ctx, s.customer, s.caller, 10_000)

	s.NoError(limiter.Allow(s.ctx, s.customer, s.caller))
}
