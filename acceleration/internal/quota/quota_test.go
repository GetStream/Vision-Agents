//go:build integration

package quota

import (
	"context"
	"errors"
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
	limiter, err := New(s.redis, Limits{MessagesPerDay: messages, TokensPerDay: tokens}, nil, slog.Default())
	s.Require().NoError(err)
	return limiter
}

func (s *QuotaSuite) TestACallerUnderTheirAllowanceIsAllowed() {
	limiter := s.limiter(3, 0)
	limiter.Debit(s.ctx, s.customer, s.caller, 10)
	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	s.NoError(s.allow(limiter, s.customer, s.caller))
}

func (s *QuotaSuite) TestTheMessageAfterTheLastOneIsRefused() {
	limiter := s.limiter(3, 0)
	for range 3 {
		s.Require().NoError(s.allow(limiter, s.customer, s.caller))
		limiter.Debit(s.ctx, s.customer, s.caller, 10)
	}

	s.ErrorIs(s.allow(limiter, s.customer, s.caller), ErrExhausted)
}

func (s *QuotaSuite) TestOneEnormousRequestSpendsTheDayBeforeTheMessagesRunOut() {
	// The point of the token limit: a caller with 200 messages left and no tokens is done.
	limiter := s.limiter(200, 1_000)
	limiter.Debit(s.ctx, s.customer, s.caller, 1_000)

	err := s.allow(limiter, s.customer, s.caller)

	s.ErrorIs(err, ErrExhausted)
	s.Contains(err.Error(), "tokens", "the caller should be told which limit they reached")
}

func (s *QuotaSuite) TestTwoUsersOfOneCustomerDoNotShareAnAllowance() {
	limiter := s.limiter(1, 0)
	spent := routing.Caller{UserID: s.caller.UserID}
	limiter.Debit(s.ctx, s.customer, spent, 10)
	s.Require().ErrorIs(s.allow(limiter, s.customer, spent), ErrExhausted)

	other := routing.Caller{UserID: s.caller.UserID + "-other"}

	s.NoError(s.allow(limiter, s.customer, other))
}

func (s *QuotaSuite) TestTwoCustomersMayNameTheSameUser() {
	// A user id belongs to the customer who minted it, so the same name under two of them
	// is two people and gets two allowances.
	limiter := s.limiter(1, 0)
	spent := routing.Caller{UserID: s.caller.UserID}
	limiter.Debit(s.ctx, s.customer, spent, 10)
	s.Require().ErrorIs(s.allow(limiter, s.customer, spent), ErrExhausted)

	s.NoError(s.allow(limiter, s.customer+"-other", spent))
}

func (s *QuotaSuite) TestTheAddressIsCountedWhenTheUserIsFresh() {
	// A backend minting a new user id per request still has one address to answer for.
	limiter := s.limiter(1, 0)
	limiter.Debit(s.ctx, s.customer, routing.Caller{UserID: "first", IP: s.caller.IP}, 10)

	err := s.allow(limiter, s.customer, routing.Caller{UserID: "second", IP: s.caller.IP})

	s.ErrorIs(err, ErrExhausted)
}

func (s *QuotaSuite) TestNobodyToCountAgainstIsNotCounted() {
	// A customer's own backend, which is trusted with its own spend.
	limiter := s.limiter(1, 1)
	backend := routing.Caller{}
	limiter.Debit(s.ctx, s.customer, backend, 10_000)
	limiter.Debit(s.ctx, s.customer, backend, 10_000)

	s.NoError(s.allow(limiter, s.customer, backend))
}

func (s *QuotaSuite) TestALimitOfZeroIsNotEnforced() {
	// Messages off, tokens on: the count still climbs, and only the token limit refuses.
	limiter := s.limiter(0, 100)
	for range 5 {
		limiter.Debit(s.ctx, s.customer, s.caller, 10)
	}
	s.Require().NoError(s.allow(limiter, s.customer, s.caller))

	limiter.Debit(s.ctx, s.customer, s.caller, 50)

	s.ErrorIs(s.allow(limiter, s.customer, s.caller), ErrExhausted)
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

func (s *QuotaSuite) TestALimiterNeedsSomewhereToCount() {
	_, err := New(nil, Limits{MessagesPerDay: 1}, nil, slog.Default())

	s.Require().Error(err)
	s.Contains(err.Error(), "redis")
}

func (s *QuotaSuite) TestANilLimiterAllowsEverything() {
	// What a deployment with no Redis to count in runs with.
	var limiter *Limiter

	limiter.Debit(s.ctx, s.customer, s.caller, 10_000)

	s.NoError(s.allow(limiter, s.customer, s.caller))
}

// allow is Allow without the status, for the tests that ask only whether a caller may go on.
func (s *QuotaSuite) allow(limiter *Limiter, customerID string, caller routing.Caller) error {
	_, err := limiter.Allow(s.ctx, customerID, caller)
	return err
}

// limitsByCustomer is a Source that knows a few customers' limits and has no opinion about
// anybody else's.
type limitsByCustomer map[string]Limits

func (l limitsByCustomer) Limits(_ context.Context, customerID string, defaults Limits) (Limits, error) {
	if limits, ok := l[customerID]; ok {
		return limits, nil
	}
	return defaults, nil
}

// unreadable is a Source whose policies cannot be read.
type unreadable struct{}

func (unreadable) Limits(context.Context, string, Limits) (Limits, error) {
	return Limits{}, errors.New("policies: connection refused")
}

func (s *QuotaSuite) limiterWith(defaults Limits, source Source) *Limiter {
	limiter, err := New(s.redis, defaults, source, slog.Default())
	s.Require().NoError(err)
	return limiter
}

func (s *QuotaSuite) TestEachAppsUsersAreHeldToTheirAppsOwnLimit() {
	other := s.customer + "-other"
	limiter := s.limiterWith(Limits{MessagesPerDay: 100}, limitsByCustomer{s.customer: {MessagesPerDay: 5}})
	for range 5 {
		s.Require().NoError(s.allow(limiter, s.customer, s.caller))
		limiter.Debit(s.ctx, s.customer, s.caller, 10)
		limiter.Debit(s.ctx, other, s.caller, 10)
	}

	s.ErrorIs(s.allow(limiter, s.customer, s.caller), ErrExhausted, "the app allows five")
	s.NoError(s.allow(limiter, other, s.caller), "the other app is held to the deployment's hundred")
}

func (s *QuotaSuite) TestALimitThatCannotBeReadIsTheDeployments() {
	limiter := s.limiterWith(Limits{MessagesPerDay: 1}, unreadable{})
	s.Require().NoError(s.allow(limiter, s.customer, s.caller))
	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	s.ErrorIs(s.allow(limiter, s.customer, s.caller), ErrExhausted)
}

func (s *QuotaSuite) TestTheDeploymentsLimitHoldsWhereNoPolicySetsOne() {
	// Control: the base allowed three messages under a limit of three and refused the
	// fourth (probe in author-1.md). A source with no opinion must change neither.
	limiter := s.limiterWith(Limits{MessagesPerDay: 3}, limitsByCustomer{})
	allowed := 0
	for s.allow(limiter, s.customer, s.caller) == nil && allowed < 10 {
		allowed++
		limiter.Debit(s.ctx, s.customer, s.caller, 10)
	}

	s.Equal(3, allowed)
}

func (s *QuotaSuite) TestACustomersOwnProcessIsNeverCounted() {
	// A backend naming the user it acts for is still the customer's own process.
	limiter := s.limiter(1, 1)
	backend := routing.Caller{UserID: s.caller.UserID, IP: s.caller.IP, ServerSide: true}
	limiter.Debit(s.ctx, s.customer, backend, 10_000)
	limiter.Debit(s.ctx, s.customer, backend, 10_000)

	s.NoError(s.allow(limiter, s.customer, backend))
	s.NoError(s.allow(limiter, s.customer, s.caller),
		"what a backend spent for a user is not the user's")
}

func (s *QuotaSuite) TestAnAddressIsCountedPerApp() {
	limiter := s.limiter(1, 0)
	address := routing.Caller{IP: s.caller.IP}
	limiter.Debit(s.ctx, s.customer, address, 10)
	s.Require().ErrorIs(s.allow(limiter, s.customer, address), ErrExhausted)

	s.NoError(s.allow(limiter, s.customer+"-other", address),
		"each app sets its own limit, so one address is counted once per app")
}

func (s *QuotaSuite) TestACallerIsToldWhatIsLeftOfTheirDay() {
	limiter := s.limiter(5, 0)
	limiter.Debit(s.ctx, s.customer, s.caller, 10)
	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	status, err := limiter.Allow(s.ctx, s.customer, s.caller)

	s.Require().NoError(err)
	s.EqualValues(5, status.Limit)
	s.EqualValues(3, status.Remaining)
	now := time.Now().UTC()
	s.Equal(time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, time.UTC), status.Reset)
}

func (s *QuotaSuite) TestACallerIsToldAboutTheLimitNearestToRefusingThem() {
	// Nine of ten messages left, and a tenth of the tokens: the tokens refuse first.
	limiter := s.limiter(10, 1_000)
	limiter.Debit(s.ctx, s.customer, s.caller, 900)

	status, err := limiter.Allow(s.ctx, s.customer, s.caller)

	s.Require().NoError(err)
	s.EqualValues(1_000, status.Limit)
	s.EqualValues(100, status.Remaining)
}

func (s *QuotaSuite) TestARefusedCallerIsToldNothingIsLeft() {
	limiter := s.limiter(1, 0)
	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	status, err := limiter.Allow(s.ctx, s.customer, s.caller)

	s.Require().ErrorIs(err, ErrExhausted)
	s.EqualValues(1, status.Limit)
	s.Zero(status.Remaining)
}

func (s *QuotaSuite) TestACustomerHeldToNoLimitIsNotCounted() {
	// A deployment with no default, and an app that set none: nothing to enforce, so nothing
	// is written either, which is what such a deployment did before it had a Limiter at all.
	limiter := s.limiterWith(Limits{}, limitsByCustomer{})
	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	status, err := limiter.Allow(s.ctx, s.customer, s.caller)

	s.Require().NoError(err)
	s.Zero(status, "an uncounted caller is told nothing")
	for _, key := range limiter.keys(s.customer, s.caller) {
		exists, err := s.redis.Do(s.ctx, s.redis.B().Exists().Key(key).Build()).AsInt64()
		s.Require().NoError(err)
		s.Zero(exists, "nothing is counted for %s", key)
	}
}

func (s *QuotaSuite) TestAnAppsLimitHoldsWhereTheDeploymentHasNone() {
	limiter := s.limiterWith(Limits{}, limitsByCustomer{s.customer: {MessagesPerDay: 1}})
	s.Require().NoError(s.allow(limiter, s.customer, s.caller))
	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	s.ErrorIs(s.allow(limiter, s.customer, s.caller), ErrExhausted)
}
