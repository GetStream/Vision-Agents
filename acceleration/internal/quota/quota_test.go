//go:build integration

package quota

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/redis/rueidis"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
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

	client, err := redisready.NewClient(rueidis.ClientOption{
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
		UserID:   fmt.Sprintf("user-%d", unique),
		IP:       fmt.Sprintf("192.0.2.%d", unique%256),
		Verified: true,
	}
}

// limiter returns a Limiter over the shared Redis with the given allowance.
func (s *QuotaSuite) limiter(messages, tokens int64) *Limiter {
	return s.limiterWith(Limits{MessagesPerDay: messages, TokensPerDay: tokens}, nil)
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

	spent, err := limiter.spent(s.ctx, deviceKeys(s.customer, s.caller, time.Now())[0])

	s.Require().NoError(err)
	s.EqualValues(2, spent[0].messages)
	s.EqualValues(500, spent[0].tokens)
}

func (s *QuotaSuite) TestADaysCountersExpireOnTheirOwn() {
	limiter := s.limiterWith(Limits{MessagesPerDay: 10}, s.appQuota(Limits{MessagesPerDay: 10}))
	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	keys := append(deviceKeys(s.customer, s.caller, time.Now()), customerKey(s.customer, s.caller.UserID, time.Now()))
	s.Len(keys, 3)
	for _, key := range keys {
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

// limitsByCustomer is a Source that knows a few apps' quotas, and named users' by
// "<app>/<user>", and has no opinion about anybody else's.
type limitsByCustomer map[string]Limits

func (l limitsByCustomer) Quota(_ context.Context, customerID, userID string, defaults Limits) Quota {
	if limits, ok := l[customerID+"/"+userID]; ok {
		by := Setter{Level: LevelUser, ID: userID}
		return Quota{Messages: Limit{PerDay: limits.MessagesPerDay, By: by}, Tokens: Limit{PerDay: limits.TokensPerDay, By: by}}
	}
	if limits, ok := l[customerID]; ok {
		by := Setter{Level: LevelApp, ID: customerID}
		return Quota{Messages: Limit{PerDay: limits.MessagesPerDay, By: by}, Tokens: Limit{PerDay: limits.TokensPerDay, By: by}}
	}
	return Default(defaults)
}

// appQuota is a source in which the suite's app set limits for its users.
func (s *QuotaSuite) appQuota(limits Limits) limitsByCustomer {
	return limitsByCustomer{s.customer: limits}
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

func (s *QuotaSuite) TestTheDeploymentsLimitHoldsWhereNoPolicySetsOne() {
	// A source with no opinion leaves the deployment's three messages a day.
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
	for _, key := range append(deviceKeys(s.customer, s.caller, time.Now()), customerKey(s.customer, s.caller.UserID, time.Now())) {
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

// stalledSource is a Source whose store answers only once the lookup is given up on, or
// after a minute when nothing gives it up.
type stalledSource struct{}

func (stalledSource) Quota(ctx context.Context, _, _ string, _ Limits) Quota {
	select {
	case <-ctx.Done():
	case <-time.After(time.Minute):
	}
	return Default(Limits{MessagesPerDay: 1})
}

func (s *QuotaSuite) TestADebitWaitsNoLongerThanItsTimeoutForTheLimits() {
	// The debit runs on the goroutine draining a finished response, which a stalled policy
	// store must not hold past the debit's own timeout.
	limiter := s.limiterWith(Limits{}, stalledSource{})
	started := time.Now()

	limiter.Debit(context.Background(), s.customer, s.caller, 10)

	s.Less(time.Since(started), writeTimeout+time.Second)
}

func (s *QuotaSuite) TestADebitWhoseLimitsWereSlowToReadIsStillCounted() {
	// Reading the limits used up the whole wait, and the response they count was answered
	// all the same: what it cost is counted once it reaches Redis.
	limiter := s.limiterWith(Limits{}, stalledSource{})

	limiter.Debit(context.Background(), s.customer, s.caller, 10)

	s.EventuallyWithT(func(c *assert.CollectT) {
		spent, err := limiter.spent(s.ctx, deviceKeys(s.customer, s.caller, time.Now())...)
		require.NoError(c, err)
		assert.Equal(c, []spend{{messages: 1, tokens: 10}, {messages: 1, tokens: 10}}, spent)
	}, writeTimeout, 10*time.Millisecond)
}

func (s *QuotaSuite) TestADebitWaitsNoLongerThanItsTimeoutForTheLimitsAndRedisTogether() {
	// A Redis too busy to answer, after limits that took the whole wait to read, must not
	// hold the goroutine draining a finished response past the debit's own timeout either.
	slow := newSlowRedis(s.T(), os.Getenv(AddressEnvVar))
	client, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{slow.address()}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	limiter, err := New(client, Limits{}, stalledSource{}, slog.Default())
	s.Require().NoError(err)
	slow.hold()
	started := time.Now()

	limiter.Debit(context.Background(), s.customer, s.caller, 10)

	s.Less(time.Since(started), writeTimeout+time.Second)
	slow.release()
}

// slowRedis passes a test's commands to the suite's Redis on connections of its own, and
// holds the replies back while it is held: a Redis too busy to answer, that no other suite
// sharing the server feels.
type slowRedis struct {
	listener net.Listener
	mu       sync.Mutex
	// open is closed while replies pass.
	open  chan struct{}
	conns []net.Conn
}

func newSlowRedis(t *testing.T, target string) *slowRedis {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	open := make(chan struct{})
	close(open)
	redis := &slowRedis{listener: listener, open: open}
	t.Cleanup(redis.close)
	go func() {
		for {
			client, err := listener.Accept()
			if err != nil {
				return
			}
			server, err := net.Dial("tcp", target)
			if err != nil {
				_ = client.Close()
				continue
			}
			redis.mu.Lock()
			redis.conns = append(redis.conns, client, server)
			redis.mu.Unlock()
			go func() { _, _ = io.Copy(server, client) }()
			go redis.reply(client, server)
		}
	}()
	return redis
}

func (r *slowRedis) address() string { return r.listener.Addr().String() }

// hold keeps every reply back until release.
func (r *slowRedis) hold() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.open = make(chan struct{})
}

func (r *slowRedis) release() {
	r.mu.Lock()
	defer r.mu.Unlock()
	select {
	case <-r.open:
	default:
		close(r.open)
	}
}

func (r *slowRedis) gate() chan struct{} {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.open
}

func (r *slowRedis) reply(client, server net.Conn) {
	buffer := make([]byte, 32<<10)
	for {
		n, err := server.Read(buffer)
		if n > 0 {
			<-r.gate()
			if _, err := client.Write(buffer[:n]); err != nil {
				return
			}
		}
		if err != nil {
			return
		}
	}
}

func (r *slowRedis) close() {
	r.release()
	_ = r.listener.Close()
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, conn := range r.conns {
		_ = conn.Close()
	}
}

// backend is the app's own process acting for the suite's user.
func (s *QuotaSuite) backend() routing.Caller {
	return routing.Caller{UserID: s.caller.UserID, IP: s.caller.IP, ServerSide: true}
}

func (s *QuotaSuite) TestTheDeploymentsDefaultsStillSkipABackend() {
	limiter := s.limiter(1, 1)
	limiter.Debit(s.ctx, s.customer, s.backend(), 10_000)

	for _, key := range append(deviceKeys(s.customer, s.caller, time.Now()), customerKey(s.customer, s.caller.UserID, time.Now())) {
		exists, err := s.redis.Do(s.ctx, s.redis.B().Exists().Key(key).Build()).AsInt64()
		s.Require().NoError(err)
		s.Zero(exists, "a backend's work is in no bucket of the deployment's: %s", key)
	}
	s.NoError(s.allow(limiter, s.customer, s.caller))
}

func (s *QuotaSuite) TestACustomerQuotaCountsABackendForTheUser() {
	limiter := s.limiterWith(Limits{MessagesPerDay: 100}, s.appQuota(Limits{MessagesPerDay: 2}))
	s.Require().NoError(s.allow(limiter, s.customer, s.caller))
	limiter.Debit(s.ctx, s.customer, s.caller, 10)
	s.Require().NoError(s.allow(limiter, s.customer, s.backend()))
	limiter.Debit(s.ctx, s.customer, s.backend(), 10)

	s.ErrorIs(s.allow(limiter, s.customer, s.caller), ErrExhausted, "from the user's device")
	s.ErrorIs(s.allow(limiter, s.customer, s.backend()), ErrExhausted, "from the app's backend")
	s.NoError(s.allow(limiter, s.customer, routing.Caller{UserID: s.caller.UserID + "-other"}),
		"another user of the same app has a day of their own")
}

func (s *QuotaSuite) TestARefusalNamesWhoseQuotaItIs() {
	limiter := s.limiterWith(Limits{}, limitsByCustomer{
		s.customer:                         {MessagesPerDay: 1},
		s.customer + "/" + s.caller.UserID: {MessagesPerDay: 1},
	})
	other := routing.Caller{UserID: s.caller.UserID + "-other"}
	limiter.Debit(s.ctx, s.customer, s.backend(), 10)
	limiter.Debit(s.ctx, s.customer, other, 10)

	own := s.allow(limiter, s.customer, s.backend())
	apps := s.allow(limiter, s.customer, other)

	s.Require().ErrorIs(own, ErrExhausted)
	s.Contains(own.Error(), "for user "+s.caller.UserID)
	s.Require().ErrorIs(apps, ErrExhausted)
	s.Contains(apps.Error(), "for every user of app "+s.customer)
}

func (s *QuotaSuite) TestTheDeploymentsTokensStillSkipABackendUnderACustomersMessages() {
	// The app set the messages only, so the tokens stay the deployment's, and count devices only.
	limiter := s.limiterWith(Limits{}, sourceFunc(func(context.Context, string, string, Limits) Quota {
		return Quota{Messages: Limit{PerDay: 3, By: Setter{Level: LevelApp, ID: s.customer}}, Tokens: Limit{PerDay: 1_000}}
	}))
	limiter.Debit(s.ctx, s.customer, s.backend(), 5_000)

	s.NoError(s.allow(limiter, s.customer, s.caller), "the backend's tokens are not the device's")
	limiter.Debit(s.ctx, s.customer, s.backend(), 10)
	limiter.Debit(s.ctx, s.customer, s.caller, 10)
	err := s.allow(limiter, s.customer, s.caller)
	s.Require().ErrorIs(err, ErrExhausted, "three messages counted, two of them the backend's")
	s.Contains(err.Error(), "3 messages a day")
}

func (s *QuotaSuite) TestUsersBehindOneAddressEachHaveTheCustomersQuota() {
	// An office, or a carrier's NAT: the customer's quota is each user's, not the address's.
	limiter := s.limiterWith(Limits{}, s.appQuota(Limits{MessagesPerDay: 1}))
	limiter.Debit(s.ctx, s.customer, routing.Caller{UserID: "first", IP: s.caller.IP, Verified: true}, 10)

	s.NoError(s.allow(limiter, s.customer, routing.Caller{UserID: "second", IP: s.caller.IP, Verified: true}))
	err := s.allow(limiter, s.customer, routing.Caller{UserID: "first", IP: s.caller.IP, Verified: true})
	s.Require().ErrorIs(err, ErrExhausted, "the user who spent it is held to it")
	s.Contains(err.Error(), "for every user of app "+s.customer)
}

func (s *QuotaSuite) TestAnAddressStillAnswersForTheDeploymentsLimit() {
	// The app set the messages, so the tokens are the deployment's, and the address answers
	// for those.
	limiter := s.limiterWith(Limits{}, sourceFunc(func(context.Context, string, string, Limits) Quota {
		return Quota{Messages: Limit{PerDay: 5, By: Setter{Level: LevelApp, ID: s.customer}}, Tokens: Limit{PerDay: 100}}
	}))
	limiter.Debit(s.ctx, s.customer, routing.Caller{UserID: "first", IP: s.caller.IP}, 100)

	err := s.allow(limiter, s.customer, routing.Caller{UserID: "second", IP: s.caller.IP})
	s.Require().ErrorIs(err, ErrExhausted, "a fresh user id behind a spent address")
	s.Contains(err.Error(), "100 tokens a day, for the router's default")
	s.NoError(s.allow(limiter, s.customer, routing.Caller{UserID: "second", IP: s.caller.IP, ServerSide: true}),
		"a backend has no address of its user's to answer for")
}

func (s *QuotaSuite) TestTodayIsWhatTheCustomersQuotaCounted() {
	limiter := s.limiterWith(Limits{}, s.appQuota(Limits{MessagesPerDay: 10}))
	limiter.Debit(s.ctx, s.customer, s.caller, 100)
	limiter.Debit(s.ctx, s.customer, s.backend(), 50)

	messages, tokens, reset, err := limiter.Today(s.ctx, s.customer, s.caller.UserID)

	s.Require().NoError(err)
	s.EqualValues(2, messages)
	s.EqualValues(150, tokens)
	s.Equal(nextMidnight(time.Now()), reset)
}

func (s *QuotaSuite) TestUserKeysAreEveryBucketAUserIsCountedIn() {
	// The app's messages count the user, and the deployment's tokens count the address too.
	limiter := s.limiterWith(Limits{}, sourceFunc(func(context.Context, string, string, Limits) Quota {
		return Quota{Messages: Limit{PerDay: 1, By: Setter{Level: LevelApp, ID: s.customer}}, Tokens: Limit{PerDay: 10}}
	}))
	limiter.Debit(s.ctx, s.customer, s.caller, 10)
	s.Require().ErrorIs(s.allow(limiter, s.customer, s.caller), ErrExhausted)

	for _, key := range UserKeys(s.customer, s.caller.UserID, time.Now()) {
		s.Require().NoError(s.redis.Do(s.ctx, s.redis.B().Del().Key(key).Build()).Error())
	}

	elsewhere := s.caller
	elsewhere.IP = "198.51.100.7"
	s.NoError(s.allow(limiter, s.customer, elsewhere), "nothing counts against them but the address they used")
	s.ErrorIs(s.allow(limiter, s.customer, s.caller), ErrExhausted, "which still answers for itself")
}

func (s *QuotaSuite) TestTheEffectiveQuotaIsTheSources() {
	limiter := s.limiterWith(Limits{MessagesPerDay: 7}, s.appQuota(Limits{TokensPerDay: 9}))

	held := limiter.Effective(s.ctx, s.customer, s.caller.UserID)

	s.EqualValues(9, held.Tokens.PerDay)
	s.Equal(Setter{Level: LevelApp, ID: s.customer}, held.Tokens.By)
	var none *Limiter
	s.Zero(none.Effective(s.ctx, s.customer, s.caller.UserID), "a deployment that counts nothing enforces nothing")
}

func (s *QuotaSuite) TestAnAnonymousNameIsNotHeldToThatUsersOwnQuota() {
	// The user's own quota of 0 is no cap; a device going by their name without a token is
	// held to the app's one message instead, and spends nothing of the user's day.
	limiter := s.limiterWith(Limits{}, limitsByCustomer{
		s.customer:                         {MessagesPerDay: 1},
		s.customer + "/" + s.caller.UserID: {MessagesPerDay: 0},
	})
	impostor := routing.Caller{UserID: s.caller.UserID, IP: "198.51.100.7"}
	limiter.Debit(s.ctx, s.customer, impostor, 10)

	err := s.allow(limiter, s.customer, impostor)

	s.Require().ErrorIs(err, ErrExhausted)
	s.Contains(err.Error(), "for every user of app "+s.customer)
	exists, err := s.redis.Do(s.ctx, s.redis.B().Exists().Key(customerKey(s.customer, s.caller.UserID, time.Now())).Build()).AsInt64()
	s.Require().NoError(err)
	s.Zero(exists, "the user's own day is untouched")
	s.NoError(s.allow(limiter, s.customer, s.caller), "the verified user keeps their own quota")
}

// sourceFunc is a Source written inline.
type sourceFunc func(ctx context.Context, customerID, userID string, defaults Limits) Quota

func (f sourceFunc) Quota(ctx context.Context, customerID, userID string, defaults Limits) Quota {
	return f(ctx, customerID, userID, defaults)
}

// unreadSource is a Source whose user's own quota could not be read: it knows the limits it
// is given and marks the answer Unread.
func unreadSource(known Quota) sourceFunc {
	return func(_ context.Context, _, _ string, _ Limits) Quota {
		known.Unread = true
		return known
	}
}

func (s *QuotaSuite) TestADebitIsCountedInTheCustomersBucketWhenTheUsersOwnQuotaCouldNotBeRead() {
	// Only the user's own policy sets a limit, and reading it failed: the app's quota is all
	// that is known, and it is none. What the user spent still reaches the customer's bucket.
	appLimit := Quota{Messages: Limit{PerDay: 2, By: Setter{Level: LevelApp, ID: s.customer}}}
	cases := []struct {
		name       string
		known      Quota
		caller     routing.Caller
		wantWarned int
	}{
		{name: "a device caller, nothing else known", caller: s.caller, wantWarned: 1},
		{name: "a customer's backend, nothing else known", caller: routing.Caller{UserID: s.caller.UserID, ServerSide: true}, wantWarned: 1},
		{name: "the deployment's limit alone is known", known: Quota{Tokens: Limit{PerDay: 100}}, caller: s.caller, wantWarned: 1},
		{name: "the app's limit is known, so no bucket was skipped", known: appLimit, caller: s.caller},
	}
	for _, tc := range cases {
		s.Run(tc.name, func() {
			customer := s.customer + "-" + strings.ReplaceAll(tc.name, " ", "-")
			var logs bytes.Buffer
			limiter, err := New(s.redis, Limits{}, unreadSource(tc.known), slog.New(slog.NewTextHandler(&logs, nil)))
			s.Require().NoError(err)

			limiter.Debit(s.ctx, customer, tc.caller, 10)

			spent, err := limiter.spent(s.ctx, customerKey(customer, tc.caller.UserID, time.Now()))
			s.Require().NoError(err)
			s.Equal(spend{messages: 1, tokens: 10}, spent[0])
			s.Equal(tc.wantWarned, strings.Count(logs.String(), "level=WARN"), logs.String())
		})
	}
}

func (s *QuotaSuite) TestTheWarningForAnUnreadUsersQuotaIsLoggedOncePerCustomerAWindow() {
	var logs bytes.Buffer
	limiter, err := New(s.redis, Limits{}, unreadSource(Quota{}), slog.New(slog.NewTextHandler(&logs, nil)))
	s.Require().NoError(err)
	other := s.customer + "-other"

	for range 20 {
		limiter.Debit(s.ctx, s.customer, s.caller, 10)
	}
	s.Equal(1, strings.Count(logs.String(), "level=WARN"), "twenty debits of one customer log once")

	limiter.Debit(s.ctx, other, s.caller, 10)
	s.Equal(2, strings.Count(logs.String(), "level=WARN"), "another customer still logs")

	spent, err := limiter.spent(s.ctx, customerKey(s.customer, s.caller.UserID, time.Now()))
	s.Require().NoError(err)
	s.Equal(spend{messages: 20, tokens: 200}, spent[0], "every debit is counted, though the log is throttled")
}

func (s *QuotaSuite) TestADebitStillCountsADeviceCallersAddressWhenTheUsersOwnQuotaCouldNotBeRead() {
	limiter := s.limiterWith(Limits{}, unreadSource(Quota{}))

	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	spent, err := limiter.spent(s.ctx, deviceKeys(s.customer, s.caller, time.Now())...)
	s.Require().NoError(err)
	s.Equal([]spend{{messages: 1, tokens: 10}, {messages: 1, tokens: 10}}, spent)
}

func (s *QuotaSuite) TestADebitCountsNothingForAUserHeldToNoLimitWhoseQuotaWasRead() {
	// The control: a quota that was read and sets nothing is not counted, as before.
	var logs bytes.Buffer
	limiter, err := New(s.redis, Limits{}, sourceFunc(func(context.Context, string, string, Limits) Quota { return Quota{} }),
		slog.New(slog.NewTextHandler(&logs, nil)))
	s.Require().NoError(err)

	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	keys := append(deviceKeys(s.customer, s.caller, time.Now()), customerKey(s.customer, s.caller.UserID, time.Now()))
	for _, key := range keys {
		exists, err := s.redis.Do(s.ctx, s.redis.B().Exists().Key(key).Build()).AsInt64()
		s.Require().NoError(err)
		s.Zero(exists, key)
	}
	s.Empty(logs.String())
}

func (s *QuotaSuite) TestARequestWhoseUsersOwnQuotaCouldNotBeReadIsStillAllowed() {
	// Admission stays open on an unread policy; only the debit changed.
	limiter := s.limiterWith(Limits{}, unreadSource(Quota{}))
	limiter.Debit(s.ctx, s.customer, s.caller, 10)

	s.NoError(s.allow(limiter, s.customer, s.caller))
}
