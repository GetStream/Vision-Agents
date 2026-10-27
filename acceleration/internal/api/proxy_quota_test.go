//go:build integration

package api

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// ProxyQuotaSuite runs a proxy-mode router that allows one message a day, with every request
// arriving from the same address the way they all arrive from the proxy in front of it.
type ProxyQuotaSuite struct {
	suite.Suite
	redis   rueidis.Client
	limiter *quota.Limiter
	// app is unique per test so that a user spent in one is a stranger in the next.
	app string
}

func TestProxyQuotaSuite(t *testing.T) {
	suite.Run(t, new(ProxyQuotaSuite))
}

func (s *ProxyQuotaSuite) SetupSuite() {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR must be set")
	}
	client, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.redis = client
	s.limiter, err = quota.New(client, quota.Limits{MessagesPerDay: 1}, nil, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
}

func (s *ProxyQuotaSuite) TearDownSuite() {
	if s.redis != nil {
		s.redis.Close()
	}
}

// SetupTest forgets the day this test's app spent, since every test here calls from the same address.
// Only its own app's buckets: the Redis is every suite's.
func (s *ProxyQuotaSuite) SetupTest() {
	s.app = strings.ReplaceAll(s.T().Name(), "/", "-")
	ctx := s.T().Context()
	spent, err := s.redis.Do(ctx, s.redis.B().Keys().Pattern("quota:*:ip:"+s.app+":*").Build()).AsStrSlice()
	s.Require().NoError(err)
	if len(spent) > 0 {
		s.Require().NoError(s.redis.Do(ctx, s.redis.B().Del().Key(spent...).Build()).Error())
	}
}

func (s *ProxyQuotaSuite) TestForgettingATestsDayLeavesAnotherAppsAddressBucketsAlone() {
	// The Redis the limiter counts in is every suite's, and another suite's day is its own.
	other := "quota:2000-01-01:ip:" + s.app + "-other:127.0.0.1"
	ctx := s.T().Context()
	s.Require().NoError(s.redis.Do(ctx, s.redis.B().Set().Key(other).Value("1").Ex(time.Minute).Build()).Error())
	s.T().Cleanup(func() { s.redis.Do(context.Background(), s.redis.B().Del().Key(other).Build()) })

	s.SetupTest()

	kept, err := s.redis.Do(ctx, s.redis.B().Exists().Key(other).Build()).AsInt64()
	s.Require().NoError(err)
	s.EqualValues(1, kept)
}

func (s *ProxyQuotaSuite) TestOneUserSpendingTheirDayDoesNotSpendEveryoneElsesBehindTheProxy() {
	// A response is charged to every bucket its caller falls in, and before the fix that
	// included the proxy's address.
	router := s.router(nil)
	s.spend(routing.Caller{UserID: "first", IP: "127.0.0.1"})

	s.Equal(http.StatusTooManyRequests, s.openSession(router, "first", ""))
	s.NotEqual(http.StatusTooManyRequests, s.openSession(router, "second", ""),
		"the proxy's address is everybody's, so it is not a bucket")
}

func (s *ProxyQuotaSuite) TestAGuestTheProxyNamesNobodyForIsStillCounted() {
	// A guest has no user id, so the address is the only bucket they can be counted in.
	router := s.router(nil)
	s.spend(routing.Caller{IP: "127.0.0.1"})

	s.Equal(http.StatusTooManyRequests, s.openSession(router, "", ""))
}

func (s *ProxyQuotaSuite) TestAnAddressTheProxyVouchesForIsStillCounted() {
	// With the proxy's ranges configured the forwarded address is the caller's own, which
	// is what catches a backend that mints a fresh user id per request.
	router := s.router([]netip.Prefix{netip.MustParsePrefix("127.0.0.0/8")})
	s.spend(routing.Caller{IP: "203.0.113.9"})

	s.Equal(http.StatusTooManyRequests, s.openSession(router, "fresh-user", "203.0.113.9"))
	s.NotEqual(http.StatusTooManyRequests, s.openSession(router, "fresh-user", "203.0.113.10"))
}

func (s *ProxyQuotaSuite) router(trusted []netip.Prefix) *httptest.Server {
	server, err := NewServer(Options{
		Routers:        map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		Auth:           auth.NewProxy(auth.ProxyOptions{DeclaresKind: true}),
		AuthMode:       auth.Proxy,
		Quota:          s.limiter,
		TrustedProxies: trusted,
		Logger:         slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	running := httptest.NewServer(server.Handler())
	s.T().Cleanup(running.Close)
	return running
}

// spend uses up a bucket's day the way a response does.
func (s *ProxyQuotaSuite) spend(caller routing.Caller) {
	s.limiter.Debit(s.T().Context(), s.app, caller, 1)
}

// openSession asks for a session as the gateway forwards a user's token, or a guest's when
// user is empty, from the proxy's address and, when forwardedFor is set, naming the client
// it forwarded for.
func (s *ProxyQuotaSuite) openSession(router *httptest.Server, user, forwardedFor string) int {
	request, err := http.NewRequest(http.MethodPost, router.URL+"/v1/agents/sessions", strings.NewReader("{}"))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(auth.AppHeader, s.app)
	request.Header.Set(auth.OrganizationHeader, "7")
	if user != "" {
		request.Header.Set(auth.UserHeader, user)
		request.Header.Set(auth.AuthTypeHeader, auth.AuthTypeJWT)
	} else {
		request.Header.Set(auth.AuthTypeHeader, auth.AuthTypeAnonymous)
	}
	if forwardedFor != "" {
		request.Header.Set(ForwardedHeader, forwardedFor)
	}
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}
