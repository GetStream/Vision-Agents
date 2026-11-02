//go:build integration

package llmrouter

import (
	"context"
	"crypto/rand"
	"log/slog"
	"os"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
)

// slowLimits is a quota source that takes a while to say what a user is held to, as a policy
// store read can.
type slowLimits time.Duration

func (d slowLimits) Quota(context.Context, string, string, quota.Limits) quota.Quota {
	time.Sleep(time.Duration(d))
	return quota.Default(quota.Limits{MessagesPerDay: 1_000})
}

func (s *StagesSuite) TestTheRouterStageCountsTheDailyLimitCheck() {
	// The daily limit is asked on every turn, a Redis round trip in production: time the
	// router spends before it asks the provider, which only the router stage can show.
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR must be set")
	}
	client, err := redisready.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	const deciding = 200 * time.Millisecond
	limiter, err := quota.New(client, quota.Limits{}, slowLimits(deciding), slog.Default())
	s.Require().NoError(err)
	config := s.config()
	session := s.session(s.provider(answering, 0, 0, answer), config)
	session.quota = limiter
	session.owner.Caller = routing.Caller{UserID: "user-" + rand.Text()}

	s.Equal("hi", s.ask(s.T().Context(), session).OutputText)

	count, took := s.stage(config, config.Model, routing.StageRouter)
	s.EqualValues(1, count)
	// The histogram keeps seconds as a float, which can read a nanosecond short.
	s.GreaterOrEqual(took, deciding-time.Microsecond)
}
