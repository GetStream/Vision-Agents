package main

import (
	"bytes"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// LimitsSuite covers what the router refuses to start with about spend limits, and what it
// warns of.
type LimitsSuite struct {
	suite.Suite
	settings config.Config
}

func TestLimitsSuite(t *testing.T) {
	suite.Run(t, new(LimitsSuite))
}

func (s *LimitsSuite) SetupTest() {
	s.settings = config.Defaults()
}

func (s *LimitsSuite) TestEnforceWithoutPostgresIsRefused() {
	for _, mode := range []config.LimitsMode{config.LimitsEnforce, config.LimitsShadow} {
		s.settings.Limits.Mode = mode

		err := limitsNeedSomewhereToRead(s.settings, nil)

		s.Require().ErrorContains(err, "ROUTER_POSTGRES_DSN", mode)
		s.ErrorContains(err, "ROUTER_LIMITS_MODE=off", "says how to run without them")
	}
}

func (s *LimitsSuite) TestAHostedRouterWithoutAUsageServiceIsRefused() {
	s.settings.Auth.Mode = string(auth.Proxy)

	err := limitsNeedSomewhereToRead(s.settings, &store.Store{})

	s.Require().ErrorContains(err, "ROUTER_USAGE_URL")
	s.settings.Usage.URL = "http://usage:8070"
	s.NoError(limitsNeedSomewhereToRead(s.settings, &store.Store{}))
}

func (s *LimitsSuite) TestASelfHostedRouterReadsItsSpendFromPostgres() {
	s.settings.Auth.Mode = string(auth.APIKey)

	s.NoError(limitsNeedSomewhereToRead(s.settings, &store.Store{}), "no usage service is no credit to check")
}

func (s *LimitsSuite) TestOffWithoutPostgresStarts() {
	s.settings.Limits.Mode = config.LimitsOff
	s.settings.Auth.Mode = string(auth.Proxy)

	s.NoError(limitsNeedSomewhereToRead(s.settings, nil))
}

func (s *LimitsSuite) TestOffAndShadowAreWarnedOf() {
	for mode, warned := range map[config.LimitsMode]string{
		config.LimitsOff:     "spend limits are off",
		config.LimitsShadow:  "spend limits are in shadow",
		config.LimitsEnforce: "",
	} {
		var logs bytes.Buffer
		s.settings.Limits.Mode = mode

		warnLimits(s.settings, slog.New(slog.NewTextHandler(&logs, nil)))

		if warned == "" {
			s.Empty(logs.String(), mode)
			continue
		}
		s.Contains(logs.String(), "level=WARN", mode)
		s.Contains(logs.String(), warned, mode)
		s.Contains(logs.String(), "ROUTER_LIMITS_MODE", mode)
	}
}
