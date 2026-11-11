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

func (s *LimitsSuite) TestTheCostAnomalyCheckRunsWhereThereIsAChannelAndPostgres() {
	logger := slog.New(slog.DiscardHandler)
	db := &store.Store{}
	s.settings.Limits.SlackToken, s.settings.Limits.SlackChannel = "xoxb-test", "C123"
	s.NotNil(costAnomalies(s.settings, db, logger))
	s.settings.Limits.Mode = config.LimitsOff
	s.NotNil(costAnomalies(s.settings, db, logger), "spend is priced and recorded with limits off too")

	s.Nil(costAnomalies(s.settings, nil, logger), "no Postgres to read the spend from")
	s.settings.Limits.SlackToken = " "
	s.Nil(costAnomalies(s.settings, db, logger), "no token")
	s.settings.Limits.SlackToken, s.settings.Limits.SlackChannel = "xoxb-test", ""
	s.Nil(costAnomalies(s.settings, db, logger), "no channel")
}

// The message names its environment and links to the staff page, so an operator reading it in the
// shared channel knows which deployment it is from and where to look.
func (s *LimitsSuite) TestTheCostAnomalyCheckNamesItsEnvironmentAndLinksTheStaffPage() {
	s.T().Setenv("SENTRY_ENVIRONMENT", "staging-test")
	s.settings.Limits.SlackToken, s.settings.Limits.SlackChannel = "xoxb-test", "C123"
	s.settings.Limits.StaffURL = "https://staff.example.test/orgs"
	logger := slog.New(slog.DiscardHandler)
	db := &store.Store{}

	options, ok := costAnomalyOptions(s.settings, db, logger)

	s.Require().True(ok)
	s.Equal("staging-test", options.Environment)
	s.Equal(s.settings.Limits.StaffURL, options.Links.Staff)
	s.Same(db, options.Store)
}

func (s *LimitsSuite) TestVerificationLinkKeepsOnlyTheDashboardsOrigin() {
	s.Equal("https://dash.example/internal/agents/organizations/{organization}",
		verificationLink("https://dash.example/{organization}/{app}"))
	s.Equal("http://localhost:3000/internal/agents/organizations/{organization}",
		verificationLink("http://localhost:3000/x"))
}

func (s *LimitsSuite) TestVerificationLinkIsEmptyWithoutAnAbsoluteAppURL() {
	for _, appURL := range []string{"", "/relative/{organization}", "::bad"} {
		s.Empty(verificationLink(appURL), appURL)
	}
}

func (s *LimitsSuite) TestVerificationNoticesNeedATokenAndAChannel() {
	s.settings.Limits.Mode = config.LimitsOff
	s.settings.Limits.SlackToken = "xoxb-token"
	s.settings.Limits.SlackChannel = "  "

	s.True(verificationNotices(s.settings).Poster == nil, "a token alone posts nothing")

	s.settings.Limits.SlackChannel = "C123"
	s.NotNil(verificationNotices(s.settings).Poster, "posts even with limits off")

	s.settings.Limits.SlackToken = " "
	s.True(verificationNotices(s.settings).Poster == nil, "a channel alone posts nothing")
}
