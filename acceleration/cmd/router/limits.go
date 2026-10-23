package main

import (
	"errors"
	"log/slog"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	usageservice "github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// warnLimits says at startup when the router refuses nothing for money, or only logs what it
// would refuse.
func warnLimits(settings config.Config, logger *slog.Logger) {
	switch settings.Limits.Mode {
	case config.LimitsOff:
		logger.Warn("spend limits are off: credit, the spend hard limit and budgets refuse nothing, so "+
			"nothing but a laptop should run this way", "set", "ROUTER_LIMITS_MODE")
	case config.LimitsShadow:
		logger.Warn("spend limits are in shadow: refusals are logged as would refuse and "+
			"the request is admitted", "set", "ROUTER_LIMITS_MODE")
	}
}

// limitsNeedSomewhereToRead refuses a router that would enforce limits it cannot read: with no
// Postgres there is no spend, and behind a proxy with no usage service there is no credit, so
// the router would sell work nobody can bill.
func limitsNeedSomewhereToRead(settings config.Config, pgStore *store.Store) error {
	if settings.Limits.Mode == config.LimitsOff {
		return nil
	}
	if pgStore == nil {
		return errors.New("spend limits need ROUTER_POSTGRES_DSN; set ROUTER_LIMITS_MODE=off to run without them")
	}
	if strings.TrimSpace(settings.Auth.Mode) == string(auth.Proxy) && settings.Usage.URL == "" {
		return errors.New("behind the proxy, spend limits need ROUTER_USAGE_URL; set ROUTER_LIMITS_MODE=off to run without them")
	}
	return nil
}

// newPolicies is the enforcer every router, LLM response and session asks, reading money from
// the usage service when there is one. Nil without a configuration store, which
// limitsNeedSomewhereToRead allows only with limits off.
func newPolicies(settings config.Config, configs *appconfig.Store, usageClient *usageservice.Client, logger *slog.Logger) (*policy.Enforcer, error) {
	if configs == nil {
		return nil, nil
	}
	return policy.New(configs, policy.Settings{Mode: settings.Limits.Mode, Usage: usageClient}, logger)
}
