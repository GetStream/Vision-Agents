package main

import (
	"errors"
	"log/slog"
	"strings"
	"time"

	"github.com/redis/rueidis"

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
		logger.Warn("spend limits are off: credit, the spend hard limit and the customers' spend limits refuse nothing, so "+
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
// the usage service when there is one, and its spend counters from the Redis it keeps them in
// when that is set. What each router holds is shared with the others through holds, the
// deployment's Redis; with none, holds stay in this process. Nil without a configuration
// store, which limitsNeedSomewhereToRead allows only with limits off. notified is told when a
// warning that an organization's credit is running out was queued for its destinations. The
// returned func closes the counters' connection.
func newPolicies(settings config.Config, configs *appconfig.Store, usageClient *usageservice.Client, holds rueidis.Client, notified func(), logger *slog.Logger) (*policy.Enforcer, func(), error) {
	if configs == nil {
		return nil, func() {}, nil
	}
	counters, err := usageCounters(settings, usageClient, logger)
	if err != nil {
		return nil, nil, err
	}
	closeCounters := func() {
		if counters != nil {
			counters.Close()
		}
	}
	enforcer, err := policy.New(configs, policy.Settings{
		Mode: settings.Limits.Mode, Usage: usageClient, Counters: counters, Holds: holds, Notified: notified,
	}, logger)
	if err != nil {
		closeCounters()
		return nil, nil, err
	}
	return enforcer, closeCounters, nil
}

// usageCounters connects to the Redis the usage service keeps its spend counters in, or nil
// when none is set or no usage service is. A Redis that cannot be reached at startup is warned
// of and left out, so the limits are read from the service itself until the router restarts.
func usageCounters(settings config.Config, usageClient *usageservice.Client, logger *slog.Logger) (rueidis.Client, error) {
	if settings.Usage.RedisURL == "" || usageClient == nil || settings.Limits.Mode == config.LimitsOff {
		return nil, nil
	}
	address := settings.Usage.RedisURL
	if !strings.Contains(address, "://") {
		address = "redis://" + address
	}
	options, err := rueidis.ParseURL(address)
	if err != nil {
		// Not wrapped: the parse error quotes the URL, and so its password.
		return nil, errors.New("ROUTER_USAGE_REDIS_URL does not parse: want redis://[user:password@]host:port[/db], rediss:// for TLS, or host:port")
	}
	options.DisableCache = true
	options.Dialer.Timeout = 2 * time.Second
	client, err := rueidis.NewClient(options)
	if err != nil {
		logger.Warn("could not reach the usage service's Redis, so spend limits read the usage service itself",
			"set", "ROUTER_USAGE_REDIS_URL", "error", err)
		return nil, nil
	}
	return client, nil
}
