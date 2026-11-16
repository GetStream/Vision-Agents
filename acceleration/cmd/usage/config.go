package main

import (
	"errors"

	"github.com/knadh/koanf/providers/env/v2"
	"github.com/knadh/koanf/v2"
)

// defaultAddr is the API's listener when USAGE_ADDR is not set.
const defaultAddr = ":8070"

// settings are what the service runs with, read from the environment alone.
type settings struct {
	// Addr serves the API.
	Addr string `koanf:"addr"`
	// MetricsAddr serves /metrics on a listener of its own; empty serves none.
	MetricsAddr string `koanf:"metrics_addr"`
	// PostgresDSN is the service's own database.
	PostgresDSN string `koanf:"postgres_dsn"`
	// IngestToken is the bearer token of /v1/events, /v1/windows and /v1/standings.
	IngestToken string `koanf:"ingest_token"`
	// RedisURL is where window totals are mirrored, the routers' Redis; empty mirrors
	// nothing, and routers read through /v1/windows alone.
	RedisURL string `koanf:"redis_url"`
	// ManageKey is what the management API takes in X-Manage-Key; empty turns it off.
	ManageKey string `koanf:"manage_key"`
	// TiersFile is the tier file whose tiers the service adds at start when it lacks them;
	// empty adds only the built-in default tier.
	TiersFile string `koanf:"tiers_file"`
}

// variables maps each environment variable to the setting it carries. Nothing else in the
// environment is read here: SENTRY_DSN and SENTRY_ENVIRONMENT are the Sentry SDK's own.
var variables = map[string]string{
	"USAGE_ADDR":         "addr",
	"USAGE_METRICS_ADDR": "metrics_addr",
	"USAGE_POSTGRES_DSN": "postgres_dsn",
	"USAGE_INGEST_TOKEN": "ingest_token",
	"USAGE_REDIS_URL":    "redis_url",
	"USAGE_MANAGE_KEY":   "manage_key",
	"USAGE_TIERS_FILE":   "tiers_file",
}

// load reads the settings the service serves with from environ.
func load(environ func() []string) (settings, error) {
	loaded, err := read(environ)
	if err != nil {
		return settings{}, err
	}
	if loaded.PostgresDSN == "" {
		return settings{}, errors.New("usage: USAGE_POSTGRES_DSN is required: the service's own database")
	}
	if loaded.IngestToken == "" {
		return settings{}, errors.New("usage: USAGE_INGEST_TOKEN is required: the bearer token producers and routers call with")
	}
	if loaded.ManageKey == loaded.IngestToken {
		return settings{}, errors.New("usage: USAGE_MANAGE_KEY cannot be USAGE_INGEST_TOKEN, which every producer holds")
	}
	return loaded, nil
}

// read reads whatever settings environ has, requiring none.
func read(environ func() []string) (settings, error) {
	k := koanf.New(".")
	provider := env.Provider(".", env.Opt{
		Prefix: "USAGE_",
		TransformFunc: func(variable, value string) (string, any) {
			return variables[variable], value
		},
		EnvironFunc: environ,
	})
	if err := k.Load(provider, nil); err != nil {
		return settings{}, err
	}
	loaded := settings{Addr: defaultAddr}
	if err := k.Unmarshal("", &loaded); err != nil {
		return settings{}, err
	}
	return loaded, nil
}
