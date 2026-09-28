// Package config holds the settings the router runs with: where Postgres and Redis are,
// how a caller proves who it is, and where the capability files live.
//
// A deployment writes them in a YAML file and points the router at it, with `--config
// /etc/router.yaml` or ROUTER_CONFIG_FILE. Saying nothing picks one of the files embedded
// here by ROUTER_ENV: a laptop, the test suites and the hosted staging deployment each
// have one, and none of them holds a secret.
//
// Every ROUTER_ variable the router used to read directly still wins over the file, so a
// chart, compose and .env keep working with nothing changed. The effective settings are
// written back into the environment on the way out, because the other commands in this
// repository and the packages that hold their own credentials still read them there.
package config

import (
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"strings"
	"time"

	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/env/v2"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/providers/rawbytes"
	"github.com/knadh/koanf/v2"
)

const (
	// EnvVar names which embedded file to start from. Unset means Local.
	EnvVar = "ROUTER_ENV"
	// FileVar is a path to a file of the deployment's own, and is what --config sets.
	FileVar = "ROUTER_CONFIG_FILE"
)

const (
	Local   = "local"
	Testing = "testing"
	Staging = "staging"
)

//go:embed *.yaml
var files embed.FS

// Config is everything the router reads before it serves anything.
type Config struct {
	// Addr is where to listen, which behind a load balancer is not where anyone reaches
	// the router. PublicURL is that, and the telephony vendors that fetch a call plan on
	// answer need it.
	Addr      string `koanf:"addr"`
	PublicURL string `koanf:"public_url"`
	LogLevel  string `koanf:"log_level"`
	// DashboardURL is where a finished plugin login sends the browser.
	DashboardURL string   `koanf:"dashboard_url"`
	CORSOrigins  []string `koanf:"cors_origins"`
	// TrustedProxies are the CIDR ranges this deployment's own proxies sit in, and they
	// decide how much of X-Forwarded-For is believed. Empty means none of it is.
	TrustedProxies []string `koanf:"trusted_proxies"`
	// RoutingConfig and PhoneConfig are paths to the capability files. Empty means the
	// ones embedded in the binary.
	RoutingConfig string `koanf:"routing_config"`
	PhoneConfig   string `koanf:"phone_config"`
	// VoicesBucketURL is where the recordings behind a customer's own voice are kept.
	VoicesBucketURL string    `koanf:"voices_bucket_url"`
	Postgres        Postgres  `koanf:"postgres"`
	Redis           Redis     `koanf:"redis"`
	Auth            Auth      `koanf:"auth"`
	RateLimit       RateLimit `koanf:"rate_limit"`
	DataMove        DataMove  `koanf:"data_move"`
	Stream          Stream    `koanf:"stream"`
}

// Postgres is where everything worth keeping is written. An empty DSN is a router that
// records nothing, which /health reports.
type Postgres struct {
	DSN string `koanf:"dsn"`
}

// Redis holds live provider health and the daily counters. An empty address is a router
// that routes without them.
type Redis struct {
	Addr     string `koanf:"addr"`
	Username string `koanf:"username"`
	Password string `koanf:"password"`
}

// Auth decides who the router believes a caller is.
type Auth struct {
	// Mode is api_key, proxy, noauth or custom. Empty means api_key.
	Mode string `koanf:"mode"`
	// KEK unseals the stored key secrets. It belongs in the environment rather than in a
	// file checked in beside the code: it is what makes a leaked backup ciphertext.
	KEK string `koanf:"kek"`
}

// RateLimit caps what one of a customer's end users may spend in a day. Either at 0 turns
// that half off.
type RateLimit struct {
	MessagesPerDay int64 `koanf:"messages_per_day"`
	TokensPerDay   int64 `koanf:"tokens_per_day"`
}

// DataMove is how long a customer moving to another deployment has to finish.
type DataMove struct {
	// Retention is how long recorded changes are kept, and how long an export keeps
	// recording them for. A move that takes longer than this starts again from an
	// export rather than resuming from a cursor nothing can honour.
	Retention time.Duration `koanf:"retention"`
}

// Stream is the app whose secret signs the call events Stream sends to the inbound hook,
// and the tokens a browser joins a call with.
type Stream struct {
	APIKey    string `koanf:"api_key"`
	APISecret string `koanf:"api_secret"`
}

// variables maps each setting to the environment variable that has always carried it.
// Both directions are read from here: the variable wins over the file on the way in, and
// the effective value is written back to it on the way out.
var variables = map[string]string{
	"addr":                "ROUTER_ADDR",
	"public_url":          "ROUTER_PUBLIC_URL",
	"log_level":           "ROUTER_LOG_LEVEL",
	"dashboard_url":       "DASHBOARD_BASE_URL",
	"cors_origins":        "ROUTER_CORS_ORIGINS",
	"trusted_proxies":     "ROUTER_TRUSTED_PROXIES",
	"routing_config":      "ROUTER_CONFIG",
	"phone_config":        "ROUTER_PHONE_CONFIG",
	"voices_bucket_url":   "ROUTER_VOICES_BUCKET_URL",
	"postgres.dsn":        "ROUTER_POSTGRES_DSN",
	"redis.addr":          "ROUTER_REDIS_ADDR",
	"redis.username":      "ROUTER_REDIS_USERNAME",
	"redis.password":      "ROUTER_REDIS_PASSWORD",
	"auth.mode":           "ROUTER_AUTH_MODE",
	"auth.kek":            "ROUTER_AUTH_KEK",
	"data_move.retention": "ROUTER_DATA_MOVE_RETENTION",
	"stream.api_key":      "STREAM_API_KEY",
	"stream.api_secret":   "STREAM_API_SECRET",

	"rate_limit.messages_per_day": "ROUTER_RATE_LIMIT_MESSAGES_PER_DAY",
	"rate_limit.tokens_per_day":   "ROUTER_RATE_LIMIT_TOKENS_PER_DAY",
}

// lists are the settings written as a comma-separated variable and as a sequence in YAML.
var lists = map[string]bool{"cors_origins": true, "trusted_proxies": true}

// Defaults are what a deployment gets for saying nothing at all.
func Defaults() Config {
	return Config{
		Addr:         ":8080",
		DashboardURL: "http://localhost:3000",
		// A day's allowance for one end user. The token limit is a backstop under the
		// message count rather than a second cap: at roughly 2,500 tokens for a turn
		// carrying instructions and some history, 200 messages is about 500,000 tokens,
		// so it should only be reached by somebody making a few enormous requests.
		RateLimit: RateLimit{MessagesPerDay: 200, TokensPerDay: 500_000},
		DataMove:  DataMove{Retention: 7 * 24 * time.Hour},
	}
}

// Load reads the settings a deployment runs with.
//
// path is the file the deployment named on the command line, and empty means the one
// FileVar names, or the embedded file for ROUTER_ENV. It returns the settings and the
// name of what they came from, for the line the router logs on the way up.
func Load(path string) (Config, string, error) {
	name := strings.TrimSpace(os.Getenv(EnvVar))
	switch name {
	case "":
		name = Local
	// The file was called development before it was one of several a self-hosted
	// deployment chooses between, and a chart still says so.
	case "development":
		name = Local
	}

	embedded, err := files.ReadFile(name + ".yaml")
	if errors.Is(err, fs.ErrNotExist) {
		return Config{}, "", fmt.Errorf("config: unknown %s %q, want %s, %s or %s", EnvVar, name, Local, Testing, Staging)
	}
	if err != nil {
		return Config{}, "", err
	}

	k := koanf.New(".")
	if err := k.Load(rawbytes.Provider(embedded), yaml.Parser()); err != nil {
		return Config{}, "", fmt.Errorf("config: parse %s: %w", name, err)
	}

	if path == "" {
		path = strings.TrimSpace(os.Getenv(FileVar))
	}
	if path != "" {
		if err := k.Load(file.Provider(path), yaml.Parser()); err != nil {
			return Config{}, "", fmt.Errorf("config: read %s: %w", path, err)
		}
		name = path
	}

	if err := k.Load(environment(), nil); err != nil {
		return Config{}, "", err
	}

	// The test suites are the one place a file wins over the environment: the store suite
	// drops the whole schema, so it must never be pointed at the database a local router
	// is using, however the shell that started it is set up.
	if name == Testing {
		if err := k.Load(rawbytes.Provider(embedded), yaml.Parser()); err != nil {
			return Config{}, "", err
		}
	}

	config := Defaults()
	if err := k.Unmarshal("", &config); err != nil {
		return Config{}, "", fmt.Errorf("config: %s: %w", name, err)
	}
	if err := config.validate(); err != nil {
		return Config{}, "", err
	}

	if err := config.export(); err != nil {
		return Config{}, "", err
	}
	return config, name, nil
}

// environment reads the ROUTER_ variables this package knows, and nothing else: a
// deployment's environment holds every provider credential too, and those belong to the
// packages that read them rather than here.
func environment() *env.Env {
	byVariable := make(map[string]string, len(variables))
	for key, variable := range variables {
		byVariable[variable] = key
	}
	return env.Provider(".", env.Opt{
		TransformFunc: func(variable, value string) (string, any) {
			key, known := byVariable[variable]
			if !known {
				return "", nil
			}
			if lists[key] {
				return key, splitList(value)
			}
			return key, value
		},
	})
}

// validate refuses settings that would otherwise be found out from a customer.
func (c Config) validate() error {
	if c.RateLimit.MessagesPerDay < 0 || c.RateLimit.TokensPerDay < 0 {
		return fmt.Errorf("config: a daily limit cannot be negative, got %d messages and %d tokens",
			c.RateLimit.MessagesPerDay, c.RateLimit.TokensPerDay)
	}
	if c.DataMove.Retention < 0 {
		return fmt.Errorf("config: data_move.retention cannot be negative, got %s", c.DataMove.Retention)
	}
	return nil
}

// export writes the settings back into the environment.
//
// The router is not the only thing that reads them: cmd/agent and cmd/phone run against
// the same database, the integration suites find it the same way, and the packages
// holding a bucket or a provider credential read their own variable. One loader that
// leaves the environment as it found it would mean two answers to where Postgres is.
func (c Config) export() error {
	values := map[string]string{
		"addr":                        c.Addr,
		"public_url":                  c.PublicURL,
		"log_level":                   c.LogLevel,
		"dashboard_url":               c.DashboardURL,
		"cors_origins":                strings.Join(c.CORSOrigins, ","),
		"trusted_proxies":             strings.Join(c.TrustedProxies, ","),
		"routing_config":              c.RoutingConfig,
		"phone_config":                c.PhoneConfig,
		"voices_bucket_url":           c.VoicesBucketURL,
		"postgres.dsn":                c.Postgres.DSN,
		"redis.addr":                  c.Redis.Addr,
		"redis.username":              c.Redis.Username,
		"redis.password":              c.Redis.Password,
		"auth.mode":                   c.Auth.Mode,
		"auth.kek":                    c.Auth.KEK,
		"stream.api_key":              c.Stream.APIKey,
		"stream.api_secret":           c.Stream.APISecret,
		"data_move.retention":         c.DataMove.Retention.String(),
		"rate_limit.messages_per_day": fmt.Sprint(c.RateLimit.MessagesPerDay),
		"rate_limit.tokens_per_day":   fmt.Sprint(c.RateLimit.TokensPerDay),
	}
	for key, value := range values {
		if value == "" {
			continue
		}
		if err := os.Setenv(variables[key], value); err != nil {
			return err
		}
	}
	return nil
}

// splitList reads a comma-separated variable, dropping the empty entries a trailing comma
// leaves behind.
func splitList(raw string) []string {
	var entries []string
	for _, entry := range strings.Split(raw, ",") {
		if trimmed := strings.TrimSpace(entry); trimmed != "" {
			entries = append(entries, trimmed)
		}
	}
	return entries
}
