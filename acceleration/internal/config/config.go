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
	"math"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/eotdefaults"
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
	VoicesBucketURL string     `koanf:"voices_bucket_url"`
	Postgres        Postgres   `koanf:"postgres"`
	Redis           Redis      `koanf:"redis"`
	Node            Node       `koanf:"node"`
	Auth            Auth       `koanf:"auth"`
	RateLimit       RateLimit  `koanf:"rate_limit"`
	DataMove        DataMove   `koanf:"data_move"`
	Stream          Stream     `koanf:"stream"`
	EOT             EOT        `koanf:"eot"`
	Agent           Agent      `koanf:"agent"`
	Connectors      Connectors `koanf:"connectors"`
	Sandbox         Sandbox    `koanf:"sandbox"`
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

// Node is how one process of a deployment is reached by the others, which is what lets a
// request for a session land on any of them.
type Node struct {
	// Advertise is the host and port this node's peers reach it at, which is not Addr:
	// Addr is where to listen, and a node listening on every interface still has one
	// address its peers use. Empty means this host's own address and the port from Addr,
	// which is right wherever a pod's address is reachable from its peers.
	Advertise string `koanf:"advertise"`
}

// Auth decides who the router believes a caller is.
type Auth struct {
	// Mode is api_key, proxy, noauth or custom. Empty means api_key.
	Mode string `koanf:"mode"`
	// KEK unseals the stored key secrets. It belongs in the environment rather than in a
	// file checked in beside the code: it is what makes a leaked backup ciphertext.
	KEK string `koanf:"kek"`
	// ProxyDeclaresKind says the proxy in front authenticates every caller and declares
	// whether it is a backend or an end user. In proxy mode the router then reads only the
	// app header and takes a caller that declares nothing for an end user. Off, a caller
	// that declares nothing is a backend, as a proxy that never declares means.
	ProxyDeclaresKind bool `koanf:"proxy_declares_kind"`
	// OpsKey is what Stream's own staff tools send as X-Ops-Key to review use cases. Empty
	// turns those endpoints off. It is never handed to a browser.
	OpsKey string `koanf:"ops_key"`
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

// Stream is the deployment's own Stream app: the one the router acts in for every
// customer in deployment mode, and for its own customer in app mode.
type Stream struct {
	APIKey    string `koanf:"api_key"`
	APISecret string `koanf:"api_secret"`
	// UserToken is a fixed token the voice edge has always preferred to minting its own,
	// for the deployment's app only.
	UserToken string `koanf:"user_token"`
	// BaseURL is the Stream API the deployment's app is reached at. Empty is Stream's
	// default. It is read here once, so every app's client is told where to go rather than
	// each reading the environment for itself.
	BaseURL string `koanf:"base_url"`
	// Tenancy says whose app the router acts in. deployment, the default, is the
	// deployment's own app for every customer, as it always was. app is each customer's
	// own, registered with its keys.
	Tenancy string `koanf:"tenancy"`
	// Fallback is what app mode does for a customer that registered no app: deployment
	// writes it into the deployment's own app, as before, and refuse writes it nowhere.
	// Unset is refuse. Deployment mode never reads it.
	Fallback string `koanf:"fallback"`
	// AppID is the deployment's own app's id, which Stream is asked for when it is not
	// set. A pin naming it is finished with the deployment's own key, in either mode.
	AppID int64 `koanf:"app_id"`
	// TrustAPIKeyHeader lets the X-Stream-Api-Key a gateway forwards choose which of the
	// calling app's registered keys mints its tokens. Off, the primary key always does.
	// Only a gateway that writes that header itself, rather than passing a caller's on,
	// may have it on.
	TrustAPIKeyHeader bool `koanf:"trust_api_key_header"`
	// DenyRegistration are Stream app ids that may never be registered as a customer's own.
	DenyRegistration []string `koanf:"deny_registration"`
}

// What Stream.Tenancy holds.
const (
	TenancyDeployment = "deployment"
	TenancyApp        = "app"
)

// What Stream.Fallback holds.
const (
	FallbackDeployment = "deployment"
	FallbackRefuse     = "refuse"
)

// EffectiveFallback is what app mode does for a customer with no app of its own: refuse
// unless the deployment said otherwise, so leaving it out fails closed.
func (s Stream) EffectiveFallback() string {
	if s.Fallback == "" {
		return FallbackRefuse
	}
	return s.Fallback
}

// Agent is how an agent holds a conversation, where that is the deployment's choice.
type Agent struct {
	// SpeculativeReplies starts a reply while the flow controller is still deciding
	// whether the words were meant for the agent, and holds it until the ruling says to
	// answer. It saves the ruling's round trip on every answered turn and pays for the
	// replies a ruling throws away. On by default; false asks for each reply only once the
	// ruling is in.
	SpeculativeReplies bool `koanf:"speculative_replies"`
}

// Connectors is whether agents may reach the customer's accounts elsewhere.
type Connectors struct {
	// Enabled builds the versioned key encryption keyring that seals connector
	// credentials, in every auth mode, and refuses to start without one. Off by default.
	Enabled bool `koanf:"enabled"`
}

// Sandbox holds an app with no approved 10DLC use case to a few numbers and a little
// traffic. It is for the hosted router: a self-hosted one registers, or not, on its own
// account, and only opt-outs are enforced there.
type Sandbox struct {
	Enabled            bool  `koanf:"enabled"`
	Recipients         int   `koanf:"recipients"`
	MessagesPerDay     int64 `koanf:"messages_per_day"`
	AudioMinutesPerDay int64 `koanf:"audio_minutes_per_day"`
}

// EOT is the optional acoustic endpoint scorer for settled cascade turns.
type EOT struct {
	Mode        string  `koanf:"mode"`
	Endpoint    string  `koanf:"endpoint"`
	IDTokenFile string  `koanf:"id_token_file"`
	Threshold   float64 `koanf:"threshold"`
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
	"node.advertise":      "ROUTER_NODE_ADVERTISE",
	"auth.mode":           "ROUTER_AUTH_MODE",
	"auth.kek":            "ROUTER_AUTH_KEK",
	"auth.ops_key":        "ROUTER_AUTH_OPS_KEY",
	"data_move.retention": "ROUTER_DATA_MOVE_RETENTION",
	"stream.api_key":      "STREAM_API_KEY",
	"stream.api_secret":   "STREAM_API_SECRET",
	"stream.base_url":     "STREAM_BASE_URL",
	"stream.user_token":   "STREAM_USER_TOKEN",
	"stream.tenancy":      "ROUTER_STREAM_TENANCY",
	"stream.fallback":     "ROUTER_STREAM_FALLBACK",
	"stream.app_id":       "ROUTER_STREAM_APP_ID",

	"stream.trust_api_key_header": "ROUTER_STREAM_TRUST_API_KEY_HEADER",
	"stream.deny_registration":    "ROUTER_STREAM_DENY_REGISTRATION",

	"eot.mode":          "ROUTER_EOT_MODE",
	"eot.endpoint":      "ROUTER_EOT_URL",
	"eot.id_token_file": "ROUTER_EOT_ID_TOKEN_FILE",
	"eot.threshold":     "ROUTER_EOT_THRESHOLD",

	"rate_limit.messages_per_day": "ROUTER_RATE_LIMIT_MESSAGES_PER_DAY",
	"rate_limit.tokens_per_day":   "ROUTER_RATE_LIMIT_TOKENS_PER_DAY",

	"agent.speculative_replies": "ROUTER_SPECULATIVE_REPLIES",
	"auth.proxy_declares_kind":  "ROUTER_AUTH_PROXY_DECLARES_KIND",
	"connectors.enabled":        "ROUTER_CONNECTORS_ENABLED",

	"sandbox.enabled":               "ROUTER_SANDBOX_ENABLED",
	"sandbox.recipients":            "ROUTER_SANDBOX_RECIPIENTS",
	"sandbox.messages_per_day":      "ROUTER_SANDBOX_MESSAGES_PER_DAY",
	"sandbox.audio_minutes_per_day": "ROUTER_SANDBOX_AUDIO_MINUTES_PER_DAY",
}

// lists are the settings written as a comma-separated variable and as a sequence in YAML.
var lists = map[string]bool{"cors_origins": true, "trusted_proxies": true, "stream.deny_registration": true}

// Defaults are what a deployment gets for saying nothing at all.
func Defaults() Config {
	return Config{
		Addr:         ":8080",
		DashboardURL: "http://localhost:3000",
		// A day's allowance for one end user. The token limit is a backstop under the
		// message count rather than a second cap: an agent with a few MCP servers sends
		// tens of thousands of tokens of tool definitions with every turn, so 200 messages
		// can come to millions of tokens.
		RateLimit: RateLimit{MessagesPerDay: 200, TokensPerDay: 5_000_000},
		DataMove:  DataMove{Retention: 7 * 24 * time.Hour},
		Agent:     Agent{SpeculativeReplies: true},
		EOT:       EOT{Endpoint: eotdefaults.HostedDemoEndpoint, Mode: "primary", Threshold: 0.5},
		Sandbox:   Sandbox{Recipients: 2, MessagesPerDay: 30, AudioMinutesPerDay: 30},
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
	modeExplicit := k.Exists("eot.mode")

	config := Defaults()
	if err := k.Unmarshal("", &config); err != nil {
		return Config{}, "", fmt.Errorf("config: %s: %w", name, err)
	}
	config.EOT.Endpoint = strings.TrimSpace(config.EOT.Endpoint)
	config.EOT.IDTokenFile = strings.TrimSpace(config.EOT.IDTokenFile)
	config.EOT.Mode = strings.TrimSpace(config.EOT.Mode)
	if !modeExplicit {
		if eotdefaults.IsHostedDemoEndpoint(config.EOT.Endpoint) {
			config.EOT.Mode = "primary"
		} else {
			config.EOT.Mode = "gate"
		}
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
	switch c.Stream.Tenancy {
	case "", TenancyDeployment:
	case TenancyApp:
		if err := c.validateAppTenancy(); err != nil {
			return err
		}
	default:
		return fmt.Errorf("config: stream.tenancy is %s or %s, got %q", TenancyDeployment, TenancyApp, c.Stream.Tenancy)
	}
	switch c.Stream.Fallback {
	case "", FallbackDeployment, FallbackRefuse:
	default:
		return fmt.Errorf("config: stream.fallback is %s or %s, got %q", FallbackDeployment, FallbackRefuse, c.Stream.Fallback)
	}
	if c.Stream.AppID < 0 {
		return fmt.Errorf("config: stream.app_id is a Stream app's id, got %d", c.Stream.AppID)
	}
	if c.DataMove.Retention < 0 {
		return fmt.Errorf("config: data_move.retention cannot be negative, got %s", c.DataMove.Retention)
	}
	if c.EOT.Mode != "gate" && c.EOT.Mode != "primary" {
		return fmt.Errorf("config: eot.mode must be gate or primary, got %q", c.EOT.Mode)
	}
	if eotdefaults.IsHostedDemoOrigin(c.EOT.Endpoint) {
		if !eotdefaults.IsHostedDemoEndpoint(c.EOT.Endpoint) {
			return errors.New("config: the hosted demo endpoint path must be /v1/eot")
		}
		if strings.TrimSpace(c.EOT.IDTokenFile) != "" {
			return errors.New("config: eot.id_token_file cannot be used with the hosted demo endpoint; configure a private endpoint")
		}
	}
	if math.IsNaN(c.EOT.Threshold) || math.IsInf(c.EOT.Threshold, 0) ||
		c.EOT.Threshold < 0 || c.EOT.Threshold > 1 {
		return fmt.Errorf("config: eot.threshold must be between 0 and 1, got %v", c.EOT.Threshold)
	}
	return nil
}

// validateAppTenancy refuses an app mode that could not keep its promises. Registered apps
// and their keys live in Postgres. A fixed user token is the deployment's own, and the voice
// edge would prefer it to a token of the app a session is in. And the customer picks whose
// stored Stream secret mints a caller's tokens, so it has to be one nobody could name for
// themselves: noauth, and a proxy that does not declare kinds, take it from the caller's own
// X-Customer-Id.
func (c Config) validateAppTenancy() error {
	if c.Postgres.DSN == "" {
		return fmt.Errorf("config: stream.tenancy=%s keeps every app's keys in Postgres: set postgres.dsn", TenancyApp)
	}
	switch strings.TrimSpace(c.Auth.Mode) {
	case "noauth":
		return fmt.Errorf("config: stream.tenancy=%s cannot use auth.mode=noauth, where a caller names "+
			"its own customer and so whose Stream app it acts in: set auth.mode=api_key, or proxy with "+
			"auth.proxy_declares_kind", TenancyApp)
	case "proxy":
		if !c.Auth.ProxyDeclaresKind {
			return fmt.Errorf("config: stream.tenancy=%s cannot use auth.mode=proxy without "+
				"auth.proxy_declares_kind, which reads a caller's own X-Customer-Id: set "+
				"auth.proxy_declares_kind=true, or auth.mode=api_key", TenancyApp)
		}
	}
	if c.Stream.UserToken != "" {
		return fmt.Errorf("config: stream.tenancy=%s cannot use stream.user_token, which is one "+
			"app's fixed token: unset it", TenancyApp)
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
		"addr":                          c.Addr,
		"public_url":                    c.PublicURL,
		"log_level":                     c.LogLevel,
		"dashboard_url":                 c.DashboardURL,
		"cors_origins":                  strings.Join(c.CORSOrigins, ","),
		"trusted_proxies":               strings.Join(c.TrustedProxies, ","),
		"routing_config":                c.RoutingConfig,
		"phone_config":                  c.PhoneConfig,
		"voices_bucket_url":             c.VoicesBucketURL,
		"postgres.dsn":                  c.Postgres.DSN,
		"redis.addr":                    c.Redis.Addr,
		"redis.username":                c.Redis.Username,
		"redis.password":                c.Redis.Password,
		"node.advertise":                c.Node.Advertise,
		"auth.mode":                     c.Auth.Mode,
		"auth.kek":                      c.Auth.KEK,
		"auth.ops_key":                  c.Auth.OpsKey,
		"auth.proxy_declares_kind":      fmt.Sprint(c.Auth.ProxyDeclaresKind),
		"stream.api_key":                c.Stream.APIKey,
		"stream.api_secret":             c.Stream.APISecret,
		"stream.base_url":               c.Stream.BaseURL,
		"stream.user_token":             c.Stream.UserToken,
		"stream.tenancy":                c.Stream.Tenancy,
		"stream.fallback":               c.Stream.Fallback,
		"stream.app_id":                 appID(c.Stream.AppID),
		"stream.trust_api_key_header":   fmt.Sprint(c.Stream.TrustAPIKeyHeader),
		"stream.deny_registration":      strings.Join(c.Stream.DenyRegistration, ","),
		"eot.endpoint":                  c.EOT.Endpoint,
		"eot.mode":                      c.EOT.Mode,
		"eot.id_token_file":             c.EOT.IDTokenFile,
		"eot.threshold":                 fmt.Sprint(c.EOT.Threshold),
		"data_move.retention":           c.DataMove.Retention.String(),
		"rate_limit.messages_per_day":   fmt.Sprint(c.RateLimit.MessagesPerDay),
		"rate_limit.tokens_per_day":     fmt.Sprint(c.RateLimit.TokensPerDay),
		"agent.speculative_replies":     fmt.Sprint(c.Agent.SpeculativeReplies),
		"connectors.enabled":            fmt.Sprint(c.Connectors.Enabled),
		"sandbox.enabled":               fmt.Sprint(c.Sandbox.Enabled),
		"sandbox.recipients":            fmt.Sprint(c.Sandbox.Recipients),
		"sandbox.messages_per_day":      fmt.Sprint(c.Sandbox.MessagesPerDay),
		"sandbox.audio_minutes_per_day": fmt.Sprint(c.Sandbox.AudioMinutesPerDay),
	}
	for key, value := range values {
		if value == "" && key != "eot.endpoint" {
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

// appID writes an app id back, or nothing for one nobody set.
func appID(id int64) string {
	if id == 0 {
		return ""
	}
	return fmt.Sprint(id)
}
