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
	"maps"
	"net/netip"
	"net/url"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/env/v2"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/providers/rawbytes"
	"github.com/knadh/koanf/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
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
	Addr string `koanf:"addr"`
	// ListenFD is a listening socket this process inherited, served in place of Addr. A
	// supervisor that keeps the socket open across processes passes it, so a new router
	// takes connections from the same queue the old one did and none is refused between
	// them. Zero means listen on Addr.
	ListenFD  int    `koanf:"listen_fd"`
	PublicURL string `koanf:"public_url"`
	LogLevel  string `koanf:"log_level"`
	// DashboardURL is where a finished plugin login sends the browser.
	DashboardURL string `koanf:"dashboard_url"`
	// TrustedProxies are the CIDR ranges this deployment's own proxies sit in, and they
	// decide how much of X-Forwarded-For is believed. Empty means none of it is.
	TrustedProxies []string `koanf:"trusted_proxies"`
	// RoutingConfig and PhoneConfig are paths to the capability files. Empty means the
	// ones embedded in the binary.
	RoutingConfig string `koanf:"routing_config"`
	PhoneConfig   string `koanf:"phone_config"`
	// PricesFile is a price file of the deployment's own, laid over the embedded vendor.json
	// when the router loads the price list into its database. Empty loads vendor.json alone.
	PricesFile string `koanf:"prices_file"`
	// VoicesBucketURL is where the recordings behind a customer's own voice are kept.
	VoicesBucketURL string `koanf:"voices_bucket_url"`
	// MockProviders builds the mock/* speech, voice and language models the load
	// benchmark runs against. They reach no vendor, recognise nothing and say nothing, so
	// a thousand conversations cost what one does. Off by default: with it off the
	// catalogue still declares them and nothing can be built for them, which is what a
	// deployment carrying real traffic wants.
	MockProviders bool `koanf:"mock_providers"`
	// PprofAddr serves net/http/pprof on its own listener, never the public one. Empty
	// means no profiling endpoint.
	PprofAddr string `koanf:"pprof_addr"`
	// MetricsAddr serves Prometheus metrics at /metrics on its own listener, never the
	// public one. Empty means no metrics endpoint.
	MetricsAddr string `koanf:"metrics_addr"`
	// Shard names the part of a deployment this router is in, when its calls are kept apart
	// from what every part shares (see Postgres.ShardDSN and Redis.Shard). Empty is a
	// deployment of one part.
	Shard      string     `koanf:"shard"`
	Postgres   Postgres   `koanf:"postgres"`
	Redis      Redis      `koanf:"redis"`
	Node       Node       `koanf:"node"`
	Shutdown   Shutdown   `koanf:"shutdown"`
	Auth       Auth       `koanf:"auth"`
	RateLimit  RateLimit  `koanf:"rate_limit"`
	DataMove   DataMove   `koanf:"data_move"`
	Stream     Stream     `koanf:"stream"`
	Chat       Chat       `koanf:"chat"`
	Agent      Agent      `koanf:"agent"`
	Connectors Connectors `koanf:"connectors"`
	Runs       Runs       `koanf:"runs"`
	Sessions   Sessions   `koanf:"sessions"`
	Episodes   Episodes   `koanf:"episodes"`
	Sandbox    Sandbox    `koanf:"sandbox"`
	// CodeSandbox is the sandbox code runs in, not the 10DLC Sandbox above.
	CodeSandbox CodeSandbox `koanf:"code_sandbox"`
	Models      Models      `koanf:"models"`
	Telnyx      Telnyx      `koanf:"telnyx"`
	Usage       Usage       `koanf:"usage"`
	Knowledge   Knowledge   `koanf:"knowledge"`
	Limits      Limits      `koanf:"limits"`
}

// Knowledge is the knowledge store's settings.
type Knowledge struct {
	// StreamCost is what the knowledge provider charges the operator, by meter: bytes_queried
	// and bytes_returned for a query, bytes_written for a write (its minimum is per request),
	// byte_hours for storage. Each call's cost under it is kept beside its row as
	// attributes.provider_cost_micros; what a customer pays is the price list's. It is here,
	// not in code, so a change in the provider's prices is a change of config.
	StreamCost map[string]pricing.Rate `koanf:"stream_cost"`
	// MinChunk is the smallest size, in bytes, a document is cut at: a smaller chunk_size is
	// raised to it.
	MinChunk int `koanf:"min_chunk"`
	// MaxHeading is how many bytes of a heading are repeated in each later passage of its
	// section and in every passage's source.
	MaxHeading int `koanf:"max_heading"`
	// Regions maps each region knowledge is kept in, such as eu, to the turbopuffer region
	// serving it, such as aws-eu-central-1. The default region, us, is always served: left
	// out here, at TURBOPUFFER_REGION, then gcp-us-central1, as before. A customer whose
	// policies name a region mapped to none has its knowledge refused.
	Regions map[string]string `koanf:"regions"`
}

// LimitsMode is how much of credit, the spend hard limit and the customers' spend limits the
// router enforces.
type LimitsMode string

const (
	// LimitsEnforce refuses what they refuse.
	LimitsEnforce LimitsMode = "enforce"
	// LimitsShadow works every refusal out, logs it as would refuse, and admits the request.
	LimitsShadow LimitsMode = "shadow"
	// LimitsOff refuses nothing for money. Work is still priced, recorded and shipped.
	LimitsOff LimitsMode = "off"
)

// Limits is what the router does about an organization's money before its work: a suspension,
// its credit, the operator's spend hard limit and its budgets.
type Limits struct {
	// Mode is enforce, shadow or off. Empty is enforce.
	Mode LimitsMode `koanf:"mode"`
}

// Usage is the usage service the router ships the rows it writes to. An empty URL is none:
// rows are written, and never shipped.
type Usage struct {
	// URL is the service's base URL, such as http://usage:8070.
	URL string `koanf:"url"`
	// Token is the service's ingest token.
	Token string `koanf:"token"`
	// ManageKey is the service's manage key (its USAGE_MANAGE_KEY), which `router usage
	// forget-user` calls the service's management API with. The router itself never uses it.
	ManageKey string `koanf:"manage_key"`
	// RedisURL is the Redis the service keeps every subject's spend in (its USAGE_REDIS_URL),
	// which the customer's spend limits are read from every second. Empty: they are read from
	// the service itself every few seconds.
	RedisURL string `koanf:"redis_url"`
}

// Models are the language models a customer serves themselves, named custom/<name>.
type Models struct {
	// PrivateEndpoints lets one be served from a private address or over plain http, such
	// as a vLLM server beside a self-hosted router. Off by default, and never on for a
	// router tenants share: there a tenant's endpoint would be a way into its network.
	PrivateEndpoints bool `koanf:"private_endpoints"`
}

// Postgres is where everything worth keeping is written. An empty DSN is a router that
// records nothing, which /health reports.
type Postgres struct {
	// DSN is the database every shard shares: tenancy, agents, connectors and phone numbers.
	// A deployment of one shard keeps everything there.
	DSN string `koanf:"dsn"`
	// ShardDSN is where this shard writes the calls, sessions and requests it runs. Empty
	// writes them to DSN.
	ShardDSN string `koanf:"shard_dsn"`
	// Shards is every shard's DSN by name, this one's included, which reads of calls,
	// sessions and requests cover. Empty reads this shard's alone.
	Shards map[string]string `koanf:"shards"`
}

// Redis holds live provider health and the daily counters. An empty address is a router
// that routes without them.
type Redis struct {
	// Addr is the Redis every shard shares: caches, counters and which node holds a session.
	Addr     string `koanf:"addr"`
	Username string `koanf:"username"`
	Password string `koanf:"password"`
	// Shard is this shard's own Redis, for provider health and a session's events. An empty
	// address uses the shared one.
	Shard ShardRedis `koanf:"shard"`
}

// ShardRedis is one shard's own Redis.
type ShardRedis struct {
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
	// Listen is a second address, where only this node's peers reach it. Empty means they
	// reach it on Addr. It is what two processes sharing one Addr need, since a peer
	// asking about a session on Addr could land on the other.
	Listen string `koanf:"listen"`
}

// Shutdown is how a router stops.
type Shutdown struct {
	// Drain is how long a stopping router keeps the sessions it holds, after it stops
	// taking new work, before it ends the rest. Zero ends them as soon as HTTP has
	// drained. A voice call lasts minutes, so a deployment that lets an old process
	// finish its calls sets this to the longest one it will wait for.
	Drain time.Duration `koanf:"drain"`
}

// Auth decides who the router believes a caller is.
type Auth struct {
	// Mode is api_key, proxy, noauth or custom. Empty means api_key.
	Mode string `koanf:"mode"`
	// KEK unseals the stored key secrets. It belongs in the environment rather than in a
	// file checked in beside the code: it is what makes a leaked backup ciphertext.
	KEK string `koanf:"kek"`
	// KMSKey names the KMS key that unwraps each ROUTER_AUTH_KEK_V<n>_WRAPPED at startup, so
	// the keyring itself is in no variable. Empty reads only the plain keys.
	KMSKey string `koanf:"kms_key"`
	// ProxyDeclaresKind says the proxy in front authenticates every caller and declares
	// whether it is a backend or an end user. In proxy mode the router then reads only the
	// app header and takes a caller that declares nothing for an end user. Off, a caller
	// that declares nothing is a backend, as a proxy that never declares means.
	ProxyDeclaresKind bool `koanf:"proxy_declares_kind"`
	// OpsKey is what Stream's own staff tools send as X-Ops-Key to review use cases. Empty
	// turns those endpoints off. It is never handed to a browser.
	OpsKey string `koanf:"ops_key"`
	// ManageKey is what Stream's proxy adds as X-Manage-Key to a staff member's call, and
	// what a self-hosted operator sends, to read and change price lists and set customers'
	// limits. Empty turns those endpoints off. It decides what customers pay, so it is never
	// the ops key.
	ManageKey string `koanf:"manage_key"`
}

// RateLimit caps what one of a customer's end users may spend in a day, for a customer
// whose policies set no number of their own. Either at 0 turns that half off.
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
	// own, acted in with the server token the gateway in front forwards for it.
	Tenancy string `koanf:"tenancy"`
	// Fallback is what app mode does for a customer with no token for its own app: deployment
	// writes it into the deployment's own app, as before, and refuse writes it nowhere.
	// Unset is refuse. Deployment mode never reads it.
	Fallback string `koanf:"fallback"`
	// AppID is the deployment's own app's id, which Stream is asked for when it is not
	// set. A pin naming it is finished with the deployment's own key, in either mode.
	AppID int64 `koanf:"app_id"`
	// MintURL is the gateway's internal mint endpoint, which signs the tokens a forwarded
	// server token cannot, and MintToken the bearer it expects.
	MintURL   string `koanf:"mint_url"`
	MintToken string `koanf:"mint_token"`
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

// Chat is the settings of the router's side of Stream Chat's own delivery of messages.
type Chat struct {
	// InboundKeys are the ed25519 public keys Stream Chat signs its own delivery of messages
	// with, each the standard base64 of 32 bytes: one, or two while Stream rotates. Only
	// Stream holds the private key, and no app's secret is one. None leaves the inbound
	// endpoint off.
	InboundKeys []string `koanf:"inbound_keys"`
}

// Agent is how an agent holds a conversation, where that is the deployment's choice.
type Agent struct {
	// SpeculativeReplies starts a reply while the flow controller is still deciding
	// whether the words were meant for the agent, and holds it until the ruling says to
	// answer. It saves the ruling's round trip on every answered turn and pays for the
	// replies a ruling throws away. Off by default.
	SpeculativeReplies bool `koanf:"speculative_replies"`
	// UnifiedReplies has the conversation model rule on a settled turn in the reply it
	// writes to it, instead of asking the flow controller first. Off by default.
	UnifiedReplies bool `koanf:"unified_replies"`
	// ChatTimings shows how long each stage of a turn took after the agent's reply in its
	// chat channel, for a developer talking to the agent. Off by default: it is not something a
	// caller should read.
	ChatTimings bool `koanf:"chat_timings"`
	// ReplyHedge is how long a reply may say nothing, neither text nor a tool call, before the same
	// request is asked of another candidate of its target as well. Whichever says something first is
	// kept and the other is cancelled. It only applies to a target with more than one candidate.
	// 1.2s by default; 0 turns it off.
	ReplyHedge time.Duration `koanf:"reply_hedge"`
	// FlowHedge is ReplyHedge for the flow controller's decisions. A decision is a few tokens
	// the caller is waiting in silence for, and one that has said nothing after this is
	// stalled rather than thinking, so it is asked of another candidate far sooner than a
	// reply is. 400ms by default; 0 turns it off.
	FlowHedge time.Duration `koanf:"flow_hedge"`
	// FlowJudge routes the flow controller to a decision model, such as
	// typesafe/jev-latest, in place of a language model. Empty keeps the language model.
	FlowJudge string `koanf:"flow_judge"`
}

// Connectors is whether agents may reach the customer's accounts elsewhere.
type Connectors struct {
	// Enabled builds the versioned key encryption keyring that seals connector
	// credentials, in every auth mode, and refuses to start without one. Off by default.
	Enabled bool `koanf:"enabled"`
	// ProxyCallsPerMinute caps the direct calls one customer sends to one connector through
	// the connection proxy in a minute, counted in Redis across every router (AI-958): all
	// customers share the router's egress, so one customer's calls could otherwise spend a
	// provider's limit for everyone. 0 turns it off; so does a deployment with no Redis.
	ProxyCallsPerMinute int64 `koanf:"proxy_calls_per_minute"`
}

// Runs is how many headless runs (POST /v1/agents/runs) one app may have going at once.
type Runs struct {
	// ConcurrentPerApp caps one app's runs in flight, counted in Redis across every router,
	// so one app's backlog cannot hold every model slot of the deployment. A run over it is a
	// 429 with Retry-After. 0 turns it off; so does a deployment with no Redis.
	ConcurrentPerApp int64 `koanf:"concurrent_per_app"`
}

// Sessions is how many sessions one app may have open at once: agent sessions however they
// were opened (the API, a call to one of its numbers, a channel message, a run) and the model
// streams of /v1/{modality}/stream.
type Sessions struct {
	// ConcurrentPerApp caps one app's sessions open, counted in Redis across every router, so
	// one app cannot take every agent and model slot the deployment has. One over it is a 429
	// with Retry-After. 0 turns it off; so does a deployment with no Redis.
	ConcurrentPerApp int64 `koanf:"concurrent_per_app"`
}

// Episodes is how a person's episodes close (T55, AI-884; omnichannel.Closer).
type Episodes struct {
	// IdleAfter is how long a text episode goes without a message before it closes and is
	// summarized. One setting for the whole router.
	IdleAfter time.Duration `koanf:"idle_after"`
}

// maxEpisodeIdle bounds IdleAfter from above. It is the only external bound: WhatsApp's
// customer service window, 24 hours from the person's last message, which AI-884 names as
// the one a text episode must close inside.
const maxEpisodeIdle = 24 * time.Hour

// Sandbox holds an app with no approved 10DLC use case to a few numbers and a little
// traffic. It is for the hosted router: a self-hosted one registers, or not, on its own
// account, and only opt-outs are enforced there.
type Sandbox struct {
	Enabled            bool  `koanf:"enabled"`
	Recipients         int   `koanf:"recipients"`
	MessagesPerDay     int64 `koanf:"messages_per_day"`
	AudioMinutesPerDay int64 `koanf:"audio_minutes_per_day"`
}

// CodeSandbox is how code the model wrote reaches a connection's API from its sandbox
// (internal/sandbox) without holding the connection's credential.
type CodeSandbox struct {
	// Broker holds the credential for code there: daytona, Daytona Secrets, or proxy, the
	// egress proxy at ProxyPublicURL.
	Broker string `koanf:"broker"`
	// SecretPrefix names this router's environment in every secret it leaves with the
	// provider, since routers sharing one Daytona organization must never take or sweep each
	// other's: stg on staging. A session with a sandbox binding is refused while it is empty.
	SecretPrefix string `koanf:"secret_prefix"`
	// ProxyAddr is where the router serves the sandbox egress proxy (internal/sandbox/egress),
	// such as :8091; empty serves none.
	ProxyAddr string `koanf:"proxy_addr"`
	// ProxyListenFD is a listening socket this process inherited, served in place of
	// ProxyAddr, as ListenFD is for Addr: a supervisor that keeps it open across processes
	// passes it, so no sandbox's CONNECT is refused while one router hands over to the next.
	// Zero means listen on ProxyAddr, which must still be set to serve the proxy at all.
	ProxyListenFD int `koanf:"proxy_listen_fd"`
	// ProxyCACert and ProxyCAKey are the PEM certificate and private key of the CA the proxy
	// signs each granted host's certificate with. Both are needed when ProxyAddr is set.
	ProxyCACert string `koanf:"proxy_ca_cert"`
	ProxyCAKey  string `koanf:"proxy_ca_key"`
	// ProxyPublicURL is the proxy as a sandbox reaches it, http or https://<IPv4 address>:port,
	// which the proxy broker needs, with ProxyCACert. With https the proxy serves TLS on
	// ProxyAddr, with a certificate for that address the CA signs.
	ProxyPublicURL string `koanf:"proxy_public_url"`
	// IdleStop is how long a sandbox runs no code before Daytona stops it, after which only its
	// disk is billed, and StoppedDelete how long a stopped one is kept before Daytona deletes
	// it. The next run starts a stopped sandbox again, or makes a new one. Both are whole
	// minutes, as Daytona takes them, and the sandbox meter keeps the same clock. The defaults
	// are Daytona's own 15 idle minutes and an hour stopped: a conversation that comes back
	// within the hour finds its files and installed packages where it left them.
	IdleStop      time.Duration `koanf:"idle_stop"`
	StoppedDelete time.Duration `koanf:"stopped_delete"`
	// Deployment names this deployment on every sandbox its routers make, as the
	// stream.deployment label, so deployments sharing a Daytona organization tell their
	// sandboxes apart. The sandbox job (router usage sandboxes) refuses to run without it and
	// touches only sandboxes carrying it. Empty labels nothing, and the job leaves the
	// sandboxes alone.
	Deployment string `koanf:"deployment"`
	// Targets maps each region sandboxes run in, such as eu, to the Daytona target they are
	// made on, such as eu. The default region, us, is always served: left out here, on
	// DAYTONA_TARGET, then Daytona's own default, as before. A session whose customer's
	// policies name a region mapped to none is refused its sandbox.
	Targets map[string]string `koanf:"targets"`
}

// sandboxDeployment is what code_sandbox.deployment may hold: a Daytona label value.
var sandboxDeployment = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,62}$`)

// secretPrefix is what code_sandbox.secret_prefix may hold. Secrets are named
// va_<prefix>_<session>_..., so a prefix with an underscore in it would make one router's
// sweep of va_stg_ match the secrets of another named stg_dev.
var secretPrefix = regexp.MustCompile(`^[a-z0-9]+$`)

// CodeSandboxBrokerDaytona is the code_sandbox.broker value for Daytona Secrets
// (daytona.BrokerName).
const CodeSandboxBrokerDaytona = "daytona"

// CodeSandboxBrokerProxy is the code_sandbox.broker value for the egress proxy
// (egress.BrokerName).
const CodeSandboxBrokerProxy = "proxy"

// Telnyx is the account Telnyx numbers are bought on and the TeXML application they answer
// on. The Telnyx provider still reads its own variables, which export writes back.
type Telnyx struct {
	APIKey string `koanf:"api_key"`
	// ConnectionID is the TeXML application. A call webhook for any other is hung up.
	ConnectionID string `koanf:"connection_id"`
	// PublicKey is Telnyx's base64 key, which checks the calls it sends to the webhook.
	PublicKey string `koanf:"public_key"`
}

// variables maps each setting to the environment variable that has always carried it.
// Both directions are read from here: the variable wins over the file on the way in, and
// the effective value is written back to it on the way out.
var variables = map[string]string{
	"addr":               "ROUTER_ADDR",
	"public_url":         "ROUTER_PUBLIC_URL",
	"log_level":          "ROUTER_LOG_LEVEL",
	"dashboard_url":      "DASHBOARD_BASE_URL",
	"trusted_proxies":    "ROUTER_TRUSTED_PROXIES",
	"routing_config":     "ROUTER_CONFIG",
	"phone_config":       "ROUTER_PHONE_CONFIG",
	"prices_file":        "ROUTER_PRICES_FILE",
	"voices_bucket_url":  "ROUTER_VOICES_BUCKET_URL",
	"mock_providers":     "ROUTER_MOCK_PROVIDERS",
	"pprof_addr":         "ROUTER_PPROF_ADDR",
	"metrics_addr":       "ROUTER_METRICS_ADDR",
	"shard":              "ROUTER_SHARD",
	"postgres.dsn":       "ROUTER_POSTGRES_DSN",
	"postgres.shard_dsn": "ROUTER_POSTGRES_SHARD_DSN",
	"postgres.shards":    "ROUTER_POSTGRES_SHARDS",
	"redis.addr":         "ROUTER_REDIS_ADDR",
	"redis.username":     "ROUTER_REDIS_USERNAME",
	"redis.password":     "ROUTER_REDIS_PASSWORD",

	"redis.shard.addr":     "ROUTER_REDIS_SHARD_ADDR",
	"redis.shard.username": "ROUTER_REDIS_SHARD_USERNAME",
	"redis.shard.password": "ROUTER_REDIS_SHARD_PASSWORD",
	"node.advertise":       "ROUTER_NODE_ADVERTISE",
	"node.listen":          "ROUTER_NODE_LISTEN",
	"listen_fd":            "ROUTER_LISTEN_FD",
	"shutdown.drain":       "ROUTER_SHUTDOWN_DRAIN",
	"auth.mode":            "ROUTER_AUTH_MODE",
	"auth.kek":             "ROUTER_AUTH_KEK",
	"auth.kms_key":         "ROUTER_AUTH_KMS_KEY",
	"auth.ops_key":         "ROUTER_AUTH_OPS_KEY",
	"auth.manage_key":      "ROUTER_AUTH_MANAGE_KEY",
	"data_move.retention":  "ROUTER_DATA_MOVE_RETENTION",
	"stream.api_key":       "STREAM_API_KEY",
	"stream.api_secret":    "STREAM_API_SECRET",
	"stream.base_url":      "STREAM_BASE_URL",
	"stream.user_token":    "STREAM_USER_TOKEN",
	"stream.tenancy":       "ROUTER_STREAM_TENANCY",
	"stream.fallback":      "ROUTER_STREAM_FALLBACK",
	"stream.app_id":        "ROUTER_STREAM_APP_ID",

	"stream.mint_url":   "ROUTER_STREAM_MINT_URL",
	"stream.mint_token": "ROUTER_STREAM_MINT_TOKEN",

	"chat.inbound_keys": "ROUTER_CHAT_INBOUND_KEYS",

	"rate_limit.messages_per_day": "ROUTER_RATE_LIMIT_MESSAGES_PER_DAY",
	"rate_limit.tokens_per_day":   "ROUTER_RATE_LIMIT_TOKENS_PER_DAY",

	"agent.speculative_replies": "ROUTER_SPECULATIVE_REPLIES",
	"agent.unified_replies":     "ROUTER_UNIFIED_REPLIES",
	"agent.chat_timings":        "ROUTER_CHAT_TIMINGS",
	"agent.reply_hedge":         "ROUTER_REPLY_HEDGE",
	"agent.flow_hedge":          "ROUTER_FLOW_HEDGE",
	"agent.flow_judge":          "ROUTER_FLOW_JUDGE",
	"auth.proxy_declares_kind":  "ROUTER_AUTH_PROXY_DECLARES_KIND",
	"connectors.enabled":        "ROUTER_CONNECTORS_ENABLED",
	"episodes.idle_after":       "ROUTER_EPISODES_IDLE_AFTER",
	"models.private_endpoints":  "ROUTER_MODELS_PRIVATE_ENDPOINTS",

	"connectors.proxy_calls_per_minute": "ROUTER_CONNECTORS_PROXY_CALLS_PER_MINUTE",
	"runs.concurrent_per_app":           "ROUTER_RUNS_CONCURRENT_PER_APP",
	"sessions.concurrent_per_app":       "ROUTER_SESSIONS_CONCURRENT_PER_APP",

	"sandbox.enabled":               "ROUTER_SANDBOX_ENABLED",
	"sandbox.recipients":            "ROUTER_SANDBOX_RECIPIENTS",
	"sandbox.messages_per_day":      "ROUTER_SANDBOX_MESSAGES_PER_DAY",
	"sandbox.audio_minutes_per_day": "ROUTER_SANDBOX_AUDIO_MINUTES_PER_DAY",

	"code_sandbox.broker":           "ROUTER_CODE_SANDBOX_BROKER",
	"code_sandbox.secret_prefix":    "ROUTER_CODE_SANDBOX_SECRET_PREFIX",
	"code_sandbox.proxy_addr":       "ROUTER_CODE_SANDBOX_PROXY_ADDR",
	"code_sandbox.proxy_listen_fd":  "ROUTER_CODE_SANDBOX_PROXY_LISTEN_FD",
	"code_sandbox.proxy_ca_cert":    "ROUTER_CODE_SANDBOX_PROXY_CA_CERT",
	"code_sandbox.proxy_ca_key":     "ROUTER_CODE_SANDBOX_PROXY_CA_KEY",
	"code_sandbox.proxy_public_url": "ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL",
	"code_sandbox.idle_stop":        "ROUTER_CODE_SANDBOX_IDLE_STOP",
	"code_sandbox.stopped_delete":   "ROUTER_CODE_SANDBOX_STOPPED_DELETE",
	"code_sandbox.deployment":       "ROUTER_CODE_SANDBOX_DEPLOYMENT",
	"code_sandbox.targets":          "ROUTER_CODE_SANDBOX_TARGETS",

	"telnyx.api_key":       "TELNYX_API_KEY",
	"telnyx.connection_id": "TELNYX_CONNECTION_ID",
	"telnyx.public_key":    "TELNYX_PUBLIC_KEY",

	"usage.url":        "ROUTER_USAGE_URL",
	"usage.token":      "ROUTER_USAGE_TOKEN",
	"usage.manage_key": "ROUTER_USAGE_MANAGE_KEY",
	"usage.redis_url":  "ROUTER_USAGE_REDIS_URL",

	"limits.mode": "ROUTER_LIMITS_MODE",

	"knowledge.min_chunk":   "ROUTER_KNOWLEDGE_MIN_CHUNK",
	"knowledge.max_heading": "ROUTER_KNOWLEDGE_MAX_HEADING",
	"knowledge.regions":     "ROUTER_KNOWLEDGE_REGIONS",
}

// lists are the settings written as a comma-separated variable and as a sequence in YAML.
var lists = map[string]bool{"trusted_proxies": true, "chat.inbound_keys": true}

// pairs are the settings written as a comma-separated variable of name=value and as a
// mapping in YAML.
var pairs = map[string]bool{"postgres.shards": true, "knowledge.regions": true, "code_sandbox.targets": true}

// providerRegion is what a turbopuffer region or a Daytona target may be called: part of a
// host name, or a value Daytona takes.
var providerRegion = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,62}$`)

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
		Agent:     Agent{ReplyHedge: 1200 * time.Millisecond, FlowHedge: 400 * time.Millisecond},
		// One hour is Kanat's decision of 2026-10-07 (D4, wave 3b), not a measurement:
		// unverified against any traffic. The only external bound is maxEpisodeIdle.
		Episodes:    Episodes{IdleAfter: time.Hour},
		Sandbox:     Sandbox{Recipients: 2, MessagesPerDay: 30, AudioMinutesPerDay: 30},
		CodeSandbox: CodeSandbox{Broker: CodeSandboxBrokerDaytona, IdleStop: 15 * time.Minute, StoppedDelete: time.Hour},
		// Slack's rate limits page (https://docs.slack.dev/apis/web-api/rate-limits, opened
		// 2026-10-09): «we do recommend you design your apps with a limit of 1 request per
		// second for any given API call». slack_bot and github are the built-ins with an
		// api_base, so the ones the proxy reaches, and that page names no limit per IP address.
		// One a second is the rate Slack tells an app to design for. Whether Slack keeps a limit
		// per egress address, and so whether 60 keeps one customer from spending it for the
		// others, is not on that page. # unverified
		// GitHub (https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api,
		// opened 2026-10-15) counts an authenticated call against the user, not the address,
		// and 60 a minute is under its «5,000 requests per hour» and «80 content-generating
		// requests per minute».
		Connectors: Connectors{ProxyCallsPerMinute: 60},
		// A choice, not a measurement: room for a moderation worker's batch without letting
		// one app's backlog take every model slot. # unverified
		Runs: Runs{ConcurrentPerApp: 20},
		// A choice, not a measurement: a hundred times the most calls at once any tier allows
		// (10), as text sessions outlive their last message by DetachedGrace and are many more.
		// # unverified
		Sessions: Sessions{ConcurrentPerApp: 1000},
		// turbopuffer's list prices (turbopuffer.com/pricing, read 2026-10-23): $1 per PB
		// queried, $0.05 per GB returned, $2 per GB written with every write billed at least
		// 10 KB, $0.33 per GB-month stored, a month taken as 730 hours. Its batch discount on
		// writes is a formula, not a rate, so it is the store's code.
		Knowledge: Knowledge{StreamCost: map[string]pricing.Rate{
			string(pricing.BytesQueried):  {USD: 1, Per: 1_000_000_000_000_000},
			string(pricing.BytesReturned): {USD: 0.05, Per: 1_000_000_000},
			string(pricing.BytesWritten):  {USD: 2, Per: 1_000_000_000, Minimum: 10_000},
			string(pricing.ByteHours):     {USD: 0.33, Per: 730_000_000_000},
		}, MinChunk: ingest.DefaultLimits.MinChunk, MaxHeading: ingest.DefaultLimits.MaxHeading},
		Limits: Limits{Mode: LimitsEnforce},
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
	if config.Limits.Mode == "" {
		config.Limits.Mode = LimitsEnforce
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
			if pairs[key] {
				return key, splitPairs(value)
			}
			return key, value
		},
	})
}

// validate refuses settings that would otherwise be found out from a customer.
func (c Config) validate() error {
	if c.Connectors.ProxyCallsPerMinute < 0 {
		return fmt.Errorf("config: connectors.proxy_calls_per_minute cannot be negative, got %d", c.Connectors.ProxyCallsPerMinute)
	}
	if c.Runs.ConcurrentPerApp < 0 {
		return fmt.Errorf("config: runs.concurrent_per_app cannot be negative, got %d", c.Runs.ConcurrentPerApp)
	}
	if c.Sessions.ConcurrentPerApp < 0 {
		return fmt.Errorf("config: sessions.concurrent_per_app cannot be negative, got %d", c.Sessions.ConcurrentPerApp)
	}
	if c.Knowledge.MinChunk < 0 || c.Knowledge.MaxHeading < 0 {
		return fmt.Errorf("config: knowledge.min_chunk and knowledge.max_heading cannot be negative, got %d and %d",
			c.Knowledge.MinChunk, c.Knowledge.MaxHeading)
	}
	// A heading repeated whole in a passage no bigger than it leaves no room for the passage.
	if c.Knowledge.MinChunk > 0 && c.Knowledge.MaxHeading >= c.Knowledge.MinChunk {
		return fmt.Errorf("config: knowledge.max_heading (%d) must be less than knowledge.min_chunk (%d)",
			c.Knowledge.MaxHeading, c.Knowledge.MinChunk)
	}
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
	if c.Episodes.IdleAfter <= 0 || c.Episodes.IdleAfter >= maxEpisodeIdle {
		return fmt.Errorf("config: episodes.idle_after is more than zero and less than %s, got %s",
			maxEpisodeIdle, c.Episodes.IdleAfter)
	}
	if c.Agent.ReplyHedge < 0 {
		return fmt.Errorf("config: agent.reply_hedge cannot be negative, got %s", c.Agent.ReplyHedge)
	}
	if c.CodeSandbox.Broker != CodeSandboxBrokerDaytona && c.CodeSandbox.Broker != CodeSandboxBrokerProxy {
		return fmt.Errorf("config: code_sandbox.broker is %s or %s, got %q", CodeSandboxBrokerDaytona, CodeSandboxBrokerProxy,
			c.CodeSandbox.Broker)
	}
	_, validProxy := ProxyURL(c.CodeSandbox.ProxyPublicURL)
	if c.CodeSandbox.Broker == CodeSandboxBrokerProxy && (!validProxy || c.CodeSandbox.ProxyCACert == "") {
		return errors.New("config: code_sandbox.broker is proxy, so code_sandbox.proxy_public_url " +
			"(http://<public IPv4 address>:port or https://<public IPv4 address>:port: a sandbox reaches that address " +
			"alone and has no DNS to look a host name up with) and proxy_ca_cert " +
			"(ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL, ROUTER_CODE_SANDBOX_PROXY_CA_CERT) must say how a sandbox reaches the proxy")
	}
	// An https URL a sandbox cannot use is served as plain HTTP by the listener, which would
	// leave a router running a different proxy from the one its settings say.
	if raw := c.CodeSandbox.ProxyPublicURL; c.CodeSandbox.ProxyAddr != "" && len(raw) >= len("https://") &&
		strings.EqualFold(raw[:len("https://")], "https://") && !validProxy {
		// The URL may carry a credential as its user, so the error names the host alone.
		host := "unparseable"
		if parsed, err := url.Parse(raw); err == nil {
			host = parsed.Host
		}
		return fmt.Errorf("config: code_sandbox.proxy_public_url (ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL) has host %q, an https URL that is not "+
			"https://<public IPv4 address>:port, so the proxy would be served in plain HTTP: fix it or use http", host)
	}
	// Every router of a proxy broker serves the proxy, so no sandbox is sent to a proxy no
	// router runs.
	if c.CodeSandbox.Broker == CodeSandboxBrokerProxy && c.CodeSandbox.ProxyAddr == "" {
		return errors.New("config: code_sandbox.broker is proxy, so this router serves the proxy: set " +
			"code_sandbox.proxy_addr (ROUTER_CODE_SANDBOX_PROXY_ADDR)")
	}
	if err := c.validateProxyListenFD(); err != nil {
		return err
	}
	if strings.Contains(c.CodeSandbox.ProxyCACert, "PRIVATE KEY") {
		return errors.New("config: code_sandbox.proxy_ca_cert (ROUTER_CODE_SANDBOX_PROXY_CA_CERT) holds a private key, " +
			"which every sandbox would be given: keep the certificate there and the key in proxy_ca_key")
	}
	if c.CodeSandbox.SecretPrefix != "" && !secretPrefix.MatchString(c.CodeSandbox.SecretPrefix) {
		return fmt.Errorf("config: code_sandbox.secret_prefix is lowercase letters and digits only, got %q: "+
			"a prefix holding _ would match another router's secrets (va_stg_ and va_stg_dev_)", c.CodeSandbox.SecretPrefix)
	}
	if c.CodeSandbox.ProxyAddr != "" && (c.CodeSandbox.ProxyCACert == "" || c.CodeSandbox.ProxyCAKey == "") {
		return fmt.Errorf("config: code_sandbox.proxy_addr is set, so code_sandbox.proxy_ca_cert and proxy_ca_key " +
			"(ROUTER_CODE_SANDBOX_PROXY_CA_CERT, ROUTER_CODE_SANDBOX_PROXY_CA_KEY) must hold the CA the proxy signs with")
	}
	// Daytona takes whole minutes, and the meter bills by the same clock: a fraction would
	// have one stop the sandbox before the other.
	for setting, interval := range map[string]time.Duration{
		"code_sandbox.idle_stop (ROUTER_CODE_SANDBOX_IDLE_STOP)":           c.CodeSandbox.IdleStop,
		"code_sandbox.stopped_delete (ROUTER_CODE_SANDBOX_STOPPED_DELETE)": c.CodeSandbox.StoppedDelete,
	} {
		if interval < time.Minute || interval%time.Minute != 0 {
			return fmt.Errorf("config: %s is a whole number of minutes, at least one, got %s", setting, interval)
		}
	}
	if c.CodeSandbox.Deployment != "" && !sandboxDeployment.MatchString(c.CodeSandbox.Deployment) {
		return fmt.Errorf("config: code_sandbox.deployment (ROUTER_CODE_SANDBOX_DEPLOYMENT) is up to 63 lowercase letters, "+
			"digits, '.', '_' and '-', starting with a letter or digit, got %q", c.CodeSandbox.Deployment)
	}
	if c.Agent.FlowHedge < 0 {
		return fmt.Errorf("config: agent.flow_hedge cannot be negative, got %s", c.Agent.FlowHedge)
	}
	if c.Auth.ManageKey != "" && c.Auth.ManageKey == c.Auth.OpsKey {
		return errors.New("config: auth.manage_key (ROUTER_AUTH_MANAGE_KEY) is the same as auth.ops_key " +
			"(ROUTER_AUTH_OPS_KEY): give the management API a key of its own")
	}
	if err := c.validateUsage(); err != nil {
		return err
	}
	for meter, rate := range c.Knowledge.StreamCost {
		if rate.Per <= 0 || rate.USD < 0 || rate.Minimum < 0 {
			return fmt.Errorf("config: knowledge.stream_cost.%s is usd per a positive number of units, got %+v", meter, rate)
		}
	}
	for _, regions := range []struct {
		setting, example string
		mapped           map[string]string
	}{
		{"knowledge.regions (ROUTER_KNOWLEDGE_REGIONS)", "the turbopuffer region serving it, such as eu=aws-eu-central-1", c.Knowledge.Regions},
		{"code_sandbox.targets (ROUTER_CODE_SANDBOX_TARGETS)", "the Daytona target serving it, such as eu=eu", c.CodeSandbox.Targets},
	} {
		for _, region := range slices.Sorted(maps.Keys(regions.mapped)) {
			if !pricing.ValidRegion(region) || !providerRegion.MatchString(regions.mapped[region]) {
				return fmt.Errorf("config: %s maps a region to %s, got %s=%q", regions.setting, regions.example, region, regions.mapped[region])
			}
		}
	}
	switch c.Limits.Mode {
	case LimitsEnforce, LimitsShadow, LimitsOff:
	default:
		return fmt.Errorf("config: limits.mode (ROUTER_LIMITS_MODE) is %s, %s or %s, got %q",
			LimitsEnforce, LimitsShadow, LimitsOff, c.Limits.Mode)
	}
	return c.validateShard()
}

// validateUsage refuses a usage service the router could not ship to. A token without a URL is
// allowed, since unsetting the URL alone is how shipping is stopped; the router warns of it.
func (c Config) validateUsage() error {
	if c.Usage.URL == "" {
		if c.Usage.RedisURL != "" {
			return errors.New("config: usage.redis_url (ROUTER_USAGE_REDIS_URL) is the usage service's Redis, so it needs usage.url (ROUTER_USAGE_URL)")
		}
		return nil
	}
	if parsed, err := url.Parse(c.Usage.URL); err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return errors.New("config: usage.url (ROUTER_USAGE_URL) is the usage service's base URL, such as http://usage:8070")
	}
	if c.Usage.Token == "" {
		return errors.New("config: usage.url (ROUTER_USAGE_URL) is set, so usage.token (ROUTER_USAGE_TOKEN) must hold " +
			"the usage service's ingest token")
	}
	return nil
}

// ProxyURL parses raw as the proxy's public URL, http or https://<IPv4 address>:port, and says
// whether it is one a sandbox can use. https is the proxy serving TLS. The URL carries the sandbox's own credential as its user, so it has
// none. The sandbox may reach that address alone, through Daytona's network allow list, which
// takes IP ranges, so it has no DNS to look a name up with, and no IPv6 route (live probe,
// 2026-10-22). The address must be public: a sandbox reaches no loopback, private (RFC 1918),
// link-local (169.254/16, the metadata address too), unspecified or broadcast address, and the
// range would let it reach that address of the provider's network. The port is 1 to 65535, and
// the URL has no path, query or fragment.
func ProxyURL(raw string) (*url.URL, bool) {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme != "http" && parsed.Scheme != "https" || parsed.User != nil || parsed.Opaque != "" || parsed.Path != "" ||
		parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
		return nil, false
	}
	address, err := netip.ParseAddr(parsed.Hostname())
	if err != nil || !address.Is4() || !address.IsGlobalUnicast() || address.IsPrivate() {
		return nil, false
	}
	port, err := strconv.Atoi(parsed.Port())
	return parsed, err == nil && port >= 1 && port <= 65535
}

// validateProxyListenFD refuses an inherited proxy socket the router would not serve: one
// without the proxy on, a descriptor that cannot be one, or the public listener's own.
func (c Config) validateProxyListenFD() error {
	fd := c.CodeSandbox.ProxyListenFD
	switch {
	case fd == 0:
		return nil
	case fd < 0:
		return fmt.Errorf("config: code_sandbox.proxy_listen_fd (ROUTER_CODE_SANDBOX_PROXY_LISTEN_FD) is a file descriptor, got %d", fd)
	case c.CodeSandbox.ProxyAddr == "":
		return errors.New("config: code_sandbox.proxy_listen_fd (ROUTER_CODE_SANDBOX_PROXY_LISTEN_FD) is the proxy's socket, " +
			"and the proxy is served only with code_sandbox.proxy_addr (ROUTER_CODE_SANDBOX_PROXY_ADDR) set")
	case fd == c.ListenFD:
		return fmt.Errorf("config: code_sandbox.proxy_listen_fd and listen_fd are both %d: the proxy and the API need a socket each", fd)
	}
	return nil
}

// validateShard refuses shard settings that would write a shard's calls where no reader
// looks for them.
func (c Config) validateShard() error {
	if c.Postgres.ShardDSN != "" && c.Postgres.DSN == "" {
		return errors.New("config: postgres.shard_dsn is a shard's half of postgres.dsn: set postgres.dsn")
	}
	if c.Redis.Shard.Addr != "" && c.Redis.Addr == "" {
		return errors.New("config: redis.shard.addr is a shard's half of redis.addr: set redis.addr")
	}
	if len(c.Postgres.Shards) == 0 {
		return nil
	}
	if c.Shard == "" {
		return errors.New("config: postgres.shards names every shard: set shard to this one's name")
	}
	own, listed := c.Postgres.Shards[c.Shard]
	if !listed {
		return fmt.Errorf("config: postgres.shards does not list this shard, %q", c.Shard)
	}
	if own != c.Postgres.ShardDSN {
		return fmt.Errorf("config: postgres.shards lists shard %q at another DSN than postgres.shard_dsn", c.Shard)
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
		return fmt.Errorf("config: stream.tenancy=%s keeps every app's server token in Postgres: set postgres.dsn", TenancyApp)
	}
	if strings.TrimSpace(c.Auth.Mode) != "proxy" || !c.Auth.ProxyDeclaresKind {
		return fmt.Errorf("config: stream.tenancy=%s acts with the server tokens the gateway in front "+
			"forwards: set auth.mode=proxy and auth.proxy_declares_kind=true", TenancyApp)
	}
	if c.Stream.MintURL == "" || c.Stream.MintToken == "" {
		return fmt.Errorf("config: stream.tenancy=%s signs agents' and guests' tokens at the gateway: "+
			"set stream.mint_url and stream.mint_token", TenancyApp)
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
		"trusted_proxies":               strings.Join(c.TrustedProxies, ","),
		"routing_config":                c.RoutingConfig,
		"phone_config":                  c.PhoneConfig,
		"prices_file":                   c.PricesFile,
		"voices_bucket_url":             c.VoicesBucketURL,
		"mock_providers":                fmt.Sprint(c.MockProviders),
		"pprof_addr":                    c.PprofAddr,
		"metrics_addr":                  c.MetricsAddr,
		"shard":                         c.Shard,
		"postgres.dsn":                  c.Postgres.DSN,
		"postgres.shard_dsn":            c.Postgres.ShardDSN,
		"postgres.shards":               joinPairs(c.Postgres.Shards),
		"redis.addr":                    c.Redis.Addr,
		"redis.username":                c.Redis.Username,
		"redis.password":                c.Redis.Password,
		"redis.shard.addr":              c.Redis.Shard.Addr,
		"redis.shard.username":          c.Redis.Shard.Username,
		"redis.shard.password":          c.Redis.Shard.Password,
		"node.advertise":                c.Node.Advertise,
		"node.listen":                   c.Node.Listen,
		"listen_fd":                     listenFD(c.ListenFD),
		"shutdown.drain":                c.Shutdown.Drain.String(),
		"auth.mode":                     c.Auth.Mode,
		"auth.kek":                      c.Auth.KEK,
		"auth.kms_key":                  c.Auth.KMSKey,
		"auth.ops_key":                  c.Auth.OpsKey,
		"auth.manage_key":               c.Auth.ManageKey,
		"auth.proxy_declares_kind":      fmt.Sprint(c.Auth.ProxyDeclaresKind),
		"stream.api_key":                c.Stream.APIKey,
		"stream.api_secret":             c.Stream.APISecret,
		"stream.base_url":               c.Stream.BaseURL,
		"stream.user_token":             c.Stream.UserToken,
		"stream.tenancy":                c.Stream.Tenancy,
		"stream.fallback":               c.Stream.Fallback,
		"stream.app_id":                 appID(c.Stream.AppID),
		"stream.mint_url":               c.Stream.MintURL,
		"stream.mint_token":             c.Stream.MintToken,
		"chat.inbound_keys":             strings.Join(c.Chat.InboundKeys, ","),
		"data_move.retention":           c.DataMove.Retention.String(),
		"rate_limit.messages_per_day":   fmt.Sprint(c.RateLimit.MessagesPerDay),
		"rate_limit.tokens_per_day":     fmt.Sprint(c.RateLimit.TokensPerDay),
		"agent.speculative_replies":     fmt.Sprint(c.Agent.SpeculativeReplies),
		"agent.unified_replies":         fmt.Sprint(c.Agent.UnifiedReplies),
		"agent.chat_timings":            fmt.Sprint(c.Agent.ChatTimings),
		"agent.reply_hedge":             c.Agent.ReplyHedge.String(),
		"agent.flow_hedge":              c.Agent.FlowHedge.String(),
		"connectors.enabled":            fmt.Sprint(c.Connectors.Enabled),
		"episodes.idle_after":           c.Episodes.IdleAfter.String(),
		"sandbox.enabled":               fmt.Sprint(c.Sandbox.Enabled),
		"sandbox.recipients":            fmt.Sprint(c.Sandbox.Recipients),
		"sandbox.messages_per_day":      fmt.Sprint(c.Sandbox.MessagesPerDay),
		"sandbox.audio_minutes_per_day": fmt.Sprint(c.Sandbox.AudioMinutesPerDay),

		"code_sandbox.broker":           c.CodeSandbox.Broker,
		"code_sandbox.secret_prefix":    c.CodeSandbox.SecretPrefix,
		"code_sandbox.proxy_addr":       c.CodeSandbox.ProxyAddr,
		"code_sandbox.proxy_listen_fd":  listenFD(c.CodeSandbox.ProxyListenFD),
		"code_sandbox.proxy_ca_cert":    c.CodeSandbox.ProxyCACert,
		"code_sandbox.proxy_ca_key":     c.CodeSandbox.ProxyCAKey,
		"code_sandbox.proxy_public_url": c.CodeSandbox.ProxyPublicURL,
		"code_sandbox.idle_stop":        c.CodeSandbox.IdleStop.String(),
		"code_sandbox.stopped_delete":   c.CodeSandbox.StoppedDelete.String(),
		"code_sandbox.deployment":       c.CodeSandbox.Deployment,
		"code_sandbox.targets":          joinPairs(c.CodeSandbox.Targets),

		"connectors.proxy_calls_per_minute": fmt.Sprint(c.Connectors.ProxyCallsPerMinute),
		"runs.concurrent_per_app":           fmt.Sprint(c.Runs.ConcurrentPerApp),
		"sessions.concurrent_per_app":       fmt.Sprint(c.Sessions.ConcurrentPerApp),

		"telnyx.api_key":       c.Telnyx.APIKey,
		"telnyx.connection_id": c.Telnyx.ConnectionID,
		"telnyx.public_key":    c.Telnyx.PublicKey,

		"usage.url":        c.Usage.URL,
		"usage.token":      c.Usage.Token,
		"usage.manage_key": c.Usage.ManageKey,
		"usage.redis_url":  c.Usage.RedisURL,

		"limits.mode": string(c.Limits.Mode),

		"knowledge.min_chunk":   fmt.Sprint(c.Knowledge.MinChunk),
		"knowledge.max_heading": fmt.Sprint(c.Knowledge.MaxHeading),
		"knowledge.regions":     joinPairs(c.Knowledge.Regions),
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

// splitPairs reads a comma-separated variable of name=value entries. An entry with no name
// is dropped, the same as an empty one.
func splitPairs(raw string) map[string]string {
	entries := map[string]string{}
	for _, entry := range splitList(raw) {
		name, value, _ := strings.Cut(entry, "=")
		if name = strings.TrimSpace(name); name != "" {
			entries[name] = strings.TrimSpace(value)
		}
	}
	return entries
}

// joinPairs writes what splitPairs reads, in name order so the variable is the same every
// time.
func joinPairs(entries map[string]string) string {
	written := make([]string, 0, len(entries))
	for _, name := range slices.Sorted(maps.Keys(entries)) {
		written = append(written, name+"="+entries[name])
	}
	return strings.Join(written, ",")
}

// listenFD writes an inherited socket back, or nothing for none: descriptor zero is standard
// input, never a listener.
func listenFD(fd int) string {
	if fd == 0 {
		return ""
	}
	return fmt.Sprint(fd)
}

// appID writes an app id back, or nothing for one nobody set.
func appID(id int64) string {
	if id == 0 {
		return ""
	}
	return fmt.Sprint(id)
}
