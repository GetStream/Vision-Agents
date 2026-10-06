// Command router serves the model router's HTTP API.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/getsentry/sentry-go"
	"github.com/hibiken/asynq"
	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/blob"
	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/apikey"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/none"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2cc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/hmacheader"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	dlctelnyx "github.com/GetStream/Vision-Agents/acceleration/internal/dlc/telnyx"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/eotdefaults"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory/mem0"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pluginevents"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/relay"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search/exa"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/simulation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tracing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/cartesia"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/elevenlabs"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/fish"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/inworld"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/users"
)

// release is the version this binary was built from, set with -X main.release
// at link time by the workflow that publishes it. Empty in a local build, which
// Sentry reads as "no release" rather than as an error.
var release string

const (
	// authKEKEnvVar and authKEKVersionEnvVar name the connector credential keyring: the
	// keys are authKEKEnvVar with a _V<n> suffix, and authKEKVersionEnvVar says which one
	// seals new rows.
	authKEKEnvVar        = "ROUTER_AUTH_KEK"
	authKEKVersionEnvVar = "ROUTER_AUTH_KEK_VERSION"
	shutdownGrace        = 10 * time.Second
	readHeaderTimeout    = 10 * time.Second
	// crawlTimeout bounds reading one page into a knowledge base. It is generous compared
	// to a search because nobody is on the phone waiting for it: a page that has to be
	// crawled live rather than served from an index takes seconds, and giving up on it
	// leaves a subscription that never works.
	crawlTimeout = 60 * time.Second
	// sentryFlushTimeout bounds how long the process spends delivering buffered
	// events on the way out. Short, because this runs while the orchestrator is
	// already counting down the termination grace period.
	sentryFlushTimeout = 2 * time.Second
	// traceFlushTimeout bounds the same thing for spans, on the same reasoning.
	traceFlushTimeout = 2 * time.Second
)

// usage is what the binary does besides serving.
const usage = `usage: router [--config path] [command]

  serve                 serve the API (the default)
  keys create           mint a credential for an app, printing the secret once
  replicate             copy another deployment's data here and follow its changes
  stream-apps           look after the Stream apps customers registered in app mode
`

func main() {
	path, command, args := arguments(os.Args[1:])
	settings, from, configErr := config.Load(path)
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel(settings)}))
	slog.SetDefault(logger)
	if configErr != nil {
		logger.Error("router stopped", "error", configErr)
		os.Exit(1)
	}
	logger.Info("loaded configuration", "from", from)

	// Dsn is deliberately not set. The SDK reads SENTRY_DSN and
	// SENTRY_ENVIRONMENT itself, so the DSN stays out of this repository and an
	// unset one disables reporting -- which is what a local run and the tests
	// want, and what self-hosting this service wants too.
	//
	// A failed Init is logged, not fatal: a malformed DSN is a configuration
	// mistake, but it is a bad reason to refuse to serve calls.
	if err := sentry.Init(sentry.ClientOptions{
		Release:          release,
		ServerName:       "acceleration-router",
		AttachStacktrace: true,
	}); err != nil {
		logger.Error("sentry is disabled", "error", err)
	}

	if err := dispatchCommand(command, args, settings, logger); err != nil {
		logger.Error("router stopped", "error", err)
		// Reported here because a startup failure never reaches an HTTP handler,
		// so the middleware in internal/api would never see it.
		sentry.CaptureException(err)
		sentry.Flush(sentryFlushTimeout)
		os.Exit(1)
	}
	// Not deferred: os.Exit above skips defers, so a single deferred flush would
	// cover only the path that does not need it.
	sentry.Flush(sentryFlushTimeout)
}

// arguments pulls --config off the command line, wherever it sits, and returns the
// command and what is left for it.
//
// It is read by hand rather than by a flag set because the subcommands have flags of
// their own, and --config belongs to all of them: it says which deployment is being
// talked about before anything decides what to do with it.
func arguments(argv []string) (path, command string, rest []string) {
	for index := 0; index < len(argv); index++ {
		argument := argv[index]
		switch {
		case argument == "--config" || argument == "-config":
			if index+1 < len(argv) {
				path = argv[index+1]
				index++
			}
		case strings.HasPrefix(argument, "--config="):
			path = strings.TrimPrefix(argument, "--config=")
		case command == "" && !strings.HasPrefix(argument, "-"):
			command = argument
		default:
			rest = append(rest, argument)
		}
	}
	if command == "" {
		command = "serve"
	}
	return path, command, rest
}

// dispatchCommand runs what the command line asked for.
func dispatchCommand(command string, args []string, settings config.Config, logger *slog.Logger) error {
	switch command {
	case "serve":
		return run(settings, logger)
	case "keys":
		return runKeys(args, settings, logger)
	case "replicate":
		return runReplicate(args, settings, logger)
	case "stream-apps":
		return runStreamApps(args, settings, logger)
	case "help", "-h", "--help":
		fmt.Print(usage)
		return nil
	default:
		return fmt.Errorf("unknown command %q\n\n%s", command, usage)
	}
}

// logLevel reads the level to log at. Debug is where the turn-taking decisions are: what
// was heard, what the flow controller made of it, and why the agent did or did not speak.
//
// It is read from the settings when they loaded and from the environment when they did
// not, so that the failure to load them is itself logged at the level asked for.
func logLevel(settings config.Config) slog.Level {
	var level slog.Level
	text := settings.LogLevel
	if text == "" {
		text = os.Getenv("ROUTER_LOG_LEVEL")
	}
	if text == "" {
		return slog.LevelInfo
	}
	if err := level.UnmarshalText([]byte(text)); err != nil {
		return slog.LevelInfo
	}
	return level
}

// newSecretSealer builds the keyring the secrets the router holds for its customers are
// sealed under: connector credentials, Stream app keys and channel credentials. Connectors
// and app mode refuse to start without one; channels use it when it is set and are off
// otherwise, so it is nil only when nothing requires it and no key is set. It does not
// depend on auth.mode: a proxy deployment holds them as much as an api_key one does.
//
// The keyring is every ROUTER_AUTH_KEK_V1, _V2 and so on that is set, with
// ROUTER_AUTH_KEK_VERSION naming the one that seals new rows. That variable picks the
// writer and is never a ceiling: moving it back to an older key must leave the newer ones
// loaded, or the rows sealed under them stop opening. auth.kek is version 1, so a
// deployment that already has it needs nothing more.
func newSecretSealer(settings config.Config) (*auth.Sealer, error) {
	var holders []string
	if settings.Connectors.Enabled {
		holders = append(holders, "connectors.enabled")
	}
	if settings.Stream.Tenancy == config.TenancyApp {
		holders = append(holders, "stream.tenancy="+config.TenancyApp)
	}
	if len(holders) == 0 {
		if !keyringSet(settings) {
			return nil, nil
		}
		return loadKeyring(settings, "channels need")
	}
	// The setting that needs the keyring is what each refusal names, so whoever reads it
	// knows which change brought it on.
	needs := strings.Join(holders, " and ") + " need"
	if len(holders) == 1 {
		needs = holders[0] + " needs"
	}
	return loadKeyring(settings, needs)
}

// keyringSet reports whether any key of the keyring is set: auth.kek, or a
// ROUTER_AUTH_KEK_V<n>.
func keyringSet(settings config.Config) bool {
	if settings.Auth.KEK != "" {
		return true
	}
	for _, variable := range os.Environ() {
		name, key, _ := strings.Cut(variable, "=")
		if strings.HasPrefix(name, authKEKEnvVar+"_V") && key != "" {
			return true
		}
	}
	return false
}

// loadKeyring reads every ROUTER_AUTH_KEK_V<n> and auth.kek into a keyring, with
// ROUTER_AUTH_KEK_VERSION naming the writer. needs says which setting requires it, for
// each refusal to name.
func loadKeyring(settings config.Config, needs string) (*auth.Sealer, error) {
	current := auth.KEKVersion
	if configured := os.Getenv(authKEKVersionEnvVar); configured != "" {
		version, err := strconv.Atoi(configured)
		if err != nil || version < 1 {
			// The value is left out on purpose: a key pasted into the wrong variable would
			// otherwise reach the logs and Sentry with this error.
			return nil, fmt.Errorf("%s %s to be a positive integer", needs, authKEKVersionEnvVar)
		}
		current = version
	}
	keys := make(map[int]string)
	if settings.Auth.KEK != "" {
		keys[1] = settings.Auth.KEK
	}
	for _, variable := range os.Environ() {
		name, key, _ := strings.Cut(variable, "=")
		suffix, ok := strings.CutPrefix(name, authKEKEnvVar+"_V")
		version, err := strconv.Atoi(suffix)
		// Only the plain spelling counts, so _V01 cannot stand in for _V1 and
		// ROUTER_AUTH_KEK_VERSION is not read as a key.
		if !ok || err != nil || version < 1 || strconv.Itoa(version) != suffix || key == "" {
			continue
		}
		if version == 1 && keys[1] != "" && keys[1] != key {
			return nil, fmt.Errorf("%s and %s_V1 are both version 1 and differ: set one of them",
				authKEKEnvVar, authKEKEnvVar)
		}
		keys[version] = key
	}
	if keys[current] == "" {
		if os.Getenv(authKEKVersionEnvVar) == "" {
			return nil, fmt.Errorf("%s a key encryption keyring to seal the secrets it holds: "+
				"set %s_V1 (%s is version 1)", needs, authKEKEnvVar, authKEKEnvVar)
		}
		// The version is not named: an all-digit key pasted into ROUTER_AUTH_KEK_VERSION
		// parses as one, and naming it would send the key to the logs and Sentry. The
		// versions that are set come from variable names, never from values.
		return nil, fmt.Errorf("%s a key for the version %s names: "+
			"set the matching %s_V<n> (versions set: %v)",
			needs, authKEKVersionEnvVar, authKEKEnvVar, slices.Sorted(maps.Keys(keys)))
	}
	return auth.NewSealerWithKeyring(current, keys)
}

// connectorHTTPTimeout bounds one outbound request oauth2_code or oauth2_client_credentials
// makes: a discovery fetch, a client registration, a code exchange or a token request. 10 s is the prototype's (defaultHTTPClient in
// internal/mcp/oauth.go on codex/connector-support at cf62af0d), not a measured one.
const connectorHTTPTimeout = 10 * time.Second

// newConnectorRegistry is the connector adapters this deployment has: none when connectors
// are off, so no connection and no custom connector can name a scheme, and oauth2_code,
// oauth2_client_credentials, api_key, bearer and none when they are on. A new scheme is one more entry in the list
// below. It follows the sealer: without a keyring no attempt or grant can be sealed,
// so a scheme registered without one would take connections nothing could ever authorize.
//
// ClientMetadataURL is where the API serves the router's client metadata document
// (api.ConnectorClientMetadataPath under public_url), and empty, which turns CIMD off, when
// public_url is not https: CIMD section 3 allows only an https client_id URL. clients finds a
// client registered in advance: the app's own from its record, the operator's from the
// environment (api.ConnectorClients).
func newConnectorRegistry(settings config.Config, clients oauth2code.ClientLookup) (core.Registry, error) {
	if !settings.Connectors.Enabled {
		return core.Registry{}, nil
	}
	code, err := oauth2code.New(connectorSchemeConfig(settings, clients))
	if err != nil {
		return core.Registry{}, err
	}
	clientCredentials, err := oauth2cc.New(oauth2cc.Config{HTTP: egress.NewClient(connectorHTTPTimeout, nil)})
	if err != nil {
		return core.Registry{}, err
	}
	schemes := map[string]core.Scheme{}
	for _, scheme := range []core.Scheme{code, clientCredentials, apikey.New(), bearer.New(), none.New()} {
		schemes[scheme.Name()] = scheme
	}
	// The verifiers a manifest's channel.verifier.kind may name; a new one is one more entry.
	verifiers := map[string]core.Verifier{}
	for _, verifier := range []core.Verifier{hmacheader.New()} {
		verifiers[verifier.Name()] = verifier
	}
	return core.Registry{Schemes: schemes, Verifiers: verifiers}, nil
}

// connectorSchemeConfig is the oauth2code.Config newConnectorRegistry starts the scheme with.
// A test runs it against a loopback fake by replacing HTTP and PublicEndpoint, which egress
// refuses loopback for, and keeps the rest.
func connectorSchemeConfig(settings config.Config, clients oauth2code.ClientLookup) oauth2code.Config {
	return oauth2code.Config{
		HTTP:              egress.NewClient(connectorHTTPTimeout, nil),
		Clients:           clients,
		ClientMetadataURL: api.ConnectorClientMetadataURL(settings.PublicURL),
	}
}

// newAuthenticator builds the authenticator the deployment's mode asks for.
//
// api_key needs both a store to look keys up in and the key that unseals their secrets, and
// says which is missing rather than starting and refusing every request for a reason only
// visible in a 401.
func newAuthenticator(settings config.Config, configs *appconfig.Store, logger *slog.Logger) (auth.Authenticator, error) {
	mode, err := auth.ParseMode(settings.Auth.Mode)
	if err != nil {
		return nil, err
	}

	switch mode {
	case auth.NoAuth:
		logger.Warn("running without authentication: anyone who can reach this router can "+
			"read and spend any customer's account, and every one of them is treated as "+
			"that customer's own backend, so nothing but a laptop should run this way",
			"mode", auth.NoAuth, "set", "auth.mode")
		return auth.New(mode, nil)
	case auth.Proxy:
		logger.Warn("authenticating nothing: the caller is whoever the headers in front of "+
			"this router say, so only a proxy that overwrites them should be able to reach it",
			"mode", auth.Proxy, "set", "auth.mode", "proxy_declares_kind", settings.Auth.ProxyDeclaresKind)
		return auth.NewProxy(auth.ProxyOptions{DeclaresKind: settings.Auth.ProxyDeclaresKind}), nil
	case auth.Custom:
		return nil, fmt.Errorf("auth.mode=%s has no authenticator in this binary: a deployment "+
			"answering for itself embeds the module and passes api.WithAuthenticator", auth.Custom)
	}

	if configs == nil {
		return nil, fmt.Errorf("auth.mode=%s needs postgres.dsn, because that is where the keys are",
			auth.APIKey)
	}
	sealer, err := loadKeyring(settings, "auth.mode="+string(auth.APIKey)+" needs")
	if err != nil {
		return nil, err
	}

	return auth.New(mode, configs.Lookup(sealer))
}

func run(settings config.Config, logger *slog.Logger) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	capabilities, err := routing.LoadConfig(settings.RoutingConfig)
	if err != nil {
		return err
	}

	// Traces go nowhere unless the deployment names a collector in the environment, so
	// this is a no-op on a laptop and on every deployment that has not asked for them.
	flushTraces, err := tracing.Setup(ctx, "acceleration-router", release)
	if err != nil {
		return err
	}
	defer func() {
		shutdown, cancel := context.WithTimeout(context.Background(), traceFlushTimeout)
		defer cancel()
		if err := flushTraces(shutdown); err != nil {
			logger.Error("could not flush the last traces", "error", err)
		}
	}()

	// Checked before anything is opened, so a deployment that turned on something holding
	// secrets without a keyring is refused at startup rather than on its first secret.
	secrets, err := newSecretSealer(settings)
	if err != nil {
		return err
	}
	// Connectors seal under the same keyring, and only when they are on.
	var connectorSecrets *auth.Sealer
	if settings.Connectors.Enabled {
		connectorSecrets = secrets
	}

	// Postgres and Redis are optional so the API can be brought up for inspection before
	// the data stores exist. /health reports what is missing.
	var pgStore *store.Store
	if settings.Postgres.DSN != "" {
		pgStore, err = openStore(ctx, settings)
		if err != nil {
			return err
		}
		defer pgStore.Close()
	} else {
		logger.Warn("no database configured, statistics will not be recorded", "setting", "postgres.dsn")
	}
	// After the store, which holds the apps' own OAuth clients oauth2_code looks up.
	connectors, err := newConnectorRegistry(settings, api.ConnectorClients(pgStore, connectorSecrets, os.Getenv))
	if err != nil {
		return err
	}
	// Nothing asks it for a credential yet: the session's dispatcher will (T21, AI-851). The
	// events endpoint revokes through it.
	connectorResolver, err := newConnectorResolver(connectors, pgStore, connectorSecrets)
	if err != nil {
		return err
	}

	var liveClient *live.Client
	if settings.Redis.Addr != "" {
		liveClient, err = live.New(live.Options{
			Address:  settings.Redis.Addr,
			Username: settings.Redis.Username,
			Password: settings.Redis.Password,
		})
		if err != nil {
			return err
		}
		defer liveClient.Close()
	} else {
		logger.Warn("no redis configured, routing will not use live health", "setting", "redis.addr")
	}

	// A daily limit is counted in Redis, so a deployment without one caps nothing. That is
	// the right way round: the limit protects against a customer's end users spending more
	// than they were meant to, and refusing every request because the counter is missing
	// would be a worse outage than the one it prevents.
	var limiter *quota.Limiter
	limits := quota.Limits{
		MessagesPerDay: settings.RateLimit.MessagesPerDay,
		TokensPerDay:   settings.RateLimit.TokensPerDay,
	}
	switch {
	case !limits.Enforced():
		logger.Warn("no daily limit configured, an end user may spend without bound",
			"setting", "rate_limit.messages_per_day")
	case liveClient == nil:
		logger.Warn("no redis configured, daily limits will not be enforced", "setting", "redis.addr")
	default:
		if limiter, err = quota.New(liveClient.Redis(), limits, logger); err != nil {
			return err
		}
		logger.Info("capping what one end user may spend in a day",
			"messages", limits.MessagesPerDay, "tokens", limits.TokensPerDay)
	}

	// What every request is measured against -- the key it presented, the app behind it,
	// their policies and what they have configured -- read through Redis and the cache
	// rueidis keeps beside it. Without Redis it is Postgres on every read, which is what
	// this deployment did before the cache existed.
	var configs *appconfig.Store
	if pgStore != nil {
		configs, err = appconfig.New(appconfig.Options{
			Store:    pgStore,
			Address:  settings.Redis.Addr,
			Username: settings.Redis.Username,
			Password: settings.Redis.Password,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer configs.Close()
	}

	// Who each app is acting for, written down once per user rather than once per
	// request: an LRU of the last few thousand this process saw, a key the replicas share
	// in Redis, and the row underneath.
	var endUsers *users.Recorder
	if pgStore != nil {
		endUsers, err = users.New(users.Options{
			Store:    pgStore,
			Address:  settings.Redis.Addr,
			Username: settings.Redis.Username,
			Password: settings.Redis.Password,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer endUsers.Close()
	}

	// Budgets, data policies and prompt injection screening are stored per organization
	// and app, so a deployment without a database enforces none of them.
	var policies *policy.Enforcer
	var gate routing.Gate
	if configs != nil {
		if policies, err = policy.New(configs, logger); err != nil {
			return err
		}
		gate = policies
	}

	if !settings.Agent.SpeculativeReplies {
		logger.Info("asking for each reply only once the flow controller has ruled",
			"env", "ROUTER_SPECULATIVE_REPLIES")
	}

	trustedProxies, err := api.TrustedProxies(settings.TrustedProxies)
	if err != nil {
		return err
	}
	if len(trustedProxies) == 0 {
		logger.Warn("no trusted proxies configured, X-Forwarded-For will be ignored",
			"setting", "trusted_proxies")
	}

	// Voices a customer brought with them live in an object bucket and a few tables. The
	// resolver only reads the tables, so a deployment with a database but no bucket can
	// still speak in voices another one prepared.
	var resolver routing.VoiceResolver
	if configs != nil {
		resolver = voices.NewResolver(configs)
	}

	bucket, err := blob.Open(ctx, settings.VoicesBucketURL)
	if err != nil {
		return err
	}
	if bucket != nil {
		defer bucket.Close()
	}

	// A modality the config says nothing about is simply not served, and its paths 404.
	routers := map[routing.Modality]routing.Inspector{}
	streams := &api.Streams{}

	if section, ok := capabilities[routing.STT]; ok {
		speech, err := sttrouter.New(sttrouter.Options{
			Config:   section,
			Registry: sttrouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Gate:     gate,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer speech.Close()
		routers[routing.STT] = speech
		streams.STT = speech

		// The recording half of the same section. It is a second router rather than a
		// second method because it routes to different models: the batch endpoints,
		// which are the ones declared realtime: false.
		recorded, err := sttrouter.NewRecordings(sttrouter.Options{
			Config:       section,
			Transcribers: sttrouter.DefaultTranscriberRegistry(),
			Store:        pgStore,
			Live:         liveClient,
			Gate:         gate,
			Logger:       logger,
		})
		if err != nil {
			return err
		}
		defer recorded.Close()
		streams.Transcriptions = recorded
	}

	if section, ok := capabilities[routing.TTS]; ok {
		voice, err := ttsrouter.New(ttsrouter.Options{
			Config:   section,
			Registry: ttsrouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Voices:   resolver,
			Gate:     gate,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer voice.Close()
		routers[routing.TTS] = voice
		streams.TTS = voice

		recorded, err := ttsrouter.NewRecordings(ttsrouter.Options{
			Config:    section,
			Recorders: ttsrouter.DefaultRecorderRegistry(),
			Store:     pgStore,
			Live:      liveClient,
			Voices:    resolver,
			Gate:      gate,
			Logger:    logger,
		})
		if err != nil {
			return err
		}
		defer recorded.Close()
		streams.Speech = recorded
	}

	// The classifier is routed for the same reason search is, and is absent for the same
	// reason: a deployment that declares no section for it runs agents that cannot be
	// given a guardrail, and says so when one is asked for rather than ignoring it. It is
	// opened before the LLM router, which screens prompt injection on it.
	var judging *lcmrouter.Router
	if section, ok := capabilities[routing.LCM]; ok {
		judging, err = lcmrouter.New(lcmrouter.Options{
			Config:   section,
			Registry: lcmrouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Gate:     gate,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer judging.Close()
		routers[routing.LCM] = judging
		streams.LCM = judging
	}

	if section, ok := capabilities[routing.LLM]; ok {
		var screen llmrouter.Screen
		if policies != nil {
			screen = policies.Screener(judging)
		}
		chat, err := llmrouter.New(llmrouter.Options{
			Config:   section,
			Registry: llmrouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Quota:    limiter,
			Gate:     gate,
			Screen:   screen,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer chat.Close()
		routers[routing.LLM] = chat
		streams.LLM = chat
	}

	if section, ok := capabilities[routing.STS]; ok {
		conversing, err := stsrouter.New(stsrouter.Options{
			Config:   section,
			Registry: stsrouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Gate:     gate,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer conversing.Close()
		routers[routing.STS] = conversing
		streams.STS = conversing
	}

	// Search is routed like the three above, so a deployment with no key for any of the
	// providers still inspects and reports on them: what stops a session searching is a
	// candidate refusing to be built, not the section being absent.
	var finding *searchrouter.Router
	if section, ok := capabilities[routing.Search]; ok {
		finding, err = searchrouter.New(searchrouter.Options{
			Config:   section,
			Registry: searchrouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Gate:     gate,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer finding.Close()
		routers[routing.Search] = finding
		streams.Search = finding
	}

	// Image generation is served when the config has a section for it, and a deployment
	// with no key for either provider still inspects and reports on them: what stops a
	// picture being drawn is a candidate refusing to be built.
	if section, ok := capabilities[routing.Image]; ok {
		imaging, err := imagerouter.New(imagerouter.Options{
			Config:   section,
			Registry: imagerouter.DefaultRegistry(),
			Store:    pgStore,
			Live:     liveClient,
			Gate:     gate,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer imaging.Close()
		routers[routing.Image] = imaging
		streams.Image = imaging
	}

	// Every Stream action taken for a customer, a call joined, a line made, a transcript
	// written, a token minted, is taken in the app this resolves for them.
	streamClients, err := newStreamClients(settings, pgStore, secrets, logger)
	if err != nil {
		return err
	}
	if err := checkDeploymentApp(ctx, settings, streamClients); err != nil {
		return err
	}
	// An app whose policies require one of its own is never written into the deployment's.
	if stored, ok := streamClients.Stored(); ok && policies != nil {
		stored.SetFloor(policies.RequiresOwnStreamApp)
	}
	if pgStore != nil {
		pgStore.SetStreamPins(streamPins(streamClients))
	}
	go learnDeploymentApp(ctx, streamClients, logger)

	registrations, dlcGate, err := buildDLC(settings, pgStore, liveClient, logger)
	if err != nil {
		return err
	}
	if registrations != nil {
		go registrations.Run(ctx, dlcPollEvery)
	}
	telephony, err := buildPhone(settings, pgStore, liveClient, streamClients, dlcGate, logger)
	if err != nil {
		return err
	}

	// Without a turbopuffer key an agent knows only what its instructions say: the lookup
	// tool is offered to no session, and there is nothing to fill either.
	var base *turbopuffer.Store
	if search, err := turbopuffer.New(turbopuffer.Options{Logger: logger}); err != nil {
		logger.Debug("nothing will be looked up or written down", "error", err)
	} else {
		base = search
		defer base.Close()
	}

	// Which node is running which session. A socket is reached by the relay below, and
	// everything else a session is asked over HTTP is carried to the node that can answer
	// it, which first has to be findable.
	directory, err := newDirectory(settings, liveClient, logger)
	if err != nil {
		return err
	}
	if directory != nil {
		defer directory.Close()
	}

	pluginAuth := &plugins.Auth{
		PublicURL:    settings.PublicURL,
		DashboardURL: settings.DashboardURL,
		Clients:      session.PluginClients(pgStore, secrets),
	}

	// An LLM-only deployment serves text sessions; voice modes validate their own
	// speech dependencies before a call is opened.
	sessions, err := buildSessions(settings, streams, pgStore, configs, liveClient, directory, telephony, base, finding, judging, streamClients, pluginAuth, logger)
	if err != nil {
		return err
	}
	if sessions != nil {
		defer sessions.Shutdown()
	}
	// In app mode every connected app is checked on, and what the router stops acting in
	// takes the sessions pinned to it with it.
	if stored, ok := streamClients.Source().(*streamapp.Stored); ok {
		ended := func(customer string, app int64) {
			if sessions != nil {
				sessions.EndPinned(customer, app)
			}
		}
		go stored.Watch(ctx, streamClients, streamapp.WatchEvery, ended)
	}

	// A campaign is a phone call, a conversation and a row, so it runs only where all
	// three are configured. Elsewhere the campaign paths say so.
	var campaigns *campaign.Runner
	if pgStore != nil && telephony != nil && sessions != nil {
		campaigns, err = campaign.New(campaign.Options{
			Store:    pgStore,
			Phone:    telephony,
			Sessions: sessions,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		defer campaigns.Close()
	}

	// A plugin event opens a conversation and is subscribed to from a row, so events need
	// both. Elsewhere a config's plugin_events are stored and nothing subscribes to them.
	var events *pluginevents.Service
	if pgStore != nil && sessions != nil {
		events, err = pluginevents.New(pluginevents.Options{
			Store:    pgStore,
			Sessions: sessions,
			Auth:     pluginAuth,
			Logger:   logger,
		})
		if err != nil {
			return err
		}
		events.Start()
		defer events.Close()
	}

	// A simulation is a conversation, a model to judge it and a row, so it too runs only
	// where all three are configured. Elsewhere a simulation can be written down but the
	// path that runs it says why it cannot.
	var simulations *simulation.Runner
	if pgStore != nil && sessions != nil && streams.LLM != nil {
		simulations, err = simulation.New(simulation.Options{
			Store:    pgStore,
			Sessions: sessions,
			LLM:      streams.LLM,
			// Speech is what an audio simulation needs and a text one does not, so it is
			// passed where it exists rather than required.
			TTS:    streams.TTS,
			STT:    streams.STT,
			Logger: logger,
		})
		if err != nil {
			return err
		}
		defer simulations.Close()

		// Runs an older process left going are nobody's to finish: the conversations were
		// held in it, and it is gone.
		if err := simulations.Abandon(ctx); err != nil {
			logger.Error("could not write off the runs an older router left going", "error", err)
		}
	}

	// Bringing a voice needs somewhere to keep the recordings, a place to record them and
	// at least one provider willing to be taught. Missing any of those, the voice paths
	// say so rather than half-working.
	voiceService, err := buildVoices(configs, bucket, logger)
	if err != nil {
		return err
	}

	// Keeping a knowledge base filled from a url needs a row, a base and a crawler.
	// Missing any of those, the url paths say so rather than storing a subscription
	// nothing would ever honour.
	pages, err := buildKnowledgeURLs(pgStore, settings.Redis.Addr, base, logger)
	if err != nil {
		return err
	}
	if pages != nil {
		if err := pages.Start(); err != nil {
			return err
		}
		defer pages.Close()
	}

	// Inbound calls are answered by whoever is connected to the dispatch socket, so the
	// pool exists whether or not anybody is: an empty pool is a call nobody answers, which
	// is a different thing from a deployment that does not dispatch at all.
	workers := dispatch.NewPool()
	if sessions != nil {
		sessions.HostTools(workers)
	}

	authenticator, err := newAuthenticator(settings, configs, logger)
	if err != nil {
		return err
	}
	authMode, err := auth.ParseMode(settings.Auth.Mode)
	if err != nil {
		return err
	}

	// The changes recorded for a customer moving away are kept for as long as the move
	// has to finish in, and then they are somebody's storage bill for nothing.
	if pgStore != nil {
		go pruneDataChanges(ctx, pgStore, settings.DataMove.Retention, logger)
	}

	// A session lives in one process's memory, so a deployment running more than one
	// node needs the sessions reachable from whichever node a socket lands on. The relay
	// is what reaches them, and Redis is all it takes: a single-node deployment gets one
	// anyway and nothing is published that nobody asks for.
	var sessionRelay *relay.Bus
	if liveClient != nil {
		sessionRelay, err = relay.New(relay.Options{Redis: liveClient.Redis(), Logger: logger})
		if err != nil {
			return err
		}
	} else {
		logger.Warn("no redis configured, so a session is only reachable on the node running it",
			"setting", "redis.addr")
	}

	options := api.Options{
		Routers:           routers,
		Voices:            voiceService,
		VoiceLibrary:      buildLibrary(logger),
		KnowledgeURLs:     pages,
		Store:             pgStore,
		Configs:           configs,
		Users:             endUsers,
		Live:              liveClient,
		Phone:             telephony,
		Sessions:          sessions,
		Relay:             sessionRelay,
		Directory:         directory,
		Streams:           streams,
		Stream:            streamClients,
		ProxyDeclaresKind: settings.Auth.ProxyDeclaresKind,
		TrustAPIKeyHeader: settings.Stream.TrustAPIKeyHeader,
		DenyRegistration:  settings.Stream.DenyRegistration,
		HookSecret:        settings.Stream.APISecret,
		Campaigns:         campaigns,
		Simulations:       simulations,
		PluginEvents:      events,
		DLC:               registrations,
		Gate:              dlcGate,
		OpsKey:            settings.Auth.OpsKey,
		Dispatch:          workers,
		Quota:             limiter,
		Policies:          policies,
		Connectors:        connectors,
		ConnectorSecrets:  connectorSecrets,
		TrustedProxies:    trustedProxies,
		AuthMode:          authMode,
		DataRetention:     settings.DataMove.Retention,
		CORSOrigins:       settings.CORSOrigins,
		PublicURL:         settings.PublicURL,
		DashboardURL:      settings.DashboardURL,
		Auth:              authenticator,
		Logger:            logger,
	}
	// A nil *resolver.Resolver in the interface would not be a nil interface, so the absence
	// stays absent, and the events endpoint takes no events without it.
	if connectorResolver != nil {
		options.ConnectorResolver = connectorResolver
		options.ConnectorEventSecrets = api.ConnectorEventSecrets(os.Getenv)
	}
	if streamClients.PerApp() {
		// Each registered app signs its own hooks and mints its own tokens, so only work in
		// the deployment's own app goes without.
		if settings.Stream.APIKey == "" {
			logger.Info("no stream.api_key or stream.api_secret set, so only registered apps act in Stream: " +
				"nothing is written into a deployment app, and the old hook paths accept nothing")
		}
	} else {
		if options.HookSecret == "" {
			logger.Warn("no stream.api_secret set, so inbound calls cannot be dispatched: "+
				"the call events Stream sends cannot be told apart from anyone who found the url",
				"hook", "POST /v1/phone/hooks/stream")
		}
		if settings.Stream.APIKey == "" {
			logger.Warn("no stream.api_key set, so nobody can join a call from a browser",
				"endpoint", "POST /v1/agents/calls/{id}/token")
		}
	}
	// A nil *turbopuffer.Store in an interface is not a nil interface, so the absence has
	// to stay absent rather than becoming a value that says it is there.
	if base != nil {
		options.Knowledge = base
	}
	// A channel's provider credentials are sealed under the same keyring as the other
	// secrets the router holds. Without one the channel paths refuse to hold them, which is
	// better than keeping a WhatsApp token in the clear, and nothing is delivered because
	// nothing can be connected. A message that arrives opens a conversation from a row, so
	// answering one needs a database and sessions as well.
	if secrets != nil {
		options.Secrets = secrets
		if pgStore != nil && sessions != nil {
			inbound, err := channels.New(channels.Options{
				Store:    pgStore,
				Sessions: sessions,
				Secrets:  options.Secrets,
				Gate:     dlcGate,
				Logger:   logger,
			})
			if err != nil {
				return err
			}
			options.Channels = inbound
			defer inbound.Close()
		}
	} else {
		logger.Warn("no key encryption keyring set (ROUTER_AUTH_KEK_V1), so no channel can be "+
			"connected: there is nowhere safe to keep a provider's credentials",
			"endpoint", "POST /v1/agents/channels")
	}

	server, err := api.NewServer(options)
	if err != nil {
		return err
	}

	address := settings.Addr

	httpServer := &http.Server{
		Addr:              address,
		Handler:           server.Handler(),
		ReadHeaderTimeout: readHeaderTimeout,
	}

	listening := make(chan error, 1)
	go func() {
		logger.Info("listening", "address", address, "modalities", len(routers))
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			listening <- err
			return
		}
		listening <- nil
	}()

	select {
	case err := <-listening:
		return err
	case <-ctx.Done():
		logger.Info("shutting down")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownGrace)
		defer cancel()
		return httpServer.Shutdown(shutdownCtx)
	}
}

// pruneDataChanges drops the recorded changes nobody can resume from any more, hourly
// until the process stops.
func pruneDataChanges(ctx context.Context, pgStore *store.Store, retention time.Duration, logger *slog.Logger) {
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()
	for {
		removed, err := pgStore.PruneDataChanges(ctx, retention)
		if err != nil {
			logger.Error("could not prune recorded changes", "error", err)
		} else if removed > 0 {
			logger.Debug("pruned recorded changes", "rows", removed)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// openStore opens the database and brings the schema up to date. Every command that
// touches a customer's rows starts here, so migrating is part of opening rather than
// something only serving does.
func openStore(ctx context.Context, settings config.Config) (*store.Store, error) {
	if settings.Postgres.DSN == "" {
		return nil, errors.New("this needs a database: set postgres.dsn")
	}
	pgStore, err := store.Open(settings.Postgres.DSN)
	if err != nil {
		return nil, err
	}
	if err := pgStore.Migrate(ctx); err != nil {
		pgStore.Close()
		return nil, err
	}
	// Seeded whether connectors are on or not: the built-ins are data with no secret in
	// them, and a deployment that turns connectors on later finds them already there.
	if err := pgStore.SeedConnectorDefinitions(ctx, providers.FS); err != nil {
		pgStore.Close()
		return nil, err
	}
	return pgStore, nil
}

// newDirectory wires the register of which node of this deployment is running which
// session, which is what lets a request about a session land on any of them.
//
// It needs Redis to keep the register in and an address this node's peers reach it at.
// Without either it returns nil, and the deployment behaves as a single node did: a
// session is only reachable through the process running it.
//
// The address is not asked for in the ordinary case. A node listens on every interface
// and is reached on one of them, so the port it listens on and the address this host
// answers at are enough to work it out; node.advertise is for the deployment where that
// is not what a peer can reach, which is any of them behind network address translation.
func newDirectory(
	settings config.Config,
	liveClient *live.Client,
	logger *slog.Logger,
) (*node.Directory, error) {
	// The absence is already reported where the relay is built, which is missing for the
	// same reason.
	if liveClient == nil {
		return nil, nil
	}

	address := settings.Node.Advertise
	if address == "" {
		found, err := node.Address(settings.Addr)
		if err != nil {
			return nil, fmt.Errorf("could not work out the address this node's peers reach it at, "+
				"so set node.advertise: %w", err)
		}
		address = found
	}
	logger.Info("reachable by this deployment's other nodes", "address", address,
		"setting", "node.advertise")

	return node.NewDirectory(node.DirectoryOptions{
		Redis:   liveClient.Redis(),
		Address: address,
		Logger:  logger,
	})
}

// buildSessions wires the part of the router that holds conversations rather than
// describing them.
//
// It returns nil when the LLM router is missing. Speech routers are optional for text. The
// factories live here rather than in the session package so the Stream edge, whose Opus
// path is cgo, stays out of everything that only needs to be tested.
func buildSessions(
	settings config.Config,
	streams *api.Streams,
	pgStore *store.Store,
	configs *appconfig.Store,
	liveClient *live.Client,
	directory *node.Directory,
	telephony *phone.Service,
	base *turbopuffer.Store,
	finding *searchrouter.Router,
	judging *lcmrouter.Router,
	stream *streamapp.Clients,
	pluginAuth *plugins.Auth,
	logger *slog.Logger,
) (*session.Manager, error) {
	if streams.LLM == nil {
		logger.Warn("not serving sessions, which need an llm router configured")
		return nil, nil
	}

	// Without a mem0 key a session starts every call knowing nothing but its
	// instructions, which is the behaviour before memory existed.
	var remembering memory.Store
	if recall, err := mem0.New(mem0.Options{Logger: logger}); err != nil {
		logger.Debug("sessions will not remember anything between calls", "error", err)
	} else {
		remembering = recall
	}

	var reading knowledge.Store
	if base != nil {
		reading = base
	}

	var eotClient *agent.EOTClient
	if endpoint := strings.TrimSpace(settings.EOT.Endpoint); endpoint != "" {
		client, err := configuredEOTClient(settings.EOT)
		if err != nil {
			logger.Warn("acoustic endpoint gate is disabled", "reason", "invalid configuration")
		} else {
			eotClient = client
			logger.Info("acoustic endpoint is configured", "mode", settings.EOT.Mode,
				"threshold", settings.EOT.Threshold)
		}
	}

	return session.NewManager(session.ManagerOptions{
		LLM:          streams.LLM,
		STT:          streams.STT,
		TTS:          streams.TTS,
		STS:          streams.STS,
		EOT:          eotClient,
		EOTMode:      agent.EOTMode(settings.EOT.Mode),
		EOTThreshold: settings.EOT.Threshold,
		Memory:       remembering,
		Knowledge:    reading,
		Search:       finding,
		Classifier:   judging,
		Phone:        telephony,
		// Off unless the deployment asks: a reply started before its ruling is paid for
		// whether or not it is spoken.
		SpeculativeReplies: &settings.Agent.SpeculativeReplies,
		Stream:             stream,
		Store:              pgStore,
		Live:               liveClient,
		Logger:             logger,
		Edge:               edgeFor(stream),
		Transcript:         transcriptFor(),
		Configs:            configs,
		Directory:          directory,
		PluginAuth:         pluginAuth,
	})
}

func configuredEOTClient(settings config.EOT) (*agent.EOTClient, error) {
	endpoint := strings.TrimSpace(settings.Endpoint)
	tokenFile := strings.TrimSpace(settings.IDTokenFile)
	if endpoint == "" {
		return nil, nil
	}
	if eotdefaults.IsHostedDemoOrigin(endpoint) {
		if tokenFile != "" {
			return nil, errors.New("eot.id_token_file cannot be used with the hosted demo endpoint; configure a private endpoint")
		}
		if !eotdefaults.IsHostedDemoEndpoint(endpoint) {
			return nil, errors.New("the hosted demo endpoint path must be /v1/eot")
		}
		return agent.NewHostedDemoEOTClient()
	}
	return agent.NewEOTClient(endpoint, tokenFile)
}

// buildKnowledgeURLs wires the control plane for pages a knowledge base is kept filled
// from.
//
// It returns nil unless there is a database to remember a subscription, a Redis to queue the
// reads on, a knowledge base to write the passages into and a key for something that can
// read a page, since a url that is recorded and never fetched is a promise nothing keeps.
// The url paths report the absence.
//
// Exa is built here rather than taken from the search router because the two want opposite
// timeouts: a search happens while somebody waits on the phone, and a live crawl of a page
// nobody is listening to can take as long as it takes.
func buildKnowledgeURLs(
	pgStore *store.Store,
	redisAddress string,
	base *turbopuffer.Store,
	logger *slog.Logger,
) (*urls.Service, error) {
	if pgStore == nil || redisAddress == "" || base == nil {
		logger.Debug("not serving knowledge urls",
			"database", pgStore != nil, "redis", redisAddress != "", "knowledge", base != nil)
		return nil, nil
	}

	reader, err := exa.New(exa.Options{Timeout: crawlTimeout, Logger: logger})
	if err != nil {
		logger.Debug("not serving knowledge urls: nothing can read a page", "error", err)
		return nil, nil
	}

	return urls.New(urls.Options{
		Store:  pgStore,
		Redis:  asynq.RedisClientOpt{Addr: redisAddress},
		Reader: reader,
		Writer: base,
		Logger: logger,
	})
}

// buildLibrary wires the catalogues the speech providers publish, so a voice can be picked
// by name. It needs neither a database nor a bucket: the voices belong to the vendor, and
// all this reads them.
//
// It returns nil when no provider that publishes one has a key here, since an empty
// catalogue and a deployment that cannot browse are not the same thing to somebody
// choosing a voice.
func buildLibrary(logger *slog.Logger) *voices.Catalogue {
	catalogue := voices.NewCatalogue()
	if lister, err := voices.NewElevenLabs(voices.ElevenLabsOptions{}); err == nil {
		catalogue.Register(elevenlabs.ProviderName, lister)
	}
	if lister, err := voices.NewCartesia(voices.CartesiaOptions{}); err == nil {
		catalogue.Register(cartesia.ProviderName, lister)
	}
	if lister, err := voices.NewInworld(voices.InworldOptions{}); err == nil {
		catalogue.Register(inworld.ProviderName, lister)
	}
	if len(catalogue.Providers()) == 0 {
		logger.Debug("no provider here publishes a voice library")
		return nil
	}
	logger.Debug("serving voice libraries", "providers", catalogue.Providers())
	return catalogue
}

// buildVoices wires the control plane for voices a customer brought with them.
//
// It returns nil when there is no database, no bucket, or no provider this deployment has
// a key for, since a voice needs a row, somewhere to keep the recordings and somebody to
// teach them to. The voice paths report the absence rather than failing halfway through an
// upload.
func buildVoices(
	configs *appconfig.Store,
	bucket *blob.Bucket,
	logger *slog.Logger,
) (*voices.Service, error) {
	if configs == nil || bucket == nil {
		logger.Debug("not serving voices of your own", "database", configs != nil, "bucket", bucket != nil)
		return nil, nil
	}

	cloners := voices.NewRegistry()
	if cloner, err := voices.NewElevenLabs(voices.ElevenLabsOptions{}); err == nil {
		cloners.Register(elevenlabs.ProviderName, cloner)
	}
	if cloner, err := voices.NewCartesia(voices.CartesiaOptions{}); err == nil {
		cloners.Register(cartesia.ProviderName, cloner)
	}
	if cloner, err := voices.NewFish(voices.FishOptions{}); err == nil {
		cloners.Register(fish.ProviderName, cloner)
	}
	if len(cloners.Providers()) == 0 {
		logger.Warn("not serving voices of your own: no provider this deployment has a key for can be taught one")
		return nil, nil
	}

	return voices.NewService(voices.Options{
		Store:   configs,
		Bucket:  bucket,
		Cloners: cloners,
		Logger:  logger,
	})
}

// buildPhone wires the telephony service. Stream credentials are only needed to attach a
// number, so a deployment without them still lists vendors and searches for numbers, and
// the operations that need them say so.
func buildPhone(
	settings config.Config,
	pgStore *store.Store,
	liveClient *live.Client,
	stream *streamapp.Clients,
	gate *dlc.Gate,
	logger *slog.Logger,
) (*phone.Service, error) {
	vendorConfig, err := phone.LoadConfig(settings.PhoneConfig)
	if err != nil {
		return nil, err
	}
	if settings.Stream.APIKey == "" || settings.Stream.APISecret == "" {
		logger.Warn("no stream credentials, numbers cannot be attached to a call in the deployment's app")
	}

	var recorder *routing.Recorder
	if pgStore != nil || liveClient != nil {
		recorder = routing.NewRecorder(routing.Phone, pgStore, liveClient, logger)
	}

	return phone.NewService(phone.ServiceOptions{
		Registry:  vendors.Registry(vendorConfig),
		Store:     pgStore,
		Apps:      phoneApps{clients: stream},
		Recorder:  recorder,
		Gate:      gate,
		PublicURL: settings.PublicURL,
		Logger:    logger,
	})
}

// dlcPollEvery is how often the use cases a vendor holds are asked after, for the reports
// its hook missed. A campaign takes days, so this is not what makes one quick.
const dlcPollEvery = 15 * time.Minute

// buildDLC wires 10DLC review and the gate every text and call passes. Without a database
// there is nothing to register or enforce. Without a Telnyx key Stream's approval is final,
// which is a deployment registering with no vendor of ours.
func buildDLC(
	settings config.Config,
	pgStore *store.Store,
	liveClient *live.Client,
	logger *slog.Logger,
) (*dlc.Service, *dlc.Gate, error) {
	if pgStore == nil {
		return nil, nil, nil
	}
	var counters rueidis.Client
	if liveClient != nil {
		counters = liveClient.Redis()
	}
	gate := dlc.NewGate(pgStore, counters, dlc.Sandbox{
		Enabled:            settings.Sandbox.Enabled,
		Recipients:         settings.Sandbox.Recipients,
		MessagesPerDay:     settings.Sandbox.MessagesPerDay,
		AudioMinutesPerDay: settings.Sandbox.AudioMinutesPerDay,
	}, logger)
	if settings.Sandbox.Enabled {
		logger.Info("sandboxing apps with no approved 10DLC use case",
			"recipients", settings.Sandbox.Recipients, "messages", settings.Sandbox.MessagesPerDay,
			"audio_minutes", settings.Sandbox.AudioMinutesPerDay)
	}

	options := dlc.Options{Store: pgStore, PublicURL: settings.PublicURL, Logger: logger}
	if registrar, err := dlctelnyx.New(dlctelnyx.Options{}); err == nil {
		options.Registrar = registrar
	} else {
		logger.Debug("no 10DLC registrar, Stream's approval of a use case is final", "error", err)
	}
	service, err := dlc.NewService(options)
	if err != nil {
		return nil, nil, err
	}
	return service, gate, nil
}
