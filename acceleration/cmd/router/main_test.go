package main

import (
	"bytes"
	"crypto/rand"
	"errors"
	"log/slog"
	"os"
	"os/exec"
	"testing"

	"github.com/getsentry/sentry-go"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
)

func TestBuildSessionsSupportsAnLLMOnlyDeployment(t *testing.T) {
	logger := slog.New(slog.DiscardHandler)
	model, err := llmrouter.New(llmrouter.Options{Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{Provider: "deepseek", Model: "DeepSeek-V4-Flash-0731", Languages: []string{"en"}}},
	}, Registry: llmrouter.DefaultRegistry(), Logger: logger})
	require.NoError(t, err)
	t.Cleanup(model.Close)
	manager, err := buildSessions(config.Defaults(), &api.Streams{LLM: model}, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, session.Connectors{}, nil, nil, nil, nil, logger)
	require.NoError(t, err)
	require.NotNil(t, manager)
	t.Cleanup(func() { require.NoError(t, manager.Shutdown()) })
	missing, err := buildSessions(config.Defaults(), &api.Streams{}, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, session.Connectors{}, nil, nil, nil, nil, logger)
	require.NoError(t, err)
	require.Nil(t, missing)
}

// TestBuildSessionsGivesTheProxyBrokerItsProxy: code_sandbox.broker proxy reaches the session
// manager with the store the proxy reads, and without Redis the router does not start.
func TestBuildSessionsGivesTheProxyBrokerItsProxy(t *testing.T) {
	logger := slog.New(slog.DiscardHandler)
	model, err := llmrouter.New(llmrouter.Options{Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{Provider: "deepseek", Model: "DeepSeek-V4-Flash-0731", Languages: []string{"en"}}},
	}, Registry: llmrouter.DefaultRegistry(), Logger: logger})
	require.NoError(t, err)
	t.Cleanup(model.Close)
	settings := config.Defaults()
	settings.CodeSandbox.Broker, settings.CodeSandbox.ProxyPublicURL = config.CodeSandboxBrokerProxy, "http://203.0.113.7:8091"
	settings.CodeSandbox.ProxyCACert = "cert"

	_, err = buildSessions(settings, &api.Streams{LLM: model}, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, session.Connectors{}, nil, nil, nil, nil, logger)
	require.ErrorContains(t, err, "redis.addr")

	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	liveClient, err := redisready.Retry(address, func() (*live.Client, error) { return live.New(live.Options{Address: address}) })
	require.NoError(t, err)
	t.Cleanup(liveClient.Close)
	manager, err := buildSessions(settings, &api.Streams{LLM: model}, nil, nil, liveClient, nil, nil, nil, nil, nil, nil, nil, nil,
		session.Connectors{}, nil, nil, nil, nil, logger)
	require.NoError(t, err)
	require.NotNil(t, manager)
	t.Cleanup(func() { require.NoError(t, manager.Shutdown()) })
}

func TestADeploymentWithNoDefaultDailyLimitStillCapsAnAppThatSetsOne(t *testing.T) {
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	logger := slog.New(slog.DiscardHandler)
	liveClient, err := redisready.Retry(address, func() (*live.Client, error) { return live.New(live.Options{Address: address}) })
	require.NoError(t, err)
	t.Cleanup(liveClient.Close)
	policies, err := policy.New(&appconfig.Store{}, policy.Settings{}, logger)
	require.NoError(t, err)
	t.Cleanup(policies.Close)

	limiter, err := newDailyLimiter(liveClient, quota.Limits{}, nil, logger)
	require.NoError(t, err)
	require.Nil(t, limiter, "nothing configured anywhere caps nobody")

	limiter, err = newDailyLimiter(liveClient, quota.Limits{}, policies, logger)
	require.NoError(t, err)
	require.NotNil(t, limiter, "an app's policy may set a limit of its own")

	limiter, err = newDailyLimiter(nil, quota.Limits{MessagesPerDay: 10}, policies, logger)
	require.NoError(t, err)
	require.Nil(t, limiter, "a limit nothing can count caps nobody")
}

// TestAnIdleRouterHoldsTheDrainForNothing is the drain's control: a router
// with no sessions holds nothing, with or without an idle scheduled-task
// worker, as before the worker counted.
func TestAnIdleRouterHoldsTheDrainForNothing(t *testing.T) {
	require.Zero(t, running(nil, nil, nil, nil, nil, nil, nil))
	require.Zero(t, running(nil, scheduling.NewScheduler(nil, nil, nil), nil, nil, nil, nil, nil))
}

// TestKnowledgeURLsAreNotServedWithoutRedis: with a database, a knowledge base and a crawler
// key but no Redis, the url paths are off rather than queueing on nothing.
func TestKnowledgeURLsAreNotServedWithoutRedis(t *testing.T) {
	t.Setenv("EXA_API_KEY", "exa-key")

	pages, err := buildKnowledgeURLs(&store.Store{}, config.Redis{}, nil, &metered.Knowledge{}, ingest.Limits{}, slog.New(slog.DiscardHandler))

	require.NoError(t, err)
	require.Nil(t, pages)
}

// A provider that echoes the request's headers in its error puts the key the router sent into the
// failure; the router's log keeps the provider's reason and not the key, and still logs at its level.
func TestTheRoutersLogKeepsAProvidersReasonAndNotTheKeyItEchoed(t *testing.T) {
	key := rand.Text()
	var out bytes.Buffer
	logger := newLogger(&out, slog.LevelWarn)
	logger.Info("below the level")
	logger.Error("agent failure", "error", errors.New("502 Bad Gateway: upstream error: failed to connect to local service\n"+
		"request headers: Authorization: Bearer sk-test-"+key))

	require.NotContains(t, out.String(), key)
	require.NotContains(t, out.String(), "below the level")
	require.Contains(t, out.String(), `msg="agent failure" error="502 Bad Gateway: upstream error: failed to connect to local service\nrequest headers: [redacted]"`)
}

// The router's Sentry events carry a failure's text: the key a provider echoed in it is taken
// out of the exception and the breadcrumbs, and the reason is kept.
func TestTheRoutersSentryEventsKeepAProvidersReasonAndNotTheKeyItEchoed(t *testing.T) {
	key := rand.Text()
	options := sentryOptions()
	require.NotNil(t, options.BeforeSend, "the router's events are sent as they are")
	var sent []*sentry.Event
	redact := options.BeforeSend
	options.BeforeSend = func(event *sentry.Event, hint *sentry.EventHint) *sentry.Event {
		sent = append(sent, redact(event, hint))
		return nil
	}
	client, err := sentry.NewClient(options)
	require.NoError(t, err)
	hub := sentry.NewHub(client, sentry.NewScope())
	echoed := "502 Bad Gateway: upstream error: failed to connect to local service\nrequest headers: Authorization: Bearer sk-test-" + key
	hub.AddBreadcrumb(&sentry.Breadcrumb{Message: echoed}, nil)
	hub.CaptureException(errors.New(echoed))

	require.Len(t, sent, 1)
	reason := "502 Bad Gateway: upstream error: failed to connect to local service\nrequest headers: [redacted]"
	require.Equal(t, reason, sent[0].Exception[len(sent[0].Exception)-1].Value)
	require.Equal(t, reason, sent[0].Breadcrumbs[0].Message)
}

// The logger main builds takes a key out of a startup failure: a config path that names one is
// unreadable, and the failure is logged without it. A path, not ROUTER_ENV: the integration
// build's testenv resets ROUTER_ENV in the child.
func TestTheRouterBinaryLogsAStartupFailureWithoutTheKeyInIt(t *testing.T) {
	if path := os.Getenv("ROUTER_MAIN_TEST_CONFIG"); path != "" {
		os.Args = []string{"router", "--config", path}
		main()
		return
	}
	key := rand.Text()
	command := exec.Command(os.Args[0], "-test.run=^TestTheRouterBinaryLogsAStartupFailureWithoutTheKeyInIt$")
	command.Env = append(os.Environ(), "ROUTER_MAIN_TEST_CONFIG=/missing/Bearer sk-test-"+key)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	require.Error(t, command.Run(), "the router started with a config it cannot read")

	require.NotContains(t, stderr.String(), key)
	require.Contains(t, stderr.String(), `msg="router stopped" error="config: read /missing/[redacted] open /missing/[redacted]`)
}
