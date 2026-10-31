package main

import (
	"log/slog"
	"os"
	"testing"

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
)

func TestBuildSessionsSupportsAnLLMOnlyDeployment(t *testing.T) {
	logger := slog.New(slog.DiscardHandler)
	model, err := llmrouter.New(llmrouter.Options{Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{Provider: "deepseek", Model: "DeepSeek-V4-Flash-0731", Languages: []string{"en"}}},
	}, Registry: llmrouter.DefaultRegistry(), Logger: logger})
	require.NoError(t, err)
	t.Cleanup(model.Close)
	manager, err := buildSessions(config.Defaults(), &api.Streams{LLM: model}, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, session.Connectors{}, nil, nil, nil, logger)
	require.NoError(t, err)
	require.NotNil(t, manager)
	t.Cleanup(func() { require.NoError(t, manager.Shutdown()) })
	missing, err := buildSessions(config.Defaults(), &api.Streams{}, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, session.Connectors{}, nil, nil, nil, logger)
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

	_, err = buildSessions(settings, &api.Streams{LLM: model}, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, session.Connectors{}, nil, nil, nil, logger)
	require.ErrorContains(t, err, "redis.addr")

	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		t.Skip("ROUTER_REDIS_ADDR not set")
	}
	liveClient, err := live.New(live.Options{Address: address})
	require.NoError(t, err)
	t.Cleanup(liveClient.Close)
	manager, err := buildSessions(settings, &api.Streams{LLM: model}, nil, nil, liveClient, nil, nil, nil, nil, nil, nil, nil, nil,
		session.Connectors{}, nil, nil, nil, logger)
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
	liveClient, err := live.New(live.Options{Address: address})
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
	require.Zero(t, running(nil, nil, nil, nil))
	require.Zero(t, running(nil, scheduling.NewScheduler(nil, nil, nil), nil, nil))
}

// TestKnowledgeURLsAreNotServedWithoutRedis: with a database, a knowledge base and a crawler
// key but no Redis, the url paths are off rather than queueing on nothing.
func TestKnowledgeURLsAreNotServedWithoutRedis(t *testing.T) {
	t.Setenv("EXA_API_KEY", "exa-key")

	pages, err := buildKnowledgeURLs(&store.Store{}, config.Redis{}, nil, &metered.Knowledge{}, ingest.Limits{}, slog.New(slog.DiscardHandler))

	require.NoError(t, err)
	require.Nil(t, pages)
}
