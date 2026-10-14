package main

import (
	"context"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
)

func TestBuildSessionsSupportsAnLLMOnlyDeployment(t *testing.T) {
	logger := slog.New(slog.DiscardHandler)
	model, err := llmrouter.New(llmrouter.Options{Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{Provider: "deepseek", Model: "DeepSeek-V4-Flash-0731", Languages: []string{"en"}}},
	}, Registry: llmrouter.DefaultRegistry(), Logger: logger})
	require.NoError(t, err)
	t.Cleanup(model.Close)
	manager, err := buildSessions(config.Defaults(), &api.Streams{LLM: model}, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, session.Connectors{}, logger)
	require.NoError(t, err)
	require.NotNil(t, manager)
	t.Cleanup(func() { require.NoError(t, manager.Shutdown()) })
	missing, err := buildSessions(config.Defaults(), &api.Streams{}, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, session.Connectors{}, logger)
	require.NoError(t, err)
	require.Nil(t, missing)
}

type noLimits struct{}

func (noLimits) Limits(_ context.Context, _ string, defaults quota.Limits) (quota.Limits, error) {
	return defaults, nil
}

func TestADeploymentWithNoDefaultStillCapsAnAppThatSetsALimit(t *testing.T) {
	require.False(t, limitsApply(quota.Limits{}, nil), "nothing configured anywhere caps nobody")
	require.True(t, limitsApply(quota.Limits{}, noLimits{}), "a policy source may hold an app's limit")
	require.True(t, limitsApply(quota.Limits{MessagesPerDay: 10}, nil), "the deployment's default alone caps")
}
