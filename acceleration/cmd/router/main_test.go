package main

import (
	"log/slog"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

func TestBuildSessionsSupportsAnLLMOnlyDeployment(t *testing.T) {
	t.Setenv("CHAT_OUTBOX_DIR", "")
	logger := slog.New(slog.DiscardHandler)
	model, err := llmrouter.New(llmrouter.Options{Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{Provider: "deepseek", Model: "DeepSeek-V4-Flash-0731", Languages: []string{"en"}}},
	}, Registry: llmrouter.DefaultRegistry(), Logger: logger})
	require.NoError(t, err)
	t.Cleanup(model.Close)
	manager, err := buildSessions(config.Defaults(), &api.Streams{LLM: model}, nil, nil, nil, nil, nil, nil, logger)
	require.NoError(t, err)
	require.NotNil(t, manager)
	t.Cleanup(func() { require.NoError(t, manager.Shutdown()) })
	missing, err := buildSessions(config.Defaults(), &api.Streams{}, nil, nil, nil, nil, nil, nil, logger)
	require.NoError(t, err)
	require.Nil(t, missing)
}
