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
	logger := slog.New(slog.DiscardHandler)
	model, err := llmrouter.New(llmrouter.Options{Config: routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{Provider: "deepseek", Model: "DeepSeek-V4-Flash-0731", Languages: []string{"en"}}},
	}, Registry: llmrouter.DefaultRegistry(), Logger: logger})
	require.NoError(t, err)
	t.Cleanup(model.Close)
	manager, err := buildSessions(config.Defaults(), &api.Streams{LLM: model}, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, logger)
	require.NoError(t, err)
	require.NotNil(t, manager)
	t.Cleanup(func() { require.NoError(t, manager.Shutdown()) })
	missing, err := buildSessions(config.Defaults(), &api.Streams{}, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, logger)
	require.NoError(t, err)
	require.Nil(t, missing)
}

func TestConfiguredEOTClientUsesHostedDefaultAndPreservesOverrides(t *testing.T) {
	defaults := config.Defaults().EOT
	if defaults.Endpoint == "" || defaults.Mode != "primary" || defaults.Threshold != 0.5 {
		t.Fatalf("unexpected hosted router defaults: %+v", defaults)
	}
	client, err := configuredEOTClient(defaults)
	require.NoError(t, err)
	require.NotNil(t, client)

	disabled := defaults
	disabled.Endpoint = ""
	client, err = configuredEOTClient(disabled)
	require.NoError(t, err)
	require.Nil(t, client)

	private := defaults
	private.Endpoint = "https://private.example/v1/eot"
	private.IDTokenFile = "/run/secrets/private-eot-token"
	client, err = configuredEOTClient(private)
	require.NoError(t, err)
	require.NotNil(t, client, "an explicit private endpoint keeps the authenticated constructor")

	publicWithToken := defaults
	publicWithToken.IDTokenFile = "/run/secrets/wrong-token"
	client, err = configuredEOTClient(publicWithToken)
	require.ErrorContains(t, err, "cannot be used with the hosted demo endpoint")
	require.Nil(t, client)

	rootForm := defaults
	rootForm.Endpoint = "https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app"
	client, err = configuredEOTClient(rootForm)
	require.NoError(t, err)
	require.NotNil(t, client)
	rootForm.IDTokenFile = "/run/secrets/wrong-token"
	client, err = configuredEOTClient(rootForm)
	require.ErrorContains(t, err, "cannot be used with the hosted demo endpoint")
	require.Nil(t, client)
}
