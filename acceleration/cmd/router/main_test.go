package main

import (
	"log/slog"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/api"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
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
	manager, err := buildSessions(config.Defaults(), &api.Streams{LLM: model}, nil, nil, nil, nil, nil, nil, nil, logger)
	require.NoError(t, err)
	require.NotNil(t, manager)
	t.Cleanup(func() { require.NoError(t, manager.Shutdown()) })
	missing, err := buildSessions(config.Defaults(), &api.Streams{}, nil, nil, nil, nil, nil, nil, nil, logger)
	require.NoError(t, err)
	require.Nil(t, missing)
}

func TestCredentialSealerLoadsCurrentAndRetainedKeyVersions(t *testing.T) {
	t.Setenv(authKEKVersionEnvVar, "2")
	t.Setenv(authKEKEnvVar+"_V1", "old-version-key")
	t.Setenv(authKEKEnvVar+"_V2", "current-version-key")

	sealer, err := newCredentialSealer(config.Config{Auth: config.Auth{KEK: "old-version-key"}})
	require.NoError(t, err)
	require.Equal(t, 2, sealer.CurrentVersion())

	oldSealer, err := auth.NewSealer("old-version-key")
	require.NoError(t, err)
	oldCiphertext, err := oldSealer.Seal("old credential")
	require.NoError(t, err)
	opened, err := sealer.OpenWithAADVersion(oldCiphertext, nil, 1)
	require.NoError(t, err)
	require.Equal(t, "old credential", opened)

	newCiphertext, err := sealer.Seal("new credential")
	require.NoError(t, err)
	opened, err = sealer.OpenWithAADVersion(newCiphertext, nil, 2)
	require.NoError(t, err)
	require.Equal(t, "new credential", opened)
}
