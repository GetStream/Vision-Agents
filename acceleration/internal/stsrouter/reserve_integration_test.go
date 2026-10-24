//go:build integration

package stsrouter

import (
	"io"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy/policytest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
)

func TestAConversationWhoseLeaseIsRefusedStopsAtItsNextTurn(t *testing.T) {
	money := policytest.Start(t, "stsrouter_reserve")
	config := routing.ProviderConfig{Provider: "stub", Model: "stub-model", Price: routing.Price{PerMillionInputTokens: 1}}
	prices, err := routing.Prices(routing.Config{routing.STS: {Providers: []routing.ProviderConfig{config}}}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, err)
	// No credit: the lease of its first minute is refused, as a renewal is once money runs out.
	owner := money.Owner(t, 0)
	recorder := money.Recorder(routing.STS, prices)
	provider := newStubSTS()
	session := newSession(provider, config, owner, recorder, recorder.For(routing.LLM))
	t.Cleanup(func() { _ = session.Close() })

	heard := session.ProcessAudio(sts.PcmData{Samples: make([]int16, 240), SampleRate: 24_000, Channels: 1}, sts.Participant{})
	typed := session.SendText("hello", sts.Participant{})

	require.ErrorIs(t, heard, policy.ErrNoCredits, "a voice model replies to every turn, so the turn is refused")
	require.ErrorIs(t, typed, policy.ErrNoCredits)
	require.Empty(t, provider.heard, "the provider heard nothing")
	require.Empty(t, provider.typed)
	require.True(t, provider.closed, "and the conversation is closed")
}
