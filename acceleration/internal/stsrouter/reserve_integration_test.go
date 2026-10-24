//go:build integration

package stsrouter

import (
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy/policytest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
)

func TestAConversationWhoseLeaseCannotBeCheckedWaitsAndGoesOn(t *testing.T) {
	redis, sharing := policytest.SharedHolds(t)
	money := policytest.Start(t, "stsrouter_reserve", sharing)
	config := routing.ProviderConfig{Provider: "stub", Model: "stub-model", Price: routing.Price{PerMillionInputTokens: 1}}
	prices, err := routing.Prices(routing.Config{routing.STS: {Providers: []routing.ProviderConfig{config}}}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, err)
	owner := money.Owner(t, 100_000_000)
	previous := leaseRenew
	leaseRenew = 50 * time.Millisecond
	t.Cleanup(func() { leaseRenew = previous })
	recorder := money.Recorder(routing.STS, prices)
	provider := newStubSTS()
	session := newSession(provider, config, owner, recorder, recorder.For(routing.LLM))
	t.Cleanup(func() { _ = session.Close() })
	frame := sts.PcmData{Samples: make([]int16, 240), SampleRate: 24_000, Channels: 1}
	require.NoError(t, session.ProcessAudio(frame, sts.Participant{}))

	redis.Stop()
	var refused error
	require.Eventually(t, func() bool {
		refused = session.ProcessAudio(frame, sts.Participant{})
		return refused != nil
	}, 3*time.Second, 10*time.Millisecond, "a renewal is refused once holds go unshared")

	require.ErrorIs(t, refused, policy.ErrSpendCheckUnavailable)
	require.ErrorIs(t, session.Answer("call-1", "done", nil), policy.ErrSpendCheckUnavailable, "a tool's answer is a turn too")
	require.False(t, provider.closed, "the money was not checked, not spent: the conversation stays open")
	redis.Restore()
	require.Eventually(t, func() bool { return session.ProcessAudio(frame, sts.Participant{}) == nil }, 5*time.Second, 10*time.Millisecond,
		"and takes turns again once a renewal lands")
	require.False(t, provider.closed)
}

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
	require.ErrorIs(t, session.Answer("call-1", "done", nil), policy.ErrNoCredits, "a tool's answer is a turn too")
	require.Empty(t, provider.answers)
	require.Empty(t, provider.heard, "the provider heard nothing")
	require.Empty(t, provider.typed)
	require.True(t, provider.closed, "and the conversation is closed")
}
