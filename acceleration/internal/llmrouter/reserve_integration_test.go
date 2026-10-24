//go:build integration

package llmrouter

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy/policytest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

func TestALongResponseStaysReservedUntilItSettles(t *testing.T) {
	money := policytest.Start(t, "llmrouter_reserve")
	// A million output tokens cost $1, which is all the organization has.
	config := routing.ProviderConfig{Provider: "stub", Model: "reserved", Price: routing.Price{PerMillionOutputTokens: 1}}
	prices, err := routing.Prices(routing.Config{routing.LLM: {Providers: []routing.ProviderConfig{config}}}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, err)
	owner := money.Owner(t, 1_000_000)
	previous := reserveRenewEvery
	reserveRenewEvery = 50 * time.Millisecond
	t.Cleanup(func() { reserveRenewEvery = previous })
	provider := newStubLLM()
	session := newSession(provider, config, owner, money.Recorder(routing.LLM, prices), nil)
	session.admit = money.Enforcer.Admit
	t.Cleanup(func() { _ = session.Close() })

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt(), MaxOutputTokens: 1_000_000})
	require.NoError(t, err)

	// Its reservation lives 100 ms at a time; the answer takes six times that.
	for deadline := time.Now().Add(600 * time.Millisecond); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		require.ErrorIs(t, money.Spend(t, owner), policy.ErrNoCredits, "a response still streaming stays held")
	}
	provider.script(0).OutputText("a long answer")
	provider.script(0).Usage(llm.Usage{InputTokens: 10, OutputTokens: 10})
	provider.script(0).Done()
	drain(stream)

	require.Eventually(t, func() bool { return money.Spend(t, owner) == nil }, 2*time.Second, 10*time.Millisecond,
		"settled at the 10 micros it cost")
}
