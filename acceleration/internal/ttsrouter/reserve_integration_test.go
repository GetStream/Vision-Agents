//go:build integration

package ttsrouter

import (
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy/policytest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// reservingSession is a session over a stub voice whose every character costs $1, for an
// organization with credit micros, reserving through the real enforcer.
func reservingSession(t *testing.T, money *policytest.Money, credit int64) (*Session, *stubTTS, routing.Owner) {
	t.Helper()
	config := routing.ProviderConfig{Provider: "stub", Model: "stub-model", Price: routing.Price{PerMillionChars: 1_000_000}}
	prices, err := routing.Prices(routing.Config{routing.TTS: {Providers: []routing.ProviderConfig{config}}}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, err)
	owner := money.Owner(t, credit)
	provider := newStubTTS()
	session := newSession(provider, config, owner, money.Recorder(routing.TTS, prices), slog.New(slog.NewTextHandler(io.Discard, nil)))
	t.Cleanup(func() { _ = session.Close() })
	return session, provider, owner
}

func TestClosingASessionReleasesWhatItsUtterancesHeld(t *testing.T) {
	money := policytest.Start(t, "ttsrouter_reserve")
	session, _, owner := reservingSession(t, money, 1_000_000)

	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a"}))
	require.ErrorIs(t, money.Spend(t, owner), policy.ErrNoCredits, "its one character is held")

	require.NoError(t, session.Close())

	require.Eventually(t, func() bool { return money.Spend(t, owner) == nil }, 2*time.Second, 10*time.Millisecond,
		"an utterance that never completed holds nothing once the session is closed")
}

func TestTextTheVoiceRefusedIsNoLongerHeld(t *testing.T) {
	money := policytest.Start(t, "ttsrouter_reserve")
	session, provider, owner := reservingSession(t, money, 2_000_000)
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a"}))
	require.NoError(t, money.Spend(t, owner), "$1 of $2 is held")

	provider.refuse = errors.New("the voice dropped the connection")
	require.Error(t, session.Synthesize(tts.Request{ID: "u1", Text: "b"}))

	require.NoError(t, money.Spend(t, owner), "what is held is the one character sent, not the two")
	provider.refuse = errors.New("still down")
	require.Error(t, session.Synthesize(tts.Request{ID: "u1", Text: "cd"}))
	require.NoError(t, money.Spend(t, owner))
}
