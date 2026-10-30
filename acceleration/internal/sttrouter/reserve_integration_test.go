//go:build integration

package sttrouter

import (
	"io"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy/policytest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func TestATranscriptionWhoseLeaseIsRefusedStopsTakingAudio(t *testing.T) {
	money := policytest.Start(t, "sttrouter_reserve")
	config := routing.ProviderConfig{Provider: "counting", Model: "live", Price: routing.Price{PerAudioHour: 1}}
	prices, err := routing.Prices(routing.Config{routing.STT: {Providers: []routing.ProviderConfig{config}}}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, err)
	// No credit: the lease of its first minute is refused, as a renewal is once money runs out.
	owner := money.Owner(t, 0)
	provider := &countingSTT{emitter: stt.NewEmitter(64)}
	session := newSession(provider, config, owner, money.Recorder(routing.STT, prices), nil, nil)
	t.Cleanup(func() { _ = session.Close() })

	err = session.ProcessAudio(speech(20), stt.Participant{ID: "caller"})

	require.ErrorIs(t, err, policy.ErrNoCredits, "a transcriber bills the audio it is sent, so the frame is refused")
	require.Zero(t, provider.heard.Load(), "the provider heard nothing")
	_, open := <-session.Events()
	require.False(t, open, "and the session is closed")
}
