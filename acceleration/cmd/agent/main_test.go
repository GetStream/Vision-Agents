package main

import (
	"bytes"
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
)

// The chorus-asr server the router's settings name is a transcriber an agent can name here
// too, with the default flow controller: it asks the server nothing until there are words
// to write, and then for the words alone.
func TestTheAudioFloorTranscribesWithTheDefaultFlowController(t *testing.T) {
	var asked atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		asked.Add(1)
		w.WriteHeader(http.StatusNotImplemented)
	}))
	t.Cleanup(srv.Close)
	for _, name := range []string{postgresEnvVar, redisEnvVar, configEnvVar} {
		t.Setenv(name, "")
	}
	t.Setenv("ROUTER_CHORUS_ASR_URL", srv.URL)

	floor, err := openAudioFloor(slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	require.NotNil(t, floor)
	mode, err := floorControlOf("", floor)
	require.NoError(t, err)
	require.Equal(t, audiofloor.Off, mode)

	routers, cleanup, err := buildRouters(context.Background(), 0, 0, floor, slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	t.Cleanup(cleanup)
	session, err := routers.stt.Start(context.Background(), sttrouter.Request{Target: "stream/chorus-asr", CustomerID: "acme"})
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })
	require.Equal(t, audiofloor.DefaultModel, session.Model())
	require.Zero(t, asked.Load(), "no agent, policy or decision")
}

// The agent's floor control is its own: stream/chorus-asr needs a server, and the router's
// old setting is warned of and acted on for nothing.
func TestTheAgentsFloorControlIsItsOwn(t *testing.T) {
	t.Setenv("ROUTER_CHORUS_ASR_URL", "")
	t.Setenv("ROUTER_CHORUS_ASR_MODE", "act")
	var said bytes.Buffer
	floor, err := openAudioFloor(slog.New(slog.NewTextHandler(&said, nil)))
	require.NoError(t, err)
	require.Nil(t, floor)
	require.Contains(t, said.String(), "ROUTER_CHORUS_ASR_MODE")
	require.Contains(t, said.String(), "floor_control")

	_, err = floorControlOf("stream/chorus-asr", nil)
	require.ErrorContains(t, err, "ROUTER_CHORUS_ASR_URL")
	chosen, err := floorControlOf("stream/chorus-asr", audiofloor.New("http://floor:8080", audiofloor.Opus, nil))
	require.NoError(t, err)
	require.Equal(t, audiofloor.Act, chosen)
	for _, named := range []string{"act", "chorus-asr"} {
		_, err = floorControlOf(named, nil)
		require.ErrorContains(t, err, "stream/chorus-asr, or empty")
	}
}
