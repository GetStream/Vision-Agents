package main

import (
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
// too, with the flow controller's mode off: it asks the server nothing until there are
// words to write, and then for the words alone.
func TestTheAudioFloorTranscribesWithTheModeOff(t *testing.T) {
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
	t.Setenv("ROUTER_CHORUS_ASR_MODE", "off")

	floor, mode, err := openAudioFloor()
	require.NoError(t, err)
	require.NotNil(t, floor)
	require.Equal(t, audiofloor.Mode("off"), mode)

	routers, cleanup, err := buildRouters(context.Background(), 0, 0, floor, slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	t.Cleanup(cleanup)
	session, err := routers.stt.Start(context.Background(), sttrouter.Request{Target: "stream/chorus-asr", CustomerID: "acme"})
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })
	require.Equal(t, audiofloor.DefaultModel, session.Model())
	require.Zero(t, asked.Load(), "no agent, policy or decision")
}

func TestWithoutAServerThereIsNoAudioFloor(t *testing.T) {
	t.Setenv("ROUTER_CHORUS_ASR_URL", "")
	t.Setenv("ROUTER_CHORUS_ASR_MODE", "")
	floor, _, err := openAudioFloor()
	require.NoError(t, err)
	require.Nil(t, floor)

	t.Setenv("ROUTER_CHORUS_ASR_MODE", "act")
	_, _, err = openAudioFloor()
	require.ErrorContains(t, err, "ROUTER_CHORUS_ASR_URL")
}
