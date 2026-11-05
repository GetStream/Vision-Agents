package main

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
)

// The audio floor server the router's settings name is a transcriber an agent can name here
// too, and the flow controller asks it as the mode says.
func TestTheAudioFloorIsTheRoutersSettings(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"end_of_turn":{"first_check_ms":200,"first_bar":0.7,"later_bar":0.2,"first_pause_logit":0.5,"interval_ms":300,"cap_ms":2500}}`))
	}))
	t.Cleanup(srv.Close)
	for _, name := range []string{postgresEnvVar, redisEnvVar, configEnvVar} {
		t.Setenv(name, "")
	}
	t.Setenv("ROUTER_AUDIO_FLOOR_URL", srv.URL)
	t.Setenv("ROUTER_AUDIO_FLOOR_MODE", "shadow")

	floor, mode, err := openAudioFloor()
	require.NoError(t, err)
	require.NotNil(t, floor)
	require.Equal(t, audiofloor.Shadow, mode)

	routers, cleanup, err := buildRouters(context.Background(), 0, 0, floor, slog.New(slog.DiscardHandler))
	require.NoError(t, err)
	t.Cleanup(cleanup)
	session, err := routers.stt.Start(context.Background(), sttrouter.Request{Target: "audiofloor/audio-floor", CustomerID: "acme"})
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })
	require.Equal(t, audiofloor.DefaultModel, session.Model())
}

func TestWithoutAServerThereIsNoAudioFloor(t *testing.T) {
	t.Setenv("ROUTER_AUDIO_FLOOR_URL", "")
	t.Setenv("ROUTER_AUDIO_FLOOR_MODE", "")
	floor, _, err := openAudioFloor()
	require.NoError(t, err)
	require.Nil(t, floor)

	t.Setenv("ROUTER_AUDIO_FLOOR_MODE", "act")
	_, _, err = openAudioFloor()
	require.ErrorContains(t, err, "ROUTER_AUDIO_FLOOR_URL")
}
