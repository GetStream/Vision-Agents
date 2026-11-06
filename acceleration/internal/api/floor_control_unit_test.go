package api

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// stream/chorus-asr asks a router with no chorus-asr server for what it cannot do, and is
// refused as a feature the deployment lacks; empty and leaving it out ask for nothing.
func TestAFloorControlThatAsksIsRefusedWithoutAServer(t *testing.T) {
	without := serverAsking(nil)
	failure, refused := without.floorControlRefusal(new("stream/chorus-asr"))
	require.True(t, refused)
	require.Equal(t, errNoChorusASR, failure)
	require.Equal(t, codeNotConfigured, failure.Code)
	require.Contains(t, failure.Message, "ROUTER_CHORUS_ASR_URL")
	for name, sent := range map[string]*string{"empty": new(""), "left out": nil} {
		_, refused := without.floorControlRefusal(sent)
		require.False(t, refused, name)
	}

	client, _ := chorusASRServer(t, chorusASRInfo)
	_, refused = serverAsking(client).floorControlRefusal(new("stream/chorus-asr"))
	require.False(t, refused)
}

// An agent that chose no floor control asks chorus-asr nothing, as before there was floor
// control: it opts in for itself.
func TestAnUnsetFloorControlIsTheDefaultFlowController(t *testing.T) {
	require.Equal(t, audiofloor.Off, session.FloorModeOf(""))
	require.Equal(t, audiofloor.Act, session.FloorModeOf(store.FloorControlChorusASR))
}

// Sent, a choice replaces the stored one, and empty is the default flow controller again;
// left out, the stored one stays. What is stored renders as it was sent, and none chosen as
// nothing.
func TestAFloorControlSentReplacesTheStoredOne(t *testing.T) {
	require.Equal(t, store.FloorControlChorusASR, floorControlOr("", new("stream/chorus-asr")))
	require.Equal(t, store.FloorControlChorusASR, floorControlOr(store.FloorControlChorusASR, nil), "left out, it stays")
	require.True(t, floorControlOr(store.FloorControlChorusASR, new("")).IsZero(), "empty is the default again")

	require.Equal(t, new("stream/chorus-asr"), renderedFloorControl(store.FloorControlChorusASR))
	require.Nil(t, renderedFloorControl(""))
}

// A session's choice is its request's over its config's, and the session's own is held to the
// router as a config's is, its end_of_turn under the floor control it runs by.
func TestASessionsFloorControlIsItsRequestsOverItsConfigs(t *testing.T) {
	config := &store.AgentConfig{ID: "c", FloorControl: store.FloorControlChorusASR}
	require.Equal(t, store.FloorControlChorusASR, specOf(CreateSessionRequest{}, "acme", config).FloorControl)
	require.True(t, specOf(CreateSessionRequest{FloorControl: new("")}, "acme", config).FloorControl.IsZero())

	failure, refused := serverAsking(nil).sessionFloorRefusal(context.Background(), CreateSessionRequest{FloorControl: new("stream/chorus-asr")}, nil)
	require.True(t, refused)
	require.Equal(t, errNoChorusASR, failure)

	client, asked := chorusASRServer(t, chorusASRInfo)
	s := serverAsking(client)
	eager := &EndOfTurn{Preset: new("eager")}
	_, refused = s.sessionFloorRefusal(context.Background(), CreateSessionRequest{EndOfTurn: eager}, nil)
	require.False(t, refused, "with the default flow controller, end_of_turn does nothing")
	require.Zero(t, asked.Load())
	failure, refused = s.sessionFloorRefusal(context.Background(), CreateSessionRequest{EndOfTurn: eager}, config)
	require.True(t, refused, "with its config's chorus-asr, end_of_turn is checked")
	require.Contains(t, failure.Message, `there is no preset "eager"`)
	_, refused = s.sessionFloorRefusal(context.Background(), CreateSessionRequest{EndOfTurn: eager, FloorControl: new("")}, config)
	require.False(t, refused, "the default flow controller by its own choice over its config's")
}

// Only the app's backend chooses what controls a session's floor.
func TestADeviceMayNotChooseASessionsFloorControl(t *testing.T) {
	failure, refused := refuseServerSideFields(context.Background(), CreateSessionRequest{FloorControl: new("stream/chorus-asr")})
	require.True(t, refused)
	require.Equal(t, errDeviceFloorControl, failure)
	backend := context.WithValue(context.Background(), serverSideContextKey{}, true)
	_, refused = refuseServerSideFields(backend, CreateSessionRequest{FloorControl: new("stream/chorus-asr")})
	require.False(t, refused)
}
