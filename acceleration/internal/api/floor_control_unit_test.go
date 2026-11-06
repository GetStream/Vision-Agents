package api

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// floorControl is chorus-asr's floor control in mode, as a client sends it.
func floorControl(mode FloorControlMode) *FloorControl {
	return &FloorControl{Model: floorModelChorusASR, Mode: mode}
}

// shadow or act asks a router with no chorus-asr server for what it cannot do, and is refused
// as a feature the deployment lacks; off and leaving it out ask for nothing.
func TestAFloorControlThatAsksIsRefusedWithoutAServer(t *testing.T) {
	without := serverAsking(nil)
	for _, mode := range []FloorControlMode{floorModeShadow, floorModeAct} {
		failure, refused := without.floorControlRefusal(floorControl(mode))
		require.True(t, refused, mode)
		require.Equal(t, errNoChorusASR, failure, mode)
		require.Equal(t, codeNotConfigured, failure.Code)
		require.Contains(t, failure.Message, "ROUTER_CHORUS_ASR_URL")
	}
	for name, sent := range map[string]*FloorControl{"off": floorControl(floorModeOff), "left out": nil} {
		_, refused := without.floorControlRefusal(sent)
		require.False(t, refused, name)
	}

	client, _ := chorusASRServer(t, chorusASRInfo)
	with := serverAsking(client)
	for _, mode := range []FloorControlMode{floorModeOff, floorModeShadow, floorModeAct} {
		_, refused := with.floorControlRefusal(floorControl(mode))
		require.False(t, refused, mode)
	}
}

// An agent that chose no floor control asks chorus-asr nothing, as before there was floor
// control: it opts in for itself.
func TestAnUnsetFloorControlIsOff(t *testing.T) {
	require.Equal(t, audiofloor.Off, session.FloorModeOf(store.FloorControl{}))
	require.False(t, session.FloorModeOf(store.FloorControl{}).Asks())
	for mode, asks := range map[string]bool{store.FloorModeOff: false, store.FloorModeShadow: true, store.FloorModeAct: true} {
		chosen := session.FloorModeOf(store.FloorControl{Model: store.FloorModelChorusASR, Mode: mode})
		require.Equal(t, audiofloor.Mode(mode), chosen)
		require.Equal(t, asks, chosen.Asks(), mode)
	}
}

// Sent, a choice replaces the stored one; left out, the stored one stays. What is stored
// renders as it was sent, and none chosen as nothing.
func TestAFloorControlSentReplacesTheStoredOne(t *testing.T) {
	acting := store.FloorControl{Model: store.FloorModelChorusASR, Mode: store.FloorModeAct}
	require.Equal(t, acting, floorControlOr(store.FloorControl{}, floorControl(floorModeAct)))
	require.Equal(t, acting, floorControlOr(acting, nil), "left out, it stays")
	require.Equal(t, store.FloorControl{Model: store.FloorModelChorusASR, Mode: store.FloorModeOff},
		floorControlOr(acting, floorControl(floorModeOff)))

	require.Equal(t, floorControl(floorModeAct), renderedFloorControl(acting))
	require.Nil(t, renderedFloorControl(store.FloorControl{}))
}

// A session's choice is its request's over its config's, and the session's own is held to the
// router as a config's is, its end_of_turn under the floor control it runs by.
func TestASessionsFloorControlIsItsRequestsOverItsConfigs(t *testing.T) {
	acting := store.FloorControl{Model: store.FloorModelChorusASR, Mode: store.FloorModeAct}
	config := &store.AgentConfig{ID: "c", FloorControl: acting}
	require.Equal(t, acting, specOf(CreateSessionRequest{}, "acme", config).FloorControl)
	require.Equal(t, store.FloorControl{Model: store.FloorModelChorusASR, Mode: store.FloorModeOff},
		specOf(CreateSessionRequest{FloorControl: floorControl(floorModeOff)}, "acme", config).FloorControl)

	failure, refused := serverAsking(nil).sessionFloorRefusal(context.Background(), CreateSessionRequest{FloorControl: floorControl(floorModeShadow)}, nil)
	require.True(t, refused)
	require.Equal(t, errNoChorusASR, failure)

	client, asked := chorusASRServer(t, chorusASRInfo)
	s := serverAsking(client)
	eager := &EndOfTurn{Preset: new("eager")}
	_, refused = s.sessionFloorRefusal(context.Background(), CreateSessionRequest{EndOfTurn: eager}, nil)
	require.False(t, refused, "with no floor control chosen, end_of_turn does nothing")
	require.Zero(t, asked.Load())
	failure, refused = s.sessionFloorRefusal(context.Background(), CreateSessionRequest{EndOfTurn: eager}, config)
	require.True(t, refused, "acting by its config's choice, end_of_turn is checked")
	require.Contains(t, failure.Message, `there is no preset "eager"`)
	_, refused = s.sessionFloorRefusal(context.Background(), CreateSessionRequest{EndOfTurn: eager, FloorControl: floorControl(floorModeOff)}, config)
	require.False(t, refused, "off by its own choice over its config's")
}

// Only the app's backend chooses whether a model rules a session's floor.
func TestADeviceMayNotChooseASessionsFloorControl(t *testing.T) {
	failure, refused := refuseServerSideFields(context.Background(), CreateSessionRequest{FloorControl: floorControl(floorModeAct)})
	require.True(t, refused)
	require.Equal(t, errDeviceFloorControl, failure)
	backend := context.WithValue(context.Background(), serverSideContextKey{}, true)
	_, refused = refuseServerSideFields(backend, CreateSessionRequest{FloorControl: floorControl(floorModeAct)})
	require.False(t, refused)
}
