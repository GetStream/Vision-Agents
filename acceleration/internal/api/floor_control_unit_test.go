package api

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// chorusASRInfo is what a chorus-asr server publishes: presets naming the points of its
// measured frontier, whose quickest has a caller wait 800 ms at p90 and fails the dictation
// guard.
const chorusASRInfo = `{"template":"t","end_of_turn":{"first_check_ms":200,"first_pause_logit":0,"interval_ms":300,"cap_ms":2500,` +
	`"default":"safe","presets":{"safe":1100,"fast":500},"frontier":[` +
	`{"max_wait_ms":500,"bars":[{"from_ms":0,"bar":0.8},{"from_ms":500,"bar":0.3}],"checks_in_a_row":1,` +
	`"measured":{"cut_ins":0.1451,"answer_ms_median":500,"answer_ms_p90":800,"dictation_pauses_cut":0.16,"slow_counts_cut":9,"slow_counts":9,"dictation_guard":false}},` +
	`{"max_wait_ms":1100,"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}],"checks_in_a_row":1,` +
	`"measured":{"cut_ins":0.0454,"answer_ms_median":1100,"answer_ms_p90":1100,"dictation_pauses_cut":0.1257,"slow_counts_cut":0,"slow_counts":9,"dictation_guard":true}},` +
	`{"max_wait_ms":1400,"bars":[{"from_ms":0,"bar":0.99},{"from_ms":1100,"bar":0.2}],"checks_in_a_row":2,` +
	`"measured":{"cut_ins":0.0307,"answer_ms_median":1400,"answer_ms_p90":1400,"dictation_pauses_cut":0.1057,"slow_counts_cut":0,"slow_counts":9,"dictation_guard":true}}]}}`

// chorusASRServer is a chorus-asr server answering /v1/info with info, or 500 when info is
// empty, and counting the times it was asked.
func chorusASRServer(t *testing.T, info string) (*audiofloor.Client, *atomic.Int32) {
	var asked atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked.Add(1)
		if info == "" {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(info))
	}))
	t.Cleanup(srv.Close)
	return audiofloor.New(srv.URL, audiofloor.Opus, srv.Client()), &asked
}

// serverAsking is a router with client as its chorus-asr server, nil for none.
func serverAsking(client *audiofloor.Client) *Server {
	return &Server{audioFloor: client, logger: slog.New(slog.DiscardHandler)}
}

// chorusASR is chorus-asr's floor control, waiting at most maxWaitMs, 0 for the server's
// default point.
func chorusASR(maxWaitMs int) store.FloorControl {
	return store.FloorControl{Model: store.FloorControlChorusASR, MaxWaitMs: maxWaitMs}
}

// stream/chorus-asr asks a router with no chorus-asr server for what it cannot do, and is
// refused as a feature the deployment lacks; empty and leaving it out ask for nothing.
func TestAFloorControlThatAsksIsRefusedWithoutAServer(t *testing.T) {
	without := serverAsking(nil)
	for name, sent := range map[string]*FloorControl{
		"model":      {Model: new("stream/chorus-asr")},
		"model+wait": {Model: new("stream/chorus-asr"), MaxWaitMs: new(800)},
	} {
		failure, refused := without.floorControlRefusal(sent, nil)
		require.True(t, refused, name)
		require.Equal(t, errNoChorusASR, failure, name)
		require.Equal(t, codeNotConfigured, failure.Code, name)
	}
	for name, sent := range map[string]*FloorControl{"empty": {}, "left out": nil} {
		_, refused := without.floorControlRefusal(sent, nil)
		require.False(t, refused, name)
	}

	client, _ := chorusASRServer(t, chorusASRInfo)
	_, refused := serverAsking(client).floorControlRefusal(&FloorControl{Model: new("stream/chorus-asr")}, nil)
	require.False(t, refused)
}

// What the schema says is held here too, for the socket's start frame, which no schema checks:
// a wait needs a model, the model is one there is, and a wait is above 0. end_of_turn, which
// max_wait_ms replaced, is refused wherever it is sent.
func TestAFloorControlTheSchemaRefusesIsRefusedWithoutIt(t *testing.T) {
	client, _ := chorusASRServer(t, chorusASRInfo)
	s := serverAsking(client)
	for sent, says := range map[*FloorControl]string{
		{MaxWaitMs: new(800)}:                                "floor_control.model is required",
		{Model: new("stream/another")}:                       `floor_control.model is "stream/another"`,
		{Model: new("stream/chorus-asr"), MaxWaitMs: new(0)}: "floor_control.max_wait_ms is a number of milliseconds above 0",
	} {
		failure, refused := s.floorControlRefusal(sent, nil)
		require.True(t, refused, says)
		require.Equal(t, ErrorTypeInvalidRequest, failure.Type, says)
		require.Contains(t, failure.Message, says)
	}
	for _, endOfTurn := range []any{map[string]any{"max_wait_ms": 800.0}, map[string]any{}, "fast"} {
		failure, refused := s.floorControlRefusal(nil, endOfTurn)
		require.True(t, refused, "%v", endOfTurn)
		require.Equal(t, errEndOfTurn, failure)
		require.Contains(t, failure.Message, "floor_control")
	}
}

// An agent that chose no floor control asks chorus-asr nothing, as before there was floor
// control: it opts in for itself. A wait is the point the session's Hearer ends turns by.
func TestAnUnsetFloorControlIsTheDefaultFlowController(t *testing.T) {
	require.Equal(t, audiofloor.Off, session.FloorModeOf(store.FloorControl{}))
	require.Equal(t, audiofloor.Act, session.FloorModeOf(chorusASR(0)))
	require.Equal(t, audiofloor.EndOfTurn{MaxWaitMs: 800}, session.EndOfTurnOf(chorusASR(800)))
	require.Equal(t, audiofloor.EndOfTurn{}, session.EndOfTurnOf(chorusASR(0)), "no wait is the server's default point")
}

// Sent, a choice replaces the stored one whole, and empty is the default flow controller
// again; left out, the stored one stays. What is stored renders as it was sent, and none
// chosen as nothing.
func TestAFloorControlSentReplacesTheStoredOne(t *testing.T) {
	stored := chorusASR(1400)
	require.Equal(t, stored, floorControlOr(stored, nil), "left out, it stays")
	require.Equal(t, chorusASR(0), floorControlOr(stored, &FloorControl{Model: new("stream/chorus-asr")}), "sent, it replaces the wait too")
	require.Equal(t, chorusASR(800), floorControlOr(store.FloorControl{}, &FloorControl{Model: new("stream/chorus-asr"), MaxWaitMs: new(800)}))
	require.True(t, floorControlOr(stored, &FloorControl{}).IsZero(), "empty is the default again")

	require.Equal(t, &FloorControl{Model: new("stream/chorus-asr"), MaxWaitMs: new(1400)}, renderedFloorControl(stored))
	require.Equal(t, &FloorControl{Model: new("stream/chorus-asr")}, renderedFloorControl(chorusASR(0)))
	require.Nil(t, renderedFloorControl(store.FloorControl{}))
}

// A wait is checked against what the server publishes: one within some measured point's p90
// wait is kept, one shorter than every point's is refused, and one choosing a point that fails
// the dictation guard is kept with what it does to callers.
func TestAMaxWaitIsCheckedAgainstWhatTheServerPublishes(t *testing.T) {
	client, _ := chorusASRServer(t, chorusASRInfo)
	s := serverAsking(client)
	for _, wait := range []int{0, 1100, 5000} {
		warning, message, ok := s.floorControlComplaint(context.Background(), chorusASR(wait), store.FloorControl{})
		require.True(t, ok, "%d: %s", wait, message)
		require.Empty(t, warning, wait)
	}
	warning, message, ok := s.floorControlComplaint(context.Background(), chorusASR(800), store.FloorControl{})
	require.True(t, ok, message)
	require.Equal(t, "floor_control.max_wait_ms chooses the 500 ms point, which fails the dictation guard: it ended 9 of 9 slow counts before their last number, "+
		"and 16.0% of real pauses inside a dictated number, ID or spelled name; an agent whose callers read those out wants a point that passes", warning)

	_, message, ok = s.floorControlComplaint(context.Background(), chorusASR(799), store.FloorControl{})
	require.False(t, ok)
	require.Equal(t, "floor_control.max_wait_ms is 799, under every wait the model was measured at: the quickest has a caller wait 800 ms at p90", message)
}

// A wait stored before is left editable, as it was checked when it was written: the server
// isn't asked about it again. One the server can't be asked about is kept with a warning.
func TestAMaxWaitIsAskedAboutOnlyWhenItChanges(t *testing.T) {
	client, asked := chorusASRServer(t, chorusASRInfo)
	_, message, ok := serverAsking(client).floorControlComplaint(context.Background(), chorusASR(100), chorusASR(100))
	require.True(t, ok, message)
	require.Zero(t, asked.Load())

	down, _ := chorusASRServer(t, "")
	warning, message, ok := serverAsking(down).floorControlComplaint(context.Background(), chorusASR(100), store.FloorControl{})
	require.True(t, ok, message)
	require.Contains(t, warning, "floor_control.max_wait_ms is kept unchecked: the chorus-asr server did not say what it publishes")
	require.NotContains(t, warning, "127.0.0.1", "where the server is stays the router's")
}

// A session's choice is its request's over its config's, and the session's own is held to the
// router as a config's is.
func TestASessionsFloorControlIsItsRequestsOverItsConfigs(t *testing.T) {
	config := &store.AgentConfig{ID: "c", FloorControl: chorusASR(1400)}
	require.Equal(t, chorusASR(1400), specOf(CreateSessionRequest{}, "acme", config).FloorControl)
	require.True(t, specOf(CreateSessionRequest{FloorControl: &FloorControl{}}, "acme", config).FloorControl.IsZero())

	failure, refused := serverAsking(nil).sessionFloorRefusal(context.Background(), CreateSessionRequest{FloorControl: &FloorControl{Model: new("stream/chorus-asr")}})
	require.True(t, refused)
	require.Equal(t, errNoChorusASR, failure)

	client, asked := chorusASRServer(t, chorusASRInfo)
	s := serverAsking(client)
	_, refused = s.sessionFloorRefusal(context.Background(), CreateSessionRequest{ConfigId: new("c")})
	require.False(t, refused, "a session that sends none takes its config's, checked when it was written")
	require.Zero(t, asked.Load())
	failure, refused = s.sessionFloorRefusal(context.Background(), CreateSessionRequest{FloorControl: &FloorControl{Model: new("stream/chorus-asr"), MaxWaitMs: new(500)}})
	require.True(t, refused, "its own wait is checked")
	require.Contains(t, failure.Message, "floor_control.max_wait_ms is 500, under every wait")
	failure, refused = s.sessionFloorRefusal(context.Background(), CreateSessionRequest{EndOfTurn: map[string]any{"preset": "fast"}})
	require.True(t, refused)
	require.Equal(t, errEndOfTurn, failure)
}

// Only the app's backend chooses what controls a session's floor.
func TestADeviceMayNotChooseASessionsFloorControl(t *testing.T) {
	sent := CreateSessionRequest{FloorControl: &FloorControl{Model: new("stream/chorus-asr")}}
	failure, refused := refuseServerSideFields(context.Background(), sent)
	require.True(t, refused)
	require.Equal(t, errDeviceFloorControl, failure)
	backend := context.WithValue(context.Background(), serverSideContextKey{}, true)
	_, refused = refuseServerSideFields(backend, sent)
	require.False(t, refused)
}
