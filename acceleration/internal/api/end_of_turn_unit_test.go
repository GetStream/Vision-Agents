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
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// chorusInfo is what a chorus-asr server publishes: two presets, and a measured frontier
// whose quickest point has a caller wait 600 ms at p90.
const chorusInfo = `{"template":"t","end_of_turn":{"first_check_ms":200,"first_pause_logit":0,"interval_ms":300,"cap_ms":2500,"default":"safe",` +
	`"presets":{"safe":{"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}]},"fast":{"bars":[{"from_ms":0,"bar":0.8},{"from_ms":500,"bar":0.3}]}},` +
	`"frontier":[` +
	`{"max_wait_ms":600,"bars":[{"from_ms":0,"bar":0.7},{"from_ms":300,"bar":0.35}],"measured":{"cut_ins":0.15,"answer_ms_median":400,"answer_ms_p90":600,"dictation_pauses_cut":0.4}},` +
	`{"max_wait_ms":1200,"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}],"measured":{"cut_ins":0.045,"answer_ms_median":1100,"answer_ms_p90":1100,"dictation_pauses_cut":0.126}}]}}`

// chorusServer is a chorus-asr server answering /v1/info with info, or 500 when info is
// empty, and counting the times it was asked.
func chorusServer(t *testing.T, info string) (*audiofloor.Client, *atomic.Int32) {
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

func serverAsking(client *audiofloor.Client, mode audiofloor.Mode) *Server {
	return &Server{audioFloor: asksAboutTurns(client, mode), logger: slog.New(slog.DiscardHandler)}
}

// What a choice says is checked whatever the router asks; what it names, against what the
// server publishes.
func TestAnEndOfTurnIsCheckedAgainstWhatTheServerPublishes(t *testing.T) {
	client, _ := chorusServer(t, chorusInfo)
	s := serverAsking(client, audiofloor.Act)
	for _, chosen := range []store.EndOfTurn{
		{},
		{Preset: "fast"},
		{MaxWaitMs: 600},
		{MaxWaitMs: 5000},
		{Bars: []store.EndOfTurnBar{{FromMs: 0, Bar: 0.9}, {FromMs: 800, Bar: 0.4}}},
	} {
		warning, message, ok := s.endOfTurnComplaint(context.Background(), chosen, store.EndOfTurn{})
		require.True(t, ok, "%+v: %s", chosen, message)
		require.Empty(t, warning)
	}
	for _, c := range []struct {
		chosen store.EndOfTurn
		says   string
	}{
		{store.EndOfTurn{Preset: "eager"}, `end_of_turn.preset: there is no preset "eager"; the model's are fast and safe`},
		{store.EndOfTurn{MaxWaitMs: 300}, "end_of_turn.max_wait_ms is 300, under every wait the model was measured at: the quickest has a caller wait 600 ms at p90"},
		{store.EndOfTurn{MaxWaitMs: 900, Preset: "fast"}, "end_of_turn names max_wait_ms and preset: it takes one of max_wait_ms, preset and bars"},
		{store.EndOfTurn{Bars: []store.EndOfTurnBar{{FromMs: 100, Bar: 0.9}}}, "end_of_turn.bars[0].from_ms is 100: the first bar begins at 0 ms into the pause"},
		{store.EndOfTurn{Bars: []store.EndOfTurnBar{{FromMs: 0, Bar: 0.9}, {FromMs: 0, Bar: 0.5}}}, "end_of_turn.bars[1].from_ms is 0, not after bars[0]'s 0"},
		{store.EndOfTurn{Bars: []store.EndOfTurnBar{{FromMs: 0, Bar: 1}}}, "end_of_turn.bars[0].bar is 1: a bar is a probability strictly between 0 and 1"},
		{store.EndOfTurn{Bars: []store.EndOfTurnBar{{FromMs: 0, Bar: 0}}}, "end_of_turn.bars[0].bar is 0: a bar is a probability strictly between 0 and 1"},
	} {
		_, message, ok := s.endOfTurnComplaint(context.Background(), c.chosen, store.EndOfTurn{})
		require.False(t, ok, "%+v", c.chosen)
		require.Contains(t, message, c.says)
	}
}

// A choice stored before is left editable, as it was checked when it was written: the server
// isn't asked about it again.
func TestAnUnchangedEndOfTurnIsNotAskedAboutAgain(t *testing.T) {
	client, asked := chorusServer(t, chorusInfo)
	s := serverAsking(client, audiofloor.Shadow)
	stored := store.EndOfTurn{Preset: "gone"}
	_, message, ok := s.endOfTurnComplaint(context.Background(), stored, stored)
	require.True(t, ok, message)
	require.Zero(t, asked.Load())
}

// A router that asks the server nothing about turns, or a server that can't say what it
// publishes, keeps the choice and says why it may not do what it says.
func TestAnEndOfTurnThatCannotBeCheckedIsKeptWithAWarning(t *testing.T) {
	client, _ := chorusServer(t, chorusInfo)
	for _, s := range []*Server{serverAsking(nil, audiofloor.Act), serverAsking(client, ""), serverAsking(client, "off")} {
		warning, message, ok := s.endOfTurnComplaint(context.Background(), store.EndOfTurn{Preset: "eager"}, store.EndOfTurn{})
		require.True(t, ok, message)
		require.Equal(t, "end_of_turn is kept, and does nothing on this router, which asks chorus-asr nothing about turns", warning)
	}

	down, _ := chorusServer(t, "")
	warning, message, ok := serverAsking(down, audiofloor.Act).endOfTurnComplaint(context.Background(), store.EndOfTurn{Preset: "eager"}, store.EndOfTurn{})
	require.True(t, ok, message)
	require.Contains(t, warning, "end_of_turn is kept unchecked: the chorus-asr server did not say what it publishes")
	require.NotContains(t, warning, "127.0.0.1", "where the server is stays the router's")

	_, message, ok = serverAsking(nil, audiofloor.Act).endOfTurnComplaint(context.Background(), store.EndOfTurn{MaxWaitMs: 1, Preset: "fast"}, store.EndOfTurn{})
	require.False(t, ok, "what a choice says is checked with no server to ask")
	require.Contains(t, message, "it takes one of max_wait_ms, preset and bars")
}

// Sent, a choice replaces the stored one whole, and an empty one is the server's default;
// left out, the stored one stays. What is stored renders as it was sent.
func TestAnEndOfTurnSentReplacesTheStoredOneWhole(t *testing.T) {
	stored := store.EndOfTurn{Preset: "fast"}
	require.Equal(t, stored, endOfTurnOr(stored, nil))
	require.Equal(t, store.EndOfTurn{MaxWaitMs: 900}, endOfTurnOr(stored, &EndOfTurn{MaxWaitMs: ptrTo(900)}))
	require.True(t, endOfTurnOr(stored, &EndOfTurn{}).IsZero())

	bars := []EndOfTurnBar{{FromMs: 0, Bar: 0.9}, {FromMs: 700, Bar: 0.3}}
	chosen := endOfTurnOr(store.EndOfTurn{}, &EndOfTurn{Bars: &bars})
	require.Equal(t, store.EndOfTurn{Bars: []store.EndOfTurnBar{{FromMs: 0, Bar: 0.9}, {FromMs: 700, Bar: 0.3}}}, chosen)
	require.Equal(t, &EndOfTurn{Bars: &bars}, renderedEndOfTurn(chosen))
	require.Equal(t, &EndOfTurn{Preset: ptrTo("fast")}, renderedEndOfTurn(stored))
	require.Nil(t, renderedEndOfTurn(store.EndOfTurn{}))
}

// A session's choice is its request's over its config's.
func TestASessionsEndOfTurnIsItsRequestsOverItsConfigs(t *testing.T) {
	config := &store.AgentConfig{ID: "c", EndOfTurn: store.EndOfTurn{Preset: "fast"}}
	require.Equal(t, store.EndOfTurn{Preset: "fast"}, specOf(CreateSessionRequest{}, "acme", config).EndOfTurn)
	require.Equal(t, store.EndOfTurn{MaxWaitMs: 900},
		specOf(CreateSessionRequest{EndOfTurn: &EndOfTurn{MaxWaitMs: ptrTo(900)}}, "acme", config).EndOfTurn)
	require.True(t, specOf(CreateSessionRequest{EndOfTurn: &EndOfTurn{}}, "acme", config).EndOfTurn.IsZero())
}

// Only the app's backend chooses how eager a session is to answer.
func TestADeviceMayNotChooseASessionsEndOfTurn(t *testing.T) {
	failure, refused := refuseServerSideFields(context.Background(), CreateSessionRequest{EndOfTurn: &EndOfTurn{Preset: ptrTo("fast")}})
	require.True(t, refused)
	require.Equal(t, errDeviceEndOfTurn, failure)
	backend := context.WithValue(context.Background(), serverSideContextKey{}, true)
	_, refused = refuseServerSideFields(backend, CreateSessionRequest{EndOfTurn: &EndOfTurn{Preset: ptrTo("fast")}})
	require.False(t, refused)
}

func ptrTo[T any](value T) *T { return &value }
