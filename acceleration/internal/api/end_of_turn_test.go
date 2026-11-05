//go:build integration

package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// EndOfTurnSuite runs a router that asks a chorus-asr server about turns, which publishes
// chorusASRInfo, on an app of each test's own.
type EndOfTurnSuite struct {
	RouterSuite
}

func TestEndOfTurnSuite(t *testing.T) {
	runSuite(t, new(EndOfTurnSuite))
}

func (s *EndOfTurnSuite) SetupSuite() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(chorusASRInfo))
	}))
	s.T().Cleanup(server.Close)
	s.audioFloor, s.audioFloorMode = audiofloor.New(server.URL, audiofloor.Opus, server.Client()), audiofloor.Shadow
	s.RouterSuite.SetupSuite()
}

func (s *EndOfTurnSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *EndOfTurnSuite) sync(name, hash string, endOfTurn any) SyncAgentResult {
	declaration := map[string]any{"name": name, "hash": hash}
	if endOfTurn != nil {
		declaration["end_of_turn"] = endOfTurn
	}
	var result SyncAgentResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync", declaration, &result))
	return result
}

func (s *EndOfTurnSuite) refusedSync(endOfTurn any) (int, string) {
	return s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "refused-" + s.utils.uuid(), "hash": "v1", "end_of_turn": endOfTurn})
}

// A sync stores each way of choosing as sent, the config serves it, and a session opened from
// the config runs by it.
func (s *EndOfTurnSuite) TestASyncStoresTheChoiceAndTheConfigServesIt() {
	for name, chosen := range map[string]EndOfTurn{
		"by-wait":   {MaxWaitMs: pointerTo(900)},
		"by-preset": {Preset: pointerTo("fast")},
		"by-bars":   {Bars: &[]EndOfTurnBar{{FromMs: 0, Bar: 0.97}, {FromMs: 900, Bar: 0.25}}},
	} {
		synced := s.sync(name, "v1", chosen)
		s.Empty(synced.Warnings, name)
		s.Equal(&chosen, synced.Config.EndOfTurn, name)

		var read AgentConfig
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+synced.Config.Id, nil, &read))
		s.Equal(&chosen, read.EndOfTurn, name)

		stored, err := s.store.AgentConfig(context.Background(), s.customerID(), synced.Config.Id)
		s.Require().NoError(err)
		s.Equal(endOfTurnOr(store.EndOfTurn{}, &chosen), stored.EndOfTurn, name)
		s.Equal(stored.EndOfTurn, specOf(CreateSessionRequest{}, s.customerID(), &stored).EndOfTurn, name)
	}
}

// Left out, the stored choice stays, as every setting a directory leaves out does; an empty
// one is the server's default again.
func (s *EndOfTurnSuite) TestAnEmptyChoiceIsTheServersDefaultAgain() {
	s.Equal(&EndOfTurn{Preset: pointerTo("fast")}, s.sync("support", "v1", map[string]any{"preset": "fast"}).Config.EndOfTurn)
	s.Equal(&EndOfTurn{Preset: pointerTo("fast")}, s.sync("support", "v2", nil).Config.EndOfTurn)
	s.Nil(s.sync("support", "v3", map[string]any{}).Config.EndOfTurn)
}

// A sync is refused, before anything is written, for a choice that says more than one thing,
// for bars that don't begin at 0 ms and rise, each strictly between 0 and 1, and for a preset
// or wait the server doesn't publish.
func (s *EndOfTurnSuite) TestASyncRefusesAChoiceTheServerCannotRunWithAClearError() {
	for _, c := range []struct {
		endOfTurn any
		says      string
	}{
		{map[string]any{"preset": "eager"}, `end_of_turn.preset: there is no preset "eager"; the model's are fast and safe`},
		{map[string]any{"max_wait_ms": 400}, "end_of_turn.max_wait_ms is 400, under every wait the model was measured at: the quickest has a caller wait 600 ms at p90"},
		{map[string]any{"max_wait_ms": 900, "preset": "fast"}, "end_of_turn names max_wait_ms and preset: it takes one of max_wait_ms, preset and bars"},
		{map[string]any{"bars": []map[string]any{{"from_ms": 300, "bar": 0.9}}}, "end_of_turn.bars[0].from_ms is 300: the first bar begins at 0 ms into the pause"},
		{map[string]any{"bars": []map[string]any{{"from_ms": 0, "bar": 0.9}, {"from_ms": 900, "bar": 0.3}, {"from_ms": 600, "bar": 0.2}}},
			"end_of_turn.bars[2].from_ms is 600, not after bars[1]'s 900: each bar begins later into the pause than the one before"},
		{map[string]any{"bars": []map[string]any{{"from_ms": 0, "bar": 1}}}, "end_of_turn.bars[0].bar is 1: a bar is a probability strictly between 0 and 1"},
		{map[string]any{"bars": []map[string]any{{"from_ms": 0, "bar": 0.9}, {"from_ms": 500, "bar": 0}}}, "end_of_turn.bars[1].bar is 0: a bar is a probability strictly between 0 and 1"},
		{map[string]any{"bars": []map[string]any{{"from_ms": 0, "bar": 1.2}}}, "end_of_turn.bars[0].bar is 1.2"},
	} {
		status, message := s.refusedSync(c.endOfTurn)
		s.Equal(http.StatusBadRequest, status, "%v", c.endOfTurn)
		s.Contains(message, c.says)
	}
	configs, err := s.store.CustomerAgentConfigs(context.Background(), s.customerID())
	s.Require().NoError(err)
	s.Empty(configs, "nothing is written for a refused sync")
}

// What the schema can say is the schema's: a key nobody reads, no bars, a wait of nothing.
func (s *EndOfTurnSuite) TestAChoiceOutsideTheSchemaIsAValidationFailure() {
	for _, endOfTurn := range []any{
		map[string]any{"max_wait": 900},
		map[string]any{"bars": []map[string]any{}},
		map[string]any{"max_wait_ms": 0},
		map[string]any{"preset": ""},
		map[string]any{"bars": []map[string]any{{"from_ms": -100, "bar": 0.9}}},
	} {
		status, _ := s.refusedSync(endOfTurn)
		s.Equal(http.StatusBadRequest, status, "%v", endOfTurn)
	}
}

// A config created, replaced or patched is held to what a sync is held to.
func (s *EndOfTurnSuite) TestEveryWriteOfAConfigIsHeldToTheSameChoice() {
	eager := &EndOfTurn{Preset: pointerTo("eager")}
	status, message := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{Name: "created", EndOfTurn: eager})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(message, `there is no preset "eager"`)

	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		AgentConfigRequest{Name: "created", EndOfTurn: &EndOfTurn{MaxWaitMs: pointerTo(1200)}}, &created))
	s.Equal(&EndOfTurn{MaxWaitMs: pointerTo(1200)}, created.EndOfTurn)

	status, message = s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id, AgentConfigPatch{EndOfTurn: &EndOfTurn{MaxWaitMs: pointerTo(100)}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(message, "end_of_turn.max_wait_ms is 100, under every wait")

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		AgentConfigPatch{EndOfTurn: &EndOfTurn{Preset: pointerTo("safe")}}, &patched))
	s.Equal(&EndOfTurn{Preset: pointerTo("safe")}, patched.EndOfTurn)

	status, message = s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+created.Id, AgentConfigRequest{Name: "created", EndOfTurn: eager})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(message, `there is no preset "eager"`)

	var replaced AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		AgentConfigRequest{Name: "created"}, &replaced))
	s.Equal(&EndOfTurn{Preset: pointerTo("safe")}, replaced.EndOfTurn, "left out on an update, the stored choice stays")
}

// A session chooses for itself over its config, server-side only, held to the same rules, and
// keeps what it was opened with.
func (s *EndOfTurnSuite) TestASessionChoosesForItselfServerSide() {
	agent := store.AgentConfig{CustomerID: s.customerID(), Name: "session-" + s.utils.uuid(), Mode: store.AgentModeText,
		LLM: "en-low-latency", EndOfTurn: store.EndOfTurn{Preset: "safe"}}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &agent))

	request := CreateSessionRequest{ConfigId: &agent.ID, EndOfTurn: &EndOfTurn{Preset: pointerTo("fast")}}
	for _, device := range []*testClient{s.client, s.guestClient, s.anonymousClient} {
		status, failure := device.failure(http.MethodPost, "/v1/agents/sessions", request)
		s.Equal(http.StatusForbidden, status, "%s", device.kind)
		s.Equal(errDeviceEndOfTurn.Message, failure, "%s", device.kind)
	}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions",
		CreateSessionRequest{ConfigId: &agent.ID, EndOfTurn: &EndOfTurn{Preset: pointerTo("eager")}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `there is no preset "eager"`)

	opened := s.serverClient.createSession(request)
	s.Require().Eventually(func() bool {
		row, err := s.store.StoredSession(context.Background(), s.customerID(), opened.Id)
		return err == nil && row.EndOfTurn.Preset == "fast"
	}, settleFor, 20*time.Millisecond, "the session keeps the choice it was opened with")
	unchosen := s.serverClient.createSession(CreateSessionRequest{ConfigId: &agent.ID})
	// A chat carried on from its row is one somebody wrote in.
	respondIn(&s.RouterSuite, s.serverClient, opened.Id, "hello")
	respondIn(&s.RouterSuite, s.serverClient, unchosen.Id, "hello")
	s.Require().Eventually(func() bool {
		for _, id := range []string{opened.Id, unchosen.Id} {
			if row, err := s.store.StoredSession(context.Background(), s.customerID(), id); err != nil || row.ConversationID == "" {
				return false
			}
		}
		return true
	}, settleFor, 20*time.Millisecond, "the chats were never kept")
	s.serverClient.stopSession(opened.Id)
	s.serverClient.stopSession(unchosen.Id)

	// Carried on from its row, as a chat that ended in another process is (sessionToAnswer).
	_, other := s.otherRouter(nil)
	ctx := context.WithValue(context.Background(), customerContextKey{}, s.customerID())
	ctx = context.WithValue(ctx, kindContextKey{}, auth.KindServer)
	reopened, err := other.reopenedFromRow(ctx, opened.Id)
	s.Require().NoError(err)
	s.Equal(store.EndOfTurn{Preset: "fast"}, reopened.EndOfTurn, "a reopened session keeps its own choice")
	reopened, err = other.reopenedFromRow(ctx, unchosen.Id)
	s.Require().NoError(err)
	s.Equal(store.EndOfTurn{Preset: "safe"}, reopened.EndOfTurn, "one that chose none takes its config's")
}
