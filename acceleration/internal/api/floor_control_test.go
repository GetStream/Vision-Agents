//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// FloorControlSuite runs a router with a chorus-asr server, which publishes chorusASRInfo, on
// an app of each test's own.
type FloorControlSuite struct {
	RouterSuite
}

func TestFloorControlSuite(t *testing.T) {
	runSuite(t, new(FloorControlSuite))
}

func (s *FloorControlSuite) SetupSuite() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(chorusASRInfo))
	}))
	s.T().Cleanup(server.Close)
	s.audioFloor = audiofloor.New(server.URL, audiofloor.Opus, server.Client())
	s.RouterSuite.SetupSuite()
}

func (s *FloorControlSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// syncFloor syncs a directory naming fields beside its name and hash, which must be taken.
func (s *RouterSuite) syncFloor(name, hash string, fields map[string]any) SyncAgentResult {
	declaration := map[string]any{"name": name, "hash": hash}
	for field, value := range fields {
		declaration[field] = value
	}
	var result SyncAgentResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync", declaration, &result))
	return result
}

// refusedWith is what a refused request was answered with: its status, code and message.
func (s *RouterSuite) refusedWith(client *testClient, method, path string, body any) (int, string, string) {
	status, payload := client.call(method, path, body)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	return status, answered.Error.Code, answered.Error.Message
}

// A sync stores the floor control as sent, the config serves it, and a session opened from
// the config runs by it.
func (s *FloorControlSuite) TestASyncStoresTheFloorControlAndTheConfigServesIt() {
	synced := s.syncFloor("ruled", "v1", map[string]any{"floor_control": "stream/chorus-asr"})
	s.Empty(synced.Warnings)
	s.Equal(new("stream/chorus-asr"), synced.Config.FloorControl)

	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+synced.Config.Id, nil, &read))
	s.Equal(new("stream/chorus-asr"), read.FloorControl)

	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), synced.Config.Id)
	s.Require().NoError(err)
	s.Equal(store.FloorControlChorusASR, stored.FloorControl)
	s.Equal(stored.FloorControl, specOf(CreateSessionRequest{}, s.customerID(), &stored).FloorControl)
}

// Left out, an agent has the default flow controller, on a router with a chorus-asr server:
// none is stored or served, and its sessions ask chorus-asr nothing. Once chosen, a sync that
// leaves it out keeps it, and an empty one is the default flow controller again.
func (s *FloorControlSuite) TestAnUnsetFloorControlIsTheDefaultFlowController() {
	unset := s.syncFloor("unset", "v1", nil)
	s.Nil(unset.Config.FloorControl)
	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), unset.Config.Id)
	s.Require().NoError(err)
	s.True(stored.FloorControl.IsZero())
	s.Equal(audiofloor.Off, session.FloorModeOf(specOf(CreateSessionRequest{}, s.customerID(), &stored).FloorControl))

	s.Equal(new("stream/chorus-asr"), s.syncFloor("support", "v1", map[string]any{"floor_control": "stream/chorus-asr"}).Config.FloorControl)
	s.Equal(new("stream/chorus-asr"), s.syncFloor("support", "v2", nil).Config.FloorControl, "left out, the stored choice stays")
	s.Nil(s.syncFloor("support", "v3", map[string]any{"floor_control": ""}).Config.FloorControl, "empty is the default flow controller again")
}

// A floor control naming no model that can control the floor, or not a string, is refused
// before anything is written.
func (s *FloorControlSuite) TestAFloorControlNamingNothingKnownIsRefused() {
	for _, floorControl := range []any{"chorus-asr", "stream/another", "act", map[string]any{"model": "chorus-asr", "mode": "act"}, true} {
		status, code, message := s.refusedWith(s.serverClient, http.MethodPost, "/v1/agents/sync",
			map[string]any{"name": "refused-" + s.utils.uuid(), "hash": "v1", "floor_control": floorControl})
		s.Equal(http.StatusBadRequest, status, "%v", floorControl)
		s.Equal("validation_failed", code, "%v: %s", floorControl, message)
	}
	configs, err := s.store.CustomerAgentConfigs(context.Background(), s.customerID())
	s.Require().NoError(err)
	s.Empty(configs, "nothing is written for a refused sync")
}

// A config created, replaced or patched takes floor control as a sync does: a patch sent
// replaces it, one left out keeps it, and an empty one is the default flow controller again.
func (s *FloorControlSuite) TestEveryWriteOfAConfigTakesTheFloorControl() {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		AgentConfigRequest{Name: "created", FloorControl: new("stream/chorus-asr")}, &created))
	s.Equal(new("stream/chorus-asr"), created.FloorControl)

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		AgentConfigPatch{Instructions: pointerTo("Sell ferry tickets.")}, &patched))
	s.Equal(new("stream/chorus-asr"), patched.FloorControl, "a patch that leaves it out keeps it")

	var replaced AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		AgentConfigRequest{Name: "created"}, &replaced))
	s.Equal(new("stream/chorus-asr"), replaced.FloorControl, "left out on an update, the stored choice stays")

	var cleared AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		AgentConfigPatch{FloorControl: new("")}, &cleared))
	s.Nil(cleared.FloorControl, "an empty one is the default flow controller again")
	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), created.Id)
	s.Require().NoError(err)
	s.True(stored.FloorControl.IsZero())

	var chosen AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		AgentConfigPatch{FloorControl: new("stream/chorus-asr")}, &chosen))
	s.Equal(new("stream/chorus-asr"), chosen.FloorControl)
}

// end_of_turn does nothing with the default flow controller, and a sync is told so; once
// chorus-asr controls the floor it is checked against what the server publishes, the choice
// the agent already had included.
func (s *FloorControlSuite) TestAnEndOfTurnIsReadByTheAgentsFloorControl() {
	eager := map[string]any{"preset": "eager"}
	kept := s.syncFloor("eager", "v1", map[string]any{"end_of_turn": eager})
	s.Equal([]string{"end_of_turn is kept, and does nothing unless floor_control is stream/chorus-asr: chorus-asr is asked nothing about the agent's turns"}, kept.Warnings)

	status, message := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "eager", "hash": "v2", "floor_control": "stream/chorus-asr"})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(message, `end_of_turn.preset: there is no preset "eager"`)

	acting := s.syncFloor("fast", "v1", map[string]any{"end_of_turn": map[string]any{"preset": "fast"}, "floor_control": "stream/chorus-asr"})
	s.Require().Len(acting.Warnings, 1)
	s.Contains(acting.Warnings[0], "end_of_turn chooses the 500 ms point, which fails the dictation guard")
}

// A session chooses for itself over its config, server-side only, and keeps what it was
// opened with, which carrying it on from its row keeps.
func (s *FloorControlSuite) TestASessionChoosesItsFloorControlServerSide() {
	agent := store.AgentConfig{CustomerID: s.customerID(), Name: "session-" + s.utils.uuid(), Mode: store.AgentModeText,
		LLM: "en-low-latency", FloorControl: store.FloorControlChorusASR}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &agent))

	request := CreateSessionRequest{ConfigId: &agent.ID, FloorControl: new("")}
	for _, device := range []*testClient{s.client, s.guestClient, s.anonymousClient} {
		status, failure := device.failure(http.MethodPost, "/v1/agents/sessions", request)
		s.Equal(http.StatusForbidden, status, "%s", device.kind)
		s.Equal(errDeviceFloorControl.Message, failure, "%s", device.kind)
	}

	opened := s.serverClient.createSession(request)
	unchosen := s.serverClient.createSession(CreateSessionRequest{ConfigId: &agent.ID})
	s.Require().Eventually(func() bool {
		row, err := s.store.StoredSession(context.Background(), s.customerID(), unchosen.Id)
		return err == nil && row.FloorControl == store.FloorControlChorusASR
	}, settleFor, 20*time.Millisecond, "a session takes its config's")
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
	row, err := s.store.StoredSession(context.Background(), s.customerID(), opened.Id)
	s.Require().NoError(err)
	s.True(row.FloorControl.IsZero(), "the session chose the default flow controller over its config's")
	s.serverClient.stopSession(opened.Id)
	s.serverClient.stopSession(unchosen.Id)

	_, other := s.otherRouter(nil)
	ctx := context.WithValue(context.Background(), customerContextKey{}, s.customerID())
	ctx = context.WithValue(ctx, kindContextKey{}, auth.KindServer)
	reopened, err := other.reopenedFromRow(ctx, unchosen.Id)
	s.Require().NoError(err)
	s.Equal(store.FloorControlChorusASR, reopened.FloorControl, "a reopened session keeps what it was opened with")
}

// FloorControlWithoutChorusASRSuite runs a router with no chorus-asr server.
type FloorControlWithoutChorusASRSuite struct {
	RouterSuite
}

func TestFloorControlWithoutChorusASRSuite(t *testing.T) {
	runSuite(t, new(FloorControlWithoutChorusASRSuite))
}

func (s *FloorControlWithoutChorusASRSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// shadow or act asks a router with no chorus-asr server for what it cannot do: every write
// that asks is refused, as a feature the deployment lacks, saying what the router needs, and
// nothing is written. off asks nothing of it.
func (s *FloorControlWithoutChorusASRSuite) TestAFloorControlThatOptsInIsRefusedWithoutAServer() {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		AgentConfigRequest{Name: "default", FloorControl: new("")}, &created))
	s.Nil(created.FloorControl)
	s.Nil(s.syncFloor("unset", "v1", nil).Config.FloorControl)

	chorusASR := new("stream/chorus-asr")
	for _, write := range []struct {
		method, path string
		body         any
	}{
		{http.MethodPost, "/v1/agents/sync", map[string]any{"name": "refused", "hash": "v1", "floor_control": chorusASR}},
		{http.MethodPost, "/v1/agents/configs", AgentConfigRequest{Name: "refused", FloorControl: chorusASR}},
		{http.MethodPut, "/v1/agents/configs/" + created.Id, AgentConfigRequest{Name: "default", FloorControl: chorusASR}},
		{http.MethodPatch, "/v1/agents/configs/" + created.Id, AgentConfigPatch{FloorControl: chorusASR}},
		{http.MethodPost, "/v1/agents/sessions", CreateSessionRequest{ConfigId: &created.Id, FloorControl: chorusASR}},
	} {
		status, code, message := s.refusedWith(s.serverClient, write.method, write.path, write.body)
		s.Equal(http.StatusBadRequest, status, "%s %s", write.method, write.path)
		s.Equal(codeNotConfigured, code, "%s %s", write.method, write.path)
		s.Equal(errNoChorusASR.Message, message)
	}
	configs, err := s.store.CustomerAgentConfigs(context.Background(), s.customerID())
	s.Require().NoError(err)
	s.Len(configs, 2, "nothing more is written")
	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), created.Id)
	s.Require().NoError(err)
	s.True(stored.FloorControl.IsZero(), "and nothing is changed")
}

// A socket's start frame is refused the floor control POST /v1/agents/sessions refuses, with
// the same message, code and type.
func (s *SocketSessionSuite) TestASocketSessionThatAsksChorusASRWithoutAServerIsRefused() {
	sent := map[string]any{"floor_control": "stream/chorus-asr"}
	_, body := s.serverClient.call(http.MethodPost, "/v1/agents/sessions", sent)
	var created ErrorResponse
	s.Require().NoError(json.Unmarshal(body, &created))
	s.Equal(errNoChorusASR.Message, created.Error.Message)

	connection := s.serverClient.opens("/v1/agents/socket")
	s.Require().NoError(connection.WriteJSON(map[string]any{"type": "start", "sample_rate": 24_000, "session": sent}))
	refused := s.readFrame(connection)

	s.Equal("error", refused["type"])
	s.Equal(created.Error.Message, refused["error"])
	s.Equal(created.Error.Code, refused["code"])
	s.Equal(string(created.Error.Type), refused["error_type"])
	_, _, err := connection.ReadMessage()
	s.True(websocket.IsCloseError(err, websocket.ClosePolicyViolation), "the socket closes: %v", err)
}
