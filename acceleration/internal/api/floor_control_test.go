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

// chorusASRFloor is chorus-asr's floor control as a client sends it, waiting at most
// maxWaitMs, or the server's default point when it is 0.
func chorusASRFloor(maxWaitMs int) *FloorControl {
	sent := &FloorControl{Model: new("stream/chorus-asr")}
	if maxWaitMs != 0 {
		sent.MaxWaitMs = new(maxWaitMs)
	}
	return sent
}

// A sync stores the floor control as sent, max_wait_ms included, the config serves it, and a
// session opened from the config runs by it: its Hearer ends turns by the point the wait
// chooses.
func (s *FloorControlSuite) TestASyncStoresTheFloorControlAndTheConfigServesIt() {
	synced := s.syncFloor("ruled", "v1", map[string]any{"floor_control": map[string]any{"model": "stream/chorus-asr", "max_wait_ms": 1200}})
	s.Empty(synced.Warnings)
	s.Equal(chorusASRFloor(1200), synced.Config.FloorControl)

	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+synced.Config.Id, nil, &read))
	s.Equal(chorusASRFloor(1200), read.FloorControl)

	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), synced.Config.Id)
	s.Require().NoError(err)
	s.Equal(store.FloorControl{Model: store.FloorControlChorusASR, MaxWaitMs: 1200}, stored.FloorControl)
	spec := specOf(CreateSessionRequest{}, s.customerID(), &stored)
	s.Equal(stored.FloorControl, spec.FloorControl)
	s.Equal(audiofloor.Act, session.FloorModeOf(spec.FloorControl))
	s.Equal(audiofloor.EndOfTurn{MaxWaitMs: 1200}, session.EndOfTurnOf(spec.FloorControl))
}

// A wait whose point fails the dictation guard is kept, and the sync says what the point was
// measured to do to callers reading out a number.
func (s *FloorControlSuite) TestASyncIsToldOfAPointThatFailsTheDictationGuard() {
	synced := s.syncFloor("eager", "v1", map[string]any{"floor_control": map[string]any{"model": "stream/chorus-asr", "max_wait_ms": 800}})
	s.Equal(chorusASRFloor(800), synced.Config.FloorControl)
	s.Equal([]string{"floor_control.max_wait_ms chooses the 500 ms point, which fails the dictation guard: it ended 9 of 9 slow counts before their last number, " +
		"and 16.0% of real pauses inside a dictated number, ID or spelled name; an agent whose callers read those out wants a point that passes"}, synced.Warnings)
}

// Left out, an agent has the default flow controller, on a router with a chorus-asr server:
// none is stored or served, and its sessions ask chorus-asr nothing. Once chosen, a sync that
// leaves it out or sends null keeps it, and an empty one is the default flow controller again.
func (s *FloorControlSuite) TestAnUnsetFloorControlIsTheDefaultFlowController() {
	unset := s.syncFloor("unset", "v1", nil)
	s.Nil(unset.Config.FloorControl)
	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), unset.Config.Id)
	s.Require().NoError(err)
	s.True(stored.FloorControl.IsZero())
	s.Equal(audiofloor.Off, session.FloorModeOf(specOf(CreateSessionRequest{}, s.customerID(), &stored).FloorControl))

	chosen := map[string]any{"model": "stream/chorus-asr", "max_wait_ms": 1400}
	s.Equal(chorusASRFloor(1400), s.syncFloor("support", "v1", map[string]any{"floor_control": chosen}).Config.FloorControl)
	s.Equal(chorusASRFloor(1400), s.syncFloor("support", "v2", nil).Config.FloorControl, "left out, the stored choice stays")
	s.Equal(chorusASRFloor(1400), s.syncFloor("support", "v3", map[string]any{"floor_control": nil}).Config.FloorControl, "null, the stored choice stays")
	s.Nil(s.syncFloor("support", "v4", map[string]any{"floor_control": map[string]any{}}).Config.FloorControl, "empty is the default flow controller again")
}

// A floor control the schema doesn't take is refused before anything is written: the string
// it once was, a wait with no model, a model nobody knows, a wait of nothing, and the
// end_of_turn it replaced, presets and bars included. So is a wait shorter than every point
// the server publishes, saying what the quickest is.
func (s *FloorControlSuite) TestAFloorControlNamingNothingKnownIsRefused() {
	for _, c := range []struct {
		fields map[string]any
		code   string
	}{
		{map[string]any{"floor_control": "stream/chorus-asr"}, "validation_failed"},
		{map[string]any{"floor_control": ""}, "validation_failed"},
		{map[string]any{"floor_control": map[string]any{"max_wait_ms": 1100}}, "invalid_request"},
		{map[string]any{"floor_control": map[string]any{"model": "stream/another"}}, "validation_failed"},
		{map[string]any{"floor_control": map[string]any{"model": "stream/chorus-asr", "max_wait_ms": 0}}, "validation_failed"},
		{map[string]any{"floor_control": map[string]any{"model": "stream/chorus-asr", "preset": "fast"}}, "validation_failed"},
		{map[string]any{"floor_control": map[string]any{"model": "stream/chorus-asr", "mode": "act"}}, "validation_failed"},
		{map[string]any{"floor_control": true}, "validation_failed"},
		{map[string]any{"floor_control": map[string]any{"model": "stream/chorus-asr"}, "end_of_turn": map[string]any{"max_wait_ms": 1100}}, "invalid_request"},
		{map[string]any{"end_of_turn": map[string]any{"preset": "fast"}}, "invalid_request"},
		{map[string]any{"end_of_turn": map[string]any{"bars": []map[string]any{{"from_ms": 0, "bar": 0.9}}}}, "invalid_request"},
		{map[string]any{"floor_control": map[string]any{"model": "stream/chorus-asr", "max_wait_ms": 799}}, "invalid_request"},
	} {
		fields := map[string]any{"name": "refused-" + s.utils.uuid(), "hash": "v1"}
		for field, value := range c.fields {
			fields[field] = value
		}
		status, code, message := s.refusedWith(s.serverClient, http.MethodPost, "/v1/agents/sync", fields)
		s.Equal(http.StatusBadRequest, status, "%v", c.fields)
		s.Equal(c.code, code, "%v: %s", c.fields, message)
		if _, sent := c.fields["end_of_turn"]; sent {
			s.Equal(errEndOfTurn.Message, message)
		}
	}
	_, message := s.serverClient.failure(http.MethodPost, "/v1/agents/sync", map[string]any{"name": "refused", "hash": "v1",
		"floor_control": map[string]any{"model": "stream/chorus-asr", "max_wait_ms": 799}})
	s.Equal("floor_control.max_wait_ms is 799, under every wait the model was measured at: the quickest has a caller wait 800 ms at p90", message)
	configs, err := s.store.CustomerAgentConfigs(context.Background(), s.customerID())
	s.Require().NoError(err)
	s.Empty(configs, "nothing is written for a refused sync")
}

// A config created, replaced or patched takes floor control as a sync does: sent, it replaces
// the stored one whole, left out or null it stays, and an empty one is the default flow
// controller again. A wait shorter than every point the server publishes is refused on each,
// and so is end_of_turn.
func (s *FloorControlSuite) TestEveryWriteOfAConfigTakesTheFloorControl() {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		AgentConfigRequest{Name: "created", FloorControl: chorusASRFloor(1400)}, &created))
	s.Equal(chorusASRFloor(1400), created.FloorControl)

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		AgentConfigPatch{Instructions: pointerTo("Sell ferry tickets.")}, &patched))
	s.Equal(chorusASRFloor(1400), patched.FloorControl, "a patch that leaves it out keeps it")
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"floor_control": nil}, &patched))
	s.Equal(chorusASRFloor(1400), patched.FloorControl, "a patch that sends null keeps it")

	var replaced AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		AgentConfigRequest{Name: "created"}, &replaced))
	s.Equal(chorusASRFloor(1400), replaced.FloorControl, "left out on an update, the stored choice stays")

	var model AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		AgentConfigPatch{FloorControl: chorusASRFloor(0)}, &model))
	s.Equal(chorusASRFloor(0), model.FloorControl, "sent, it replaces the stored one whole, its wait included")

	var cleared AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		AgentConfigPatch{FloorControl: &FloorControl{}}, &cleared))
	s.Nil(cleared.FloorControl, "an empty one is the default flow controller again")
	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), created.Id)
	s.Require().NoError(err)
	s.True(stored.FloorControl.IsZero())

	for _, write := range []struct {
		method, path string
		body         any
	}{
		{http.MethodPost, "/v1/agents/configs", AgentConfigRequest{Name: "refused", FloorControl: chorusASRFloor(100)}},
		{http.MethodPut, "/v1/agents/configs/" + created.Id, AgentConfigRequest{Name: "created", FloorControl: chorusASRFloor(100)}},
		{http.MethodPatch, "/v1/agents/configs/" + created.Id, AgentConfigPatch{FloorControl: chorusASRFloor(100)}},
		{http.MethodPost, "/v1/agents/configs", map[string]any{"name": "refused", "end_of_turn": map[string]any{"max_wait_ms": 1100}}},
		{http.MethodPut, "/v1/agents/configs/" + created.Id, map[string]any{"name": "created", "end_of_turn": map[string]any{}}},
		{http.MethodPatch, "/v1/agents/configs/" + created.Id, map[string]any{"end_of_turn": map[string]any{"preset": "fast"}}},
		{http.MethodPatch, "/v1/agents/configs/" + created.Id, map[string]any{"floor_control": "stream/chorus-asr"}},
	} {
		status, _ := s.serverClient.failure(write.method, write.path, write.body)
		s.Equal(http.StatusBadRequest, status, "%s %s %v", write.method, write.path, write.body)
	}
	stored, err = s.store.AgentConfig(context.Background(), s.customerID(), created.Id)
	s.Require().NoError(err)
	s.True(stored.FloorControl.IsZero(), "nothing refused is written")
}

// A session chooses for itself over its config, server-side only, held to the same rules, and
// keeps what it was opened with, which carrying it on from its row keeps.
func (s *FloorControlSuite) TestASessionChoosesItsFloorControlServerSide() {
	agent := store.AgentConfig{CustomerID: s.customerID(), Name: "session-" + s.utils.uuid(), Mode: store.AgentModeText,
		LLM: "en-low-latency", FloorControl: store.FloorControl{Model: store.FloorControlChorusASR, MaxWaitMs: 1400}}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &agent))

	request := CreateSessionRequest{ConfigId: &agent.ID, FloorControl: chorusASRFloor(1100)}
	for _, device := range []*testClient{s.client, s.guestClient, s.anonymousClient} {
		status, failure := device.failure(http.MethodPost, "/v1/agents/sessions", request)
		s.Equal(http.StatusForbidden, status, "%s", device.kind)
		s.Equal(errDeviceFloorControl.Message, failure, "%s", device.kind)
	}
	for _, refused := range []any{
		CreateSessionRequest{ConfigId: &agent.ID, FloorControl: chorusASRFloor(100)},
		map[string]any{"config_id": agent.ID, "end_of_turn": map[string]any{"max_wait_ms": 1100}},
		map[string]any{"config_id": agent.ID, "floor_control": "stream/chorus-asr"},
	} {
		status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions", refused)
		s.Equal(http.StatusBadRequest, status, "%v", refused)
	}

	opened := s.serverClient.createSession(request)
	unchosen := s.serverClient.createSession(CreateSessionRequest{ConfigId: &agent.ID})
	s.Require().Eventually(func() bool {
		row, err := s.store.StoredSession(context.Background(), s.customerID(), opened.Id)
		return err == nil && row.FloorControl.MaxWaitMs == 1100
	}, settleFor, 20*time.Millisecond, "the session keeps the choice it was opened with")
	s.Require().Eventually(func() bool {
		row, err := s.store.StoredSession(context.Background(), s.customerID(), unchosen.Id)
		return err == nil && row.FloorControl == agent.FloorControl
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
	s.serverClient.stopSession(opened.Id)
	s.serverClient.stopSession(unchosen.Id)

	_, other := s.otherRouter(nil)
	ctx := context.WithValue(context.Background(), customerContextKey{}, s.customerID())
	ctx = context.WithValue(ctx, kindContextKey{}, auth.KindServer)
	reopened, err := other.reopenedFromRow(ctx, opened.Id)
	s.Require().NoError(err)
	s.Equal(store.FloorControl{Model: store.FloorControlChorusASR, MaxWaitMs: 1100}, reopened.FloorControl, "a reopened session keeps its own choice")
	reopened, err = other.reopenedFromRow(ctx, unchosen.Id)
	s.Require().NoError(err)
	s.Equal(agent.FloorControl, reopened.FloorControl, "one that chose none takes its config's")
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

// stream/chorus-asr asks a router with no chorus-asr server for what it cannot do: every write
// that names it is refused, as a feature the deployment lacks, saying what the router needs,
// and nothing is written. An empty floor control asks nothing of it.
func (s *FloorControlWithoutChorusASRSuite) TestAFloorControlThatOptsInIsRefusedWithoutAServer() {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		AgentConfigRequest{Name: "default", FloorControl: &FloorControl{}}, &created))
	s.Nil(created.FloorControl)
	s.Nil(s.syncFloor("unset", "v1", nil).Config.FloorControl)

	chorusASR := chorusASRFloor(1100)
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
	sent := map[string]any{"floor_control": map[string]any{"model": "stream/chorus-asr", "max_wait_ms": 1100}}
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

// A socket's start frame is held to what the schema holds POST /v1/agents/sessions to: a wait
// with no model and end_of_turn are refused, and the string floor_control once was is no start
// frame at all.
func (s *SocketSessionSuite) TestASocketSessionIsRefusedTheFloorControlTheSchemaRefuses() {
	for _, c := range []struct {
		session map[string]any
		says    string
	}{
		{map[string]any{"floor_control": map[string]any{"max_wait_ms": 1100}}, "floor_control.model is required"},
		{map[string]any{"end_of_turn": map[string]any{"max_wait_ms": 1100}}, errEndOfTurn.Message},
		{map[string]any{"floor_control": "stream/chorus-asr"}, "the first frame must be a start frame naming the session"},
	} {
		connection := s.serverClient.opens("/v1/agents/socket")
		s.Require().NoError(connection.WriteJSON(map[string]any{"type": "start", "sample_rate": 24_000, "session": c.session}))
		refused := s.readFrame(connection)
		s.Equal("error", refused["type"], "%v", c.session)
		s.Contains(refused["error"], c.says, "%v", c.session)
		_ = connection.Close()
	}
}
