//go:build integration

package api

import (
	"encoding/json"
	"maps"
	"net/http"
	"slices"
	"strings"
)

// leftOut is a connector a client left out of a sync, as it sends it.
func leftOut(alias, connection string) map[string]any {
	return map[string]any{"name": alias, "connector_id": "linear", "connection": connection}
}

func (s *SyncSuite) TestASyncStoresTheConnectorsItLeftOut() {
	synced := s.sync(map[string]any{"name": "support", "hash": "v1",
		"connectors":          []map[string]any{sessionSlack("inbox")},
		"left_out_connectors": []map[string]any{leftOut("issues", "company")}})

	want := []LeftOutConnector{{Name: "issues", ConnectorId: "linear", Connection: "company"}}
	s.Equal(want, value(synced.Config.LeftOutConnectors))
	var got AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+synced.Config.Id, nil, &got))
	s.Equal(want, value(got.LeftOutConnectors))
	s.Equal(want, value(s.configsNamed("support")[0].LeftOutConnectors))
}

func (s *SyncSuite) TestASyncWithConnectorsReplacesTheOnesLeftOut() {
	s.sync(map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{},
		"left_out_connectors": []map[string]any{leftOut("issues", "company")}})

	second := s.sync(map[string]any{"name": "support", "hash": "v2", "connectors": []map[string]any{},
		"left_out_connectors": []map[string]any{leftOut("tracker", "work")}})

	want := []LeftOutConnector{{Name: "tracker", ConnectorId: "linear", Connection: "work"}}
	s.Equal(want, value(second.Config.LeftOutConnectors))
	s.Equal(want, value(s.configsNamed("support")[0].LeftOutConnectors))
}

// The connection turned up, so the client bound it and left nothing out.
func (s *SyncSuite) TestASyncWithConnectorsAndNoneLeftOutClearsThem() {
	s.sync(map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{},
		"left_out_connectors": []map[string]any{leftOut("issues", "company")}})

	second := s.sync(map[string]any{"name": "support", "hash": "v2",
		"connectors": []map[string]any{sessionSlack("inbox")}})

	s.Nil(second.Config.LeftOutConnectors)
	s.Nil(s.configsNamed("support")[0].LeftOutConnectors)
}

// Without connectors there is nothing the list could be the left-out part of, so it is
// refused rather than dropped, and nothing is stored.
func (s *SyncSuite) TestASyncLeavingOutConnectorsWithoutConnectorsIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "support", "hash": "v1", "instructions": "Be brief.",
			"left_out_connectors": []map[string]any{leftOut("issues", "company")}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "left_out_connectors goes with connectors")
	s.Empty(s.configsNamed("support"))
}

// An empty list is nothing to refuse: without connectors it is ignored and the stored list stays.
func (s *SyncSuite) TestASyncSendingAnEmptyListWithoutConnectorsLeavesTheOnesLeftOut() {
	s.sync(map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{},
		"left_out_connectors": []map[string]any{leftOut("issues", "company")}})

	second := s.sync(map[string]any{"name": "support", "hash": "v2", "instructions": "Be brief.",
		"left_out_connectors": []map[string]any{}})

	want := []LeftOutConnector{{Name: "issues", ConnectorId: "linear", Connection: "company"}}
	s.Equal(want, value(second.Config.LeftOutConnectors))
	s.Equal(want, value(s.configsNamed("support")[0].LeftOutConnectors))
}

// A sync that says nothing about the bindings leaves them and the list left out as they are.
func (s *SyncSuite) TestASyncSendingNeitherLeavesTheOnesLeftOut() {
	s.sync(map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{},
		"left_out_connectors": []map[string]any{leftOut("issues", "company")}})

	second := s.sync(map[string]any{"name": "support", "hash": "v2", "instructions": "Be brief."})

	want := []LeftOutConnector{{Name: "issues", ConnectorId: "linear", Connection: "company"}}
	s.Equal(want, value(second.Config.LeftOutConnectors))
	s.Equal(want, value(s.configsNamed("support")[0].LeftOutConnectors))
}

// A save from the dashboard does not send them, and must not wipe what the sync stored; one
// that does send them is not heard either, since only a sync writes them.
func (s *SyncSuite) TestSavingAConfigKeepsTheConnectorsASyncLeftOut() {
	synced := s.sync(map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{},
		"left_out_connectors": []map[string]any{leftOut("issues", "company")}})

	var saved AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+synced.Config.Id,
		map[string]any{"name": "support", "instructions": "Be brief.",
			"left_out_connectors": []map[string]any{leftOut("tracker", "work")}}, &saved))

	want := []LeftOutConnector{{Name: "issues", ConnectorId: "linear", Connection: "company"}}
	s.Equal(want, value(saved.LeftOutConnectors))
	s.Equal(want, value(s.configsNamed("support")[0].LeftOutConnectors))
}

func (s *SyncSuite) TestPatchingAConfigKeepsTheConnectorsASyncLeftOut() {
	synced := s.sync(map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{},
		"left_out_connectors": []map[string]any{leftOut("issues", "company")}})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+synced.Config.Id,
		AgentConfigPatch{Greeting: &Greeting{Text: "Hello there"}}, &patched))

	want := []LeftOutConnector{{Name: "issues", ConnectorId: "linear", Connection: "company"}}
	s.Equal(want, value(patched.LeftOutConnectors))
	s.Equal(want, value(s.configsNamed("support")[0].LeftOutConnectors))
}

// Probed on 7f4c3e16, before the field existed: a patch naming it is refused the same way.
func (s *SyncSuite) TestPatchingTheConnectorsASyncLeftOutIsRefused() {
	synced := s.sync(map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{}})

	status, body := s.serverClient.call(http.MethodPatch, "/v1/agents/configs/"+synced.Config.Id,
		map[string]any{"left_out_connectors": []map[string]any{leftOut("issues", "company")}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "unexpected property (body.left_out_connectors)")
	s.Nil(s.configsNamed("support")[0].LeftOutConnectors)
}

func (s *SyncSuite) TestASyncLeavingOutAConnectorWithNoConnectionIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{},
			"left_out_connectors": []map[string]any{leftOut("issues", "")}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "left_out_connectors")
	s.Empty(s.configsNamed("support"))
}

func (s *SyncSuite) TestASyncLeavingOutMoreThan64ConnectorsIsRefused() {
	many := make([]map[string]any, 0, 65)
	for range 65 {
		many = append(many, leftOut("issues", "company"))
	}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{}, "left_out_connectors": many})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "left_out_connectors")
	s.Empty(s.configsNamed("support"))
}

// The keys a sync and a GET answered on 7f4c3e16, before left-out connectors existed, for a
// sync that left nothing out (probe in the PR). They answer the same now.
func (s *SyncSuite) TestASyncLeavingNothingOutAnswersAsBefore() {
	status, body := s.serverClient.call(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "support", "hash": "v1", "connectors": []map[string]any{sessionSlack("inbox")}})
	s.Require().Equal(http.StatusOK, status)
	var synced struct {
		Config map[string]json.RawMessage `json:"config"`
	}
	s.Require().NoError(json.Unmarshal(body, &synced))
	s.Equal("connectors created_at dispatch episode_cards harness id mode name region sync_hash tools updated_at video",
		strings.Join(slices.Sorted(maps.Keys(synced.Config)), " "))

	status, body = s.serverClient.call(http.MethodGet, "/v1/agents/configs/"+strings.Trim(string(synced.Config["id"]), `"`), nil)
	s.Require().Equal(http.StatusOK, status)
	var got map[string]json.RawMessage
	s.Require().NoError(json.Unmarshal(body, &got))
	s.Equal("connectors created_at dispatch duration episode_cards harness id mode name region sync_hash tools updated_at video",
		strings.Join(slices.Sorted(maps.Keys(got)), " "))
}
