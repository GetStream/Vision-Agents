//go:build integration

package api

import (
	"context"
	"net/http"
	"net/url"
	"slices"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// meanwhile runs then while first writes the config: a transaction holds the config's row,
// first queues for it, then then, so then reads the config before first writes it and writes
// after. It returns both statuses.
func (s *PluginsSuite) meanwhile(configID string, first, then func() int) (int, int) {
	ctx := s.T().Context()
	tx, err := s.store.DB().BeginTx(ctx, nil)
	s.Require().NoError(err)
	// A failed wait below still lets both finish; after the commit this does nothing.
	defer func() { _ = tx.Rollback() }()
	_, err = tx.ExecContext(ctx, "SELECT 1 FROM agent_configs WHERE id = ? FOR UPDATE", configID)
	s.Require().NoError(err)

	firsts, thens := make(chan int, 1), make(chan int, 1)
	go func() { firsts <- first() }()
	s.waitForLockWaiters(1, "the first write waits on the row")
	go func() { thens <- then() }()
	s.waitForLockWaiters(2, "the second write waits behind the first")
	s.Require().NoError(tx.Commit())
	return <-firsts, <-thens
}

// auditedFields are the fields the config's audit entries of action record as changed, sorted.
func (s *PluginsSuite) auditedFields(configID, action string) []string {
	entries, err := s.store.AuditEntries(s.T().Context(), s.customerID(),
		store.AuditEntryFilter{AgentID: configID, ResourceType: store.AuditAgentConfig, Limit: 100})
	s.Require().NoError(err)
	var fields []string
	for _, entry := range entries {
		if entry.Action != action {
			continue
		}
		for _, change := range entry.Changes {
			fields = append(fields, change.Field)
		}
	}
	slices.Sort(fields)
	return fields
}

// waitForLockWaiters waits until n statements of the suite's database wait on a lock.
func (s *PluginsSuite) waitForLockWaiters(n int, what string) {
	s.Require().Eventually(func() bool {
		var waiting int
		s.Require().NoError(s.store.DB().QueryRowContext(s.T().Context(),
			"SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'").Scan(&waiting))
		return waiting >= n
	}, 5*time.Second, 10*time.Millisecond, what)
}

// appLoginOf is the app's login of plugin on the config, finished at the callback.
func (s *PluginsSuite) appLoginOf(configID, plugin string) func() int {
	exchanged := false
	state := s.pendingLogin(configID, plugin, "", s.tokenServer(&exchanged).URL)
	return func() int {
		status, _ := s.callback(state)
		return status
	}
}

// patching is a PATCH of the config with body.
func (s *PluginsSuite) patching(configID string, body map[string]any) func() int {
	return func() int { return s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+configID, body, nil) }
}

// The base, bb0ccea76, answered 200 and dropped linear.
func (s *PluginsSuite) TestAPatchOfAnotherFieldKeepsAPluginALoginNamedMeanwhile() {
	agent := s.data.createAgentConfig()

	logged, patched := s.meanwhile(agent.Id, s.appLoginOf(agent.Id, "linear"),
		s.patching(agent.Id, map[string]any{"instructions": "be kind"}))

	s.Equal(http.StatusFound, logged)
	s.Equal(http.StatusOK, patched)
	config, err := s.store.AgentConfig(s.T().Context(), s.customerID(), agent.Id)
	s.Require().NoError(err)
	s.Equal("be kind", config.Instructions)
	s.Equal([]store.PluginEntry{{Name: "linear"}}, config.Plugins)
	s.Equal([]string{"instructions"}, s.auditedFields(agent.Id, store.AuditUpdated),
		"the login's plugin is no change of the patch")
}

// A PATCH that binds nessy passed its own check on the config it read, without the plugin a
// login named since, so it is checked again on the plugins it writes and refused. The base,
// bb0ccea76, answered 200, bound nessy and dropped linear.
func (s *PluginsSuite) TestAPatchBindingNessyIsRefusedBesideAPluginALoginNamedMeanwhile() {
	s.useApp(s.data.createAppAdmitting(store.AppSettings{InternalConnectors: new(true)}))
	agent := s.data.createAgentConfig()
	var failure string

	logged, patched := s.meanwhile(agent.Id, s.appLoginOf(agent.Id, "linear"), func() int {
		var status int
		status, failure = s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+agent.Id,
			map[string]any{"connectors": []map[string]any{sessionNessy("customers")}})
		return status
	})

	s.Equal(http.StatusFound, logged)
	s.Equal(http.StatusBadRequest, patched)
	s.Contains(failure, `and plugin "linear" can send data out`)
	config, err := s.store.AgentConfig(s.T().Context(), s.customerID(), agent.Id)
	s.Require().NoError(err)
	s.Empty(config.Connectors)
	s.Equal([]store.PluginEntry{{Name: "linear"}}, config.Plugins)
}

// A sync whose agent.yaml leaves plugins out leaves them alone (its operation's doc), so it
// keeps one a login named meanwhile. The base, bb0ccea76, answered 200 and dropped linear.
func (s *PluginsSuite) TestASyncWithoutPluginsKeepsAPluginALoginNamedMeanwhile() {
	name := "assistant-" + s.utils.uuid()
	var synced SyncAgentResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": name, "hash": "v1", "mode": "text", "instructions": "be brief"}, &synced))

	logged, resynced := s.meanwhile(synced.Config.Id, s.appLoginOf(synced.Config.Id, "linear"), func() int {
		return s.serverClient.do(http.MethodPost, "/v1/agents/sync",
			map[string]any{"name": name, "hash": "v2", "mode": "text", "instructions": "be kind"}, nil)
	})

	s.Equal(http.StatusFound, logged)
	s.Equal(http.StatusOK, resynced)
	config, err := s.store.AgentConfig(s.T().Context(), s.customerID(), synced.Config.Id)
	s.Require().NoError(err)
	s.Equal("be kind", config.Instructions)
	s.Equal([]store.PluginEntry{{Name: "linear"}}, config.Plugins)
	s.NotContains(s.auditedFields(synced.Config.Id, store.AuditSynced), "plugins",
		"the login's plugin is no change of the sync")
}

// A client set with user names its plugin beside the ones stored then. The base, bb0ccea76,
// answered 200 and dropped linear.
func (s *PluginsSuite) TestAUserPluginsClientKeepsAPluginALoginNamedMeanwhile() {
	agent := s.data.createAgentConfig()

	logged, set := s.meanwhile(agent.Id, s.appLoginOf(agent.Id, "linear"), func() int {
		return s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+agent.Id+"/plugins/google_calendar/client",
			SetPluginClientRequest{ClientId: "acme-client", User: pointerTo(true)}, nil)
	})

	s.Equal(http.StatusFound, logged)
	s.Equal(http.StatusOK, set)
	s.Equal([]string{"linear", "google_calendar"}, s.storedPlugins(agent.Id))
}

// A client set with user for a plugin a login named since the config was read leaves it named
// once, as the write finds it named and adds nothing.
func (s *PluginsSuite) TestAUserPluginsClientForAPluginALoginNamedMeanwhileLeavesItNamedOnce() {
	agent := s.data.createAgentConfig()

	logged, set := s.meanwhile(agent.Id, s.appLoginOf(agent.Id, "google_calendar"), func() int {
		return s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+agent.Id+"/plugins/google_calendar/client",
			SetPluginClientRequest{ClientId: "acme-client", User: pointerTo(true)}, nil)
	})

	s.Equal(http.StatusFound, logged)
	s.Equal(http.StatusOK, set)
	s.Equal([]string{"google_calendar"}, s.storedPlugins(agent.Id))
}

// A client set with user names a plugin that may send data out, so beside nessy, bound since
// the config was read, it is refused on the config it writes. The base, bb0ccea76, answered
// 200, named google_calendar and dropped nessy.
func (s *PluginsSuite) TestAUserPluginsClientIsRefusedBesideNessyBoundMeanwhile() {
	s.useApp(s.data.createAppAdmitting(store.AppSettings{InternalConnectors: new(true)}))
	agent := s.data.createAgentConfig()
	var failure string

	patched, set := s.meanwhile(agent.Id,
		s.patching(agent.Id, map[string]any{"connectors": []map[string]any{sessionNessy("customers")}}), func() int {
			var status int
			status, failure = s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+agent.Id+"/plugins/google_calendar/client",
				SetPluginClientRequest{ClientId: "acme-client", User: pointerTo(true)})
			return status
		})

	s.Equal(http.StatusOK, patched)
	s.Equal(http.StatusBadRequest, set)
	s.Contains(failure, `and plugin "google_calendar" can send data out`)
	config, err := s.store.AgentConfig(s.T().Context(), s.customerID(), agent.Id)
	s.Require().NoError(err)
	s.Require().Len(config.Connectors, 1)
	s.Equal("nessy", config.Connectors[0].ConnectorID)
	s.Empty(config.Plugins)
}

// Disconnecting one plugin takes out that one, not one a login named meanwhile. The base,
// bb0ccea76, answered 204 and dropped linear.
func (s *PluginsSuite) TestADisconnectKeepsAPluginALoginNamedMeanwhile() {
	agent := s.data.createAgentConfig()
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+agent.Id+"/plugins/google_calendar/client",
		SetPluginClientRequest{ClientId: "acme-client", User: pointerTo(true)}, nil))

	logged, disconnected := s.meanwhile(agent.Id, s.appLoginOf(agent.Id, "linear"), func() int {
		status, _ := s.serverClient.call(http.MethodDelete, "/v1/agents/configs/"+agent.Id+"/plugins/google_calendar", nil)
		return status
	})

	s.Equal(http.StatusFound, logged)
	s.Equal(http.StatusNoContent, disconnected)
	s.Equal([]string{"linear"}, s.storedPlugins(agent.Id))
}

// The app's login checks the config it writes on the connection its write holds: with one
// connection in the pool, it is finished. The base, bb0ccea76, waited for a second
// connection until the browser gave up.
func (s *PluginsSuite) TestAnAppLoginIsCheckedOnTheConnectionItsWriteHolds() {
	agent := s.data.createAgentConfig()
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+agent.Id,
		map[string]any{"connectors": []map[string]any{sessionGitHub("code", false)}}, nil))
	exchanged := false
	state := s.pendingLogin(agent.Id, "linear", "", s.tokenServer(&exchanged).URL)
	s.store.DB().SetMaxOpenConns(1)
	defer s.store.DB().SetMaxOpenConns(suiteConnections)
	// Ten seconds is far more than a login on loopback takes, and ends a wait that never would.
	ctx, cancel := context.WithTimeout(s.T().Context(), 10*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet,
		s.server.URL+plugins.CallbackPath+"?state="+url.QueryEscape(state)+"&code=the-code", nil)
	s.Require().NoError(err)
	browser := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}

	response, err := browser.Do(request)

	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Equal(http.StatusFound, response.StatusCode)
	s.store.DB().SetMaxOpenConns(suiteConnections)
	s.Equal([]string{"linear"}, s.storedPlugins(agent.Id))
}

// TestAWriteWithNothingMeanwhileStoresThePluginsAsBefore is the control: with no write in
// between, a PATCH of another field keeps the plugins, a PATCH or a PUT naming plugins writes
// them, and a PUT without plugins writes none. The base, bb0ccea76, stored the same four
// (logs/control-on-base.log in the PR's ledger).
func (s *PluginsSuite) TestAWriteWithNothingMeanwhileStoresThePluginsAsBefore() {
	var agent AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "agent-" + s.utils.uuid(), "llm": "llm-flow", "plugins": []string{"linear"},
	}, &agent))
	path := "/v1/agents/configs/" + agent.Id
	var seen [][]string

	for _, write := range []struct {
		method string
		body   map[string]any
	}{
		{http.MethodPatch, map[string]any{"instructions": "be kind"}},
		{http.MethodPatch, map[string]any{"plugins": []string{"linear", "google_calendar"}}},
		{http.MethodPut, map[string]any{"name": agent.Name, "llm": "llm-flow", "plugins": []string{"google_calendar"}}},
		{http.MethodPut, map[string]any{"name": agent.Name, "llm": "llm-flow"}},
	} {
		var written AgentConfig
		s.Require().Equal(http.StatusOK, s.serverClient.do(write.method, path, write.body, &written), write.body)
		var answered []string
		if written.Plugins != nil {
			for _, entry := range *written.Plugins {
				answered = append(answered, entry.Name)
			}
		}
		s.Equal(s.storedPlugins(agent.Id), answered, "the answer is what is stored")
		seen = append(seen, s.storedPlugins(agent.Id))
	}

	s.Equal([][]string{{"linear"}, {"linear", "google_calendar"}, {"google_calendar"}, nil}, seen)
}
