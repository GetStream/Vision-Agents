//go:build integration

package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// AI-1049: a channel connection belongs to one live agent config. A test copy of the agent
// (the dashboard's draft_of tag) leaves it answering; a second live config binding it as
// fixed is refused with a 409 naming the first.

// TestATestCopyOfTheAgentLeavesItAnswering pins AI-1049: on base the copy's binding made two
// configs, and every message was dropped.
func (s *SlackChannelSuite) TestATestCopyOfTheAgentLeavesItAnswering() {
	copied := s.saved(http.MethodPost, "/v1/agents/configs", s.botAgent("copy-"+s.utils.uuid(), map[string]string{store.DraftOfTag: s.config.ID}))
	s.Require().NotEqual(s.config.ID, copied)

	status, _ := s.deliver(s.message("U0000ALICE", "Can you check the build?", "1759740000.004900", ""), 0)

	s.Equal(http.StatusOK, status)
	channel := s.threadChannel("C0000CHAN:1759740000.004900")
	s.written(channel, 1)
	data, found := s.chat.Channel(channel)
	s.Require().True(found)
	custom, _ := data["custom"].(map[string]any)
	s.Equal(s.config.ID, custom[ConfigField], "the live config answers, not its test copy")
}

// TestASecondAgentBindingTheBotIsRefusedNamingTheFirst: create, update, patch, sync and
// save-as-new are each a 409 that names the config that has the connection.
func (s *SlackChannelSuite) TestASecondAgentBindingTheBotIsRefusedNamingTheFirst() {
	unbound := s.saved(http.MethodPost, "/v1/agents/configs", map[string]any{"name": "other-" + s.utils.uuid(), "mode": "text"})
	for name, write := range map[string]func() (int, ErrorDetail){
		"create": func() (int, ErrorDetail) {
			return s.refusal(http.MethodPost, "/v1/agents/configs", s.botAgent("second-"+s.utils.uuid(), nil))
		},
		"update": func() (int, ErrorDetail) {
			return s.refusal(http.MethodPut, "/v1/agents/configs/"+unbound, s.botAgent("other-"+s.utils.uuid(), nil))
		},
		"patch": func() (int, ErrorDetail) {
			return s.refusal(http.MethodPatch, "/v1/agents/configs/"+unbound, map[string]any{"connectors": s.botBindings()})
		},
		"sync": func() (int, ErrorDetail) {
			return s.refusal(http.MethodPost, "/v1/agents/sync", map[string]any{
				"name": "synced-" + s.utils.uuid(), "hash": s.utils.uuid(), "connectors": s.botBindings(),
			})
		},
	} {
		status, failure := write()
		s.Equal(http.StatusConflict, status, name)
		s.Equal(ErrorTypeConflict, failure.Type, name)
		s.Equal(codeChannelConnectionTaken, failure.Code, name)
		s.Contains(failure.Message, `agent config "`+s.config.Name+`" (`+s.config.ID+`)`, name)
		s.Contains(failure.Message, s.bot.ConnectionID, name)
		s.Contains(failure.Message, `Remove the binding from "`+s.config.Name+`"`, name)
	}
	var stored AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+unbound, nil, &stored))
	s.Empty(value(stored.Connectors), "the refused update and patch wrote nothing")
}

// TestPromotingATestCopyIsRefused: a copy that drops its draft_of tag becomes a live config,
// a second one on the connection.
func (s *SlackChannelSuite) TestPromotingATestCopyIsRefused() {
	copied := s.saved(http.MethodPost, "/v1/agents/configs", s.botAgent("copy-"+s.utils.uuid(), map[string]string{store.DraftOfTag: s.config.ID}))

	status, failure := s.refusal(http.MethodPut, "/v1/agents/configs/"+copied, s.botAgent("copy-"+s.utils.uuid(), nil))

	s.Equal(http.StatusConflict, status)
	s.Equal(codeChannelConnectionTaken, failure.Code)
}

// TestTheAgentThatAnswersIsSavedAsBefore is the control: the one config that binds the
// connection saves with its binding, as on base (200; probe in the PR's author log).
func (s *SlackChannelSuite) TestTheAgentThatAnswersIsSavedAsBefore() {
	s.Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+s.config.ID, s.botAgent(s.config.Name, nil), nil))
	s.Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+s.config.ID,
		map[string]any{"instructions": "Be brief."}, nil))
}

// TestASessionBindingOfTheBotIsNotRefused: only a fixed binding answers the connection's
// messages.
func (s *SlackChannelSuite) TestASessionBindingOfTheBotIsNotRefused() {
	s.saved(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "session-" + s.utils.uuid(), "mode": "text",
		"connectors": []map[string]any{{"name": "slack", "connector_id": "slack_bot",
			"connection": map[string]any{"type": "session"}, "tools": []map[string]any{}}},
	})
}

// TestAConnectionTwoAgentsAlreadySharedIsAnsweredByTheOldest: configs that both bound the
// connection before the 409 existed are not silenced: the oldest answers, the router warns
// naming the other, and the other can still be saved with its binding.
func (s *SlackChannelSuite) TestAConnectionTwoAgentsAlreadySharedIsAnsweredByTheOldest() {
	newer := s.sharedAsBefore()

	status, _ := s.deliver(s.message("U0000ALICE", "Can you check the build?", "1759740000.005000", ""), 0)

	s.Equal(http.StatusOK, status)
	channel := s.threadChannel("C0000CHAN:1759740000.005000")
	s.written(channel, 1)
	data, found := s.chat.Channel(channel)
	s.Require().True(found)
	custom, _ := data["custom"].(map[string]any)
	s.Equal(s.config.ID, custom[ConfigField], "the oldest config answers")
	s.Contains(s.logged.String(), `level=WARN msg="more than one agent config binds the connection a message came in on: the oldest answers"`)
	s.Contains(s.logged.String(), "config="+s.config.ID+" not_answering=["+newer+"]")
	s.Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+newer,
		map[string]any{"instructions": "Be brief."}, nil), "a save that keeps the binding is not a new bind")
}

// TestTwoAgentsBindingTheBotAtOnceAreOneOwner: two routers saving at once take turns, so one
// save is stored and the other is the 409.
func (s *SlackChannelSuite) TestTwoAgentsBindingTheBotAtOnceAreOneOwner() {
	bot := s.connectedBot("T0000RACE"+strings.ToUpper(s.utils.uuid()[:8]), "xoxb-synthetic-race")
	const pairs = 8
	statuses := make([]int, 2*pairs)
	var started, done sync.WaitGroup
	started.Add(1)
	for i := range statuses {
		done.Add(1)
		go func() {
			defer done.Done()
			started.Wait()
			body := s.botAgent("race-"+s.utils.uuid(), nil)
			body["connectors"].([]map[string]any)[0]["connection"] = map[string]any{"type": "fixed", "connection_id": bot.ConnectionID}
			statuses[i], _ = s.serverClient.call(http.MethodPost, "/v1/agents/configs", body)
		}()
	}
	started.Done()
	done.Wait()

	created := 0
	for _, status := range statuses {
		s.Contains([]int{http.StatusCreated, http.StatusConflict}, status)
		if status == http.StatusCreated {
			created++
		}
	}
	s.Equal(1, created, "one live config binds the connection")
	configs, err := s.store.AgentConfigsBindingConnection(context.Background(), s.customerID(), bot.ConnectionID)
	s.Require().NoError(err)
	s.Len(configs, 1)
}

// TestAPluginMigrationCannotGiveTheBotASecondAgent: router plugins migrate adds bindings
// through AddConnectorBinding, which refuses the same way.
func (s *SlackChannelSuite) TestAPluginMigrationCannotGiveTheBotASecondAgent() {
	other := s.saved(http.MethodPost, "/v1/agents/configs", map[string]any{"name": "migrated-" + s.utils.uuid(), "mode": "text"})

	_, added, err := s.store.AddConnectorBinding(context.Background(), s.customerID(), other, store.ConnectorBinding{
		Name: "slack", ConnectorID: "slack_bot", Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: s.bot.ConnectionID},
	})

	taken, ok := errors.AsType[*store.ChannelConnectionTakenError](err)
	s.Require().True(ok, "%v", err)
	s.False(added)
	s.Equal(s.config.ID, taken.OwnerID)
	s.Equal(s.config.Name, taken.OwnerName)
}

// TestAReplyAfterAStopOnALineTwoAgentsSharedIsNotSent: on a Linq line two configs bound before
// AI-1049, the oldest answers, and who a chat's replies reach is read with it (replyTo), so a
// STOP keeps the reply away as it does with one config.
func (s *LinqChannelSuite) TestAReplyAfterAStopOnALineTwoAgentsSharedIsNotSent() {
	_, err := s.store.DB().ExecContext(context.Background(), "UPDATE agent_configs SET tags = ? WHERE id = ?",
		`{"`+store.DraftOfTag+`":"`+s.config.ID+`"}`, s.config.ID)
	s.Require().NoError(err)
	newer := store.AgentConfig{CustomerID: s.customerID(), Name: "linq-newer-" + s.utils.uuid(), Mode: store.AgentModeText,
		LLM: "noted/noted-model", Connectors: s.config.Connectors}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &newer))
	_, err = s.store.DB().ExecContext(context.Background(), "UPDATE agent_configs SET tags = '{}' WHERE id = ?", s.config.ID)
	s.Require().NoError(err)
	chat, person := s.utils.uuid(), "+12025550198"
	s.deliver(s.received(chat, s.line, person, "Hi"), time.Now())
	channel := s.threadChannel(chat)
	s.written(channel, 1)
	s.deliver(s.received(chat, s.line, person, "STOP"), time.Now())
	s.took(1)

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	s.Never(func() bool { return len(s.linq.sent()) > 1 }, dropped, 20*time.Millisecond)
	s.Equal(telnyxStopped, s.linq.sent()[0].text)
}

// botAgent is a text agent config body binding the bot connection as fixed, with tags.
func (s *SlackChannelSuite) botAgent(name string, tags map[string]string) map[string]any {
	body := map[string]any{"name": name, "mode": "text", "llm": "noted/noted-model", "connectors": s.botBindings()}
	if tags != nil {
		body["tags"] = tags
	}
	return body
}

// botBindings is the one fixed binding of the bot connection.
func (s *SlackChannelSuite) botBindings() []map[string]any {
	return []map[string]any{{"name": "slack", "connector_id": "slack_bot",
		"connection": map[string]any{"type": "fixed", "connection_id": s.bot.ConnectionID}, "tools": []map[string]any{}}}
}

// saved writes a config through the API, requires it stored, and returns its id.
func (s *SlackChannelSuite) saved(method, path string, body map[string]any) string {
	status, payload := s.serverClient.call(method, path, body)
	s.Require().Contains([]int{http.StatusOK, http.StatusCreated}, status, string(payload))
	var stored AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &stored))
	return stored.Id
}

// refusal is a write's status and the failure it answered with.
func (s *SlackChannelSuite) refusal(method, path string, body any) (int, ErrorDetail) {
	status, payload := s.serverClient.call(method, path, body)
	var answer ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answer), string(payload))
	return status, answer.Error
}

// sharedAsBefore is a second live config binding the bot connection, as a save before AI-1049
// could store one: made while the suite's config is tagged a test copy, straight in its row,
// which the API now refuses to do.
func (s *SlackChannelSuite) sharedAsBefore() string {
	s.tagged(s.config.ID, `{"`+store.DraftOfTag+`":"`+s.config.ID+`"}`)
	newer := s.saved(http.MethodPost, "/v1/agents/configs", s.botAgent("newer-"+s.utils.uuid(), nil))
	s.tagged(s.config.ID, `{}`)
	return newer
}

// tagged sets a config's tags in its row.
func (s *SlackChannelSuite) tagged(configID, tags string) {
	_, err := s.store.DB().ExecContext(context.Background(), "UPDATE agent_configs SET tags = ? WHERE id = ?", tags, configID)
	s.Require().NoError(err)
}
