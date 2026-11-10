//go:build integration

package api

import (
	"context"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A config answers in agent channels alone until it names a type, keeps the types it names
// through an update that leaves them out, and has them replaced whole by a patch.
func (s *ConfigsSuite) TestAConfigAnswersInNoOtherChannelTypeUnlessItNamesOne() {
	unnamed := s.createConfig(map[string]any{"name": "support"})
	s.Nil(unnamed.ChatChannels)
	s.storedChatChannels(unnamed.Id, nil)

	created := s.createConfig(map[string]any{"name": "rota", "chat_channels": []map[string]any{{"type": "callouts"}}})
	s.Equal(&[]AgentChatChannel{{Type: "callouts"}}, created.ChatChannels)
	s.storedChatChannels(created.Id, []store.ChatChannel{{Type: "callouts"}})

	var saved AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id, map[string]any{"name": "rota"}, &saved))
	s.Equal(&[]AgentChatChannel{{Type: "callouts"}}, saved.ChatChannels, "a list left out on an update stays")
	s.storedChatChannels(created.Id, []store.ChatChannel{{Type: "callouts"}})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"chat_channels": []map[string]any{{"type": "shifts"}, {"type": "team_2"}}}, &patched))
	s.Equal(&[]AgentChatChannel{{Type: "shifts"}, {Type: "team_2"}}, patched.ChatChannels)
	s.storedChatChannels(created.Id, []store.ChatChannel{{Type: "shifts"}, {Type: "team_2"}})

	var emptied AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "rota", "chat_channels": []map[string]any{}}, &emptied))
	s.Nil(emptied.ChatChannels, "an empty list removes them")
	s.storedChatChannels(created.Id, nil)
}

// chat_channels names each type once, never agent, as a cid's type is written, and at most 16,
// on create, update and patch.
func (s *ConfigsSuite) TestChatChannelsTakesOnlyTypesACIDCanName() {
	created := s.createConfig(map[string]any{"name": "rota", "chat_channels": []map[string]any{{"type": "callouts"}}})
	tooMany := make([]map[string]any, 17)
	for i := range tooMany {
		tooMany[i] = map[string]any{"type": "type-" + string(rune('a'+i))}
	}
	for _, sent := range []any{
		[]map[string]any{{"type": "agent"}},
		[]map[string]any{{"type": "callouts"}, {"type": "callouts"}},
		[]map[string]any{{"type": "call outs"}},
		[]map[string]any{{"type": "callouts:team"}},
		[]map[string]any{{"type": ""}},
		[]map[string]any{{}},
		[]string{"callouts"},
		tooMany,
	} {
		for _, write := range []struct{ method, path string }{
			{http.MethodPost, "/v1/agents/configs"},
			{http.MethodPut, "/v1/agents/configs/" + created.Id},
			{http.MethodPatch, "/v1/agents/configs/" + created.Id},
		} {
			status, body := s.serverClient.call(write.method, write.path, map[string]any{"name": "rota-" + s.utils.uuid(), "chat_channels": sent})
			s.Equal(http.StatusBadRequest, status, "%s %v: %s", write.method, sent, body)
		}
	}
	s.storedChatChannels(created.Id, []store.ChatChannel{{Type: "callouts"}})
}

// storedChatChannels checks the chat_channels a config holds in the store.
func (s *ConfigsSuite) storedChatChannels(id string, want []store.ChatChannel) {
	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), id)
	s.Require().NoError(err)
	s.Equal(want, stored.ChatChannels)
}
