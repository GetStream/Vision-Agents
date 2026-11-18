//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"
)

// SessionUISuite covers whether a session's model is told its agent's ui_instructions: after
// the instructions by default, and not at all in a session its caller opened with ui false,
// for as long as that conversation goes on.
type SessionUISuite struct {
	RouterSuite
}

func TestSessionUISuite(t *testing.T) {
	runSuite(t, new(SessionUISuite))
}

func (s *SessionUISuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SessionUISuite) TestASessionIsToldItsAgentsUIAfterItsInstructions() {
	config, instructions, ui := s.uiAgent()

	opened := s.client.createSession(s.echoing(config, nil))

	s.Contains(s.answerAs(s.client, opened.Id), instructions+"\n\n"+ui)
}

func (s *SessionUISuite) TestADeviceThatTurnsUIOffIsAnsweredWithoutIt() {
	config, instructions, ui := s.uiAgent()

	opened := s.client.createSession(s.echoing(config, pointerTo(false)))

	told := s.answerAs(s.client, opened.Id)
	s.Contains(told, instructions)
	s.NotContains(told, ui)
}

// A backend's instructions replace the config's, and the agent's UI still follows them.
func (s *SessionUISuite) TestABackendsOwnInstructionsAreFollowedByTheAgentsUI() {
	config, instructions, ui := s.uiAgent()
	request := s.echoing(config, nil)
	brief := "Be a customer whose order " + s.utils.uuid() + " never came."
	request.Instructions = &brief

	told := s.answerAs(s.serverClient, s.serverClient.createSession(request).Id)

	s.Contains(told, brief+"\n\n"+ui)
	s.NotContains(told, instructions)
}

// A stopped session is no longer held, so answering it again carries it on from its row.
func (s *SessionUISuite) TestAConversationCarriedOnFromItsRowKeepsUIOff() {
	config, instructions, ui := s.uiAgent()
	opened := s.client.createSession(s.echoing(config, pointerTo(false)))
	s.Require().Eventually(func() bool { return s.stored(opened.Id) }, settleFor, 20*time.Millisecond,
		"the row is written")
	s.client.stopSession(opened.Id)

	told := s.answerAs(s.client, opened.Id)
	s.Contains(told, instructions)
	s.NotContains(told, ui)
}

// uiAgent is the id of an agent config of the suite's app with instructions and UI
// instructions, returned with them. It answers on the model that echoes what it was told,
// which a conversation carried on from its row runs on too.
func (s *SessionUISuite) uiAgent() (string, string, string) {
	instructions := "Answer about " + s.utils.uuid() + "."
	ui := "## Interactive UI Generation\n\nDraw card " + s.utils.uuid() + " when it helps."
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Llm: pointerTo("echo/echo-model"), Instructions: &instructions, UiInstructions: &ui,
	}, &created))
	return created.Id, instructions, ui
}

// echoing opens a conversation in writing under config.
func (s *SessionUISuite) echoing(config string, ui *bool) CreateSessionRequest {
	return CreateSessionRequest{ConfigId: &config, Ui: ui}
}

// answerAs asks as's session a question and returns what it answered.
func (s *SessionUISuite) answerAs(as *testClient, id string) string {
	watching := as.opens("/v1/agents/sessions/" + id + "/events")
	s.Require().Equal(http.StatusAccepted, as.do(http.MethodPost, "/v1/agents/sessions/"+id+"/responses",
		CreateResponseRequest{Text: "Where should we eat tonight?", RequestId: pointerTo(s.utils.uuid())}, nil))
	answered, _ := s.await(watching, "responded")["text"].(string)
	return answered
}

func (s *SessionUISuite) stored(id string) bool {
	var found string
	return s.store.ShardDB().QueryRowContext(context.Background(),
		"SELECT id FROM agent_sessions WHERE id = ?", id).Scan(&found) == nil
}

// A client that does not know the field must not take an agent's UI away by saving it.
func (s *ConfigsSuite) TestAnUpdateLeavingTheUIInstructionsOutKeepsThem() {
	created := s.createConfig(map[string]any{"name": "concierge", "ui_instructions": "Draw a card."})

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+created.Id, map[string]any{"name": "concierge"}, nil))

	s.Equal("Draw a card.", s.uiInstructions(created.Id))
}

func (s *ConfigsSuite) TestAPatchSetsKeepsAndRemovesTheUIInstructions() {
	created := s.createConfig(map[string]any{"name": "concierge"})

	s.patch(created.Id, map[string]any{"ui_instructions": "Draw a card."})
	s.Equal("Draw a card.", s.uiInstructions(created.Id), "a patch sets them")

	s.patch(created.Id, map[string]any{"instructions": "Be brief."})
	s.Equal("Draw a card.", s.uiInstructions(created.Id), "a patch leaving them out keeps them")

	s.patch(created.Id, map[string]any{"ui_instructions": ""})
	s.Empty(s.uiInstructions(created.Id), "an empty string removes them")
}

// The directory is the whole of what the agent is told, so a sync of one whose agent.yaml no
// longer says ui takes the UI away.
func (s *SyncSuite) TestASyncStoresTheDirectorysUIInstructionsAndOneWithoutThemRemovesThem() {
	synced := s.sync(map[string]any{"name": "concierge", "hash": "v1", "instructions": "Be brief.",
		"ui_instructions": "Draw a card."})
	s.Equal("Draw a card.", value(synced.Config.UiInstructions))

	s.sync(map[string]any{"name": "concierge", "hash": "v2", "instructions": "Be brief."})

	var stored AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+synced.Config.Id, nil, &stored))
	s.Nil(stored.UiInstructions)
	s.Equal("Be brief.", value(stored.Instructions))
}

// uiInstructions are what the config is read back with, empty when it has none.
func (s *ConfigsSuite) uiInstructions(id string) string {
	var stored AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+id, nil, &stored))
	return value(stored.UiInstructions)
}
