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

func (s *ConfigsSuite) TestAPatchSetsTheUIInstructions() {
	created := s.createConfig(map[string]any{"name": "concierge"})

	s.patch(created.Id, map[string]any{"ui_instructions": "Draw a card."})

	s.Equal("Draw a card.", s.uiInstructions(created.Id))
}

func (s *ConfigsSuite) TestAPatchLeavingTheUIInstructionsOutKeepsThem() {
	created := s.createConfig(map[string]any{"name": "concierge", "ui_instructions": "Draw a card."})

	s.patch(created.Id, map[string]any{"instructions": "Be brief."})

	stored := s.readConfig(created.Id)
	s.Equal("Be brief.", value(stored.Instructions), "the patch was applied")
	s.Equal("Draw a card.", value(stored.UiInstructions))
}

func (s *ConfigsSuite) TestAPatchWithEmptyUIInstructionsRemovesThem() {
	created := s.createConfig(map[string]any{"name": "concierge", "ui_instructions": "Draw a card."})
	s.Require().Equal("Draw a card.", s.uiInstructions(created.Id), "the config starts with them")

	s.patch(created.Id, map[string]any{"ui_instructions": ""})

	s.Nil(s.readConfig(created.Id).UiInstructions)
}

// The directory is the whole of what the agent is told, so a sync of one whose agent.yaml no
// longer says ui takes the UI away.
func (s *SyncSuite) TestASyncStoresTheDirectorysUIInstructionsAndOneWithoutThemRemovesThem() {
	synced := s.sync(map[string]any{"name": "concierge", "hash": "v1", "instructions": "Be brief.",
		"ui_instructions": "Draw a card."})
	s.Equal("Draw a card.", value(s.storedConfig(synced.Config.Id).UiInstructions), "the first sync stores them")

	s.sync(map[string]any{"name": "concierge", "hash": "v2", "instructions": "Be briefer."})

	stored := s.storedConfig(synced.Config.Id)
	s.Nil(stored.UiInstructions, "a sync without them removes them")
	s.Equal("Be briefer.", value(stored.Instructions), "the second sync was written")
}

// A router older than ui_instructions kept a sync's hash and dropped them, so the SDK sends the
// same sync again once the router keeps them, and it is written.
func (s *SyncSuite) TestASyncOfTheSameHashWritesUIInstructionsTheConfigLacks() {
	synced := s.sync(map[string]any{"name": "concierge", "hash": "v1", "instructions": "Be brief."})

	again := s.sync(map[string]any{"name": "concierge", "hash": "v1", "instructions": "Be brief.",
		"ui_instructions": "Draw a card."})

	s.False(again.Unchanged)
	s.Equal("Draw a card.", value(s.storedConfig(synced.Config.Id).UiInstructions))
}

// The control: a config that has its UI is not written again for the same hash.
func (s *SyncSuite) TestASyncOfTheSameHashWithTheUIStoredIsUnchanged() {
	s.sync(map[string]any{"name": "concierge", "hash": "v1", "ui_instructions": "Draw a card."})

	again := s.sync(map[string]any{"name": "concierge", "hash": "v1", "ui_instructions": "Draw a card."})

	s.True(again.Unchanged)
}

// storedConfig is the config as the router reads it back, not as a write answered.
func (s *SyncSuite) storedConfig(id string) AgentConfig {
	var stored AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+id, nil, &stored))
	return stored
}

// uiInstructions are what the config is read back with, empty when it has none.
func (s *ConfigsSuite) uiInstructions(id string) string {
	return value(s.readConfig(id).UiInstructions)
}

// readConfig is the config as the router reads it back, not as a write answered.
func (s *ConfigsSuite) readConfig(id string) AgentConfig {
	var stored AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+id, nil, &stored))
	return stored
}
