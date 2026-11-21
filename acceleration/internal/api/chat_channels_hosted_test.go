//go:build integration

package api

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// AI-3464: in a channel of a type a config answers in, a call of a tool the app's backend hosts
// (a dispatch worker) names the person who wrote the message its turn answers, as a session
// opened with that person's token does, so the backend can act for them: Anya's turn names Anya,
// and Ben's next one Ben.

func (s *AppHooksSuite) TestAHostedCallInAChannelTypeAnsweringMentionsIsForEachTurnsWriter() {
	s.hostedCallsAreForEachWriter(store.ChatChannelAnswerMentioned)
}

func (s *AppHooksSuite) TestAHostedCallInAChannelTypeAnsweringEveryMessageIsForEachTurnsWriter() {
	s.hostedCallsAreForEachWriter(store.ChatChannelAnswerAlways)
}

// hostedCallsAreForEachWriter has Anya, then Ben, ask a config answering in callouts as answer
// says, on the model that reaches for lookup_order, which the backend's worker hosts for it and
// a tool group of the config names.
func (s *AppHooksSuite) hostedCallsAreForEachWriter(answer string) {
	s.holdsType("callouts", memberWrites)
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "rota-" + s.utils.uuid(), Mode: store.AgentModeText,
		LLM: "tooling/tool-model", ChatChannels: []store.ChatChannel{{Type: "callouts", Answer: answer}},
		ToolGroups: []store.ToolGroup{{Name: "orders", Tools: []string{lookupOrder}}}}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	worker := s.hostsLookupOrder(config.ID)
	mention := ""
	if answer == store.ChatChannelAnswerMentioned {
		mention = fmt.Sprintf(`[{"id": %q, "name": %q}]`, config.ID, config.Name)
	}
	anya, ben := s.person("Anya"), s.person("Ben")
	channel := s.teamChannel("callouts", anya, ben)
	cid := "callouts:" + channel

	s.says("callouts", channel, anya, "where is my order 12?", mention)
	s.Equal(map[string]any{"id": anya, "name": "Anya"}, s.hostedCallUser(worker), "Anya's turn")
	s.Require().Eventually(func() bool { return s.replies(cid) == 1 }, settleFor, 20*time.Millisecond)

	s.says("callouts", channel, ben, "and where is mine?", mention)
	s.Equal(map[string]any{"id": ben, "name": "Ben"}, s.hostedCallUser(worker), "Ben's turn after Anya's")
	s.Require().Eventually(func() bool { return s.replies(cid) == 2 }, settleFor, 20*time.Millisecond)
}

// hostsLookupOrder connects a worker of the app's backend hosting lookup_order for the agent id given.
func (s *RouterSuite) hostsLookupOrder(agentID string) *websocket.Conn {
	worker := s.serverClient.opens("/v1/dispatch")
	s.await(worker, "ready")
	s.Require().NoError(s.dispatch.Host(s.dispatch.Workers(s.customerID())[0], agentID,
		[]dispatch.Tool{{Name: lookupOrder, Description: "find an order by its number"}}, time.Minute))
	return worker
}

// hostedCallUser is the user the next call the worker is handed names, nil when it names none,
// once the worker has answered it.
func (s *RouterSuite) hostedCallUser(worker *websocket.Conn) any {
	call := s.await(worker, "tool_call")
	s.Require().Equal(lookupOrder, call["name"])
	s.Require().NoError(worker.WriteJSON(map[string]any{"type": "tool_result", "id": call["id"], "output": "it ships tomorrow"}))
	return call["user"]
}

// The control: a session a person opened with their own token names them on a hosted call, as
// on base b466fb795 (probe of this test there).
func (s *SessionToolsSuite) TestAHostedCallInASessionOpenedWithAUsersTokenIsForThem() {
	agent := "agent-" + s.utils.uuid()
	worker := s.hostsLookupOrder(agent)
	device := s.data.createUser()
	request := textSession(nil)
	request.AgentId, request.Llm = &agent, pointerTo("tooling/tool-model")
	opened := device.createSession(request)

	status, body := s.serverClient.actingFor(device).call(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond", RespondRequest{Text: "where is my order", RequestId: pointerTo("request-" + s.utils.uuid())})
	s.Require().Equal(http.StatusOK, status, string(body))

	s.Equal(map[string]any{"id": device.userID}, s.hostedCallUser(worker))
}
