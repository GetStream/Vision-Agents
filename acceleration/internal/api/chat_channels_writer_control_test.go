//go:build integration

package api

import (
	"context"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TestAMessageToASessionInAnAgentChannelReachesTheModelAsAtBase is the control for the writer's
// label: a session the app's backend opened on an agent channel answers Sam's message there
// (answerMessage) with what base 7460d0281 sends for it, no label, no id and no note about
// several people. The expected request is the one a probe of this test printed on 7460d0281.
func (s *AppHooksSuite) TestAMessageToASessionInAnAgentChannelReachesTheModelAsAtBase() {
	s.forgetNoted()
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "config-" + s.utils.uuid(), Mode: store.AgentModeText,
		LLM: "noted/noted-model", Instructions: "be brief"}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	s.serverClient.createSession(CreateSessionRequest{AgentId: &s.channelID, Agent: &config.Name, Incognito: pointerTo(true)})

	s.Require().Equal(http.StatusOK, s.fromApp(s.appPath("/v1/chat/hooks/stream"), s.messageIn("{}")))

	s.Require().Eventually(func() bool { return len(s.notedInputs()) > 0 }, settleFor, 20*time.Millisecond, "the message was not answered")
	request, everything := s.askedWith("is my order on its way?")
	s.Equal("be brief", withoutClock(request.Instructions))
	s.Equal([]string{"is my order on its way?"}, contentsOf(request.Input))
	for _, message := range request.Input {
		s.Empty(message.Speaker)
		s.Equal(llm.User, message.Role)
	}
	s.NotContains(everything, "sam", "the writer's id")
	s.NotContains(everything, "Sam", "the writer's name")
}
