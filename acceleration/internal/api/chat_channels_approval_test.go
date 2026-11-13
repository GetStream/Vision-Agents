//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
)

// AI-3154: in a channel of a type a config answers in, a connector call the config holds for
// approval asks the person whose message the reply answers, and only they, while a member of
// the channel, or the app's backend may answer it (POST .../conversations/{cid}/tool_approvals).

// Anya's message asks for the call: its step asks her. Ben, a member too, is refused, as are
// someone outside the channel and an anonymous caller going by her id; nothing reaches the
// provider until she allows it, and her answer is taken once.
func (s *AppHooksSuite) TestAnApprovalInAChannelTypeGoesToTheWriterOfTheMessage() {
	s.holdsType("callouts", memberWrites)
	s.approvingIn("callouts")
	anya, ben := s.person("Anya"), s.person("Ben")
	channel := s.teamChannel("callouts", anya, ben)
	cid := "callouts:" + channel

	s.says("callouts", channel, anya, "can you check the crm for my shift?", "")
	step, request := s.awaitingIn(cid)
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.Equal(anya, step["target_user_id"])
	s.Equal(http.StatusForbidden, s.approves(s.data.signedInAs(ben), cid, step, request, true), "another member")
	s.Equal(http.StatusNotFound, s.approves(s.data.createUser(), cid, step, request, true), "someone outside the channel")
	s.Equal(http.StatusForbidden, s.approves(s.data.claiming(anya), cid, step, request, true), "an anonymous caller")
	s.Equal(http.StatusNotFound, s.approves(s.data.signedInAs(anya), cid, step, "thread_another", true), "another request")
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP), "nothing reached the provider before her answer")
	s.Equal(http.StatusNoContent, s.approves(s.data.signedInAs(anya), cid, step, request, true))
	s.Require().NotNil(s.replyIn(cid), "the reply never finished")
	s.Greater(s.provider.Hits(fakeprovider.PathMCP), before, "allowed, the call ran")
	s.Equal("completed", s.stepIn(cid)["status"])
	s.Equal(http.StatusNotFound, s.approves(s.data.signedInAs(anya), cid, step, request, true), "answered already")
}

// The app's backend may answer the writer's call: declined, it never runs.
func (s *AppHooksSuite) TestTheAppsBackendMayAnswerAWritersApproval() {
	s.holdsType("callouts", memberWrites)
	s.approvingIn("callouts")
	anya := s.person("Anya")
	channel := s.teamChannel("callouts", anya)
	cid := "callouts:" + channel
	s.says("callouts", channel, anya, "can you check the crm for my shift?", "")
	step, request := s.awaitingIn(cid)
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.Equal(http.StatusNoContent, s.approves(s.serverClient, cid, step, request, false))

	s.Require().NotNil(s.replyIn(cid), "the reply never finished")
	s.Equal("cancelled", s.stepIn(cid)["status"])
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP), "declined, nothing reached the provider")
}

// The session holding Anya's call runs on the node that took the hook; her answer, and the
// backend's, may land on another node, as a load balancer sends it with two replicas: the other
// node carries it to the holding one (onOwningNode), and the call runs.
func (s *AppHooksSuite) TestAnApprovalAnsweredOnAnotherNodeReachesTheCall() {
	other := s.otherNode()
	s.holdsType("callouts", memberWrites)
	s.approvingIn("callouts")
	anya, ben := s.person("Anya"), s.person("Ben")
	channel := s.teamChannel("callouts", anya, ben)
	cid := "callouts:" + channel
	s.says("callouts", channel, anya, "can you check the crm for my shift?", "")
	step, request := s.awaitingIn(cid)
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.Equal(http.StatusForbidden, s.approves(s.data.signedInAs(ben).on(other), cid, step, request, true),
		"another member, from the other node")
	s.Equal(http.StatusNoContent, s.approves(s.data.signedInAs(anya).on(other), cid, step, request, true),
		"her answer from the other node never reached the call")
	s.Require().NotNil(s.replyIn(cid), "the reply never finished")
	s.Greater(s.provider.Hits(fakeprovider.PathMCP), before, "allowed, the call ran")
	s.Equal(http.StatusNotFound, s.approves(s.serverClient.on(other), cid, step, request, true), "answered already")
}

// ChannelTypeTurnSuite is AppHooksSuite with the ten minutes a held call waits for its
// person outside a test, so what ends the call is the turn, not the wait.
type ChannelTypeTurnSuite struct{ AppHooksSuite }

func TestChannelTypeTurnSuite(t *testing.T) { runSuite(t, new(ChannelTypeTurnSuite)) }

func (s *ChannelTypeTurnSuite) SetupSuite() {
	s.approvalWait = 10 * time.Minute
	s.AppHooksSuite.SetupSuite()
}

// BeforeTest runs this suite's own test alone: the rest are AppHooksSuite's, run there.
func (s *ChannelTypeTurnSuite) BeforeTest(suiteName, testName string) {
	if testName != "TestAWritersApprovalAfterTheTurnEndsIsTooLate" {
		s.T().Skip("an AppHooksSuite test, run there")
	}
	s.AppHooksSuite.BeforeTest(suiteName, testName)
}

// The Router answers one message in a channel of a type for at most askTimeout (two minutes)
// and then closes the session it opened for it, so Anya's call, held for her, is stopped then,
// within seconds, eight minutes before the call's own wait would end: her answer after it is a
// 404, and nothing reaches the provider. The operation's description and the connectors skill
// say two minutes.
func (s *ChannelTypeTurnSuite) TestAWritersApprovalAfterTheTurnEndsIsTooLate() {
	s.holdsType("callouts", memberWrites)
	s.approvingIn("callouts")
	anya := s.person("Anya")
	channel := s.teamChannel("callouts", anya)
	cid := "callouts:" + channel
	start := time.Now()
	s.says("callouts", channel, anya, "can you check the crm for my shift?", "")
	step, request := s.awaitingIn(cid)
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.Require().Eventually(func() bool { return s.stepIn(cid)["status"] == "cancelled" },
		askTimeout+30*time.Second, time.Second, "the call outlived the turn")
	s.T().Logf("the call was stopped %s after Anya wrote", time.Since(start).Round(time.Second))
	s.GreaterOrEqual(time.Since(start), askTimeout, "the call was stopped before the turn ended")
	s.Equal(http.StatusNotFound, s.approves(s.data.signedInAs(anya), cid, step, request, true))
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP), "nothing reached the provider")
}

// Anya's and Ben's calls wait at once, each in a channel of the type: each asks its own writer,
// and neither may answer the other's.
func (s *AppHooksSuite) TestEachWritersApprovalGoesToThem() {
	s.holdsType("callouts", memberWrites)
	s.approvingIn("callouts")
	anya, ben := s.person("Anya"), s.person("Ben")
	first, second := "callouts:"+s.teamChannel("callouts", anya, ben), "callouts:"+s.teamChannel("callouts", anya, ben)
	s.says("callouts", first[len("callouts:"):], anya, "can you check the crm for my shift?", "")
	s.says("callouts", second[len("callouts:"):], ben, "can you check the crm for mine?", "")
	hers, herRequest := s.awaitingIn(first)
	his, hisRequest := s.awaitingIn(second)

	s.Equal(anya, hers["target_user_id"])
	s.Equal(ben, his["target_user_id"])
	s.Equal(http.StatusForbidden, s.approves(s.data.signedInAs(anya), second, his, hisRequest, true))
	s.Equal(http.StatusForbidden, s.approves(s.data.signedInAs(ben), first, hers, herRequest, true))
	s.Equal(http.StatusNoContent, s.approves(s.data.signedInAs(ben), second, his, hisRequest, true))
	s.Equal(http.StatusNoContent, s.approves(s.data.signedInAs(anya), first, hers, herRequest, true))
	s.Require().NotNil(s.replyIn(first))
	s.Require().NotNil(s.replyIn(second))
}

// Once Anya's turn is over, Ben's next one in the same channel asks Ben, not her.
func (s *AppHooksSuite) TestTheNextWritersApprovalGoesToThem() {
	s.holdsType("callouts", memberWrites)
	s.approvingIn("callouts")
	anya, ben := s.person("Anya"), s.person("Ben")
	channel := s.teamChannel("callouts", anya, ben)
	cid := "callouts:" + channel
	s.says("callouts", channel, anya, "can you check the crm for my shift?", "")
	hers, herRequest := s.awaitingIn(cid)
	s.Require().Equal(http.StatusNoContent, s.approves(s.data.signedInAs(anya), cid, hers, herRequest, true))
	s.Require().Eventually(func() bool { return s.replies(cid) == 1 }, settleFor, 20*time.Millisecond)

	s.says("callouts", channel, ben, "and for mine?", "")
	his, hisRequest := s.awaitingIn(cid)

	s.NotEqual(herRequest, hisRequest)
	s.Equal(ben, his["target_user_id"])
	s.Equal(http.StatusForbidden, s.approves(s.data.signedInAs(anya), cid, his, hisRequest, true))
	s.Equal(http.StatusNoContent, s.approves(s.data.signedInAs(ben), cid, his, hisRequest, true))
}

// Anya leaves the channel while her call waits: she can no longer answer it; the backend still can.
func (s *AppHooksSuite) TestAWriterWhoLeftTheChannelCannotAnswer() {
	s.holdsType("callouts", memberWrites)
	s.approvingIn("callouts")
	anya, ben := s.person("Anya"), s.person("Ben")
	channel := s.teamChannel("callouts", anya, ben)
	cid := "callouts:" + channel
	s.says("callouts", channel, anya, "can you check the crm for my shift?", "")
	step, request := s.awaitingIn(cid)
	_, err := s.chat.Client.Chat().UpdateChannel(context.Background(), "callouts", channel,
		&getstream.UpdateChannelRequest{RemoveMembers: []string{anya}})
	s.Require().NoError(err)

	s.Equal(http.StatusNotFound, s.approves(s.data.signedInAs(anya), cid, step, request, true))
	s.Equal(http.StatusNoContent, s.approves(s.serverClient, cid, step, request, true))
	s.Require().NotNil(s.replyIn(cid))
}

// approvingIn is a config of the app's customer answering every message in channels of the
// type named, on the model that reaches for crm's echo, whose crm binding is the app's own
// connection to the suite's fake and holds echo for a person to allow (require_approval).
func (s *AppHooksSuite) approvingIn(channelType string) string {
	connector := s.customConnector(s.provider)
	connection, echo := s.bearerConnection(s.token, nil, connector)
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "rota-" + s.utils.uuid(), "mode": "text", "llm": "connecting/connector-model",
		"chat_channels": []map[string]any{{"type": channelType}},
		"connectors": []map[string]any{{"name": "crm", "connector_id": connector,
			"connection": map[string]any{"type": "fixed", "connection_id": connection},
			"tools":      []map[string]any{echo}, "require_approval": []string{"echo"}}},
	}, &created))
	return created.Id
}

// teamChannel is a new channel of the type named, made by the customer's app with the members
// given and nothing in it.
func (s *AppHooksSuite) teamChannel(channelType string, members ...string) string {
	channel, creator := "team-"+s.utils.uuid(), members[0]
	listed := make([]getstream.ChannelMemberRequest, 0, len(members))
	for _, member := range members {
		listed = append(listed, getstream.ChannelMemberRequest{UserID: member})
	}
	_, err := s.chat.Client.Chat().GetOrCreateChannel(context.Background(), channelType, channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &creator, Members: listed}})
	s.Require().NoError(err)
	return channel
}

// awaitingIn is the step of crm's echo once the agent's reply in the channel cid shows it
// waiting for a person to allow it, and the request id that reply answers.
func (s *AppHooksSuite) awaitingIn(cid string) (map[string]any, string) {
	var step map[string]any
	request := ""
	s.Require().Eventually(func() bool {
		for _, stored := range s.chat.Stored(cid) {
			if found := echoStepOf(stored); found != nil && found["status"] == "awaiting_approval" {
				custom, _ := stored["custom"].(map[string]any)
				runtime, _ := custom["support_runtime"].(map[string]any)
				request, _ = runtime["command_id"].(string)
				step = found
				return request != ""
			}
		}
		return false
	}, settleFor, 20*time.Millisecond, "no reply in %s asks for crm's echo to be allowed", cid)
	return step, request
}

// stepIn is the latest step of crm's echo in the channel cid.
func (s *AppHooksSuite) stepIn(cid string) map[string]any {
	var step map[string]any
	for _, stored := range s.chat.Stored(cid) {
		if found := echoStepOf(stored); found != nil {
			step = found
		}
	}
	return step
}

// approves answers the call of step, of the reply to request in the conversation cid, as the
// client given, and is the status it is answered with.
func (s *AppHooksSuite) approves(as *testClient, cid string, step map[string]any, request string, allowed bool) int {
	return as.do(http.MethodPost, "/v1/agents/conversations/"+cid+"/tool_approvals",
		map[string]any{"tool_call_id": step["id"], "request_id": request, "allowed": allowed}, nil)
}

// echoStepOf is the ai_tool_call step of crm's echo among a stored Chat message's attachments.
func echoStepOf(stored map[string]any) map[string]any {
	attachments, _ := stored["attachments"].([]any)
	for _, attachment := range attachments {
		if found, _ := attachment.(map[string]any); found["type"] == "ai_tool_call" && found["name"] == connectorEcho {
			return found
		}
	}
	return nil
}
