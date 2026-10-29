//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type CallsSuite struct {
	RouterSuite

	// base is when the calls in a test happened. It is a fixed hour, and the app is new in
	// every test, so what is listed is only what the test itself started.
	base time.Time
}

func TestCallsSuite(t *testing.T) {
	runSuite(t, new(CallsSuite))
}

// SetupTest gives every test an app of its own, because the running calls are everything
// one customer has not finished.
func (s *CallsSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.base = time.Date(2026, 4, 1, 9, 0, 0, 0, time.UTC)
}

func (s *CallsSuite) TestACallIsFoundAfterTheSessionRunningItIsGone() {
	// A session lives in a map in memory. This is the whole point of the row: the call
	// is still there once the process that held it is not.
	number := s.utils.number()
	call := s.started(store.Call{Direction: store.Outbound, ToNumber: number})

	var read Call
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/calls/"+call.ID, nil, &read))
	s.Equal(call.CallID, read.CallId)
	s.Equal(Outbound, read.Direction)
	s.Equal(number, value(read.ToNumber))
	s.Nil(read.EndedAt, "a call nobody has ended is still running")
}

func (s *CallsSuite) TestAJoinTokenSaysWhichCallToJoinAndWhoAsIt() {
	// The browser is handed a token and a call, never the secret: whoever holds this can
	// join one call as one user until it expires, and can sign nothing of their own.
	call := s.started(store.Call{})

	var minted CallToken
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/calls/"+call.ID+"/token", map[string]any{}, &minted))

	s.Equal(suiteStreamKey, minted.ApiKey)
	s.Equal(call.CallID, minted.CallId, "the Stream call, not the id we hold it by")
	s.Equal(defaultCallType, minted.CallType, "a call nothing said otherwise about")
	s.NotEqual(call.AgentID, minted.UserId, "a listener is not the agent")
	s.True(minted.ExpiresAt.After(time.Now()), "a token that has expired is no use")

	claimed := jwt.MapClaims{}
	_, err := jwt.ParseWithClaims(minted.Token, claimed, func(*jwt.Token) (any, error) {
		return []byte(suiteStreamSecret), nil
	})
	s.Require().NoError(err, "the token is signed with the app secret")
	s.Equal(minted.UserId, claimed["user_id"], "and it is signed for the user it names")
}

func (s *CallsSuite) TestAJoinTokenIsMintedForTheUserTheCallerAsksFor() {
	// In its own Stream app every user id is the customer's to name.
	id := streamAppID()
	s.useApp(s.numberedApp(id))
	key := "key-" + s.customerID()
	s.giveApp(s.customerID(), id, key)
	call := s.started(store.Call{StreamAppPK: id})

	var minted CallToken
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/calls/"+call.ID+"/token",
		map[string]any{"user_id": "thierry", "user_name": "Thierry"}, &minted))

	s.Equal("thierry", minted.UserId)
	s.Equal("Thierry", minted.UserName)

	claimed := jwt.MapClaims{}
	_, err := jwt.ParseWithClaims(minted.Token, claimed, func(*jwt.Token) (any, error) {
		return []byte(key + "-secret"), nil
	})
	s.Require().NoError(err)
	s.Equal("thierry", claimed["user_id"])
}

func (s *CallsSuite) TestAJoinTokenInASharedAppIsMintedOnlyForAUserTheCustomerHolds() {
	// A Stream token is good for everything its user may do in the app, not for one call,
	// and in the deployment's app every customer's users are side by side.
	call := s.started(store.Call{})
	mine, theirs := s.guestOf(s.app), s.guestOf(s.data.createApp())

	for _, user := range []string{"listener-" + call.ID, mine} {
		var minted CallToken
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
			"/v1/agents/calls/"+call.ID+"/token", map[string]any{"user_id": user}, &minted), user)
		s.Equal(user, minted.UserId)
	}
	for _, user := range []string{"thierry", theirs, "listener-" + s.utils.uuid()} {
		status, failure := s.serverClient.failure(http.MethodPost,
			"/v1/agents/calls/"+call.ID+"/token", map[string]any{"user_id": user})
		s.Equal(http.StatusBadRequest, status, user)
		s.Equal(errUserInSharedApp.Message, failure, user)
	}
}

func (s *CallsSuite) TestAChatTokenInASharedAppIsMintedOnlyForAUserTheCustomerHolds() {
	mine, theirs := s.guestOf(s.app), s.guestOf(s.data.createApp())

	for _, user := range []string{"reader-" + s.customerID(), mine} {
		agent := "agent-" + s.utils.uuid()
		var minted ChatToken
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/chat-token",
			map[string]any{"agent_id": agent, "user_id": user}, &minted), user)
		s.Equal(user, minted.UserId)
		s.Contains(s.chat.Members(agent), user)
	}
	for _, user := range []string{"thierry-" + s.utils.uuid(), theirs, "reader-" + s.utils.uuid()} {
		agent := "agent-" + s.utils.uuid()
		_, existed := s.chat.User(user)
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/chat-token",
			map[string]any{"agent_id": agent, "user_id": user})
		s.Equal(http.StatusBadRequest, status, user)
		s.Equal(errUserInSharedApp.Message, failure, user)
		_, opened := s.chat.Channel(agent)
		s.False(opened, "a refused reader opens no channel")
		_, exists := s.chat.User(user)
		s.Equal(existed, exists, "nor is made a user")
	}
}

func (s *CallsSuite) TestAChatTokenIsRefusedForAnotherCustomersAgentChannel() {
	// Agent ids are the customers' own words, so in a shared app two customers can name
	// the same channel; the stamp says whose it is.
	agent := "agent-" + s.utils.uuid()
	s.inChannel(agent, s.data.createApp().app.ID, s.base, "said to another customer")
	before := s.chat.Members(agent)

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/chat-token",
		map[string]any{"agent_id": agent})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(errChannelOfAnother.Message, failure)
	s.Equal(before, s.chat.Members(agent), "nobody is added to it")
}

func (s *CallsSuite) TestATokenInASharedAppIsNotMintedForAGuestOnlySeenInTheCustomersOwnToken() {
	// The customer signs its guest tokens itself, for any id it likes, so a guest seen in
	// one is no sign the id is not another customer's user.
	call := s.started(store.Call{})
	seen := s.guestSeenInAToken()

	for path, body := range map[string]map[string]any{
		"/v1/agents/calls/" + call.ID + "/token": {"user_id": seen},
		"/v1/agents/chat-token":                  {"agent_id": "agent-" + s.utils.uuid(), "user_id": seen},
	} {
		status, failure := s.serverClient.failure(http.MethodPost, path, body)
		s.Equal(http.StatusBadRequest, status, path)
		s.Equal(errUserInSharedApp.Message, failure, path)
	}
}

func (s *CallsSuite) TestAChatTokenReadsTheCustomersOwnAndUnstampedChannels() {
	stamped, unstamped := "agent-"+s.utils.uuid(), "agent-"+s.utils.uuid()
	s.inChannel(stamped, s.customerID(), s.base, "said to this customer")
	creator := "vision-agent"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(context.Background(), chatlog.ChannelType, unstamped,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &creator}})
	s.Require().NoError(err)

	for _, agent := range []string{stamped, unstamped} {
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/chat-token",
			map[string]any{"agent_id": agent}, nil), agent)
		s.Contains(s.chat.Members(agent), "reader-"+s.customerID(), agent)
	}
}

func (s *CallsSuite) TestAChannelAChatTokenOpensIsStampedForTheCustomer() {
	// Stamped, a channel opened before the agent first speaks is not another customer's
	// to join afterwards.
	agent := "agent-" + s.utils.uuid()
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/chat-token",
		map[string]any{"agent_id": agent}, nil))

	opened, _ := s.chat.Channel(agent)
	custom, _ := opened["custom"].(map[string]any)
	s.Equal(s.customerID(), custom[conversation.CustomerField])
}

func (s *CallsSuite) TestACallAnotherAppHoldsCannotBeJoined() {
	// Handing out a token for somebody else's call would be handing out their call.
	call := s.started(store.Call{})

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/calls/"+call.ID+"/token", map[string]any{}, nil)
	})
}

func (s *CallsSuite) TestTheRunningCallsAreTheOnesThatHaveNotEnded() {
	running := s.started(store.Call{StartedAt: s.base})
	finished := s.started(store.Call{StartedAt: s.base.Add(-time.Hour)})
	s.Require().NoError(s.store.FinishCall(context.Background(), finished.ID, s.base))

	var listed []Call
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/calls?running=true", nil, &listed))
	s.Require().Len(listed, 1)
	s.Equal(running.ID, listed[0].Id)

	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/calls", nil, &listed))
	s.Require().Len(listed, 2, "both calls happened")
	s.Equal(running.ID, listed[0].Id, "newest first")
	s.NotNil(listed[1].EndedAt)
}

func (s *CallsSuite) TestWhatACallDecidedIsFoundByTheRowRecordingIt() {
	// A dashboard holds the row's id, and the agent wrote its reasoning against the call
	// it joined. Reading one by the other is what puts the decision log on the page.
	// Decisions are kept for an hour, so this call is a recent one.
	call := s.started(store.Call{StartedAt: time.Now().Add(-time.Minute)})
	s.Require().NoError(s.live.RecordCallEvents(context.Background(), []live.CallEvent{{
		CustomerID: s.customerID(), CallID: call.CallID, AgentID: call.AgentID,
		At: time.Now(), Kind: "answer", Reason: "a complete thought",
		Said: "how is the weather",
	}}))

	var decided []CallEvent
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/calls/"+call.ID+"/events", nil, &decided))
	s.Require().Len(decided, 1)
	s.Equal(DecisionKind("answer"), decided[0].Kind)
	s.Equal("how is the weather", value(decided[0].Said))
}

func (s *CallsSuite) TestAFinishedNativeCallNamesTheConversationAndSubagentModels() {
	call := s.started(store.Call{STS: "stub/stub-sts", Subagent: "stub/stub-llm"})
	for modality, model := range map[string]string{"sts": "stub-sts", "llm": "stub-llm"} {
		s.Require().NoError(s.store.RecordRequest(context.Background(), &store.Request{
			CustomerID: s.customerID(), AgentID: call.AgentID,
			Modality: modality, Provider: "stub", Model: model,
			StartedAt: s.base.Add(time.Second), Success: true,
		}))
	}
	s.Require().NoError(s.store.FinishCall(context.Background(), call.ID, s.base.Add(time.Minute)))

	var rendered Call
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/calls/"+call.ID, nil, &rendered))
	s.Equal("stub/stub-sts", value(rendered.StsUsed))
	s.Equal("stub/stub-llm", value(rendered.SubagentUsed))
	s.Nil(rendered.SttUsed)
	s.Nil(rendered.LlmUsed)
	s.Nil(rendered.TtsUsed)
}

func (s *CallsSuite) TestAFinishedCallReportsWhatItSpentAndWhoItSpokeTo() {
	call := s.started(store.Call{UserID: "ada"})
	s.Require().NoError(s.store.RecordRequest(context.Background(), &store.Request{
		CustomerID: s.customerID(), AgentID: call.AgentID,
		Modality: "llm", Provider: "stub", Model: "stub-llm",
		StartedAt:   s.base.Add(time.Second),
		InputTokens: 900, CachedInputTokens: 400, OutputTokens: 150,
		CostMicros: 2500, Success: true,
	}))
	s.Require().NoError(s.store.FinishCall(context.Background(), call.ID, s.base.Add(time.Minute)))

	var rendered Call
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/calls/"+call.ID, nil, &rendered))
	s.Equal("ada", value(rendered.UserId))
	s.Require().NotNil(rendered.Usage)
	s.Equal(int64(900), rendered.Usage.InputTokens)
	s.Equal(int64(400), rendered.Usage.CachedInputTokens)
	s.Equal(int64(150), rendered.Usage.OutputTokens)
	s.Equal(int64(2500), rendered.Usage.CostMicros)
	s.Equal(int64(1), rendered.Usage.Requests)
}

func (s *CallsSuite) TestACallStillRunningIsNotToldWhatItHasSpentSoFar() {
	call := s.started(store.Call{})
	s.Require().NoError(s.store.RecordRequest(context.Background(), &store.Request{
		CustomerID: s.customerID(), AgentID: call.AgentID,
		Modality: "llm", Provider: "stub", Model: "stub-llm",
		StartedAt: s.base.Add(time.Second), InputTokens: 900, Success: true,
	}))

	var rendered Call
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/calls/"+call.ID, nil, &rendered))
	s.Nil(rendered.Usage, "what a conversation cost is a question asked after it")
}

func (s *CallsSuite) TestACallsTokensAreReadWhileItRunsByModelAndByWhatThePromptsCarried() {
	call := s.started(store.Call{})
	for _, request := range []store.Request{
		{Modality: "llm", Provider: "stub", Model: "stub-llm", InputTokens: 900, CachedInputTokens: 400,
			OutputTokens: 150, InputParts: store.InputParts{InstructionTokens: 300, MessageTokens: 100, ToolDefinitionTokens: 500}},
		{Modality: "llm", Provider: "stub", Model: "stub-llm", InputTokens: 100, OutputTokens: 10,
			InputParts: store.InputParts{MessageTokens: 40, ImageTokens: 60}},
		{Modality: "llm", Provider: "vision", Model: "vision-model", InputTokens: 50, OutputTokens: 5,
			InputParts: store.InputParts{VideoTokens: 50}},
		{Modality: "tts", Provider: "stub", Model: "stub-tts", Characters: 80},
	} {
		request.CustomerID, request.AgentID = s.customerID(), call.AgentID
		request.StartedAt, request.Success = s.base.Add(time.Second), true
		s.Require().NoError(s.store.RecordRequest(context.Background(), &request))
	}

	var spent CallTokens
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/calls/"+call.ID+"/tokens", nil, &spent))
	s.Equal(int64(1050), spent.InputTokens)
	s.Equal(int64(400), spent.CachedInputTokens)
	s.Equal(int64(165), spent.OutputTokens)
	s.Equal(InputParts{Instructions: 300, Messages: 140, ToolDefinitions: 500, Images: 60, Video: 50}, spent.InputParts)
	s.Require().Len(spent.Models, 3, "a voice is a model too, though it reads no tokens")
	s.Equal("stub-llm", spent.Models[0].Model, "the busiest first")
	s.Equal(int64(2), spent.Models[0].Requests)
	s.Equal(InputParts{Video: 50}, spent.Models[1].InputParts)
}

func (s *CallsSuite) TestACallsCostIsSplitByWhereItCameFrom() {
	call := s.started(store.Call{})
	for _, request := range []store.Request{
		{Modality: "llm", Provider: "stub", Model: "stub-llm", InputTokens: 1000, OutputTokens: 100,
			CostMicros: 1500, OutputCostMicros: 500,
			InputParts: store.InputParts{InstructionTokens: 200, ToolDefinitionTokens: 799, MessageTokens: 1}},
		{Modality: "llm", Provider: "stub", Model: "stub-llm", InputTokens: 1000, CostMicros: 1000},
		{Modality: "tts", Provider: "stub", Model: "stub-tts", Characters: 80, CostMicros: 300},
	} {
		request.CustomerID, request.AgentID = s.customerID(), call.AgentID
		request.StartedAt, request.Success = s.base.Add(time.Second), true
		s.Require().NoError(s.store.RecordRequest(context.Background(), &request))
	}

	var spent CallTokens
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/calls/"+call.ID+"/tokens", nil, &spent))
	s.Equal(int64(2800), spent.CostMicros)
	s.Equal([]CostSource{
		{Source: "input", CostMicros: 1000},
		{Source: "tool_definitions", CostMicros: 799},
		{Source: "output", CostMicros: 500},
		{Source: "tts", CostMicros: 300},
		{Source: "instructions", CostMicros: 200},
		{Source: "messages", CostMicros: 1},
	}, spent.CostSources, "the prompt recorded without a split is input")
}

func (s *CallsSuite) TestAConversationRecordsWhatItsPromptsWereMadeOf() {
	request := textSession(nil)
	request.Llm, request.ConfigId = pointerTo("counted/counted-model"), s.data.instructedAgent("Answer about "+s.utils.uuid())
	opened := s.serverClient.createSession(request)
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+opened.Id+"/respond", SayRequest{Text: "How many tokens is this?"}, nil))

	var spent CallTokens
	s.Require().Eventually(func() bool {
		return s.serverClient.do(http.MethodGet, "/v1/agents/calls/"+opened.Id+"/tokens", nil, &spent) == http.StatusOK &&
			spent.InputTokens >= 1000
	}, settleFor, 20*time.Millisecond, "the turn's request is recorded")
	parts := spent.InputParts
	s.Equal(spent.InputTokens, parts.Instructions+parts.Messages+parts.ToolDefinitions+parts.ToolUse+parts.Images+parts.Video,
		"the parts add up to what the provider counted")
	s.Positive(parts.Instructions)
	s.Positive(parts.Messages)
}

func (s *CallsSuite) TestAnotherAppsCallIsNotFound() {
	call := s.started(store.Call{})

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/calls/"+call.ID, nil, nil)
	})
}

func (s *CallsSuite) TestNobodyIsListedACallWithoutCredentials() {
	s.started(store.Call{})

	s.Equal(http.StatusUnauthorized,
		s.unauthenticatedClient.do(http.MethodGet, "/v1/agents/calls", nil, nil))
}

func (s *CallsSuite) TestTheTranscriptOfAConversationBoundCallIsReadFromTheConversation() {
	// A voice call bound to a conversation writes into the conversation's channel, not
	// the agent's own, so reading the agent's channel would find nothing it said.
	call := s.started(store.Call{})
	conversation := "support-" + s.utils.uuid()
	s.Require().NoError(s.store.SaveSession(context.Background(), &store.AgentSession{
		ID: call.ID, CustomerID: s.customerID(), AgentID: call.AgentID,
		ConversationID: "agent:" + conversation, CallID: call.CallID,
	}))
	s.inChannel(conversation, s.customerID(), call.StartedAt.Add(time.Minute), "said in the conversation")
	s.inChannel(call.AgentID, s.customerID(), call.StartedAt.Add(time.Minute), "not where this call wrote")

	var read []TranscriptMessage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/calls/"+call.ID+"/transcript", nil, &read))

	s.Require().Len(read, 1)
	s.Equal("said in the conversation", read[0].Text)
}

func (s *CallsSuite) TestACallWithoutASessionRowReadsItsAgentChannel() {
	// Calls from before session rows were kept have only their agent to go by.
	call := s.started(store.Call{})
	s.inChannel(call.AgentID, s.customerID(), call.StartedAt.Add(time.Minute), "said on the call")

	var read []TranscriptMessage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/calls/"+call.ID+"/transcript", nil, &read))

	s.Require().Len(read, 1)
	s.Equal("said on the call", read[0].Text)
}

func (s *CallsSuite) TestReadingACallsTranscriptCreatesNoChannel() {
	call := s.started(store.Call{})

	var read []TranscriptMessage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/calls/"+call.ID+"/transcript", nil, &read))

	s.Empty(read)
	_, exists := s.chat.Channel(call.AgentID)
	s.False(exists, "a call nobody wrote down leaves nothing behind in the app")
}

func (s *CallsSuite) TestATimelinePairsWhatEachCallerSaidWithTheAgentsAnswer() {
	// A caller's line is stored when they start talking, so it is older than the turn it
	// started. The times are a staging call's: Alice's line at 18:26:10.608, her turn at
	// 18:26:11.940, Bob's line at 18:26:21.409, his turn at 18:26:22.468.
	ctx := context.Background()
	call := s.started(store.Call{StartedAt: time.Now().UTC().Truncate(time.Millisecond).Add(-time.Minute)})
	at := func(ms int) time.Time { return call.StartedAt.Add(time.Duration(ms) * time.Millisecond) }
	s.inChannelAs(call.AgentID, s.customerID(), "alice", at(10_608), "My favorite color is blue.")
	s.inChannelAs(call.AgentID, s.customerID(), call.AgentID, at(13_100), "Got it, your favorite color is blue.")
	s.inChannelAs(call.AgentID, s.customerID(), "bob", at(21_409), "My favorite color is green.")
	s.inChannelAs(call.AgentID, s.customerID(), call.AgentID, at(23_500), "Noted, Bob, your favorite is green.")
	for _, turn := range []live.Turn{{TurnID: "turn-1", StartedAt: at(11_940)}, {TurnID: "turn-2", StartedAt: at(22_468)}} {
		turn.CustomerID, turn.AgentID, turn.CallID = s.customerID(), call.AgentID, call.CallID
		s.Require().NoError(s.live.RecordTurn(ctx, turn))
	}

	var timeline []TimelineEntry
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/calls/"+call.ID+"/timeline", nil, &timeline))

	s.Require().Len(timeline, 2)
	s.Equal("My favorite color is blue.", value(timeline[0].Heard))
	s.Equal("Got it, your favorite color is blue.", value(timeline[0].Said))
	s.Equal("My favorite color is green.", value(timeline[1].Heard))
	s.Equal("Noted, Bob, your favorite is green.", value(timeline[1].Said))
}

func (s *CallsSuite) TestATimelineOfACallNobodyWroteDownHasTurnsWithoutWords() {
	ctx := context.Background()
	call := s.started(store.Call{StartedAt: time.Now().UTC().Truncate(time.Millisecond).Add(-time.Minute)})
	s.Require().NoError(s.live.RecordTurn(ctx, live.Turn{CustomerID: s.customerID(), AgentID: call.AgentID,
		CallID: call.CallID, TurnID: "turn-1", StartedAt: call.StartedAt.Add(time.Second)}))

	var timeline []TimelineEntry
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/calls/"+call.ID+"/timeline", nil, &timeline))

	s.Require().Len(timeline, 1)
	s.Equal("turn-1", timeline[0].TurnId)
	s.Nil(timeline[0].Heard)
	s.Nil(timeline[0].Said)
}

// inChannel writes one line into an agent channel, created and stamped for a customer the
// way the router creates the channels it writes to.
func (s *CallsSuite) inChannel(channel, customer string, at time.Time, text string) {
	s.inChannelAs(channel, customer, "alice", at, text)
}

// inChannelAs is inChannel written as somebody: a line written as the agent's own id is
// what the agent said.
func (s *CallsSuite) inChannelAs(channel, customer, speaker string, at time.Time, text string) {
	ctx := context.Background()
	creator := "vision-agent"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{
			CreatedByID: &creator, Custom: map[string]any{conversation.CustomerField: customer},
		}})
	s.Require().NoError(err)
	s.chat.At(at)
	_, err = s.chat.Client.Chat().SendMessage(ctx, chatlog.ChannelType, channel,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{Text: &text, UserID: &speaker}})
	s.Require().NoError(err)
}

// guestOf is a guest an app's page made, the suite's app being the one given afterwards.
func (s *CallsSuite) guestOf(app testApp) string {
	mine := s.app
	s.useApp(app)
	defer s.useApp(mine)
	var guest GuestUser
	s.Require().Equal(http.StatusCreated, s.anonymousClient.do(http.MethodPost, "/v1/agents/guests", nil, &guest))
	return guest.Id
}

// started records a call the suite's app is on, filling in whatever the test did not name.
func (s *CallsSuite) TestCallsCanBeListedByHowBadlyTheyWentFor() {
	ctx := context.Background()
	bad, good, unreviewed := s.started(store.Call{}), s.started(store.Call{}), s.started(store.Call{})
	two, five := 2, 5
	s.Require().NoError(s.store.ReviewCall(ctx, s.customerID(), bad.ID, "Stalled.", &two, ""))
	s.Require().NoError(s.store.ReviewCall(ctx, s.customerID(), good.ID, "Helped.", &five, ""))

	var listed []Call
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/calls?max_review_score=3", nil, &listed))
	s.Require().Len(listed, 1, "neither the good call nor %s, which nobody reviewed", unreviewed.ID)
	s.Equal(bad.ID, listed[0].Id)

	s.Equal(http.StatusBadRequest, s.serverClient.do(http.MethodGet,
		"/v1/agents/calls?max_review_score=9", nil, nil))
}

func (s *CallsSuite) started(call store.Call) store.Call {
	call.CustomerID = s.customerID()
	if call.ID == "" {
		call.ID = s.utils.uuid()
	}
	if call.CallID == "" {
		call.CallID = s.utils.callID()
	}
	if call.AgentID == "" {
		call.AgentID = "agent-" + s.utils.uuid()
	}
	if call.StartedAt.IsZero() {
		call.StartedAt = s.base
	}
	s.Require().NoError(s.store.StartCall(context.Background(), &call))
	return call
}
