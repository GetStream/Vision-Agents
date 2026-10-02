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
	call := s.started(store.Call{})

	var minted CallToken
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/calls/"+call.ID+"/token",
		map[string]any{"user_id": "thierry", "user_name": "Thierry"}, &minted))

	s.Equal("thierry", minted.UserId)
	s.Equal("Thierry", minted.UserName)

	claimed := jwt.MapClaims{}
	_, err := jwt.ParseWithClaims(minted.Token, claimed, func(*jwt.Token) (any, error) {
		return []byte(suiteStreamSecret), nil
	})
	s.Require().NoError(err)
	s.Equal("thierry", claimed["user_id"])
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
	call := s.started(store.Call{})
	s.Require().NoError(s.store.RecordCallEvents(context.Background(), []store.CallEvent{{
		CustomerID: s.customerID(), CallID: call.CallID, AgentID: call.AgentID,
		At: s.base.Add(time.Second), Kind: "answer", Reason: "a complete thought",
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

// inChannel writes one line into an agent channel, created and stamped for a customer the
// way the router creates the channels it writes to.
func (s *CallsSuite) inChannel(channel, customer string, at time.Time, text string) {
	ctx := context.Background()
	creator := "vision-agent"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{
			CreatedByID: &creator, Custom: map[string]any{conversation.CustomerField: customer},
		}})
	s.Require().NoError(err)
	s.chat.At(at)
	speaker := "alice"
	_, err = s.chat.Client.Chat().SendMessage(ctx, chatlog.ChannelType, channel,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{Text: &text, UserID: &speaker}})
	s.Require().NoError(err)
}

// started records a call the suite's app is on, filling in whatever the test did not name.
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
