//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// StreamAppsSuite gives the suite's customer a Stream app of its own and checks that what
// the API does in Stream for them happens there, and not in the deployment's app.
type StreamAppsSuite struct {
	RouterSuite

	own *chattest.Server
}

func TestStreamAppsSuite(t *testing.T) {
	runSuite(t, new(StreamAppsSuite))
}

func (s *StreamAppsSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.own = s.giveApp(s.customerID(), 4242, "own-key")
}

func (s *StreamAppsSuite) TestACallTokenNamesTheIdentitysKey() {
	// The browser joins whichever app the token names, so it has to be the one the agent
	// joined the call in.
	call := s.called(4242)

	var minted CallToken
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/calls/"+call.ID+"/token", map[string]any{}, &minted))

	s.Equal("own-key", minted.ApiKey)
}

func (s *StreamAppsSuite) TestACallTokenIsMintedInTheCallsApp() {
	// A call made before the customer had an app of its own was made in the deployment's,
	// and joining it means joining it there.
	call := s.called(0)

	var minted CallToken
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/calls/"+call.ID+"/token", map[string]any{}, &minted))

	s.Equal(suiteStreamKey, minted.ApiKey)
}

func (s *StreamAppsSuite) TestACallInAnAppTheCustomerLeftMintsNothing() {
	call := s.called(7)

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/calls/"+call.ID+"/token", map[string]any{})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "no longer acts in")
}

func (s *StreamAppsSuite) TestATranscriptIsReadFromTheCallsApp() {
	call := s.called(0)
	s.lineIn(s.chat, call.AgentID, call.StartedAt.Add(time.Minute), "said in the deployment's app")
	s.lineIn(s.own, call.AgentID, call.StartedAt.Add(time.Minute), "not where this call was")

	var read []TranscriptMessage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/calls/"+call.ID+"/transcript", nil, &read))

	s.Require().Len(read, 1)
	s.Equal("said in the deployment's app", read[0].Text)
}

func (s *StreamAppsSuite) TestAClaimJoinsEachChannelInItsOwnApp() {
	// A guest's conversations may span the time before and after the customer had an app
	// of its own, and each is joined where it is.
	guest := guestPrefix + s.utils.uuid()
	s.Require().NoError(s.store.RecordGuest(context.Background(), &store.GuestUser{
		ID: guest, CustomerID: s.customerID(), Name: "Guest",
	}))
	before, after := "support-"+s.utils.uuid(), "support-"+s.utils.uuid()
	s.talkedIn(s.chat, guest, before, 0)
	s.talkedIn(s.own, guest, after, 4242)
	account := "account-" + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/guests/claim",
		ClaimGuestRequest{GuestId: guest, UserId: account}, nil))

	s.Contains(s.chat.Members(before), account)
	s.Contains(s.own.Members(after), account)
	s.NotContains(s.own.Members(before), account)
}

// called records a call the suite's customer made in the app given, zero being the
// deployment's.
func (s *StreamAppsSuite) called(app int64) store.Call {
	call := store.Call{
		ID: s.utils.uuid(), CustomerID: s.customerID(), StreamAppPK: app, CallID: s.utils.callID(),
		AgentID: "agent-" + s.utils.uuid(), StartedAt: time.Now().UTC().Add(-time.Hour),
	}
	s.Require().NoError(s.store.StartCall(context.Background(), &call))
	return call
}

// lineIn writes one line into a channel of one app, creating it stamped for the customer.
func (s *StreamAppsSuite) lineIn(chat *chattest.Server, channel string, at time.Time, text string) {
	ctx := context.Background()
	creator, speaker := "vision-agent", "alice"
	_, err := chat.Client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{
			CreatedByID: &creator, Custom: map[string]any{conversation.CustomerField: s.customerID()},
		}})
	s.Require().NoError(err)
	chat.At(at)
	_, err = chat.Client.Chat().SendMessage(ctx, chatlog.ChannelType, channel,
		&getstream.SendMessageRequest{Message: getstream.MessageRequest{Text: &text, UserID: &speaker}})
	s.Require().NoError(err)
}

// talkedIn is a finished conversation a user had in one app's channel.
func (s *StreamAppsSuite) talkedIn(chat *chattest.Server, user, channel string, app int64) {
	creator := "vision-agent"
	_, err := chat.Client.Chat().GetOrCreateChannel(context.Background(), chatlog.ChannelType, channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &creator}})
	s.Require().NoError(err)
	s.Require().NoError(s.store.SaveSession(context.Background(), &store.AgentSession{
		ID: s.utils.uuid(), CustomerID: s.customerID(), StreamAppPK: app, UserID: user,
		AgentID: "agent", ConversationID: chatlog.ChannelType + ":" + channel, State: store.SessionClosed,
	}))
}

func (s *StreamAppsSuite) TestAChatTokenNamesTheCallingAppsKey() {
	agent := "agent-" + s.utils.uuid()

	var minted ChatToken
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/chat-token", map[string]any{"agent_id": agent}, &minted))

	s.Equal("own-key", minted.ApiKey)
	_, inOwn := s.own.Channel(agent)
	s.True(inOwn, "the channel is opened in the customer's own app")
	_, inDeployment := s.chat.Channel(agent)
	s.False(inDeployment)
}

func (s *StreamAppsSuite) TestAChatTokenLeavesAnExistingReaderAlone() {
	// The reader is a real person in the customer's app; minting them a token must not
	// write them back as an id and a name.
	s.own.PutUser(map[string]any{"id": "reader-ada", "name": "Ada Lovelace", "image": "https://example.com/ada.png"})

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/chat-token",
		map[string]any{"agent_id": "agent-" + s.utils.uuid(), "user_id": "reader-ada"}, nil))

	reader, _ := s.own.User("reader-ada")
	s.Equal("Ada Lovelace", reader["name"])
	s.Equal("https://example.com/ada.png", reader["image"])
}

func (s *StreamAppsSuite) TestAGuestIsCreatedInTheCallingApp() {
	var guest GuestUser
	s.Require().Equal(http.StatusCreated, s.anonymousClient.do(http.MethodPost, "/v1/agents/guests", nil, &guest))

	_, inOwn := s.own.User(guest.Id)
	s.True(inOwn, "a guest's token is only good in the app it was created in")
	_, inDeployment := s.chat.User(guest.Id)
	s.False(inDeployment)
}

func (s *StreamAppsSuite) TestASessionRecordsTheAppItWasCreatedIn() {
	// The rows a session leaves behind say which app it was in, so what is finished or read
	// back later is done there, wherever the customer acts by then.
	call := s.utils.callID()
	created := s.serverClient.createSession(CreateSessionRequest{CallId: &call})

	s.Require().Eventually(func() bool {
		stored, err := s.store.StoredSession(context.Background(), s.customerID(), created.Id)
		return err == nil && stored.StreamAppPK == 4242
	}, settleFor, 10*time.Millisecond, "the session row carries its app")
	s.Require().Eventually(func() bool {
		row, err := s.store.Call(context.Background(), s.customerID(), created.Id)
		return err == nil && row.StreamAppPK == 4242
	}, settleFor, 10*time.Millisecond, "the call row carries its app")
}

func (s *StreamAppsSuite) TestAnInboundCallOnALegacyNumberJoinsTheDeploymentAppsCall() {
	// The number was attached before the customer had an app of its own, so callers still
	// land in the deployment's app, and that is where the agent has to be.
	call := s.attached(0)

	created := s.serverClient.createSession(CreateSessionRequest{CallId: &call})

	s.Equal(int64(0), s.pinOf(created.Id))
}

func (s *StreamAppsSuite) TestAnInboundCallOnANumberInTheCustomersAppJoinsThere() {
	call := s.attached(4242)

	created := s.serverClient.createSession(CreateSessionRequest{CallId: &call})

	s.Equal(int64(4242), s.pinOf(created.Id))
}

// attached is the call a number the customer holds routes callers into, attached in the
// app given.
func (s *StreamAppsSuite) attached(app int64) string {
	ctx := context.Background()
	e164 := s.utils.number()
	call := "phone-" + e164
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164: e164, Vendor: "telnyx", Country: "US", CustomerID: s.customerID(), PurchasedAt: time.Now().UTC(),
	}))
	s.Require().NoError(s.store.AttachNumber(ctx, s.customerID(), e164, store.NumberAttachment{
		TrunkID: "trunk-" + s.utils.uuid(), StreamAppPK: app, CallType: "agent", CallID: call,
	}))
	return call
}

// pinOf is the app a session was pinned to, read off its row once it is written.
func (s *StreamAppsSuite) pinOf(id string) int64 {
	var stored store.AgentSession
	s.Require().Eventually(func() bool {
		var err error
		stored, err = s.store.StoredSession(context.Background(), s.customerID(), id)
		return err == nil
	}, settleFor, 10*time.Millisecond, "the session was never written down")
	return stored.StreamAppPK
}
