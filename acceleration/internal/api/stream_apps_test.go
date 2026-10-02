//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"

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
	call := store.Call{
		ID: s.utils.uuid(), CustomerID: s.customerID(), CallID: s.utils.callID(),
		AgentID: "agent-" + s.utils.uuid(), StartedAt: time.Now().UTC(),
	}
	s.Require().NoError(s.store.StartCall(context.Background(), &call))

	var minted CallToken
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/calls/"+call.ID+"/token", map[string]any{}, &minted))

	s.Equal("own-key", minted.ApiKey)
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
