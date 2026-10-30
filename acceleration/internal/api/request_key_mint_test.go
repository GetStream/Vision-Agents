//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// RequestKeyMintSuite mints user tokens in a customer's own app that is called with more
// than one key, behind a gateway that signs a user token with the secret of the key it is
// asked for. The router holds the app's server token for one key at a time, whichever
// lasts longest.
type RequestKeyMintSuite struct {
	RouterSuite
	id int64
}

func TestRequestKeyMintSuite(t *testing.T) {
	runSuite(t, new(RequestKeyMintSuite))
}

func (s *RequestKeyMintSuite) SetupSuite() {
	s.appMode, s.proxied = true, true
	s.RouterSuite.SetupSuite()
}

func (s *RequestKeyMintSuite) SetupTest() {
	s.id = streamAppID()
	s.useApp(s.numberedApp(s.id))
}

// holding makes the router hold the app's server token for key, lasting longer than any a
// request forwards, so a request made with another key does not replace it.
func (s *RequestKeyMintSuite) holding(key string) {
	s.answerAs(key, s.id)
	s.Require().NoError(s.stream.KeepToken(context.Background(), s.customerID(), key,
		appToken(jwt.MapClaims{"exp": time.Now().Add(48 * time.Hour).Unix()})))
}

// gatewayCall is a caller of the app as the gateway forwards it once it has checked the
// key the caller came with: it names the app, the kind of caller and that key, and writes
// the server token it minted for the key, when it could mint one.
func (s *RequestKeyMintSuite) gatewayCall(authType, user, key string, withToken bool) *testClient {
	s.answerAs(key, s.id)
	header := http.Header{}
	header.Set(auth.AppHeader, s.customerID())
	header.Set(auth.OrganizationHeader, s.app.organization.ID)
	header.Set(auth.AuthTypeHeader, authType)
	if user != "" {
		header.Set(auth.UserHeader, user)
	}
	header.Set(auth.APIKeyIDHeader, key)
	if withToken {
		header.Set(auth.ServerTokenKeyHeader, key)
		header.Set(auth.ServerTokenHeader, appToken(jwt.MapClaims{"exp": time.Now().Add(24 * time.Hour).Unix()}))
	}
	return &testClient{suite: &s.RouterSuite, header: header, userID: user}
}

// calledWith is the suite's backend, forwarded with the server token for key.
func (s *RequestKeyMintSuite) calledWith(key string) *testClient {
	return s.gatewayCall(auth.AuthTypeServer, "", key, true)
}

// signedFor is whether a user token verifies with the secret of key.
func signedFor(token, key string) bool {
	_, err := jwt.Parse(token, func(*jwt.Token) (any, error) { return []byte(keySecret(key)), nil })
	return err == nil
}

func (s *RequestKeyMintSuite) guestToken(client *testClient) string {
	var guest GuestUser
	s.Require().Equal(http.StatusCreated, client.do(http.MethodPost, "/v1/agents/guests", nil, &guest))
	return guest.Token
}

func (s *RequestKeyMintSuite) callToken(client *testClient) CallToken {
	call := store.Call{ID: s.utils.uuid(), CustomerID: s.customerID(), CallID: s.utils.callID(),
		AgentID: "agent-" + s.utils.uuid(), StreamAppPK: s.id, StartedAt: time.Now()}
	s.Require().NoError(s.store.StartCall(context.Background(), &call))
	var minted CallToken
	s.Require().Equal(http.StatusOK, client.do(http.MethodPost, "/v1/agents/calls/"+call.ID+"/token",
		map[string]any{}, &minted))
	return minted
}

func (s *RequestKeyMintSuite) chatToken(client *testClient) ChatToken {
	var minted ChatToken
	s.Require().Equal(http.StatusOK, client.do(http.MethodPost, "/v1/agents/chat-token",
		map[string]any{"agent_id": "agent-" + s.utils.uuid()}, &minted))
	return minted
}

func (s *RequestKeyMintSuite) TestAGuestIsMintedForTheKeyTheRequestCameWith() {
	s.holding("key-b")

	token := s.guestToken(s.calledWith("key-a"))

	s.True(signedFor(token, "key-a"), "the caller connects with key-a")
	s.False(signedFor(token, "key-b"))
}

func (s *RequestKeyMintSuite) TestAGuestIsMintedForTheKeyAPageCameWith() {
	// A page minting a guest is the main path of that call: an end user, not the backend.
	s.holding("key-b")

	for _, caller := range []*testClient{
		s.gatewayCall(auth.AuthTypeAnonymous, s.utils.uuid(), "key-a", true),
		s.gatewayCall(auth.AuthTypeJWT, s.utils.uuid(), "key-a", true),
	} {
		token := s.guestToken(caller)

		s.True(signedFor(token, "key-a"))
		s.False(signedFor(token, "key-b"))
	}
}

func (s *RequestKeyMintSuite) TestARequestWithNoServerTokenIsMintedForTheKeyTheGatewayNamedForTheCaller() {
	// A gateway that could not mint a server token still names the key it checked.
	s.holding("key-b")

	backend := s.gatewayCall(auth.AuthTypeServer, "", "key-a", false)
	page := s.gatewayCall(auth.AuthTypeAnonymous, s.utils.uuid(), "key-a", false)

	s.True(signedFor(s.guestToken(page), "key-a"))
	s.True(signedFor(s.guestToken(backend), "key-a"))
	minted := s.callToken(backend)
	s.Equal("key-a", minted.ApiKey)
	s.True(signedFor(minted.Token, "key-a"))
}

func (s *RequestKeyMintSuite) TestACallTokenIsMintedForAndNamesTheKeyTheRequestCameWith() {
	s.holding("key-b")

	minted := s.callToken(s.calledWith("key-a"))

	s.Equal("key-a", minted.ApiKey)
	s.True(signedFor(minted.Token, "key-a"))
}

func (s *RequestKeyMintSuite) TestAChatTokenIsMintedForAndNamesTheKeyTheRequestCameWith() {
	s.holding("key-b")

	minted := s.chatToken(s.calledWith("key-a"))

	s.Equal("key-a", minted.ApiKey)
	s.True(signedFor(minted.Token, "key-a"))
}

func (s *RequestKeyMintSuite) TestAnAppCalledWithOneKeyMintsForIt() {
	s.holding("key-a")
	client := s.calledWith("key-a")

	s.True(signedFor(s.guestToken(client), "key-a"))
	minted := s.callToken(client)
	s.Equal("key-a", minted.ApiKey)
	s.True(signedFor(minted.Token, "key-a"))
}

func (s *RequestKeyMintSuite) TestARequestTheGatewayVouchedNoKeyForMintsForTheKeyHeld() {
	// A request that names no key at all is minted for the key held.
	s.holding("key-b")
	client := s.gatewayCall(auth.AuthTypeServer, "", "key-b", false)
	client.header.Del(auth.APIKeyIDHeader)

	s.True(signedFor(s.guestToken(client), "key-b"))
	minted := s.callToken(client)
	s.Equal("key-b", minted.ApiKey)
	s.True(signedFor(minted.Token, "key-b"))
	chat := s.chatToken(client)
	s.Equal("key-b", chat.ApiKey)
	s.True(signedFor(chat.Token, "key-b"))
}
