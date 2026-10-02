//go:build integration

package api

import (
	"net/http"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
)

// AppModeSessionsSuite opens sessions with the suite's apps answering as app mode does,
// where each customer acts in an app of its own and a conversation is kept only somewhere
// safe to keep it.
type AppModeSessionsSuite struct {
	RouterSuite
}

func TestAppModeSessionsSuite(t *testing.T) {
	runSuite(t, new(AppModeSessionsSuite))
}

func (s *AppModeSessionsSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.setApps(s.customerID(), func(apps *suiteApps) { apps.perApp = true })
}

func (s *AppModeSessionsSuite) TestAPersistedTextSessionWithoutAnIdentityIsRefused() {
	// The conversation used to be held and quietly not kept anywhere.
	s.setApps(s.customerID(), func(apps *suiteApps) { apps.nowhere[s.customerID()] = true })

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions", textSession(nil))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "register this app's Stream keys")
}

func (s *AppModeSessionsSuite) TestAnIncognitoSessionNeedsNoStreamApp() {
	s.setApps(s.customerID(), func(apps *suiteApps) { apps.nowhere[s.customerID()] = true })
	incognito := textSession(nil)
	incognito.Incognito = pointerTo(true)

	created := s.serverClient.createSession(incognito)

	s.NotEmpty(created.Id)
}

func (s *AppModeSessionsSuite) TestAPersistedTextSessionIntoAnUnsafeAppIsRefused() {
	own := s.giveApp(s.customerID(), 4242, "own-key")
	own.SetApp(chattest.App{
		ID:           4242,
		ChannelTypes: map[string]map[string][]string{"agent": {"user": {"update-channel-owner"}}},
		CallTypes:    []string{"agent"},
	})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions", textSession(nil))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "restrict the channel type's grants")
}

func (s *AppModeSessionsSuite) TestAPersistedTextSessionIntoAnAppWithoutTheChannelTypeIsRefused() {
	own := s.giveApp(s.customerID(), 4242, "own-key")
	own.SetApp(chattest.App{ID: 4242, CallTypes: []string{"agent"}})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions", textSession(nil))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "no agent channel type")
}

func (s *AppModeSessionsSuite) TestAResumedLegacyConversationAsksForAFork() {
	// Begun in the shared app before the customer registered its own; once the fallback
	// is off it is read there and not added to.
	created := s.serverClient.createSession(textSession(nil))
	s.Require().NotNil(created.ConversationId)
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodPost,
		"/v1/agents/sessions/"+created.Id+"/stop", nil, nil))
	s.giveApp(s.customerID(), 4242, "own-key")
	s.setApps(s.customerID(), func(apps *suiteApps) { apps.readOnly[s.customerID()] = true })

	resumed := textSession(nil)
	resumed.ConversationId = created.ConversationId
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions", resumed)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "can only be read there: fork it")
}

func (s *AppModeSessionsSuite) TestAVoiceCallOnAConversationKeptInAnotherAppIsRefused() {
	// The conversation is in the customer's own app; the call rings a number attached in
	// the shared one. The agent would join the call there and look for the conversation's
	// channel where it is not.
	s.giveApp(s.customerID(), 4242, "own-key")
	created := s.serverClient.createSession(textSession(nil))
	s.Require().NotNil(created.ConversationId)
	call := s.attachedIn(0)

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/sessions",
		CreateSessionRequest{CallId: &call, ConversationId: created.ConversationId})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "kept in another Stream app")
}

func (s *AppModeSessionsSuite) TestDisconnectingEndsSessionsPinnedToTheApp() {
	// What the router no longer acts in, nothing it holds goes on acting in.
	s.giveApp(s.customerID(), 4242, "own-key")
	s.serverClient.createSession(textSession(nil))
	s.setApps(s.customerID(), func(apps *suiteApps) { delete(apps.own, s.customerID()) })
	s.serverClient.createSession(textSession(nil))

	s.Equal(1, s.manager.EndPinned(s.customerID(), 4242))
	s.Zero(s.manager.EndPinned(s.customerID(), 4242), "it ended already")
	s.Equal(1, s.manager.EndPinned(s.customerID(), 0), "the session in the deployment's app was left running")
}
