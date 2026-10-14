//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
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
	s.Contains(failure, "call through the gateway with this app's own Stream keys")
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

func (s *AppModeSessionsSuite) TestAVoiceSessionIntoAnAppWithoutTheCallTypeIsRefused() {
	// Stream used to refuse the call itself, saying only that the call type does not exist.
	own := s.giveApp(s.customerID(), 4242, "own-key")
	own.SetApp(chattest.App{ID: 4242, ChannelTypes: map[string]map[string][]string{"agent": {}}})

	status, failure := s.errorOf(http.MethodPost, "/v1/agents/sessions", CreateSessionRequest{StartVoice: pointerTo(true)})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeNotConfigured, failure.Code)
	s.Contains(failure.Message, `getstream api CreateCallType --request '{"name":"agent"}'`)
	// The check fires before anything is written to the customer's app: no orphan channel.
	for _, r := range own.Requests("own-key") {
		s.False(r.Writes(), "a refused voice create must not write to the customer's app (got %s %s)", r.Method, r.Path)
	}
}

func (s *AppModeSessionsSuite) TestStartingVoiceInAnAppWithoutTheCallTypeIsRefused() {
	own := s.giveApp(s.customerID(), 4242, "own-key")
	own.SetApp(chattest.App{ID: 4242, ChannelTypes: map[string]map[string][]string{"agent": {}}})
	opened := s.serverClient.createSession(textSession(nil))

	status, failure := s.errorOf(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/voice", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Equal(codeNotConfigured, failure.Code)
	s.Contains(failure.Message, `getstream api CreateCallType --request '{"name":"agent"}'`)
	s.Empty(s.serverClient.getSession(opened.Id).CallId, "the conversation carries on in writing")
}

func (s *AppModeSessionsSuite) TestAVoiceSessionIntoAnAppWithTheCallTypeJoinsIt() {
	s.giveApp(s.customerID(), 4242, "own-key")

	opened := s.serverClient.createSession(CreateSessionRequest{StartVoice: pointerTo(true)})

	s.Equal(opened.Id, opened.CallId)
}

func (s *AppModeSessionsSuite) TestAVoiceSessionIntoAnAppNobodyCouldAskIsNotRefusedForTheCallType() {
	// Only Stream saying the call type is missing refuses: an app that could not be read is
	// left to the call, as it was before the check.
	closed := httptest.NewServer(nil)
	closed.Close()
	s.setApps(s.customerID(), func(apps *suiteApps) {
		apps.own[s.customerID()] = streamapp.Identity{
			CustomerID: s.customerID(), StreamApp: 4242, APIKey: "own-key",
			Secret: streamapp.NewSecret("own-key-secret"), BaseURL: closed.URL, Registered: true,
		}
	})
	incognito := CreateSessionRequest{StartVoice: pointerTo(true), Incognito: pointerTo(true)}

	opened := s.serverClient.createSession(incognito)

	s.Equal(opened.Id, opened.CallId)
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

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/sessions/"+created.Id+"/respond", SayRequest{Text: "still there?"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "can only be read there: fork it")
}

func (s *AppModeSessionsSuite) TestAVoiceSessionForAnUnregisteredAppNeedsNoAgentCallType() {
	// MC: the callable check is skipped when the identity is not Registered (deployment
	// mode acting in a customer's own app key without a forwarded registration), so the
	// session is not refused even if the app has no agent call type.
	own := chattest.NewServer(s.T())
	own.SetApp(chattest.App{ID: 4242, ChannelTypes: map[string]map[string][]string{"agent": {}}})
	s.setApps(s.customerID(), func(apps *suiteApps) {
		apps.own[s.customerID()] = streamapp.Identity{
			CustomerID: s.customerID(), StreamApp: 4242, APIKey: "own-key",
			Secret: streamapp.NewSecret("own-key-secret"), BaseURL: own.URL,
			Registered: false,
		}
	})
	incognito := CreateSessionRequest{StartVoice: pointerTo(true), Incognito: pointerTo(true)}

	opened := s.serverClient.createSession(incognito)

	s.Equal(opened.Id, opened.CallId)
}

func (s *AppModeSessionsSuite) TestASocketVoiceSessionNeedsNoAgentCallType() {
	// ME: socket sessions carry their own edge (spec.Edge != nil), so callable is skipped
	// even when the customer's Registered app has no agent call type.
	own := s.giveApp(s.customerID(), 4242, "own-key")
	own.SetApp(chattest.App{ID: 4242, ChannelTypes: map[string]map[string][]string{"agent": {}}})

	connection := s.serverClient.opens("/v1/agents/socket")
	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type": "start", "sample_rate": 24_000,
		"session": map[string]any{"incognito": true},
	}))
	answered := s.await(connection, "session")

	s.NotEmpty(answered["session"].(map[string]any)["id"])
}

// errorOf makes a request and returns the failure it was answered with.
func (s *AppModeSessionsSuite) errorOf(method, path string, body any) (int, ErrorDetail) {
	status, payload := s.serverClient.call(method, path, body)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	return status, answered.Error
}
