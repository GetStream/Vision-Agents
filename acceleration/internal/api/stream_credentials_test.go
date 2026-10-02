//go:build integration

package api

import (
	"context"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// StreamCredentialsSuite registers apps' own Stream apps on a router in app mode. Every app
// the suite makes has an id that is a Stream app id, and the suite's Stream answers as that
// app, so a key checked with Stream belongs to the caller unless a test says otherwise.
type StreamCredentialsSuite struct {
	RouterSuite
}

func TestStreamCredentialsSuite(t *testing.T) {
	runSuite(t, new(StreamCredentialsSuite))
}

func (s *StreamCredentialsSuite) SetupSuite() {
	s.appMode, s.trustAPIKeyHeader = true, true
	s.denied = []string{strconv.FormatInt(streamAppID(), 10)}
	s.RouterSuite.SetupSuite()
}

func (s *StreamCredentialsSuite) SetupTest() {
	s.useApp(s.numberedApp(streamAppID()))
	s.answerAs(s.appID())
}

// streamAppID is an app id nothing else in the database has.
func streamAppID() int64 { return 1_000_000_000 + rand.Int64N(8_000_000_000) }

// numberedApp is an app of its own whose id is a Stream app id.
func (s *RouterSuite) numberedApp(id int64) testApp {
	ctx := context.Background()
	organization := store.Organization{Name: "organization-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateOrganization(ctx, &organization))
	app := store.App{ID: strconv.FormatInt(id, 10), OrganizationID: organization.ID, Name: "app-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateApp(ctx, &app))
	return s.data.keyed(organization, app)
}

func (s *RouterSuite) appID() int64 {
	id, err := strconv.ParseInt(s.customerID(), 10, 64)
	s.Require().NoError(err)
	return id
}

// answerAs makes the suite's Stream say it is app id, set up as the router needs.
func (s *RouterSuite) answerAs(id int64) {
	s.chat.SetApp(chattest.App{
		ID:           id,
		ChannelTypes: map[string]map[string][]string{"agent": {"channel_member": {"read-channel", "create-message"}}},
		CallTypes:    []string{"agent"},
	})
}

type keyInput = map[string]any

func key(apiKey, secret string) keyInput { return keyInput{"api_key": apiKey, "api_secret": secret} }

func (s *RouterSuite) put(body map[string]any) (int, string) {
	status, answered := s.serverClient.call(http.MethodPut, "/v1/settings/app/stream/credentials", body)
	return status, string(answered)
}

// registered registers the keys given, the first primary, against the revision given.
func (s *RouterSuite) registered(revision int64, keys ...keyInput) StreamSettings {
	var settings AppSettings
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/settings/app/stream/credentials",
		map[string]any{"keys": keys, "expected_revision": revision}, &settings))
	return settings.Stream
}

func (s *StreamCredentialsSuite) TestRegisteringProvesTheKeysBelongToTheCaller() {
	stream := s.registered(0, key("own-key-"+s.utils.uuid(), "a-long-stream-secret"))

	s.Require().NotNil(stream.State)
	s.Equal(StreamAppState(store.StreamAppConnected), *stream.State)
	s.Equal(s.appID(), *stream.StreamAppID)
	s.Equal(int64(1), *stream.Revision)
	s.Equal(WritesIntoThisApp, stream.WritesInto)
	s.Require().Len(stream.Keys, 1)
	s.Equal("cret", stream.Keys[0].SecretLast4)
	s.False(*stream.AllowGuests, "a registered app mints no guests until it says so")
}

func (s *StreamCredentialsSuite) TestAnotherAppsKeysAreRefused() {
	s.answerAs(s.appID() + 1)

	status, failure := s.put(map[string]any{"keys": []keyInput{key("other-key", "a-long-stream-secret")}, "expected_revision": 0})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "belongs to another Stream app")
}

func (s *StreamCredentialsSuite) TestAnAppWithoutAuthChecksIsRefused() {
	s.chat.SetApp(chattest.App{ID: s.appID(), DisableAuthChecks: true})

	status, failure := s.put(map[string]any{"keys": []keyInput{key("own-key", "a-long-stream-secret")}, "expected_revision": 0})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "does not check the tokens")
}

func (s *StreamCredentialsSuite) TestASuspendedAppIsRefused() {
	s.chat.SetApp(chattest.App{ID: s.appID(), Suspended: true})

	status, failure := s.put(map[string]any{"keys": []keyInput{key("own-key", "a-long-stream-secret")}, "expected_revision": 0})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "suspended")
}

func (s *StreamCredentialsSuite) TestADeniedAppCannotRegister() {
	denied, err := strconv.ParseInt(s.denied[0], 10, 64)
	s.Require().NoError(err)
	s.useApp(s.numberedApp(denied))
	s.answerAs(denied)

	status, _ := s.put(map[string]any{"keys": []keyInput{key("own-key", "a-long-stream-secret")}, "expected_revision": 0})

	s.Equal(http.StatusForbidden, status)
}

func (s *StreamCredentialsSuite) TestNoResponseEverCarriesASecret() {
	secret := "a-secret-nobody-should-read-" + s.utils.uuid()
	apiKey := "own-key-" + s.utils.uuid()
	status, registered := s.put(map[string]any{"keys": []keyInput{key(apiKey, secret)}, "expected_revision": 0})
	s.Require().Equal(http.StatusOK, status)
	_, read := s.serverClient.call(http.MethodGet, "/v1/settings/app", nil)
	_, checked := s.serverClient.call(http.MethodPost, "/v1/settings/app/stream/check", map[string]any{})
	_, conflict := s.put(map[string]any{"keys": []keyInput{key(apiKey, secret)}, "expected_revision": 0})

	for _, answer := range []string{registered, string(read), string(checked)} {
		s.NotContains(answer, secret)
		s.Contains(answer, apiKey, "the key itself is named")
	}
	s.NotContains(conflict, secret)
}

func (s *StreamCredentialsSuite) TestARejectedSecretIsNeverEchoed() {
	tooLong := "S3CRET" + strings.Repeat("x", 300)
	for name, secret := range map[string]any{"over-long": tooLong, "not a string": 918273645546372} {
		s.Run(name, func() {
			status, failure := s.put(map[string]any{
				"keys": []keyInput{{"api_key": "own-key", "api_secret": secret}}, "expected_revision": 0,
			})

			s.Equal(http.StatusBadRequest, status)
			s.NotContains(failure, "S3CRET")
			s.NotContains(failure, "918273645546372")
		})
	}
}

func (s *StreamCredentialsSuite) TestDisconnectingNeedsAProof() {
	apiKey := "own-key-" + s.utils.uuid()
	s.registered(0, key(apiKey, "a-long-stream-secret"))

	status, failure := s.put(map[string]any{"keys": []keyInput{}, "expected_revision": 1})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "proof")

	var settings AppSettings
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/settings/app/stream/credentials",
		map[string]any{"keys": []keyInput{}, "expected_revision": 1, "proof": key(apiKey, "a-long-stream-secret")}, &settings))
	s.Equal(StreamAppState(store.StreamAppDisconnected), *settings.Stream.State)
	s.Empty(settings.Stream.Keys)
	s.Equal(WritesIntoNowhere, settings.Stream.WritesInto, "a disconnected app never falls back")
}

func (s *StreamCredentialsSuite) TestAStaleRevisionIsAConflict() {
	s.registered(0, key("own-key-"+s.utils.uuid(), "a-long-stream-secret"))

	status, _ := s.put(map[string]any{"keys": []keyInput{key("own-key-"+s.utils.uuid(), "a-long-stream-secret")}, "expected_revision": 0})

	s.Equal(http.StatusConflict, status)
}

func (s *StreamCredentialsSuite) TestCheckReportsAMissingAgentCallType() {
	s.registered(0, key("own-key-"+s.utils.uuid(), "a-long-stream-secret"))
	s.chat.SetApp(chattest.App{
		ID: s.appID(), ChannelTypes: map[string]map[string][]string{"agent": {"channel_member": {"read-channel"}}},
	})

	var checked StreamCheck
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/settings/app/stream/check", map[string]any{}, &checked))

	s.Equal(StreamTypeState("missing"), checked.Settings.Stream.CallType)
	s.NotNil(checked.Reattach)
}

func (s *StreamCredentialsSuite) TestStreamCredentialsAreServerSideOnly() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPut, "/v1/settings/app/stream/credentials",
			map[string]any{"keys": []keyInput{key("own-key-"+s.utils.uuid(), "a-long-stream-secret")}, "expected_revision": 0}, nil)
	})
}

func (s *StreamCredentialsSuite) TestTheCallersKeyMintsItsTokens() {
	primary, secondary := "primary-"+s.utils.uuid(), "secondary-"+s.utils.uuid()
	s.registered(0, key(primary, "a-long-stream-secret"), key(secondary, "another-long-secret"))

	s.Equal(secondary, s.chatTokenKey(secondary))
	s.Equal(primary, s.chatTokenKey(""))
}

func (s *StreamCredentialsSuite) TestAnotherAppsPublicKeyNeverChoosesTheMintingKey() {
	primary := "primary-" + s.utils.uuid()
	s.registered(0, key(primary, "a-long-stream-secret"))

	s.Equal(primary, s.chatTokenKey(suiteStreamKey))
	s.Equal(primary, s.chatTokenKey("nobody's-key"))
}

func (s *StreamCredentialsSuite) TestDroppingAKeyEndsTheSessionsActingInTheApp() {
	first, second := "first-"+s.utils.uuid(), "second-"+s.utils.uuid()
	s.registered(0, key(first, "a-long-stream-secret"), key(second, "another-long-secret"))
	s.serverClient.createSession(textSession(nil))

	s.registered(1, key(second, "another-long-secret"))

	s.Zero(s.manager.EndPinned(s.customerID(), s.appID()), "the session ended with the key")
}

// chatTokenKey is the key a chat token is minted with when the gateway names apiKey.
func (s *StreamCredentialsSuite) chatTokenKey(apiKey string) string {
	client := *s.serverClient
	client.header = s.serverClient.header.Clone()
	if apiKey != "" {
		client.header.Set(mintingKeyHeader, apiKey)
	}
	var minted ChatToken
	s.Require().Equal(http.StatusOK, client.do(http.MethodPost, "/v1/agents/chat-token",
		map[string]any{"agent_id": "agent-" + s.utils.uuid()}, &minted))
	return minted.ApiKey
}

func (s *StreamAppsSuite) TestStreamCredentialsAreRefusedInDeploymentMode() {
	status, failure := s.serverClient.failure(http.MethodPut, "/v1/settings/app/stream/credentials",
		map[string]any{"keys": []keyInput{key("own-key", "a-long-stream-secret")}, "expected_revision": 0})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "stream.tenancy=app")
}

func (s *StreamCredentialsSuite) TestAMalformedBodyNeverEchoesTheSecretsInIt() {
	// A validation error names the value it refused, and for a missing or unexpected field
	// that value is the whole object around it, secrets and all.
	secret := "a-secret-in-a-bad-body-" + s.utils.uuid()
	many := make([]keyInput, 9)
	for i := range many {
		many[i] = key("key-"+strconv.Itoa(i), secret)
	}
	for name, body := range map[string]map[string]any{
		"no revision":     {"keys": []keyInput{key("own-key", secret)}},
		"no api key":      {"keys": []keyInput{{"api_secret": secret}}, "expected_revision": 0},
		"too many keys":   {"keys": many, "expected_revision": 0},
		"a mistyped name": {"keys": []keyInput{key("own-key", secret)}, "expected_revision": "zero"},
	} {
		s.Run(name, func() {
			status, failure := s.put(body)

			s.Equal(http.StatusBadRequest, status)
			s.NotContains(failure, secret)
		})
	}
}

func (s *StreamCredentialsSuite) TestADisconnectedAppIsToldItMintsNoGuests() {
	apiKey := "own-key-" + s.utils.uuid()
	s.registered(0, key(apiKey, "a-long-stream-secret"))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/settings/app/stream/credentials",
		map[string]any{"keys": []keyInput{}, "expected_revision": 1, "proof": key(apiKey, "a-long-stream-secret")}, nil))

	status, failure := s.anonymousClient.failure(http.MethodPost, "/v1/agents/guests", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "disconnected")
}
