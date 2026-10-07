package fakeprovider_test

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/hmacheader"
)

// The suite is an outside package on purpose: it uses only what another package's test can.
type FakeProviderSuite struct {
	suite.Suite
}

func TestFakeProviderSuite(t *testing.T) {
	suite.Run(t, new(FakeProviderSuite))
}

func (s *FakeProviderSuite) TestACodeIsExchangedOnlyWithItsPKCEVerifier() {
	srv := fakeprovider.New(s.T())
	authorize, verifier := s.authorizeURL(srv, nil)
	callback := s.consent(srv, authorize)

	status, body := s.exchange(srv, callback.Query().Get("code"), "another-verifier-of-enough-length-to-look-real")
	s.Equal(http.StatusBadRequest, status)
	s.Equal("invalid_grant", body["error"])

	status, body = s.exchange(srv, callback.Query().Get("code"), verifier)
	s.Equal(http.StatusOK, status)
	s.NotEmpty(body["access_token"])
	s.NotEmpty(body["refresh_token"])
}

func (s *FakeProviderSuite) TestAnAuthorizationWithoutAnS256ChallengeIsRefused() {
	srv := fakeprovider.New(s.T())
	missing, _ := s.authorizeURL(srv, url.Values{"code_challenge": {""}})
	s.Equal("invalid_request", s.consent(srv, missing).Query().Get("error"))

	plain, _ := s.authorizeURL(srv, url.Values{"code_challenge_method": {"plain"}})
	s.Equal("invalid_request", s.consent(srv, plain).Query().Get("error"))
}

func (s *FakeProviderSuite) TestTheCallbackNamesTheIssuerFromTheMetadata() {
	srv := fakeprovider.New(s.T())
	metadata := s.getJSON(srv, fakeprovider.PathAuthorizationServer)
	s.Equal(true, metadata["authorization_response_iss_parameter_supported"])

	authorize, _ := s.authorizeURL(srv, nil)
	s.Equal(metadata["issuer"], s.consent(srv, authorize).Query().Get("iss"))
}

func (s *FakeProviderSuite) TestForeignIssuerNamesAnotherIssuerInTheCallback() {
	srv := fakeprovider.New(s.T(), fakeprovider.ForeignIssuer)
	authorize, _ := s.authorizeURL(srv, nil)
	callback := s.consent(srv, authorize)
	s.Equal(fakeprovider.ForeignIssuerURL, callback.Query().Get("iss"))
	s.NotEqual(srv.URL, callback.Query().Get("iss"))
}

func (s *FakeProviderSuite) TestConsentDeniedSendsTheBrowserBackWithAccessDenied() {
	srv := fakeprovider.New(s.T(), fakeprovider.ConsentDenied)
	authorize, _ := s.authorizeURL(srv, url.Values{"state": {"kept-state"}})
	callback := s.consent(srv, authorize)
	s.Equal("access_denied", callback.Query().Get("error"))
	s.Equal("kept-state", callback.Query().Get("state"))
	s.Empty(callback.Query().Get("code"))
}

func (s *FakeProviderSuite) TestACodeRedeemedTwiceRevokesItsGrant() {
	srv := fakeprovider.New(s.T())
	authorize, verifier := s.authorizeURL(srv, nil)
	code := s.consent(srv, authorize).Query().Get("code")
	_, first := s.exchange(srv, code, verifier)

	status, body := s.exchange(srv, code, verifier)
	s.Equal(http.StatusBadRequest, status)
	s.Equal("invalid_grant", body["error"])
	s.Equal(http.StatusUnauthorized, s.call(srv, first["access_token"].(string)).StatusCode)
}

func (s *FakeProviderSuite) TestACodeRedeemedAgainAfterItExpiredStillRevokesItsGrant() {
	srv := fakeprovider.New(s.T())
	authorize, verifier := s.authorizeURL(srv, nil)
	code := s.consent(srv, authorize).Query().Get("code")
	_, first := s.exchange(srv, code, verifier)

	srv.Advance(11 * time.Minute) // past codeTTL, RFC 6749 §4.1.2's 10 minutes
	status, body := s.exchange(srv, code, verifier)
	s.Equal(http.StatusBadRequest, status)
	s.Equal("invalid_grant", body["error"])
	s.Equal(http.StatusUnauthorized, s.call(srv, first["access_token"].(string)).StatusCode)
}

func (s *FakeProviderSuite) TestADynamicallyRegisteredPublicClientCanConnect() {
	srv := fakeprovider.New(s.T())
	response, err := srv.Client().Post(srv.URL+fakeprovider.PathRegister, "application/json",
		strings.NewReader(`{"redirect_uris":["https://dcr.example/callback"],"token_endpoint_auth_method":"none"}`))
	s.Require().NoError(err)
	defer response.Body.Close()
	s.Require().Equal(http.StatusCreated, response.StatusCode)
	var registered map[string]any
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&registered))
	s.NotContains(registered, "client_secret")

	clientID := registered["client_id"].(string)
	verifier := "dcr-verifier-dcr-verifier-dcr-verifier-dcr-verifier"
	authorize := srv.URL + fakeprovider.PathAuthorize + "?" + url.Values{
		"response_type": {"code"}, "client_id": {clientID}, "redirect_uri": {"https://dcr.example/callback"},
		"code_challenge": {challenge(verifier)}, "code_challenge_method": {"S256"},
	}.Encode()
	code := s.consent(srv, authorize).Query().Get("code")
	status, body := s.post(srv, fakeprovider.PathToken, url.Values{
		"grant_type": {"authorization_code"}, "code": {code}, "client_id": {clientID},
		"redirect_uri": {"https://dcr.example/callback"}, "code_verifier": {verifier},
	}, false)
	s.Equal(http.StatusOK, status)
	s.NotEmpty(body["access_token"])
}

func (s *FakeProviderSuite) TestAClientMetadataDocumentClientConnectsWithItsURLAsClientID() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientMetadataDocuments)
	s.Equal(true, s.getJSON(srv, fakeprovider.PathAuthorizationServer)["client_id_metadata_document_supported"])
	clientID, fetches := s.serveClientMetadata(srv, func(clientID string) map[string]any {
		return map[string]any{"client_id": clientID, "client_name": "Test", "redirect_uris": []string{"https://cimd.example/callback"}, "token_endpoint_auth_method": "none"}
	})

	verifier := "cimd-verifier-cimd-verifier-cimd-verifier-cimd-verifier"
	authorize := srv.URL + fakeprovider.PathAuthorize + "?" + url.Values{
		"response_type": {"code"}, "client_id": {clientID}, "redirect_uri": {"https://cimd.example/callback"},
		"code_challenge": {challenge(verifier)}, "code_challenge_method": {"S256"},
	}.Encode()
	code := s.consent(srv, authorize).Query().Get("code")
	s.Equal(1, *fetches, "the server fetched the document at the client_id URL")
	status, body := s.post(srv, fakeprovider.PathToken, url.Values{
		"grant_type": {"authorization_code"}, "code": {code}, "client_id": {clientID},
		"redirect_uri": {"https://cimd.example/callback"}, "code_verifier": {verifier},
	}, false)
	s.Equal(http.StatusOK, status)
	s.NotEmpty(body["access_token"])
	s.Equal(0, srv.Hits(fakeprovider.PathRegister), "no dynamic registration")
}

func (s *FakeProviderSuite) TestAClientMetadataDocumentThatNamesAnotherClientIsRefused() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientMetadataDocuments)
	clientID, _ := s.serveClientMetadata(srv, func(string) map[string]any {
		return map[string]any{"client_id": "https://elsewhere.example/client", "client_name": "Test", "redirect_uris": []string{"https://cimd.example/callback"}}
	})
	_, err := srv.Consent(srv.URL + fakeprovider.PathAuthorize + "?" + url.Values{
		"response_type": {"code"}, "client_id": {clientID}, "redirect_uri": {"https://cimd.example/callback"},
		"code_challenge": {challenge("v")}, "code_challenge_method": {"S256"},
	}.Encode())
	s.Error(err, "no redirect for a client the document does not describe")
}

func (s *FakeProviderSuite) TestAClientMetadataDocumentWithASecretIsRefused() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientMetadataDocuments)
	clientID, _ := s.serveClientMetadata(srv, func(clientID string) map[string]any {
		return map[string]any{"client_id": clientID, "client_name": "Test", "redirect_uris": []string{"https://cimd.example/callback"}, "token_endpoint_auth_method": "client_secret_post"}
	})
	_, err := srv.Consent(srv.URL + fakeprovider.PathAuthorize + "?" + url.Values{
		"response_type": {"code"}, "client_id": {clientID}, "redirect_uri": {"https://cimd.example/callback"},
		"code_challenge": {challenge("v")}, "code_challenge_method": {"S256"},
	}.Encode())
	s.Error(err)
}

func (s *FakeProviderSuite) TestWithoutClientMetadataDocumentsAURLClientIsUnknown() {
	srv := fakeprovider.New(s.T())
	s.NotContains(s.getJSON(srv, fakeprovider.PathAuthorizationServer), "client_id_metadata_document_supported")
	clientID, fetches := s.serveClientMetadata(srv, func(clientID string) map[string]any {
		return map[string]any{"client_id": clientID, "client_name": "Test", "redirect_uris": []string{"https://cimd.example/callback"}}
	})
	_, err := srv.Consent(srv.URL + fakeprovider.PathAuthorize + "?" + url.Values{
		"response_type": {"code"}, "client_id": {clientID}, "redirect_uri": {"https://cimd.example/callback"},
		"code_challenge": {challenge("v")}, "code_challenge_method": {"S256"},
	}.Encode())
	s.Error(err)
	s.Equal(0, *fetches)
}

func (s *FakeProviderSuite) TestStrictRotationRefusesAReplayAndRevokesTheGrant() {
	srv := fakeprovider.New(s.T())
	first := s.connect(srv, nil)
	status, rotated := s.refresh(srv, first["refresh_token"].(string))
	s.Require().Equal(http.StatusOK, status)
	s.NotEqual(first["refresh_token"], rotated["refresh_token"])

	status, body := s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusBadRequest, status)
	s.Equal("invalid_grant", body["error"])
	status, _ = s.refresh(srv, rotated["refresh_token"].(string))
	s.Equal(http.StatusBadRequest, status, "a replay revokes the active refresh token too")
}

func (s *FakeProviderSuite) TestRotatingRefreshWithGraceAcceptsTheOldTokenUntilTheWindowCloses() {
	srv := fakeprovider.New(s.T(), fakeprovider.RotatingRefreshWithGrace)
	first := s.connect(srv, nil)
	old := first["refresh_token"].(string)
	status, rotated := s.refresh(srv, old)
	s.Require().Equal(http.StatusOK, status)
	s.NotEqual(old, rotated["refresh_token"])

	status, _ = s.refresh(srv, old)
	s.Equal(http.StatusOK, status, "inside the grace window the old token still works")

	srv.Advance(fakeprovider.Grace)
	status, body := s.refresh(srv, old)
	s.Equal(http.StatusBadRequest, status)
	s.Equal("invalid_grant", body["error"])
}

func (s *FakeProviderSuite) TestAnExpiredGraceTokenIsRefusedAndTheRotatedOneKeepsWorking() {
	srv := fakeprovider.New(s.T(), fakeprovider.RotatingRefreshWithGrace)
	first := s.connect(srv, nil)
	old := first["refresh_token"].(string)
	status, second := s.refresh(srv, old)
	s.Require().Equal(http.StatusOK, status)

	srv.Advance(fakeprovider.Grace)
	status, third := s.refresh(srv, second["refresh_token"].(string))
	s.Require().Equal(http.StatusOK, status)
	status, _ = s.refresh(srv, old)
	s.Equal(http.StatusBadRequest, status, "past the window the old token is refused")

	status, _ = s.refresh(srv, third["refresh_token"].(string))
	s.Equal(http.StatusOK, status, "an expired old token does not end the grant")
	s.Equal(http.StatusOK, s.call(srv, third["access_token"].(string)).StatusCode)
}

func (s *FakeProviderSuite) TestNonRotatingRefreshKeepsTheSameRefreshToken() {
	srv := fakeprovider.New(s.T(), fakeprovider.NonRotatingRefresh)
	first := s.connect(srv, nil)
	for range 3 {
		status, body := s.refresh(srv, first["refresh_token"].(string))
		s.Require().Equal(http.StatusOK, status)
		s.NotContains(body, "refresh_token")
		s.NotEqual(first["access_token"], body["access_token"])
	}
}

func (s *FakeProviderSuite) TestNoRefreshTokenLeavesAnAccessTokenThatExpires() {
	srv := fakeprovider.New(s.T(), fakeprovider.NoRefreshToken)
	first := s.connect(srv, nil)
	s.NotContains(first, "refresh_token")
	access := first["access_token"].(string)
	s.Equal(http.StatusOK, s.call(srv, access).StatusCode)

	srv.Advance(fakeprovider.AccessTTL)
	response := s.call(srv, access)
	s.Equal(http.StatusUnauthorized, response.StatusCode)
	s.Contains(response.Header.Get("WWW-Authenticate"), `error="invalid_token"`)
}

func (s *FakeProviderSuite) TestBareChallengeRefusesATokenWithResourceMetadataAndNoError() {
	srv := fakeprovider.New(s.T(), fakeprovider.BareChallenge)
	access := s.connect(srv, nil)["access_token"].(string)
	s.Equal(http.StatusOK, s.call(srv, access).StatusCode)

	srv.Advance(fakeprovider.AccessTTL)
	for _, token := range []string{access, "a-token-nobody-issued"} {
		response := s.call(srv, token)
		s.Equal(http.StatusUnauthorized, response.StatusCode)
		s.Equal(`Bearer resource_metadata="`+srv.URL+fakeprovider.PathProtectedResource+`"`, response.Header.Get("WWW-Authenticate"))
	}
}

func (s *FakeProviderSuite) TestInvalidGrantRefusesEveryRefresh() {
	srv := fakeprovider.New(s.T(), fakeprovider.InvalidGrant)
	first := s.connect(srv, nil)
	status, body := s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusBadRequest, status)
	s.Equal("invalid_grant", body["error"])
	s.Equal(http.StatusOK, s.call(srv, first["access_token"].(string)).StatusCode,
		"only the refresh is refused; the access token works until it expires")
}

func (s *FakeProviderSuite) TestLostResponseSpendsTheRefreshTokenAndAnswersNothing() {
	srv := fakeprovider.New(s.T())
	first := s.connect(srv, nil)
	srv.Use(fakeprovider.LostResponse)

	_, err := srv.Client().PostForm(srv.URL+fakeprovider.PathToken, url.Values{
		"grant_type": {"refresh_token"}, "refresh_token": {first["refresh_token"].(string)},
		"client_id": {srv.ClientID}, "client_secret": {srv.ClientSecret},
	})
	s.Require().Error(err, "the connection closes before any response")
	s.Equal(1, srv.Refreshes())

	srv.Use()
	status, body := s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusBadRequest, status, "the refresh was committed, so the old token is spent")
	s.Equal("invalid_grant", body["error"])
}

func (s *FakeProviderSuite) TestUnavailableAnswers503AndLeavesTheGrantAlone() {
	srv := fakeprovider.New(s.T())
	first := s.connect(srv, nil)
	srv.Use(fakeprovider.Unavailable)
	status, _ := s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusServiceUnavailable, status)

	srv.Use()
	status, _ = s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusOK, status)
}

func (s *FakeProviderSuite) TestInsufficientScopeAsksForTheScopeAndAStepUpConsentPasses() {
	srv := fakeprovider.New(s.T(), fakeprovider.InsufficientScope)
	narrow := s.connect(srv, url.Values{"scope": {"files:read"}})
	response := s.call(srv, narrow["access_token"].(string))
	s.Equal(http.StatusForbidden, response.StatusCode)
	challenge := response.Header.Get("WWW-Authenticate")
	s.Contains(challenge, `error="insufficient_scope"`)
	s.Contains(challenge, `scope="files:read `+fakeprovider.RequiredScope+`"`)
	s.Contains(challenge, `resource_metadata="`+srv.URL+fakeprovider.PathProtectedResource+`"`)

	wide := s.connect(srv, url.Values{"scope": {"files:read " + fakeprovider.RequiredScope}})
	s.Equal(http.StatusOK, s.call(srv, wide["access_token"].(string)).StatusCode)
}

func (s *FakeProviderSuite) TestClaimsChallengeAnswers401WithClaimsAndAConsentWithThemPasses() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClaimsChallenge)
	first := s.connect(srv, nil)
	response := s.call(srv, first["access_token"].(string))
	s.Equal(http.StatusUnauthorized, response.StatusCode)
	challenge := response.Header.Get("WWW-Authenticate")
	s.Contains(challenge, `error="insufficient_claims"`)
	_, encoded, found := strings.Cut(challenge, `claims="`)
	s.Require().True(found)
	claims, err := base64.StdEncoding.DecodeString(strings.TrimSuffix(encoded, `"`))
	s.Require().NoError(err)
	s.JSONEq(fakeprovider.ClaimsChallengeJSON, string(claims))

	// As Microsoft's page says: decode the base64, pass it back as the claims parameter.
	stepped := s.connect(srv, url.Values{"claims": {string(claims)}})
	s.Equal(http.StatusOK, s.call(srv, stepped["access_token"].(string)).StatusCode)
}

func (s *FakeProviderSuite) TestRateLimitedAnswers429WithRetryAfter() {
	srv := fakeprovider.New(s.T())
	first := s.connect(srv, nil)
	srv.Use(fakeprovider.RateLimited)
	response := s.call(srv, first["access_token"].(string))
	s.Equal(http.StatusTooManyRequests, response.StatusCode)
	s.Equal("30", response.Header.Get("Retry-After"))
}

func (s *FakeProviderSuite) TestRateLimitedRefusesARefreshWith429AndSpendsNothing() {
	srv := fakeprovider.New(s.T())
	first := s.connect(srv, nil)
	srv.Use(fakeprovider.RateLimited)
	request, err := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathToken, strings.NewReader(url.Values{
		"grant_type": {"refresh_token"}, "refresh_token": {first["refresh_token"].(string)},
	}.Encode()))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.SetBasicAuth(srv.ClientID, srv.ClientSecret)
	response, err := srv.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Equal(http.StatusTooManyRequests, response.StatusCode)
	s.Equal("30", response.Header.Get("Retry-After"))

	srv.Use()
	status, _ := s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusOK, status, "the refused refresh spent nothing")
}

func (s *FakeProviderSuite) TestLostResponseWithGraceLeavesTheSpentTokenUsableInTheWindow() {
	srv := fakeprovider.New(s.T())
	first := s.connect(srv, nil)
	srv.Use(fakeprovider.LostResponse, fakeprovider.RotatingRefreshWithGrace)
	s.Require().Error(s.lostRefresh(srv, first["refresh_token"].(string)))
	s.Require().Error(s.lostRefresh(srv, first["refresh_token"].(string)), "every refresh is lost")
	s.Equal(2, srv.Refreshes())

	srv.Use(fakeprovider.RotatingRefreshWithGrace)
	status, _ := s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusOK, status, "inside the grace window the spent token still works")
}

func (s *FakeProviderSuite) TestLostResponseOnceDropsOneAnswerAndAnswersTheNext() {
	srv := fakeprovider.New(s.T())
	first := s.connect(srv, nil)
	srv.Use(fakeprovider.LostResponseOnce, fakeprovider.RotatingRefreshWithGrace)
	s.Require().Error(s.lostRefresh(srv, first["refresh_token"].(string)))

	status, body := s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusOK, status, "the retry inside the window is answered")
	s.NotEqual(first["refresh_token"], body["refresh_token"])
	s.Equal(2, srv.Refreshes())
}

func (s *FakeProviderSuite) TestServerErrorAnswers500AfterTheRotationWasCommitted() {
	srv := fakeprovider.New(s.T())
	first := s.connect(srv, nil)
	srv.Use(fakeprovider.ServerError)
	status, body := s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusInternalServerError, status)
	s.Equal("server_error", body["error"])

	srv.Use()
	status, body = s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusBadRequest, status, "the refresh took effect, so the old token is spent")
	s.Equal("invalid_grant", body["error"])
}

func (s *FakeProviderSuite) TestServerErrorUnderCommaScopesIsSlacksInternalError() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	first := s.connect(srv, nil)
	srv.Use(fakeprovider.CommaScopes, fakeprovider.ServerError)
	status, body := s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusOK, status)
	s.Equal(false, body["ok"])
	s.Equal("internal_error", body["error"])
}

func (s *FakeProviderSuite) TestCutOffRefusalSends400ThenLosesTheBodyAndSpendsNothing() {
	srv := fakeprovider.New(s.T())
	first := s.connect(srv, nil)
	srv.Use(fakeprovider.CutOffRefusal)
	response, err := srv.Client().PostForm(srv.URL+fakeprovider.PathToken, url.Values{
		"grant_type": {"refresh_token"}, "refresh_token": {first["refresh_token"].(string)},
		"client_id": {srv.ClientID}, "client_secret": {srv.ClientSecret},
	})
	s.Require().NoError(err, "the status line and headers arrive")
	s.Equal(http.StatusBadRequest, response.StatusCode)
	_, err = io.ReadAll(response.Body)
	s.Require().Error(err, "the body does not")
	s.Require().NoError(response.Body.Close())

	srv.Use()
	status, _ := s.refresh(srv, first["refresh_token"].(string))
	s.Equal(http.StatusOK, status, "the refused refresh spent nothing")
}

func (s *FakeProviderSuite) TestAccessTokenNotRevocableRefusesAnAccessTokenAndRevokesARefreshToken() {
	srv := fakeprovider.New(s.T(), fakeprovider.AccessTokenNotRevocable)
	first := s.connect(srv, nil)
	status, body := s.post(srv, fakeprovider.PathRevoke, url.Values{"token": {first["access_token"].(string)}}, true)
	s.Equal(http.StatusBadRequest, status)
	s.Equal("unsupported_token_type", body["error"])
	s.Equal(http.StatusOK, s.call(srv, first["access_token"].(string)).StatusCode, "nothing was revoked")

	status, _ = s.post(srv, fakeprovider.PathRevoke, url.Values{"token": {first["refresh_token"].(string)}}, true)
	s.Equal(http.StatusOK, status)
	s.Equal(http.StatusUnauthorized, s.call(srv, first["access_token"].(string)).StatusCode, "revoking the refresh token ended the grant")
}

func (s *FakeProviderSuite) TestARefreshScopeIsRecordedAndCannotWidenTheGrant() {
	srv := fakeprovider.New(s.T())
	first := s.connect(srv, url.Values{"scope": {"files:read"}})
	status, second := s.post(srv, fakeprovider.PathToken, url.Values{
		"grant_type": {"refresh_token"}, "refresh_token": {first["refresh_token"].(string)}, "scope": {"files:read"},
	}, true)
	s.Require().Equal(http.StatusOK, status)
	scope, sent := srv.RefreshScope()
	s.True(sent)
	s.Equal("files:read", scope)

	status, body := s.post(srv, fakeprovider.PathToken, url.Values{
		"grant_type": {"refresh_token"}, "refresh_token": {second["refresh_token"].(string)}, "scope": {"files:read files:write"},
	}, true)
	s.Equal(http.StatusBadRequest, status)
	s.Equal("invalid_scope", body["error"])

	status, _ = s.refresh(srv, second["refresh_token"].(string))
	s.Equal(http.StatusOK, status, "the refused widening spent nothing")
	_, sent = srv.RefreshScope()
	s.False(sent)
}

func (s *FakeProviderSuite) TestCommaScopesAnswersInSlackShapeThatTheSlackManifestCaptures() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	body := s.connect(srv, url.Values{"scope": {"channels:history,chat:write"}, "user_scope": {"search:read"}})
	s.Equal(true, body["ok"])
	s.Equal("channels:history,chat:write", body["scope"])
	user := body["authed_user"].(map[string]any)
	s.Equal("search:read", user["scope"])
	s.Equal(http.StatusOK, s.call(srv, user["access_token"].(string)).StatusCode)

	// The shape agrees with core's recorded Slack response: same top-level keys, and the
	// Slack fixture manifest captures the fake's team and user.
	var recorded map[string]any
	s.Require().NoError(json.Unmarshal(s.read("../core/testdata/recorded/slack.token.json"), &recorded))
	for key := range recorded {
		s.Contains(body, key)
	}
	account := s.apply("../core/testdata/manifests/slack.yaml", nil, body)
	s.Equal(srv.TeamID, account.Metadata["team_id"])
	s.Equal(srv.UserID, account.Metadata["user_id"])
}

func (s *FakeProviderSuite) TestClientCredentialsIssuesAnAccessTokenAloneToAnAuthenticatedClient() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	grant := url.Values{"grant_type": {"client_credentials"}}

	status, body := s.post(srv, fakeprovider.PathToken, grant, true)
	s.Equal(http.StatusOK, status)
	s.NotContains(body, "refresh_token", "RFC 6749 §4.4.3")
	s.Equal(http.StatusOK, s.call(srv, body["access_token"].(string)).StatusCode)

	status, body = s.post(srv, fakeprovider.PathToken, grant, false)
	s.Equal(http.StatusUnauthorized, status, "§4.4.2: the client must authenticate")
	s.Equal("invalid_client", body["error"])
	s.Equal(2, srv.ClientCredentialsGrants())

	srv.Use()
	status, body = s.post(srv, fakeprovider.PathToken, grant, true)
	s.Equal(http.StatusBadRequest, status)
	s.Equal("unsupported_grant_type", body["error"], "without the personality the grant is not offered")
}

func (s *FakeProviderSuite) TestClientCredentialsTokensExpireAndRevokeOneByOne() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	grant := url.Values{"grant_type": {"client_credentials"}}
	_, first := s.post(srv, fakeprovider.PathToken, grant, true)
	_, second := s.post(srv, fakeprovider.PathToken, grant, true)

	status, _ := s.post(srv, fakeprovider.PathRevoke, url.Values{"token": {first["access_token"].(string)}}, true)
	s.Equal(http.StatusOK, status)
	s.Equal(http.StatusUnauthorized, s.call(srv, first["access_token"].(string)).StatusCode)
	s.Equal(http.StatusOK, s.call(srv, second["access_token"].(string)).StatusCode, "each request is a grant of its own")

	srv.Advance(fakeprovider.AccessTTL)
	s.Equal(http.StatusUnauthorized, s.call(srv, second["access_token"].(string)).StatusCode)
}

func (s *FakeProviderSuite) TestNoExpiresInLeavesTheLifetimeOutButTokensStillEnd() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials, fakeprovider.NoExpiresIn)
	_, body := s.post(srv, fakeprovider.PathToken, url.Values{"grant_type": {"client_credentials"}}, true)
	s.NotContains(body, "expires_in")
	s.NotContains(s.connect(srv, nil), "expires_in", "the code exchange too")

	srv.Advance(fakeprovider.AccessTTL)
	s.Equal(http.StatusUnauthorized, s.call(srv, body["access_token"].(string)).StatusCode)
}

func (s *FakeProviderSuite) TestIdentityURLPutsTheIdentityURLThatTheSalesforceManifestCapturesInTheTokenResponse() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials, fakeprovider.IdentityURL)
	_, body := s.post(srv, fakeprovider.PathToken, url.Values{"grant_type": {"client_credentials"}}, true)
	s.Equal(srv.IdentityURL, body["id"])

	account := s.apply("../providers/salesforce.yaml", nil, body)
	s.Equal(srv.IdentityURL, account.AccountID)
	s.Equal(srv.IdentityURL, s.connect(srv, nil)["id"], "the code exchange carries it too")
}

func (s *FakeProviderSuite) TestSwitchAccountMakesTheNextConsentAnotherUsersAndKeepsTheFirstGrantsUser() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	first := s.connect(srv, url.Values{"scope": {"chat:write"}})

	other := srv.SwitchAccount()
	second := s.connect(srv, url.Values{"scope": {"chat:write"}})

	s.NotEqual(srv.UserID, other)
	s.Equal(other, second["authed_user"].(map[string]any)["id"])
	s.Equal(srv.TeamID, second["team"].(map[string]any)["id"], "another user of the same workspace")
	status, refreshed := s.refresh(srv, first["refresh_token"].(string))
	s.Require().Equal(http.StatusOK, status)
	s.Equal(srv.UserID, refreshed["authed_user"].(map[string]any)["id"], "the first grant is still the first user's")
}

func (s *FakeProviderSuite) TestCommaScopesAnswersARefreshInTheSameSlackShape() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	first := s.connect(srv, url.Values{"scope": {"channels:history,chat:write"}, "user_scope": {"search:read"}})
	s.Equal(float64(43200), first["expires_in"])
	user := first["authed_user"].(map[string]any)
	s.Equal(float64(43200), user["expires_in"])
	s.Require().NotEmpty(user["refresh_token"])

	status, refreshed := s.refresh(srv, first["refresh_token"].(string))
	s.Require().Equal(http.StatusOK, status)
	s.Equal(true, refreshed["ok"])
	s.Equal("bot", refreshed["token_type"])
	s.Equal("channels:history,chat:write", refreshed["scope"])
	s.Equal(float64(43200), refreshed["expires_in"])
	var recorded map[string]any
	s.Require().NoError(json.Unmarshal(s.read("../core/testdata/recorded/slack.token.json"), &recorded))
	for key := range recorded {
		s.Contains(refreshed, key)
	}
	account := s.apply("../core/testdata/manifests/slack.yaml", nil, refreshed)
	s.Equal(srv.TeamID, account.Metadata["team_id"])

	status, userRefreshed := s.refresh(srv, user["refresh_token"].(string))
	s.Require().Equal(http.StatusOK, status)
	s.Equal("user", userRefreshed["token_type"])
	s.Equal("search:read", userRefreshed["scope"])
	s.Equal(http.StatusOK, s.call(srv, userRefreshed["access_token"].(string)).StatusCode)
}

func (s *FakeProviderSuite) TestCommaScopesCodeRedeemedTwiceRevokesTheUserGrantToo() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	authorize, verifier := s.authorizeURL(srv, url.Values{"scope": {"channels:history"}, "user_scope": {"search:read"}})
	code := s.consent(srv, authorize).Query().Get("code")
	status, first := s.exchange(srv, code, verifier)
	s.Require().Equal(http.StatusOK, status, first)
	user := first["authed_user"].(map[string]any)
	s.Require().Equal(http.StatusOK, s.call(srv, user["access_token"].(string)).StatusCode)

	_, again := s.exchange(srv, code, verifier)
	s.Equal(false, again["ok"])
	s.Equal(http.StatusUnauthorized, s.call(srv, first["access_token"].(string)).StatusCode)
	s.Equal(http.StatusUnauthorized, s.call(srv, user["access_token"].(string)).StatusCode)
}

func (s *FakeProviderSuite) TestCommaScopesNamesTokenErrorsAsSlackDoes() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	authorize, verifier := s.authorizeURL(srv, nil)
	code := s.consent(srv, authorize).Query().Get("code")

	_, body := s.post(srv, fakeprovider.PathToken, url.Values{
		"grant_type": {"authorization_code"}, "code": {code}, "redirect_uri": {"https://client.example/other"},
		"code_verifier": {verifier},
	}, true)
	s.Equal("bad_redirect_uri", body["error"])

	_, body = s.exchange(srv, code, "another-verifier-of-enough-length-to-look-real")
	s.Equal("invalid_code_verifier", body["error"])

	_, body = s.post(srv, fakeprovider.PathToken, url.Values{"grant_type": {"password"}}, true)
	s.Equal("invalid_grant_type", body["error"])
}

func (s *FakeProviderSuite) TestCommaScopesRefusesASpaceSeparatedScope() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	authorize, _ := s.authorizeURL(srv, url.Values{"scope": {"channels:history chat:write"}})
	s.Equal("invalid_scope", s.consent(srv, authorize).Query().Get("error"))
}

func (s *FakeProviderSuite) TestCommaScopesReportsTokenErrorsAsOkFalseWithStatus200() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	status, body := s.refresh(srv, "a-refresh-token-nobody-issued")
	s.Equal(http.StatusOK, status)
	s.Equal(false, body["ok"])
	s.Equal("invalid_refresh_token", body["error"])
}

func (s *FakeProviderSuite) TestCallbackRealmIDPutsTheRealmInTheCallbackThatTheQuickBooksManifestCaptures() {
	srv := fakeprovider.New(s.T(), fakeprovider.CallbackRealmID)
	authorize, verifier := s.authorizeURL(srv, nil)
	callback := s.consent(srv, authorize)
	s.Equal(srv.RealmID, callback.Query().Get("realmId"))
	_, body := s.exchange(srv, callback.Query().Get("code"), verifier)

	account := s.apply("../core/testdata/manifests/quickbooks.yaml", callback.Query(), body)
	s.Equal(srv.RealmID, account.Metadata["realm_id"])
	s.Equal(srv.RealmID, account.AccountID)
	s.Contains(account.Unverified, "realm_id")

	status, refreshed := s.refresh(srv, body["refresh_token"].(string))
	s.Require().Equal(http.StatusOK, status)
	s.Equal(body["x_refresh_token_expires_in"], refreshed["x_refresh_token_expires_in"])
}

func (s *FakeProviderSuite) TestSignedCallbackIsSignedWithTheClientSecret() {
	srv := fakeprovider.New(s.T(), fakeprovider.SignedCallback)
	authorize, _ := s.authorizeURL(srv, url.Values{"state": {"nonce"}})
	query := s.consent(srv, authorize).Query()
	s.Equal(srv.Shop, query.Get("shop"))
	s.Regexp(`^[a-zA-Z0-9][a-zA-Z0-9\-]*\.myshopify\.com$`, query.Get("shop"), "shopify.dev's shop pattern")
	s.NotEmpty(query.Get("timestamp"))

	// Recomputed by hand, as shopify.dev describes it, not with fakeprovider.Sign.
	var pairs []string
	for _, key := range []string{"code", "host", "iss", "shop", "state", "timestamp"} {
		pairs = append(pairs, key+"="+query.Get(key))
	}
	s.Equal(hmacHex(srv.ClientSecret, strings.Join(pairs, "&")), query.Get("hmac"))
	s.Equal(query.Get("hmac"), fakeprovider.Sign(query, srv.ClientSecret))

	query.Set("shop", "other-shop.myshopify.com")
	s.NotEqual(query.Get("hmac"), fakeprovider.Sign(query, srv.ClientSecret), "a changed parameter breaks the signature")
}

func (s *FakeProviderSuite) TestTheMCPEndpointServesTheModernAndTheLegacyEra() {
	srv := fakeprovider.New(s.T())
	access := s.connect(srv, nil)["access_token"].(string)

	meta := map[string]any{"io.modelcontextprotocol/protocolVersion": "2026-07-28"}
	discover := s.rpc(srv, access, "2026-07-28", "server/discover", map[string]any{"_meta": meta})
	s.Equal("complete", discover["result"].(map[string]any)["resultType"])
	first := s.rpc(srv, access, "2026-07-28", "tools/list", map[string]any{"_meta": meta})["result"].(map[string]any)
	s.Equal("page-2", first["nextCursor"])
	second := s.rpc(srv, access, "2026-07-28", "tools/list", map[string]any{"_meta": meta, "cursor": "page-2"})["result"].(map[string]any)
	s.NotContains(second, "nextCursor")
	s.Equal("echo", first["tools"].([]any)[0].(map[string]any)["name"])
	s.Equal("fail", second["tools"].([]any)[0].(map[string]any)["name"])

	initialize := s.rpc(srv, access, "", "initialize", map[string]any{"protocolVersion": "2025-11-25"})
	s.Equal("2025-11-25", initialize["result"].(map[string]any)["protocolVersion"])
	echo := s.rpc(srv, access, "2025-11-25", "tools/call", map[string]any{"name": "echo", "arguments": map[string]any{"text": "hi"}})
	s.Equal("hi", echo["result"].(map[string]any)["content"].([]any)[0].(map[string]any)["text"])
	fail := s.rpc(srv, access, "2025-11-25", "tools/call", map[string]any{"name": "fail"})
	s.Equal(true, fail["result"].(map[string]any)["isError"])
}

func (s *FakeProviderSuite) TestAModernRequestWhoseHeaderDisagreesWithItsBodyIsRefused() {
	srv := fakeprovider.New(s.T())
	access := s.connect(srv, nil)["access_token"].(string)
	body, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": 1, "method": "tools/list",
		"params": map[string]any{"_meta": map[string]any{"io.modelcontextprotocol/protocolVersion": "2026-07-28"}}})
	request, err := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathMCP, strings.NewReader(string(body)))
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer "+access)
	request.Header.Set("MCP-Protocol-Version", "2026-07-28")
	request.Header.Set("Mcp-Method", "tools/call")
	response, err := srv.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	s.Equal(http.StatusBadRequest, response.StatusCode)
}

func (s *FakeProviderSuite) TestARequestWithoutATokenIsToldWhereDiscoveryStarts() {
	srv := fakeprovider.New(s.T())
	response := s.call(srv, "")
	s.Equal(http.StatusUnauthorized, response.StatusCode)
	s.Equal(`Bearer resource_metadata="`+srv.URL+fakeprovider.PathProtectedResource+`"`, response.Header.Get("WWW-Authenticate"))

	resource := s.getJSON(srv, fakeprovider.PathProtectedResource)
	s.Equal(srv.URL+fakeprovider.PathMCP, resource["resource"])
	s.Equal([]any{srv.URL}, resource["authorization_servers"])
}

func (s *FakeProviderSuite) TestTheServerClosesWhenItsTestEnds() {
	var srv *fakeprovider.Server
	s.Run("a test that uses it", func() {
		srv = fakeprovider.New(s.T())
		s.Equal(http.StatusOK, s.getStatus(srv, fakeprovider.PathAuthorizationServer))
	})
	_, err := srv.Client().Get(srv.URL + fakeprovider.PathAuthorizationServer)
	s.Error(err)
}

// authorizeURL is an authorization request for the preregistered client with a fresh PKCE
// pair; extra replaces any parameter.
func (s *FakeProviderSuite) authorizeURL(srv *fakeprovider.Server, extra url.Values) (string, string) {
	verifier := "verifier-" + strings.Repeat("x", 43)
	query := url.Values{
		"response_type": {"code"}, "client_id": {srv.ClientID}, "redirect_uri": {fakeprovider.RedirectURI},
		"code_challenge": {challenge(verifier)}, "code_challenge_method": {"S256"}, "state": {"state"},
		"resource": {srv.URL + fakeprovider.PathMCP},
	}
	for k, v := range extra {
		query[k] = v
	}
	return srv.URL + fakeprovider.PathAuthorize + "?" + query.Encode(), verifier
}

// serveClientMetadata starts a TLS server that serves the document document returns at
// /client, points srv's fetches at it, and returns the client_id URL and a count of fetches.
func (s *FakeProviderSuite) serveClientMetadata(srv *fakeprovider.Server, document func(clientID string) map[string]any) (string, *int) {
	var clientID string
	fetches := new(int)
	host := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*fetches++
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(document(clientID))
	}))
	s.T().Cleanup(host.Close)
	clientID = host.URL + "/client"
	srv.FetchClientMetadataWith(host.Client())
	return clientID, fetches
}

func (s *FakeProviderSuite) consent(srv *fakeprovider.Server, authorize string) *url.URL {
	callback, err := srv.Consent(authorize)
	s.Require().NoError(err)
	return callback
}

// connect is a whole consent: authorize, approve, exchange. It returns the token response.
func (s *FakeProviderSuite) connect(srv *fakeprovider.Server, extra url.Values) map[string]any {
	authorize, verifier := s.authorizeURL(srv, extra)
	callback := s.consent(srv, authorize)
	status, body := s.exchange(srv, callback.Query().Get("code"), verifier)
	s.Require().Equal(http.StatusOK, status, body)
	return body
}

func (s *FakeProviderSuite) exchange(srv *fakeprovider.Server, code, verifier string) (int, map[string]any) {
	return s.post(srv, fakeprovider.PathToken, url.Values{
		"grant_type": {"authorization_code"}, "code": {code}, "redirect_uri": {fakeprovider.RedirectURI},
		"code_verifier": {verifier}, "resource": {srv.URL + fakeprovider.PathMCP},
	}, true)
}

func (s *FakeProviderSuite) refresh(srv *fakeprovider.Server, refreshToken string) (int, map[string]any) {
	return s.post(srv, fakeprovider.PathToken, url.Values{"grant_type": {"refresh_token"}, "refresh_token": {refreshToken}}, true)
}

// lostRefresh is a refresh whose answer a lost-response personality drops; it returns the
// transport error.
func (s *FakeProviderSuite) lostRefresh(srv *fakeprovider.Server, refreshToken string) error {
	response, err := srv.Client().PostForm(srv.URL+fakeprovider.PathToken, url.Values{
		"grant_type": {"refresh_token"}, "refresh_token": {refreshToken},
		"client_id": {srv.ClientID}, "client_secret": {srv.ClientSecret},
	})
	if err == nil {
		s.Require().NoError(response.Body.Close())
	}
	return err
}

// post sends a form, authenticated as the preregistered client with client_secret_basic
// when basic is set.
func (s *FakeProviderSuite) post(srv *fakeprovider.Server, path string, form url.Values, basic bool) (int, map[string]any) {
	request, err := http.NewRequest(http.MethodPost, srv.URL+path, strings.NewReader(form.Encode()))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if basic {
		request.SetBasicAuth(srv.ClientID, srv.ClientSecret)
	}
	response, err := srv.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	var body map[string]any
	_ = json.NewDecoder(response.Body).Decode(&body)
	return response.StatusCode, body
}

// call is a legacy tools/call of echo with token, the request every resource personality
// answers.
func (s *FakeProviderSuite) call(srv *fakeprovider.Server, token string) *http.Response {
	request, err := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathMCP,
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}`))
	s.Require().NoError(err)
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response, err := srv.Client().Do(request)
	s.Require().NoError(err)
	_, _ = io.Copy(io.Discard, response.Body)
	s.Require().NoError(response.Body.Close())
	return response
}

// rpc sends one JSON-RPC request with the headers MCP 2026-07-28 asks for when version is
// set, and returns the decoded answer.
func (s *FakeProviderSuite) rpc(srv *fakeprovider.Server, token, version, method string, params map[string]any) map[string]any {
	body, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
	s.Require().NoError(err)
	request, err := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathMCP, strings.NewReader(string(body)))
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer "+token)
	if version != "" {
		request.Header.Set("MCP-Protocol-Version", version)
		request.Header.Set("Mcp-Method", method)
		if name, ok := params["name"].(string); ok {
			request.Header.Set("Mcp-Name", name)
		}
	}
	response, err := srv.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	s.Require().Equal(http.StatusOK, response.StatusCode)
	var answer map[string]any
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&answer))
	s.Require().NotContains(answer, "error")
	return answer
}

func (s *FakeProviderSuite) getJSON(srv *fakeprovider.Server, path string) map[string]any {
	response, err := srv.Client().Get(srv.URL + path)
	s.Require().NoError(err)
	defer response.Body.Close()
	var body map[string]any
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&body))
	return body
}

func (s *FakeProviderSuite) getStatus(srv *fakeprovider.Server, path string) int {
	response, err := srv.Client().Get(srv.URL + path)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}

// An event Deliver posts is one the core's Slack bot fixture verifies and reads, retry
// headers and all (https://docs.slack.dev/apis/events-api/, «Retries»).
func (s *FakeProviderSuite) TestSlackChannelDeliversAnEventSignedAsSlackSignsIt() {
	srv := fakeprovider.New(s.T(), fakeprovider.SlackChannel)
	manifest, err := core.ParseManifest(s.read("../core/testdata/manifests/slack_bot.yaml"))
	s.Require().NoError(err)
	var read core.VerifiedEvent
	var retry string
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		read, err = hmacheader.New().Verify(r, body, manifest, []byte("synthetic-signing-secret"))
		retry = r.Header.Get("X-Slack-Retry-Num")
		w.WriteHeader(http.StatusOK)
	}))
	defer endpoint.Close()

	status, _ := srv.Deliver(endpoint.URL, "synthetic-signing-secret", s.read("../core/testdata/recorded/slack_bot.message.json"), 2)

	s.Equal(http.StatusOK, status)
	s.Require().NoError(err)
	s.Require().Len(read.Messages, 1)
	s.Equal("1759740000.000200", read.Messages[0].ProviderMessageID)
	s.Equal("2", retry)
}

func (s *FakeProviderSuite) TestSlackChannelPostsAReplyWithTheBotTokenItIssued() {
	srv := fakeprovider.New(s.T(), fakeprovider.SlackChannel)
	token := srv.InstallBot()

	answer := s.postMessage(srv, token, `{"channel":"C0000CHAN","thread_ts":"1759740000.000100","text":"Done"}`)

	s.Equal(true, answer["ok"])
	s.NotEmpty(answer["ts"])
	posts := srv.Posts()
	s.Require().Len(posts, 1)
	s.Equal(fakeprovider.Post{Channel: "C0000CHAN", ThreadTS: "1759740000.000100", Text: "Done", Token: posts[0].Token}, posts[0])
	s.True(posts[0].Token == token)
}

// Slack refuses with HTTP 200 and ok false (https://docs.slack.dev/reference/methods/chat.postMessage).
func (s *FakeProviderSuite) TestSlackChannelRefusesATokenItDidNotIssueOrThatWasRevokedWithOkFalse() {
	srv := fakeprovider.New(s.T(), fakeprovider.SlackChannel)
	token := srv.InstallBot()
	srv.RevokeBot(token)

	s.Equal(map[string]any{"ok": false, "error": "invalid_auth"}, s.postMessage(srv, token, `{"channel":"C0000CHAN","text":"Done"}`))
	s.Equal(map[string]any{"ok": false, "error": "invalid_auth"}, s.postMessage(srv, "xoxb-not-issued", `{"channel":"C0000CHAN","text":"Done"}`))
	s.Empty(srv.Posts())
}

func (s *FakeProviderSuite) TestSlackChannelFailsTheNextPostsItIsToldToAndThenPosts() {
	srv := fakeprovider.New(s.T(), fakeprovider.SlackChannel)
	token := srv.InstallBot()
	srv.FailPosts(1)
	request, err := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathChatPostMessage, strings.NewReader(`{"channel":"C0000CHAN","text":"Done"}`))
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer "+token)

	response, err := srv.Client().Do(request)
	s.Require().NoError(err)
	_ = response.Body.Close()

	s.Equal(http.StatusServiceUnavailable, response.StatusCode)
	s.Empty(srv.Posts())
	s.Equal(true, s.postMessage(srv, token, `{"channel":"C0000CHAN","text":"Done"}`)["ok"])
	s.Len(srv.Posts(), 1)
}

func (s *FakeProviderSuite) TestChatPostMessageIsNotServedWithoutSlackChannel() {
	srv := fakeprovider.New(s.T())
	request, err := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathChatPostMessage, strings.NewReader(`{}`))
	s.Require().NoError(err)

	response, err := srv.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusNotFound, response.StatusCode)
}

// postMessage calls chat.postMessage with token and returns its answer, which is HTTP 200 in
// every case.
func (s *FakeProviderSuite) postMessage(srv *fakeprovider.Server, token, body string) map[string]any {
	request, err := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathChatPostMessage, strings.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+token)
	response, err := srv.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	s.Require().Equal(http.StatusOK, response.StatusCode)
	var answer map[string]any
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&answer))
	return answer
}

// apply runs a core fixture manifest's capture and identity rules over what the fake sent.
func (s *FakeProviderSuite) apply(manifestPath string, callback url.Values, token map[string]any) core.AccountInfo {
	manifest, err := core.ParseManifest(s.read(manifestPath))
	s.Require().NoError(err)
	resolved, err := manifest.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	raw, err := json.Marshal(token)
	s.Require().NoError(err)
	account, err := resolved.Apply(callback, raw)
	s.Require().NoError(err)
	return account
}

func (s *FakeProviderSuite) read(path string) []byte {
	raw, err := os.ReadFile(path)
	s.Require().NoError(err)
	return raw
}

// challenge is RFC 7636 §4.2 S256: BASE64URL(SHA256(verifier)).
func challenge(verifier string) string {
	digest := sha256.Sum256([]byte(verifier))
	return base64.RawURLEncoding.EncodeToString(digest[:])
}

func hmacHex(secret, message string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(message))
	return hex.EncodeToString(mac.Sum(nil))
}

func (s *FakeProviderSuite) TestAConfigRefreshTokenRotatesOnce() {
	srv := fakeprovider.New(s.T())
	refresh := srv.NewConfigToken()

	first := s.slack(srv, "tooling.tokens.rotate", "", url.Values{"refresh_token": {refresh}})
	second := s.slack(srv, "tooling.tokens.rotate", "", url.Values{"refresh_token": {refresh}})

	s.Equal(true, first["ok"])
	s.Equal(first["iat"].(float64)+fakeprovider.ConfigTokenTTL.Seconds(), first["exp"])
	s.Equal("invalid_refresh_token", second["error"])
	s.Equal(1, srv.ConfigTokenRotations())
}

func (s *FakeProviderSuite) TestASlackAppIsMadeOnlyWithAConfigTokenThatHasNotExpired() {
	srv := fakeprovider.New(s.T())
	token := s.configToken(srv)
	manifest := url.Values{"manifest": {`{"display_information":{"name":"Acme"},"settings":{}}`}}

	created := s.slack(srv, "apps.manifest.create", token, manifest)
	srv.Advance(fakeprovider.ConfigTokenTTL)
	expired := s.slack(srv, "apps.manifest.create", token, manifest)

	s.Equal(true, created["ok"])
	s.Equal(created["app_id"], srv.SlackApps()[0].AppID)
	s.Equal("token_expired", expired["error"])
	s.Len(srv.SlackApps(), 1)
}

func (s *FakeProviderSuite) TestADeletedSlackAppIsNotFound() {
	srv := fakeprovider.New(s.T())
	token := s.configToken(srv)
	manifest := `{"display_information":{"name":"Acme"},"settings":{}}`
	app := s.slack(srv, "apps.manifest.create", token, url.Values{"manifest": {manifest}})["app_id"].(string)

	s.Equal(true, s.slack(srv, "apps.manifest.delete", token, url.Values{"app_id": {app}})["ok"])
	s.Equal("app_not_found", s.slack(srv, "apps.manifest.delete", token, url.Values{"app_id": {app}})["error"])
	s.Equal("app_not_found", s.slack(srv, "apps.manifest.update", token, url.Values{"app_id": {app}, "manifest": {manifest}})["error"])
}

func (s *FakeProviderSuite) TestASlowConfigRotationAnswersOnlyAfterItsDelay() {
	srv := fakeprovider.New(s.T(), fakeprovider.SlowConfigRotation)
	started := time.Now()

	s.configToken(srv)

	s.GreaterOrEqual(time.Since(started), 200*time.Millisecond)
}

// configToken is a configuration token rotated from one the fake's admin generated.
func (s *FakeProviderSuite) configToken(srv *fakeprovider.Server) string {
	token, ok := s.slack(srv, "tooling.tokens.rotate", "", url.Values{"refresh_token": {srv.NewConfigToken()}})["token"].(string)
	s.Require().True(ok)
	return token
}

// slack posts form to one of the fake Slack's methods, with token as a bearer token when one
// is given, and returns the JSON it answered.
func (s *FakeProviderSuite) slack(srv *fakeprovider.Server, method, token string, form url.Values) map[string]any {
	request, err := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathSlackAPI+method, strings.NewReader(form.Encode()))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response, err := srv.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	s.Require().Equal(http.StatusOK, response.StatusCode)
	var answered map[string]any
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&answered))
	return answered
}
