//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

const consentReturnURLsPath = "/v1/agents/consent-return-urls"

// ConsentReturnURLsSuite is an app's list of the pages that may finish its connector
// consents: put, read back, refused, and kept from other apps. Each test has an app of its own.
type ConsentReturnURLsSuite struct {
	RouterSuite
}

func TestConsentReturnURLsSuite(t *testing.T) {
	runSuite(t, new(ConsentReturnURLsSuite))
}

func (s *ConsentReturnURLsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// put registers urls as the app's backend and says how it was answered.
func (s *ConsentReturnURLsSuite) put(urls ...string) (int, string) {
	// None is sent as [], which is how an empty list is written; null is refused.
	return s.serverClient.failure(http.MethodPut, consentReturnURLsPath, ConsentReturnUrls{Urls: append([]string{}, urls...)})
}

// registered is the app's list as GET answers it.
func (s *ConsentReturnURLsSuite) registered(as *testClient) []string {
	var answered ConsentReturnUrls
	s.Require().Equal(http.StatusOK, as.do(http.MethodGet, consentReturnURLsPath, nil, &answered))
	return answered.Urls
}

// assertRefused checks that a list is a 400 naming why, and that the list stored is unchanged.
func (s *ConsentReturnURLsSuite) assertRefused(why string, urls ...string) {
	s.T().Helper()
	kept := []string{"https://kept.example/done"}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, consentReturnURLsPath, ConsentReturnUrls{Urls: kept}, nil))

	status, message := s.put(urls...)

	s.Equal(http.StatusBadRequest, status, message)
	s.Contains(message, why)
	s.Equal(kept, s.registered(s.serverClient))
}

func (s *ConsentReturnURLsSuite) TestOnlyTheAppsBackendMaySetConsentReturnURLs() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPut, consentReturnURLsPath, ConsentReturnUrls{Urls: []string{"https://app.example/done"}}, nil)
	})
}

func (s *ConsentReturnURLsSuite) TestOnlyTheAppsBackendMayReadConsentReturnURLs() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, consentReturnURLsPath, nil, nil)
	})
}

func (s *ConsentReturnURLsSuite) TestAnAppThatRegisteredNoneReadsAnEmptyList() {
	status, body := s.serverClient.call(http.MethodGet, consentReturnURLsPath, nil)

	s.Equal(http.StatusOK, status)
	var fields map[string]json.RawMessage
	s.Require().NoError(json.Unmarshal(body, &fields))
	s.JSONEq(`[]`, string(fields["urls"]))
}

func (s *ConsentReturnURLsSuite) TestTheListPutIsAnsweredAndReadBackExactlyAsSent() {
	urls := []string{
		"https://app.example/connectors/done?tab=tools",
		"https://app.example/",
		"http://localhost:3000/done",
		"http://127.0.0.1:8080/done",
		"http://[::1]:5173/",
	}

	var answered ConsentReturnUrls
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, consentReturnURLsPath, ConsentReturnUrls{Urls: urls}, &answered))

	s.Equal(urls, answered.Urls)
	s.Equal(urls, s.registered(s.serverClient))
}

func (s *ConsentReturnURLsSuite) TestAPutReplacesTheWholeList() {
	first, second := "https://first.example/done", "https://second.example/done"
	status, message := s.put(first)
	s.Require().Equal(http.StatusOK, status, message)

	status, message = s.put(second)

	s.Require().Equal(http.StatusOK, status, message)
	s.Equal([]string{second}, s.registered(s.serverClient))
}

func (s *ConsentReturnURLsSuite) TestAnEmptyListRemovesThemAll() {
	status, message := s.put("https://app.example/done")
	s.Require().Equal(http.StatusOK, status, message)

	status, message = s.put()

	s.Require().Equal(http.StatusOK, status, message)
	s.Empty(s.registered(s.serverClient))
	settings, err := s.store.AppSettingsFor(context.Background(), s.customerID())
	s.Require().NoError(err)
	s.Nil(settings.ConsentReturnURLs)
}

func (s *ConsentReturnURLsSuite) TestPlainHTTPToAHostThatIsNotLoopbackIsRefused() {
	s.assertRefused("http is taken for a loopback host alone", "http://app.example/done")
	s.assertRefused("http is taken for a loopback host alone", "http://localhost.app.example/done")
	s.assertRefused("http is taken for a loopback host alone", "http://10.0.0.1/done")
}

func (s *ConsentReturnURLsSuite) TestASchemeOtherThanHTTPSIsRefused() {
	s.assertRefused("the scheme must be https", "myapp://callback")
	s.assertRefused("the scheme must be https", "ftp://app.example/done")
}

func (s *ConsentReturnURLsSuite) TestAURLThatIsNotAbsoluteIsRefused() {
	s.assertRefused("not an absolute URL with a host", "/connectors/done")
	s.assertRefused("not an absolute URL with a host", "app.example/done")
	s.assertRefused("not an absolute URL with a host", "javascript:alert(1)")
	s.assertRefused("not an absolute URL with a host", "https:///done")
}

func (s *ConsentReturnURLsSuite) TestAURLCarryingAUserNameIsRefused() {
	s.assertRefused("user name or password", "https://admin@app.example/done")
	s.assertRefused("user name or password", "https://admin:secret@app.example/done")
}

func (s *ConsentReturnURLsSuite) TestAURLWithAFragmentIsRefused() {
	s.assertRefused("fragment", "https://app.example/done#tools")
	s.assertRefused("fragment", "https://app.example/done#")
}

func (s *ConsentReturnURLsSuite) TestAURLListedTwiceIsRefused() {
	s.assertRefused("urls[2]: the same URL as urls[0]",
		"https://app.example/done", "https://other.example/done", "https://app.example/done")
}

func (s *ConsentReturnURLsSuite) TestTenURLsAreTakenAndAnEleventhIsRefused() {
	urls := make([]string, 0, maxConsentReturnURLs+1)
	for len(urls) < maxConsentReturnURLs {
		urls = append(urls, "https://app.example/"+s.utils.uuid())
	}
	status, message := s.put(urls...)
	s.Require().Equal(http.StatusOK, status, message)

	s.assertRefused("validation failed", append(urls, "https://app.example/eleventh")...)
}

func (s *ConsentReturnURLsSuite) TestAURLOf2048BytesIsTakenAndALongerOneRefused() {
	prefix := "https://app.example/"
	longest := prefix + strings.Repeat("a", maxConsentReturnURLLength-len(prefix))
	status, message := s.put(longest)
	s.Require().Equal(http.StatusOK, status, message)

	s.assertRefused("longer than 2048 bytes", longest+"a")
}

func (s *ConsentReturnURLsSuite) TestAnotherAppNeitherReadsNorReplacesTheAppsList() {
	mine := []string{"https://app.example/done"}
	status, message := s.put(mine...)
	s.Require().Equal(http.StatusOK, status, message)
	stranger := s.data.backendOfAnotherApp()

	s.Empty(s.registered(stranger), "another app reads its own list, which is empty")
	s.Require().Equal(http.StatusOK, stranger.do(http.MethodPut, consentReturnURLsPath,
		ConsentReturnUrls{Urls: []string{"https://stranger.example/done"}}, nil))

	s.Equal(mine, s.registered(s.serverClient))
	s.Equal([]string{"https://stranger.example/done"}, s.registered(stranger))
}

func (s *ConsentReturnURLsSuite) TestARouterBehindAProxyKeepsNoListAndAnswers501() {
	s.assertNotImplemented(auth.Proxy, auth.NewProxy(auth.ProxyOptions{}))
}

func (s *ConsentReturnURLsSuite) TestARouterWithoutAuthenticationKeepsNoListAndAnswers501() {
	noauth, err := auth.New(auth.NoAuth, nil)
	s.Require().NoError(err)

	s.assertNotImplemented(auth.NoAuth, noauth)
}

// assertNotImplemented runs a router in mode over the suite's database, where the suite's
// app is a row, and names that app the way such a deployment is told the customer: neither
// reading nor writing reaches the row.
func (s *ConsentReturnURLsSuite) assertNotImplemented(mode auth.Mode, authenticator auth.Authenticator) {
	server, err := NewServer(Options{
		Routers:  map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		Store:    s.store,
		Auth:     authenticator,
		AuthMode: mode,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	router := httptest.NewServer(server.Handler())
	s.T().Cleanup(router.Close)

	for _, method := range []string{http.MethodPut, http.MethodGet} {
		body, err := json.Marshal(ConsentReturnUrls{Urls: []string{"https://app.example/done"}})
		s.Require().NoError(err)
		request, err := http.NewRequest(method, router.URL+consentReturnURLsPath, bytes.NewReader(body))
		s.Require().NoError(err)
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set(auth.AppHeader, s.customerID())
		request.Header.Set(CustomerHeader, s.customerID())
		response, err := http.DefaultClient.Do(request)
		s.Require().NoError(err)
		var answered ErrorResponse
		s.Require().NoError(json.NewDecoder(response.Body).Decode(&answered))
		s.Require().NoError(response.Body.Close())

		s.Equal(http.StatusNotImplemented, response.StatusCode, method)
		s.Equal(errNoAppSettings.Message, answered.Error.Message, method)
		s.Equal(ErrorTypeNotImplemented, answered.Error.Type, method)
	}
	settings, err := s.store.AppSettingsFor(context.Background(), s.customerID())
	s.Require().NoError(err)
	s.Nil(settings.ConsentReturnURLs, "the app's row is left alone")
}
