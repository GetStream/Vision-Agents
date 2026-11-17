//go:build integration

package api

import (
	"context"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// baseLaunchPage is the launch page base 1145e341 served for every attempt, its nonce written
// NONCE (probe-base.txt in the AI-1383 ledger, 2026-10-16). A consent started without a
// return_url, and an attempt that cannot be read, must still get exactly this page.
const baseLaunchPage = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="referrer" content="no-referrer"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Connecting your account</title></head>
<body><p id="status">Connecting securely to the provider…</p><script nonce="NONCE">
const dashboardOrigin = "https://dashboard.example";
const status = document.getElementById('status');
let handoffStarted = false;
window.addEventListener('message', async (event) => {
  if (handoffStarted || event.source !== window.opener || event.origin !== dashboardOrigin || !event.data || event.data.type !== 'va.connector.oauth.handoff' || typeof event.data.handoff_token !== 'string') return;
  handoffStarted = true;
  try {
    const response = await fetch(window.location.pathname, { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ handoff_token: event.data.handoff_token }) });
    if (!response.ok) throw new Error('This consent has expired. Start again from the dashboard.');
    const authorization = await response.json();
    window.opener = null;
    window.location.replace(authorization.authorization_url);
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : 'Could not reach the provider.';
  }
});
if (window.opener) window.opener.postMessage({ type: 'va.connector.oauth.ready' }, dashboardOrigin);
else status.textContent = 'Start from the dashboard, so this browser can be checked.';
</script></body></html>`

// appReturnURL is a page of the app's own with a query of its own, which the callback keeps.
const appReturnURL = "https://app.example/connectors/done?tab=tools&from=chat"

// nativeReturnURL is a native app's own scheme, in the reverse domain form RFC 8252 section 7.1
// asks for, with a query of its own.
const nativeReturnURL = "io.getstream.assistant:/gmail?from=composer"

func (s *AuthorizationsSuite) TestAConsentStartedWithAReturnURLIsLaunchedFromItsOriginAndGoesBackThere() {
	id := s.connection("")
	started := s.startReturningTo(id, appReturnURL)
	alice := s.browser()

	page := alice.get(started.LaunchURL)
	finished := alice.finish(s.consent(alice.handOff(started)))

	s.Equal(launchPageFrom("https://app.example"), s.launchPage(page),
		"the base page, with the return_url's origin in place of the dashboard's and no word of a dashboard")
	s.Equal("https://app.example/connectors/done?"+url.Values{
		"connection_id": {id}, "status": {consentConnected}, "tab": {"tools"}, "from": {"chat"},
	}.Encode(), finished.Header.Get("Location"))
	s.Equal(ConnectionStatus(store.ConnectionConnected), s.get(id).Status)
}

func (s *AuthorizationsSuite) TestAConsentStartedWithoutAReturnURLIsLaunchedAndEndsExactlyAsOnBase() {
	for _, body := range []any{nil, map[string]string{}, AuthorizationRequest{ReturnURL: ""}} {
		id := s.connection("")
		var started Authorization
		s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections/"+id+"/authorizations", body, &started))
		alice := s.browser()

		page := alice.get(started.LaunchURL)
		finished := alice.get(s.consent(alice.handOff(started)).String())

		s.Equal(baseLaunchPage, s.launchPage(page), "body %v", body)
		// Base 1145e341's Location and body (probe-base.txt).
		s.Equal("https://dashboard.example/connections?connection_id="+id+"&status=connected", finished.Header.Get("Location"))
		s.Equal(`<a href="https://dashboard.example/connections?connection_id=`+id+`&amp;status=connected">Found</a>.`+"\n\n", s.body(finished))
	}
}

func (s *AuthorizationsSuite) TestTheLaunchPageOfAConsentWithAReturnURLDoesNotSendThePersonToTheDashboard() {
	started := s.startReturningTo(s.connection(""), appReturnURL)

	page := s.launchPage(s.browser().get(started.LaunchURL))

	s.Contains(page, "This consent has expired. Start again from the page that opened this.")
	s.Contains(page, "Start from the page that opened this, so this browser can be checked.")
	s.NotContains(page, "dashboard.")
	s.NotContains(page, "from the dashboard")
}

func (s *AuthorizationsSuite) TestTheLaunchPageOfAnUnknownAttemptIsTheDashboardsPage() {
	page := s.browser().get(consentPublicURL + connectorLaunchPath + store.NewID())

	s.Equal(baseLaunchPage, s.launchPage(page))
}

func (s *AuthorizationsSuite) TestTheLaunchPageOfAFinishedConsentIsTheDashboardsPage() {
	id := s.connection("")
	started := s.startReturningTo(id, appReturnURL)
	alice := s.browser()
	alice.finish(s.consent(alice.handOff(started)))

	page := alice.get(started.LaunchURL)

	s.Equal(baseLaunchPage, s.launchPage(page), "a used attempt names no origin")
}

func (s *AuthorizationsSuite) TestAHandoffFromTheReturnURLsOriginIsRefused() {
	started := s.startReturningTo(s.connection(""), appReturnURL)

	response := s.browser().post(started.LaunchURL, "https://app.example", map[string]string{"handoff_token": started.HandoffToken})

	s.Equal(http.StatusForbidden, response.StatusCode, "the launch page's own fetch, from the router's origin, is the only handoff")
	s.Empty(response.Cookies())
}

func (s *AuthorizationsSuite) TestADeniedConsentAlsoGoesBackToTheReturnURL() {
	id := s.connection("")
	alice := s.browser()
	authorize := alice.handOff(s.startReturningTo(id, appReturnURL))
	s.provider.Use(fakeprovider.CommaScopes, fakeprovider.ConsentDenied)

	finished := alice.finish(s.consent(authorize))

	s.Equal(s.returnedTo(appReturnURL, id, consentDenied), finished.Header.Get("Location"))
	s.Equal(ConnectionStatus(store.ConnectionPending), s.get(id).Status)
}

func (s *AuthorizationsSuite) TestAConsentTheProviderRefusedAlsoGoesBackToTheReturnURL() {
	id := s.connection("")
	alice := s.browser()
	authorize := alice.handOff(s.startReturningTo(id, appReturnURL))
	s.provider.Use(fakeprovider.CommaScopes, fakeprovider.ConsentDenied)
	callback := s.consent(authorize)
	query := callback.Query()
	// RFC 6749 section 4.1.2.1's unauthorized_client, the refusal that leaves a reason.
	query.Set("error", "unauthorized_client")
	callback.RawQuery = query.Encode()

	finished := alice.finish(callback)

	s.Equal(s.returnedTo(appReturnURL, id, consentFailed), finished.Header.Get("Location"))
	s.Require().NotNil(s.get(id).LastValidation, "the refusal still leaves its reason")
}

func (s *AuthorizationsSuite) TestAConsentForAnotherAccountAlsoGoesBackToTheReturnURL() {
	id := s.connection("")
	s.connect(id)
	s.provider.SwitchAccount()
	alice := s.browser()

	finished := alice.finish(s.consent(alice.handOff(s.startReturningTo(id, appReturnURL))))

	s.Equal(s.returnedTo(appReturnURL, id, consentAccountMismatch), finished.Header.Get("Location"))
}

func (s *AuthorizationsSuite) TestTheRoutersConnectionIDAndStatusReplaceTheOnesInTheReturnURL() {
	id := s.connection("")
	returnURL := "https://app.example/done?status=mine&connection_id=other&tab=tools"
	alice := s.browser()

	finished := alice.finish(s.consent(alice.handOff(s.startReturningTo(id, returnURL))))

	s.Equal("https://app.example/done?"+url.Values{
		"connection_id": {id}, "status": {consentConnected}, "tab": {"tools"},
	}.Encode(), finished.Header.Get("Location"), "one of each, the router's")
}

func (s *AuthorizationsSuite) TestTheReturnURLsQueryComesBackReEncodedWithItsKeysSorted() {
	id := s.connection("")
	alice := s.browser()

	finished := alice.finish(s.consent(alice.handOff(s.startReturningTo(id, "https://app.example/done?b&a=1&a=2"))))

	s.Equal("https://app.example/done?a=1&a=2&b=&connection_id="+id+"&status="+consentConnected, finished.Header.Get("Location"))
}

func (s *AuthorizationsSuite) TestAReturnURLOnALoopbackHostIsTakenOverPlainHTTP() {
	// RFC 8252 section 8.3's loopback hosts, each with the origin the launch page names.
	for returnURL, origin := range map[string]string{
		"http://localhost:3000/done":  "http://localhost:3000",
		"http://127.0.0.1:8080/done":  "http://127.0.0.1:8080",
		"http://127.1.2.3/done":       "http://127.1.2.3",
		"http://[::1]:5173/?mode=dev": "http://[::1]:5173",
	} {
		started := s.startReturningTo(s.connection(""), returnURL)

		page := s.launchPage(s.browser().get(started.LaunchURL))

		s.Contains(page, `const dashboardOrigin = "`+origin+`";`, returnURL)
	}
}

func (s *AuthorizationsSuite) TestAReturnURLThatIsNotAbsoluteIsRefused() {
	s.assertReturnURLRefused("not an absolute URL with a host", "/connectors/done")
	s.assertReturnURLRefused("not an absolute URL with a host", "app.example/done")
	s.assertReturnURLRefused("not an absolute URL with a host", "javascript:alert(1)")
	s.assertReturnURLRefused("not an absolute URL with a host", "https:///done")
	// Scheme-relative: no scheme, so not absolute, whatever the host.
	s.assertReturnURLRefused("not an absolute URL with a host", "//app.example/x")
}

func (s *AuthorizationsSuite) TestAReturnURLWhoseHostIsNotASCIIIsRefused() {
	// A browser's origin holds the punycode host, which the launch page would never be sent.
	s.assertReturnURLRefused("non-ASCII", "https://bücher.example/done")
	s.startReturningTo(s.connection(""), "https://xn--bcher-kva.example/done")
}

func (s *AuthorizationsSuite) TestAReturnURLWithAnIPv4MappedHostIsRefused() {
	// A browser's origin writes it [::ffff:7f00:1], which the launch page would never be sent.
	s.assertReturnURLRefused("IPv4-mapped", "http://[::ffff:127.0.0.1]:3000/done")
	s.assertReturnURLRefused("IPv4-mapped", "https://[::ffff:10.0.0.1]/done")
}

func (s *AuthorizationsSuite) TestAReturnURLOverPlainHTTPToAHostThatIsNotLoopbackIsRefused() {
	s.assertReturnURLRefused("http is taken for a loopback host alone", "http://app.example/done")
	s.assertReturnURLRefused("http is taken for a loopback host alone", "http://localhost.app.example/done")
	s.assertReturnURLRefused("http is taken for a loopback host alone", "http://10.0.0.1/done")
}

func (s *AuthorizationsSuite) TestAReturnURLWithASchemeOtherThanHTTPSIsRefused() {
	s.assertReturnURLRefused("the scheme must be https", "myapp://callback")
	s.assertReturnURLRefused("the scheme must be https", "ftp://app.example/done")
}

// TestAConsentReturningToANativeAppIsHandedOffFromItsFragmentAndGoesBackToTheApp: an app has
// no page to post the token from, so it opens launch_url#handoff_token=... in its web
// authentication sheet. The page reads the token from its fragment, which the router is never
// sent, and the callback hands the sheet back to the app's own scheme.
func (s *AuthorizationsSuite) TestAConsentReturningToANativeAppIsHandedOffFromItsFragmentAndGoesBackToTheApp() {
	id := s.connection("")
	started := s.startReturningTo(id, nativeReturnURL)
	alice := s.browser()

	page := s.launchPage(alice.get(started.LaunchURL + "#handoff_token=" + started.HandoffToken))
	finished := alice.finish(s.consent(alice.handOff(started)))

	s.Contains(page, `const handoffToken = new URLSearchParams(window.location.hash.slice(1)).get('handoff_token');`)
	s.Contains(page, `history.replaceState(null, '', window.location.pathname);`)
	s.NotContains(page, "dashboardOrigin", "there is no opener to take it from")
	s.NotContains(page, "dashboard", "no word of a dashboard the person never used")
	s.NotContains(page, started.HandoffToken, "the fragment never reaches the router")
	s.Equal(s.returnedTo(nativeReturnURL, id, consentConnected), finished.Header.Get("Location"))
	s.Equal(ConnectionStatus(store.ConnectionConnected), s.get(id).Status)
}

func (s *AuthorizationsSuite) TestANativeAppsOwnSchemeInReverseDomainFormIsTaken() {
	for _, returnURL := range []string{"com.example.app:/done", "com.example.app://callback?x=1", nativeReturnURL} {
		started := s.startReturningTo(s.connection(""), returnURL)

		s.NotContains(s.launchPage(s.browser().get(started.LaunchURL)), "dashboardOrigin", returnURL)
	}
}

func (s *AuthorizationsSuite) TestANativeAppsSchemeKeepsTheRulesOfAnyReturnURL() {
	s.assertReturnURLRefused("fragment", "com.example.app:/done#tools")
	s.assertReturnURLRefused("user name or password", "com.example.app://admin@callback/done")
	s.assertReturnURLRefused("a query that is not key=value pairs", "com.example.app:/done?a=%zz")
}

func (s *AuthorizationsSuite) TestAReturnURLCarryingAUserNameIsRefused() {
	s.assertReturnURLRefused("user name or password", "https://admin@app.example/done")
	s.assertReturnURLRefused("user name or password", "https://admin:secret@app.example/done")
}

func (s *AuthorizationsSuite) TestAReturnURLWithAFragmentIsRefused() {
	s.assertReturnURLRefused("fragment", "https://app.example/done#tools")
	s.assertReturnURLRefused("fragment", "https://app.example/done#")
}

func (s *AuthorizationsSuite) TestAReturnURLWhoseQueryIsNotKeyValuePairsIsRefused() {
	s.assertReturnURLRefused("a query that is not key=value pairs", "https://app.example/done?a=%zz")
	s.assertReturnURLRefused("a query that is not key=value pairs", "https://app.example/done?a=1;b=2")
}

func (s *AuthorizationsSuite) TestAReturnURLOf2048BytesIsTakenAndALongerOneRefused() {
	prefix := "https://app.example/"
	longest := prefix + strings.Repeat("a", maxReturnURLBytes-len(prefix))
	s.startReturningTo(s.connection(""), longest)

	s.assertReturnURLRefused("longer than 2048 bytes", longest+"a")
}

// startReturningTo is the app's backend starting a consent that goes back to returnURL.
func (s *AuthorizationsSuite) startReturningTo(id, returnURL string) Authorization {
	var started Authorization
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections/"+id+"/authorizations",
		AuthorizationRequest{ReturnURL: returnURL}, &started))
	return started
}

// assertReturnURLRefused checks that a consent with returnURL is a 400 naming why, and that no
// attempt was begun.
func (s *AuthorizationsSuite) assertReturnURLRefused(why, returnURL string) {
	s.T().Helper()
	id := s.connection("")

	status, message := s.serverClient.failure(http.MethodPost, "/v1/agents/connections/"+id+"/authorizations", AuthorizationRequest{ReturnURL: returnURL})

	s.Equal(http.StatusBadRequest, status, returnURL)
	s.True(strings.HasPrefix(message, "return_url: "), message)
	s.Contains(message, why, returnURL)
	var attempts int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM connector_authorization_attempts WHERE connection_id = ?", id).Scan(&attempts))
	s.Zero(attempts, returnURL)
}

// launchPage is a launch page's body with its nonce written NONCE, after checking the nonce is
// the one its Content-Security-Policy allows.
func (s *AuthorizationsSuite) launchPage(response *http.Response) string {
	return launchPageIn(&s.RouterSuite, response)
}

// launchPageIn is launchPage for any suite.
func launchPageIn(s *RouterSuite, response *http.Response) string {
	s.Require().Equal(http.StatusOK, response.StatusCode)
	nonce := regexp.MustCompile(`'nonce-([A-Za-z0-9_-]+)'`).FindStringSubmatch(response.Header.Get("Content-Security-Policy"))
	s.Require().Len(nonce, 2)
	defer response.Body.Close()
	read, err := readAll(response)
	s.Require().NoError(err)
	return strings.ReplaceAll(string(read), nonce[1], "NONCE")
}

// launchPageFrom is the base page as an attempt with a return_url of origin serves it: that
// origin in place of the dashboard's, and no word of a dashboard.
func launchPageFrom(origin string) string {
	page := strings.Replace(baseLaunchPage, `"https://dashboard.example"`, `"`+origin+`"`, 1)
	page = strings.Replace(page, "Start again from the dashboard.", "Start again from the page that opened this.", 1)
	return strings.Replace(page, "Start from the dashboard, so", "Start from the page that opened this, so", 1)
}

// returnedTo is where the callback sends the browser of a consent started with returnURL.
func (s *AuthorizationsSuite) returnedTo(returnURL, id, status string) string {
	destination, err := url.Parse(returnURL)
	s.Require().NoError(err)
	query := destination.Query()
	query.Set("connection_id", id)
	query.Set("status", status)
	destination.RawQuery = query.Encode()
	return destination.String()
}
