//go:build integration

package api

import (
	"crypto/rand"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/twiliosig"
)

// twilioTestToken is the Auth Token of Twilio's own signature example
// (https://www.twilio.com/docs/usage/webhooks/webhooks-security), not a real account's.
const twilioTestToken = "12345"

// twilioSignedURL is the URL Twilio is told to call: the suite's public URL, trimmed of its
// trailing slash, then the path.
const twilioSignedURL = "http://localhost:8080" + twilioEventsHookPath

// TwilioEventsSuite is the Twilio account sending Monitor alerts and usage-trigger callbacks
// to the router, signed with the account's Auth Token as Twilio signs them.
//
// The counter is process-wide and its labels are a fixed set, so each test reads the change
// on the series it sends to. No other suite sends Twilio events, and a suite's tests run one
// at a time.
type TwilioEventsSuite struct {
	RouterSuite
	logged *lockedLog
	since  int
}

func TestTwilioEventsSuite(t *testing.T) {
	runSuite(t, new(TwilioEventsSuite))
}

func (s *TwilioEventsSuite) SetupSuite() {
	s.twilioAuthToken = twilioTestToken
	// With a trailing slash, as an operator may write it.
	s.publicURL = "http://localhost:8080/"
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

func (s *TwilioEventsSuite) SetupTest() {
	s.since = len(s.logged.String())
}

// A Debugger alert for 10001, signed with a signature computed apart from twiliosig, so a Sign
// that is wrong the same way on both sides cannot pass:
//
//	python3 -c 'import hmac,hashlib,base64; u="http://localhost:8080/v1/phone/hooks/twilio/events";
//	p={"Sid":"NO00000000000000000000000000000001","Level":"ERROR","PayloadType":"application/json",
//	"Payload":"{\"error_code\":\"10001\"}"}; s=u+"".join(k+p[k] for k in sorted(p));
//	print(base64.b64encode(hmac.new(b"12345",s.encode(),hashlib.sha1).digest()).decode())'
func (s *TwilioEventsSuite) TestASignedAccountNotActiveAlertIsCountedAtPageAndLogged() {
	const signature = "e7MiSS/EXrsnJyEC2sbIOpeew4w="
	before := accountEvents("twilio", "error.10001", "page")

	status, _ := s.deliver(twilioAlert("NO00000000000000000000000000000001", `{"error_code":"10001"}`), "",
		withSignature(signature))

	s.Equal(http.StatusNoContent, status)
	s.Equal(1.0, accountEvents("twilio", "error.10001", "page")-before)
	s.Equal(1, s.loggedLines("received a twilio account event"))
	s.Equal(1, s.loggedLines("vendor=twilio", "event=error.10001", "level=page",
		"sid=NO00000000000000000000000000000001", "error_code=10001", "twilio_level=error"))
	s.NotContains(s.logged.String()[s.since:], twilioTestToken)
	s.NotContains(s.logged.String()[s.since:], signature)
}

// The docs give error_code no type, so a number is the same code as the string.
func (s *TwilioEventsSuite) TestAnAlertWhoseErrorCodeIsANumberIsCountedUnderItsCode() {
	before := accountEvents("twilio", "error.30002", "page")

	status, _ := s.deliver(twilioAlert(rand.Text(), `{"error_code":30002}`), "", s.signed(twilioSignedURL))

	s.Equal(http.StatusNoContent, status)
	s.Equal(1.0, accountEvents("twilio", "error.30002", "page")-before)
}

func (s *TwilioEventsSuite) TestAnAlertWithAWarnCodeIsCountedAtWarn() {
	before := accountEvents("twilio", "error.32017", "warn")

	status, _ := s.deliver(twilioAlert(rand.Text(), `{"error_code":"32017"}`), "", s.signed(twilioSignedURL))

	s.Equal(http.StatusNoContent, status)
	s.Equal(1.0, accountEvents("twilio", "error.32017", "warn")-before)
}

// A code the router does not name adds no series of its own, and the log still says what
// Twilio sent.
func (s *TwilioEventsSuite) TestAnUnlistedCodeIsCountedAsErrorOtherAtInfo() {
	before := accountEvents("twilio", "error.other", "info")

	status, _ := s.deliver(twilioAlert(rand.Text(), `{"error_code":"11200"}`), "", s.signed(twilioSignedURL))

	s.Equal(http.StatusNoContent, status)
	s.Equal(1.0, accountEvents("twilio", "error.other", "info")-before)
	s.Zero(accountEvents("twilio", "error.11200", "info"))
	s.Equal(1, s.loggedLines("event=error.other", "error_code=11200"))
}

// Twilio signed the alert, so it is counted even when its Payload names no code the router
// can read.
func (s *TwilioEventsSuite) TestAnAlertWithNoReadableCodeIsCountedAsErrorOther() {
	payloads := []string{`not json`, `{}`, `{"error_code":null}`, `{"error_code":true}`, `{"error_code":{"code":"10001"}}`,
		`[]`, `"10001"`, `{"ErrorCode":"10001"}`}
	before := accountEvents("twilio", "error.other", "info")
	page := accountEvents("twilio", "error.10001", "page")

	for _, payload := range payloads {
		status, _ := s.deliver(twilioAlert(rand.Text(), payload), "", s.signed(twilioSignedURL))
		s.Equal(http.StatusNoContent, status, payload)
	}

	s.Equal(float64(len(payloads)), accountEvents("twilio", "error.other", "info")-before)
	s.Zero(accountEvents("twilio", "error.10001", "page") - page)
	s.Equal(len(payloads), s.loggedLines("event=error.other", `error_code=""`))
}

// The callback's parameters as https://www.twilio.com/docs/usage/api/usage-trigger lists them.
func (s *TwilioEventsSuite) TestASignedTotalPriceTriggerIsCountedAtPageAndLogged() {
	sid := "UT" + rand.Text()
	before := accountEvents("twilio", "usage_trigger.totalprice", "page")

	form := twilioUsageTrigger(sid, "totalprice")
	status, _ := s.deliver(form, "", s.signed(twilioSignedURL))

	s.Equal(http.StatusNoContent, status)
	s.Equal(1.0, accountEvents("twilio", "usage_trigger.totalprice", "page")-before)
	s.Equal(1, s.loggedLines("received a twilio account event"))
	s.neverLogged(twiliosig.Sign(twilioTestToken, twilioSignedURL, form))
	s.Equal(1, s.loggedLines("vendor=twilio", "event=usage_trigger.totalprice", "level=page",
		"usage_trigger_sid="+sid, "usage_category=totalprice"))
}

func (s *TwilioEventsSuite) TestACallsTriggerIsCountedAtWarn() {
	before := accountEvents("twilio", "usage_trigger.calls", "warn")

	status, _ := s.deliver(twilioUsageTrigger("UT"+rand.Text(), "calls"), "", s.signed(twilioSignedURL))

	s.Equal(http.StatusNoContent, status)
	s.Equal(1.0, accountEvents("twilio", "usage_trigger.calls", "warn")-before)
}

func (s *TwilioEventsSuite) TestAnUnlistedCategoryIsCountedAsUsageTriggerOtherAtWarn() {
	category := "sms-" + strings.ToLower(rand.Text())
	before := accountEvents("twilio", "usage_trigger.other", "warn")

	status, _ := s.deliver(twilioUsageTrigger("UT"+rand.Text(), category), "", s.signed(twilioSignedURL))

	s.Equal(http.StatusNoContent, status)
	s.Equal(1.0, accountEvents("twilio", "usage_trigger.other", "warn")-before)
	s.Zero(accountEvents("twilio", "usage_trigger."+category, "warn"))
	s.Equal(1, s.loggedLines("event=usage_trigger.other", "usage_category="+category))
}

// The alert as Twilio signed it, with one byte of the signature changed.
func (s *TwilioEventsSuite) TestAWronglySignedAlertIsRefusedAndNotCounted() {
	form := twilioAlert(rand.Text(), `{"error_code":"10001"}`)
	signature := []byte(twiliosig.Sign(twilioTestToken, twilioSignedURL, form))
	signature[0] ^= 1
	before := accountEvents("twilio", "error.10001", "page")

	status, answer := s.deliver(form, "", withSignature(string(signature)))

	s.Equal(http.StatusUnauthorized, status)
	s.Contains(answer, "unauthenticated")
	s.Zero(accountEvents("twilio", "error.10001", "page") - before)
	s.Equal(1, s.loggedLines("level=WARN", "refused a twilio account event"))
	s.neverLogged(string(signature))
}

func (s *TwilioEventsSuite) TestAnUnsignedAlertIsRefusedAndNotCounted() {
	before := accountEvents("twilio", "error.10001", "page")

	status, _ := s.deliver(twilioAlert(rand.Text(), `{"error_code":"10001"}`), "", withSignature(""))

	s.Equal(http.StatusUnauthorized, status)
	s.Zero(accountEvents("twilio", "error.10001", "page") - before)
}

// A signature Twilio made for another URL, such as the router's address behind the proxy or
// another of its hooks, is not one for this route.
func (s *TwilioEventsSuite) TestASignatureOverAnotherURLIsRefused() {
	before := accountEvents("twilio", "usage_trigger.totalprice", "page")
	for _, signedURL := range []string{s.server.URL + twilioEventsHookPath, "http://localhost:8080/v1/phone/hooks/telnyx/events",
		"https://localhost:8080" + twilioEventsHookPath, "http://localhost:8080/" + twilioEventsHookPath} {
		status, _ := s.deliver(twilioUsageTrigger("UT"+rand.Text(), "totalprice"), "", s.signed(signedURL))
		s.Equal(http.StatusUnauthorized, status, signedURL)
	}
	s.Zero(accountEvents("twilio", "usage_trigger.totalprice", "page") - before)
}

// A parameter added after Twilio signed, here one that turns a calls trigger into a page, is
// refused.
func (s *TwilioEventsSuite) TestAFormChangedAfterSigningIsRefused() {
	form := twilioUsageTrigger("UT"+rand.Text(), "calls")
	signature := twiliosig.Sign(twilioTestToken, twilioSignedURL, form)
	form.Set("UsageCategory", "totalprice")
	before := accountEvents("twilio", "usage_trigger.totalprice", "page")

	status, _ := s.deliver(form, "", withSignature(signature))

	s.Equal(http.StatusUnauthorized, status)
	s.Zero(accountEvents("twilio", "usage_trigger.totalprice", "page") - before)
}

// Twilio signs the URL it called with its query string, so a webhook set with one is checked
// with it, and a query added to a URL signed without one is refused.
func (s *TwilioEventsSuite) TestTheQueryStringIsPartOfTheSignedURL() {
	before := accountEvents("twilio", "usage_trigger.calls", "warn")

	signed, _ := s.deliver(twilioUsageTrigger("UT"+rand.Text(), "calls"), "source=console",
		s.signed(twilioSignedURL+"?source=console"))
	added, _ := s.deliver(twilioUsageTrigger("UT"+rand.Text(), "calls"), "source=console", s.signed(twilioSignedURL))

	// Twilio signed the query as it sent it, so it is not sorted or decoded before the check.
	const raw = "z=1&a=b%20c"
	unsorted, _ := s.deliver(twilioUsageTrigger("UT"+rand.Text(), "calls"), raw, s.signed(twilioSignedURL+"?"+raw))

	s.Equal(http.StatusNoContent, signed)
	s.Equal(http.StatusUnauthorized, added)
	s.Equal(http.StatusNoContent, unsorted)
	s.Equal(2.0, accountEvents("twilio", "usage_trigger.calls", "warn")-before)
}

// Signed or not, a form that is neither an alert nor a usage trigger names nothing to count.
// Each refusal is logged, so a Twilio shape the router does not read shows up, and the form,
// which may hold numbers, is not.
func (s *TwilioEventsSuite) TestASignedFormThatIsNoEventIsRefusedAndNotCounted() {
	// Stands in for a caller's number.
	caller := rand.Text()
	forms := []url.Values{
		{},
		{"Sid": {"NO1"}, "Level": {"ERROR"}},
		{"Payload": {`{"error_code":"10001"}`}, "From": {caller}},
		{"Sid": {"NO1"}, "Payload": {""}},
		{"UsageCategory": {"totalprice"}, "AccountSid": {"AC1"}},
		{"UsageTriggerSid": {""}, "UsageCategory": {"totalprice"}},
	}
	alerts := accountEvents("twilio", "error.other", "info")
	page := accountEvents("twilio", "error.10001", "page")
	triggers := accountEvents("twilio", "usage_trigger.totalprice", "page")

	var signatures []string
	for _, form := range forms {
		status, _ := s.deliver(form, "", s.signed(twilioSignedURL))
		s.Equal(http.StatusBadRequest, status, form.Encode())
		signatures = append(signatures, twiliosig.Sign(twilioTestToken, twilioSignedURL, form))
	}

	s.Zero(accountEvents("twilio", "error.other", "info") - alerts)
	s.Zero(accountEvents("twilio", "error.10001", "page") - page)
	s.Zero(accountEvents("twilio", "usage_trigger.totalprice", "page") - triggers)
	s.Equal(len(forms), s.loggedLines("level=WARN", "refused a signed twilio account event that is no event"))
	s.NotContains(s.logged.String()[s.since:], caller)
	s.neverLogged(signatures...)
}

// A body that is no form cannot be checked against a signature over its parameters.
func (s *TwilioEventsSuite) TestABodyThatIsNoFormIsRefused() {
	status, answer := s.post("Sid=NO1&Payload=%zz", "", "")

	s.Equal(http.StatusBadRequest, status)
	s.Contains(answer, "that is not a form")
}

// The 1 MiB the spec names, written out rather than read from the constant, so a raised limit
// fails here.
func (s *TwilioEventsSuite) TestAnEventOverTheLimitIsRefusedBeforeItIsRead() {
	before := accountEvents("twilio", "error.other", "info")
	form := twilioAlert(rand.Text(), `{"pad":"`+strings.Repeat("x", 1<<20)+`"}`)

	status, _ := s.deliver(form, "", s.signed(twilioSignedURL))

	s.Equal(http.StatusRequestEntityTooLarge, status)
	s.Zero(accountEvents("twilio", "error.other", "info") - before)
}

// An alert near the limit, as one carrying a failed webhook's request and response may be, is
// still read: the limit is not the 64 KiB of a Telnyx call.
func (s *TwilioEventsSuite) TestAnAlertLargerThanATelnyxCallIsCounted() {
	before := accountEvents("twilio", "error.10001", "page")
	form := twilioAlert(rand.Text(), `{"error_code":"10001","more_info":"`+strings.Repeat("x", 2*telnyxCallLimit)+`"}`)

	status, _ := s.deliver(form, "", s.signed(twilioSignedURL))

	s.Equal(http.StatusNoContent, status)
	s.Equal(1.0, accountEvents("twilio", "error.10001", "page")-before)
}

// Without the public URL no signature Twilio made can match, so the route is not there rather
// than refusing every alert as unsigned.
func (s *TwilioEventsSuite) TestWithoutAPublicURLNoEventIsCounted() {
	public := s.router.publicURL
	s.T().Cleanup(func() { s.router.publicURL = public })
	s.router.publicURL = "/"
	before := accountEvents("twilio", "error.10001", "page")

	status, answer := s.deliver(twilioAlert(rand.Text(), `{"error_code":"10001"}`), "", s.signed(twilioEventsHookPath))

	s.Equal(http.StatusNotFound, status)
	s.Contains(answer, "Twilio events are not configured")
	s.Zero(accountEvents("twilio", "error.10001", "page") - before)
}

// neverLogged fails when what the current test logged holds the token or any of the signatures.
func (s *TwilioEventsSuite) neverLogged(signatures ...string) {
	logged := s.logged.String()[s.since:]
	s.NotContains(logged, twilioTestToken)
	for _, signature := range signatures {
		s.NotContains(logged, signature)
	}
}

// loggedLines counts the lines the current test logged that hold every part.
func (s *TwilioEventsSuite) loggedLines(parts ...string) int {
	n := 0
	for _, line := range strings.Split(s.logged.String()[s.since:], "\n") {
		found := true
		for _, part := range parts {
			found = found && strings.Contains(line, part)
		}
		if found {
			n++
		}
	}
	return n
}

// signed signs a form as Twilio does for a webhook at signedURL.
func (s *TwilioEventsSuite) signed(signedURL string) func(url.Values) string {
	return func(form url.Values) string {
		return twiliosig.Sign(twilioTestToken, signedURL, form)
	}
}

// withSignature sends signature, empty for none, whatever the form is.
func withSignature(signature string) func(url.Values) string {
	return func(url.Values) string { return signature }
}

func (s *TwilioEventsSuite) deliver(form url.Values, query string, sign func(url.Values) string) (int, string) {
	return s.post(form.Encode(), query, sign(form))
}

func (s *TwilioEventsSuite) post(body, query, signature string) (int, string) {
	return postTwilioEvent(&s.RouterSuite, body, query, signature)
}

// postTwilioEvent posts a form as Twilio does and answers with what came back.
func postTwilioEvent(s *RouterSuite, body, query, signature string) (int, string) {
	target := s.server.URL + twilioEventsHookPath
	if query != "" {
		target += "?" + query
	}
	request, err := http.NewRequest(http.MethodPost, target, strings.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if signature != "" {
		request.Header.Set(twiliosig.SignatureHeader, signature)
	}
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	answered, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, string(answered)
}

// twilioAlert is a Monitor alert as the Debugger webhook sends it, in the shape of
// https://www.twilio.com/docs/usage/troubleshooting/debugging-event-webhooks.
func twilioAlert(sid, payload string) url.Values {
	return url.Values{"Sid": {sid}, "Level": {"ERROR"}, "PayloadType": {"application/json"}, "Payload": {payload}}
}

// twilioUsageTrigger is a usage trigger's callback, with the parameters of
// https://www.twilio.com/docs/usage/api/usage-trigger (CallbackUrl request parameters).
func twilioUsageTrigger(sid, category string) url.Values {
	return url.Values{"AccountSid": {"AC00000000000000000000000000000000"}, "UsageTriggerSid": {sid},
		"DateFired": {"2026-11-13"}, "Recurring": {"daily"}, "UsageCategory": {category}, "TriggerBy": {"price"},
		"TriggerValue": {"50"}, "CurrentValue": {"60"}, "IdempotencyToken": {rand.Text()},
		"UsageRecordUri": {"/2010-04-01/Accounts/AC00000000000000000000000000000000/Usage/Records/Today"}}
}

// TwilioEventsUnconfiguredSuite is a deployment with no Twilio webhook token.
type TwilioEventsUnconfiguredSuite struct {
	RouterSuite
}

func TestTwilioEventsUnconfiguredSuite(t *testing.T) {
	runSuite(t, new(TwilioEventsUnconfiguredSuite))
}

func (s *TwilioEventsUnconfiguredSuite) SetupSuite() {
	// The public URL is set, so only the missing token turns the route off.
	s.publicURL = "http://localhost:8080"
	s.RouterSuite.SetupSuite()
}

// Without the token nothing could tell Twilio's events from anybody's, so none is counted,
// signed or not. It reads a series TwilioEventsSuite, which runs beside it, never sends.
func (s *TwilioEventsUnconfiguredSuite) TestWithoutATokenNoEventIsCounted() {
	form := url.Values{"Sid": {rand.Text()}, "Payload": {`{"error_code":"20003"}`}}
	before := accountEvents("twilio", "error.20003", "warn")

	unsigned, answer := postTwilioEvent(&s.RouterSuite, form.Encode(), "", "")
	signed, _ := postTwilioEvent(&s.RouterSuite, form.Encode(), "", twiliosig.Sign(twilioTestToken, twilioSignedURL, form))

	s.Equal(http.StatusNotFound, unsigned)
	s.Equal(http.StatusNotFound, signed)
	s.Contains(answer, `"type":"not_found"`)
	s.Contains(answer, "Twilio events are not configured")
	s.Zero(accountEvents("twilio", "error.20003", "warn") - before)
}
