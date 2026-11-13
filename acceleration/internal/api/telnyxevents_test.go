//go:build integration

package api

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// TelnyxEventsSuite is the Telnyx account's notification channel sending account events to
// the router, signed with the account's key as Telnyx signs them.
//
// The counter is process-wide and its labels are a fixed set, so each test reads the change
// on the series it sends to. No other suite sends the Telnyx account events these tests count,
// and a suite's tests run one at a time.
type TelnyxEventsSuite struct {
	RouterSuite
	private ed25519.PrivateKey
	logged  *lockedLog
	since   int
}

func TestTelnyxEventsSuite(t *testing.T) {
	runSuite(t, new(TelnyxEventsSuite))
}

func (s *TelnyxEventsSuite) SetupSuite() {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)
	s.private = private
	s.telnyxPublicKey = public
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

func (s *TelnyxEventsSuite) SetupTest() {
	s.since = len(s.logged.String())
}

func (s *TelnyxEventsSuite) TestASignedPortOutIsCountedAtPageAndLogged() {
	id := rand.Text()
	before := accountEvents("telnyx", "portout.status_changed", "page")

	status, _ := s.deliver(telnyxAccountEvent("portout.status_changed", id, `{"status":"pending"}`), s.signer(time.Now()))

	s.Equal(http.StatusNoContent, status)
	s.Equal(1.0, accountEvents("telnyx", "portout.status_changed", "page")-before)
	s.Equal(1, s.loggedLines("vendor=telnyx", "event=portout.status_changed", "level=page", "id="+id))
}

// A number order that did not succeed left numbers unbought; one that succeeded needs nobody.
func (s *TelnyxEventsSuite) TestANumberOrderIsCountedAtTheLevelItsStatusGives() {
	warn := accountEvents("telnyx", "number_order.complete", "warn")
	info := accountEvents("telnyx", "number_order.complete", "info")

	failed, _ := s.deliver(telnyxAccountEvent("number_order.complete", rand.Text(), `{"status":"failure"}`), s.signer(time.Now()))
	bought, _ := s.deliver(telnyxAccountEvent("number_order.complete", rand.Text(), `{"status":"success"}`), s.signer(time.Now()))

	s.Equal(http.StatusNoContent, failed)
	s.Equal(http.StatusNoContent, bought)
	s.Equal(1.0, accountEvents("telnyx", "number_order.complete", "warn")-warn)
	s.Equal(1.0, accountEvents("telnyx", "number_order.complete", "info")-info)
}

// The port-in deletion and messaging events carry their fields at the top level, with no data
// around them, as in https://developers.telnyx.com/docs/numbers/porting/port-in-notifications.md.
func (s *TelnyxEventsSuite) TestAPortInEventWithItsFieldsAtTheTopLevelIsCounted() {
	for _, eventType := range []string{"porting_order.deleted", "porting_order.messaging_changed"} {
		id := rand.Text()
		before := accountEvents("telnyx", eventType, "info")

		status, _ := s.deliver(`{"id":"`+id+`","event_type":"`+eventType+`","occurred_at":"2023-01-01T00:00:00Z",`+
			`"format":"v2","payload":{"id":"x","deleted_at":"2023-01-01T00:00:00Z"}}`, s.signer(time.Now()))

		s.Equal(http.StatusNoContent, status, eventType)
		s.Equal(1.0, accountEvents("telnyx", eventType, "info")-before, eventType)
		s.Equal(1, s.loggedLines("event="+eventType, "level=info", "id="+id), eventType)
	}
}

// Another event's status need not be a string, and it does not stop the event being counted.
func (s *TelnyxEventsSuite) TestAnEventWhoseStatusIsNotAStringIsCounted() {
	before := accountEvents("telnyx", "porting_order.status_changed", "info")

	status, _ := s.deliver(telnyxAccountEvent("porting_order.status_changed", rand.Text(),
		`{"status":{"value":"ported"}}`), s.signer(time.Now()))

	s.Equal(http.StatusNoContent, status)
	s.Equal(1.0, accountEvents("telnyx", "porting_order.status_changed", "info")-before)
}

// A type the router does not name adds no series of its own, and the log still says what
// Telnyx sent.
func (s *TelnyxEventsSuite) TestATypeTheRouterDoesNotNameIsCountedAsOther() {
	eventType := "number." + strings.ToLower(rand.Text())
	before := accountEvents("telnyx", "other", "info")

	status, _ := s.deliver(telnyxAccountEvent(eventType, rand.Text(), `{}`), s.signer(time.Now()))

	s.Equal(http.StatusNoContent, status)
	s.Equal(1.0, accountEvents("telnyx", "other", "info")-before)
	s.Zero(accountEvents("telnyx", eventType, "info"))
	s.Equal(1, s.loggedLines("event=other", "event_type="+eventType))
}

// The body as Telnyx signed it, with one byte of the signature changed.
func (s *TelnyxEventsSuite) TestAWronglySignedEventIsRefusedAndNotCounted() {
	body := telnyxAccountEvent("portout.status_changed", rand.Text(), `{}`)
	before := accountEvents("telnyx", "portout.status_changed", "page")
	sign := s.signer(time.Now())

	status, answer := s.deliver(body, func(r *http.Request, body string) {
		sign(r, body)
		signature, err := base64.StdEncoding.DecodeString(r.Header.Get("Telnyx-Signature-Ed25519"))
		s.Require().NoError(err)
		signature[0] ^= 1
		r.Header.Set("Telnyx-Signature-Ed25519", base64.StdEncoding.EncodeToString(signature))
	})

	s.Equal(http.StatusUnauthorized, status)
	s.Contains(answer, "unauthenticated")
	s.Zero(accountEvents("telnyx", "portout.status_changed", "page") - before)
}

func (s *TelnyxEventsSuite) TestAnUnsignedEventIsRefusedAndNotCounted() {
	before := accountEvents("telnyx", "portout.status_changed", "page")

	status, _ := s.deliver(telnyxAccountEvent("portout.status_changed", rand.Text(), `{}`), func(*http.Request, string) {})

	s.Equal(http.StatusUnauthorized, status)
	s.Zero(accountEvents("telnyx", "portout.status_changed", "page") - before)
}

// A captured event cannot be sent again once its timestamp is past the tolerance.
func (s *TelnyxEventsSuite) TestAStaleEventIsRefusedAndNotCounted() {
	before := accountEvents("telnyx", "portout.status_changed", "page")

	status, _ := s.deliver(telnyxAccountEvent("portout.status_changed", rand.Text(), `{}`), s.signer(time.Now().Add(-10*time.Minute)))

	s.Equal(http.StatusUnauthorized, status)
	s.Zero(accountEvents("telnyx", "portout.status_changed", "page") - before)
}

// Signed or not, a body that is no event names nothing to count. Each refusal is logged, so
// a Telnyx shape the router does not read shows up, and the body, which may hold numbers, is not.
func (s *TelnyxEventsSuite) TestASignedBodyThatIsNoEventIsRefusedAndNotCounted() {
	before := accountEvents("telnyx", "other", "info")
	bodies := []string{`not json`, `{}`, `{"data":{}}`, `{"data":{"event_type":""}}`,
		`{"data":{"event_type":7}}`, `{"data":{"event_type":"number.order","id":7}}`,
		`{"event_type":""}`, `{"event_type":7}`, `{"event_type":"number.order","id":7}`, `null`, `[]`}
	for _, body := range bodies {
		status, _ := s.deliver(body, s.signer(time.Now()))
		s.Equal(http.StatusBadRequest, status, body)
	}
	s.Zero(accountEvents("telnyx", "other", "info") - before)
	s.Equal(len(bodies), s.loggedLines("level=WARN", "refused a signed telnyx account event that is no event"))
	s.NotContains(s.logged.String()[s.since:], "number.order")
}

func (s *TelnyxEventsSuite) TestAnEventOverTheLimitIsRefusedBeforeItIsRead() {
	before := accountEvents("telnyx", "other", "info")
	body := telnyxAccountEvent("number.big", rand.Text(), `{"pad":"`+strings.Repeat("x", telnyxCallLimit)+`"}`)

	status, _ := s.deliver(body, s.signer(time.Now()))

	s.Equal(http.StatusRequestEntityTooLarge, status)
	s.Zero(accountEvents("telnyx", "other", "info") - before)
}

// loggedLines counts the lines the current test logged that hold every part.
func (s *TelnyxEventsSuite) loggedLines(parts ...string) int {
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

// signer signs a delivery as Telnyx does, over "{timestamp}|{body}", at the time given.
func (s *TelnyxEventsSuite) signer(at time.Time) func(*http.Request, string) {
	return func(r *http.Request, body string) {
		timestamp := strconv.FormatInt(at.Unix(), 10)
		signature := ed25519.Sign(s.private, []byte(timestamp+"|"+body))
		r.Header.Set("Telnyx-Timestamp", timestamp)
		r.Header.Set("Telnyx-Signature-Ed25519", base64.StdEncoding.EncodeToString(signature))
	}
}

func (s *TelnyxEventsSuite) deliver(body string, sign func(*http.Request, string)) (int, string) {
	return deliverTelnyxEvent(&s.RouterSuite, body, sign)
}

// deliverTelnyxEvent posts an account event as Telnyx does and answers with what came back.
func deliverTelnyxEvent(s *RouterSuite, body string, sign func(*http.Request, string)) (int, string) {
	request, err := http.NewRequest(http.MethodPost, s.server.URL+"/v1/phone/hooks/telnyx/events", strings.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	sign(request, body)
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	answered, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, string(answered)
}

// telnyxAccountEvent is an event as Telnyx's notification channel sends it, in the shape of
// https://developers.telnyx.com/docs/numbers/porting/port-out-notifications.md.
func telnyxAccountEvent(eventType, id, payload string) string {
	return `{"data":{"event_type":"` + eventType + `","id":"` + id + `","occurred_at":"2026-11-12T10:00:00Z",` +
		`"payload":` + payload + `,"record_type":"event"},"meta":{"attempt":1,"delivered_to":"https://router.example/v1/phone/hooks/telnyx/events"}}`
}

// accountEvents is the value of one vendor_account_events_total series as /metrics serves
// it, 0 while it does not exist.
func accountEvents(vendor, event, level string) float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		panic(err)
	}
	want := map[string]string{"vendor": vendor, "event": event, "level": level}
	for _, family := range families {
		if family.GetName() != "vendor_account_events_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			matched := len(metric.GetLabel()) == len(want)
			for _, pair := range metric.GetLabel() {
				matched = matched && want[pair.GetName()] == pair.GetValue()
			}
			if matched {
				return metric.GetCounter().GetValue()
			}
		}
	}
	return 0
}

// TelnyxEventsUnconfiguredSuite is a deployment with no Telnyx public key.
type TelnyxEventsUnconfiguredSuite struct {
	RouterSuite
}

func TestTelnyxEventsUnconfiguredSuite(t *testing.T) {
	runSuite(t, new(TelnyxEventsUnconfiguredSuite))
}

// Without the key nothing could tell Telnyx's events from anybody's, so none is counted. It
// reads a series TelnyxEventsSuite, which runs beside it, never sends.
func (s *TelnyxEventsUnconfiguredSuite) TestWithoutAKeyNoEventIsCounted() {
	before := accountEvents("telnyx", "portout.foc_date_changed", "page")

	status, answer := deliverTelnyxEvent(&s.RouterSuite, telnyxAccountEvent("portout.foc_date_changed", rand.Text(), `{}`),
		func(*http.Request, string) {})

	s.Equal(http.StatusNotFound, status)
	s.Contains(answer, "telnyx events are not configured")
	s.Zero(accountEvents("telnyx", "portout.foc_date_changed", "page") - before)
}
