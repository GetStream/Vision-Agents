//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// dropped is how long a hook that should wake nobody is given to prove it. A hook that
// dispatches does so before it answers, so this only has to outlast the answer.
const dropped = 200 * time.Millisecond

type CallHooksSuite struct {
	RouterSuite

	// e164 is the number rung in a test, and callID the call it arrives as. Both are
	// unique, because a number is held by one customer at a time.
	e164   string
	callID string
}

func TestCallHooksSuite(t *testing.T) {
	runSuite(t, new(CallHooksSuite))
}

func (s *CallHooksSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.e164 = s.utils.number()
	s.callID = "phone-" + s.e164
}

func (s *CallHooksSuite) TestAnArrivingCallReachesTheWorkerOfWhoeverHoldsTheNumber() {
	s.hold("default", s.callID)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	select {
	case call := <-worker.Calls():
		s.Equal(s.callID, call.CallID)
		s.Equal("default", call.CallType)
		s.Equal(s.e164, call.CalledNumber, "the worker has to know which line rang")
		s.Equal("+15550001111", call.CallerNumber)
		s.Equal("support", call.Custom["line"])
		s.False(call.At.IsZero())
	case <-time.After(settleFor):
		s.Fail("the call never reached the worker")
	}
}

func (s *CallHooksSuite) TestACallOnACustomNamedLineStillFindsItsOwner() {
	// A number attached to a call of its own is the case the stored binding exists for:
	// nothing about "the-support-line" says which number it belongs to.
	line := "line-" + s.utils.uuid()
	s.hold("support", line)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("support:"+line))

	select {
	case call := <-worker.Calls():
		s.Equal(line, call.CallID)
		s.Equal("support", call.CallType)
		s.Equal(s.e164, call.CalledNumber)
	case <-time.After(settleFor):
		s.Fail("the call never reached the worker")
	}
}

func (s *CallHooksSuite) TestAnotherCustomersWorkerIsNotGivenTheCall() {
	// Two customers' workers are two rotations, and a call is one customer's.
	s.hold("default", s.callID)
	somebodyElse, release := s.dispatch.Register(s.utils.uuid(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	s.nothingReaches(somebodyElse.Calls(), "a call was misrouted")
}

func (s *CallHooksSuite) TestACallOnANumberNobodyHoldsIsAcceptedAndDropped() {
	// Every video call in the app arrives at this hook too. Retrying would not make one
	// answerable, so it is accepted and nothing is woken.
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("default:standup-"+s.utils.uuid()))

	s.nothingReaches(worker.Calls(), "a video call was answered")
}

func (s *CallHooksSuite) TestACallOnAReleasedNumberIsNotAnswered() {
	// The number is gone, so whoever holds it now is not this customer.
	s.hold("default", s.callID)
	s.Require().NoError(s.store.ReleaseNumber(
		context.Background(), s.customerID(), s.e164, time.Now().UTC()))
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Require().Equal(http.StatusOK, s.arrive("default:"+s.callID))

	s.nothingReaches(worker.Calls(), "a released number was answered")
}

func (s *CallHooksSuite) TestAnUnsignedCallEventIsRefused() {
	status, _ := s.deliver("/v1/phone/hooks/stream", s.arriving("default:"+s.callID), "")

	s.Equal(http.StatusUnauthorized, status)
}

func (s *CallHooksSuite) TestACallEventSignedWithTheWrongSecretIsRefused() {
	body := s.arriving("default:" + s.callID)

	status, _ := s.deliver("/v1/phone/hooks/stream", body, sign(body, "somebody-elses-secret"))

	s.Equal(http.StatusUnauthorized, status)
}

func (s *CallHooksSuite) TestATamperedCallEventIsRefused() {
	body := s.arriving("default:" + s.callID)
	signature := sign(body, suiteStreamSecret)
	tampered := strings.Replace(body, "+15550001111", "+15559998888", 1)

	status, _ := s.deliver("/v1/phone/hooks/stream", tampered, signature)

	s.Equal(http.StatusUnauthorized, status)
}

func (s *CallHooksSuite) TestASessionEndedEventIsAccepted() {
	ended := fmt.Sprintf(`{"type":"call.session_ended","call_cid":"default:%s",`+
		`"session_id":"session-1","created_at":"2026-08-27T12:05:00Z",`+
		`"call":{"cid":"default:%s","id":"%s","type":"default","custom":{}}}`,
		s.callID, s.callID, s.callID)

	s.Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", ended))
}

func (s *CallHooksSuite) TestAnEventTypeThisVersionHasNeverHeardOfIsAccepted() {
	unknown := fmt.Sprintf(`{"type":"call.something_new","call_cid":"default:%s"}`, s.callID)

	s.Equal(http.StatusOK, s.signedly("/v1/phone/hooks/stream", unknown),
		"a new event type must not look like an outage to Stream")
}

func (s *CallHooksSuite) TestSomethingThatIsNotACallEventIsRefused() {
	// Correctly signed, but there is no event in it to act on.
	s.Equal(http.StatusBadRequest, s.signedly("/v1/phone/hooks/stream", `{"not":"an event"}`))
}

func (s *CallHooksSuite) TestTheHookIgnoresWhoeverTheCallerClaimsToBe() {
	// Stream is not a customer, so a credential is neither required nor read. Sending one
	// changes nothing, which is what stops a caller thinking it scopes the hook.
	s.hold("default", s.callID)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	body := s.arriving("default:" + s.callID)
	request := s.request("/v1/phone/hooks/stream", body, sign(body, suiteStreamSecret))
	request.Header.Set(CustomerHeader, "globex")
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())

	s.Require().Equal(http.StatusOK, response.StatusCode)
	select {
	case call := <-worker.Calls():
		s.Equal(s.callID, call.CallID, "the number decided whose call it is")
	case <-time.After(settleFor):
		s.Fail("the call never reached the worker")
	}
}

// hold records the number for the suite's app and attaches it to a call.
func (s *CallHooksSuite) hold(callType, callID string) {
	ctx := context.Background()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{
		E164:        s.e164,
		Vendor:      "telnyx",
		Country:     "US",
		CustomerID:  s.customerID(),
		PurchasedAt: time.Now().UTC(),
	}))
	s.Require().NoError(s.store.AttachNumber(
		ctx, s.customerID(), s.e164, "trunk-"+s.utils.uuid(), callType, callID))
}

// arrive delivers a signed call.session_started for one call, as Stream would.
func (s *CallHooksSuite) arrive(cid string) int {
	return s.signedly("/v1/phone/hooks/stream", s.arriving(cid))
}

// arriving is what Stream sends when a caller lands in a call.
func (s *CallHooksSuite) arriving(cid string) string {
	id := cid[strings.Index(cid, ":")+1:]
	callType := cid[:strings.Index(cid, ":")]
	return fmt.Sprintf(`{
  "type": "call.session_started",
  "call_cid": %q,
  "session_id": "session-1",
  "created_at": "2026-08-27T12:00:00Z",
  "call": {
    "cid": %q,
    "id": %q,
    "type": %q,
    "custom": {"line": "support"},
    "session": {
      "id": "session-1",
      "participants": [
        {"user_session_id": "s1", "role": "user", "joined_at": "2026-08-27T12:00:00Z",
         "user": {"id": "sip-+15550001111"}}
      ]
    }
  }
}`, cid, cid, id, callType)
}

// nothingReaches fails when a call arrives on a channel that should stay empty.
func (s *CallHooksSuite) nothingReaches(calls <-chan dispatch.Call, message string) {
	select {
	case call := <-calls:
		s.Failf(message, "%s reached a worker", call.CallID)
	case <-time.After(dropped):
	}
}

// signedly delivers a body signed with the app secret, and answers with the status.
func (s *RouterSuite) signedly(path, body string) int {
	status, _ := s.deliver(path, body, sign(body, suiteStreamSecret))
	return status
}

// deliver posts a hook body with whatever signature it was given.
func (s *RouterSuite) deliver(path, body, signature string) (int, string) {
	response, err := s.server.Client().Do(s.request(path, body, signature))
	s.Require().NoError(err)
	defer response.Body.Close()

	answered, err := readAll(response)
	s.Require().NoError(err)
	return response.StatusCode, string(answered)
}

func (s *RouterSuite) request(path, body, signature string) *http.Request {
	request, err := http.NewRequest(http.MethodPost, s.server.URL+path, bytes.NewReader([]byte(body)))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(signatureHeader, signature)
	return request
}

// sign is the HMAC Stream signs a delivery with.
func sign(body, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(body))
	return hex.EncodeToString(mac.Sum(nil))
}
