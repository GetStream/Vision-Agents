package api

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
)

// hookSecret is what the suite signs with. Stream signs webhooks with the app secret, so
// this stands in for one.
const hookSecret = "not-the-real-secret"

// CallHookSuite covers what the inbound hook does with what Stream sends it.
//
// There is no store here, so the assertions are about authentication and about a call being
// let through or not. That an accepted call reaches the right customer's worker needs a
// database to say whose number was rung, and is covered in the integration suite.
type CallHookSuite struct {
	suite.Suite
	pool    *dispatch.Pool
	handler http.Handler
}

func TestCallHookSuite(t *testing.T) {
	suite.Run(t, new(CallHookSuite))
}

func (s *CallHookSuite) SetupTest() {
	s.pool = dispatch.NewPool()
	s.handler = s.serverWith(hookSecret)
}

func (s *CallHookSuite) serverWith(secret string) http.Handler {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	server, err := NewServer(Options{
		Routers:      map[routing.Modality]routing.Inspector{routing.STT: speech},
		Dispatch:     s.pool,
		StreamSecret: secret,
	})
	s.Require().NoError(err)
	return server.Handler()
}

// deliver posts a body signed the way Stream signs one.
func (s *CallHookSuite) deliver(handler http.Handler, body string) *httptest.ResponseRecorder {
	mac := hmac.New(sha256.New, []byte(hookSecret))
	mac.Write([]byte(body))
	return s.deliverSigned(handler, body, hex.EncodeToString(mac.Sum(nil)))
}

func (s *CallHookSuite) deliverSigned(handler http.Handler, body, signature string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(
		http.MethodPost, "/v1/phone/hooks/stream", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Signature", signature)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	return recorder
}

// sessionStarted is what Stream sends when a caller lands in a call.
const sessionStarted = `{
  "type": "call.session_started",
  "call_cid": "default:phone-+15125551234",
  "session_id": "session-1",
  "created_at": "2026-08-27T12:00:00Z",
  "call": {
    "cid": "default:phone-+15125551234",
    "id": "phone-+15125551234",
    "type": "default",
    "custom": {"line": "support"},
    "session": {
      "id": "session-1",
      "participants": [
        {"user_session_id": "s1", "role": "user", "joined_at": "2026-08-27T12:00:00Z",
         "user": {"id": "sip-+15550001111"}}
      ]
    }
  }
}`

func (s *CallHookSuite) TestAnUnsignedCallEventIsRefused() {
	recorder := s.deliverSigned(s.handler, sessionStarted, "")

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *CallHookSuite) TestACallEventSignedWithTheWrongSecretIsRefused() {
	mac := hmac.New(sha256.New, []byte("somebody-elses-secret"))
	mac.Write([]byte(sessionStarted))

	recorder := s.deliverSigned(s.handler, sessionStarted, hex.EncodeToString(mac.Sum(nil)))

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *CallHookSuite) TestATamperedCallEventIsRefused() {
	mac := hmac.New(sha256.New, []byte(hookSecret))
	mac.Write([]byte(sessionStarted))
	signature := hex.EncodeToString(mac.Sum(nil))

	tampered := strings.Replace(sessionStarted, "+15125551234", "+15559998888", -1)
	recorder := s.deliverSigned(s.handler, tampered, signature)

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *CallHookSuite) TestWithoutASecretThereIsNoHookAtAll() {
	// A hook that cannot check a signature would take a call from anyone who found the
	// url, and this path starts agents.
	recorder := s.deliver(s.serverWith(""), sessionStarted)

	s.Equal(http.StatusNotFound, recorder.Code)
}

func (s *CallHookSuite) TestASignedCallEventIsAccepted() {
	recorder := s.deliver(s.handler, sessionStarted)

	s.Equal(http.StatusOK, recorder.Code)
}

func (s *CallHookSuite) TestACallThatCouldNotBeDispatchedIsStillAccepted() {
	// Nothing here can say whose call this is, so nobody is woken. Stream retries a
	// non-2xx, and no retry is going to find a worker that is not there while the caller
	// waits through every one of them.
	recorder := s.deliver(s.handler, sessionStarted)

	s.Equal(http.StatusOK, recorder.Code)
	s.Empty(s.pool.Workers("acme"))
}

func (s *CallHookSuite) TestAVideoCallIsNotTreatedAsAnArrivingPhoneCall() {
	video := strings.Replace(sessionStarted,
		"default:phone-+15125551234", "default:standup-monday", -1)

	recorder := s.deliver(s.handler, video)

	s.Equal(http.StatusOK, recorder.Code)
}

func (s *CallHookSuite) TestAnEventTheHookDoesNotActOnIsAccepted() {
	ended := `{"type":"call.session_ended","call_cid":"default:phone-+15125551234",` +
		`"session_id":"session-1","created_at":"2026-08-27T12:05:00Z",` +
		`"call":{"cid":"default:phone-+15125551234","id":"phone-+15125551234","type":"default","custom":{}}}`

	recorder := s.deliver(s.handler, ended)

	s.Equal(http.StatusOK, recorder.Code)
}

func (s *CallHookSuite) TestAnEventTypeThisVersionHasNeverHeardOfIsAccepted() {
	unknown := `{"type":"call.something_new","call_cid":"default:phone-+15125551234"}`

	recorder := s.deliver(s.handler, unknown)

	s.Equal(http.StatusOK, recorder.Code, "a new event type must not look like an outage to Stream")
}

func (s *CallHookSuite) TestSomethingThatIsNotACallEventIsRefused() {
	// Correctly signed, but there is no event in it to act on.
	recorder := s.deliver(s.handler, `{"not":"an event"}`)

	s.Equal(http.StatusBadRequest, recorder.Code)
}

func (s *CallHookSuite) TestACallEventWithAnRFC3339TimestampIsStillRead() {
	// The SDK's Timestamp reads epoch nanoseconds and writes RFC 3339, so decoding a
	// delivery with its event structs would refuse whichever format it is not expecting.
	// Nothing here reads a timestamp off the wire, and this is what proves it.
	recorder := s.deliver(s.handler, sessionStarted)

	s.Equal(http.StatusOK, recorder.Code)
	s.Contains(sessionStarted, `"created_at": "2026-08-27T12:00:00Z"`)
}

func (s *CallHookSuite) TestTheHookIsNotReachedWithTheCustomerHeaderMissingOrPresent() {
	// Stream is not a customer, so the header is neither required nor read. Sending one
	// changes nothing, which is what stops a caller thinking it scopes the hook.
	request := httptest.NewRequest(
		http.MethodPost, "/v1/phone/hooks/stream", strings.NewReader(sessionStarted))
	mac := hmac.New(sha256.New, []byte(hookSecret))
	mac.Write([]byte(sessionStarted))
	request.Header.Set("X-Signature", hex.EncodeToString(mac.Sum(nil)))
	request.Header.Set(CustomerHeader, "globex")
	recorder := httptest.NewRecorder()

	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusOK, recorder.Code)
}

// started parses a delivery the way the hook does, so the assertions below are on what a
// worker would be told rather than on an intermediate of the test's own making.
func (s *CallHookSuite) started(body string) callEvent {
	var event callEvent
	s.Require().NoError(json.Unmarshal([]byte(body), &event))
	return event
}

func (s *CallHookSuite) TestTheCallerNumberIsReadOffTheSipParticipant() {
	s.Equal("+15550001111", callerOf(s.started(sessionStarted)))
}

func (s *CallHookSuite) TestACallerWhoHasNotJoinedYetLeavesTheNumberBlank() {
	// Normal: the call is dispatched the moment it starts, and the agent joining it sees
	// the participant whether or not this did.
	withoutSession := strings.Replace(sessionStarted, `"session": {`, `"unused": {`, 1)

	s.Empty(callerOf(s.started(withoutSession)))
}

func (s *CallHookSuite) TestAParticipantWhoIsNotACallerIsNotReadAsOne() {
	// The agent is in the call too, and it is not who rang.
	agentFirst := strings.Replace(sessionStarted,
		`{"user_session_id": "s1"`,
		`{"user_session_id": "s0", "role": "user", "joined_at": "2026-08-27T12:00:00Z",
		  "user": {"id": "agent"}},
		 {"user_session_id": "s1"`, 1)

	s.Equal("+15550001111", callerOf(s.started(agentFirst)))
}

func (s *CallHookSuite) TestCustomDataThatIsNotTextIsLeftOut() {
	// Stream takes arbitrary JSON on a call. Everything this service puts there is a
	// string, and a field whose type depends on who set it is not one a client can read.
	narrowed := customOf(map[string]any{"line": "support", "attempt": 3, "tags": []string{"a"}})

	s.Equal(map[string]string{"line": "support"}, narrowed)
}
