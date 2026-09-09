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

// MessageHookSuite covers what the message hook does with what Stream sends it.
//
// There is no store here, so the assertions are about authentication and about which
// messages are let through to be routed at all. That a message reaches the right customer's
// worker needs a database to say whose channel it is, and is covered in the integration
// suite.
type MessageHookSuite struct {
	suite.Suite
	pool    *dispatch.Pool
	handler http.Handler
}

func TestMessageHookSuite(t *testing.T) {
	suite.Run(t, new(MessageHookSuite))
}

func (s *MessageHookSuite) SetupTest() {
	s.pool = dispatch.NewPool()
	s.handler = s.serverWith(hookSecret)
}

func (s *MessageHookSuite) serverWith(secret string) http.Handler {
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
func (s *MessageHookSuite) deliver(handler http.Handler, body string) *httptest.ResponseRecorder {
	mac := hmac.New(sha256.New, []byte(hookSecret))
	mac.Write([]byte(body))
	return s.deliverSigned(handler, body, hex.EncodeToString(mac.Sum(nil)))
}

func (s *MessageHookSuite) deliverSigned(
	handler http.Handler, body, signature string,
) *httptest.ResponseRecorder {
	request := httptest.NewRequest(
		http.MethodPost, "/v1/chat/hooks/stream", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Signature", signature)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	return recorder
}

// typed is what Stream sends when a person writes to an agent's channel.
const typed = `{
  "type": "message.new",
  "cid": "agent:call-1",
  "channel_id": "call-1",
  "channel_type": "agent",
  "created_at": "2026-09-08T12:00:00Z",
  "message": {
    "id": "message-1",
    "text": "is my invoice reissuable to another company?",
    "user": {"id": "sam", "name": "Sam"}
  }
}`

func (s *MessageHookSuite) TestAnUnsignedMessageEventIsRefused() {
	recorder := s.deliverSigned(s.handler, typed, "")

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *MessageHookSuite) TestAMessageEventSignedWithTheWrongSecretIsRefused() {
	mac := hmac.New(sha256.New, []byte("somebody-elses-secret"))
	mac.Write([]byte(typed))

	recorder := s.deliverSigned(s.handler, typed, hex.EncodeToString(mac.Sum(nil)))

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *MessageHookSuite) TestATamperedMessageEventIsRefused() {
	mac := hmac.New(sha256.New, []byte(hookSecret))
	mac.Write([]byte(typed))
	signature := hex.EncodeToString(mac.Sum(nil))

	tampered := strings.Replace(typed, "call-1", "somebody-elses-call", -1)
	recorder := s.deliverSigned(s.handler, tampered, signature)

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *MessageHookSuite) TestWithoutASecretThereIsNoHookAtAll() {
	// A hook that cannot check a signature would answer anyone who found the url, and this
	// path starts agents.
	recorder := s.deliver(s.serverWith(""), typed)

	s.Equal(http.StatusNotFound, recorder.Code)
}

func (s *MessageHookSuite) TestASignedMessageEventIsAccepted() {
	recorder := s.deliver(s.handler, typed)

	s.Equal(http.StatusOK, recorder.Code)
}

func (s *MessageHookSuite) TestAMessageNobodyCouldAnswerIsStillAccepted() {
	// Nothing here can say whose channel this is, so nobody is woken. Stream retries a
	// non-2xx, and no retry is going to find a worker that is not there.
	recorder := s.deliver(s.handler, typed)

	s.Equal(http.StatusOK, recorder.Code)
	s.Empty(s.pool.Workers("acme"))
}

// written parses a delivery the way the hook does, so the assertions below are on what the
// hook would act on rather than on an intermediate of the test's own making.
func (s *MessageHookSuite) written(body string) messageEvent {
	var event messageEvent
	s.Require().NoError(json.Unmarshal([]byte(body), &event))
	return event
}

func (s *MessageHookSuite) TestSomethingSomebodyTypedToAnAgentIsAddressedToIt() {
	s.True(addressed(s.written(typed)))
}

func (s *MessageHookSuite) TestAMessageOutsideAnAgentChannelIsNotAddressedToOne() {
	// Every message in the app arrives here, and a team's own channel is not a way to
	// reach an agent.
	elsewhere := strings.Replace(typed, `"channel_type": "agent"`, `"channel_type": "messaging"`, 1)

	s.False(addressed(s.written(elsewhere)))
}

func (s *MessageHookSuite) TestSpeechTheAgentHasAlreadyAnsweredIsNotAddressedToIt() {
	// The agent writes what it heard into the same channel. Answering that would be the
	// agent replying to a question it answered as it was asked.
	heard := strings.Replace(typed,
		`"user": {"id": "sam", "name": "Sam"}`,
		`"user": {"id": "sam", "name": "Sam"}, "custom": {"source": "speech"}`, 1)

	s.False(addressed(s.written(heard)))
}

func (s *MessageHookSuite) TestTheAgentsOwnReplyIsNotAddressedToIt() {
	// Otherwise every answer is a new question, and the agent talks to itself forever.
	reply := strings.Replace(typed,
		`"user": {"id": "sam", "name": "Sam"}`,
		`"user": {"id": "support-agent"}, "custom": {"source": "agent", "generating": false}`, 1)

	s.False(addressed(s.written(reply)))
}

func (s *MessageHookSuite) TestAReplyStillBeingWrittenIsNotAddressedToTheAgentEither() {
	// A streamed reply is stored the moment it starts and updated as it goes, so the
	// first piece of every answer arrives here while the model is still writing it.
	partial := strings.Replace(typed,
		`"user": {"id": "sam", "name": "Sam"}`,
		`"user": {"id": "support-agent"}, "custom": {"source": "agent", "generating": true}`, 1)

	s.False(addressed(s.written(partial)))
}

func (s *MessageHookSuite) TestAMessageWithNothingWrittenInItIsNotAddressedToAnAgent() {
	// An attachment on its own is this: there is nothing to answer.
	empty := strings.Replace(typed,
		`"text": "is my invoice reissuable to another company?"`, `"text": ""`, 1)

	s.False(addressed(s.written(empty)))
}

func (s *MessageHookSuite) TestAnEventTypeThisVersionHasNeverHeardOfIsAccepted() {
	unknown := `{"type":"message.something_new","channel_type":"agent","channel_id":"call-1"}`

	recorder := s.deliver(s.handler, unknown)

	s.Equal(http.StatusOK, recorder.Code, "a new event type must not look like an outage to Stream")
}

func (s *MessageHookSuite) TestAnEventTheHookDoesNotActOnIsAccepted() {
	updated := `{"type":"message.updated","channel_type":"agent","channel_id":"call-1",` +
		`"message":{"id":"message-1","text":"edited","user":{"id":"sam"}}}`

	recorder := s.deliver(s.handler, updated)

	s.Equal(http.StatusOK, recorder.Code)
}

func (s *MessageHookSuite) TestSomethingThatIsNotAMessageEventIsRefused() {
	// Correctly signed, but there is no event in it to act on.
	recorder := s.deliver(s.handler, `{"not":"an event"}`)

	s.Equal(http.StatusBadRequest, recorder.Code)
}

func (s *MessageHookSuite) TestTheHookIsNotReachedWithTheCustomerHeaderMissingOrPresent() {
	// Stream is not a customer, so the header is neither required nor read.
	request := httptest.NewRequest(
		http.MethodPost, "/v1/chat/hooks/stream", strings.NewReader(typed))
	mac := hmac.New(sha256.New, []byte(hookSecret))
	mac.Write([]byte(typed))
	request.Header.Set("X-Signature", hex.EncodeToString(mac.Sum(nil)))
	request.Header.Set(CustomerHeader, "globex")
	recorder := httptest.NewRecorder()

	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusOK, recorder.Code)
}
