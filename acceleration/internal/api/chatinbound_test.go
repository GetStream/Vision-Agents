package api

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatinbound"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// deliveryApp is the deployment's own Stream app in ChatInboundRefusalSuite.
const deliveryApp = 1257545

// ChatInboundRefusalSuite is what the Stream Chat delivery endpoint answers before a message
// reaches anything that answers it: who it refuses, what body it reads, and which app it
// serves. A router with no store is enough, since none of it reads one: the message it signs
// is the agent's own, which is taken and never queued. That messages are queued and answered
// is ChatInboundSuite's.
type ChatInboundRefusalSuite struct {
	suite.Suite
	current, previous ed25519.PrivateKey
	router            *httptest.Server
}

func TestChatInboundRefusalSuite(t *testing.T) {
	suite.Run(t, new(ChatInboundRefusalSuite))
}

func (s *ChatInboundRefusalSuite) SetupTest() {
	s.current = ed25519.NewKeyFromSeed([]byte("the-current-delivery-key-seed-32"))
	s.previous = ed25519.NewKeyFromSeed([]byte("the-previous-delivery-key-seed32"))
	s.router = s.serve(deliveryApp, s.current, s.previous)
}

func (s *ChatInboundRefusalSuite) TestADeliverySignedByStreamIsTaken() {
	s.Equal(http.StatusOK, s.send(s.signed(s.message(deliveryApp), s.current)))
}

func (s *ChatInboundRefusalSuite) TestADeliveryWithNoSignatureIsRefused() {
	request := s.signed(s.message(deliveryApp), s.current)
	request.Header.Del(chatinbound.SignatureHeader)

	s.Equal(http.StatusUnauthorized, s.send(request))
}

func (s *ChatInboundRefusalSuite) TestADeliverySignedWithAnotherKeyIsRefused() {
	stranger := ed25519.NewKeyFromSeed([]byte("somebody-else-s-key-seed-32bytes"))

	s.Equal(http.StatusUnauthorized, s.send(s.signed(s.message(deliveryApp), stranger)))
}

func (s *ChatInboundRefusalSuite) TestAStaleDeliveryIsRefused() {
	s.Equal(http.StatusUnauthorized, s.send(s.signedAt(s.message(deliveryApp), s.current, time.Now().Add(-6*time.Minute))))
}

func (s *ChatInboundRefusalSuite) TestADeliveryFromTheFutureIsRefused() {
	s.Equal(http.StatusUnauthorized, s.send(s.signedAt(s.message(deliveryApp), s.current, time.Now().Add(6*time.Minute))))
}

func (s *ChatInboundRefusalSuite) TestABodyChangedAfterSigningIsRefused() {
	body := s.message(deliveryApp)
	signed := s.signed(body, s.current)
	request := s.post(strings.Replace(body, "this morning", "yesterday", 1))
	request.Header = signed.Header

	s.Equal(http.StatusUnauthorized, s.send(request))
}

// What the message webhook takes, the HMAC under the app's API secret in X-Signature, is
// what a customer's own server can make, so it proves nothing here.
func (s *ChatInboundRefusalSuite) TestADeliverySignedOnlyWithTheAppSecretIsRefused() {
	body := s.message(deliveryApp)
	request := s.post(body)
	request.Header.Set(signatureHeader, streamSignature([]byte(body), "hook-secret"))

	s.Equal(http.StatusUnauthorized, s.send(request))
	s.Equal(http.StatusOK, s.sendTo(chat.MessageHookPath, body, request.Header), "the webhook takes the same signature")
}

// A token of the app's, which its end user's device holds, signs nothing Stream sends.
func (s *ChatInboundRefusalSuite) TestAClientTokenIsRefused() {
	request := s.post(s.message(deliveryApp))
	request.Header.Set("Authorization", "Bearer a-client-token")

	s.Equal(http.StatusUnauthorized, s.send(request))
}

// The app's key is all a server-side caller of the app has; it signs nothing Stream sends.
func (s *ChatInboundRefusalSuite) TestABareAPIKeyIsRefused() {
	request := s.post(s.message(deliveryApp))
	request.Header.Set(auth.APIKeyHeader, "the-apps-key")

	s.Equal(http.StatusUnauthorized, s.send(request))
}

func (s *ChatInboundRefusalSuite) TestTheOldKeyIsTakenWhileBothAreSet() {
	s.Equal(http.StatusOK, s.send(s.signed(s.message(deliveryApp), s.previous)))
}

func (s *ChatInboundRefusalSuite) TestOneValidSignatureAmongTwoIsTaken() {
	body := s.message(deliveryApp)
	request := s.signed(body, s.current)
	stranger := ed25519.NewKeyFromSeed([]byte("somebody-else-s-key-seed-32bytes"))
	forged := s.signed(body, stranger).Header.Get(chatinbound.SignatureHeader)
	request.Header.Set(chatinbound.SignatureHeader, forged+" "+request.Header.Get(chatinbound.SignatureHeader))

	s.Equal(http.StatusOK, s.send(request))
}

// The signature is checked before the body is read: a body nothing could parse, signed by
// nobody, is a stranger and not a bad request.
func (s *ChatInboundRefusalSuite) TestAnUnreadableBodyWithABadSignatureIsUnauthenticated() {
	stranger := ed25519.NewKeyFromSeed([]byte("somebody-else-s-key-seed-32bytes"))

	s.Equal(http.StatusUnauthorized, s.send(s.signed(`{"app_id":`, stranger)))
}

func (s *ChatInboundRefusalSuite) TestASignedBodyThatIsNotJSONIsABadRequest() {
	s.Equal(http.StatusBadRequest, s.send(s.signed(`{"app_id":`, s.current)))
}

func (s *ChatInboundRefusalSuite) TestASignedEmptyObjectIsABadRequest() {
	s.Equal(http.StatusBadRequest, s.send(s.signed(`{}`, s.current)))
}

func (s *ChatInboundRefusalSuite) TestASignedNullIsABadRequest() {
	s.Equal(http.StatusBadRequest, s.send(s.signed(`null`, s.current)))
}

func (s *ChatInboundRefusalSuite) TestASignedArrayIsABadRequest() {
	s.Equal(http.StatusBadRequest, s.send(s.signed(`[]`, s.current)))
}

func (s *ChatInboundRefusalSuite) TestAnAppIDWrittenAsAStringIsABadRequest() {
	body := strings.Replace(s.message(deliveryApp), `"app_id": 1257545`, `"app_id": "1257545"`, 1)

	s.Equal(http.StatusBadRequest, s.send(s.signed(body, s.current)))
}

func (s *ChatInboundRefusalSuite) TestAnAppIDOfZeroIsABadRequest() {
	s.Equal(http.StatusBadRequest, s.send(s.signed(s.message(0), s.current)))
}

func (s *ChatInboundRefusalSuite) TestANegativeAppIDIsABadRequest() {
	s.Equal(http.StatusBadRequest, s.send(s.signed(s.message(-1), s.current)))
}

func (s *ChatInboundRefusalSuite) TestAMessageOfTheWrongShapeIsABadRequest() {
	body := `{"type": "message.new", "app_id": 1257545, "channel_type": "agent", "channel_id": "chat-1", "message": "hello"}`

	s.Equal(http.StatusBadRequest, s.send(s.signed(body, s.current)))
}

// Stream retries anything but a 2xx, and an event this endpoint does not answer is not
// answerable on the second delivery either.
func (s *ChatInboundRefusalSuite) TestAnEventThatIsNotAMessageIsTakenAndIgnored() {
	body := `{"type": "reaction.new", "app_id": 1257545, "channel_type": "agent", "channel_id": "chat-1"}`

	s.Equal(http.StatusOK, s.send(s.signed(body, s.current)))
}

// The limit is the webhook's, a megabyte, and is held before the signature is checked, so a
// stranger cannot make the router read more.
func (s *ChatInboundRefusalSuite) TestADeliveryOverAMegabyteIsTooLarge() {
	body := `{"type": "message.new", "app_id": 1257545, "padding": "` + strings.Repeat("a", hookBodyLimit) + `"}`

	s.Equal(http.StatusRequestEntityTooLarge, s.send(s.signed(body, s.current)))
}

// The content type is the sender's to say; the signature says what the body is.
func (s *ChatInboundRefusalSuite) TestAnotherContentTypeIsTaken() {
	request := s.signed(s.message(deliveryApp), s.current)
	request.Header.Set("Content-Type", "text/plain")

	s.Equal(http.StatusOK, s.send(request))
}

// A message.new names the message the queue holds it under, so one without an id is not a
// message Stream Chat delivers.
func (s *ChatInboundRefusalSuite) TestANewMessageWithoutAnIDIsABadRequest() {
	body := strings.Replace(s.message(deliveryApp), `"id": "message-1", `, "", 1)

	s.Equal(http.StatusBadRequest, s.send(s.signed(body, s.current)))
}

// A delivery the router cannot queue is one Stream Chat must make again, not one it took.
func (s *ChatInboundRefusalSuite) TestAMessageWithNoDatabaseToQueueItInAsksForARetry() {
	body := strings.Replace(s.message(deliveryApp), `, "source": "agent"`, "", 1)

	s.Equal(http.StatusServiceUnavailable, s.send(s.signed(body, s.current)))
}

// A router in deployment mode acts in its own app alone.
func (s *ChatInboundRefusalSuite) TestAMessageInAnotherAppIsNotServedHere() {
	s.Equal(http.StatusNotFound, s.send(s.signed(s.message(deliveryApp+1), s.current)))
}

func (s *ChatInboundRefusalSuite) TestNoKeyAnswers404() {
	s.router = s.serve(deliveryApp)

	s.Equal(http.StatusNotFound, s.send(s.signed(s.message(deliveryApp), s.current)))
}

// Until the deployment learns which app is its own, it cannot tell its own app's message from
// another's, and Stream retries.
func (s *ChatInboundRefusalSuite) TestADeploymentThatDoesNotKnowItsAppYetAsksForARetry() {
	s.router = s.serve(0, s.current)

	request := s.signed(s.message(deliveryApp), s.current)
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())

	s.Equal(http.StatusServiceUnavailable, response.StatusCode)
	s.NotEmpty(response.Header.Get("Retry-After"))
}

// A router with no Stream app of its own serves no app's messages.
func (s *ChatInboundRefusalSuite) TestARouterWithNoStreamAppServesNone() {
	server, err := NewServer(Options{
		Routers:         map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		AuthMode:        auth.Custom,
		Auth:            s.authenticator(),
		ChatInboundKeys: []ed25519.PublicKey{public(s.current)},
		Logger:          slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.router = httptest.NewServer(server.Handler())
	s.T().Cleanup(s.router.Close)

	s.Equal(http.StatusNotFound, s.send(s.signed(s.message(deliveryApp), s.current)))
}

// serve is a router in deployment mode whose own app is app, zero for one it has yet to
// learn, taking deliveries signed with keys.
func (s *ChatInboundRefusalSuite) serve(app int64, keys ...ed25519.PrivateKey) *httptest.Server {
	var held []ed25519.PublicKey
	for _, key := range keys {
		held = append(held, public(key))
	}
	deployment := streamapp.NewDeployment(streamapp.DeploymentOptions{APIKey: "the-apps-key", Secret: "hook-secret", App: app})
	server, err := NewServer(Options{
		Routers:         map[routing.Modality]routing.Inspector{routing.LLM: idleInspector{}},
		AuthMode:        auth.Custom,
		Auth:            s.authenticator(),
		Stream:          streamapp.NewClients(deployment, streamapp.ClientsOptions{}),
		HookSecret:      "hook-secret",
		ChatInboundKeys: held,
		Logger:          slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	router := httptest.NewServer(server.Handler())
	s.T().Cleanup(router.Close)
	return router
}

// authenticator takes a bearer token for an end user's device and the app's key alone for its
// server, the two kinds of caller an app has.
func (s *ChatInboundRefusalSuite) authenticator() auth.Authenticator {
	return auth.Func(func(_ context.Context, r *http.Request) (auth.Principal, error) {
		if r.Header.Get("Authorization") != "" {
			return auth.Principal{AppID: "1257545", OrganizationID: "org-1", Kind: auth.KindAuthenticated, UserID: "sam"}, nil
		}
		if r.Header.Get(auth.APIKeyHeader) != "" {
			return auth.Principal{AppID: "1257545", OrganizationID: "org-1", Kind: auth.KindServer, ServerSide: true}, nil
		}
		return auth.Principal{}, auth.ErrUnauthenticated
	})
}

// message is a message.new in an agent channel, as Stream Chat delivers it for app: the
// agent's own reply, which carries a source (addressed), so the endpoint takes it without
// queueing it.
func (s *ChatInboundRefusalSuite) message(app int64) string {
	return `{
  "type": "message.new",
  "app_id": ` + strconv.FormatInt(app, 10) + `,
  "channel_type": "agent",
  "channel_id": "chat-1",
  "message": {"id": "message-1", "text": "it left this morning", "user": {"id": "agent", "name": "Agent"}, "source": "agent"}
}`
}

func (s *ChatInboundRefusalSuite) signed(body string, key ed25519.PrivateKey) *http.Request {
	return s.signedAt(body, key, time.Now())
}

func (s *ChatInboundRefusalSuite) signedAt(body string, key ed25519.PrivateKey, at time.Time) *http.Request {
	request := s.post(body)
	for name, values := range signDelivery(key, "msg_"+strconv.FormatInt(at.UnixNano(), 36), at, body) {
		request.Header[name] = values
	}
	return request
}

func (s *ChatInboundRefusalSuite) post(body string) *http.Request {
	request, err := http.NewRequest(http.MethodPost, s.router.URL+chatInboundPath, strings.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	return request
}

func (s *ChatInboundRefusalSuite) send(request *http.Request) int {
	request.URL.Host = strings.TrimPrefix(s.router.URL, "http://")
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}

func (s *ChatInboundRefusalSuite) sendTo(path, body string, header http.Header) int {
	request, err := http.NewRequest(http.MethodPost, s.router.URL+path, strings.NewReader(body))
	s.Require().NoError(err)
	request.Header = header
	return s.send(request)
}

// signDelivery is the headers Stream Chat signs a delivery with: Standard Webhooks' v1a, the
// ed25519 signature of {id}.{timestamp}.{body}.
func signDelivery(key ed25519.PrivateKey, id string, at time.Time, body string) http.Header {
	timestamp := strconv.FormatInt(at.Unix(), 10)
	header := http.Header{}
	header.Set(chatinbound.IDHeader, id)
	header.Set(chatinbound.TimestampHeader, timestamp)
	header.Set(chatinbound.SignatureHeader, "v1a,"+base64.StdEncoding.EncodeToString(
		ed25519.Sign(key, []byte(id+"."+timestamp+"."+body))))
	return header
}

func public(key ed25519.PrivateKey) ed25519.PublicKey {
	return key.Public().(ed25519.PublicKey)
}

// ChatInboundLeaseSuite is the queue's lease against the answers it holds a delivery for.
type ChatInboundLeaseSuite struct {
	suite.Suite
}

func TestChatInboundLeaseSuite(t *testing.T) {
	suite.Run(t, new(ChatInboundLeaseSuite))
}

// A thread's answer waits up to askTimeout for the turn and up to askTimeout for the reply,
// then lets go of the turn (answerThread): a lease shorter than that has a second router
// answer the message again (TestAnAnswerThatOutlastsItsLeaseIsAnsweredAgainAndSaysSo).
func (s *ChatInboundLeaseSuite) TestTheLeaseOutlastsTheLongestAnswer() {
	s.Greater(chatinbound.Lease, 2*askTimeout+20*threadTurnPoll)
}
