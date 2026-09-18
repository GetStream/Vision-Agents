package guardrail

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// secret is the app secret both sides sign with.
const secret = "super-secret"

// The scheme as one fixed vector. sdks/go/agents, which is where a customer gets the
// verifier from, pins the same three inputs to the same digest in a test of its own. The
// two sides cannot import each other - one is published and the other is internal - so
// this literal is what holds them to the same bytes: change the signing scheme and both
// tests fail, which is the point.
const (
	vectorTimestamp = "1789000000"
	vectorBody      = `{"customer_id":"acme","text":"how do I make a pizza"}`
	vectorSignature = "7b994e1f8d9d21f421462dff92f0f9c4a5a025d3a26454eee7135657ad672278"
)

// verify is what a customer's server does with a request, written out here rather than
// called from the SDK so that this test proves the recipe the SDK documents rather than
// proving the SDK agrees with itself.
func verify(timestamp, signature string, body []byte) bool {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(timestamp))
	mac.Write([]byte("."))
	mac.Write(body)
	return hmac.Equal([]byte(hex.EncodeToString(mac.Sum(nil))), []byte(signature))
}

// server stands in for a customer's own endpoint. It checks the signature the way a
// customer would and records whether it was satisfied.
type server struct {
	http *httptest.Server

	answer  Answer
	status  int
	asked   Ask
	signed  bool
	stamped string
	calls   int
	replied string
}

func newServer() *server {
	stub := &server{answer: Answer{Allow: true}, status: http.StatusOK}
	stub.http = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		stub.calls++
		body, _ := io.ReadAll(r.Body)
		stub.stamped = r.Header.Get(TimestampHeader)
		stub.signed = verify(stub.stamped, r.Header.Get(SignatureHeader), body)
		_ = json.Unmarshal(body, &stub.asked)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(stub.status)
		if stub.replied != "" {
			_, _ = w.Write([]byte(stub.replied))
			return
		}
		_ = json.NewEncoder(w).Encode(stub.answer)
	}))
	return stub
}

type WebhookSuite struct {
	suite.Suite
	ctx    context.Context
	server *server
}

func TestWebhookSuite(t *testing.T) {
	suite.Run(t, new(WebhookSuite))
}

func (s *WebhookSuite) SetupTest() {
	s.ctx = context.Background()
	s.server = newServer()
	s.T().Cleanup(s.server.http.Close)
}

// guardrail builds a webhook guardrail pointed at the stub server.
func (s *WebhookSuite) guardrail() Guardrail {
	screening, err := New(s.ctx, Policy{
		Kind:    KindWebhook,
		Mode:    ModeParallel,
		URL:     s.server.http.URL,
		Refusal: "I can only help with questions about Stream.",
		Text:    "My server decides.",
	}, Deps{
		Owner: routing.Owner{
			CustomerID: "acme", AgentID: "support", CallID: "call-9",
		},
		Secret: secret,
		Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = screening.Close() })
	return screening
}

func (s *WebhookSuite) TestTheServerDecidesAndTheRequestIsSignedForIt() {
	s.server.answer = Answer{Allow: true}

	verdict, err := s.guardrail().Check(s.ctx, "turn-1", "how do I install stream-chat-react")
	s.Require().NoError(err)

	s.True(verdict.Allowed)
	s.True(s.server.signed, "the signature a customer is told to check did not verify")
	s.Equal("acme", s.server.asked.CustomerID)
	s.Equal("support", s.server.asked.AgentID)
	s.Equal("call-9", s.server.asked.CallID)
	s.Equal("turn-1", s.server.asked.TurnID)
	s.Equal("user", s.server.asked.Role)
	s.Equal("how do I install stream-chat-react", s.server.asked.Text)
	s.Equal("My server decides.", s.server.asked.Policy)
}

func (s *WebhookSuite) TestARefusalMayCarryTheServersOwnWords() {
	// A server knows things about who is asking that a file written in advance does not,
	// so it may say something more useful than the policy's one line.
	s.server.answer = Answer{
		Allow:   false,
		Reason:  "this customer is not on the plan that includes billing questions",
		Message: "Billing questions go to your account manager.",
	}

	verdict, err := s.guardrail().Check(s.ctx, "turn-1", "can you refund me")
	s.Require().NoError(err)

	s.False(verdict.Allowed)
	s.Equal("Billing questions go to your account manager.", verdict.Refusal)
	s.Contains(verdict.Reason, "not on the plan")
}

func (s *WebhookSuite) TestARefusalWithNoWordsFallsBackToThePolicysOwn() {
	s.server.answer = Answer{Allow: false}

	screening := s.guardrail()
	verdict, err := screening.Check(s.ctx, "turn-1", "how do I make a pizza")
	s.Require().NoError(err)

	s.False(verdict.Allowed)
	s.Empty(verdict.Refusal)
	s.Equal("I can only help with questions about Stream.", screening.Policy().Refusal)
}

func (s *WebhookSuite) TestAServerThatFailedIsReportedRatherThanReadAsAVerdict() {
	s.server.status = http.StatusInternalServerError
	s.server.replied = `{"error":"database is down"}`

	_, err := s.guardrail().Check(s.ctx, "turn-1", "anything")

	s.ErrorContains(err, "500")
	s.ErrorContains(err, "database is down")
}

func (s *WebhookSuite) TestAnAnswerThatWillNotParseIsAFailureRatherThanAnAllow() {
	// A body nobody can read has not decided anything, and reading it as an allow would
	// mean a server that started returning HTML silently stopped screening.
	s.server.replied = `<html>maintenance</html>`

	_, err := s.guardrail().Check(s.ctx, "turn-1", "anything")

	s.Require().Error(err)
}

func (s *WebhookSuite) TestAWebhookWithNoSecretToSignWithIsRefusedWhenBuilt() {
	// Unsigned, the server cannot tell our request from anyone who found the url, and what
	// it decides on that request gates an agent's replies.
	_, err := New(s.ctx, Policy{
		Kind: KindWebhook, Mode: ModeParallel, URL: s.server.http.URL, Refusal: "No.",
	}, Deps{Owner: routing.Owner{CustomerID: "acme"}, Logger: slog.New(slog.DiscardHandler)})

	s.ErrorContains(err, "secret")
}

func (s *WebhookSuite) TestTheSchemeIsTheOneCustomersAreToldToVerify() {
	s.Equal(vectorSignature, Sign(secret, vectorTimestamp, []byte(vectorBody)),
		"the signing scheme changed, so every customer's receiver is now rejecting us")
}

func (s *WebhookSuite) TestTheTimestampIsSignedSoARequestCannotBeReplayed() {
	// A signature over the body alone stays valid forever, so anyone who captured one
	// "allow" could replay it to wave through whatever they liked. The timestamp is inside
	// the signed string, so it cannot be moved forward without the secret, and a receiver
	// that rejects an old one is rejecting a replay.
	body := []byte(vectorBody)
	now := strconv.FormatInt(time.Now().Unix(), 10)
	earlier := strconv.FormatInt(time.Now().Add(-10*time.Minute).Unix(), 10)

	s.NotEqual(Sign(secret, now, body), Sign(secret, earlier, body),
		"the same body signs the same whenever it was sent, so age cannot be checked")
	s.False(verify(earlier, Sign(secret, now, body), body),
		"a signature can be paired with a timestamp it was not made for")
}

func (s *WebhookSuite) TestASignatureCoversTheWholeBodyAndOnlyThisSecretMakesIt() {
	body := []byte(vectorBody)
	signature := Sign(secret, vectorTimestamp, body)

	s.True(verify(vectorTimestamp, signature, body))
	s.False(verify(vectorTimestamp, signature, []byte(`{"text":"tampered"}`)),
		"a body that was changed after signing still verifies")
	s.NotEqual(signature, Sign("wrong-secret", vectorTimestamp, body),
		"anyone can produce a valid signature")
	s.False(verify(vectorTimestamp, "", body))
}

func (s *WebhookSuite) TestEveryRequestCarriesTheTimestampItWasSignedWith() {
	// Without the header a receiver has nothing to reproduce the signature from, so the
	// signature it cannot check is a signature it has to ignore.
	_, err := s.guardrail().Check(s.ctx, "turn-1", "anything")
	s.Require().NoError(err)

	s.NotEmpty(s.server.stamped)
	seconds, err := strconv.ParseInt(s.server.stamped, 10, 64)
	s.Require().NoError(err)
	s.WithinDuration(time.Now(), time.Unix(seconds, 0), time.Minute)
}
