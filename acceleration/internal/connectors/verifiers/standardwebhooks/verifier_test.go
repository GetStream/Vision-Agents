package standardwebhooks_test

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/standardwebhooks"
)

// key is the subscription's signing key in these tests: 32 made-up bytes, the size of Linq's
// example secret, whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw7Jxx2Oll+OE=
// (https://docs.linqapp.com/guides/webhooks/index.md).
var key = []byte("synthetic-linq-signing-key-32byt")

// secret is key as Linq shows it: whsec_ and base64 (Standard Webhooks, «Signature scheme»).
var secret = []byte("whsec_" + base64.StdEncoding.EncodeToString(key))

// The core's synthetic Linq events (internal/connectors/core/testdata/recorded).
var recorded = filepath.Join("..", "..", "core", "testdata", "recorded")

// VerifierSuite runs standard_webhooks with the built-in Linq manifest (providers/linq.yaml).
// Requests are signed as the specification says
// (https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md):
// v1, and the base64 HMAC-SHA256 of {webhook-id}.{webhook-timestamp}.{body}.
type VerifierSuite struct {
	suite.Suite
	verifier *standardwebhooks.Verifier
	linq     core.Manifest
}

func TestVerifierSuite(t *testing.T) {
	suite.Run(t, new(VerifierSuite))
}

func (s *VerifierSuite) SetupSuite() {
	s.verifier = standardwebhooks.New()
	raw, err := providers.FS.ReadFile("linq.yaml")
	s.Require().NoError(err)
	s.linq, err = core.ParseManifest(raw)
	s.Require().NoError(err)
}

func (s *VerifierSuite) TestItIsRegisteredUnderItsKind() {
	s.Equal("standard_webhooks", s.verifier.Name())
}

func (s *VerifierSuite) TestASignedMessageIsOneInboundMessage() {
	body := s.event("linq.received.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.linq, secret)

	s.Require().NoError(err)
	s.Equal(core.VerifiedEvent{Messages: []core.InboundMessage{{
		ConnectorID:       "linq",
		ProviderUnitID:    "+12025550100",
		ThreadKey:         "00000000-0000-4000-8000-0000000000c1",
		AuthorID:          "+12025550199",
		Text:              "Hi, is my order ready?",
		ProviderMessageID: "00000000-0000-4000-8000-0000000000e1",
		Raw:               body,
	}}}, event)
}

// Linq's message.sent is the line's own reply, «Outbound message confirmed as sent from your
// phone number» (https://docs.linqapp.com/channel/imessage/guides/webhooks/events/index.md).
func (s *VerifierSuite) TestTheLinesOwnSentMessageIsNotRead() {
	body := s.event("linq.sent.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.linq, secret)

	s.Require().NoError(err)
	s.Empty(event.Messages)
	s.Equal([]string{"match $.data.direction"}, event.Skipped, "the rule that skipped it, for the events route to log (AI-990 F21)")
}

// A message.received that says it is outbound is not a person's: the manifest matches
// data.direction too, as internal/channels/linq.go reads it.
func (s *VerifierSuite) TestAReceivedEventThatIsOutboundIsNotRead() {
	body := bytes.Replace(s.event("linq.received.json"), []byte(`"direction":"inbound"`), []byte(`"direction":"outbound"`), 1)

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.linq, secret)

	s.Require().NoError(err)
	s.Empty(event.Messages)
}

// The specification: the secret is «base64 encoded, prefixed with whsec_»;
// internal/channels/linq.go also takes it without the prefix.
func (s *VerifierSuite) TestASecretWithoutItsPrefixIsTheSameSecret() {
	body := s.event("linq.received.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.linq, []byte(base64.StdEncoding.EncodeToString(key)))

	s.Require().NoError(err)
	s.Len(event.Messages, 1)
}

// «Multiple signatures are space delimited» while a secret is rotated: any one may match.
func (s *VerifierSuite) TestOneMatchingSignatureAmongSeveralIsEnough() {
	body := s.event("linq.received.json")
	request := s.signed(body, time.Now())
	request.Header.Set("Webhook-Signature", "v1,"+base64.StdEncoding.EncodeToString([]byte("an old key's signature"))+" "+
		request.Header.Get("Webhook-Signature"))

	event, err := s.verifier.Verify(request, body, s.linq, secret)

	s.Require().NoError(err)
	s.Len(event.Messages, 1)
}

func (s *VerifierSuite) TestAnUnsignedRequestIsRefused() {
	body := s.event("linq.received.json")
	request := s.signed(body, time.Now())
	request.Header.Del("Webhook-Signature")

	event, err := s.verifier.Verify(request, body, s.linq, secret)

	s.ErrorIs(err, standardwebhooks.ErrUnsigned)
	s.Zero(event)
}

func (s *VerifierSuite) TestASignatureWithAnotherSecretIsRefused() {
	body := s.event("linq.received.json")
	other := []byte("whsec_" + base64.StdEncoding.EncodeToString([]byte("another-synthetic-signing-key-32")))

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.linq, other)

	s.ErrorIs(err, standardwebhooks.ErrUnsigned)
	s.Zero(event)
}

func (s *VerifierSuite) TestABodyChangedAfterSigningIsRefused() {
	body := s.event("linq.received.json")
	request := s.signed(body, time.Now())
	changed := bytes.Replace(body, []byte("is my order ready?"), []byte("send it to my new address"), 1)

	_, err := s.verifier.Verify(request, changed, s.linq, secret)

	s.ErrorIs(err, standardwebhooks.ErrUnsigned)
}

// The id is signed, so a recorded delivery replayed under another id breaks the signature.
func (s *VerifierSuite) TestAnIDChangedAfterSigningIsRefused() {
	body := s.event("linq.received.json")
	request := s.signed(body, time.Now())
	request.Header.Set("Webhook-Id", "msg_another")

	_, err := s.verifier.Verify(request, body, s.linq, secret)

	s.ErrorIs(err, standardwebhooks.ErrUnsigned)
}

// The timestamp is signed, so moving it to pass the age check breaks the signature.
func (s *VerifierSuite) TestATimestampChangedAfterSigningIsRefused() {
	body := s.event("linq.received.json")
	request := s.signed(body, time.Now().Add(-10*time.Minute))
	request.Header.Set("Webhook-Timestamp", strconv.FormatInt(time.Now().Unix(), 10))

	_, err := s.verifier.Verify(request, body, s.linq, secret)

	s.ErrorIs(err, standardwebhooks.ErrUnsigned)
}

// The specification makes webhook-id required, so a request without one is refused even when
// it is signed over an empty id.
func (s *VerifierSuite) TestARequestWithoutAnIDIsRefused() {
	body := s.event("linq.received.json")
	request := s.signedAs("", body, time.Now(), key)
	request.Header.Del("Webhook-Id")

	_, err := s.verifier.Verify(request, body, s.linq, secret)

	s.ErrorIs(err, standardwebhooks.ErrUnsigned)
}

// Linq: «Reject if the timestamp is more than 5 minutes old»
// (https://docs.linqapp.com/guides/webhooks/index.md); linq.yaml's max_age.
func (s *VerifierSuite) TestARequestSignedMoreThanFiveMinutesAgoIsRefused() {
	body := s.event("linq.received.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now().Add(-5*time.Minute-time.Second)), body, s.linq, secret)

	s.ErrorIs(err, standardwebhooks.ErrStale)
	s.Zero(event)
}

func (s *VerifierSuite) TestARequestSignedMoreThanFiveMinutesAheadIsRefused() {
	body := s.event("linq.received.json")

	_, err := s.verifier.Verify(s.signed(body, time.Now().Add(5*time.Minute+time.Second)), body, s.linq, secret)

	s.ErrorIs(err, standardwebhooks.ErrStale)
}

func (s *VerifierSuite) TestARequestSignedWithinFiveMinutesIsTaken() {
	body := s.event("linq.received.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now().Add(-4*time.Minute)), body, s.linq, secret)

	s.Require().NoError(err)
	s.Len(event.Messages, 1)
}

func (s *VerifierSuite) TestATimestampThatIsNotUnixSecondsIsRefused() {
	body := s.event("linq.received.json")
	request := s.signed(body, time.Now())
	request.Header.Set("Webhook-Timestamp", time.Now().Format(time.RFC3339))

	_, err := s.verifier.Verify(request, body, s.linq, secret)

	s.ErrorIs(err, standardwebhooks.ErrUnsigned)
}

// v1a is the specification's asymmetric signature, which Linq does not name.
func (s *VerifierSuite) TestASignatureOfAnotherVersionIsRefused() {
	body := s.event("linq.received.json")
	request := s.signed(body, time.Now())
	written := request.Header.Get("Webhook-Signature")
	request.Header.Set("Webhook-Signature", "v1a,"+written[len("v1,"):])

	_, err := s.verifier.Verify(request, body, s.linq, secret)

	s.ErrorIs(err, standardwebhooks.ErrUnsigned)
}

// An HMAC under an empty key is one anybody can compute, so no secret takes nothing, even a
// request signed with the empty key.
func (s *VerifierSuite) TestNoSecretTakesNothing() {
	body := s.event("linq.received.json")

	for _, none := range [][]byte{nil, []byte("whsec_")} {
		_, err := s.verifier.Verify(s.signedWith(body, time.Now(), nil), body, s.linq, none)
		s.ErrorIs(err, standardwebhooks.ErrUnsigned, "%q", none)
	}
}

func (s *VerifierSuite) TestASecretThatIsNotBase64IsRefused() {
	body := s.event("linq.received.json")

	_, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.linq, []byte("whsec_not base64!"))

	s.ErrorIs(err, standardwebhooks.ErrUnsigned)
}

func (s *VerifierSuite) TestASignedBodyThatIsNotJSONIsNothingToActOn() {
	body := []byte("not-json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.linq, secret)

	s.Require().NoError(err)
	s.Zero(event)
}

func (s *VerifierSuite) TestAManifestWithAnotherVerifierKindIsRefused() {
	body := s.event("linq.received.json")
	other := s.linq
	channel := *other.Channel
	channel.Verifier.Kind = core.VerifierHMACHeader
	other.Channel = &channel

	_, err := s.verifier.Verify(s.signed(body, time.Now()), body, other, secret)

	s.ErrorContains(err, "has no standard_webhooks verifier")
}

func (s *VerifierSuite) event(name string) []byte {
	raw, err := os.ReadFile(filepath.Join(recorded, name))
	s.Require().NoError(err)
	return raw
}

func (s *VerifierSuite) signed(body []byte, at time.Time) *http.Request {
	return s.signedWith(body, at, key)
}

func (s *VerifierSuite) signedWith(body []byte, at time.Time, key []byte) *http.Request {
	return s.signedAs("msg_synthetic", body, at, key)
}

// signedAs is delivery id of body signed at at with key, as Linq sends one.
func (s *VerifierSuite) signedAs(id string, body []byte, at time.Time, key []byte) *http.Request {
	timestamp := strconv.FormatInt(at.Unix(), 10)
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(id + "." + timestamp + "."))
	mac.Write(body)
	request := httptest.NewRequest(http.MethodPost, "/v1/connectors/events/linq/synthetic", bytes.NewReader(body))
	request.Header.Set("Webhook-Id", id)
	request.Header.Set("Webhook-Timestamp", timestamp)
	request.Header.Set("Webhook-Signature", "v1,"+base64.StdEncoding.EncodeToString(mac.Sum(nil)))
	return request
}
