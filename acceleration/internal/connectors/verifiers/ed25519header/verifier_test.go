package ed25519header_test

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
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
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/ed25519header"
)

// VerifierSuite runs ed25519 with the built-in Telnyx manifest (providers/telnyx.yaml) on
// synthetic events shaped as the example of
// https://developers.telnyx.com/docs/messaging/messages/receiving-webhooks (opened October 8,
// 2026), signed as that page says: the base64 Ed25519 signature of {timestamp}|{body}.
type VerifierSuite struct {
	suite.Suite
	verifier *ed25519header.Verifier
	telnyx   core.Manifest
	// private signs, and public, base64 as Telnyx shows it, is the secret.
	private ed25519.PrivateKey
	public  []byte
}

func TestVerifierSuite(t *testing.T) {
	suite.Run(t, new(VerifierSuite))
}

func (s *VerifierSuite) SetupSuite() {
	s.verifier = ed25519header.New()
	raw, err := providers.FS.ReadFile("telnyx.yaml")
	s.Require().NoError(err)
	s.telnyx, err = core.ParseManifest(raw)
	s.Require().NoError(err)
	public, private, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)
	s.private, s.public = private, []byte(base64.StdEncoding.EncodeToString(public))
}

func (s *VerifierSuite) TestItIsRegisteredUnderItsKind() {
	s.Equal("ed25519", s.verifier.Name())
}

func (s *VerifierSuite) TestASignedMessageIsOneInboundMessage() {
	body := s.event("telnyx.received.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.telnyx, s.public)

	s.Require().NoError(err)
	s.Equal(core.VerifiedEvent{Messages: []core.InboundMessage{{
		ConnectorID:       "telnyx",
		ProviderUnitID:    "+17735550002",
		ThreadKey:         "+13125550001",
		AuthorID:          "+13125550001",
		Text:              "Hello from Telnyx!",
		ProviderMessageID: "00000000-0000-4000-8000-0000000000b1",
		Raw:               body,
	}}}, event)
}

// message.sent is the number's own reply, delivered to the same URL, so the agent's answer is
// not read back.
func (s *VerifierSuite) TestTheNumbersOwnSentMessageIsNotRead() {
	body := s.event("telnyx.sent.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.telnyx, s.public)

	s.Require().NoError(err)
	s.Empty(event.Messages)
	s.Equal([]string{"match $.data.event_type"}, event.Skipped, "the rule that skipped it, for the events route to log (AI-990 F21)")
}

func (s *VerifierSuite) TestAnUnsignedRequestIsRefused() {
	body := s.event("telnyx.received.json")
	request := s.signed(body, time.Now())
	request.Header.Del("Telnyx-Signature-Ed25519")

	event, err := s.verifier.Verify(request, body, s.telnyx, s.public)

	s.ErrorIs(err, ed25519header.ErrUnsigned)
	s.Zero(event)
}

func (s *VerifierSuite) TestASignatureByAnotherKeyIsRefused() {
	body := s.event("telnyx.received.json")
	_, other, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)

	event, err := s.verifier.Verify(s.signedWith(body, time.Now(), other), body, s.telnyx, s.public)

	s.ErrorIs(err, ed25519header.ErrUnsigned)
	s.Zero(event)
}

func (s *VerifierSuite) TestABodyChangedAfterSigningIsRefused() {
	body := s.event("telnyx.received.json")
	request := s.signed(body, time.Now())
	changed := bytes.Replace(body, []byte("Hello from Telnyx!"), []byte("STOP"), 1)

	_, err := s.verifier.Verify(request, changed, s.telnyx, s.public)

	s.ErrorIs(err, ed25519header.ErrUnsigned)
}

// The timestamp is signed, so moving it to pass the age check breaks the signature.
func (s *VerifierSuite) TestATimestampChangedAfterSigningIsRefused() {
	body := s.event("telnyx.received.json")
	request := s.signed(body, time.Now().Add(-10*time.Minute))
	request.Header.Set("Telnyx-Timestamp", strconv.FormatInt(time.Now().Unix(), 10))

	_, err := s.verifier.Verify(request, body, s.telnyx, s.public)

	s.ErrorIs(err, ed25519header.ErrUnsigned)
}

// Telnyx: «reject webhooks where telnyx-timestamp is more than 5 minutes old»; telnyx.yaml's
// max_age.
func (s *VerifierSuite) TestARequestSignedMoreThanFiveMinutesAgoIsRefused() {
	body := s.event("telnyx.received.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now().Add(-5*time.Minute-time.Second)), body, s.telnyx, s.public)

	s.ErrorIs(err, ed25519header.ErrStale)
	s.Zero(event)
}

func (s *VerifierSuite) TestARequestSignedMoreThanFiveMinutesAheadIsRefused() {
	body := s.event("telnyx.received.json")

	_, err := s.verifier.Verify(s.signed(body, time.Now().Add(5*time.Minute+time.Second)), body, s.telnyx, s.public)

	s.ErrorIs(err, ed25519header.ErrStale)
}

func (s *VerifierSuite) TestARequestSignedWithinFiveMinutesIsTaken() {
	body := s.event("telnyx.received.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now().Add(-4*time.Minute)), body, s.telnyx, s.public)

	s.Require().NoError(err)
	s.Len(event.Messages, 1)
}

func (s *VerifierSuite) TestATimestampThatIsNotUnixSecondsIsRefused() {
	body := s.event("telnyx.received.json")
	request := s.signed(body, time.Now())
	request.Header.Set("Telnyx-Timestamp", time.Now().Format(time.RFC3339))

	_, err := s.verifier.Verify(request, body, s.telnyx, s.public)

	s.ErrorIs(err, ed25519header.ErrUnsigned)
}

// A public key is 32 bytes (RFC 8032, section 5.1.5): no key, one that is not base64 and one
// of another size take nothing.
func (s *VerifierSuite) TestASecretThatIsNoPublicKeyTakesNothing() {
	body := s.event("telnyx.received.json")

	for _, secret := range [][]byte{nil, []byte("not base64!"), []byte(base64.StdEncoding.EncodeToString([]byte("short")))} {
		_, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.telnyx, secret)
		s.ErrorIs(err, ed25519header.ErrUnsigned, "%q", secret)
	}
}

func (s *VerifierSuite) TestASignatureThatIsNotBase64IsRefused() {
	body := s.event("telnyx.received.json")
	request := s.signed(body, time.Now())
	request.Header.Set("Telnyx-Signature-Ed25519", "not base64!")

	_, err := s.verifier.Verify(request, body, s.telnyx, s.public)

	s.ErrorIs(err, ed25519header.ErrUnsigned)
}

func (s *VerifierSuite) TestASignedBodyThatIsNotJSONIsNothingToActOn() {
	body := []byte("not-json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.telnyx, s.public)

	s.Require().NoError(err)
	s.Zero(event)
}

func (s *VerifierSuite) TestAManifestWithAnotherVerifierKindIsRefused() {
	body := s.event("telnyx.received.json")
	other := s.telnyx
	channel := *other.Channel
	channel.Verifier.Kind = core.VerifierHMACHeader
	other.Channel = &channel

	_, err := s.verifier.Verify(s.signed(body, time.Now()), body, other, s.public)

	s.ErrorContains(err, "has no ed25519 verifier")
}

func (s *VerifierSuite) event(name string) []byte {
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	s.Require().NoError(err)
	return raw
}

func (s *VerifierSuite) signed(body []byte, at time.Time) *http.Request {
	return s.signedWith(body, at, s.private)
}

// signedWith is body signed at at with key, as Telnyx sends one.
func (s *VerifierSuite) signedWith(body []byte, at time.Time, key ed25519.PrivateKey) *http.Request {
	timestamp := strconv.FormatInt(at.Unix(), 10)
	signature := ed25519.Sign(key, append([]byte(timestamp+"|"), body...))
	request := httptest.NewRequest(http.MethodPost, "/v1/connectors/events/telnyx/synthetic", bytes.NewReader(body))
	request.Header.Set("Telnyx-Signature-Ed25519", base64.StdEncoding.EncodeToString(signature))
	request.Header.Set("Telnyx-Timestamp", timestamp)
	return request
}
