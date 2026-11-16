package hmacheader_test

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
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
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/hmacheader"
)

// signingSecret is the operator's Slack signing secret in these tests. Made up.
const signingSecret = "synthetic-signing-secret"

// The core's synthetic Slack events (internal/connectors/core/testdata/recorded).
var recorded = filepath.Join("..", "..", "core", "testdata", "recorded")

// VerifierSuite runs hmac_header with the built-in Slack manifest (providers/slack.yaml), whose
// block reads tokens_revoked and app_uninstalled, and with the core's slack_bot fixture, whose
// block also reads messages. Requests are signed the way Slack's page says
// (https://docs.slack.dev/authentication/verifying-requests-from-slack): v0= and the hex
// HMAC-SHA256 of v0:{timestamp}:{body}.
type VerifierSuite struct {
	suite.Suite
	verifier *hmacheader.Verifier
	slack    core.Manifest
	bot      core.Manifest
}

func TestVerifierSuite(t *testing.T) {
	suite.Run(t, new(VerifierSuite))
}

func (s *VerifierSuite) SetupSuite() {
	s.verifier = hmacheader.New()
	raw, err := providers.FS.ReadFile("slack.yaml")
	s.Require().NoError(err)
	s.slack, err = core.ParseManifest(raw)
	s.Require().NoError(err)
	raw, err = os.ReadFile(filepath.Join("..", "..", "core", "testdata", "manifests", "slack_bot.yaml"))
	s.Require().NoError(err)
	s.bot, err = core.ParseManifest(raw)
	s.Require().NoError(err)
}

func (s *VerifierSuite) TestItIsRegisteredUnderItsKind() {
	s.Equal("hmac_header", s.verifier.Name())
}

func (s *VerifierSuite) TestASignedRevocationIsOneSignalForEachUser() {
	body := s.event("slack_bot.tokens_revoked.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.slack, []byte(signingSecret))

	s.Require().NoError(err)
	s.Equal(core.VerifiedEvent{Signals: []core.Signal{
		{ConnectorID: "slack", Identity: map[string]string{"team_id": "T0000TEAM", "user_id": "U0000USER"}, Kind: core.SignalRevoked},
		{ConnectorID: "slack", Identity: map[string]string{"team_id": "T0000TEAM", "user_id": "U0000OTHER"}, Kind: core.SignalRevoked},
	}}, event)
}

func (s *VerifierSuite) TestASignedUninstallNamesTheWorkspace() {
	body := s.event("slack_bot.app_uninstalled.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.slack, []byte(signingSecret))

	s.Require().NoError(err)
	s.Equal([]core.Signal{{ConnectorID: "slack", Identity: map[string]string{"team_id": "T0000TEAM"}, Kind: core.SignalUninstalled}}, event.Signals)
}

func (s *VerifierSuite) TestASignedURLVerificationIsOnlyItsChallenge() {
	body := s.event("slack_bot.challenge.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.slack, []byte(signingSecret))

	s.Require().NoError(err)
	s.Equal(core.VerifiedEvent{Challenge: "synthetic-challenge-value"}, event)
}

func (s *VerifierSuite) TestASignedMessageIsOneInboundMessage() {
	body := s.event("slack_bot.message.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.bot, []byte(signingSecret))

	s.Require().NoError(err)
	s.Empty(event.Signals)
	s.Equal([]core.InboundMessage{{
		ConnectorID:       "slack_bot",
		ProviderUnitID:    "T0000TEAM",
		ThreadKey:         "C0000CHAN:1759740000.000100",
		AuthorID:          "U0000USER",
		Text:              "Can you \"check\" the build?\nThanks",
		ProviderMessageID: "1759740000.000200",
		Raw:               body,
	}}, event.Messages)
}

func (s *VerifierSuite) TestAnUnsignedRequestIsRefused() {
	body := s.event("slack_bot.tokens_revoked.json")
	request := s.signed(body, time.Now())
	request.Header.Del("X-Slack-Signature")

	event, err := s.verifier.Verify(request, body, s.slack, []byte(signingSecret))

	s.ErrorIs(err, hmacheader.ErrUnsigned)
	s.Zero(event)
}

func (s *VerifierSuite) TestASignatureWithAnotherSecretIsRefused() {
	body := s.event("slack_bot.tokens_revoked.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.slack, []byte("another-secret"))

	s.ErrorIs(err, hmacheader.ErrUnsigned)
	s.Zero(event)
}

func (s *VerifierSuite) TestABodyChangedAfterSigningIsRefused() {
	body := s.event("slack_bot.tokens_revoked.json")
	request := s.signed(body, time.Now())
	changed := bytes.Replace(body, []byte("U0000OTHER"), []byte("U0000VICTIM"), 1)

	_, err := s.verifier.Verify(request, changed, s.slack, []byte(signingSecret))

	s.ErrorIs(err, hmacheader.ErrUnsigned)
}

// The timestamp is signed, so moving it to pass the age check breaks the signature.
func (s *VerifierSuite) TestATimestampChangedAfterSigningIsRefused() {
	body := s.event("slack_bot.tokens_revoked.json")
	request := s.signed(body, time.Now().Add(-10*time.Minute))
	request.Header.Set("X-Slack-Request-Timestamp", strconv.FormatInt(time.Now().Unix(), 10))

	_, err := s.verifier.Verify(request, body, s.slack, []byte(signingSecret))

	s.ErrorIs(err, hmacheader.ErrUnsigned)
}

// Slack: «The request timestamp is more than five minutes from local time», refuse it.
func (s *VerifierSuite) TestARequestSignedMoreThanFiveMinutesAgoIsRefused() {
	body := s.event("slack_bot.tokens_revoked.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now().Add(-5*time.Minute-time.Second)), body, s.slack, []byte(signingSecret))

	s.ErrorIs(err, hmacheader.ErrStale)
	s.Zero(event)
}

func (s *VerifierSuite) TestARequestSignedMoreThanFiveMinutesAheadIsRefused() {
	body := s.event("slack_bot.tokens_revoked.json")

	_, err := s.verifier.Verify(s.signed(body, time.Now().Add(5*time.Minute+time.Second)), body, s.slack, []byte(signingSecret))

	s.ErrorIs(err, hmacheader.ErrStale)
}

func (s *VerifierSuite) TestARequestSignedWithinFiveMinutesIsTaken() {
	body := s.event("slack_bot.tokens_revoked.json")

	event, err := s.verifier.Verify(s.signed(body, time.Now().Add(-4*time.Minute)), body, s.slack, []byte(signingSecret))

	s.Require().NoError(err)
	s.Len(event.Signals, 2)
}

func (s *VerifierSuite) TestATimestampThatIsNotUnixSecondsIsRefused() {
	body := s.event("slack_bot.tokens_revoked.json")
	request := s.signed(body, time.Now())
	request.Header.Set("X-Slack-Request-Timestamp", time.Now().Format(time.RFC3339))

	_, err := s.verifier.Verify(request, body, s.slack, []byte(signingSecret))

	s.ErrorIs(err, hmacheader.ErrUnsigned)
}

func (s *VerifierSuite) TestASignatureWithoutItsVersionPrefixIsRefused() {
	body := s.event("slack_bot.tokens_revoked.json")
	request := s.signed(body, time.Now())
	request.Header.Set("X-Slack-Signature", request.Header.Get("X-Slack-Signature")[len("v0="):])

	_, err := s.verifier.Verify(request, body, s.slack, []byte(signingSecret))

	s.ErrorIs(err, hmacheader.ErrUnsigned)
}

// An HMAC under an empty key is one anybody can compute, so a deployment with no secret
// takes nothing, even a request signed with the empty key.
func (s *VerifierSuite) TestNoSecretTakesNothing() {
	body := s.event("slack_bot.tokens_revoked.json")
	request := s.signedWith(body, time.Now(), nil)

	_, err := s.verifier.Verify(request, body, s.slack, nil)

	s.ErrorIs(err, hmacheader.ErrUnsigned)
}

// The body goes into the signed bytes as it is, so one that holds the text {timestamp} is
// signed and checked over those exact bytes.
func (s *VerifierSuite) TestABodyHoldingAPlaceholderIsSignedAsItIs() {
	body := []byte(`{"type":"event_callback","team_id":"T0000TEAM","event":{"type":"tokens_revoked","tokens":{"oauth":["{timestamp}"]}}}`)

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.slack, []byte(signingSecret))

	s.Require().NoError(err)
	s.Equal([]core.Signal{{ConnectorID: "slack", Identity: map[string]string{"team_id": "T0000TEAM", "user_id": "{timestamp}"}, Kind: core.SignalRevoked}}, event.Signals)
}

func (s *VerifierSuite) TestASignedBodyThatIsNotJSONIsNothingToActOn() {
	body := []byte("payload=not-json")

	event, err := s.verifier.Verify(s.signed(body, time.Now()), body, s.slack, []byte(signingSecret))

	s.Require().NoError(err)
	s.Zero(event)
}

func (s *VerifierSuite) TestAManifestWithAnotherVerifierKindIsRefused() {
	body := s.event("slack_bot.tokens_revoked.json")
	other := s.slack
	channel := *other.Channel
	channel.Verifier.Kind = core.VerifierSecretHeader
	other.Channel = &channel

	_, err := s.verifier.Verify(s.signed(body, time.Now()), body, other, []byte(signingSecret))

	s.ErrorContains(err, "has no hmac_header verifier")
}

func (s *VerifierSuite) event(name string) []byte {
	raw, err := os.ReadFile(filepath.Join(recorded, name))
	s.Require().NoError(err)
	return raw
}

// signed is the request Slack would send with body at at, signed with signingSecret.
func (s *VerifierSuite) signed(body []byte, at time.Time) *http.Request {
	return s.signedWith(body, at, []byte(signingSecret))
}

func (s *VerifierSuite) signedWith(body []byte, at time.Time, secret []byte) *http.Request {
	timestamp := strconv.FormatInt(at.Unix(), 10)
	mac := hmac.New(sha256.New, secret)
	mac.Write([]byte("v0:" + timestamp + ":"))
	mac.Write(body)
	request := httptest.NewRequest(http.MethodPost, "/v1/agents/connectors/events/slack", bytes.NewReader(body))
	request.Header.Set("X-Slack-Request-Timestamp", timestamp)
	request.Header.Set("X-Slack-Signature", "v0="+hex.EncodeToString(mac.Sum(nil)))
	return request
}
