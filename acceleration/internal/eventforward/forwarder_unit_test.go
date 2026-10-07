package eventforward

import (
	"net/http"
	"net/url"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
)

// SigningSuite is what a forward carries, decided before anything is sent: the provider's
// headers kept, the webhook-id and the signature.
type SigningSuite struct {
	suite.Suite
}

func TestSigningSuite(t *testing.T) {
	suite.Run(t, new(SigningSuite))
}

// slackVerifier is slack_bot's channel.verifier, whose headers ProviderHeaders keeps.
var slackVerifier = core.Manifest{Channel: &core.ChannelRule{Verifier: core.VerifierRule{
	Header: "X-Slack-Signature", TimestampHeader: "X-Slack-Request-Timestamp",
}}}

func (s *SigningSuite) TestProviderHeadersKeepWhatAReceiverVerifiesWithAndNothingElse() {
	header := http.Header{}
	header.Set("Content-Type", "application/x-www-form-urlencoded")
	header.Set("X-Slack-Signature", "v0=abc")
	header.Set("X-Slack-Request-Timestamp", "1759740000")
	header.Set("X-Slack-Retry-Num", "1")
	header.Set("X-Forwarded-For", "203.0.113.7")
	header.Set("Authorization", "Bearer synthetic")

	kept := ProviderHeaders(slackVerifier, header)

	s.Equal(map[string]string{
		"Content-Type":              "application/x-www-form-urlencoded",
		"X-Slack-Signature":         "v0=abc",
		"X-Slack-Request-Timestamp": "1759740000",
	}, kept)
}

func (s *SigningSuite) TestASignatureVerifiesWithEachSecretItWasSignedWithAndNoOther() {
	current, previous, other := s.secret(), s.secret(), s.secret()
	at, body := time.Now(), []byte(`{"type":"event_callback"}`)

	signature, err := sign("msg_one", at, body, current, previous)

	s.Require().NoError(err)
	header := http.Header{}
	header.Set(headerID, "msg_one")
	header.Set(headerTimestamp, strconv.FormatInt(at.Unix(), 10))
	header.Set(headerSignature, signature)
	s.NoError(plugins.VerifyWebhook(current, header, body, at))
	s.NoError(plugins.VerifyWebhook(previous, header, body, at))
	s.ErrorIs(plugins.VerifyWebhook(other, header, body, at), plugins.ErrBadSignature)
}

// https://www.standardwebhooks.com/: the id «remains the same no matter how many times a
// webhook that has failed is retried», and must hold no «.».
func (s *SigningSuite) TestADeliveryWithoutAProviderIDIsKeyedByItsBody() {
	s.Equal(deliveryID("", []byte("one")), deliveryID("", []byte("one")))
	s.NotEqual(deliveryID("", []byte("one")), deliveryID("", []byte("two")))
	s.NotContains(deliveryID("", []byte("one")), ".")
}

// Two Slack events whose bodies are byte for byte the same are still two events when their
// event_ids differ, and one event delivered again with another body is still one.
func (s *SigningSuite) TestTheProvidersEventIDIsTheWebhookIDNotTheBody() {
	body := []byte(`{"type":"event_callback"}`)

	s.NotEqual(deliveryID("Ev0000ONE", body), deliveryID("Ev0000TWO", body))
	s.Equal(deliveryID("Ev0000ONE", body), deliveryID("Ev0000ONE", []byte(`{"type":"event_callback","retried":true}`)))
	s.NotContains(deliveryID("1.2.abc", body), ".", "a trigger_id's dots stay out of webhook-id")
}

// slack_bot.yaml names event_id and trigger_id, and signs X-Slack-Request-Timestamp with a
// max_age of 5m.
func (s *SigningSuite) TestASlackEventIsKeyedByItsEventIDAndItsHeadersVerifyForFiveMinutes() {
	header := http.Header{}
	header.Set("X-Slack-Request-Timestamp", "1759740000")

	event := ProviderEvent(s.slackBot(), header, []byte(`{"type":"event_callback","event_id":"Ev0000ONE"}`))

	s.Equal("Ev0000ONE", event.ID)
	s.Equal(time.Unix(1759740000, 0).Add(5*time.Minute), event.HeadersUntil)
}

func (s *SigningSuite) TestASlackInteractionIsKeyedByItsTriggerID() {
	body := url.Values{"payload": {`{"type":"block_actions","trigger_id":"1.2.abc"}`}}.Encode()

	s.Equal("1.2.abc", ProviderEvent(s.slackBot(), http.Header{}, []byte(body)).ID)
}

func (s *SigningSuite) slackBot() core.Manifest {
	raw, err := providers.FS.ReadFile("slack_bot.yaml")
	s.Require().NoError(err)
	m, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	return m
}

func (s *SigningSuite) secret() string {
	secret, err := plugins.NewWebhookSecret()
	s.Require().NoError(err)
	return secret
}
