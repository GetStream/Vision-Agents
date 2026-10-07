package eventforward

import (
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
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
func (s *SigningSuite) TestTheSameBodyIsTheSameWebhookID() {
	s.Equal(deliveryID([]byte("one")), deliveryID([]byte("one")))
	s.NotEqual(deliveryID([]byte("one")), deliveryID([]byte("two")))
	s.NotContains(deliveryID([]byte("one")), ".")
}

func (s *SigningSuite) secret() string {
	secret, err := plugins.NewWebhookSecret()
	s.Require().NoError(err)
	return secret
}
