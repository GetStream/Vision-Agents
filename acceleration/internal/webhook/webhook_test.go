package webhook

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type WebhookSuite struct {
	suite.Suite
	secret string
}

func TestWebhookSuite(t *testing.T) {
	suite.Run(t, new(WebhookSuite))
}

func (s *WebhookSuite) SetupTest() {
	secret, err := NewSecret()
	s.Require().NoError(err)
	s.secret = secret
}

func (s *WebhookSuite) TestTheSignatureIsStandardWebhooks() {
	// The example in the Standard Webhooks specification, which its libraries test against.
	// plugins.SignWebhook gave the same before it moved here.
	signature, err := Sign("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw", "msg_p5jXN8AQM9LWM0D4loKWxJek",
		time.Unix(1614265330, 0), []byte(`{"test": 2432232314}`))

	s.Require().NoError(err)
	s.Equal("v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE=", signature)
}

func (s *WebhookSuite) TestASecretIsThirtyTwoRandomBytesAfterWhsec() {
	encoded, ok := strings.CutPrefix(s.secret, "whsec_")
	s.Require().True(ok, s.secret)
	key, err := base64.StdEncoding.DecodeString(encoded)
	s.Require().NoError(err)
	s.Len(key, 32)
}

func (s *WebhookSuite) TestASecretThatIsNotBase64SignsNothing() {
	_, err := Sign("whsec_!!", "msg_1", time.Now(), nil)

	s.Error(err)
}

func (s *WebhookSuite) TestADeliverySignedWithTheSecretIsAccepted() {
	body := []byte(`{"eventId":"evt_1","name":"issue.created","data":{}}`)
	header := s.signed("evt_1", time.Now(), body, s.secret)

	s.NoError(Verify(s.secret, header, body, time.Now()))
}

func (s *WebhookSuite) TestADeliverySignedWithAnotherSecretIsRefused() {
	body := []byte(`{"eventId":"evt_1"}`)

	s.ErrorIs(Verify(s.secret, s.signed("evt_1", time.Now(), body, s.another()), body, time.Now()), ErrBadSignature)
}

func (s *WebhookSuite) TestABodyChangedAfterSigningIsRefused() {
	header := s.signed("evt_1", time.Now(), []byte(`{"eventId":"evt_1"}`), s.secret)

	s.ErrorIs(Verify(s.secret, header, []byte(`{"eventId":"evt_2"}`), time.Now()), ErrBadSignature)
}

func (s *WebhookSuite) TestASignatureUpToFiveMinutesEitherWayIsAccepted() {
	body := []byte(`{"eventId":"evt_1"}`)
	// Whole seconds: the header carries Unix seconds.
	now := time.Unix(1761100000, 0)

	s.NoError(Verify(s.secret, s.signed("evt_1", now.Add(-5*time.Minute), body, s.secret), body, now))
	s.NoError(Verify(s.secret, s.signed("evt_1", now.Add(5*time.Minute), body, s.secret), body, now))
	s.ErrorIs(Verify(s.secret, s.signed("evt_1", now.Add(5*time.Minute+time.Second), body, s.secret), body, now), ErrBadSignature)
	s.ErrorIs(Verify(s.secret, s.signed("evt_1", now.Add(-5*time.Minute-time.Second), body, s.secret), body, now), ErrBadSignature)
}

func (s *WebhookSuite) TestADeliveryWithoutAnIDOrANumericTimestampIsRefused() {
	body := []byte(`{"eventId":"evt_1"}`)
	now := time.Now()
	// Signed over an empty id, which a missing header would otherwise match.
	noID := s.signed("", now, body, s.secret)
	badStamp := s.signed("evt_1", now, body, s.secret)
	badStamp.Set("webhook-timestamp", "soon")
	noStamp := s.signed("evt_1", now, body, s.secret)
	noStamp.Del("webhook-timestamp")

	s.ErrorIs(Verify(s.secret, noID, body, now), ErrBadSignature)
	// Refused as unsigned, not as signed at the Unix epoch.
	s.Equal(ErrBadSignature, Verify(s.secret, badStamp, body, now))
	s.Equal(ErrBadSignature, Verify(s.secret, noStamp, body, now))
}

func (s *WebhookSuite) TestEitherSignatureOfARotationIsEnough() {
	body := []byte(`{"eventId":"evt_1"}`)
	now := time.Now()
	stale, err := Sign(s.another(), "evt_1", now, body)
	s.Require().NoError(err)
	header := s.signed("evt_1", now, body, s.secret)
	header.Set("webhook-signature", stale+" "+header.Get("webhook-signature"))

	s.NoError(Verify(s.secret, header, body, now))
}

// The JSON plugins.Event read and wrote before it moved here.
func (s *WebhookSuite) TestAnEventReadsTheDeliveryThePluginPackageRead() {
	var event Event
	s.Require().NoError(json.Unmarshal([]byte(`{"type":"verification","challenge":"c","eventId":"e","name":"n","timestamp":"t","data":{"a":1}}`), &event))
	s.Equal(Event{Type: "verification", Challenge: "c", EventID: "e", Name: "n", Timestamp: "t", Data: json.RawMessage(`{"a":1}`)}, event)
	empty, err := json.Marshal(Event{})
	s.Require().NoError(err)
	s.Equal(`{}`, string(empty))
}

// plugins.MaxEventBytes before it moved here.
func (s *WebhookSuite) TestADeliveryMayHold256KiB() {
	s.Equal(262144, MaxEventBytes)
}

func (s *WebhookSuite) another() string {
	secret, err := NewSecret()
	s.Require().NoError(err)
	return secret
}

func (s *WebhookSuite) signed(id string, at time.Time, body []byte, secret string) http.Header {
	signature, err := Sign(secret, id, at, body)
	s.Require().NoError(err)
	header := http.Header{}
	header.Set("webhook-id", id)
	header.Set("webhook-timestamp", strconv.FormatInt(at.Unix(), 10))
	header.Set("webhook-signature", signature)
	return header
}
