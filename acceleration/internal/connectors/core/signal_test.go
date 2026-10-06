package core

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"
)

// fakeSignatureHeader and fakeSecret belong to the fake provider below; no real provider
// uses them.
const (
	fakeSignatureHeader = "X-Fake-Signature"
	fakeSecret          = "fake-signing-secret"
)

var errFakeSignature = errors.New("fake: signature does not match")

// fakeVerifier is a provider with one HMAC-SHA256 header over the raw body and a JSON body
// that can carry a handshake, a revocation and a batch of messages at once, so one test
// can see every part of a VerifiedEvent.
type fakeVerifier struct{}

var _ Verifier = fakeVerifier{}

type fakeEvent struct {
	Type      string   `json:"type"`
	Challenge string   `json:"challenge"`
	Unit      string   `json:"unit"`
	Revoked   []string `json:"revoked"`
	Messages  []struct {
		Thread string `json:"thread"`
		From   string `json:"from"`
		Text   string `json:"text"`
	} `json:"messages"`
}

func (fakeVerifier) Name() string { return "fake_hmac" }

func (fakeVerifier) Verify(r *http.Request, body []byte, m ResolvedManifest) (VerifiedEvent, error) {
	got, err := hex.DecodeString(r.Header.Get(fakeSignatureHeader))
	if err != nil || !hmac.Equal(got, fakeSign(body)) {
		return VerifiedEvent{}, errFakeSignature
	}
	var event fakeEvent
	if err := json.Unmarshal(body, &event); err != nil {
		return VerifiedEvent{}, err
	}
	if event.Type == "handshake" {
		return VerifiedEvent{Challenge: event.Challenge}, nil
	}
	var verified VerifiedEvent
	for _, account := range event.Revoked {
		verified.Signals = append(verified.Signals, Signal{ConnectorID: m.ConnectorID, AccountID: account, Kind: SignalRevoked})
	}
	for _, message := range event.Messages {
		verified.Messages = append(verified.Messages, InboundMessage{
			ConnectorID:    m.ConnectorID,
			ProviderUnitID: event.Unit,
			ThreadKey:      message.Thread,
			AuthorID:       message.From,
			Text:           message.Text,
			Raw:            body,
		})
	}
	return verified, nil
}

func fakeSign(body []byte) []byte {
	mac := hmac.New(sha256.New, []byte(fakeSecret))
	mac.Write(body)
	return mac.Sum(nil)
}

// VerifierSuite proves the shape of what a Verifier returns, through a fake provider: one
// request gives the resolver its signals and the channel bridge its messages.
type VerifierSuite struct {
	suite.Suite
	verifier Verifier
	manifest ResolvedManifest
}

func TestVerifierSuite(t *testing.T) {
	suite.Run(t, new(VerifierSuite))
}

func (s *VerifierSuite) SetupTest() {
	s.verifier = fakeVerifier{}
	s.manifest = ResolvedManifest{ConnectorID: "fake"}
}

func (s *VerifierSuite) TestOneEventCarriesASignalAndAMessage() {
	body := []byte(`{"type":"event","unit":"W1","revoked":["A1"],"messages":[{"thread":"C1:100.1","from":"P1","text":"hello"}]}`)

	verified, err := s.verifier.Verify(s.signed(body), body, s.manifest)

	s.Require().NoError(err)
	s.Equal(VerifiedEvent{
		Signals: []Signal{{ConnectorID: "fake", AccountID: "A1", Kind: SignalRevoked}},
		Messages: []InboundMessage{{
			ConnectorID:    "fake",
			ProviderUnitID: "W1",
			ThreadKey:      "C1:100.1",
			AuthorID:       "P1",
			Text:           "hello",
			Raw:            body,
		}},
	}, verified)
}

func (s *VerifierSuite) TestABatchedDeliveryGivesEveryMessageTheSameRawBody() {
	body := []byte(`{"type":"event","unit":"W1","messages":[{"thread":"t1","from":"P1","text":"one"},{"thread":"t2","from":"P2","text":"two"}]}`)

	verified, err := s.verifier.Verify(s.signed(body), body, s.manifest)

	s.Require().NoError(err)
	s.Empty(verified.Signals)
	s.Require().Len(verified.Messages, 2)
	s.Equal([]string{"t1", "t2"}, []string{verified.Messages[0].ThreadKey, verified.Messages[1].ThreadKey})
	s.Equal(body, verified.Messages[0].Raw)
	s.Equal(body, verified.Messages[1].Raw)
}

func (s *VerifierSuite) TestAHandshakeAnswersWithItsChallengeOnly() {
	body := []byte(`{"type":"handshake","challenge":"c-123"}`)

	verified, err := s.verifier.Verify(s.signed(body), body, s.manifest)

	s.Require().NoError(err)
	s.Equal(VerifiedEvent{Challenge: "c-123"}, verified)
}

func (s *VerifierSuite) TestABadSignatureYieldsNothingToActOn() {
	body := []byte(`{"type":"event","unit":"W1","revoked":["A1"],"messages":[{"thread":"t1","from":"P1","text":"hello"}]}`)
	request := s.signed(body)
	request.Header.Set(fakeSignatureHeader, hex.EncodeToString(fakeSign([]byte("other body"))))

	verified, err := s.verifier.Verify(request, body, s.manifest)

	s.ErrorIs(err, errFakeSignature)
	s.Zero(verified)
}

// signed is the request the fake provider would send with body.
func (s *VerifierSuite) signed(body []byte) *http.Request {
	request := httptest.NewRequest(http.MethodPost, "/events/fake", bytes.NewReader(body))
	request.Header.Set(fakeSignatureHeader, hex.EncodeToString(fakeSign(body)))
	return request
}
