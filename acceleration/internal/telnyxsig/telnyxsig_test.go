package telnyxsig

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type SignatureSuite struct {
	suite.Suite
	public  ed25519.PublicKey
	private ed25519.PrivateKey
	now     time.Time
	body    []byte
}

func TestSignature(t *testing.T) { suite.Run(t, new(SignatureSuite)) }

func (s *SignatureSuite) SetupTest() {
	var err error
	s.public, s.private, err = ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)
	s.now = time.Unix(1_791_900_000, 0)
	s.body = []byte("CallSid=v3%3Aabc&Direction=inbound&To=%2B15125550100")
}

// sign is what Telnyx sends: the timestamp, and the base64 signature over "ts|body".
func (s *SignatureSuite) sign(key ed25519.PrivateKey, at time.Time, body []byte) (string, string) {
	timestamp := strconv.FormatInt(at.Unix(), 10)
	return timestamp, base64.StdEncoding.EncodeToString(ed25519.Sign(key, append([]byte(timestamp+"|"), body...)))
}

func (s *SignatureSuite) TestASignedBodyPasses() {
	timestamp, signature := s.sign(s.private, s.now, s.body)

	s.NoError(Verify(s.public, timestamp, signature, s.body, s.now))
}

func (s *SignatureSuite) TestABodyChangedAfterSigningFails() {
	timestamp, signature := s.sign(s.private, s.now, s.body)
	changed := []byte("CallSid=v3%3Aabc&Direction=inbound&To=%2B15125550199")

	s.Error(Verify(s.public, timestamp, signature, changed, s.now))
}

func (s *SignatureSuite) TestAnotherKeysSignatureFails() {
	_, other, err := ed25519.GenerateKey(rand.Reader)
	s.Require().NoError(err)
	timestamp, signature := s.sign(other, s.now, s.body)

	s.Error(Verify(s.public, timestamp, signature, s.body, s.now))
}

func (s *SignatureSuite) TestASignatureOlderThanFiveMinutesFails() {
	timestamp, signature := s.sign(s.private, s.now.Add(-301*time.Second), s.body)

	s.ErrorContains(Verify(s.public, timestamp, signature, s.body, s.now), "too old")
}

func (s *SignatureSuite) TestASignatureFromTheFarFutureFails() {
	timestamp, signature := s.sign(s.private, s.now.Add(301*time.Second), s.body)

	s.Error(Verify(s.public, timestamp, signature, s.body, s.now))
}

// The timestamp is signed, so moving it into the window breaks the signature.
func (s *SignatureSuite) TestAMovedTimestampFails() {
	_, signature := s.sign(s.private, s.now.Add(-time.Hour), s.body)

	s.Error(Verify(s.public, strconv.FormatInt(s.now.Unix(), 10), signature, s.body, s.now))
}

func (s *SignatureSuite) TestAMissingTimestampFails() {
	_, signature := s.sign(s.private, s.now, s.body)

	s.Error(Verify(s.public, "", signature, s.body, s.now))
}

func (s *SignatureSuite) TestAMissingSignatureFails() {
	timestamp, _ := s.sign(s.private, s.now, s.body)

	s.Error(Verify(s.public, timestamp, "", s.body, s.now))
}

func (s *SignatureSuite) TestWithoutAKeyNothingPasses() {
	timestamp, signature := s.sign(s.private, s.now, s.body)

	s.Error(Verify(nil, timestamp, signature, s.body, s.now))
}

func (s *SignatureSuite) TestAKeyIsBase64Ed25519() {
	key, err := ParsePublicKey(base64.StdEncoding.EncodeToString(s.public))
	s.Require().NoError(err)
	s.Equal(s.public, key)

	_, err = ParsePublicKey("not a key")
	s.Error(err)
	_, err = ParsePublicKey(base64.StdEncoding.EncodeToString([]byte("short")))
	s.Error(err)
}
