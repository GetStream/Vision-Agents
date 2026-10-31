package chatinbound_test

import (
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatinbound"
)

// The vector was signed by OpenSSL 4.0.3, not by this package, so it pins the signed content
// ({id}.{timestamp}.{body}) and the encodings rather than Go agreeing with itself:
//
//	seed=$(printf 'chat-delivery-test-vector-seed-1' | xxd -p -c 64)
//	echo "302e020100300506032b657004220420$seed" | xxd -r -p > key.der   # PKCS#8 of the seed
//	openssl pkey -inform DER -in key.der -out key.pem
//	openssl pkey -in key.pem -pubout -outform DER | tail -c 32 | base64
//	printf '<vectorID>.<vectorTimestamp>.<vectorBody>' | openssl pkeyutl -sign -inkey key.pem -rawin | base64
const (
	vectorSeed      = "chat-delivery-test-vector-seed-1"
	vectorPublicKey = "5/7U2kOnmM/KwTJGEqvh2gSDhVIZ+0XHxq6ZmMky/Us="
	vectorID        = "msg_2KWPBgLlAfxdpx2AI54pPJ85f4W"
	vectorTimestamp = 1761670800
	vectorBody      = `{"type":"message.new","app_id":1257545}`
	vectorSignature = "lJA4Pri8yjuczMDikHg3tThCxDDFQ8WE6rLRw90NUduCXOiVO4ZmE/m2CbR4tsVPcPP4egKT3rXug8kUaDp/Cw=="
)

// ChatInboundSuite is Stream signing a delivery the way Standard Webhooks signs one with a
// public key, and the router checking it.
type ChatInboundSuite struct {
	suite.Suite
	current, previous ed25519.PrivateKey
	keys              []ed25519.PublicKey
	now               time.Time
}

func TestChatInboundSuite(t *testing.T) {
	suite.Run(t, new(ChatInboundSuite))
}

func (s *ChatInboundSuite) SetupTest() {
	s.current = ed25519.NewKeyFromSeed([]byte("the-current-delivery-key-seed-32"))
	s.previous = ed25519.NewKeyFromSeed([]byte("the-previous-delivery-key-seed32"))
	s.keys = []ed25519.PublicKey{s.public(s.current)}
	s.now = time.Unix(1761670800, 0)
}

func (s *ChatInboundSuite) TestTheVectorOpenSSLSignedIsVerified() {
	keys, err := chatinbound.ParsePublicKeys([]string{vectorPublicKey})
	s.Require().NoError(err)
	s.Equal(s.public(ed25519.NewKeyFromSeed([]byte(vectorSeed))), keys[0])
	header := http.Header{}
	header.Set("webhook-id", vectorID)
	header.Set("webhook-timestamp", strconv.Itoa(vectorTimestamp))
	header.Set("webhook-signature", "v1a,"+vectorSignature)

	s.NoError(chatinbound.Verify(keys, header, []byte(vectorBody), time.Unix(vectorTimestamp, 0)))
}

func (s *ChatInboundSuite) TestADeliverySignedWithTheKeyIsVerified() {
	body := []byte(`{"type":"message.new"}`)

	s.NoError(chatinbound.Verify(s.keys, s.signed(body, s.current), body, s.now))
}

func (s *ChatInboundSuite) TestNoSignatureIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	header.Del(chatinbound.SignatureHeader)

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

func (s *ChatInboundSuite) TestNoIDIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signedAs("", s.now, body, s.current)

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

func (s *ChatInboundSuite) TestNoTimestampIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	header.Del(chatinbound.TimestampHeader)

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

// The specification's timestamp is integer seconds; a date is another shape, signed or not.
func (s *ChatInboundSuite) TestATimestampThatIsNotWholeSecondsIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	for _, written := range []string{s.now.Format(time.RFC3339), "1761670800.5", "1.7616708e9"} {
		header := http.Header{}
		header.Set(chatinbound.IDHeader, "msg_1")
		header.Set(chatinbound.TimestampHeader, written)
		header.Set(chatinbound.SignatureHeader, "v1a,"+base64.StdEncoding.EncodeToString(
			ed25519.Sign(s.current, []byte("msg_1."+written+"."+string(body)))))

		s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned, written)
	}
}

func (s *ChatInboundSuite) TestASignatureUnderAnotherKeyIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	stranger := ed25519.NewKeyFromSeed([]byte("somebody-else-s-key-seed-32bytes"))

	s.ErrorIs(chatinbound.Verify(s.keys, s.signed(body, stranger), body, s.now), chatinbound.ErrUnsigned)
}

func (s *ChatInboundSuite) TestABodyChangedAfterSigningIsRefused() {
	body := []byte(`{"type":"message.new","app_id":1257545}`)
	header := s.signed(body, s.current)

	s.ErrorIs(chatinbound.Verify(s.keys, header, []byte(`{"type":"message.new","app_id":1257546}`), s.now),
		chatinbound.ErrUnsigned)
}

// The id is signed, so a replay cannot pass as a new delivery under another id.
func (s *ChatInboundSuite) TestAnIDChangedAfterSigningIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	header.Set(chatinbound.IDHeader, "msg_another")

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

// The timestamp is signed, so an old request cannot be made fresh by writing a new one.
func (s *ChatInboundSuite) TestATimestampChangedAfterSigningIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signedAs("msg_1", s.now.Add(-10*time.Minute), body, s.current)
	header.Set(chatinbound.TimestampHeader, strconv.FormatInt(s.now.Unix(), 10))

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

func (s *ChatInboundSuite) TestADeliveryFiveMinutesOldIsVerified() {
	body := []byte(`{"type":"message.new"}`)

	s.NoError(chatinbound.Verify(s.keys, s.signedAs("msg_1", s.now.Add(-5*time.Minute), body, s.current), body, s.now))
}

func (s *ChatInboundSuite) TestADeliveryOlderThanFiveMinutesIsStale() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signedAs("msg_1", s.now.Add(-5*time.Minute-time.Second), body, s.current)

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrStale)
}

func (s *ChatInboundSuite) TestADeliveryFiveMinutesAheadIsVerified() {
	body := []byte(`{"type":"message.new"}`)

	s.NoError(chatinbound.Verify(s.keys, s.signedAs("msg_1", s.now.Add(5*time.Minute), body, s.current), body, s.now))
}

func (s *ChatInboundSuite) TestADeliveryFromFurtherAheadIsStale() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signedAs("msg_1", s.now.Add(5*time.Minute+time.Second), body, s.current)

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrStale)
}

// v1 is the HMAC under a shared secret. Signed with the app's API secret, it is a request the
// customer's own server can make.
func (s *ChatInboundSuite) TestAnHMACUnderTheAppSecretIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	mac := hmac.New(sha256.New, []byte("the-apps-api-secret"))
	mac.Write([]byte("msg_1." + header.Get(chatinbound.TimestampHeader) + "." + string(body)))
	header.Set(chatinbound.SignatureHeader, "v1,"+base64.StdEncoding.EncodeToString(mac.Sum(nil)))

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

// Today's webhook signs with X-Signature, the hex HMAC of the body under the app's API secret.
func (s *ChatInboundSuite) TestTodaysXSignatureIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	mac := hmac.New(sha256.New, []byte("the-apps-api-secret"))
	mac.Write(body)
	header := http.Header{}
	header.Set("X-Signature", hex.EncodeToString(mac.Sum(nil)))

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

// A v1a signature with its version left off is not a signature the specification writes.
func (s *ChatInboundSuite) TestASignatureWithoutItsVersionIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	header.Set(chatinbound.SignatureHeader, strings.TrimPrefix(header.Get(chatinbound.SignatureHeader), "v1a,"))

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

func (s *ChatInboundSuite) TestASignatureThatIsNotBase64IsRefused() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	header.Set(chatinbound.SignatureHeader, "v1a,not*base64")

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

func (s *ChatInboundSuite) TestASignatureWithAStrayCharacterAfterItIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	header.Set(chatinbound.SignatureHeader, header.Get(chatinbound.SignatureHeader)+"!")

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

func (s *ChatInboundSuite) TestASignatureOfTheWrongLengthIsRefused() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	signature, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(header.Get(chatinbound.SignatureHeader), "v1a,"))
	s.Require().NoError(err)
	header.Set(chatinbound.SignatureHeader, "v1a,"+base64.StdEncoding.EncodeToString(signature[:32]))

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

func (s *ChatInboundSuite) TestTheOldKeyIsAcceptedWhileBothAreSet() {
	body := []byte(`{"type":"message.new"}`)
	both := []ed25519.PublicKey{s.public(s.current), s.public(s.previous)}

	s.NoError(chatinbound.Verify(both, s.signed(body, s.previous), body, s.now))
	s.ErrorIs(chatinbound.Verify(s.keys, s.signed(body, s.previous), body, s.now), chatinbound.ErrUnsigned,
		"once it is dropped it verifies nothing")
}

func (s *ChatInboundSuite) TestTwoSignaturesOneValidAreAccepted() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	stranger := ed25519.NewKeyFromSeed([]byte("somebody-else-s-key-seed-32bytes"))
	forged := s.signed(body, stranger).Get(chatinbound.SignatureHeader)
	header.Set(chatinbound.SignatureHeader, forged+" "+header.Get(chatinbound.SignatureHeader))

	s.NoError(chatinbound.Verify(s.keys, header, body, s.now))
}

// A v1 HMAC and an entry that is not base64 are passed over, not taken as the answer: the
// valid v1a after them still verifies.
func (s *ChatInboundSuite) TestAV1AndAMalformedEntryBeforeAValidOneAreSkipped() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	header.Set(chatinbound.SignatureHeader, "v1,"+base64.StdEncoding.EncodeToString([]byte("an-hmac"))+
		" v1a,not*base64 "+header.Get(chatinbound.SignatureHeader))

	s.NoError(chatinbound.Verify(s.keys, header, body, s.now))
}

// Four entries are read: a valid signature fourth still verifies.
func (s *ChatInboundSuite) TestAValidSignatureFourthIsVerified() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	header.Set(chatinbound.SignatureHeader, s.forged(body, 3)+" "+header.Get(chatinbound.SignatureHeader))

	s.NoError(chatinbound.Verify(s.keys, header, body, s.now))
}

// Past four entries the rest is ignored, so a header full of forgeries costs four checks a
// key, not one per entry.
func (s *ChatInboundSuite) TestASignatureFifthIsIgnored() {
	body := []byte(`{"type":"message.new"}`)
	header := s.signed(body, s.current)
	header.Set(chatinbound.SignatureHeader, s.forged(body, 4)+" "+header.Get(chatinbound.SignatureHeader))

	s.ErrorIs(chatinbound.Verify(s.keys, header, body, s.now), chatinbound.ErrUnsigned)
}

func (s *ChatInboundSuite) TestNoKeysVerifyNothing() {
	body := []byte(`{"type":"message.new"}`)

	s.ErrorIs(chatinbound.Verify(nil, s.signed(body, s.current), body, s.now), chatinbound.ErrUnsigned)
}

func (s *ChatInboundSuite) TestTwoKeysAreRead() {
	keys, err := chatinbound.ParsePublicKeys([]string{s.written(s.current), s.written(s.previous)})

	s.Require().NoError(err)
	s.Equal([]ed25519.PublicKey{s.public(s.current), s.public(s.previous)}, keys)
}

func (s *ChatInboundSuite) TestNoKeysAreNone() {
	keys, err := chatinbound.ParsePublicKeys(nil)

	s.Require().NoError(err)
	s.Nil(keys)
}

func (s *ChatInboundSuite) TestAThirdKeyIsRefused() {
	third := ed25519.NewKeyFromSeed([]byte("a-third-delivery-key-seed-32byte"))

	_, err := chatinbound.ParsePublicKeys([]string{s.written(s.current), s.written(s.previous), s.written(third)})

	s.ErrorContains(err, "at most two")
}

func (s *ChatInboundSuite) TestAKeyThatIsNotBase64IsRefused() {
	_, err := chatinbound.ParsePublicKeys([]string{s.written(s.current), "not*base64"})

	s.ErrorContains(err, "delivery key 2 is not")
	s.NotContains(err.Error(), "not*base64", "the error names the key's place, not its value")
}

// Base64 decodes up to the first byte it cannot read, so a key with a stray character after it
// still yields its 32 bytes, and only the decoding error tells it apart.
func (s *ChatInboundSuite) TestAKeyWithAStrayCharacterAfterItIsRefused() {
	_, err := chatinbound.ParsePublicKeys([]string{s.written(s.current) + "!"})

	s.ErrorContains(err, "delivery key 1 is not")
}

func (s *ChatInboundSuite) TestAKeyOfTheWrongSizeIsRefused() {
	_, err := chatinbound.ParsePublicKeys([]string{base64.StdEncoding.EncodeToString(s.public(s.current)[:31])})

	s.ErrorContains(err, "delivery key 1 is not")
}

// The specification writes an asymmetric key with a whpk_ prefix; the router's value is the
// bare base64, so a prefixed one is a value written for something else.
func (s *ChatInboundSuite) TestAPrefixedKeyIsRefused() {
	_, err := chatinbound.ParsePublicKeys([]string{"whpk_" + s.written(s.current)})

	s.ErrorContains(err, "delivery key 1 is not")
}

// Go's ed25519 private key is 64 bytes, so pasting it where the public key goes is refused.
// The 32-byte seed Chat holds cannot be told apart by size; it fails closed, verifying
// nothing Stream signs.
func (s *ChatInboundSuite) TestAPrivateKeyIsRefused() {
	_, err := chatinbound.ParsePublicKeys([]string{base64.StdEncoding.EncodeToString(s.current)})

	s.ErrorContains(err, "delivery key 1 is not")
}

// signed is a delivery's headers as Stream sends them, made now with one key.
func (s *ChatInboundSuite) signed(body []byte, key ed25519.PrivateKey) http.Header {
	return s.signedAs("msg_1", s.now, body, key)
}

func (s *ChatInboundSuite) signedAs(id string, at time.Time, body []byte, key ed25519.PrivateKey) http.Header {
	timestamp := strconv.FormatInt(at.Unix(), 10)
	header := http.Header{}
	header.Set(chatinbound.IDHeader, id)
	header.Set(chatinbound.TimestampHeader, timestamp)
	header.Set(chatinbound.SignatureHeader, "v1a,"+base64.StdEncoding.EncodeToString(
		ed25519.Sign(key, []byte(id+"."+timestamp+"."+string(body)))))
	return header
}

// forged is n well-formed v1a entries signed by a key the router does not hold.
func (s *ChatInboundSuite) forged(body []byte, n int) string {
	stranger := ed25519.NewKeyFromSeed([]byte("somebody-else-s-key-seed-32bytes"))
	entries := make([]string, n)
	for i := range entries {
		entries[i] = s.signed(body, stranger).Get(chatinbound.SignatureHeader)
	}
	return strings.Join(entries, " ")
}

func (s *ChatInboundSuite) public(key ed25519.PrivateKey) ed25519.PublicKey {
	public, ok := key.Public().(ed25519.PublicKey)
	s.Require().True(ok)
	return public
}

func (s *ChatInboundSuite) written(key ed25519.PrivateKey) string {
	return base64.StdEncoding.EncodeToString(s.public(key))
}
