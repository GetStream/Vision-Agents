package oauth2code

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// PrivateKeyJWTSuite checks the private_key_jwt assertion on its own: nothing in the scheme
// sends one yet, and the fake provider does not accept one.
type PrivateKeyJWTSuite struct {
	suite.Suite
	key *rsa.PrivateKey
	now time.Time
}

func TestPrivateKeyJWTSuite(t *testing.T) {
	suite.Run(t, new(PrivateKeyJWTSuite))
}

func (s *PrivateKeyJWTSuite) SetupSuite() {
	// 2048 bits: RFC 7518 sections 3.3 and 3.5 require «2048 bits or larger» for RS256 and
	// PS256.
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	s.Require().NoError(err)
	s.key = key
	s.now = time.Unix(1_800_000_000, 0)
}

func (s *PrivateKeyJWTSuite) TestAnRS256AssertionVerifiesWithThePublicKey() {
	header, signingInput, signature := s.assertion(PrivateKeyJWT{Key: s.key, Alg: "RS256"})
	s.Equal("RS256", header["alg"])
	digest := sha256.Sum256([]byte(signingInput))
	s.NoError(rsa.VerifyPKCS1v15(&s.key.PublicKey, crypto.SHA256, digest[:], signature))
}

func (s *PrivateKeyJWTSuite) TestAPS256AssertionVerifiesWithThePublicKey() {
	header, signingInput, signature := s.assertion(PrivateKeyJWT{Key: s.key, Alg: "PS256"})
	s.Equal("PS256", header["alg"])
	digest := sha256.Sum256([]byte(signingInput))
	s.NoError(rsa.VerifyPSS(&s.key.PublicKey, crypto.SHA256, digest[:], signature, &rsa.PSSOptions{SaltLength: rsa.PSSSaltLengthEqualsHash}))
}

func (s *PrivateKeyJWTSuite) TestTheAssertionNamesTheClientTheAudienceAndAShortLife() {
	form, err := PrivateKeyJWT{Key: s.key, Alg: "RS256"}.Form("client-1", "https://issuer.example/token", s.now)
	s.Require().NoError(err)
	s.Equal("urn:ietf:params:oauth:client-assertion-type:jwt-bearer", form.Get("client_assertion_type"))
	claims := s.segment(form.Get("client_assertion"), 1)
	s.Equal("client-1", claims["iss"])
	s.Equal("client-1", claims["sub"])
	s.Equal("https://issuer.example/token", claims["aud"])
	s.InDelta(s.now.Unix(), claims["iat"], 0)
	s.InDelta(s.now.Unix(), claims["nbf"], 0)
	s.InDelta(s.now.Add(5*time.Minute).Unix(), claims["exp"], 0)
	s.Len(claims["jti"], 43)

	again, err := PrivateKeyJWT{Key: s.key, Alg: "RS256"}.Form("client-1", "https://issuer.example/token", s.now)
	s.Require().NoError(err)
	s.NotEqual(claims["jti"], s.segment(again.Get("client_assertion"), 1)["jti"], "each assertion has its own jti")
}

func (s *PrivateKeyJWTSuite) TestTheKeyIDAndCertificateThumbprintGoInTheHeader() {
	thumbprint := sha256.Sum256([]byte("certificate DER"))
	form, err := PrivateKeyJWT{Key: s.key, Alg: "PS256", KeyID: "key-1", CertificateSHA256: thumbprint[:]}.Form("client-1", "https://issuer.example", s.now)
	s.Require().NoError(err)
	header := s.segment(form.Get("client_assertion"), 0)
	s.Equal("key-1", header["kid"])
	s.Equal(base64.RawURLEncoding.EncodeToString(thumbprint[:]), header["x5t#S256"])
}

func (s *PrivateKeyJWTSuite) TestAnAlgorithmOtherThanRS256OrPS256IsRefused() {
	_, err := PrivateKeyJWT{Key: s.key, Alg: "HS256"}.Form("client-1", "https://issuer.example", s.now)
	s.Error(err)
}

func (s *PrivateKeyJWTSuite) TestNoKeyIsRefused() {
	_, err := PrivateKeyJWT{Alg: "RS256"}.Form("client-1", "https://issuer.example", s.now)
	s.Error(err)
}

func (s *PrivateKeyJWTSuite) assertion(k PrivateKeyJWT) (map[string]any, string, []byte) {
	form, err := k.Form("client-1", "https://issuer.example/token", s.now)
	s.Require().NoError(err)
	jwt := form.Get("client_assertion")
	cut := strings.LastIndex(jwt, ".")
	signature, err := base64.RawURLEncoding.DecodeString(jwt[cut+1:])
	s.Require().NoError(err)
	return s.segment(jwt, 0), jwt[:cut], signature
}

func (s *PrivateKeyJWTSuite) segment(jwt string, i int) map[string]any {
	parts := strings.Split(jwt, ".")
	s.Require().Len(parts, 3)
	raw, err := base64.RawURLEncoding.DecodeString(parts[i])
	s.Require().NoError(err)
	var out map[string]any
	s.Require().NoError(json.Unmarshal(raw, &out))
	return out
}
