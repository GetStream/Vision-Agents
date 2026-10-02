package oauth2code

import (
	"encoding/base64"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// IDTokenSuite covers the id_token claims check on its own, since the fake provider issues
// no id_token. The token is unsigned on purpose: OpenID Connect Core 1.0 section 3.1.3.7
// item 6 lets TLS to the token endpoint stand in for the signature, so a valid one passes
// whatever its signature segment holds.
type IDTokenSuite struct {
	suite.Suite
	now time.Time
}

func TestIDTokenSuite(t *testing.T) {
	suite.Run(t, new(IDTokenSuite))
}

func (s *IDTokenSuite) SetupTest() {
	s.now = time.Unix(1_800_000_000, 0)
}

func (s *IDTokenSuite) TestAnIDTokenFromTheIssuerForThisClientPassesUnsigned() {
	s.NoError(checkIDToken(s.token(map[string]any{"iss": "https://issuer.example", "aud": "client", "exp": s.now.Unix() + 60}), "https://issuer.example", "client", s.now))
	s.NoError(checkIDToken(s.token(map[string]any{"iss": "https://issuer.example", "aud": []string{"client"}, "exp": s.now.Unix() + 60}), "https://issuer.example", "client", s.now))
	s.NoError(checkIDToken(s.token(map[string]any{"iss": "https://issuer.example", "aud": "client", "azp": "client", "exp": s.now.Unix() + 60}), "https://issuer.example", "client", s.now))
}

// TestAnIDTokenWithAnotherAudienceTooIsRefused is OpenID Connect Core 1.0 section 3.1.3.7
// item 3 in errata set 2: a token that «contains additional audiences not trusted by the
// Client» is rejected, and this client trusts no other audience.
func (s *IDTokenSuite) TestAnIDTokenWithAnotherAudienceTooIsRefused() {
	s.Error(checkIDToken(s.token(map[string]any{"iss": "https://issuer.example", "aud": []string{"other", "client"}, "exp": s.now.Unix() + 60}), "https://issuer.example", "client", s.now))
	s.Error(checkIDToken(s.token(map[string]any{"iss": "https://issuer.example", "aud": []string{"other", "client"}, "azp": "client", "exp": s.now.Unix() + 60}), "https://issuer.example", "client", s.now))
}

func (s *IDTokenSuite) TestAnIDTokenAuthorizedForAnotherPartyIsRefused() {
	s.Error(checkIDToken(s.token(map[string]any{"iss": "https://issuer.example", "aud": "client", "azp": "other", "exp": s.now.Unix() + 60}), "https://issuer.example", "client", s.now))
}

func (s *IDTokenSuite) TestAnIDTokenFromAnotherIssuerIsRefused() {
	s.Error(checkIDToken(s.token(map[string]any{"iss": "https://attacker.example", "aud": "client", "exp": s.now.Unix() + 60}), "https://issuer.example", "client", s.now))
}

func (s *IDTokenSuite) TestAnIDTokenForAnotherClientIsRefused() {
	s.Error(checkIDToken(s.token(map[string]any{"iss": "https://issuer.example", "aud": "other", "exp": s.now.Unix() + 60}), "https://issuer.example", "client", s.now))
}

func (s *IDTokenSuite) TestAnExpiredIDTokenIsRefused() {
	s.Error(checkIDToken(s.token(map[string]any{"iss": "https://issuer.example", "aud": "client", "exp": s.now.Unix()}), "https://issuer.example", "client", s.now))
	s.Error(checkIDToken(s.token(map[string]any{"iss": "https://issuer.example", "aud": "client"}), "https://issuer.example", "client", s.now))
}

func (s *IDTokenSuite) TestAMissingIDTokenIsRefused() {
	s.Error(checkIDToken(tokenResponse{AccessToken: "a"}, "https://issuer.example", "client", s.now))
}

// token is a token response whose id_token carries claims, with an empty signature.
func (s *IDTokenSuite) token(claims map[string]any) tokenResponse {
	payload, err := json.Marshal(claims)
	s.Require().NoError(err)
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256"}`))
	return tokenResponse{AccessToken: "a", IDToken: header + "." + base64.RawURLEncoding.EncodeToString(payload) + "."}
}
