package oauth2code

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"time"
)

// clientAssertionType is RFC 7523 section 2.2's client_assertion_type for a JWT, which
// OpenID Connect Core 1.0 section 9 requires of private_key_jwt.
const clientAssertionType = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"

// assertionLifetime is how long an assertion is accepted. RFC 7523 section 3 item 4 makes
// exp required but sets no length; five minutes is the short end of what Microsoft asks of
// a certificate credential (architecture doc, stress-test row 8: «exp of 5 to 10 minutes»,
// from learn.microsoft.com/en-us/entra/identity-platform/certificate-credentials), and one
// token request needs far less.
const assertionLifetime = 5 * time.Minute

// PrivateKeyJWT is the private_key_jwt client authentication method: the client signs a
// JWT with its own key instead of sending a secret (OpenID Connect Core 1.0 section 9; RFC
// 7523 section 2.2 and section 3; RFC 7521 section 4.2). Nothing selects it yet:
// supportedMethods leaves it out until a client record can hold a private key (T19).
type PrivateKeyJWT struct {
	Key *rsa.PrivateKey
	// Alg is the manifest's client.alg, RS256 or PS256 (RFC 7518 sections 3.3 and 3.5;
	// core.Manifest validates it to one of those two).
	Alg string
	// KeyID, when set, is the kid header (RFC 7515 section 4.1.4), how a server holding
	// several of the client's keys finds this one.
	KeyID string
	// CertificateSHA256, when set, is the SHA-256 digest of the DER certificate for Key, sent
	// as the x5t#S256 header (RFC 7515 section 4.1.8), which Microsoft requires of a
	// certificate credential (architecture doc, stress-test row 8).
	CertificateSHA256 []byte
}

// Form is the client_assertion_type and client_assertion that authenticate clientID at a
// token endpoint, signed at now. audience is the aud claim: OpenID Connect Core 1.0
// section 9 says it «SHOULD be the URL of the Authorization Server's Token Endpoint», which
// is what Microsoft takes, while draft-ietf-oauth-rfc7523bis-11 section 4 requires the
// issuer identifier and forbids the token endpoint. Which one is the caller's, from the
// manifest, so this takes it as given.
func (k PrivateKeyJWT) Form(clientID, audience string, now time.Time) (url.Values, error) {
	if k.Key == nil {
		return nil, errors.New("oauth2code: private_key_jwt needs a key")
	}
	if clientID == "" || audience == "" {
		return nil, errors.New("oauth2code: private_key_jwt needs a client id and an audience")
	}
	jti, err := random()
	if err != nil {
		return nil, err
	}
	header := map[string]any{"alg": k.Alg, "typ": "JWT"}
	if k.KeyID != "" {
		header["kid"] = k.KeyID
	}
	if len(k.CertificateSHA256) > 0 {
		header["x5t#S256"] = base64.RawURLEncoding.EncodeToString(k.CertificateSHA256)
	}
	// OpenID Connect Core 1.0 section 9: iss and sub are the client_id, aud the audience,
	// jti unique, exp required, iat optional. nbf is RFC 7523 section 3 item 5's, which
	// Microsoft lists (architecture doc, stress-test row 8).
	claims := map[string]any{
		"iss": clientID,
		"sub": clientID,
		"aud": audience,
		"jti": jti,
		"iat": now.Unix(),
		"nbf": now.Unix(),
		"exp": now.Add(assertionLifetime).Unix(),
	}
	signingInput, err := jwsSigningInput(header, claims)
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256([]byte(signingInput))
	var signature []byte
	switch k.Alg {
	case "RS256":
		// RFC 7518 section 3.3: RSASSA-PKCS1-v1_5 with SHA-256.
		signature, err = rsa.SignPKCS1v15(rand.Reader, k.Key, crypto.SHA256, digest[:])
	case "PS256":
		// RFC 7518 section 3.5: RSASSA-PSS with SHA-256, MGF1 with SHA-256, and a salt «the
		// same size as the hash function output».
		signature, err = rsa.SignPSS(rand.Reader, k.Key, crypto.SHA256, digest[:], &rsa.PSSOptions{SaltLength: rsa.PSSSaltLengthEqualsHash})
	default:
		return nil, fmt.Errorf("oauth2code: private_key_jwt alg %q is not RS256 or PS256", k.Alg)
	}
	if err != nil {
		return nil, err
	}
	form := url.Values{}
	form.Set("client_assertion_type", clientAssertionType)
	// RFC 7521 section 4.2 names the parameter; RFC 7523 section 2.2: «It MUST NOT contain
	// more than one JWT».
	form.Set("client_assertion", signingInput+"."+base64.RawURLEncoding.EncodeToString(signature))
	return form, nil
}

// jwsSigningInput is RFC 7515 section 5.1's signing input of the compact serialization:
// BASE64URL(header) "." BASE64URL(payload).
func jwsSigningInput(header, claims map[string]any) (string, error) {
	h, err := json.Marshal(header)
	if err != nil {
		return "", err
	}
	c, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(h) + "." + base64.RawURLEncoding.EncodeToString(c), nil
}
