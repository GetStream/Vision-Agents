package core

import (
	"crypto/sha256"
	"encoding/hex"
	"time"
)

// fingerprintBytes is how much of a token's SHA-256 a fingerprint keeps: 4 bytes, 8 hex
// characters, as AI-990 asked. It tells one token of a connection from the next; two
// different tokens share one with odds of 1 in 2^32, which a log reader can live with.
const fingerprintBytes = 4

// Fingerprint names token in a log line or an audit row without revealing it: the lowercase
// hex of the first 4 bytes of its SHA-256. Two equal fingerprints say the same token was
// kept, two different ones that it was replaced. The empty token has the empty fingerprint.
//
// It is a hash prefix, not the token's last characters: any raw characters of a token are
// secret material, a part of it a reader holds and a guess no longer has to find, while a hash
// identifies the token without carrying any of it. It is the only way a token reaches a log.
func Fingerprint(token string) string {
	if token == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:fingerprintBytes])
}

// Fingerprinter is a Scheme that can name the tokens in its stored credentials by
// Fingerprint, so a credential's life can be followed in logs and the audit without a secret
// (AI-990). It is optional: a scheme that does not implement it is logged without
// fingerprints.
type Fingerprinter interface {
	// Fingerprints are stored's tokens as Fingerprint names them, with their expiry. An
	// error means stored is not credentials this scheme wrote, and never quotes them.
	Fingerprints(stored StoredCredentials) (CredentialFingerprints, error)
}

// CredentialFingerprints names the tokens of one set of stored credentials without a secret.
type CredentialFingerprints struct {
	// Access and Refresh are the tokens' Fingerprint, empty when there is no such token.
	Access  string
	Refresh string
	// AccessExpiresAt and RefreshExpiresAt are zero when the provider did not say.
	AccessExpiresAt  time.Time
	RefreshExpiresAt time.Time
}

// CredentialChange is what one credential event did to a connection's tokens: the
// fingerprints before it, zero for a first grant, and after it.
type CredentialChange struct {
	Previous CredentialFingerprints
	Current  CredentialFingerprints
}

// Rotated says the event replaced a refresh token the connection already had, as a provider
// that rotates them does on every refresh (RFC 6749 section 6).
func (c CredentialChange) Rotated() bool {
	return c.Previous.Refresh != "" && c.Previous.Refresh != c.Current.Refresh
}

// LogAttrs are the change as slog key-value pairs, the names the audit's columns have too.
func (c CredentialChange) LogAttrs() []any {
	return []any{
		"previous_access_fingerprint", c.Previous.Access, "access_fingerprint", c.Current.Access,
		"previous_refresh_fingerprint", c.Previous.Refresh, "refresh_fingerprint", c.Current.Refresh,
		"rotated", c.Rotated(),
		"access_expires_at", c.Current.AccessExpiresAt, "refresh_expires_at", c.Current.RefreshExpiresAt,
	}
}

// FingerprintsOf is stored's fingerprints by the scheme in schemes that wrote it, and the
// zero value when that scheme is not a Fingerprinter or cannot read them: a missing
// fingerprint leaves a log line short, never a credential event undone.
func FingerprintsOf(schemes map[string]Scheme, stored StoredCredentials) CredentialFingerprints {
	fingerprinter, ok := schemes[stored.Scheme].(Fingerprinter)
	if !ok {
		return CredentialFingerprints{}
	}
	fingerprints, err := fingerprinter.Fingerprints(stored)
	if err != nil {
		return CredentialFingerprints{}
	}
	return fingerprints
}
