// Package telnyxsig checks that a request came from Telnyx. Telnyx signs every webhook,
// 10DLC reports and call webhooks alike, with Ed25519 over "{timestamp}|{body}".
package telnyxsig

import (
	"crypto/ed25519"
	"encoding/base64"
	"errors"
	"strconv"
	"time"
)

// PublicKeyEnvVar holds the account's base64 Ed25519 public key, from the Telnyx portal.
const PublicKeyEnvVar = "TELNYX_PUBLIC_KEY"

const (
	SignatureHeader = "Telnyx-Signature-Ed25519"
	TimestampHeader = "Telnyx-Timestamp"
)

// Tolerance is how far a signed timestamp may be from now, either way, and still be
// believed, so a captured request cannot be replayed later.
const Tolerance = 5 * time.Minute

// ParsePublicKey reads the key as the Telnyx portal shows it.
func ParsePublicKey(encoded string) (ed25519.PublicKey, error) {
	key, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil || len(key) != ed25519.PublicKeySize {
		return nil, errors.New("telnyx: " + PublicKeyEnvVar + " is not a base64 Ed25519 public key")
	}
	return key, nil
}

// Verify checks the signature over the timestamp and the raw body.
func Verify(publicKey ed25519.PublicKey, timestamp, signature string, body []byte, now time.Time) error {
	if len(publicKey) != ed25519.PublicKeySize {
		return errors.New("no Telnyx public key to check the report against")
	}
	seconds, err := strconv.ParseInt(timestamp, 10, 64)
	if err != nil {
		return errors.New("the report carries no timestamp")
	}
	if age := now.Sub(time.Unix(seconds, 0)); age > Tolerance || age < -Tolerance {
		return errors.New("the report is too old to act on")
	}
	decoded, err := base64.StdEncoding.DecodeString(signature)
	if err != nil || len(decoded) == 0 {
		return errors.New("the report is not signed")
	}
	signed := append([]byte(strconv.FormatInt(seconds, 10)+"|"), body...)
	if !ed25519.Verify(publicKey, signed, decoded) {
		return errors.New("the report is not signed by Telnyx")
	}
	return nil
}
