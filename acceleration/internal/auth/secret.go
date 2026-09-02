package auth

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"fmt"
)

// KEKVersion names which key encryption key sealed a row, so that key can be rotated by
// re-wrapping rows rather than by reissuing every secret.
const KEKVersion = 1

// Sealer wraps and unwraps the secrets held in the database.
//
// The received wisdom is to hash an API secret and never hold it back, and it does not
// apply here: verifying a token means recomputing a signature, which means holding the key
// material. Choosing signing is choosing recoverable secrets, so they are encrypted under
// a key that lives outside the database and a leaked backup yields ciphertext.
type Sealer struct {
	aead cipher.AEAD
}

// NewSealer derives the key encryption key from configuration. The value is stretched
// through SHA-256 so an operator can supply a passphrase rather than exactly 32 bytes.
func NewSealer(kek string) (*Sealer, error) {
	if kek == "" {
		return nil, errors.New("auth: a key encryption key is required to store secrets")
	}
	sum := sha256.Sum256([]byte(kek))
	block, err := aes.NewCipher(sum[:])
	if err != nil {
		return nil, fmt.Errorf("auth: new cipher: %w", err)
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("auth: new gcm: %w", err)
	}
	return &Sealer{aead: aead}, nil
}

// Seal encrypts a secret for storage. The nonce is prepended to the ciphertext rather than
// stored beside it, because the two are only ever used together.
func (s *Sealer) Seal(secret string) ([]byte, error) {
	nonce := make([]byte, s.aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, fmt.Errorf("auth: read random: %w", err)
	}
	return s.aead.Seal(nonce, nonce, []byte(secret), nil), nil
}

// Open decrypts a stored secret.
func (s *Sealer) Open(sealed []byte) (string, error) {
	if len(sealed) < s.aead.NonceSize() {
		return "", errors.New("auth: sealed secret is too short")
	}
	nonce, body := sealed[:s.aead.NonceSize()], sealed[s.aead.NonceSize():]
	plain, err := s.aead.Open(nil, nonce, body, nil)
	if err != nil {
		return "", fmt.Errorf("auth: open secret: %w", err)
	}
	return string(plain), nil
}
