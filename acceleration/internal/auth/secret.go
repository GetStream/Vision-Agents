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
	currentVersion int
	aeads          map[int]cipher.AEAD
}

// NewSealer derives the AES key from configuration. Supply a high-entropy secret from the
// deployment's secret store; SHA-256 normalizes its length but does not strengthen a weak
// passphrase.
func NewSealer(kek string) (*Sealer, error) {
	return NewSealerWithKeyring(KEKVersion, map[int]string{KEKVersion: kek})
}

// NewSealerWithKeyring creates a sealer that writes with currentVersion and can still read
// records wrapped by retained older keys. Operators should remove an old key only after
// every record carrying its version has been rewrapped or retired.
func NewSealerWithKeyring(currentVersion int, keys map[int]string) (*Sealer, error) {
	if currentVersion < 1 {
		return nil, errors.New("auth: key version must be positive")
	}
	if keys[currentVersion] == "" {
		return nil, fmt.Errorf("auth: key encryption key version %d is required", currentVersion)
	}
	aeads := make(map[int]cipher.AEAD, len(keys))
	for version, kek := range keys {
		if version < 1 || kek == "" {
			return nil, errors.New("auth: keyring versions and keys must be non-empty and positive")
		}
		sum := sha256.Sum256([]byte(kek))
		block, err := aes.NewCipher(sum[:])
		if err != nil {
			return nil, fmt.Errorf("auth: new cipher for key version %d: %w", version, err)
		}
		aead, err := cipher.NewGCM(block)
		if err != nil {
			return nil, fmt.Errorf("auth: new gcm for key version %d: %w", version, err)
		}
		aeads[version] = aead
	}
	return &Sealer{currentVersion: currentVersion, aeads: aeads}, nil
}

// CurrentVersion is the key version used by new Seal and SealWithAAD operations.
func (s *Sealer) CurrentVersion() int {
	return s.currentVersion
}

// Seal encrypts a secret for storage. The nonce is prepended to the ciphertext rather than
// stored beside it, because the two are only ever used together.
func (s *Sealer) Seal(secret string) ([]byte, error) {
	return s.SealWithAAD(secret, nil)
}

// SealWithAAD encrypts a secret bound to its owning record. The additional authenticated
// data is not encrypted, but moving ciphertext to another record makes decryption fail.
func (s *Sealer) SealWithAAD(secret string, additional []byte) ([]byte, error) {
	return s.SealWithAADVersion(secret, additional, s.currentVersion)
}

// SealWithAADVersion encrypts a secret with one key in the configured keyring.
func (s *Sealer) SealWithAADVersion(secret string, additional []byte, version int) ([]byte, error) {
	aead, ok := s.aeads[version]
	if !ok {
		return nil, fmt.Errorf("auth: key encryption key version %d is unavailable", version)
	}
	nonce := make([]byte, aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, fmt.Errorf("auth: read random: %w", err)
	}
	return aead.Seal(nonce, nonce, []byte(secret), additional), nil
}

// Open decrypts a stored secret.
func (s *Sealer) Open(sealed []byte) (string, error) {
	return s.OpenWithAAD(sealed, nil)
}

// OpenWithAAD decrypts a secret only when the caller supplies the record context used to
// seal it.
func (s *Sealer) OpenWithAAD(sealed, additional []byte) (string, error) {
	return s.OpenWithAADVersion(sealed, additional, s.currentVersion)
}

// OpenWithAADVersion decrypts a record using its persisted key version.
func (s *Sealer) OpenWithAADVersion(sealed, additional []byte, version int) (string, error) {
	aead, ok := s.aeads[version]
	if !ok {
		return "", fmt.Errorf("auth: key encryption key version %d is unavailable", version)
	}
	if len(sealed) < aead.NonceSize() {
		return "", errors.New("auth: sealed secret is too short")
	}
	nonce, body := sealed[:aead.NonceSize()], sealed[aead.NonceSize():]
	plain, err := aead.Open(nil, nonce, body, additional)
	if err != nil {
		return "", fmt.Errorf("auth: open secret: %w", err)
	}
	return string(plain), nil
}
