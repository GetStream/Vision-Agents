package auth

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"hash/crc32"
	"strings"
)

// Environment separates a key that spends money from one that does not. It is in the
// credential itself so that a production secret pasted into a test config is visible as
// one before it is used rather than after.
type Environment string

const (
	Live Environment = "live"
	Test Environment = "test"
)

const (
	keyPrefix    = "vak"
	secretPrefix = "vas"
)

// NewCredential mints the two halves of a credential. The key id travels in the clear and
// names which secret to verify with; the secret is shown once and then only ever held
// sealed.
func NewCredential(env Environment) (key, secret string, err error) {
	if env != Live && env != Test {
		return "", "", fmt.Errorf("auth: unknown environment %q", env)
	}

	body := make([]byte, 8)
	if _, err := rand.Read(body); err != nil {
		return "", "", fmt.Errorf("auth: read random: %w", err)
	}
	// The checksum lets a client library reject a truncated paste locally, with a message
	// that says the key is malformed rather than a 401 that reads as a permissions problem.
	id := hex.EncodeToString(body)
	key = fmt.Sprintf("%s_%s_%s%08x", keyPrefix, env, id, crc32.ChecksumIEEE([]byte(id)))

	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", "", fmt.Errorf("auth: read random: %w", err)
	}
	secret = fmt.Sprintf("%s_%s_%s", secretPrefix, env, base64.RawURLEncoding.EncodeToString(raw))

	return key, secret, nil
}

// ValidKey reports whether a key is well formed and its checksum matches. It says nothing
// about whether the key exists: it is the cheap check that runs before the database one.
func ValidKey(key string) bool {
	parts := strings.Split(key, "_")
	if len(parts) != 3 || parts[0] != keyPrefix {
		return false
	}
	if Environment(parts[1]) != Live && Environment(parts[1]) != Test {
		return false
	}
	if len(parts[2]) != 24 {
		return false
	}
	id, sum := parts[2][:16], parts[2][16:]
	if _, err := hex.DecodeString(id); err != nil {
		return false
	}
	return sum == fmt.Sprintf("%08x", crc32.ChecksumIEEE([]byte(id)))
}

// Last4 is all of a secret the dashboard ever shows again.
func Last4(secret string) string {
	if len(secret) < 4 {
		return ""
	}
	return secret[len(secret)-4:]
}
