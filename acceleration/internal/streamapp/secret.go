package streamapp

import (
	"fmt"
	"log/slog"
)

// Secret is a Stream app secret. Every way Go has of printing a value prints it redacted,
// so an identity can be logged, wrapped in an error or marshalled without leaking it.
type Secret struct {
	value string
}

// NewSecret wraps a secret.
func NewSecret(value string) Secret { return Secret{value: value} }

// Reveal is the secret itself, for signing with. It is the one way to get it.
func (s Secret) Reveal() string { return s.value }

// Empty reports whether there is no secret.
func (s Secret) Empty() bool { return s.value == "" }

const redacted = "[redacted]"

func (s Secret) String() string   { return redacted }
func (s Secret) GoString() string { return redacted }

// Format covers every verb, %x and %q included, which String alone does not.
func (s Secret) Format(f fmt.State, _ rune) { _, _ = f.Write([]byte(redacted)) }

// MarshalJSON keeps a secret out of anything encoded, an error body or a log line alike.
func (s Secret) MarshalJSON() ([]byte, error) { return []byte(`"` + redacted + `"`), nil }

// LogValue keeps a secret out of structured logs.
func (s Secret) LogValue() slog.Value { return slog.StringValue(redacted) }
