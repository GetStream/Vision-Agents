// Package hmacheader is the hmac_header verifier: an HMAC of the raw body, and optionally of
// a timestamp header, sent in a request header, with every parameter read from the
// manifest's channel.verifier block (core/channel.go). Slack's request signing is one.
package hmacheader

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrUnsigned is a request this verifier cannot prove the provider sent: no signature, one in
// another shape, one that does not match, or no secret to check it with. It never says which,
// so an answer teaches a forger nothing.
var ErrUnsigned = errors.New("hmacheader: the request is not signed by the provider")

// ErrStale is a signed timestamp further from now than channel.verifier.max_age, in either
// direction: a recording of a real request replayed later, or a clock far ahead.
var ErrStale = errors.New("hmacheader: the request's signed timestamp is too far from now")

// Verifier implements core.Verifier for channel.verifier.kind hmac_header.
type Verifier struct{}

var _ core.Verifier = (*Verifier)(nil)

// New is the verifier, judging timestamps by the system clock.
func New() *Verifier {
	return &Verifier{}
}

// Name is the kind a manifest names it by.
func (*Verifier) Name() string { return string(core.VerifierHMACHeader) }

// Verify checks the request's signature against secret, by m's channel.verifier, before
// anything reads the body, then reads the body with m's channel block.
//
// The header holds prefix followed by the digest, written in encoding. The HMAC is over the
// signed template with {body} the raw body and {timestamp} the timestamp header as sent.
// That header is decimal Unix seconds, which is what Slack sends («Verifying requests from
// Slack», https://docs.slack.dev/authentication/verifying-requests-from-slack, opened
// October 6, 2026); a provider writing it another way needs a parameter for it in
// core.VerifierRule first. The digests are compared with hmac.Equal, in constant time, as
// that page asks («use an hmac compare function instead of directly comparing»).
func (v *Verifier) Verify(r *http.Request, body []byte, m core.Manifest, secret []byte) (core.VerifiedEvent, error) {
	if m.Channel == nil || m.Channel.Verifier.Kind != core.VerifierHMACHeader {
		return core.VerifiedEvent{}, stack.Wrap(fmt.Errorf("hmacheader: manifest %q has no hmac_header verifier", m.ID))
	}
	rule := m.Channel.Verifier
	if len(secret) == 0 {
		// An HMAC under an empty key is one anybody can compute.
		return core.VerifiedEvent{}, stack.Wrap(ErrUnsigned)
	}
	written, prefixed := strings.CutPrefix(r.Header.Get(rule.Header), rule.Prefix)
	if !prefixed || written == "" {
		return core.VerifiedEvent{}, stack.Wrap(ErrUnsigned)
	}
	sent, err := decode(rule.Encoding, written)
	if err != nil {
		return core.VerifiedEvent{}, stack.Wrap(ErrUnsigned)
	}
	timestamp := ""
	if rule.TimestampHeader != "" {
		timestamp = r.Header.Get(rule.TimestampHeader)
		seconds, err := strconv.ParseInt(timestamp, 10, 64)
		if err != nil {
			return core.VerifiedEvent{}, stack.Wrap(ErrUnsigned)
		}
		if time.Since(time.Unix(seconds, 0)).Abs() > time.Duration(rule.MaxAge) {
			return core.VerifiedEvent{}, stack.Wrap(ErrStale)
		}
	}
	newHash, err := algorithm(rule.Algorithm)
	if err != nil {
		return core.VerifiedEvent{}, err
	}
	mac := hmac.New(newHash, secret)
	// Replacer reads only the template and never scans what it put in, so a body that holds
	// the text {timestamp} is signed as it is. Validate allows no other brace.
	_, _ = strings.NewReplacer("{body}", string(body), "{timestamp}", timestamp).WriteString(mac, rule.Signed)
	if !hmac.Equal(mac.Sum(nil), sent) {
		return core.VerifiedEvent{}, stack.Wrap(ErrUnsigned)
	}

	read, err := m.Channel.Read(m.ID, body)
	if err != nil {
		// The signature proved the provider sent body, so a body the block does not describe
		// is a verified request with nothing to act on (core.Verifier).
		return core.VerifiedEvent{}, nil
	}
	event := core.VerifiedEvent{Challenge: read.Challenge, Signals: read.Signals, Skipped: read.Skipped}
	for _, message := range read.Messages {
		event.Messages = append(event.Messages, message.InboundMessage)
	}
	return event, nil
}

// algorithm is the hash of channel.verifier.algorithm. sha256 is the only one core allows
// (hmacAlgorithms in core/channel.go).
func algorithm(name string) (func() hash.Hash, error) {
	switch name {
	case "sha256":
		return sha256.New, nil
	default:
		return nil, stack.Wrap(fmt.Errorf("hmacheader: algorithm %q is not one this verifier computes", name))
	}
}

// decode reads the digest as channel.verifier.encoding writes it. hex is the only encoding
// core allows (hmacEncodings in core/channel.go).
func decode(encoding, written string) ([]byte, error) {
	switch encoding {
	case "hex":
		return hex.DecodeString(written)
	default:
		return nil, fmt.Errorf("hmacheader: encoding %q is not one this verifier reads", encoding)
	}
}
