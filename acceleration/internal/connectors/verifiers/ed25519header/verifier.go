// Package ed25519header is the ed25519 verifier: an Ed25519 signature (RFC 8032) of the raw
// body, and optionally of a timestamp header, sent in base64 in a request header and checked
// with the provider's public key, every parameter but the algorithm read from the manifest's
// channel.verifier block (core/channel.go). Telnyx signs its webhooks this way
// (https://developers.telnyx.com/docs/messaging/messages/receiving-webhooks, opened October 8,
// 2026), as internal/channels/telnyx.go verifies them.
package ed25519header

import (
	"crypto/ed25519"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrUnsigned is a request this verifier cannot prove the provider sent: no signature, one in
// another shape, one that does not match, or no usable public key. It never says which, so an
// answer teaches a forger nothing.
var ErrUnsigned = errors.New("ed25519header: the request is not signed by the provider")

// ErrStale is a signed timestamp further from now than channel.verifier.max_age, in either
// direction: a recording of a real request replayed later, or a clock far ahead.
var ErrStale = errors.New("ed25519header: the request's signed timestamp is too far from now")

// Verifier implements core.Verifier for channel.verifier.kind ed25519.
type Verifier struct{}

var _ core.Verifier = (*Verifier)(nil)

// New is the verifier, judging timestamps by the system clock.
func New() *Verifier {
	return &Verifier{}
}

// Name is the kind a manifest names it by.
func (*Verifier) Name() string { return string(core.VerifierEd25519) }

// Verify checks the request's signature with the public key secret, by m's
// channel.verifier, before anything reads the body, then reads the body with m's channel
// block.
//
// secret is the provider's public key in base64, as Telnyx shows it («Public Key» under «Keys
// & Credentials»; internal/channels/telnyx.go reads it so). The header holds the base64
// signature. It covers the signed template with {body} the raw body and {timestamp} the
// timestamp header as sent, decimal Unix seconds, which is what Telnyx sends: «telnyx-timestamp
// — Unix timestamp of when signed», signed as «{timestamp}|{json_payload}» (the page above).
func (v *Verifier) Verify(r *http.Request, body []byte, m core.Manifest, secret []byte) (core.VerifiedEvent, error) {
	if m.Channel == nil || m.Channel.Verifier.Kind != core.VerifierEd25519 {
		return core.VerifiedEvent{}, stack.Wrap(fmt.Errorf("ed25519header: manifest %q has no ed25519 verifier", m.ID))
	}
	rule := m.Channel.Verifier
	key, err := base64.StdEncoding.DecodeString(strings.TrimSpace(string(secret)))
	if err != nil || len(key) != ed25519.PublicKeySize {
		return core.VerifiedEvent{}, stack.Wrap(ErrUnsigned)
	}
	signature, err := base64.StdEncoding.DecodeString(r.Header.Get(rule.Header))
	if err != nil || len(signature) != ed25519.SignatureSize {
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
	// Replacer reads only the template and never scans what it put in, so a body that holds
	// the text {timestamp} is signed as it is. Validate allows no other brace.
	signed := strings.NewReplacer("{body}", string(body), "{timestamp}", timestamp).Replace(rule.Signed)
	if !ed25519.Verify(ed25519.PublicKey(key), []byte(signed), signature) {
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
