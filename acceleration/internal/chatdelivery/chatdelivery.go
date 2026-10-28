// Package chatdelivery checks that a request came from Stream Chat's own delivery of
// messages to the router, and from nobody else.
//
// The request is signed as Standard Webhooks signs one with a public key
// (https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md,
// «Asymmetric signatures»): v1a, the ed25519 signature of {webhook-id}.{webhook-timestamp}.{body}.
// Stream holds the private key and the router only the public one, so a router that leaks
// its configuration still cannot sign a delivery. No app's API secret is ever a key here:
// a customer's server holds that secret, and anything signed with it is the customer's.
//
// The specification's Go library (github.com/standard-webhooks/standard-webhooks/libraries,
// v0.0.1, its only version) checks only the symmetric v1, so the check is written here on
// crypto/ed25519.
package chatdelivery

import (
	"crypto/ed25519"
	"encoding/base64"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// The specification's headers: «webhook-id», «webhook-timestamp» (integer Unix seconds) and
// «webhook-signature», a space-delimited list of signatures.
const (
	IDHeader        = "Webhook-Id"
	TimestampHeader = "Webhook-Timestamp"
	SignatureHeader = "Webhook-Signature"
)

// signaturePrefix marks an ed25519 signature: «v1a, followed by a comma (,), and the base64
// encoded signature». A v1 signature is an HMAC under a shared secret, which the router does
// not take.
const signaturePrefix = "v1a,"

// Tolerance is how far a signed timestamp may be from now, either way, and still be
// believed, so a captured request cannot be replayed later. Five minutes, as Stripe
// (https://docs.stripe.com/webhooks#replay-attacks) and Slack
// (https://api.slack.com/authentication/verifying-requests-from-slack) allow.
const Tolerance = 5 * time.Minute

// maxKeys is the keys the router holds at once: the current one, and the one before it
// while Stream rotates.
const maxKeys = 2

// ErrUnsigned is a request Stream cannot be proven to have sent: no id, timestamp or
// signature, one in another shape, or none that a key verifies. It never says which, so an
// answer teaches a forger nothing.
var ErrUnsigned = errors.New("chatdelivery: the request is not signed by Stream")

// ErrStale is a signed timestamp further from now than Tolerance, in either direction: a
// recording of a real request replayed later, or a clock far ahead.
var ErrStale = errors.New("chatdelivery: the request's signed timestamp is too far from now")

// ParsePublicKeys reads the keys as configured: each the standard base64 of a 32-byte
// ed25519 public key, one or two of them. None is no keys, which leaves deliveries off.
func ParsePublicKeys(written []string) ([]ed25519.PublicKey, error) {
	if len(written) > maxKeys {
		return nil, stack.Wrap(errors.New("chatdelivery: at most two delivery keys are held at once, the current one and the one before it"))
	}
	var keys []ed25519.PublicKey
	for _, encoded := range written {
		key, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || len(key) != ed25519.PublicKeySize {
			// The value is a public key, but naming its position is enough to find it.
			return nil, stack.Wrap(errors.New("chatdelivery: delivery key " + strconv.Itoa(len(keys)+1) +
				" is not the base64 of a 32-byte ed25519 public key"))
		}
		keys = append(keys, key)
	}
	return keys, nil
}

// Verify checks a delivery's signature before anything reads its body. It answers nil when
// one of the header's v1a signatures verifies {id}.{timestamp}.{body} under one of keys,
// since Stream signs with both keys while it rotates («Multiple signatures are space
// delimited»).
func Verify(keys []ed25519.PublicKey, header http.Header, body []byte, now time.Time) error {
	id, timestamp := header.Get(IDHeader), header.Get(TimestampHeader)
	seconds, err := strconv.ParseInt(timestamp, 10, 64)
	if len(keys) == 0 || id == "" || err != nil {
		return stack.Wrap(ErrUnsigned)
	}
	if now.Sub(time.Unix(seconds, 0)).Abs() > Tolerance {
		return stack.Wrap(ErrStale)
	}
	signed := append([]byte(id+"."+timestamp+"."), body...)
	for _, signature := range strings.Fields(header.Get(SignatureHeader)) {
		written, ok := strings.CutPrefix(signature, signaturePrefix)
		if !ok {
			continue
		}
		sent, err := base64.StdEncoding.DecodeString(written)
		if err != nil || len(sent) != ed25519.SignatureSize {
			continue
		}
		for _, key := range keys {
			if ed25519.Verify(key, signed, sent) {
				return nil
			}
		}
	}
	return stack.Wrap(ErrUnsigned)
}
