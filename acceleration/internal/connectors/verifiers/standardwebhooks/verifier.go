// Package standardwebhooks is the standard_webhooks verifier: the Standard Webhooks signature
// (https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md,
// opened October 8, 2026), whose headers, signed content and secret format the specification
// fixes, so the manifest's channel.verifier block gives it only max_age (core/channel.go).
// Linq signs its webhooks this way (https://docs.linqapp.com/guides/webhooks/index.md, opened
// October 8, 2026).
package standardwebhooks

import (
	"crypto/hmac"
	"crypto/sha256"
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

// The specification's headers: «webhook-id», «webhook-timestamp» (integer Unix seconds) and
// «webhook-signature», a space-delimited list of signatures. Linq's page names the same three.
const (
	idHeader        = "Webhook-Id"
	timestampHeader = "Webhook-Timestamp"
	signatureHeader = "Webhook-Signature"
)

// The specification: a symmetric signature is «v1, followed by a comma (,), and the base64
// encoded signature», and a symmetric secret is «base64 encoded, prefixed with whsec_». The
// asymmetric v1a signatures are not read: Linq's page names only v1.
const (
	signaturePrefix = "v1,"
	secretPrefix    = "whsec_"
)

// ErrUnsigned is a request this verifier cannot prove the provider sent: no id, timestamp or
// signature, one in another shape, none that matches, or no usable secret. It never says
// which, so an answer teaches a forger nothing.
var ErrUnsigned = errors.New("standardwebhooks: the request is not signed by the provider")

// ErrStale is a signed timestamp further from now than channel.verifier.max_age, in either
// direction: a recording of a real request replayed later, or a clock far ahead.
var ErrStale = errors.New("standardwebhooks: the request's signed timestamp is too far from now")

// Verifier implements core.Verifier for channel.verifier.kind standard_webhooks.
type Verifier struct{}

var _ core.Verifier = (*Verifier)(nil)

// New is the verifier, judging timestamps by the system clock.
func New() *Verifier {
	return &Verifier{}
}

// Name is the kind a manifest names it by.
func (*Verifier) Name() string { return string(core.VerifierStandardWebhooks) }

// Verify checks the request's signature against secret before anything reads the body, then
// reads the body with m's channel block.
//
// The secret is the whsec_ secret as the provider shows it; the prefix is optional, as
// internal/channels/linq.go reads it, and the rest is base64. The HMAC-SHA256 is over
// {webhook-id}.{webhook-timestamp}.{body}, and any one of the header's v1 signatures may
// match, since a provider rotating its secret signs with both («Multiple signatures are space
// delimited»). Each is compared with hmac.Equal, in constant time, as the specification asks
// («use a constant time comparison function»).
func (v *Verifier) Verify(r *http.Request, body []byte, m core.Manifest, secret []byte) (core.VerifiedEvent, error) {
	if m.Channel == nil || m.Channel.Verifier.Kind != core.VerifierStandardWebhooks {
		return core.VerifiedEvent{}, stack.Wrap(fmt.Errorf("standardwebhooks: manifest %q has no standard_webhooks verifier", m.ID))
	}
	key, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(string(secret), secretPrefix))
	if err != nil || len(key) == 0 {
		// An HMAC under an empty key is one anybody can compute.
		return core.VerifiedEvent{}, stack.Wrap(ErrUnsigned)
	}
	id, timestamp := r.Header.Get(idHeader), r.Header.Get(timestampHeader)
	seconds, err := strconv.ParseInt(timestamp, 10, 64)
	if id == "" || err != nil {
		return core.VerifiedEvent{}, stack.Wrap(ErrUnsigned)
	}
	if time.Since(time.Unix(seconds, 0)).Abs() > time.Duration(m.Channel.Verifier.MaxAge) {
		return core.VerifiedEvent{}, stack.Wrap(ErrStale)
	}
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(id + "." + timestamp + "."))
	mac.Write(body)
	if !signed(r.Header.Get(signatureHeader), mac.Sum(nil)) {
		return core.VerifiedEvent{}, stack.Wrap(ErrUnsigned)
	}

	read, err := m.Channel.Read(m.ID, body)
	if err != nil {
		// The signature proved the provider sent body, so a body the block does not describe
		// is a verified request with nothing to act on (core.Verifier).
		return core.VerifiedEvent{}, nil
	}
	event := core.VerifiedEvent{Challenge: read.Challenge, Signals: read.Signals}
	for _, message := range read.Messages {
		event.Messages = append(event.Messages, message.InboundMessage)
	}
	return event, nil
}

// signed is whether one of the header's v1 signatures is digest.
func signed(header string, digest []byte) bool {
	for _, signature := range strings.Fields(header) {
		written, ok := strings.CutPrefix(signature, signaturePrefix)
		if !ok {
			continue
		}
		sent, err := base64.StdEncoding.DecodeString(written)
		if err == nil && hmac.Equal(sent, digest) {
			return true
		}
	}
	return false
}
