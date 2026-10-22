// Package webhook signs and checks the router's Standard Webhooks deliveries
// (https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md),
// and holds the body an MCP Events server delivers to one.
package webhook

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// MaxEventBytes is the most a server may send in one delivery.
const MaxEventBytes = 256 * 1024

// tolerance is how far a delivery's signing time may be from now, the Standard Webhooks
// libraries' default.
const tolerance = 5 * time.Minute

const secretPrefix = "whsec_"

// ErrBadSignature is a delivery not signed with the subscription's secret.
var ErrBadSignature = errors.New("webhook: the delivery is not signed with the subscription's secret")

// Event is one delivery to a callback. Type is set only on a control notification, such as
// the verification a server sends before any event.
type Event struct {
	Type      string          `json:"type,omitempty"`
	Challenge string          `json:"challenge,omitempty"`
	EventID   string          `json:"eventId,omitempty"`
	Name      string          `json:"name,omitempty"`
	Timestamp string          `json:"timestamp,omitempty"`
	Data      json.RawMessage `json:"data,omitempty"`
}

// NewSecret is a fresh signing secret, in the whsec_ form servers require.
func NewSecret() (string, error) {
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		return "", err
	}
	return secretPrefix + base64.StdEncoding.EncodeToString(key), nil
}

// Sign is the Standard Webhooks signature of one delivery, as the webhook-signature header
// carries it.
func Sign(secret, id string, at time.Time, body []byte) (string, error) {
	key, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(secret, secretPrefix))
	if err != nil {
		return "", fmt.Errorf("webhook: the secret is not base64: %w", err)
	}
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(id + "." + strconv.FormatInt(at.Unix(), 10) + "."))
	mac.Write(body)
	return "v1," + base64.StdEncoding.EncodeToString(mac.Sum(nil)), nil
}

// Verify checks a delivery's Standard Webhooks headers against the subscription's secret.
// Any one of the space-separated signatures matching is enough, which is how a server
// rotating the secret signs with both.
func Verify(secret string, header http.Header, body []byte, now time.Time) error {
	id := header.Get("webhook-id")
	stamp := header.Get("webhook-timestamp")
	if id == "" {
		return ErrBadSignature
	}
	seconds, err := strconv.ParseInt(stamp, 10, 64)
	if err != nil {
		return ErrBadSignature
	}
	if math.Abs(now.Sub(time.Unix(seconds, 0)).Seconds()) > tolerance.Seconds() {
		return fmt.Errorf("%w: signed too long ago", ErrBadSignature)
	}
	expected, err := Sign(secret, id, time.Unix(seconds, 0), body)
	if err != nil {
		return err
	}
	for _, signature := range strings.Fields(header.Get("webhook-signature")) {
		if hmac.Equal([]byte(signature), []byte(expected)) {
			return nil
		}
	}
	return ErrBadSignature
}
