package plugins

import (
	"context"
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

// The client half of MCP Events: subscribing on a plugin's server with a callback the router
// serves, and checking that a delivery to it was signed by that server.
// https://developers.openai.com/plugins/build/mcp-events, opened 2026-10-03, a draft.

// EventsVersion is the MCP protocol version events need.
const EventsVersion = "2026-07-28"

// EventsPath is where deliveries arrive, followed by the subscription's token.
const EventsPath = "/v1/agents/plugins/events/"

// MaxEventBytes is the most a server may send in one delivery.
const MaxEventBytes = 256 * 1024

// webhookTolerance is how far a delivery's signing time may be from now, the Standard
// Webhooks libraries' default.
const webhookTolerance = 5 * time.Minute

const secretPrefix = "whsec_"

// ErrNoEvents is a server that offers no events.
var ErrNoEvents = errors.New("plugins: the server offers no events")

// ErrBadSignature is a delivery not signed with the subscription's secret.
var ErrBadSignature = errors.New("plugins: the delivery is not signed with the subscription's secret")

// Delivery is where a server sends a subscription's events.
type Delivery struct {
	URL    string
	Secret string
}

// Subscribed is what the server granted.
type Subscribed struct {
	ID string
	// RefreshBefore is when the server stops delivering unless subscribed to again. Nil
	// is a subscription that does not expire.
	RefreshBefore *time.Time
}

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

type discoverResult struct {
	Capabilities struct {
		Events json.RawMessage `json:"events"`
	} `json:"capabilities"`
}

type subscribeResult struct {
	ID            string     `json:"id"`
	RefreshBefore *time.Time `json:"refreshBefore"`
}

// Subscribe creates or refreshes a subscription to one event. The server derives the
// subscription from the login, the callback, the event and its arguments, so subscribing
// again with the same four refreshes it.
func Subscribe(ctx context.Context, conn Connection, event string, arguments map[string]any, delivery Delivery, transport *http.Client) (Subscribed, error) {
	opened, err := discover(ctx, conn, transport)
	if err != nil {
		return Subscribed{}, err
	}
	raw, err := opened.call(ctx, "events/subscribe", map[string]any{
		"name":      event,
		"arguments": argumentsOf(arguments),
		"delivery": map[string]string{
			"mode":   "webhook",
			"url":    delivery.URL,
			"secret": delivery.Secret,
		},
		"cursor": nil,
	})
	if err != nil {
		return Subscribed{}, err
	}
	var granted subscribeResult
	if err := json.Unmarshal(raw, &granted); err != nil {
		return Subscribed{}, fmt.Errorf("plugins: events/subscribe: %w", err)
	}
	return Subscribed{ID: granted.ID, RefreshBefore: granted.RefreshBefore}, nil
}

// Unsubscribe stops a subscription, named the way it was made. Stopping one the server no
// longer has is not an error.
func Unsubscribe(ctx context.Context, conn Connection, event string, arguments map[string]any, url string, transport *http.Client) error {
	opened, err := discover(ctx, conn, transport)
	if err != nil {
		return err
	}
	_, err = opened.call(ctx, "events/unsubscribe", map[string]any{
		"name":      event,
		"arguments": argumentsOf(arguments),
		"delivery":  map[string]string{"mode": "webhook", "url": url},
	})
	return err
}

// discover opens an MCP 2.0 client, which needs no initialize, and checks the server
// offers events.
func discover(ctx context.Context, conn Connection, transport *http.Client) (*client, error) {
	if transport == nil {
		transport = http.DefaultClient
	}
	opened := &client{
		pluginID: conn.PluginID,
		endpoint: conn.Endpoint,
		token:    conn.AccessToken,
		http:     transport,
		nextID:   1,
		version:  EventsVersion,
	}
	raw, err := opened.call(ctx, "server/discover", map[string]any{})
	if err != nil {
		return nil, err
	}
	var discovered discoverResult
	if err := json.Unmarshal(raw, &discovered); err != nil {
		return nil, fmt.Errorf("plugins: server/discover: %w", err)
	}
	if len(discovered.Capabilities.Events) == 0 || string(discovered.Capabilities.Events) == "null" {
		return nil, fmt.Errorf("%w: %s", ErrNoEvents, conn.PluginID)
	}
	return opened, nil
}

func argumentsOf(arguments map[string]any) map[string]any {
	if arguments == nil {
		return map[string]any{}
	}
	return arguments
}

// EventsURL is the callback a subscription with this token is delivered to.
func (a *Auth) EventsURL(token string) string {
	return strings.TrimRight(a.PublicURL, "/") + EventsPath + token
}

// NewWebhookSecret is a fresh signing secret, in the whsec_ form servers require.
func NewWebhookSecret() (string, error) {
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		return "", err
	}
	return secretPrefix + base64.StdEncoding.EncodeToString(key), nil
}

// SignWebhook is the Standard Webhooks signature of one delivery, as the webhook-signature
// header carries it.
func SignWebhook(secret, id string, at time.Time, body []byte) (string, error) {
	key, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(secret, secretPrefix))
	if err != nil {
		return "", fmt.Errorf("plugins: the secret is not base64: %w", err)
	}
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(id + "." + strconv.FormatInt(at.Unix(), 10) + "."))
	mac.Write(body)
	return "v1," + base64.StdEncoding.EncodeToString(mac.Sum(nil)), nil
}

// VerifyWebhook checks a delivery's Standard Webhooks headers against the subscription's
// secret. Any one of the space-separated signatures matching is enough, which is how a
// server rotating the secret signs with both.
func VerifyWebhook(secret string, header http.Header, body []byte, now time.Time) error {
	id := header.Get("webhook-id")
	stamp := header.Get("webhook-timestamp")
	if id == "" || stamp == "" {
		return ErrBadSignature
	}
	seconds, err := strconv.ParseInt(stamp, 10, 64)
	if err != nil {
		return ErrBadSignature
	}
	if math.Abs(now.Sub(time.Unix(seconds, 0)).Seconds()) > webhookTolerance.Seconds() {
		return fmt.Errorf("%w: signed too long ago", ErrBadSignature)
	}
	expected, err := SignWebhook(secret, id, time.Unix(seconds, 0), body)
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
