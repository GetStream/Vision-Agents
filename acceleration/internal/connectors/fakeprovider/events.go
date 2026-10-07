package fakeprovider

import (
	"bytes"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// MCP Events' webhook delivery as the draft describes a server's side of it:
// https://github.com/modelcontextprotocol/experimental-ext-triggers-events,
// docs/design-sketch-proposal.md at 6682596d («Webhook-Based Delivery», «Webhook Security»).
// No production MCP server offers events yet (plugin skill, «Neither Sentry nor Google Calendar
// offers events yet»), so this is what the router's client is tested against.

// EventsTTL is the lifetime the server grants a subscription, the refreshBefore it answers.
// Synthetic: the draft leaves the grant to the server («Recommended grants»: «from a few
// minutes up to about a day»), and a test moves past it with Advance.
const EventsTTL = time.Hour

// The draft's error codes («Error codes»).
const (
	codeNotFound              = -32011
	codeCallbackEndpointError = -32015
)

// callbackTimeout bounds the verification POST to a callback. Synthetic: the draft names «a
// timeout on the order of 5 seconds» as common for a delivery.
const callbackTimeout = 5 * time.Second

// EventSubscription is one subscription the server holds, as a test reads it back.
type EventSubscription struct {
	// ID is the server-derived id the draft returns and sends as X-MCP-Subscription-Id.
	ID        string
	Name      string
	Arguments map[string]any
	URL       string
	// Secret is the whsec_ secret the client supplied, which signs every delivery.
	Secret        string
	RefreshBefore time.Time
	// Subscribes is how many events/subscribe calls made or refreshed it.
	Subscribes int
}

// eventSub is a subscription with the grant it was made under: the draft keys a subscription
// on «(principal, delivery.url, name, arguments)», and the grant is the principal here.
type eventSub struct {
	EventSubscription
	principal *grant
}

type eventParams struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
	Delivery  struct {
		Mode   string `json:"mode"`
		URL    string `json:"url"`
		Secret string `json:"secret"`
	} `json:"delivery"`
}

// GrantEventsFor makes every later events/subscribe grant ttl instead of EventsTTL; a negative
// one answers a refreshBefore already past, as a server whose clock is behind, or with a bug,
// does. Zero is EventsTTL again.
func (s *Server) GrantEventsFor(ttl time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.eventsTTL = ttl
}

// EventSubscriptions are the subscriptions the server holds now, in no order.
func (s *Server) EventSubscriptions() []EventSubscription {
	s.mu.Lock()
	defer s.mu.Unlock()
	held := make([]EventSubscription, 0, len(s.eventSubs))
	for _, sub := range s.eventSubs {
		held = append(held, sub.EventSubscription)
	}
	return held
}

// Emit delivers one occurrence of event to every subscription to it, signed with each one's
// own secret, and returns the status each callback answered, by subscription id. The event
// id is fresh.
func (s *Server) Emit(event string, data map[string]any) map[string]int {
	answered := map[string]int{}
	eventID := "evt_" + synthetic("event")
	for _, sub := range s.EventSubscriptions() {
		if sub.Name == event {
			answered[sub.ID] = s.DeliverEvent(sub.URL, sub.Secret, sub.ID, eventID, event, data)
		}
	}
	return answered
}

// DeliverEvent posts one event occurrence to url signed with secret, as the server delivers
// it («Webhook Event Delivery»), and returns the status the callback answered. A test signs
// with another subscription's secret to forge one.
func (s *Server) DeliverEvent(url, secret, subscriptionID, eventID, event string, data map[string]any) int {
	s.t.Helper()
	body, err := json.Marshal(map[string]any{"eventId": eventID, "name": event,
		"timestamp": s.now().UTC().Format(time.RFC3339), "data": data, "cursor": nil})
	if err != nil {
		s.t.Fatalf("fakeprovider: %v", err)
	}
	response, err := postSigned(url, secret, subscriptionID, eventID, s.now(), body)
	if err != nil {
		s.t.Fatalf("fakeprovider: deliver %s: %v", event, err)
	}
	defer func() { _ = response.Body.Close() }()
	return response.StatusCode
}

// subscribeEvent is events/subscribe: the secret checked, the callback verified with a signed
// challenge before anything else, then the subscription made or refreshed under the token's
// grant. The caller holds s.mu.
func (s *Server) subscribeEvent(w http.ResponseWriter, id json.RawMessage, token *accessToken, raw []byte) (map[string]any, bool) {
	var params eventParams
	if err := json.Unmarshal(raw, &params); err != nil || params.Name == "" || params.Delivery.URL == "" {
		writeRPCError(w, http.StatusOK, id, codeInvalidParams, "name and delivery.url are required")
		return nil, false
	}
	// «Servers MUST reject a delivery.secret that is not whsec_ followed by base64 decoding to
	// 24–64 bytes».
	key, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(params.Delivery.Secret, "whsec_"))
	if params.Delivery.Mode != "webhook" || !strings.HasPrefix(params.Delivery.Secret, "whsec_") || err != nil || len(key) < 24 || len(key) > 64 {
		writeRPCError(w, http.StatusOK, id, codeInvalidParams, "delivery must be a webhook with a whsec_ secret of 24 to 64 bytes")
		return nil, false
	}
	slot := eventSlot(token.grant, params)
	sub := s.eventSubs[slot]
	if sub == nil {
		// «a server MUST NOT begin delivering to a callback URL until the endpoint's intent to
		// receive deliveries is confirmed», here by the handshake, (a) of the four.
		if !s.verifyCallback(params.Delivery.URL, params.Delivery.Secret) {
			writeRPCErrorData(w, http.StatusOK, id, codeCallbackEndpointError, "the callback did not echo the challenge",
				map[string]any{"reason": "challenge_failed"})
			return nil, false
		}
		digest := sha256.Sum256([]byte(slot))
		sub = &eventSub{principal: token.grant, EventSubscription: EventSubscription{
			ID: "sub_" + hex.EncodeToString(digest[:8]), Name: params.Name, Arguments: params.Arguments, URL: params.Delivery.URL,
		}}
		if s.eventSubs == nil {
			s.eventSubs = map[string]*eventSub{}
		}
		s.eventSubs[slot] = sub
	}
	// «delivery.secret | Replaced»; «TTL | Re-granted».
	sub.Secret = params.Delivery.Secret
	ttl := EventsTTL
	if s.eventsTTL != 0 {
		ttl = s.eventsTTL
	}
	sub.RefreshBefore = s.now().Add(ttl).UTC().Truncate(time.Second)
	sub.Subscribes++
	return map[string]any{"id": sub.ID, "refreshBefore": sub.RefreshBefore.Format(time.RFC3339), "cursor": nil, "truncated": false}, true
}

// unsubscribeEvent is events/unsubscribe, by the same key; none is NotFound. The caller holds
// s.mu.
func (s *Server) unsubscribeEvent(w http.ResponseWriter, id json.RawMessage, token *accessToken, raw []byte) (map[string]any, bool) {
	var params eventParams
	if err := json.Unmarshal(raw, &params); err != nil {
		writeRPCError(w, http.StatusOK, id, codeInvalidParams, "invalid params")
		return nil, false
	}
	slot := eventSlot(token.grant, params)
	if s.eventSubs[slot] == nil {
		writeRPCErrorData(w, http.StatusOK, id, codeNotFound, "no such subscription", map[string]any{"kind": "subscription"})
		return nil, false
	}
	delete(s.eventSubs, slot)
	return map[string]any{}, true
}

// verifyCallback posts the verification envelope with a fresh challenge, signed like a
// delivery, and reports whether the callback echoed it in a 2xx, compared in constant time.
func (s *Server) verifyCallback(url, secret string) bool {
	challenge := synthetic("challenge")
	body, _ := json.Marshal(map[string]string{"type": "verification", "challenge": challenge})
	response, err := postSigned(url, secret, "", "msg_verification_"+synthetic("v")[2:], s.now(), body)
	if err != nil {
		return false
	}
	defer func() { _ = response.Body.Close() }()
	var echoed struct {
		Challenge string `json:"challenge"`
	}
	raw, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
	if response.StatusCode/100 != 2 || json.Unmarshal(raw, &echoed) != nil {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(echoed.Challenge), []byte(challenge)) == 1
}

// eventSlot is the draft's subscription key: the principal, the URL, the event and its
// arguments by canonical JSON (encoding/json sorts map keys).
func eventSlot(g *grant, params eventParams) string {
	arguments, _ := json.Marshal(params.Arguments)
	return fmt.Sprintf("%p", g) + "\x00" + params.Delivery.URL + "\x00" + params.Name + "\x00" + string(arguments)
}

// postSigned posts body with the Standard Webhooks headers and the draft's
// X-MCP-Subscription-Id («Signature scheme (Standard Webhooks profile)»).
func postSigned(url, secret, subscriptionID, webhookID string, at time.Time, body []byte) (*http.Response, error) {
	key, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(secret, "whsec_"))
	if err != nil {
		return nil, err
	}
	stamp := strconv.FormatInt(at.Unix(), 10)
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(webhookID + "." + stamp + "."))
	mac.Write(body)
	request, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("webhook-id", webhookID)
	request.Header.Set("webhook-timestamp", stamp)
	request.Header.Set("webhook-signature", "v1,"+base64.StdEncoding.EncodeToString(mac.Sum(nil)))
	if subscriptionID != "" {
		request.Header.Set("X-MCP-Subscription-Id", subscriptionID)
	}
	return (&http.Client{Timeout: callbackTimeout}).Do(request)
}

// NewWebhookSecret is a whsec_ secret of 32 random bytes, for a test that needs one the server
// never saw.
func NewWebhookSecret() string {
	key := make([]byte, 32)
	_, _ = rand.Read(key)
	return "whsec_" + base64.StdEncoding.EncodeToString(key)
}
