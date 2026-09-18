package agents

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"time"
)

// GuardrailTimestampHeader and GuardrailSignatureHeader are how a guardrail webhook proves
// where it came from.
const (
	GuardrailTimestampHeader = "X-Timestamp"
	GuardrailSignatureHeader = "X-Signature"
)

// GuardrailMaxSkew is how old a request may be. A signature is only worth having if it
// stops being valid, and this is when.
const GuardrailMaxSkew = 5 * time.Minute

// GuardrailAsk is what a webhook guardrail posts to your server. Reply with a
// GuardrailAnswer.
type GuardrailAsk struct {
	CustomerID string `json:"customer_id"`
	AgentID    string `json:"agent_id"`
	CallID     string `json:"call_id"`
	TurnID     string `json:"turn_id"`
	// Role is who said it, which is "user" today.
	Role string `json:"role"`
	Text string `json:"text"`
	// Policy is the prose from the agent's guardrail.md, so your server can decide without
	// holding a copy of a file it does not own.
	Policy string `json:"policy"`
}

// GuardrailAnswer is what your server replies.
type GuardrailAnswer struct {
	Allow bool `json:"allow"`
	// Reason is recorded against the turn. It is not said to the caller.
	Reason string `json:"reason"`
	// Message is what the agent should say instead. Empty means the refusal from the
	// agent's own guardrail.md.
	Message string `json:"message"`
}

// VerifyGuardrailWebhook reports whether a request really came from the guardrail, using
// the app secret.
//
// Take the timestamp and the signature from the headers above and pass the raw body, before
// any JSON decoding: the signature is over the exact bytes that arrived, so a body that has
// been unmarshalled and marshalled again will not match. Verify before acting, not after.
//
//	if !agents.VerifyGuardrailWebhook(secret,
//	    r.Header.Get(agents.GuardrailTimestampHeader),
//	    r.Header.Get(agents.GuardrailSignatureHeader), body) {
//	    http.Error(w, "not from the guardrail", http.StatusUnauthorized)
//	    return
//	}
func VerifyGuardrailWebhook(secret, timestamp, signature string, body []byte) bool {
	if secret == "" || timestamp == "" || signature == "" {
		return false
	}

	// The age is checked before the signature because a valid signature on an old request
	// is exactly what a replay is: someone who captured one request and sent it again.
	// The timestamp is inside the signed string, so it cannot be moved forward without
	// the secret.
	seconds, err := strconv.ParseInt(timestamp, 10, 64)
	if err != nil {
		return false
	}
	age := time.Since(time.Unix(seconds, 0))
	if age < -GuardrailMaxSkew || age > GuardrailMaxSkew {
		return false
	}

	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(timestamp))
	mac.Write([]byte("."))
	mac.Write(body)
	expected := hex.EncodeToString(mac.Sum(nil))

	return hmac.Equal([]byte(expected), []byte(signature))
}
