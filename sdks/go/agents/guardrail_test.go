package agents

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"testing"
	"time"
)

// The signing scheme as one fixed vector. The backend that signs these requests pins the
// same three inputs to the same digest in a test of its own; the two cannot import each
// other, so this literal is what holds them to the same bytes.
const (
	vectorSecret    = "super-secret"
	vectorTimestamp = "1789000000"
	vectorBody      = `{"customer_id":"acme","text":"how do I make a pizza"}`
	vectorSignature = "7b994e1f8d9d21f421462dff92f0f9c4a5a025d3a26454eee7135657ad672278"
)

// sign is what the backend does, so a test can produce a request with a timestamp of its
// choosing. The vector above is what says this matches the real signer.
func sign(secret, timestamp string, body []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(timestamp))
	mac.Write([]byte("."))
	mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))
}

func stamp(offset time.Duration) string {
	return strconv.FormatInt(time.Now().Add(offset).Unix(), 10)
}

func TestTheSignerAndTheVerifierAgreeOnTheScheme(t *testing.T) {
	if got := sign(vectorSecret, vectorTimestamp, []byte(vectorBody)); got != vectorSignature {
		t.Fatalf("the scheme this verifies has changed: %s", got)
	}
}

func TestAGuardrailWebhookVerifies(t *testing.T) {
	body := []byte(vectorBody)
	now := stamp(0)

	if !VerifyGuardrailWebhook(vectorSecret, now, sign(vectorSecret, now, body), body) {
		t.Error("a request the guardrail really sent was rejected")
	}
}

func TestAGuardrailWebhookIsRejectedWhenItCouldNotHaveComeFromTheGuardrail(t *testing.T) {
	body := []byte(vectorBody)
	now := stamp(0)
	signature := sign(vectorSecret, now, body)

	tests := []struct {
		name      string
		secret    string
		timestamp string
		signature string
		body      []byte
	}{
		{
			// The whole point of the timestamp being inside the signed string: a captured
			// "allow" replayed tomorrow is not an allow.
			name: "signed ten minutes ago", secret: vectorSecret,
			timestamp: stamp(-10 * time.Minute),
			signature: sign(vectorSecret, stamp(-10*time.Minute), body), body: body,
		},
		{
			name: "signed ten minutes from now", secret: vectorSecret,
			timestamp: stamp(10 * time.Minute),
			signature: sign(vectorSecret, stamp(10*time.Minute), body), body: body,
		},
		{
			name: "a body changed after it was signed", secret: vectorSecret,
			timestamp: now, signature: signature, body: []byte(`{"text":"tampered"}`),
		},
		{
			name: "signed with somebody else's secret", secret: vectorSecret,
			timestamp: now, signature: sign("not-the-secret", now, body), body: body,
		},
		{
			name: "a signature moved onto another timestamp", secret: vectorSecret,
			timestamp: stamp(-time.Minute), signature: signature, body: body,
		},
		{
			name: "no signature at all", secret: vectorSecret,
			timestamp: now, signature: "", body: body,
		},
		{
			name: "no timestamp at all", secret: vectorSecret,
			timestamp: "", signature: signature, body: body,
		},
		{
			name: "a timestamp that is not a time", secret: vectorSecret,
			timestamp: "yesterday", signature: signature, body: body,
		},
		{
			// A deployment with no secret configured cannot tell anyone apart, so it
			// must not accept everyone.
			name: "no secret configured", secret: "",
			timestamp: now, signature: signature, body: body,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if VerifyGuardrailWebhook(test.secret, test.timestamp, test.signature, test.body) {
				t.Error("accepted")
			}
		})
	}
}
