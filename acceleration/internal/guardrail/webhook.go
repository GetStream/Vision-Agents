package guardrail

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// TimestampHeader and SignatureHeader are how a webhook proves it came from here.
const (
	TimestampHeader = "X-Timestamp"
	SignatureHeader = "X-Signature"
)

// errorBodyLimit caps how much of an unusable reply is read into an error message.
const errorBodyLimit = 2048

// webhook screens a turn by asking the customer's own server.
type webhook struct {
	policy Policy
	// owner names the call the turn belongs to, so the server being asked knows whose
	// agent it is deciding for.
	owner  routing.Owner
	secret string
	client *http.Client
	logger *slog.Logger
}

// Ask is the body posted to the customer's endpoint.
type Ask struct {
	CustomerID string `json:"customer_id"`
	AgentID    string `json:"agent_id"`
	CallID     string `json:"call_id"`
	TurnID     string `json:"turn_id"`
	// Role is who said it. Only "user" today, and named so that screening what the agent
	// replied does not need a second shape.
	Role string `json:"role"`
	Text string `json:"text"`
	// Policy is the prose from guardrail.md, sent so a server can answer without holding
	// a copy of a file it does not own.
	Policy string `json:"policy"`
}

// Answer is what the customer's server replies.
type Answer struct {
	Allow bool `json:"allow"`
	// Reason is for our log and the Blocked event, not for the caller.
	Reason string `json:"reason"`
	// Message is what the agent should say instead. Empty falls back to the policy's own
	// refusal.
	Message string `json:"message"`
}

func newWebhook(policy Policy, deps Deps) (Guardrail, error) {
	if deps.Secret == "" {
		// Unsigned, the customer's server cannot tell our request from anyone who found
		// the URL, and what it decides on that request gates an agent's replies.
		return nil, errors.New(
			"guardrail: a webhook guardrail needs the app secret to sign with, and this deployment has none")
	}

	client := deps.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: checkTimeout}
	}
	return &webhook{
		policy: policy,
		owner:  deps.Owner,
		secret: deps.Secret,
		client: client,
		logger: deps.Logger,
	}, nil
}

func (w *webhook) Policy() Policy { return w.policy }

func (w *webhook) Close() error { return nil }

// Check posts the turn and reads the verdict off the reply.
func (w *webhook) Check(ctx context.Context, turnID, text string) (Verdict, error) {
	ctx, cancel := context.WithTimeout(ctx, checkTimeout)
	defer cancel()

	body, err := json.Marshal(Ask{
		CustomerID: w.owner.CustomerID,
		AgentID:    w.owner.AgentID,
		CallID:     w.owner.CallID,
		TurnID:     turnID,
		Role:       "user",
		Text:       text,
		Policy:     w.policy.Text,
	})
	if err != nil {
		return Verdict{}, fmt.Errorf("guardrail: encode the turn: %w", err)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, w.policy.URL, bytes.NewReader(body))
	if err != nil {
		return Verdict{}, fmt.Errorf("guardrail: build the request: %w", err)
	}
	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(TimestampHeader, timestamp)
	request.Header.Set(SignatureHeader, Sign(w.secret, timestamp, body))

	response, err := w.client.Do(request)
	if err != nil {
		return Verdict{}, fmt.Errorf("guardrail: ask %s: %w", w.policy.URL, err)
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		unusable, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return Verdict{}, fmt.Errorf("guardrail: %s returned %d: %s",
			w.policy.URL, response.StatusCode, bytes.TrimSpace(unusable))
	}

	var answer Answer
	if err := json.NewDecoder(response.Body).Decode(&answer); err != nil {
		return Verdict{}, fmt.Errorf("guardrail: read the verdict from %s: %w", w.policy.URL, err)
	}

	if answer.Allow {
		return allowed(0), nil
	}
	reason := answer.Reason
	if reason == "" {
		reason = "the customer's own server refused this turn"
	}
	return Verdict{Reason: reason, Refusal: answer.Message}, nil
}

// Sign is the signature a receiver has to reproduce: the hex HMAC-SHA256, keyed by the app
// secret, of the timestamp and the body joined by a dot.
//
// The timestamp is inside the signed string rather than beside it, which is the whole point
// of it being there: a signature over the body alone stays valid forever, so anyone who
// captured one request could replay it whenever they liked.
func Sign(secret, timestamp string, body []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(timestamp))
	mac.Write([]byte("."))
	mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))
}
