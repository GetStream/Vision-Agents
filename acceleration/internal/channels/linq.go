package channels

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// linqURL is Linq's partner API.
const linqURL = "https://api.linqapp.com/api/partner/v3"

// Standard Webhooks puts the delivery's id, the time it was signed and the signatures in
// these headers.
const (
	webhookIDHeader        = "Webhook-Id"
	webhookTimestampHeader = "Webhook-Timestamp"
	webhookSignatureHeader = "Webhook-Signature"
)

// linqProvider carries iMessage through Linq, which falls back to RCS or SMS when Apple
// cannot deliver.
type linqProvider struct {
	client  *http.Client
	baseURL string
}

func (p *linqProvider) Kind() Kind { return IMessage }

// Verify checks the Standard Webhooks signature: an HMAC over the delivery's id, the time it
// was signed and the body, any one of which may match since a key being rotated signs twice.
func (p *linqProvider) Verify(account Account, header http.Header, body []byte, now time.Time) error {
	key, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(account.Signing, "whsec_"))
	if err != nil || len(key) == 0 {
		return fmt.Errorf("%w: this line has no usable webhook secret", ErrUnsigned)
	}
	delivery := header.Get(webhookIDHeader)
	stamp := header.Get(webhookTimestampHeader)
	seconds, err := strconv.ParseInt(stamp, 10, 64)
	if delivery == "" || err != nil {
		return fmt.Errorf("%w: the delivery carries no id and timestamp", ErrUnsigned)
	}
	if !fresh(time.Unix(seconds, 0), now) {
		return fmt.Errorf("%w: the delivery is too old to act on", ErrUnsigned)
	}

	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(delivery + "." + stamp + "."))
	mac.Write(body)
	expected := base64.StdEncoding.EncodeToString(mac.Sum(nil))
	for _, signature := range strings.Fields(header.Get(webhookSignatureHeader)) {
		candidate, ok := strings.CutPrefix(signature, "v1,")
		if ok && hmac.Equal([]byte(candidate), []byte(expected)) {
			return nil
		}
	}
	return ErrUnsigned
}

// linqDelivery is the part of a Linq webhook a message is read out of. It reads the
// 2026-02-03 version, which a subscription asks for with ?version=2026-02-03 on its URL.
type linqDelivery struct {
	EventType string `json:"event_type"`
	Data      struct {
		ID        string `json:"id"`
		Direction string `json:"direction"`
		Parts     []struct {
			Type  string `json:"type"`
			Value string `json:"value"`
		} `json:"parts"`
		SenderHandle struct {
			Handle string `json:"handle"`
			Name   string `json:"name"`
		} `json:"sender_handle"`
		Chat struct {
			ID          string `json:"id"`
			OwnerHandle struct {
				Handle string `json:"handle"`
			} `json:"owner_handle"`
		} `json:"chat"`
	} `json:"data"`
}

// Parse reads the message out of a delivery. Linq reports on what this router sent down the
// same webhook, which is what the direction tells apart.
func (p *linqProvider) Parse(body []byte) ([]Message, error) {
	var delivered linqDelivery
	if err := json.Unmarshal(body, &delivered); err != nil {
		return nil, fmt.Errorf("channels: that is not a Linq delivery: %w", err)
	}
	data := delivered.Data
	if delivered.EventType != "message.received" || data.Direction != "inbound" {
		return nil, nil
	}
	var written []string
	for _, part := range data.Parts {
		if (part.Type == "text" || part.Type == "link") && part.Value != "" {
			written = append(written, part.Value)
		}
	}
	text := strings.Join(written, "\n")
	if text == "" || data.Chat.ID == "" {
		return nil, nil
	}
	return []Message{{
		Kind: IMessage,
		ID:   data.ID,
		From: data.SenderHandle.Handle,
		To:   data.Chat.OwnerHandle.Handle,
		// A reply is posted to the chat rather than to a person, which is also what makes a
		// group chat answerable.
		Thread: data.Chat.ID,
		Name:   data.SenderHandle.Name,
		Text:   text,
	}}, nil
}

// Send writes the reply back as one message of several parts: the text, then each file.
// iMessage previews a URL, so a login goes in the text.
func (p *linqProvider) Send(ctx context.Context, account Account, thread string, reply Reply) error {
	text := reply.Text
	if link := reply.Link; link != nil {
		text = strings.TrimSpace(text + "\n\n" + link.Text + ": " + link.URL)
	}
	var parts []map[string]any
	if text != "" {
		parts = append(parts, map[string]any{"type": "text", "value": text})
	}
	for _, file := range reply.Files {
		if file.URL != "" {
			parts = append(parts, map[string]any{"type": "media", "url": file.URL})
		}
	}
	if len(parts) == 0 {
		return nil
	}
	body := map[string]any{"message": map[string]any{"parts": parts}}
	return post(ctx, p.client, fmt.Sprintf("%s/chats/%s/messages", p.host(), url.PathEscape(thread)),
		account.Token, body)
}

func (p *linqProvider) host() string {
	if p.baseURL != "" {
		return strings.TrimSuffix(p.baseURL, "/")
	}
	return linqURL
}
