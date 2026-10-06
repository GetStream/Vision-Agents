package channels

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// telnyxURL is Telnyx's API.
const telnyxURL = "https://api.telnyx.com/v2"

// Telnyx signs a delivery with Ed25519 and puts the signature and the time it signed in
// these headers.
const (
	telnyxSignatureHeader = "Telnyx-Signature-Ed25519"
	telnyxTimestampHeader = "Telnyx-Timestamp"
)

// telnyxMaxMedia is how many media URLs Telnyx takes on one message.
const telnyxMaxMedia = 10

// telnyxProvider carries text messages through Telnyx Messaging.
type telnyxProvider struct {
	client  *http.Client
	baseURL string
}

func (p *telnyxProvider) Kind() Kind { return SMS }

// Verify checks Telnyx's Ed25519 signature over the timestamp and the body.
//
// The timestamp is signed with the body, so it cannot be moved: a delivery recorded today
// and replayed next week carries its own age with it, and is refused for it.
func (p *telnyxProvider) Verify(account Account, header http.Header, body []byte, now time.Time) error {
	key, err := base64.StdEncoding.DecodeString(account.Signing)
	if err != nil || len(key) != ed25519.PublicKeySize {
		return fmt.Errorf("%w: this line has no usable Telnyx public key", ErrUnsigned)
	}
	seconds, err := strconv.ParseInt(header.Get(telnyxTimestampHeader), 10, 64)
	if err != nil {
		return fmt.Errorf("%w: the delivery carries no timestamp", ErrUnsigned)
	}
	if !fresh(time.Unix(seconds, 0), now) {
		return fmt.Errorf("%w: the delivery is too old to act on", ErrUnsigned)
	}
	signature, err := base64.StdEncoding.DecodeString(header.Get(telnyxSignatureHeader))
	if err != nil {
		return ErrUnsigned
	}
	signed := append([]byte(strconv.FormatInt(seconds, 10)+"|"), body...)
	if !ed25519.Verify(ed25519.PublicKey(key), signed, signature) {
		return ErrUnsigned
	}
	return nil
}

// telnyxDelivery is the part of a Telnyx webhook a message is read out of.
type telnyxDelivery struct {
	Data struct {
		EventType string `json:"event_type"`
		Payload   struct {
			ID   string `json:"id"`
			Text string `json:"text"`
			From struct {
				PhoneNumber string `json:"phone_number"`
			} `json:"from"`
			To []struct {
				PhoneNumber string `json:"phone_number"`
			} `json:"to"`
		} `json:"payload"`
	} `json:"data"`
}

// Parse reads the message out of a delivery. Telnyx reports on messages this router sent
// down the same webhook, and those are not messages to answer.
func (p *telnyxProvider) Parse(body []byte) ([]Message, error) {
	var delivered telnyxDelivery
	if err := json.Unmarshal(body, &delivered); err != nil {
		return nil, fmt.Errorf("channels: that is not a Telnyx delivery: %w", err)
	}
	if delivered.Data.EventType != "message.received" {
		return nil, nil
	}
	message := delivered.Data.Payload
	if message.Text == "" || message.From.PhoneNumber == "" {
		return nil, nil
	}
	to := ""
	if len(message.To) > 0 {
		to = message.To[0].PhoneNumber
	}
	return []Message{{
		Kind:   SMS,
		ID:     message.ID,
		From:   message.From.PhoneNumber,
		To:     to,
		Thread: message.From.PhoneNumber,
		Text:   message.Text,
	}}, nil
}

// Send writes the reply back. Files go as media URLs, which makes it an MMS, batched because
// Telnyx takes ten a message.
func (p *telnyxProvider) Send(ctx context.Context, account Account, thread string, reply Reply) error {
	text := reply.Text
	if link := reply.Link; link != nil {
		text = strings.TrimSpace(text + "\n\n" + link.Text + ": " + link.URL)
	}
	var urls []string
	for _, file := range reply.Files {
		if file.URL != "" {
			urls = append(urls, file.URL)
		}
	}

	var bodies []map[string]any
	for start := 0; start < len(urls); start += telnyxMaxMedia {
		body := map[string]any{"to": thread, "from": account.E164}
		body["media_urls"] = urls[start:min(start+telnyxMaxMedia, len(urls))]
		bodies = append(bodies, body)
	}
	if text != "" {
		// The text goes first, so an answer reads before whatever it made arrives.
		first := map[string]any{"to": thread, "from": account.E164, "text": text}
		bodies = append([]map[string]any{first}, bodies...)
	}

	for _, body := range bodies {
		if err := post(ctx, p.client, p.host()+"/messages", account.Token, body); err != nil {
			return err
		}
	}
	return nil
}

// ConfigureMessaging points the number's messaging at a URL, so texts to it are delivered
// here. It is how a number bought through /v1/phone/numbers becomes a channel: Telnyx sends
// a number's messages wherever its own record says, and nowhere until it says here.
func (p *telnyxProvider) ConfigureMessaging(ctx context.Context, account Account, hook string) error {
	body := map[string]any{"url": hook, "webhook_url": hook}
	url := fmt.Sprintf("%s/phone_numbers/%s/messaging", p.host(), account.AccountID)
	return patch(ctx, p.client, url, account.Token, body)
}

func (p *telnyxProvider) host() string {
	if p.baseURL != "" {
		return strings.TrimSuffix(p.baseURL, "/")
	}
	return telnyxURL
}
