package channels

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// graphURL is Meta's Cloud API.
const graphURL = "https://graph.facebook.com/v23.0"

// signatureHeader is where Meta puts the HMAC of the body.
const signatureHeader = "X-Hub-Signature-256"

// whatsAppProvider carries WhatsApp through Meta's Cloud API.
type whatsAppProvider struct {
	client *http.Client
	// baseURL is Meta's host, which a test replaces.
	baseURL string
}

func (p *whatsAppProvider) Kind() Kind { return WhatsApp }

// Verify checks Meta's HMAC of the body under the app secret.
//
// There is no timestamp to check: Meta signs the body alone, so a replay is caught by the
// message having been answered already rather than by its age.
func (p *whatsAppProvider) Verify(account Account, header http.Header, body []byte, _ time.Time) error {
	if account.Signing == "" {
		return fmt.Errorf("%w: this line has no app secret to check it against", ErrUnsigned)
	}
	mac := hmac.New(sha256.New, []byte(account.Signing))
	mac.Write(body)
	expected := "sha256=" + hex.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(header.Get(signatureHeader)), []byte(expected)) {
		return ErrUnsigned
	}
	return nil
}

// whatsAppDelivery is the part of a Cloud API webhook body a message is read out of.
type whatsAppDelivery struct {
	Entry []struct {
		Changes []struct {
			Value struct {
				Metadata struct {
					PhoneNumberID      string `json:"phone_number_id"`
					DisplayPhoneNumber string `json:"display_phone_number"`
				} `json:"metadata"`
				Contacts []struct {
					WaID    string `json:"wa_id"`
					Profile struct {
						Name string `json:"name"`
					} `json:"profile"`
				} `json:"contacts"`
				Messages []struct {
					ID   string `json:"id"`
					From string `json:"from"`
					Type string `json:"type"`
					Text struct {
						Body string `json:"body"`
					} `json:"text"`
					Button struct {
						Text string `json:"text"`
					} `json:"button"`
					Interactive struct {
						ButtonReply struct {
							Title string `json:"title"`
						} `json:"button_reply"`
						ListReply struct {
							Title string `json:"title"`
						} `json:"list_reply"`
					} `json:"interactive"`
				} `json:"messages"`
			} `json:"value"`
		} `json:"changes"`
	} `json:"entry"`
}

// Parse reads the messages people wrote out of a delivery.
//
// Only what somebody typed is a message here: media, reactions and delivery reports are
// left out, because an agent that is handed a photo it cannot fetch would answer as though
// it had seen one.
func (p *whatsAppProvider) Parse(body []byte) ([]Message, error) {
	var delivered whatsAppDelivery
	if err := json.Unmarshal(body, &delivered); err != nil {
		return nil, fmt.Errorf("channels: that is not a WhatsApp delivery: %w", err)
	}
	var messages []Message
	for _, entry := range delivered.Entry {
		for _, change := range entry.Changes {
			value := change.Value
			names := map[string]string{}
			for _, contact := range value.Contacts {
				names[contact.WaID] = contact.Profile.Name
			}
			for _, item := range value.Messages {
				text := ""
				switch item.Type {
				case "text":
					text = item.Text.Body
				case "button":
					text = item.Button.Text
				case "interactive":
					text = item.Interactive.ButtonReply.Title
					if text == "" {
						text = item.Interactive.ListReply.Title
					}
				}
				if text == "" || item.From == "" {
					continue
				}
				messages = append(messages, Message{
					Kind:   WhatsApp,
					ID:     item.ID,
					From:   e164(item.From),
					To:     e164(value.Metadata.DisplayPhoneNumber),
					Thread: item.From,
					Name:   names[item.From],
					Text:   text,
				})
			}
		}
	}
	return messages, nil
}

// Send writes the reply back, one request per part: the text, then each file, then the
// button for a login. Meta takes one thing per message.
func (p *whatsAppProvider) Send(ctx context.Context, account Account, thread string, reply Reply) error {
	var bodies []map[string]any
	if reply.Text != "" {
		bodies = append(bodies, map[string]any{
			"type": "text",
			"text": map[string]any{"body": reply.Text},
		})
	}
	for _, file := range reply.Files {
		kind := whatsAppMediaType(file.MimeType)
		media := map[string]any{"link": file.URL}
		if kind == "document" && file.Name != "" {
			media["filename"] = file.Name
		}
		bodies = append(bodies, map[string]any{"type": kind, kind: media})
	}
	// The WhatsApp form of a plugin_authorization attachment: a button that opens the login
	// rather than a URL somebody has to copy out of a message.
	if link := reply.Link; link != nil {
		bodies = append(bodies, map[string]any{
			"type": "interactive",
			"interactive": map[string]any{
				"type": "cta_url",
				"body": map[string]any{"text": link.Text},
				"action": map[string]any{
					"name":       "cta_url",
					"parameters": map[string]any{"display_text": "Connect", "url": link.URL},
				},
			},
		})
	}

	for _, body := range bodies {
		body["messaging_product"] = "whatsapp"
		body["recipient_type"] = "individual"
		body["to"] = thread
		url := fmt.Sprintf("%s/%s/messages", p.host(), account.AccountID)
		if err := post(ctx, p.client, url, account.Token, body); err != nil {
			return err
		}
	}
	return nil
}

func (p *whatsAppProvider) host() string {
	if p.baseURL != "" {
		return strings.TrimSuffix(p.baseURL, "/")
	}
	return graphURL
}

// whatsAppMediaType is which of Meta's four media messages carries a file.
func whatsAppMediaType(mime string) string {
	switch {
	case strings.HasPrefix(mime, "image/"):
		return "image"
	case strings.HasPrefix(mime, "video/"):
		return "video"
	case strings.HasPrefix(mime, "audio/"):
		return "audio"
	default:
		return "document"
	}
}

// e164 puts a number in the shape a config names and an identity is stored under. Meta sends
// digits with no plus, and a config is written the way somebody would say the number.
func e164(number string) string {
	digits := strings.Map(func(r rune) rune {
		if r >= '0' && r <= '9' {
			return r
		}
		return -1
	}, number)
	if digits == "" {
		return ""
	}
	return "+" + digits
}
