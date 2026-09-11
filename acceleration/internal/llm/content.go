package llm

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"net/url"
	"strings"
)

const (
	// ModalityImage is an input kind a model may accept besides text.
	ModalityImage = "image"

	// OmittedImage replaces an image a model cannot see, so a conversation that fails over
	// to a text-only model still has something to say about the turn.
	OmittedImage = "[image omitted: model does not accept images]"
)

// TextParts is a tool result or a turn that is only words.
func TextParts(text string) []ContentPart {
	if text == "" {
		return nil
	}
	return []ContentPart{{Text: text}}
}

// TextOf concatenates the text of parts, which is what an event that cannot carry images
// reports.
func TextOf(parts []ContentPart) string {
	var b strings.Builder
	for _, part := range parts {
		if part.Text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(part.Text)
	}
	return b.String()
}

// HasImage reports whether any message carries an image part.
func HasImage(messages []Message) bool {
	for _, message := range messages {
		if message.HasImage() {
			return true
		}
	}
	return false
}

// HasImage reports whether this turn carries an image part.
func (m Message) HasImage() bool {
	for _, part := range m.Parts {
		if part.Image != nil {
			return true
		}
	}
	return false
}

// HasImage reports whether the request's input carries an image part.
func (p ResponseParams) HasImage() bool {
	return HasImage(p.Input)
}

// DataURI is the image as a data URI, which is the wire form sockets and chat-completions
// providers speak.
func (p ImagePart) DataURI() string {
	if p.URL != "" {
		return p.URL
	}
	mime := p.MIME
	if mime == "" {
		mime = "image/jpeg"
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(p.Data)
}

// ParseDataURI reads a data URI into an ImagePart. URL sources are refused: frames are
// local bytes, and a URL the upstream cannot fetch would be a 400 halfway through a call.
func ParseDataURI(value string) (ImagePart, error) {
	if !strings.HasPrefix(value, "data:") {
		return ImagePart{}, fmt.Errorf("llm: image_url must be a data URI")
	}
	rest := strings.TrimPrefix(value, "data:")
	mime, encoded, found := strings.Cut(rest, ";base64,")
	if !found {
		return ImagePart{}, fmt.Errorf("llm: image_url must be a base64 data URI")
	}
	if mime == "" {
		return ImagePart{}, fmt.Errorf("llm: image_url is missing a mime type")
	}
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return ImagePart{}, fmt.Errorf("llm: image_url is not valid base64: %w", err)
	}
	if len(raw) == 0 {
		return ImagePart{}, fmt.Errorf("llm: image_url carries no bytes")
	}
	return ImagePart{MIME: mime, Data: raw}, nil
}

// OmitImages replaces every image with OmittedImage, which is what a text-only model is
// given instead of bytes it cannot see.
func OmitImages(messages []Message) []Message {
	if !HasImage(messages) {
		return messages
	}
	out := append([]Message(nil), messages...)
	for i := range out {
		if out[i].HasImage() {
			out[i] = omitMessageImages(out[i], OmittedImage)
		}
	}
	return out
}

func omitMessageImages(message Message, placeholder string) Message {
	parts := make([]ContentPart, 0, len(message.Parts))
	replaced := false
	for _, part := range message.Parts {
		if part.Image == nil {
			parts = append(parts, part)
			continue
		}
		if !replaced {
			parts = append(parts, ContentPart{Text: placeholder})
			replaced = true
		}
	}
	message.Parts = parts
	return message
}

// SameMessage reports whether two turns are the same, including image bytes. Prompt-cache
// resume compares this way: an image that changed is a different conversation.
func SameMessage(a, b Message) bool {
	if a.Role != b.Role || a.Content != b.Content || a.ToolCallID != b.ToolCallID ||
		len(a.ToolCalls) != len(b.ToolCalls) || len(a.Parts) != len(b.Parts) {
		return false
	}
	for i, call := range a.ToolCalls {
		if call != b.ToolCalls[i] {
			return false
		}
	}
	for i, part := range a.Parts {
		if !samePart(part, b.Parts[i]) {
			return false
		}
	}
	return true
}

func samePart(a, b ContentPart) bool {
	if a.Text != b.Text {
		return false
	}
	if (a.Image == nil) != (b.Image == nil) {
		return false
	}
	if a.Image == nil {
		return true
	}
	return a.Image.URL == b.Image.URL && a.Image.MIME == b.Image.MIME && a.Image.Detail == b.Image.Detail &&
		bytes.Equal(a.Image.Data, b.Image.Data)
}

// Validate checks image sources before they reach a provider.
func (p ImagePart) Validate() error {
	if (p.URL == "") == (len(p.Data) == 0) {
		return fmt.Errorf("llm: an image needs exactly one of URL or bytes")
	}
	if p.Detail != "" && p.Detail != "auto" && p.Detail != "low" && p.Detail != "high" {
		return fmt.Errorf("llm: invalid image detail %q", p.Detail)
	}
	if p.URL != "" {
		source, err := url.Parse(p.URL)
		if err != nil || (source.Scheme != "https" && source.Scheme != "http") || source.Hostname() == "" || source.User != nil {
			return fmt.Errorf("llm: image URL must be an absolute HTTP(S) URL without credentials")
		}
	} else {
		switch p.MIME {
		case "", "image/jpeg", "image/png", "image/webp", "image/gif":
		default:
			return fmt.Errorf("llm: unsupported image media type %q", p.MIME)
		}
	}
	return nil
}
