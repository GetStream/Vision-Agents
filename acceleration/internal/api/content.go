package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// maxSocketMessage bounds one incoming websocket frame, sized for a JPEG plus JSON.
const maxSocketMessage = 5 << 20

type wirePart struct {
	Type     string     `json:"type"`
	Text     string     `json:"text"`
	ImageURL *wireImage `json:"image_url,omitempty"`
}
type wireImage struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

func parseContent(raw json.RawMessage) (string, []llm.ContentPart, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return "", nil, nil
	}
	if trimmed[0] == '"' {
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return "", nil, err
		}
		return text, nil, nil
	}
	parts, err := parseParts(trimmed)
	return "", parts, err
}

func parseToolOutput(raw json.RawMessage) ([]llm.ContentPart, error) {
	text, parts, err := parseContent(raw)
	if err != nil {
		return nil, err
	}
	if len(parts) > 0 {
		return parts, nil
	}
	return llm.TextParts(text), nil
}

func parseParts(raw json.RawMessage) ([]llm.ContentPart, error) {
	var wires []wirePart
	if err := json.Unmarshal(raw, &wires); err != nil {
		return nil, fmt.Errorf("parts: %w", err)
	}
	return partsFromWire(wires)
}

func partsFromWire(wires []wirePart) ([]llm.ContentPart, error) {
	parts := make([]llm.ContentPart, 0, len(wires))
	for _, wire := range wires {
		switch wire.Type {
		case "text":
			if wire.ImageURL != nil {
				return nil, fmt.Errorf("a text part cannot contain an image")
			}
			if wire.Text != "" {
				parts = append(parts, llm.ContentPart{Text: wire.Text})
			}
		case "image_url":
			if wire.Text != "" {
				return nil, fmt.Errorf("an image part cannot contain text")
			}
			if wire.ImageURL == nil {
				return nil, fmt.Errorf("image_url is required")
			}
			image, err := imageFromURL(wire.ImageURL.URL, wire.ImageURL.Detail)
			if err != nil {
				return nil, err
			}
			parts = append(parts, llm.ContentPart{Image: &image})
		default:
			return nil, fmt.Errorf("unknown content part type %q", wire.Type)
		}
	}
	return parts, nil
}

func imagesFromWire(wires []wireImage) ([]llm.ImagePart, error) {
	images := make([]llm.ImagePart, 0, len(wires))
	for _, wire := range wires {
		image, err := imageFromURL(wire.URL, wire.Detail)
		if err != nil {
			return nil, err
		}
		images = append(images, image)
	}
	return images, nil
}

func imageFromURL(url, detail string) (llm.ImagePart, error) {
	image := llm.ImagePart{URL: url, Detail: detail}
	if strings.HasPrefix(url, "data:") {
		parsed, err := llm.ParseDataURI(url)
		if err != nil {
			return llm.ImagePart{}, err
		}
		image = parsed
		image.Detail = detail
	}
	return image, image.Validate()
}
