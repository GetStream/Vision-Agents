package voices

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/cartesia"
)

// cartesiaBaseURL is the production endpoint. The streaming provider's is a WebSocket URL,
// so this one is its own rather than shared.
const cartesiaBaseURL = "https://api.cartesia.ai"

// cartesiaAPIVersion pins the request and response shapes. Cartesia dates its API rather
// than numbering it.
const cartesiaAPIVersion = "2026-08-14"

// CartesiaOptions configures the cloner. APIKey falls back to CARTESIA_API_KEY.
type CartesiaOptions struct {
	APIKey  string
	BaseURL string
	// Language is the ISO code the recordings are in, which the endpoint requires.
	Language string
	Timeout  time.Duration
}

// Cartesia prepares a voice with Cartesia instant cloning.
type Cartesia struct {
	options CartesiaOptions
	client  *http.Client
}

// NewCartesia validates the options and returns a cloner.
func NewCartesia(options CartesiaOptions) (*Cartesia, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("CARTESIA_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("voices: api key is required (set CARTESIA_API_KEY)")
	}
	if options.BaseURL == "" {
		options.BaseURL = cartesiaBaseURL
	}
	if options.Language == "" {
		options.Language = "en"
	}
	return &Cartesia{options: options, client: client(options.Timeout)}, nil
}

// Prepare uploads the first recording and returns the voice id sessions ask for. Cartesia
// clones from one clip, so the rest are not sent: a few seconds of clean speech is what
// the endpoint asks for, and sending more would not make the clone better.
func (c *Cartesia) Prepare(ctx context.Context, request Request) (string, error) {
	if err := request.Validate(); err != nil {
		return "", err
	}

	body := newForm()
	if err := body.file("clip", request.Samples[0]); err != nil {
		return "", err
	}
	if err := body.field("name", request.Name); err != nil {
		return "", err
	}
	if err := body.field("description", request.Description); err != nil {
		return "", err
	}
	if err := body.field("language", c.options.Language); err != nil {
		return "", err
	}
	content, contentType, err := body.done()
	if err != nil {
		return "", err
	}

	url := strings.TrimSuffix(c.options.BaseURL, "/") + "/voices/clone"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, url, content)
	if err != nil {
		return "", err
	}
	c.authorize(httpRequest)
	httpRequest.Header.Set("Content-Type", contentType)

	response, err := c.client.Do(httpRequest)
	if err != nil {
		return "", fmt.Errorf("voices: cartesia clone: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", refused(cartesia.ProviderName, response)
	}

	var created struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(response.Body).Decode(&created); err != nil {
		return "", fmt.Errorf("voices: cartesia clone: decode: %w", err)
	}
	if created.ID == "" {
		return "", errors.New("voices: cartesia took the recording but named no voice")
	}
	return created.ID, nil
}

// Delete takes the voice back off Cartesia.
func (c *Cartesia) Delete(ctx context.Context, externalID string) error {
	if externalID == "" {
		return nil
	}

	url := strings.TrimSuffix(c.options.BaseURL, "/") + "/voices/" + externalID
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		return err
	}
	c.authorize(httpRequest)

	response, err := c.client.Do(httpRequest)
	if err != nil {
		return fmt.Errorf("voices: cartesia delete: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode == http.StatusNotFound {
		return nil
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return refused(cartesia.ProviderName, response)
	}
	return nil
}

func (c *Cartesia) authorize(request *http.Request) {
	request.Header.Set("Authorization", "Bearer "+c.options.APIKey)
	request.Header.Set("Cartesia-Version", cartesiaAPIVersion)
}
