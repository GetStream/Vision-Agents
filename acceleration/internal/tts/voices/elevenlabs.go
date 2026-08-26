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

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/elevenlabs"
)

// elevenLabsBaseURL is the production endpoint. The streaming provider's is a WebSocket
// URL, so this one is its own rather than shared.
const elevenLabsBaseURL = "https://api.elevenlabs.io"

// ElevenLabsOptions configures the cloner. APIKey falls back to ELEVENLABS_API_KEY.
type ElevenLabsOptions struct {
	APIKey  string
	BaseURL string
	Timeout time.Duration
}

// ElevenLabs prepares a voice with ElevenLabs instant voice cloning.
type ElevenLabs struct {
	options ElevenLabsOptions
	client  *http.Client
}

// NewElevenLabs validates the options and returns a cloner.
func NewElevenLabs(options ElevenLabsOptions) (*ElevenLabs, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("ELEVENLABS_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("voices: api key is required (set ELEVENLABS_API_KEY)")
	}
	if options.BaseURL == "" {
		options.BaseURL = elevenLabsBaseURL
	}
	return &ElevenLabs{options: options, client: client(options.Timeout)}, nil
}

// Prepare uploads the recordings and returns the voice id sessions ask for.
func (e *ElevenLabs) Prepare(ctx context.Context, request Request) (string, error) {
	if err := request.Validate(); err != nil {
		return "", err
	}

	body := newForm()
	if err := body.field("name", request.Name); err != nil {
		return "", err
	}
	if err := body.field("description", request.Description); err != nil {
		return "", err
	}
	// Every recording goes under the same field name, which is how the endpoint takes
	// more than one.
	for _, sample := range request.Samples {
		if err := body.file("files", sample); err != nil {
			return "", err
		}
	}
	content, contentType, err := body.done()
	if err != nil {
		return "", err
	}

	url := strings.TrimSuffix(e.options.BaseURL, "/") + "/v1/voices/add"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, url, content)
	if err != nil {
		return "", err
	}
	httpRequest.Header.Set("xi-api-key", e.options.APIKey)
	httpRequest.Header.Set("Content-Type", contentType)

	response, err := e.client.Do(httpRequest)
	if err != nil {
		return "", fmt.Errorf("voices: elevenlabs clone: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", refused(elevenlabs.ProviderName, response)
	}

	var created struct {
		VoiceID string `json:"voice_id"`
	}
	if err := json.NewDecoder(response.Body).Decode(&created); err != nil {
		return "", fmt.Errorf("voices: elevenlabs clone: decode: %w", err)
	}
	if created.VoiceID == "" {
		return "", errors.New("voices: elevenlabs took the recordings but named no voice")
	}
	return created.VoiceID, nil
}

// Delete takes the voice back off ElevenLabs.
func (e *ElevenLabs) Delete(ctx context.Context, externalID string) error {
	if externalID == "" {
		return nil
	}

	url := strings.TrimSuffix(e.options.BaseURL, "/") + "/v1/voices/" + externalID
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		return err
	}
	httpRequest.Header.Set("xi-api-key", e.options.APIKey)

	response, err := e.client.Do(httpRequest)
	if err != nil {
		return fmt.Errorf("voices: elevenlabs delete: %w", err)
	}
	defer response.Body.Close()

	// A voice that is already gone is not a failure, because the caller wanted it gone.
	if response.StatusCode == http.StatusNotFound {
		return nil
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return refused(elevenlabs.ProviderName, response)
	}
	return nil
}
