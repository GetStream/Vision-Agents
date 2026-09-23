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

// Speak says a line in the voice with the model a call would use.
func (e *ElevenLabs) Speak(ctx context.Context, externalID, text string) (Speech, error) {
	url := strings.TrimSuffix(e.options.BaseURL, "/") + "/v1/text-to-speech/" + externalID +
		"?output_format=" + elevenlabs.DefaultRecordingFormat
	header := http.Header{}
	header.Set("xi-api-key", e.options.APIKey)
	payload := map[string]string{"text": text, "model_id": elevenlabs.DefaultModel}
	return speak(ctx, e.client, elevenlabs.ProviderName, url, "audio/mpeg", header, payload)
}

// List reads the voices this account can speak in: the premade library plus anything
// cloned into it. Legacy voices are left out, since they are the ones ElevenLabs itself no
// longer offers.
func (e *ElevenLabs) List(ctx context.Context) ([]Library, error) {
	url := strings.TrimSuffix(e.options.BaseURL, "/") + "/v1/voices?show_legacy=false"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	httpRequest.Header.Set("xi-api-key", e.options.APIKey)

	response, err := e.client.Do(httpRequest)
	if err != nil {
		return nil, fmt.Errorf("voices: elevenlabs list: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, refused(elevenlabs.ProviderName, response)
	}
	var listed struct {
		Voices []struct {
			VoiceID     string            `json:"voice_id"`
			Name        string            `json:"name"`
			Category    string            `json:"category"`
			Description string            `json:"description"`
			PreviewURL  string            `json:"preview_url"`
			Labels      map[string]string `json:"labels"`
		} `json:"voices"`
	}
	if err := json.NewDecoder(response.Body).Decode(&listed); err != nil {
		return nil, fmt.Errorf("voices: elevenlabs list: decode: %w", err)
	}

	found := make([]Library, 0, len(listed.Voices))
	for _, voice := range listed.Voices {
		if voice.VoiceID == "" {
			continue
		}
		description := voice.Description
		if description == "" {
			description = voice.Labels["descriptive"]
		}
		found = append(found, Library{
			ID:          voice.VoiceID,
			Name:        voice.Name,
			Description: description,
			Gender:      voice.Labels["gender"],
			Accent:      voice.Labels["accent"],
			Language:    voice.Labels["language"],
			Tags:        labelTags(voice.Labels, "age", "use_case"),
			// Premade voices are the shared library; everything else was made here.
			Own:     voice.Category != "premade",
			Preview: voice.PreviewURL != "",
		})
	}
	return found, nil
}

// Preview fetches the sample ElevenLabs already published for the voice, which is why it
// costs nothing to hear. Voices without one are refused rather than synthesised.
func (e *ElevenLabs) Preview(ctx context.Context, id string) (Speech, error) {
	url := strings.TrimSuffix(e.options.BaseURL, "/") + "/v1/voices/" + id
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return Speech{}, err
	}
	httpRequest.Header.Set("xi-api-key", e.options.APIKey)

	response, err := e.client.Do(httpRequest)
	if err != nil {
		return Speech{}, fmt.Errorf("voices: elevenlabs preview: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return Speech{}, refused(elevenlabs.ProviderName, response)
	}
	var voice struct {
		PreviewURL string `json:"preview_url"`
	}
	if err := json.NewDecoder(response.Body).Decode(&voice); err != nil {
		return Speech{}, fmt.Errorf("voices: elevenlabs preview: decode: %w", err)
	}
	if voice.PreviewURL == "" {
		return Speech{}, fmt.Errorf("voices: elevenlabs has published no sample of %s", id)
	}
	return fetchPreview(ctx, e.client, elevenlabs.ProviderName, voice.PreviewURL, nil, "audio/mpeg")
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
