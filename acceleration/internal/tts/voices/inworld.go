package voices

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/inworld"
)

// inworldBaseURL is the production endpoint. The streaming provider's is a WebSocket URL,
// so this one is its own rather than shared.
const inworldBaseURL = "https://api.inworld.ai"

// InworldOptions configures the library reader. APIKey falls back to INWORLD_API_KEY.
type InworldOptions struct {
	APIKey  string
	BaseURL string
	Timeout time.Duration
}

// Inworld reads the voices Inworld offers. It is a Lister and not a Cloner: cloning with
// Inworld is a separate endpoint the router does not use.
type Inworld struct {
	options InworldOptions
	client  *http.Client
}

// NewInworld validates the options and returns a library reader.
func NewInworld(options InworldOptions) (*Inworld, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("INWORLD_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("voices: api key is required (set INWORLD_API_KEY)")
	}
	if options.BaseURL == "" {
		options.BaseURL = inworldBaseURL
	}
	return &Inworld{options: options, client: client(options.Timeout)}, nil
}

// List reads Inworld's voices, both the ones it ships and any cloned into the workspace.
func (i *Inworld) List(ctx context.Context) ([]Library, error) {
	address := strings.TrimSuffix(i.options.BaseURL, "/") + "/voices/v1/voices"
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Authorization", "Basic "+i.options.APIKey)

	response, err := i.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("voices: inworld list: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, refused(inworld.ProviderName, response)
	}
	var listed struct {
		Voices []struct {
			VoiceID     string   `json:"voiceId"`
			DisplayName string   `json:"displayName"`
			Description string   `json:"description"`
			Gender      string   `json:"gender"`
			AgeGroup    string   `json:"ageGroup"`
			LangCode    string   `json:"langCode"`
			Source      string   `json:"source"`
			Tags        []string `json:"tags"`
		} `json:"voices"`
	}
	if err := json.NewDecoder(response.Body).Decode(&listed); err != nil {
		return nil, fmt.Errorf("voices: inworld list: decode: %w", err)
	}

	found := make([]Library, 0, len(listed.Voices))
	for _, voice := range listed.Voices {
		if voice.VoiceID == "" {
			continue
		}
		name := voice.DisplayName
		if name == "" {
			name = voice.VoiceID
		}
		tags := voice.Tags
		if voice.AgeGroup != "" {
			tags = append([]string{voice.AgeGroup}, tags...)
		}
		found = append(found, Library{
			ID:          voice.VoiceID,
			Name:        name,
			Description: voice.Description,
			Gender:      voice.Gender,
			Language:    voice.LangCode,
			Tags:        tags,
			// SYSTEM is Inworld's own library; anything else was made in this workspace.
			Own: voice.Source != "" && voice.Source != "SYSTEM",
			// Inworld publishes no sample, but its preview endpoint is not metered, so
			// every voice can be heard.
			Preview: true,
		})
	}
	return found, nil
}

// Preview asks Inworld for a sample. The endpoint is not billed and the line it speaks is
// Inworld's own, so this is browsing rather than synthesis.
func (i *Inworld) Preview(ctx context.Context, id string) (Speech, error) {
	address := strings.TrimSuffix(i.options.BaseURL, "/") + "/tts/v1/voice:preview" +
		"?voice_id=" + url.QueryEscape(id) + "&model_id=" + url.QueryEscape(inworld.DefaultModel)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
	if err != nil {
		return Speech{}, err
	}
	request.Header.Set("Authorization", "Basic "+i.options.APIKey)

	response, err := i.client.Do(request)
	if err != nil {
		return Speech{}, fmt.Errorf("voices: inworld preview: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return Speech{}, refused(inworld.ProviderName, response)
	}
	var preview struct {
		AudioContent string `json:"audioContent"`
	}
	if err := json.NewDecoder(response.Body).Decode(&preview); err != nil {
		return Speech{}, fmt.Errorf("voices: inworld preview: decode: %w", err)
	}
	audio, err := base64.StdEncoding.DecodeString(preview.AudioContent)
	if err != nil {
		return Speech{}, fmt.Errorf("voices: inworld preview: decode audio: %w", err)
	}
	if len(audio) == 0 {
		return Speech{}, fmt.Errorf("voices: inworld has no sample of %s", id)
	}
	return Speech{Audio: audio, ContentType: "audio/mpeg"}, nil
}
