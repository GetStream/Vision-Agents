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

// Speak says a line in the voice with the model a call would use.
func (c *Cartesia) Speak(ctx context.Context, externalID, text string) (Speech, error) {
	url := strings.TrimSuffix(c.options.BaseURL, "/") + "/tts/bytes"
	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.options.APIKey)
	header.Set("Cartesia-Version", cartesiaAPIVersion)
	payload := map[string]any{
		"model_id":   cartesia.DefaultModel,
		"transcript": text,
		"voice":      map[string]string{"id": externalID},
		"language":   c.options.Language,
		"output_format": map[string]any{
			"container":   "wav",
			"encoding":    "pcm_s16le",
			"sample_rate": cartesia.DefaultSampleRate,
		},
	}
	return speak(ctx, c.client, cartesia.ProviderName, url, "audio/wav", header, payload)
}

// cartesiaPageLimit is the largest page the endpoint serves, and cartesiaPages bounds how
// many are read: a library that needs more than a thousand entries to find a voice needs
// a search box, not a longer list.
const (
	cartesiaPageLimit = 100
	cartesiaPages     = 10
)

// List reads Cartesia's library a page at a time. The preview URL is asked for by name,
// because Cartesia only fills it in when it is expanded.
func (c *Cartesia) List(ctx context.Context) ([]Library, error) {
	var (
		found []Library
		after string
	)
	for page := 0; page < cartesiaPages; page++ {
		url := fmt.Sprintf("%s/voices?limit=%d&expand[]=preview_file_url",
			strings.TrimSuffix(c.options.BaseURL, "/"), cartesiaPageLimit)
		if after != "" {
			url += "&starting_after=" + after
		}
		httpRequest, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return nil, err
		}
		c.authorize(httpRequest)

		response, err := c.client.Do(httpRequest)
		if err != nil {
			return nil, fmt.Errorf("voices: cartesia list: %w", err)
		}
		var listed struct {
			Data []struct {
				ID          string `json:"id"`
				Name        string `json:"name"`
				Description string `json:"description"`
				Tagline     string `json:"tagline"`
				Gender      string `json:"gender"`
				Accent      string `json:"accent"`
				Language    string `json:"language"`
				IsOwner     bool   `json:"is_owner"`
				Preview     string `json:"preview_file_url"`
			} `json:"data"`
			HasMore  bool   `json:"has_more"`
			NextPage string `json:"next_page"`
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			err = refused(cartesia.ProviderName, response)
			response.Body.Close()
			return nil, err
		}
		err = json.NewDecoder(response.Body).Decode(&listed)
		response.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("voices: cartesia list: decode: %w", err)
		}

		for _, voice := range listed.Data {
			if voice.ID == "" {
				continue
			}
			description := voice.Description
			if description == "" {
				description = voice.Tagline
			}
			found = append(found, Library{
				ID:          voice.ID,
				Name:        voice.Name,
				Description: description,
				Gender:      voice.Gender,
				Accent:      voice.Accent,
				Language:    voice.Language,
				Own:         voice.IsOwner,
				Preview:     voice.Preview != "",
			})
		}
		if !listed.HasMore || listed.NextPage == "" {
			break
		}
		after = listed.NextPage
	}
	return found, nil
}

// Preview fetches the sample Cartesia holds for the voice. The URL it hands out wants the
// same key the listing did, so the audio is read here rather than handed to a browser.
func (c *Cartesia) Preview(ctx context.Context, id string) (Speech, error) {
	url := strings.TrimSuffix(c.options.BaseURL, "/") + "/voices/" + id + "?expand[]=preview_file_url"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return Speech{}, err
	}
	c.authorize(httpRequest)

	response, err := c.client.Do(httpRequest)
	if err != nil {
		return Speech{}, fmt.Errorf("voices: cartesia preview: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return Speech{}, refused(cartesia.ProviderName, response)
	}
	var voice struct {
		Preview string `json:"preview_file_url"`
	}
	if err := json.NewDecoder(response.Body).Decode(&voice); err != nil {
		return Speech{}, fmt.Errorf("voices: cartesia preview: decode: %w", err)
	}
	if voice.Preview == "" {
		return Speech{}, fmt.Errorf("voices: cartesia has published no sample of %s", id)
	}
	header := http.Header{}
	header.Set("Authorization", "Bearer "+c.options.APIKey)
	header.Set("Cartesia-Version", cartesiaAPIVersion)
	return fetchPreview(ctx, c.client, cartesia.ProviderName, voice.Preview, header, "audio/wav")
}

func (c *Cartesia) authorize(request *http.Request) {
	request.Header.Set("Authorization", "Bearer "+c.options.APIKey)
	request.Header.Set("Cartesia-Version", cartesiaAPIVersion)
}
