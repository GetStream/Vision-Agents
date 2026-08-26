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

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/fish"
)

// fishBaseURL is the production endpoint.
const fishBaseURL = "https://api.fish.audio"

// FishOptions configures the cloner. APIKey falls back to FISH_API_KEY.
type FishOptions struct {
	APIKey  string
	BaseURL string
	Timeout time.Duration
}

// Fish prepares a voice by creating a Fish model from the recordings.
type Fish struct {
	options FishOptions
	client  *http.Client
}

// NewFish validates the options and returns a cloner.
func NewFish(options FishOptions) (*Fish, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("FISH_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("voices: api key is required (set FISH_API_KEY)")
	}
	if options.BaseURL == "" {
		options.BaseURL = fishBaseURL
	}
	return &Fish{options: options, client: client(options.Timeout)}, nil
}

// Prepare creates the model and returns the reference id sessions ask for.
func (f *Fish) Prepare(ctx context.Context, request Request) (string, error) {
	if err := request.Validate(); err != nil {
		return "", err
	}

	body := newForm()
	if err := body.field("type", "tts"); err != nil {
		return "", err
	}
	if err := body.field("title", request.Name); err != nil {
		return "", err
	}
	if err := body.field("description", request.Description); err != nil {
		return "", err
	}
	// Private, because a customer's own voice has no business on a discovery page. Fast,
	// because it is the only training mode and it leaves the voice usable at once.
	if err := body.field("visibility", "private"); err != nil {
		return "", err
	}
	if err := body.field("train_mode", "fast"); err != nil {
		return "", err
	}
	for _, sample := range request.Samples {
		if err := body.file("voices", sample); err != nil {
			return "", err
		}
		// Transcripts are positional, so one is sent per recording even when it is blank.
		if err := f.transcript(body, sample.Transcript); err != nil {
			return "", err
		}
	}
	content, contentType, err := body.done()
	if err != nil {
		return "", err
	}

	url := strings.TrimSuffix(f.options.BaseURL, "/") + "/model"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, url, content)
	if err != nil {
		return "", err
	}
	httpRequest.Header.Set("Authorization", "Bearer "+f.options.APIKey)
	httpRequest.Header.Set("Content-Type", contentType)

	response, err := f.client.Do(httpRequest)
	if err != nil {
		return "", fmt.Errorf("voices: fish clone: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", refused(fish.ProviderName, response)
	}

	var created struct {
		ID    string `json:"_id"`
		State string `json:"state"`
	}
	if err := json.NewDecoder(response.Body).Decode(&created); err != nil {
		return "", fmt.Errorf("voices: fish clone: decode: %w", err)
	}
	if created.ID == "" {
		return "", errors.New("voices: fish took the recordings but named no model")
	}
	if created.State == "failed" {
		return "", errors.New("voices: fish could not train a model from these recordings")
	}
	return created.ID, nil
}

// Delete removes the model from Fish.
func (f *Fish) Delete(ctx context.Context, externalID string) error {
	if externalID == "" {
		return nil
	}

	url := strings.TrimSuffix(f.options.BaseURL, "/") + "/model/" + externalID
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		return err
	}
	httpRequest.Header.Set("Authorization", "Bearer "+f.options.APIKey)

	response, err := f.client.Do(httpRequest)
	if err != nil {
		return fmt.Errorf("voices: fish delete: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode == http.StatusNotFound {
		return nil
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return refused(fish.ProviderName, response)
	}
	return nil
}

// transcript writes a texts part, which has to be sent even when empty so the transcripts
// line up with the recordings they belong to.
func (f *Fish) transcript(body *form, text string) error {
	return body.writer.WriteField("texts", text)
}
