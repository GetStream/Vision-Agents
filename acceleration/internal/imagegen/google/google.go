// Package google draws pictures with Gemini's native image output.
//
// Gemini draws in the same generateContent call it answers text in, asked for an image
// rather than words. Imagen, Google's dedicated image models, are not reachable this way:
// the Gemini API stopped serving Imagen 4, so the Gemini image models are what is left.
//
// A call draws one picture, so a request for several is that many calls. Gemini is asked
// for a shape and nothing else: it takes no size in pixels, no seed and no negative prompt,
// and a request naming one is refused rather than drawn without it.
package google

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
)

// ProviderName is how this provider is named in stats.
const ProviderName = "google"

const apiKeyEnvVar = "GOOGLE_API_KEY"

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

const (
	// requestTimeout bounds one picture. The generation as a whole is bounded by whoever
	// asked for it.
	requestTimeout = 120 * time.Second
	// responseLimit caps one answer, which carries one picture base64 encoded.
	responseLimit = imagegen.MaxBytes/3*4 + 1<<20
	// errorBodyLimit caps how much of a refusal is repeated in an error.
	errorBodyLimit = 512
	// imageSize is the resolution asked for, which is the one the configured price is for.
	imageSize = "1K"
)

// shapes are the aspect ratios Gemini draws.
var shapes = []string{"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}

// filtered are the reasons Gemini gives for stopping on a safety rule rather than
// finishing the picture.
var filtered = []string{
	"SAFETY", "IMAGE_SAFETY", "PROHIBITED_CONTENT", "IMAGE_PROHIBITED_CONTENT",
	"BLOCKLIST", "SPII", "RECITATION", "IMAGE_RECITATION",
}

// Options configures a Provider. The key falls back to the environment, the way every
// other provider in this service is configured.
type Options struct {
	// APIKey defaults to GOOGLE_API_KEY.
	APIKey string
	// Model is the Gemini model id, e.g. gemini-3.1-flash-image.
	Model string
	// BaseURL replaces the Gemini API host, which is what a test needs.
	BaseURL string
}

// Provider draws with one Gemini model. It satisfies imagegen.Provider.
type Provider struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("google: " + apiKeyEnvVar + " is required")
	}
	if options.Model == "" {
		return nil, errors.New("google: a model is required")
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}

	return &Provider{
		apiKey:  options.APIKey,
		model:   options.Model,
		baseURL: strings.TrimSuffix(options.BaseURL, "/"),
		client: &http.Client{
			Timeout:       requestTimeout,
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
	}, nil
}

// Provider implements imagegen.Provider.
func (p *Provider) Provider() string { return ProviderName }

// Model is the Gemini model id.
func (p *Provider) Model() string { return p.model }

// Start opens nothing: the key was checked when this was built.
func (p *Provider) Start(context.Context) error { return nil }

// Close releases nothing.
func (p *Provider) Close() error { return nil }

type generateRequest struct {
	Contents         []content        `json:"contents"`
	GenerationConfig generationConfig `json:"generationConfig"`
}

type content struct {
	Parts []textPart `json:"parts"`
}

type textPart struct {
	Text string `json:"text"`
}

type generationConfig struct {
	ResponseModalities []string    `json:"responseModalities"`
	ImageConfig        imageConfig `json:"imageConfig"`
}

type imageConfig struct {
	AspectRatio string `json:"aspectRatio"`
	ImageSize   string `json:"imageSize"`
}

type generateResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				InlineData *struct {
					MimeType string `json:"mimeType"`
					Data     string `json:"data"`
				} `json:"inlineData"`
				Thought bool `json:"thought"`
			} `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	PromptFeedback struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
}

// Generate draws the request, one call a picture.
func (p *Provider) Generate(ctx context.Context, request imagegen.Request) (imagegen.Result, error) {
	if request.Width != 0 || request.Seed != nil || request.NegativePrompt != "" || request.Format != "" {
		return imagegen.Result{}, imagegen.Fail(imagegen.UnsupportedOption, false,
			errors.New("google: Gemini is asked for a shape and nothing else"))
	}
	shape := request.AspectRatio
	if shape == "" {
		shape = "1:1"
	}
	if !slices.Contains(shapes, shape) {
		return imagegen.Result{}, imagegen.Fail(imagegen.UnsupportedOption, false,
			fmt.Errorf("google: Gemini draws %s, not %s", strings.Join(shapes, ", "), shape))
	}

	body, err := json.Marshal(generateRequest{
		Contents: []content{{Parts: []textPart{{Text: request.Prompt}}}},
		GenerationConfig: generationConfig{
			ResponseModalities: []string{"IMAGE"},
			ImageConfig:        imageConfig{AspectRatio: shape, ImageSize: imageSize},
		},
	})
	if err != nil {
		return imagegen.Result{}, imagegen.Fail(imagegen.ProviderFailed, false, err)
	}

	result := imagegen.Result{Images: make([]imagegen.Image, 0, request.Count())}
	for drawn := range request.Count() {
		// Once one picture has been drawn the request has been billed for, so a later
		// failure is not worth asking anybody else about.
		picture, err := p.draw(ctx, body, drawn > 0)
		if err != nil {
			return imagegen.Result{}, err
		}
		result.Images = append(result.Images, picture)
	}
	return result, nil
}

// draw asks Gemini for one picture.
func (p *Provider) draw(ctx context.Context, body []byte, accepted bool) (imagegen.Image, error) {
	endpoint := p.baseURL + "/models/" + p.model + ":generateContent"
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return imagegen.Image{}, imagegen.Fail(imagegen.ProviderFailed, accepted, fmt.Errorf("google: %w", err))
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("x-goog-api-key", p.apiKey)

	response, err := p.client.Do(request)
	if err != nil {
		return imagegen.Image{}, imagegen.Fail(imagegen.CodeOf(err), accepted, fmt.Errorf("google: %w", err))
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		refusal, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return imagegen.Image{}, imagegen.Fail(imagegen.ProviderFailed, accepted,
			fmt.Errorf("google: Gemini answered %d: %s", response.StatusCode, strings.TrimSpace(string(refusal))))
	}

	// From here Gemini has drawn, and billed, whatever it drew.
	answer, err := io.ReadAll(io.LimitReader(response.Body, responseLimit+1))
	if err != nil {
		return imagegen.Image{}, imagegen.Fail(imagegen.CodeOf(err), true, fmt.Errorf("google: reading the answer: %w", err))
	}
	if len(answer) > responseLimit {
		return imagegen.Image{}, imagegen.Fail(imagegen.ProviderFailed, true,
			fmt.Errorf("google: the answer is larger than %d bytes", responseLimit))
	}
	var output generateResponse
	if err := json.Unmarshal(answer, &output); err != nil {
		return imagegen.Image{}, imagegen.Fail(imagegen.ProviderFailed, true, fmt.Errorf("google: the answer is not JSON: %w", err))
	}
	if reason := output.PromptFeedback.BlockReason; reason != "" {
		return imagegen.Image{}, imagegen.Fail(imagegen.ContentFiltered, true,
			fmt.Errorf("google: Gemini refused the prompt: %s", reason))
	}

	var reasons []string
	var unusable error
	for _, candidate := range output.Candidates {
		for _, part := range candidate.Content.Parts {
			// A thinking model can sketch before it draws, and a sketch is not the answer.
			if part.Thought || part.InlineData == nil {
				continue
			}
			raw, err := base64.StdEncoding.DecodeString(part.InlineData.Data)
			if err != nil {
				unusable = fmt.Errorf("google: a picture is not base64: %w", err)
				continue
			}
			picture, err := imagegen.Verify(raw)
			if err != nil {
				unusable = err
				continue
			}
			if picture.MediaType != part.InlineData.MimeType {
				unusable = fmt.Errorf("google: a picture labelled %s is %s", part.InlineData.MimeType, picture.MediaType)
				continue
			}
			return picture, nil
		}
		reasons = append(reasons, candidate.FinishReason)
	}

	if slices.ContainsFunc(reasons, func(reason string) bool { return slices.Contains(filtered, reason) }) {
		return imagegen.Image{}, imagegen.Fail(imagegen.ContentFiltered, true,
			fmt.Errorf("google: Gemini stopped drawing on a safety rule: %s", strings.Join(reasons, ", ")))
	}
	if unusable != nil {
		return imagegen.Image{}, imagegen.Fail(imagegen.ProviderFailed, true, unusable)
	}
	return imagegen.Image{}, imagegen.Fail(imagegen.ProviderFailed, true,
		fmt.Errorf("google: Gemini drew nothing (finished with %s)", strings.Join(reasons, ", ")))
}
