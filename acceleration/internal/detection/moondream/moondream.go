// Package moondream finds objects with Moondream's cloud API.
//
// Moondream is open vocabulary: it finds whatever noun it is asked for, which is what lets
// a camera look for "a package" when no fixed-class model was trained on one. It looks for
// one word a call, so a request for several is that many calls, and it scores nothing, so
// every box it returns reports a confidence of 1.
package moondream

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
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ProviderName is how this provider is named in stats.
const ProviderName = "moondream"

// APIKeyEnvVar is where the key is read from.
const APIKeyEnvVar = "MOONDREAM_API_KEY"

// DefaultBaseURL is Moondream's cloud API.
const DefaultBaseURL = "https://api.moondream.ai/v1"

const (
	requestTimeout = 20 * time.Second
	responseLimit  = 1 << 20
	errorBodyLimit = 512
)

// Options configures a Provider.
type Options struct {
	// APIKey defaults to MOONDREAM_API_KEY.
	APIKey string
	// Model is the Moondream variant, e.g. moondream3-preview.
	Model string
	// BaseURL replaces the API host, which is what a test needs.
	BaseURL string
}

// Provider finds objects with one Moondream variant. It satisfies detection.Provider.
type Provider struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(APIKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New("moondream: " + APIKeyEnvVar + " is required"))
	}
	if options.Model == "" {
		return nil, stack.Wrap(errors.New("moondream: a model is required"))
	}
	if options.BaseURL == "" {
		options.BaseURL = DefaultBaseURL
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

// Provider implements detection.Provider.
func (p *Provider) Provider() string { return ProviderName }

// Model is the Moondream variant.
func (p *Provider) Model() string { return p.model }

// Start opens nothing: the key was checked when this was built.
func (p *Provider) Start(context.Context) error { return nil }

// Close releases nothing.
func (p *Provider) Close() error { return nil }

type detectRequest struct {
	ImageURL string `json:"image_url"`
	Object   string `json:"object"`
	Variant  string `json:"variant"`
}

type detectResponse struct {
	Objects []struct {
		XMin float64 `json:"x_min"`
		YMin float64 `json:"y_min"`
		XMax float64 `json:"x_max"`
		YMax float64 `json:"y_max"`
	} `json:"objects"`
}

// Detect looks for each class in turn.
func (p *Provider) Detect(ctx context.Context, request detection.Request) (detection.Result, error) {
	if len(request.Classes) == 0 {
		return detection.Result{}, detection.Fail(detection.UnsupportedOption,
			errors.New("moondream: Moondream has to be told what to look for, and no classes were named"))
	}
	if request.MinConfidence != nil || request.Embeddings || len(request.Overwrite) > 0 {
		return detection.Result{}, detection.Fail(detection.UnsupportedOption,
			errors.New("moondream: Moondream takes classes and nothing else"))
	}
	image := "data:" + request.MediaType + ";base64," + base64.StdEncoding.EncodeToString(request.Image)

	result := detection.Result{Billed: len(request.Classes)}
	for _, class := range request.Classes {
		var answer detectResponse
		if err := p.post(ctx, "/detect", detectRequest{ImageURL: image, Object: class, Variant: p.model}, &answer); err != nil {
			return detection.Result{}, err
		}
		for _, object := range answer.Objects {
			result.Detections = append(result.Detections, detection.Detection{
				Label:      class,
				Confidence: 1,
				Box:        detection.Box{XMin: object.XMin, YMin: object.YMin, XMax: object.XMax, YMax: object.YMax},
			})
		}
	}
	return result, nil
}

// post sends one JSON call and decodes the answer into out.
func (p *Provider) post(ctx context.Context, path string, body any, out any) error {
	encoded, err := json.Marshal(body)
	if err != nil {
		return detection.Fail(detection.ProviderFailed, err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+path, bytes.NewReader(encoded))
	if err != nil {
		return detection.Fail(detection.ProviderFailed, fmt.Errorf("moondream: %w", err))
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Moondream-Auth", p.apiKey)

	response, err := p.client.Do(request)
	if err != nil {
		return detection.Fail(detection.CodeOf(err), fmt.Errorf("moondream: %w", err))
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		refusal, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return detection.Fail(detection.ProviderFailed,
			fmt.Errorf("moondream: Moondream answered %d: %s", response.StatusCode, strings.TrimSpace(string(refusal))))
	}
	answer, err := io.ReadAll(io.LimitReader(response.Body, responseLimit))
	if err != nil {
		return detection.Fail(detection.CodeOf(err), fmt.Errorf("moondream: reading the answer: %w", err))
	}
	if err := json.Unmarshal(answer, out); err != nil {
		return detection.Fail(detection.ProviderFailed, fmt.Errorf("moondream: the answer is not JSON: %w", err))
	}
	return nil
}
