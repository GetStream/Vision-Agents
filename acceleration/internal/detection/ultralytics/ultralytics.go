// Package ultralytics finds objects with YOLO through the Ultralytics Platform.
//
// The shared API serves any public model by its path, anonymously or with a key, and is
// limited to 20 requests a minute. A deployment that needs more runs a dedicated endpoint
// and names it in ULTRALYTICS_ENDPOINT. That is the operator's to set rather than a
// caller's, since whatever address it names is sent the key and the image.
package ultralytics

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ProviderName is how this provider is named in stats.
const ProviderName = "ultralytics"

const (
	// APIKeyEnvVar is where the key is read from. Public models need none.
	APIKeyEnvVar = "ULTRALYTICS_API_KEY"
	// EndpointEnvVar names a dedicated endpoint's /predict URL, in place of the shared API.
	EndpointEnvVar = "ULTRALYTICS_ENDPOINT"
)

// DefaultBaseURL is the Platform API's model path.
const DefaultBaseURL = "https://platform.ultralytics.com/api/models"

// defaultOwner is whose model a two-part path such as yolo26/yolo26n names.
const defaultOwner = "ultralytics"

const (
	requestTimeout = 20 * time.Second
	responseLimit  = 2 << 20
	errorBodyLimit = 512
)

// Options configures a Provider.
type Options struct {
	// APIKey defaults to ULTRALYTICS_API_KEY.
	APIKey string
	// Model is "project/model" for one of Ultralytics' own, e.g. yolo26/yolo26n, or
	// "owner/project/model".
	Model string
	// BaseURL replaces the Platform API's model path, which is what a test needs.
	BaseURL string
	// Endpoint is a dedicated /predict URL. Defaults to ULTRALYTICS_ENDPOINT.
	Endpoint string
}

// Provider finds objects with one YOLO model. It satisfies detection.Provider.
type Provider struct {
	apiKey   string
	model    string
	endpoint string
	client   *http.Client
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	keys := keypool.FromEnv(options.APIKey, APIKeyEnvVar)
	if options.Endpoint == "" {
		options.Endpoint = os.Getenv(EndpointEnvVar)
	}
	path := strings.Split(options.Model, "/")
	if len(path) == 2 {
		path = append([]string{defaultOwner}, path...)
	}
	if len(path) != 3 || path[0] == "" || path[1] == "" || path[2] == "" {
		return nil, stack.Wrap(fmt.Errorf("ultralytics: %q is not a model path such as yolo26/yolo26n", options.Model))
	}
	if options.BaseURL == "" {
		options.BaseURL = DefaultBaseURL
	}
	endpoint := options.Endpoint
	if endpoint == "" {
		endpoint = strings.TrimSuffix(options.BaseURL, "/") + "/" + strings.Join(path, "/") + "/predict"
	}
	return &Provider{
		apiKey:   keys.Peek(),
		model:    options.Model,
		endpoint: endpoint,
		client: keys.WrapClient(&http.Client{
			Timeout:       requestTimeout,
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		}, keypool.Bearer),
	}, nil
}

// Provider implements detection.Provider.
func (p *Provider) Provider() string { return ProviderName }

// Model is the model path.
func (p *Provider) Model() string { return p.model }

// Start opens nothing.
func (p *Provider) Start(context.Context) error { return nil }

// Close releases nothing.
func (p *Provider) Close() error { return nil }

type predictResponse struct {
	Images []struct {
		Results []struct {
			Name       string  `json:"name"`
			Confidence float64 `json:"confidence"`
			Box        struct {
				X1 float64 `json:"x1"`
				Y1 float64 `json:"y1"`
				X2 float64 `json:"x2"`
				Y2 float64 `json:"y2"`
			} `json:"box"`
		} `json:"results"`
	} `json:"images"`
}

// Detect runs the model over the image, asking for boxes as fractions.
func (p *Provider) Detect(ctx context.Context, request detection.Request) (detection.Result, error) {
	if request.Embeddings || len(request.Overwrite) > 0 {
		return detection.Result{}, detection.Fail(detection.UnsupportedOption,
			errors.New("ultralytics: YOLO returns boxes and nothing else"))
	}

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	file, err := form.CreateFormFile("file", "frame."+strings.TrimPrefix(request.MediaType, "image/"))
	if err == nil {
		_, err = file.Write(request.Image)
	}
	if err == nil {
		err = form.WriteField("normalize", "true")
	}
	if err == nil && request.MinConfidence != nil {
		// Ultralytics refuses a floor below 0.01.
		err = form.WriteField("conf", strconv.FormatFloat(max(*request.MinConfidence, 0.01), 'f', -1, 64))
	}
	if err == nil {
		err = form.Close()
	}
	if err != nil {
		return detection.Result{}, detection.Fail(detection.ProviderFailed, fmt.Errorf("ultralytics: %w", err))
	}

	call, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint, &body)
	if err != nil {
		return detection.Result{}, detection.Fail(detection.ProviderFailed, fmt.Errorf("ultralytics: %w", err))
	}
	call.Header.Set("Content-Type", form.FormDataContentType())
	if p.apiKey != "" {
		call.Header.Set("Authorization", "Bearer "+p.apiKey)
	}
	response, err := p.client.Do(call)
	if err != nil {
		return detection.Result{}, detection.Fail(detection.CodeOf(err), fmt.Errorf("ultralytics: %w", err))
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		refusal, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return detection.Result{}, detection.Fail(detection.ProviderFailed,
			fmt.Errorf("ultralytics: Ultralytics answered %d: %s", response.StatusCode, strings.TrimSpace(string(refusal))))
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, responseLimit))
	if err != nil {
		return detection.Result{}, detection.Fail(detection.CodeOf(err), fmt.Errorf("ultralytics: reading the answer: %w", err))
	}
	var answer predictResponse
	if err := json.Unmarshal(raw, &answer); err != nil {
		return detection.Result{}, detection.Fail(detection.ProviderFailed, fmt.Errorf("ultralytics: the answer is not JSON: %w", err))
	}

	var result detection.Result
	for _, picture := range answer.Images {
		for _, found := range picture.Results {
			result.Detections = append(result.Detections, detection.Detection{
				Label:      found.Name,
				Confidence: found.Confidence,
				Box:        detection.Box{XMin: found.Box.X1, YMin: found.Box.Y1, XMax: found.Box.X2, YMax: found.Box.Y2},
			})
		}
	}
	return result, nil
}
