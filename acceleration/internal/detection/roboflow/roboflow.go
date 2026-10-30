// Package roboflow finds objects with Roboflow's serverless API, through a hosted model or
// a Workflow.
//
// The model name says which. A "project/version" such as coco/3 is a hosted model, called
// with the image as the body. "faces" is a Workflow built in here: face detection, a crop
// of each face, and a CLIP embedding of each crop, so every box carries a vector a caller
// can match against people it has seen. "workflow" is the caller's own Workflow, named in
// an overwrite by its workspace and id or sent whole as a specification. A Workflow must
// output "predictions", and may output "embeddings", one per box in the same order.
package roboflow

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ProviderName is how this provider is named in stats.
const ProviderName = "roboflow"

// APIKeyEnvVar is where the key is read from.
const APIKeyEnvVar = "ROBOFLOW_API_KEY"

// DefaultBaseURL is Roboflow's serverless API, which serves both models and Workflows.
const DefaultBaseURL = "https://serverless.roboflow.com"

// The two model names that are Workflows rather than hosted models.
const (
	ModelFaces    = "faces"
	ModelWorkflow = "workflow"
)

const (
	// requestTimeout is Roboflow's own cap on a Workflow, 20 seconds, and some slack.
	requestTimeout = 25 * time.Second
	// responseLimit is Roboflow's own cap on a Workflow's answer.
	responseLimit  = 6 << 20
	errorBodyLimit = 512
)

// facesSpecification is the Workflow behind ModelFaces. CLIP describes how a crop looks
// rather than who it is, so its vectors re-identify a person in the same light and clothes
// and are not face identification.
var facesSpecification = json.RawMessage(`{
  "version": "1.0",
  "inputs": [{"type": "WorkflowImage", "name": "image"}],
  "steps": [
    {"type": "roboflow_core/roboflow_object_detection_model@v2", "name": "faces", "images": "$inputs.image", "model_id": "face-detection-mik1i/18", "confidence": 0.4},
    {"type": "roboflow_core/dynamic_crop@v1", "name": "crops", "images": "$inputs.image", "predictions": "$steps.faces.predictions"},
    {"type": "roboflow_core/clip@v1", "name": "embed", "data": "$steps.crops.crops"}
  ],
  "outputs": [
    {"type": "JsonField", "name": "predictions", "selector": "$steps.faces.predictions"},
    {"type": "JsonField", "name": "embeddings", "selector": "$steps.embed.embedding"}
  ]
}`)

// Options configures a Provider.
type Options struct {
	// APIKey defaults to ROBOFLOW_API_KEY.
	APIKey string
	// Model is a "project/version", ModelFaces or ModelWorkflow.
	Model string
	// BaseURL replaces the API host, which is what a test needs.
	BaseURL string
}

// Provider finds objects with one Roboflow model or Workflow. It satisfies
// detection.Provider.
type Provider struct {
	keys    *keypool.Pool
	model   string
	baseURL string
	client  *http.Client
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	keys := keypool.FromEnv(options.APIKey, APIKeyEnvVar)
	if keys.Len() == 0 {
		return nil, stack.Wrap(errors.New("roboflow: " + APIKeyEnvVar + " is required"))
	}
	if options.Model != ModelFaces && options.Model != ModelWorkflow {
		project, version, found := strings.Cut(options.Model, "/")
		if _, err := strconv.Atoi(version); !found || project == "" || err != nil {
			return nil, stack.Wrap(fmt.Errorf("roboflow: %q is not a project/version such as coco/3, %s or %s", options.Model, ModelFaces, ModelWorkflow))
		}
	}
	if options.BaseURL == "" {
		options.BaseURL = DefaultBaseURL
	}
	return &Provider{
		keys:    keys,
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

// Model is the project/version or Workflow name.
func (p *Provider) Model() string { return p.model }

// Start opens nothing: the key was checked when this was built.
func (p *Provider) Start(context.Context) error { return nil }

// Close releases nothing.
func (p *Provider) Close() error { return nil }

// Overwrite is what a caller may set for Roboflow. Only ModelWorkflow reads it.
type Overwrite struct {
	// Workspace and WorkflowID name a saved Workflow.
	Workspace  string `json:"workspace"`
	WorkflowID string `json:"workflow_id"`
	// Specification is a whole Workflow, in place of a saved one.
	Specification json.RawMessage `json:"specification"`
}

// predictions is how Roboflow writes boxes, from a model and from a Workflow alike.
type predictions struct {
	Image struct {
		Width  int `json:"width"`
		Height int `json:"height"`
	} `json:"image"`
	Predictions []struct {
		X          float64 `json:"x"`
		Y          float64 `json:"y"`
		Width      float64 `json:"width"`
		Height     float64 `json:"height"`
		Confidence float64 `json:"confidence"`
		Class      string  `json:"class"`
	} `json:"predictions"`
}

type workflowResponse struct {
	Outputs []struct {
		Predictions *predictions `json:"predictions"`
		Embeddings  [][]float32  `json:"embeddings"`
	} `json:"outputs"`
}

// Detect runs the model or Workflow over the image.
func (p *Provider) Detect(ctx context.Context, request detection.Request) (detection.Result, error) {
	encoded := base64.StdEncoding.EncodeToString(request.Image)
	switch p.model {
	case ModelFaces:
		if len(request.Overwrite) > 0 {
			return detection.Result{}, detection.Fail(detection.UnsupportedOption,
				errors.New("roboflow: faces is a fixed Workflow and takes no overwrite"))
		}
		return p.workflow(ctx, "/workflows/run", facesSpecification, encoded, request)
	case ModelWorkflow:
		var overwrite Overwrite
		if len(request.Overwrite) > 0 {
			if err := json.Unmarshal(request.Overwrite, &overwrite); err != nil {
				return detection.Result{}, detection.Fail(detection.UnsupportedOption, fmt.Errorf("roboflow: the overwrite is not valid: %w", err))
			}
		}
		switch {
		case len(overwrite.Specification) > 0:
			return p.workflow(ctx, "/workflows/run", overwrite.Specification, encoded, request)
		case overwrite.Workspace != "" && overwrite.WorkflowID != "":
			path := "/infer/workflows/" + url.PathEscape(overwrite.Workspace) + "/" + url.PathEscape(overwrite.WorkflowID)
			return p.workflow(ctx, path, nil, encoded, request)
		default:
			return detection.Result{}, detection.Fail(detection.UnsupportedOption,
				errors.New("roboflow: a workflow needs a workspace and a workflow_id, or a specification, in overwrites.roboflow"))
		}
	}

	if len(request.Overwrite) > 0 || request.Embeddings {
		return detection.Result{}, detection.Fail(detection.UnsupportedOption,
			fmt.Errorf("roboflow: the hosted model %s returns boxes and nothing else", p.model))
	}
	build := func(key string) (string, []byte, error) {
		query := url.Values{"api_key": {key}}
		if request.MinConfidence != nil {
			query.Set("confidence", strconv.FormatFloat(*request.MinConfidence, 'f', -1, 64))
		}
		return p.baseURL + "/" + p.model + "?" + query.Encode(), []byte(encoded), nil
	}
	var answer predictions
	if err := p.post(ctx, "application/x-www-form-urlencoded", build, &answer); err != nil {
		return detection.Result{}, err
	}
	return boxesOf(answer, nil, request), nil
}

// workflow runs a saved Workflow at path, or the specification when one is given.
func (p *Provider) workflow(ctx context.Context, path string, specification json.RawMessage, encoded string, request detection.Request) (detection.Result, error) {
	build := func(key string) (string, []byte, error) {
		body := map[string]any{
			"api_key": key,
			"inputs":  map[string]any{"image": map[string]string{"type": "base64", "value": encoded}},
		}
		if specification != nil {
			body["specification"] = specification
		}
		payload, err := json.Marshal(body)
		return p.baseURL + path, payload, err
	}
	var answer workflowResponse
	if err := p.post(ctx, "application/json", build, &answer); err != nil {
		return detection.Result{}, err
	}
	if len(answer.Outputs) == 0 || answer.Outputs[0].Predictions == nil {
		return detection.Result{}, detection.Fail(detection.ProviderFailed,
			errors.New("roboflow: the Workflow has no predictions output"))
	}
	output := answer.Outputs[0]
	if request.Embeddings && len(output.Embeddings) != len(output.Predictions.Predictions) {
		return detection.Result{}, detection.Fail(detection.UnsupportedOption,
			fmt.Errorf("roboflow: the Workflow returned %d embeddings for %d boxes", len(output.Embeddings), len(output.Predictions.Predictions)))
	}
	return boxesOf(*output.Predictions, output.Embeddings, request), nil
}

// boxesOf turns Roboflow's centre-and-size pixels into fractions of the image.
func boxesOf(answer predictions, embeddings [][]float32, request detection.Request) detection.Result {
	width, height := answer.Image.Width, answer.Image.Height
	if width < 1 || height < 1 {
		width, height = request.Width, request.Height
	}
	var result detection.Result
	for i, prediction := range answer.Predictions {
		found := detection.Detection{
			Label:      prediction.Class,
			Confidence: prediction.Confidence,
			Box:        detection.FromCenter(prediction.X, prediction.Y, prediction.Width, prediction.Height, width, height),
		}
		if i < len(embeddings) {
			found.Embedding = embeddings[i]
		}
		result.Detections = append(result.Detections, found)
	}
	return result
}

// post sends one call, built by build for a key from the pool and again for the next key
// on a 429, and decodes the answer into out.
func (p *Provider) post(ctx context.Context, contentType string, build func(key string) (string, []byte, error), out any) error {
	var buildErr error
	response, err := p.keys.Send(ctx, func(key string) (*http.Response, error) {
		endpoint, body, err := build(key)
		if err != nil {
			buildErr = err
			return nil, err
		}
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			buildErr = errors.New("roboflow: the request could not be built")
			return nil, buildErr
		}
		request.Header.Set("Content-Type", contentType)
		return p.client.Do(request)
	})
	if buildErr != nil {
		return detection.Fail(detection.ProviderFailed, buildErr)
	}
	if err != nil {
		// A hosted model's URL carries the key in its query, escaped, and an error from the
		// client repeats the URL.
		var failed *url.Error
		if errors.As(err, &failed) {
			failed.URL, _, _ = strings.Cut(failed.URL, "?")
		}
		return detection.Fail(detection.CodeOf(err), errors.New("roboflow: "+err.Error()))
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		refusal, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return detection.Fail(detection.ProviderFailed,
			fmt.Errorf("roboflow: Roboflow answered %d: %s", response.StatusCode, strings.TrimSpace(string(refusal))))
	}
	answer, err := io.ReadAll(io.LimitReader(response.Body, responseLimit))
	if err != nil {
		return detection.Fail(detection.CodeOf(err), fmt.Errorf("roboflow: reading the answer: %w", err))
	}
	if err := json.Unmarshal(answer, out); err != nil {
		return detection.Fail(detection.ProviderFailed, fmt.Errorf("roboflow: the answer is not JSON: %w", err))
	}
	return nil
}
