// Package fal draws pictures through FAL's queue API.
//
// FAL runs a generation as a job: it is submitted, polled until it is done, and its result
// fetched. A job this gives up on is cancelled rather than left to draw for nobody, and
// the result is asked for inline so the picture arrives in the answer. FAL would otherwise
// hand back a link to its own storage, which this never fetches from or passes on.
//
// Every address the queue answers with is checked before it is called: it has to be the
// queue host this was built against, and the model's own app on it. The key goes with
// every call, so an answer pointing somewhere else would be an answer asking for it.
package fal

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
)

// ProviderName is how this provider is named in stats.
const ProviderName = "fal"

const apiKeyEnvVar = "FAL_KEY"

const defaultBaseURL = "https://queue.fal.run"

// pollInterval is how often a job is asked whether it is done. A picture takes seconds to
// draw, so asking more often than this buys nothing.
var pollInterval = 3 * time.Second

// slots is how many jobs this process holds at FAL at once. A job waiting for a slot has
// not been submitted, so it is not billed and can still go elsewhere.
var slots = make(chan struct{}, 2)

const (
	// requestTimeout bounds one call to the queue. The generation as a whole is bounded
	// by whoever asked for it.
	requestTimeout = 30 * time.Second
	// cancelTimeout bounds cancelling a job this gave up on, which happens after the
	// caller's own deadline has passed.
	cancelTimeout = 5 * time.Second
	// minSide and maxSide are the sizes FAL will draw at.
	minSide = 512
	maxSide = 2048
	// statusLimit caps what a submission or a status check may answer with, and
	// resultLimit the result, which carries every picture base64 encoded.
	statusLimit = 1 << 20
	resultLimit = imagegen.MaxImages * (imagegen.MaxBytes/3*4 + 1<<20)
	// errorBodyLimit caps how much of a refusal is repeated in an error.
	errorBodyLimit = 512
)

// shapes are the aspect ratios FAL names, as the size it draws them at.
var shapes = map[string]string{
	"1:1":  "square_hd",
	"4:3":  "landscape_4_3",
	"3:4":  "portrait_4_3",
	"16:9": "landscape_16_9",
	"9:16": "portrait_16_9",
}

// Options configures a Provider. The key falls back to the environment, the way every
// other provider in this service is configured.
type Options struct {
	// APIKey defaults to FAL_KEY.
	APIKey string
	// Model is FAL's endpoint id, e.g. alibaba/qwen-image-3/text-to-image.
	Model string
	// BaseURL replaces the queue host, which is what a test needs. Every address the
	// queue answers with is held to it.
	BaseURL string
	Logger  *slog.Logger
}

// Provider draws with one FAL model. It satisfies imagegen.Provider.
type Provider struct {
	apiKey string
	model  string
	base   *url.URL
	// app is the path every address for this model's jobs sits under: the first two
	// segments of the model id, which is how FAL names a job's status and result.
	app    string
	client *http.Client
	logger *slog.Logger
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("fal: " + apiKeyEnvVar + " is required")
	}
	segments := strings.Split(options.Model, "/")
	if len(segments) < 2 || slices.Contains(segments, "") {
		return nil, fmt.Errorf("fal: %q is not a FAL model such as alibaba/qwen-image-3/text-to-image", options.Model)
	}
	if options.BaseURL == "" {
		options.BaseURL = defaultBaseURL
	}
	base, err := url.Parse(options.BaseURL)
	if err != nil || base.Host == "" {
		return nil, fmt.Errorf("fal: %q is not a queue address", options.BaseURL)
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	return &Provider{
		apiKey: options.APIKey,
		model:  options.Model,
		base:   base,
		app:    "/" + segments[0] + "/" + segments[1],
		client: &http.Client{
			Timeout: requestTimeout,
			// A redirect is an address nobody checked.
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
		logger: options.Logger,
	}, nil
}

// Provider implements imagegen.Provider.
func (p *Provider) Provider() string { return ProviderName }

// Model is FAL's endpoint id.
func (p *Provider) Model() string { return p.model }

// Start opens nothing: the key was checked when this was built.
func (p *Provider) Start(context.Context) error { return nil }

// Close releases nothing.
func (p *Provider) Close() error { return nil }

// input is what FAL's image endpoints take.
type input struct {
	Prompt              string `json:"prompt"`
	NegativePrompt      string `json:"negative_prompt,omitempty"`
	ImageSize           any    `json:"image_size"`
	NumImages           int    `json:"num_images"`
	Seed                *int64 `json:"seed,omitempty"`
	OutputFormat        string `json:"output_format"`
	SyncMode            bool   `json:"sync_mode"`
	EnableSafetyChecker bool   `json:"enable_safety_checker"`
}

// size is a size FAL is given in pixels rather than by name.
type size struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

// job is what a submission answers with: where to ask about it.
type job struct {
	RequestID   string `json:"request_id"`
	StatusURL   string `json:"status_url"`
	ResponseURL string `json:"response_url"`
	CancelURL   string `json:"cancel_url"`
}

// Generate submits the request, waits for it and returns what it drew.
func (p *Provider) Generate(ctx context.Context, request imagegen.Request) (imagegen.Result, error) {
	format := request.Format
	if format == "" {
		format = imagegen.FormatPNG
	}
	body, err := p.input(request, format)
	if err != nil {
		return imagegen.Result{}, imagegen.Fail(imagegen.UnsupportedOption, false, err)
	}

	select {
	case slots <- struct{}{}:
		defer func() { <-slots }()
	case <-ctx.Done():
		return imagegen.Result{}, failure(false, ctx.Err())
	}

	answer, err := p.call(ctx, http.MethodPost, p.base.JoinPath(p.model).String(), body, statusLimit)
	if err != nil {
		return imagegen.Result{}, failure(false, err)
	}

	// From here the job is FAL's, and may be billed, so whatever goes wrong is not worth
	// asking anybody else about.
	var submitted job
	if json.Unmarshal(answer, &submitted) != nil ||
		!p.pinned(submitted.StatusURL) || !p.pinned(submitted.ResponseURL) || !p.pinned(submitted.CancelURL) {
		return imagegen.Result{}, failure(true, errors.New("fal: the queue answered with somewhere this will not go"))
	}

	finished := false
	defer func() {
		if finished {
			return
		}
		cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), cancelTimeout)
		defer cancel()
		if _, err := p.call(cleanup, http.MethodPut, submitted.CancelURL, nil, statusLimit); err != nil {
			p.logger.Warn("could not cancel an image job", "request", submitted.RequestID, "error", err)
		}
	}()

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return imagegen.Result{}, failure(true, ctx.Err())
		case <-ticker.C:
		}

		answer, err := p.call(ctx, http.MethodGet, submitted.StatusURL, nil, statusLimit)
		if err != nil {
			return imagegen.Result{}, failure(true, err)
		}
		var status struct {
			Status string `json:"status"`
		}
		if err := json.Unmarshal(answer, &status); err != nil {
			return imagegen.Result{}, failure(true, fmt.Errorf("fal: the status is not JSON: %w", err))
		}

		switch status.Status {
		case "IN_QUEUE", "IN_PROGRESS":
			continue
		case "COMPLETED":
			finished = true
			answer, err := p.call(ctx, http.MethodGet, submitted.ResponseURL, nil, resultLimit)
			if err != nil {
				return imagegen.Result{}, failure(true, err)
			}
			return p.decode(answer, request.Count(), format)
		default:
			return imagegen.Result{}, failure(true, fmt.Errorf("fal: the job is %q", status.Status))
		}
	}
}

// input builds what FAL is sent. A size or shape it cannot draw is refused here, before
// anything is submitted, so the request can still go elsewhere.
func (p *Provider) input(request imagegen.Request, format string) ([]byte, error) {
	sent := input{
		Prompt:              request.Prompt,
		NegativePrompt:      request.NegativePrompt,
		ImageSize:           shapes["1:1"],
		NumImages:           request.Count(),
		Seed:                request.Seed,
		OutputFormat:        format,
		SyncMode:            true,
		EnableSafetyChecker: true,
	}
	switch {
	case request.Width != 0:
		if request.Width < minSide || request.Width > maxSide || request.Height < minSide || request.Height > maxSide {
			return nil, fmt.Errorf("fal: %s draws from %d to %d pixels a side, not %dx%d",
				p.model, minSide, maxSide, request.Width, request.Height)
		}
		sent.ImageSize = size{Width: request.Width, Height: request.Height}
	case request.AspectRatio != "":
		shape, ok := shapes[request.AspectRatio]
		if !ok {
			return nil, fmt.Errorf("fal: %s draws 1:1, 4:3, 3:4, 16:9 and 9:16, not %s", p.model, request.AspectRatio)
		}
		sent.ImageSize = shape
	}
	return json.Marshal(sent)
}

// decode reads the pictures out of a finished job.
func (p *Provider) decode(answer []byte, count int, format string) (imagegen.Result, error) {
	var output struct {
		Images []struct {
			URL string `json:"url"`
		} `json:"images"`
		Seed            *int64 `json:"seed"`
		HasNSFWConcepts []bool `json:"has_nsfw_concepts"`
	}
	if err := json.Unmarshal(answer, &output); err != nil {
		return imagegen.Result{}, failure(true, fmt.Errorf("fal: the result is not JSON: %w", err))
	}
	if slices.Contains(output.HasNSFWConcepts, true) {
		return imagegen.Result{}, imagegen.Fail(imagegen.ContentFiltered, true,
			errors.New("fal: the safety checker flagged the picture"))
	}
	if len(output.Images) != count {
		return imagegen.Result{}, failure(true, fmt.Errorf("fal: asked for %d pictures and got %d", count, len(output.Images)))
	}

	prefix := "data:image/" + format + ";base64,"
	result := imagegen.Result{Images: make([]imagegen.Image, 0, count)}
	for _, drawn := range output.Images {
		encoded, ok := strings.CutPrefix(drawn.URL, prefix)
		if !ok {
			return imagegen.Result{}, failure(true, errors.New("fal: a picture came back as something other than inline "+format))
		}
		raw, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return imagegen.Result{}, failure(true, fmt.Errorf("fal: a picture is not base64: %w", err))
		}
		picture, err := imagegen.Verify(raw)
		if err != nil {
			return imagegen.Result{}, failure(true, err)
		}
		if picture.MediaType != "image/"+format || picture.Width > maxSide || picture.Height > maxSide {
			return imagegen.Result{}, failure(true, fmt.Errorf("fal: a picture is a %dx%d %s",
				picture.Width, picture.Height, picture.MediaType))
		}
		picture.Seed = output.Seed
		result.Images = append(result.Images, picture)
	}
	return result, nil
}

// refusal is an answer from the queue that was not a success.
type refusal struct {
	status int
	body   []byte
}

func (r *refusal) Error() string {
	body := r.body
	if len(body) > errorBodyLimit {
		body = body[:errorBodyLimit]
	}
	return fmt.Sprintf("fal: the queue answered %d: %s", r.status, strings.TrimSpace(string(body)))
}

// failure codes an error from the queue. FAL reports a prompt or a picture its moderation
// refused as a content policy violation, which is the one refusal that is not a failure.
func failure(accepted bool, err error) *imagegen.Error {
	var refused *refusal
	if errors.As(err, &refused) && bytes.Contains(refused.body, []byte("content_policy_violation")) {
		return imagegen.Fail(imagegen.ContentFiltered, accepted, err)
	}
	return imagegen.Fail(imagegen.CodeOf(err), accepted, err)
}

// call makes one request to the queue and returns what it answered.
func (p *Provider) call(ctx context.Context, method, endpoint string, body []byte, limit int64) ([]byte, error) {
	if !p.pinned(endpoint) {
		return nil, fmt.Errorf("fal: %s is not this model's queue", endpoint)
	}
	request, err := http.NewRequestWithContext(ctx, method, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("fal: %w", err)
	}
	request.Header.Set("Authorization", "Key "+p.apiKey)
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}

	response, err := p.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("fal: %w", err)
	}
	defer response.Body.Close()

	answer, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if err != nil {
		return nil, fmt.Errorf("fal: reading the answer: %w", err)
	}
	if int64(len(answer)) > limit {
		return nil, fmt.Errorf("fal: the answer is larger than %d bytes", limit)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, &refusal{status: response.StatusCode, body: answer}
	}
	return answer, nil
}

// pinned reports whether an address is this model's app on the queue host this was built
// against, with nothing on it that could take a request somewhere else.
func (p *Provider) pinned(raw string) bool {
	address, err := url.Parse(raw)
	if err != nil {
		return false
	}
	return address.Scheme == p.base.Scheme && address.Host == p.base.Host &&
		address.User == nil && address.Opaque == "" && address.RawQuery == "" && address.Fragment == "" &&
		path.Clean(address.Path) == address.Path &&
		(address.Path == p.app || strings.HasPrefix(address.Path, p.app+"/"))
}
