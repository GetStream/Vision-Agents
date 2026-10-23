// Package detectionrouter routes object detection and records what each request cost.
//
// Resolving a target, ranking candidates and failing over are generic and live in
// internal/routing. What this package adds is the detection shape: which providers exist,
// which candidate gets which overwrite, and what counts as a unit of work, which is one
// image however many boxes came back.
//
// Nothing is kept. The boxes go back to the caller and the only thing written down is the
// stat row saying what they cost.
package detectionrouter

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Registry is the set of detection providers a build can construct.
type Registry = routing.Registry[detection.Provider]

// DefaultTarget is where a request that names nowhere goes: the fast tier, since a worker
// sampling a camera is waiting on every answer.
const DefaultTarget = "detection-fast"

// Options configures a Router. Store and Live are optional: without them the router still
// routes, it just stops recording.
type Options struct {
	Config   routing.ModalityConfig
	Registry *Registry
	Store    *store.Store
	Live     *live.Client
	Gate     routing.Gate
	Writer   *routing.Writer
	// Pricer prices every row recorded. Nil is routing.Prices of Config.
	Pricer pricing.Pricer
	Logger *slog.Logger
}

// Request is an image to look at and who pays for it.
type Request struct {
	// CustomerID owns the request. It is what every statistic is keyed by.
	CustomerID string
	// Tags are the customer's own cost labels, recorded on the row this request writes.
	Tags routing.Tags
	// Target is a "provider/model" name or a capability shortcut. Empty means
	// DefaultTarget.
	Target string
	// Providers is a priority list of where to try, in the order given, and it wins over
	// Target when it holds anything.
	Providers []string
	// Overwrites are settings for one provider, keyed by provider name. One naming a
	// provider that is not a candidate is refused as the typo it almost certainly is.
	Overwrites map[string]json.RawMessage
	// Detection is what to look at. Its Overwrite is set per candidate from Overwrites.
	Detection detection.Request
}

// Found is what was detected, by whom, and what it cost.
type Found struct {
	// Provider and Model are who looked, or who failed to. Both are empty when nothing
	// got as far as a provider.
	Provider   string
	Model      string
	Detections []detection.Detection
	// CostMicros is millionths of a dollar, priced from the provider's configured rates.
	CostMicros int64
}

// Router selects a detection provider and looks with it.
type Router struct {
	*routing.Router[detection.Provider]
}

// New validates the options and returns a Router.
func New(options Options) (*Router, error) {
	core, err := routing.New(routing.Options[detection.Provider]{
		Modality: routing.Detection,
		Config:   options.Config,
		Registry: options.Registry,
		Store:    options.Store,
		Live:     options.Live,
		Gate:     options.Gate,
		Writer:   options.Writer,
		Pricer:   options.Pricer,
		Logger:   options.Logger,
	})
	if err != nil {
		return nil, err
	}
	return &Router{Router: core}, nil
}

// Detect looks at the image with the best candidate that will have it.
//
// A candidate that fails hands the request to the next one the target resolved to. Unlike
// a picture being drawn, a detection is cheap and over in a second, so asking the next
// vendor after one failed costs a few cents at most and is what a caller waiting on the
// answer wants.
//
// A failure is a *detection.Error. Any other error is a request that names nowhere this
// router can send it.
func (r *Router) Detect(ctx context.Context, request Request) (Found, error) {
	core := routing.Request{
		CustomerID:  request.CustomerID,
		Attribution: routing.AttributionFrom(ctx),
		Tags:        request.Tags,
		Target:      request.Target,
		Providers:   request.Providers,
		Terms:       request.Detection.Terms(),
	}
	if core.Target == "" && len(core.Providers) == 0 {
		core.Target = DefaultTarget
	}

	candidates, err := r.Candidates(ctx, core)
	if err != nil {
		return Found{}, err
	}
	for vendor := range request.Overwrites {
		if !slices.ContainsFunc(candidates, func(candidate routing.Candidate) bool { return candidate.Config.Provider == vendor }) {
			return Found{}, detection.Fail(detection.UnsupportedOption,
				fmt.Errorf("detectionrouter: there is an overwrite for %s, which nothing %s resolves to", vendor, where(core)))
		}
	}
	honouring := func(candidate routing.Candidate) bool { return candidate.Config.Supports(core.Terms) }
	if !slices.ContainsFunc(candidates, honouring) {
		return Found{}, detection.Fail(detection.UnsupportedOption,
			fmt.Errorf("detectionrouter: nothing %s resolves to can honour %s", where(core), named(core.Terms)))
	}

	var failures []error
	for _, candidate := range candidates {
		if !honouring(candidate) {
			continue
		}
		if err := ctx.Err(); err != nil {
			return Found{}, detection.Fail(detection.CodeOf(err), err)
		}

		// The list would win over the target, so it is cleared to ask this one alone.
		one := core
		one.Target = candidate.Config.Name()
		one.Providers = nil
		provider, config, err := r.Select(ctx, one)
		if err != nil {
			failures = append(failures, err)
			continue
		}

		asked := request.Detection
		asked.Overwrite = request.Overwrites[config.Provider]
		found, err := r.look(ctx, provider, config, core.Owner(), asked)
		if err == nil {
			return found, nil
		}
		if ctx.Err() != nil {
			return found, err
		}
		failures = append(failures, fmt.Errorf("%s: %w", config.Name(), err))
	}

	// A request every candidate refused for what it asked is the caller's to change, not a
	// provider that failed.
	code := detection.UnsupportedOption
	for _, failure := range failures {
		if detection.CodeOf(failure) != detection.UnsupportedOption {
			code = detection.ProviderFailed
		}
	}
	return Found{}, detection.Fail(code, errors.Join(failures...))
}

// look asks one provider for boxes and records what it cost.
func (r *Router) look(
	ctx context.Context,
	provider detection.Provider,
	config routing.ProviderConfig,
	owner routing.Owner,
	request detection.Request,
) (Found, error) {
	defer provider.Close()

	result, err := routing.Call(ctx, routing.Meter{
		Recorder: r.Recorder(), Config: config, Owner: owner,
		Code: func(err error) string { return string(detection.CodeOf(err)) },
	}, func(ctx context.Context) (detection.Result, error) {
		return provider.Detect(ctx, request)
	})
	found := Found{Provider: config.Provider, Model: config.Model}
	if err != nil {
		return found, err
	}
	for _, detected := range result.Detections {
		if !request.Keeps(detected.Label) {
			continue
		}
		if request.MinConfidence != nil && detected.Confidence < *request.MinConfidence {
			continue
		}
		if !request.Embeddings {
			detected.Embedding = nil
		}
		detected.Box = detected.Box.Clamped()
		found.Detections = append(found.Detections, detected)
	}
	found.CostMicros = r.Quote(config, result.Used()).CostMicros
	return found, nil
}

// where names what a request asked for, for an error about it.
func where(request routing.Request) string {
	if len(request.Providers) > 0 {
		return strings.Join(request.Providers, ", ")
	}
	return request.Target
}

func named(terms []options.Term) string {
	names := make([]string, 0, len(terms))
	for _, term := range terms {
		names = append(names, string(term))
	}
	return strings.Join(names, " and ")
}
