// Package imagerouter routes image generation and records what each request cost.
//
// Resolving a target, ranking candidates and failing over are generic and live in
// internal/routing. What this package adds is the image shape: which providers exist, when
// a failed generation is worth asking of the next candidate, and what counts as a unit of
// work, which is one request however many pictures it asked for.
//
// Nothing is kept. The pictures go back to the caller and the only thing written down is
// the stat row saying what they cost.
package imagerouter

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Registry is the set of image providers a build can construct.
type Registry = routing.Registry[imagegen.Provider]

// DefaultTarget is where a request that names nowhere goes: the fast tier, since whoever
// asked is usually waiting on the answer.
const DefaultTarget = "image-fast"

// Options configures a Router. Store and Live are optional: without them the router still
// routes, it just stops recording.
type Options struct {
	Config   routing.ModalityConfig
	Registry *Registry
	Store    *store.Store
	Live     *live.Client
	Gate     routing.Gate
	Logger   *slog.Logger
}

// Request is a prompt to draw and who pays for it.
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
	// Image is what to draw. A size, shape, seed, negative prompt or format named here
	// narrows the candidates to the models that declared it.
	Image imagegen.Request
}

// Generation is what was drawn, by whom, and what it cost.
type Generation struct {
	// Provider and Model are who drew it, or who refused to. Both are empty when nothing
	// got as far as a provider.
	Provider string
	Model    string
	Images   []imagegen.Image
	// CostMicros is millionths of a dollar, priced from the provider's configured rates.
	// A generation that failed costs nothing.
	CostMicros int64
}

// Router selects an image provider and draws with it.
type Router struct {
	*routing.Router[imagegen.Provider]
}

// New validates the options and returns a Router.
func New(options Options) (*Router, error) {
	core, err := routing.New(routing.Options[imagegen.Provider]{
		Modality: routing.Image,
		Config:   options.Config,
		Registry: options.Registry,
		Store:    options.Store,
		Live:     options.Live,
		Gate:     options.Gate,
		Logger:   options.Logger,
	})
	if err != nil {
		return nil, err
	}
	return &Router{Router: core}, nil
}

// Generate draws the request with the best candidate that will have it.
//
// A candidate that fails before it accepted the job - no key, a refusal at the door, a
// size it cannot draw - hands the request to the next one the target resolved to, in the
// order it resolved to them. Once a provider has accepted the job nothing else is asked:
// it may have billed and may still be drawing, and a second vendor's attempt would be paid
// for twice. A picture a safety filter refused is not asked again either, since asking
// the next vendor is shopping for a laxer filter.
//
// A failure is an *imagegen.Error. Any other error is a request that names nowhere this
// router can send it.
func (r *Router) Generate(ctx context.Context, request Request) (Generation, error) {
	core := routing.Request{
		CustomerID: request.CustomerID,
		Tags:       request.Tags,
		Target:     request.Target,
		Providers:  request.Providers,
		Terms:      request.Image.Terms(),
	}
	if core.Target == "" && len(core.Providers) == 0 {
		core.Target = DefaultTarget
	}

	candidates, err := r.Candidates(ctx, core)
	if err != nil {
		return Generation{}, err
	}
	honouring := func(candidate routing.Candidate) bool { return candidate.Config.Supports(core.Terms) }
	if !slices.ContainsFunc(candidates, honouring) {
		return Generation{}, imagegen.Fail(imagegen.UnsupportedOption, false,
			fmt.Errorf("imagerouter: nothing %s resolves to can honour %s", where(core), named(core.Terms)))
	}

	var failures []error
	for _, candidate := range candidates {
		if !honouring(candidate) {
			continue
		}
		if err := ctx.Err(); err != nil {
			return Generation{}, imagegen.Fail(imagegen.CodeOf(err), false, err)
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

		generation, err := r.draw(ctx, provider, config, core.Owner(), request.Image)
		if err == nil {
			return generation, nil
		}
		if imagegen.Accepted(err) || imagegen.CodeOf(err) == imagegen.ContentFiltered || ctx.Err() != nil {
			return generation, err
		}
		failures = append(failures, fmt.Errorf("%s: %w", config.Name(), err))
	}

	// A request every candidate refused for its size or shape is the caller's to change,
	// not a provider that failed.
	code := imagegen.UnsupportedOption
	for _, failure := range failures {
		if imagegen.CodeOf(failure) != imagegen.UnsupportedOption {
			code = imagegen.ProviderFailed
		}
	}
	return Generation{}, imagegen.Fail(code, false, errors.Join(failures...))
}

// draw asks one provider for the pictures and records what they cost.
func (r *Router) draw(
	ctx context.Context,
	provider imagegen.Provider,
	config routing.ProviderConfig,
	owner routing.Owner,
	request imagegen.Request,
) (Generation, error) {
	defer provider.Close()

	started := time.Now()
	result, err := provider.Generate(ctx, request)

	// A failed generation is not billed, so the row records what was waited for without
	// charging for it.
	var usage routing.Usage
	if err == nil {
		usage.Images = int64(len(result.Images))
		for _, picture := range result.Images {
			usage.Pixels += int64(picture.Width) * int64(picture.Height)
		}
	}
	stat := routing.Stat{
		Owner:     owner,
		StartedAt: started.UTC(),
		LatencyMs: routing.MsSince(started),
		Usage:     usage,
		Success:   err == nil,
	}
	if err != nil {
		stat.ErrorCode = string(imagegen.CodeOf(err))
		stat.ErrorMessage = err.Error()
	}
	r.Recorder().Record(config, stat)

	generation := Generation{Provider: config.Provider, Model: config.Model}
	if err != nil {
		return generation, err
	}
	generation.Images = result.Images
	generation.CostMicros = config.Price.CostMicros(usage)
	return generation, nil
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
