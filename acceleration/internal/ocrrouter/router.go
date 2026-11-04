// Package ocrrouter routes page reading and records what each page cost.
//
// Resolving a target, ranking candidates and failing over are generic and live in
// internal/routing. What this package adds is the OCR shape: which providers exist, how
// long one is given before the page goes to the next, and what counts as a unit of work,
// which is one page.
//
// Nothing is kept. The text goes back to the caller and the only thing written down is the
// stat row saying what the page cost and who read it.
package ocrrouter

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Registry is the set of OCR providers a build can construct.
type Registry = routing.Registry[ocr.Provider]

// DefaultTarget is where a request that names nowhere goes.
const DefaultTarget = "ocr-default"

// DefaultAttemptTimeout is how long one candidate has to read a page before it goes to the
// next. Mistral OCR reads one in a few seconds and a vision model in ten or twenty, so a
// candidate still going at this point is stuck rather than slow.
const DefaultAttemptTimeout = 45 * time.Second

// estimate is what a page may cost, set aside before it is read: a page for a model billed
// by the page, and a picture and a full transcription for one billed by the token. Each
// model's price rates only its own meters, so each reserves what it would bill.
var estimate = pricing.Quantities{pricing.Pages: 1, pricing.InputTokens: 2000, pricing.OutputTokens: 8192}

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
	// AttemptTimeout is how long one candidate has. Zero is DefaultAttemptTimeout.
	AttemptTimeout time.Duration
	Logger         *slog.Logger
}

// Request is a page to read and who pays for it.
type Request struct {
	// CustomerID owns the request. It is what every statistic is keyed by.
	CustomerID string
	// Tags are the customer's own cost labels, recorded on the row this page writes.
	Tags routing.Tags
	// DataPolicy is what the caller requires of what happens to the page afterwards, on
	// top of the floor the customer's policies set. A page of a document can hold personal
	// data, so a caller may ask for no training, a retention or a region.
	DataPolicy options.DataPolicy
	// Target is a "provider/model" name or a shortcut. Empty means DefaultTarget.
	Target string
	// Providers is a priority list of where to try, in the order given, and it wins over
	// Target when it holds anything.
	Providers []string
	// Page is what to read.
	Page ocr.Request
}

// Reading is what a page says, who read it, and what it cost.
type Reading struct {
	// Provider and Model are who read it, or who failed to. Both are empty when nothing
	// got as far as a provider.
	Provider string
	Model    string
	Markdown string
	// CostMicros is millionths of a dollar, priced from the provider's configured rates.
	CostMicros int64
}

// Router selects an OCR provider and reads with it.
type Router struct {
	*routing.Router[ocr.Provider]
	attempt time.Duration
}

// New validates the options and returns a Router.
func New(options Options) (*Router, error) {
	core, err := routing.New(routing.Options[ocr.Provider]{
		Modality: routing.OCR,
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
	attempt := options.AttemptTimeout
	if attempt <= 0 {
		attempt = DefaultAttemptTimeout
	}
	return &Router{Router: core, attempt: attempt}, nil
}

// Read reads the page with the best candidate that will have it.
//
// A candidate that fails or runs out of its time hands the page to the next one the target
// resolved to. A page is cheap and the caller is waiting on it, so asking the next model
// after one failed costs a fraction of a cent and is what the caller wants. Only models
// that can read a picture are ever candidates: the config lists OCR models and vision
// language models, and a language model that cannot see is refused when it is built.
//
// A failure is an *ocr.Error. Any other error is a request that names nowhere this router
// can send it.
func (r *Router) Read(ctx context.Context, request Request) (Reading, error) {
	core := routing.Request{
		CustomerID:  request.CustomerID,
		Attribution: routing.AttributionFrom(ctx),
		Spent:       routing.SpentFrom(ctx),
		Tags:        request.Tags,
		DataPolicy:  request.DataPolicy,
		Target:      request.Target,
		Providers:   request.Providers,
	}
	if core.Target == "" && len(core.Providers) == 0 {
		core.Target = DefaultTarget
	}

	candidates, err := r.Candidates(ctx, core)
	if err != nil {
		return Reading{}, err
	}

	var failures []error
	for _, candidate := range candidates {
		if err := ctx.Err(); err != nil {
			return Reading{}, ocr.Fail(ocr.CodeOf(err), err)
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

		reading, err := r.read(ctx, provider, config, core.Owner(), request.Page)
		if err == nil {
			return reading, nil
		}
		if ctx.Err() != nil || routing.ReservationRefused(err) {
			return reading, err
		}
		failures = append(failures, fmt.Errorf("%s: %w", config.Name(), err))
	}
	if len(failures) == 0 {
		return Reading{}, ocr.Fail(ocr.ProviderFailed, fmt.Errorf("ocrrouter: nothing %s resolves to can read a page", where(core)))
	}
	return Reading{}, ocr.Fail(ocr.ProviderFailed, errors.Join(failures...))
}

// read asks one provider for the page, within its own time, and records what it cost.
func (r *Router) read(
	ctx context.Context,
	provider ocr.Provider,
	config routing.ProviderConfig,
	owner routing.Owner,
	page ocr.Request,
) (Reading, error) {
	defer provider.Close()

	attempt, cancel := context.WithTimeout(ctx, r.attempt)
	defer cancel()
	result, err := routing.Call(attempt, routing.Meter{
		Recorder: r.Recorder(), Config: config, Owner: owner,
		Code:     func(err error) string { return string(ocr.CodeOf(err)) },
		Estimate: estimate,
	}, func(ctx context.Context) (ocr.Result, error) {
		return provider.Read(ctx, page)
	})
	reading := Reading{Provider: config.Provider, Model: config.Model}
	if err != nil {
		return reading, err
	}
	reading.Markdown = result.Markdown
	reading.CostMicros = r.Quote(config, result.Used()).CostMicros
	return reading, nil
}

// where names what a request asked for, for an error about it.
func where(request routing.Request) string {
	if len(request.Providers) > 0 {
		return strings.Join(request.Providers, ", ")
	}
	return request.Target
}
