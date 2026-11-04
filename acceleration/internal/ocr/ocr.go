// Package ocr is the page reading contract: a picture of one page in, its text out as
// Markdown.
//
// Like detection, a reading is one request and one answer, so the contract is one method.
// A page is the unit because it is what OCR models bill by and what a citation points at:
// a document is read a page at a time, and each page is its own row, priced and failed
// over on its own.
//
// Two kinds of model read a page. An OCR model such as Mistral OCR is built for it and
// billed by the page; a vision language model is asked to transcribe it and billed by the
// token. Both answer in Markdown, headings, lists and tables kept.
package ocr

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const (
	// MaxBytes caps one page. A page drawn at 160 dpi as a JPEG is well under a megabyte,
	// and a photo of one from a phone is a few.
	MaxBytes = 10 << 20
	// MaxSide caps either side of a page, checked before it is decoded so a small file
	// claiming to be enormous is refused rather than inflated.
	MaxSide = 10000
)

// Request is one page to read.
type Request struct {
	// Image is a JPEG or PNG of the page.
	Image []byte
	// MediaType is image/jpeg or image/png, read off the image by Validate.
	MediaType string
}

// Validate reports whether the page is a picture that can be read, and says what kind.
func (r *Request) Validate() error {
	if len(r.Image) == 0 {
		return stack.Wrap(errors.New("ocr: there is no page to read"))
	}
	if len(r.Image) > MaxBytes {
		return stack.Wrap(fmt.Errorf("ocr: a page may be at most %d bytes", MaxBytes))
	}
	config, format, err := image.DecodeConfig(bytes.NewReader(r.Image))
	if err != nil {
		return stack.Wrap(fmt.Errorf("ocr: the page is not a JPEG or PNG: %w", err))
	}
	if format != "jpeg" && format != "png" {
		return stack.Wrap(fmt.Errorf("ocr: the page is %s, not a JPEG or PNG", format))
	}
	if config.Width < 1 || config.Height < 1 || config.Width > MaxSide || config.Height > MaxSide {
		return stack.Wrap(fmt.Errorf("ocr: the page is %dx%d, and neither side may be over %d", config.Width, config.Height, MaxSide))
	}
	r.MediaType = "image/" + format
	return nil
}

// Result is what a provider read.
type Result struct {
	// Markdown is the page's text. Empty is a page with no text on it, which is an answer
	// rather than a failure.
	Markdown string
	// Pages is how many pages the provider billed, which is one unless it says otherwise.
	Pages int64
	// InputTokens and OutputTokens are what a vision language model was billed for reading
	// the page, and zero for a model billed by the page.
	InputTokens  int64
	OutputTokens int64
}

// Used is the page and the tokens it took, so the row is priced by whichever of them its
// model's price rates.
func (r Result) Used() pricing.Quantities {
	used := pricing.Quantities{pricing.Pages: max(r.Pages, 1)}
	if r.InputTokens > 0 {
		used[pricing.InputTokens] = r.InputTokens
	}
	if r.OutputTokens > 0 {
		used[pricing.OutputTokens] = r.OutputTokens
	}
	return used
}

// Code is why a reading failed, in the few words a caller can act on.
type Code string

const (
	// ProviderFailed is everything a provider did wrong.
	ProviderFailed Code = "provider_failed"
	// Timeout is a reading that ran out of time.
	Timeout Code = "timeout"
	// Cancelled is a caller that stopped waiting.
	Cancelled Code = "cancelled"
)

// Error is a failed reading and why.
type Error struct {
	Code Code
	Err  error
}

func (e *Error) Error() string { return e.Err.Error() }

func (e *Error) Unwrap() error { return e.Err }

// Fail wraps an error in a code.
func Fail(code Code, err error) *Error {
	return &Error{Code: code, Err: err}
}

// CodeOf is why err happened. A context that ran out is a timeout or a cancellation
// whatever the provider said about it, since that is what the caller did.
func CodeOf(err error) Code {
	switch {
	case errors.Is(err, context.DeadlineExceeded):
		return Timeout
	case errors.Is(err, context.Canceled):
		return Cancelled
	}
	var failure *Error
	if errors.As(err, &failure) {
		return failure.Code
	}
	return ProviderFailed
}

// Provider reads pages.
//
// It satisfies routing.Provider, so a reading is routed, ranked and billed the way the
// other modalities are. A provider missing its key fails when it is built, which is where
// the router picks the next candidate.
type Provider interface {
	// Read reads one page. A failure is an *Error.
	Read(ctx context.Context, request Request) (Result, error)
	Start(ctx context.Context) error
	Close() error
	// Provider is the stable provider name used in stats, e.g. "mistral".
	Provider() string
	// Model is the model identifier used in stats, e.g. "mistral-ocr-4-1".
	Model() string
}
