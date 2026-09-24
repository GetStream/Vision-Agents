// Package imagegen is the image generation contract: a prompt in, pictures out.
//
// It is named for what it does rather than image, which is the standard library's package
// for what comes back. Like search, a generation is one request and one answer: there is
// nothing to stream and no session to hold, so the contract is one method.
//
// What a provider hands back is bytes, never a link. A vendor's URL points at storage it
// controls and expires on a schedule it sets, and a caller who was handed one would have
// to trust both; the picture itself has neither problem. Every picture is decoded before
// it is returned, so what arrives is known to be an image rather than taken on trust.
package imagegen

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"regexp"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

const (
	// MaxImages caps how many pictures one request may ask for. Every one of them comes
	// back in the same response, so the cap is what keeps a response to a size a caller
	// can hold.
	MaxImages = 4
	// MaxBytes caps one picture.
	MaxBytes = 10 << 20
	// MaxSide caps either side of one picture, which is checked before the picture is
	// decoded so a small file claiming to be enormous is refused rather than inflated.
	MaxSide = 4096
)

// Output formats a caller may ask for. Both decode with the standard library, which is
// what lets every picture be checked before it is returned.
const (
	FormatPNG  = "png"
	FormatJPEG = "jpeg"
)

// aspectRatioPattern is what an aspect ratio looks like: two whole numbers and a colon.
var aspectRatioPattern = regexp.MustCompile(`^[1-9][0-9]*:[1-9][0-9]*$`)

// Request is one prompt to draw.
type Request struct {
	// Prompt is what to draw, in the caller's own words.
	Prompt string
	// NegativePrompt is what to keep out of the picture.
	NegativePrompt string
	// Width and Height are the size asked for, in pixels. Zero leaves it to the aspect
	// ratio or to the model.
	Width  int
	Height int
	// AspectRatio is a shape such as 16:9, for a model that is asked for a shape rather
	// than a size.
	AspectRatio string
	// N is how many pictures to draw. Zero means one.
	N int
	// Seed makes a picture reproducible, for a model that reads one.
	Seed *int64
	// Format is png or jpeg. Empty leaves it to the model.
	Format string
}

// Count is how many pictures the request asks for.
func (r Request) Count() int {
	if r.N <= 0 {
		return 1
	}
	return r.N
}

// Validate reports whether the request asks for something that can be drawn at all. Which
// sizes and shapes a model accepts is the model's business, and is refused by it.
func (r Request) Validate() error {
	if strings.TrimSpace(r.Prompt) == "" {
		return errors.New("imagegen: there is nothing to draw")
	}
	if r.N < 0 || r.N > MaxImages {
		return fmt.Errorf("imagegen: n must be between 1 and %d", MaxImages)
	}
	if (r.Width == 0) != (r.Height == 0) || r.Width < 0 || r.Height < 0 {
		return errors.New("imagegen: a size needs a width and a height")
	}
	if r.Width > MaxSide || r.Height > MaxSide {
		return fmt.Errorf("imagegen: neither side may be larger than %d pixels", MaxSide)
	}
	if r.AspectRatio != "" && !aspectRatioPattern.MatchString(r.AspectRatio) {
		return fmt.Errorf("imagegen: aspect ratio %q is not two numbers such as 16:9", r.AspectRatio)
	}
	if r.Format != "" && r.Format != FormatPNG && r.Format != FormatJPEG {
		return fmt.Errorf("imagegen: format %q is not png or jpeg", r.Format)
	}
	if r.Seed != nil && *r.Seed < 0 {
		return errors.New("imagegen: a seed cannot be negative")
	}
	return nil
}

// Terms is what the request asks of a model beyond a prompt. A model that did not declare
// one of them is not asked.
func (r Request) Terms() []options.Term {
	var asked []options.Term
	if r.Width != 0 {
		asked = append(asked, options.Size)
	}
	if r.AspectRatio != "" {
		asked = append(asked, options.AspectRatio)
	}
	if r.Seed != nil {
		asked = append(asked, options.Seed)
	}
	if r.NegativePrompt != "" {
		asked = append(asked, options.NegativePrompt)
	}
	if r.Format != "" {
		asked = append(asked, options.Format)
	}
	return asked
}

// Image is one picture, decoded and known to be what it says it is.
type Image struct {
	Data []byte
	// MediaType is image/png or image/jpeg, read off the picture rather than the vendor's
	// label for it.
	MediaType string
	Width     int
	Height    int
	// Seed is the seed the provider reports, which draws the same picture again from the
	// same prompt. Nil when it reports none.
	Seed *int64
}

// Verify decodes a picture a provider returned and says what it is.
func Verify(data []byte) (Image, error) {
	if len(data) == 0 {
		return Image{}, errors.New("imagegen: the picture is empty")
	}
	if len(data) > MaxBytes {
		return Image{}, fmt.Errorf("imagegen: the picture is larger than %d bytes", MaxBytes)
	}
	config, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return Image{}, fmt.Errorf("imagegen: the picture does not decode: %w", err)
	}
	if format != FormatPNG && format != FormatJPEG {
		return Image{}, fmt.Errorf("imagegen: the picture is %s, not png or jpeg", format)
	}
	if config.Width < 1 || config.Height < 1 || config.Width > MaxSide || config.Height > MaxSide {
		return Image{}, fmt.Errorf("imagegen: the picture is %dx%d", config.Width, config.Height)
	}
	if _, _, err := image.Decode(bytes.NewReader(data)); err != nil {
		return Image{}, fmt.Errorf("imagegen: the picture does not decode: %w", err)
	}
	return Image{Data: data, MediaType: "image/" + format, Width: config.Width, Height: config.Height}, nil
}

// Code is why a generation failed, in the few words a caller can act on.
type Code string

const (
	// ContentFiltered is a prompt or a picture the provider's safety filter refused. It is
	// never retried elsewhere: asking the next vendor is shopping for a laxer filter.
	ContentFiltered Code = "content_filtered"
	// UnsupportedOption is a size, shape or setting nothing asked could honour.
	UnsupportedOption Code = "unsupported_option"
	// ProviderFailed is everything else a provider did wrong.
	ProviderFailed Code = "provider_failed"
	// Timeout is a generation that ran out of time.
	Timeout Code = "timeout"
	// Cancelled is a caller that stopped waiting.
	Cancelled Code = "cancelled"
)

// Error is a failed generation and why.
type Error struct {
	Code Code
	// Accepted says the provider had taken the job before it failed. A job it took may
	// have been billed and may still be drawing, so it is not asked of anybody else.
	Accepted bool
	Err      error
}

func (e *Error) Error() string { return e.Err.Error() }

func (e *Error) Unwrap() error { return e.Err }

// Fail wraps an error in a code.
func Fail(code Code, accepted bool, err error) *Error {
	return &Error{Code: code, Accepted: accepted, Err: err}
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

// Accepted reports whether a provider had taken the job before err.
func Accepted(err error) bool {
	var failure *Error
	return errors.As(err, &failure) && failure.Accepted
}

// Result is what a provider drew.
type Result struct {
	Images []Image
}

// Provider is an image generation provider.
//
// It satisfies routing.Provider, so a generation is routed, ranked and billed the way the
// other modalities are. Start and Close are what that contract asks for rather than
// anything a picture needs: a provider missing its key fails when it is built, which is
// where the router picks the next candidate.
type Provider interface {
	// Generate draws the request. A failure is an *Error, so the router can tell one the
	// provider never accepted, which is worth asking elsewhere, from one it did.
	Generate(ctx context.Context, request Request) (Result, error)
	Start(ctx context.Context) error
	Close() error
	// Provider is the stable provider name used in stats, e.g. "fal".
	Provider() string
	// Model is the model identifier used in stats, e.g. "alibaba/qwen-image-3/text-to-image".
	Model() string
}
