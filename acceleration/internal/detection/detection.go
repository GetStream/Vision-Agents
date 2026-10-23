// Package detection is the object detection contract: one image in, boxes out.
//
// Like image generation, a detection is one request and one answer, so the contract is one
// method. What it adds is a common shape for the boxes. Every vendor draws a box its own
// way (a centre and a size in pixels, two corners in pixels, two corners as fractions), and
// what comes back from here is always two corners as fractions of the image, so a caller
// can draw a box without knowing how large the picture was.
//
// The router never sees a call's video. A worker that holds the frames samples them and
// sends them here one at a time.
package detection

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const (
	// MaxBytes caps the image. Ultralytics' shared API takes about 4.5 MB, so a frame is
	// expected to be a JPEG well under that.
	MaxBytes = 4 << 20
	// MaxSide caps either side, checked before decoding.
	MaxSide = 4096
	// MaxClasses caps how many words one request may look for. Moondream looks for one
	// word a call, so each is a round trip.
	MaxClasses = 8
)

// Request is one image to look at.
type Request struct {
	// Image is a JPEG or PNG.
	Image []byte
	// MediaType is image/jpeg or image/png, read off the image by Validate.
	MediaType string
	// Width and Height are the image's size, read off it by Validate.
	Width  int
	Height int
	// Classes are the labels to keep. For an open-vocabulary model they are what it looks
	// for; for a fixed-class model they filter what it found. Empty keeps everything a
	// fixed-class model found, and is refused by an open-vocabulary one, which has to be
	// told what to look for.
	Classes []string
	// MinConfidence drops boxes scored below it. Nil keeps the model's own default.
	MinConfidence *float64
	// Embeddings asks for a vector on every box.
	Embeddings bool
	// Overwrite is this provider's own settings, which the vocabulary has no word for, as
	// the caller sent them. A provider decodes the fields it knows.
	Overwrite json.RawMessage
}

// Validate decodes the image header, fills in its type and size, and checks the rest.
func (r *Request) Validate() error {
	if len(r.Image) == 0 {
		return stack.Wrap(errors.New("detection: there is no image"))
	}
	if len(r.Image) > MaxBytes {
		return stack.Wrap(fmt.Errorf("detection: the image is larger than %d MB", MaxBytes>>20))
	}
	config, format, err := image.DecodeConfig(bytes.NewReader(r.Image))
	if err != nil {
		return stack.Wrap(fmt.Errorf("detection: the image does not decode: %w", err))
	}
	if format != "jpeg" && format != "png" {
		return stack.Wrap(fmt.Errorf("detection: the image is %s, not jpeg or png", format))
	}
	if config.Width < 1 || config.Height < 1 || config.Width > MaxSide || config.Height > MaxSide {
		return stack.Wrap(fmt.Errorf("detection: the image is %dx%d, and neither side may exceed %d", config.Width, config.Height, MaxSide))
	}
	r.MediaType, r.Width, r.Height = "image/"+format, config.Width, config.Height

	if len(r.Classes) > MaxClasses {
		return stack.Wrap(fmt.Errorf("detection: at most %d classes", MaxClasses))
	}
	for _, class := range r.Classes {
		if strings.TrimSpace(class) == "" {
			return stack.Wrap(errors.New("detection: a class cannot be empty"))
		}
	}
	if r.MinConfidence != nil && (*r.MinConfidence < 0 || *r.MinConfidence > 1) {
		return stack.Wrap(errors.New("detection: min_confidence must be between 0 and 1"))
	}
	return nil
}

// Terms is what the request asks of a model beyond finding boxes. A model that did not
// declare one of them is not asked.
func (r Request) Terms() []options.Term {
	var asked []options.Term
	if r.MinConfidence != nil {
		asked = append(asked, options.Confidence)
	}
	if r.Embeddings {
		asked = append(asked, options.Embeddings)
	}
	return asked
}

// Keeps reports whether a label is one the request asked for, ignoring case.
func (r Request) Keeps(label string) bool {
	if len(r.Classes) == 0 {
		return true
	}
	return slices.ContainsFunc(r.Classes, func(class string) bool { return strings.EqualFold(class, label) })
}

// Box is a rectangle as fractions of the image, 0 at the top left and 1 at the bottom
// right.
type Box struct {
	XMin float64
	YMin float64
	XMax float64
	YMax float64
}

// Clamped is the box held inside the image, which a vendor rounding a pixel edge can
// overshoot.
func (b Box) Clamped() Box {
	clamp := func(v float64) float64 { return min(max(v, 0), 1) }
	return Box{XMin: clamp(b.XMin), YMin: clamp(b.YMin), XMax: clamp(b.XMax), YMax: clamp(b.YMax)}
}

// FromCenter is a box given as a centre and a size in pixels, which is how Roboflow
// draws one.
func FromCenter(x, y, width, height float64, imageWidth, imageHeight int) Box {
	w, h := float64(imageWidth), float64(imageHeight)
	return Box{
		XMin: (x - width/2) / w,
		YMin: (y - height/2) / h,
		XMax: (x + width/2) / w,
		YMax: (y + height/2) / h,
	}.Clamped()
}

// Detection is one thing found.
type Detection struct {
	Label string
	// Confidence is the model's score from 0 to 1. A model that scores nothing reports 1.
	Confidence float64
	Box        Box
	// Embedding is a vector describing what is in the box. Nil unless asked for.
	Embedding []float32
}

// Result is what a provider found.
type Result struct {
	Detections []Detection
	// Billed is how many images the vendor charged for, which is more than one for a
	// vendor asked once per class. Zero means one.
	Billed int
}

// Used is the images the vendor charged for.
func (r Result) Used() pricing.Quantities {
	return pricing.Quantities{pricing.Images: int64(max(r.Billed, 1))}
}

// Code is why a detection failed, in the few words a caller can act on.
type Code string

const (
	// UnsupportedOption is a class, setting or overwrite nothing asked could honour.
	UnsupportedOption Code = "unsupported_option"
	// ProviderFailed is everything else a provider did wrong.
	ProviderFailed Code = "provider_failed"
	// Timeout is a detection that ran out of time.
	Timeout Code = "timeout"
	// Cancelled is a caller that stopped waiting.
	Cancelled Code = "cancelled"
)

// Error is a failed detection and why.
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

// Provider is a detection provider. It satisfies routing.Provider.
type Provider interface {
	// Detect looks at the image. A failure is an *Error.
	Detect(ctx context.Context, request Request) (Result, error)
	Start(ctx context.Context) error
	Close() error
	// Provider is the stable provider name used in stats, e.g. "moondream".
	Provider() string
	// Model is the model identifier used in stats, e.g. "moondream3".
	Model() string
}
