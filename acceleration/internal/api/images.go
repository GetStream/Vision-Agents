package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// imageDeadline bounds one generation. A queued provider can hold a job for minutes, and
// a caller waiting on one response is owed an answer rather than a connection held open
// until something upstream gives up.
const imageDeadline = 240 * time.Second

// errNoImages is what the image path says on a deployment that does not generate images.
var errNoImages = notConfigured("this deployment does not generate images")

// generateImage draws pictures from a prompt and returns them.
//
// It is answered inline and nothing is kept: the pictures are in the response and the
// only thing written down is what they cost. A generation that reached a provider is a 200
// whether or not it drew, the way an inline recording is a 202 whether or not it spoke,
// since the failure is part of the answer rather than something wrong with the request.
// A caller that hangs up cancels the job at the provider, because the request's context
// is what the provider waits on.
func (s *Server) generateImage(ctx context.Context, request *generateImageRequest) (*generateImageResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.streams == nil || s.streams.Image == nil {
		return nil, errNoImages
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}

	drawing, err := imageRequestOf(request.Body)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	if err := drawing.Validate(); err != nil {
		return nil, invalidRequest(err.Error())
	}
	tags := tagsSent(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return nil, invalidRequest(err.Error())
	}

	sent := value(request.Body.Options)
	ctx, cancel := context.WithTimeout(ctx, imageDeadline)
	defer cancel()
	generation, err := s.streams.Image.Generate(ctx, imagerouter.Request{
		CustomerID: customerID,
		Tags:       tags,
		Target:     value(sent.Target),
		Providers:  value(sent.Providers),
		Image:      drawing,
	})
	var failure *imagegen.Error
	if err != nil && !errors.As(err, &failure) {
		return nil, invalidRequest(err.Error())
	}
	return &generateImageResponse{Body: imageGenerationOf(generation, err)}, nil
}

// imageRequestOf reads what to draw out of the request.
func imageRequestOf(body *ImageGenerationRequest) (imagegen.Request, error) {
	sent := value(body.Options)
	if sent.N != nil && *sent.N < 1 {
		return imagegen.Request{}, stack.Wrap(fmt.Errorf("n must be between 1 and %d", imagegen.MaxImages))
	}
	drawing := imagegen.Request{
		Prompt:         body.Prompt,
		NegativePrompt: value(sent.NegativePrompt),
		AspectRatio:    value(sent.AspectRatio),
		N:              count(sent.N),
		Seed:           sent.Seed,
		Format:         string(value(sent.OutputFormat)),
	}
	if size := value(sent.Size); size != "" {
		width, height, _ := strings.Cut(size, "x")
		w, widthErr := strconv.Atoi(width)
		h, heightErr := strconv.Atoi(height)
		if widthErr != nil || heightErr != nil || w < 1 || h < 1 {
			return imagegen.Request{}, stack.Wrap(fmt.Errorf("size %q is not a width and a height such as 1024x1024", size))
		}
		drawing.Width, drawing.Height = w, h
	}
	return drawing, nil
}

// imageGenerationOf renders a generation for the wire, and why it failed when it did.
func imageGenerationOf(generation imagerouter.Generation, failure error) ImageGeneration {
	rendered := ImageGeneration{
		Id:         "img_" + uuid.NewString(),
		Status:     ImageGenerationStatusCompleted,
		Provider:   optional(generation.Provider),
		Model:      optional(generation.Model),
		Images:     make([]GeneratedImage, 0, len(generation.Images)),
		CostMicros: generation.CostMicros,
	}
	for _, picture := range generation.Images {
		rendered.Images = append(rendered.Images, GeneratedImage{
			Data:      picture.Data,
			MediaType: GeneratedImageMediaType(picture.MediaType),
			Width:     picture.Width,
			Height:    picture.Height,
			Seed:      picture.Seed,
		})
	}
	if failure != nil {
		code := ImageErrorCode(imagegen.CodeOf(failure))
		rendered.Status = ImageGenerationStatusFailed
		rendered.ErrorCode = &code
		rendered.Error = optional(failure.Error())
	}
	return rendered
}

// registerImages declares the operations served in images.go.
func (s *Server) registerImages(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "generateImage",
		Method:      http.MethodPost,
		Path:        "/v1/image/generations",
		Summary:     "Draw pictures from a prompt, and return them",
		Description: "Routed like search: a target or a priority list picks the model, failover and billing " +
			"work as they do everywhere else, and one request is one stat row counting its pictures. " +
			"The pictures come back in the response as bytes, never as a link, and nothing is " +
			"stored, so the id cannot be fetched again.\n" +
			"The request is answered when the pictures are drawn, within 240 seconds; a caller that " +
			"hangs up cancels the job at the provider. A generation that got as far as a provider " +
			"answers 200 whether it drew or not: a failed one carries status failed, an error_code " +
			"and the error, and costs nothing. A request that could not be routed at all, or that " +
			"asks for something no model could draw, is a 400.\n" +
			"A failed generation is asked of the next candidate only when the provider never " +
			"accepted the job, and never after a safety filter refused it, since asking the next " +
			"vendor is shopping for a laxer filter.",
		Responses: map[string]*huma.Response{
			"200": {Description: "What was drawn, or why nothing was"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.generateImage)
}

type generateImageRequest struct {
	Body *ImageGenerationRequest `required:"true"`
}

type generateImageResponse struct {
	Body ImageGeneration
}

// GeneratedImage is the GeneratedImage schema.
type GeneratedImage struct {
	Data      []byte                  `json:"data" doc:"The picture, base64. Decoded and checked before it was returned, and never more than 10 MiB." format:"byte" nullable:"false"`
	Height    int                     `json:"height"`
	MediaType GeneratedImageMediaType `json:"media_type" doc:"What the picture is, read off the picture itself rather than the provider's label." enum:"image/png,image/jpeg"`
	Seed      *int64                  `json:"seed,omitempty" doc:"The seed the provider reports, which draws the same picture again. Absent when it reports none."`
	Width     int                     `json:"width"`
}

func (*GeneratedImage) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["height"].Format = ""
	schema.Properties["width"].Format = ""
	return schema
}

// GeneratedImageMediaType is the GeneratedImageMediaType schema.
type GeneratedImageMediaType string

// Defines values for GeneratedImageMediaType.
const (
	Imagejpeg GeneratedImageMediaType = "image/jpeg"
	Imagepng  GeneratedImageMediaType = "image/png"
)

// Valid indicates whether the value is a known member of the GeneratedImageMediaType enum.
func (e GeneratedImageMediaType) Valid() bool {
	switch e {
	case Imagejpeg:
		return true
	case Imagepng:
		return true
	default:
		return false
	}
}

// ImageErrorCode Why a generation drew nothing, absent when it completed. content_filtered is a safety filter refusing the prompt or the picture; unsupported_option a size, shape or setting no candidate could honour; provider_failed anything else a provider did wrong; timeout the 240 seconds running out; cancelled the caller hanging up.
type ImageErrorCode string

// Defines values for ImageErrorCode.
const (
	ImageErrorCodeCancelled         ImageErrorCode = "cancelled"
	ImageErrorCodeContentFiltered   ImageErrorCode = "content_filtered"
	ImageErrorCodeProviderFailed    ImageErrorCode = "provider_failed"
	ImageErrorCodeTimeout           ImageErrorCode = "timeout"
	ImageErrorCodeUnsupportedOption ImageErrorCode = "unsupported_option"
)

// Valid indicates whether the value is a known member of the ImageErrorCode enum.
func (e ImageErrorCode) Valid() bool {
	switch e {
	case ImageErrorCodeCancelled:
		return true
	case ImageErrorCodeContentFiltered:
		return true
	case ImageErrorCodeProviderFailed:
		return true
	case ImageErrorCodeTimeout:
		return true
	case ImageErrorCodeUnsupportedOption:
		return true
	default:
		return false
	}
}

func (ImageErrorCode) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ImageErrorCode", "Why a generation drew nothing, absent when it completed. content_filtered is a safety filter refusing the prompt or the picture; unsupported_option a size, shape or setting no candidate could honour; provider_failed anything else a provider did wrong; timeout the 240 seconds running out; cancelled the caller hanging up.", "content_filtered", "unsupported_option", "provider_failed", "timeout", "cancelled")
}

// ImageGeneration is the ImageGeneration schema.
type ImageGeneration struct {
	CostMicros int64                 `json:"cost_micros" doc:"Millionths of a dollar, priced per picture or per megapixel from what came back. Zero when it failed."`
	Error      *string               `json:"error,omitempty" doc:"What went wrong, in words. Absent when the generation completed."`
	ErrorCode  *ImageErrorCode       `json:"error_code,omitempty"`
	Id         string                `json:"id" doc:"This response's own id, for logs. Nothing is stored under it." example:"img_1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed"`
	Images     []GeneratedImage      `json:"images" doc:"The pictures, as many as were asked for. Empty when the generation failed." nullable:"false"`
	Model      *string               `json:"model,omitempty" example:"alibaba/qwen-image-3/text-to-image"`
	Provider   *string               `json:"provider,omitempty" doc:"Who drew it, or who refused to. Absent when nothing got as far as a provider." example:"fal"`
	Status     ImageGenerationStatus `json:"status"`
}

// ImageGenerationRequest is the ImageGenerationRequest schema.
type ImageGenerationRequest struct {
	Options *ImageOptions      `json:"options,omitempty"`
	Prompt  string             `json:"prompt" doc:"What to draw, in the caller's own words." example:"A yellow watering can beside a seedling, flat illustration, no text"`
	Tags    *map[string]string `json:"tags,omitempty"`
}
