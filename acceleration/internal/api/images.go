package api

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
)

// imageDeadline bounds one generation. A queued provider can hold a job for minutes, and
// a caller waiting on one response is owed an answer rather than a connection held open
// until something upstream gives up.
const imageDeadline = 240 * time.Second

// noImages is what the image path says on a deployment that does not generate images.
const noImages = "this deployment does not generate images"

// GenerateImage draws pictures from a prompt and returns them.
//
// It is answered inline and nothing is kept: the pictures are in the response and the
// only thing written down is what they cost. A generation that reached a provider is a 200
// whether or not it drew, the way an inline recording is a 202 whether or not it spoke,
// since the failure is part of the answer rather than something wrong with the request.
// A caller that hangs up cancels the job at the provider, because the request's context
// is what the provider waits on.
func (s *Server) GenerateImage(ctx context.Context, request GenerateImageRequestObject) (GenerateImageResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GenerateImage401JSONResponse{missingCustomer()}, nil
	}
	if s.streams == nil || s.streams.Image == nil {
		return GenerateImage404JSONResponse{NotFoundJSONResponse{Error: noImages}}, nil
	}
	if request.Body == nil {
		return GenerateImage400JSONResponse{badRequest("a request body is required")}, nil
	}

	drawing, err := imageRequestOf(request.Body)
	if err != nil {
		return GenerateImage400JSONResponse{badRequest(err.Error())}, nil
	}
	if err := drawing.Validate(); err != nil {
		return GenerateImage400JSONResponse{badRequest(err.Error())}, nil
	}
	tags := tagsSent(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return GenerateImage400JSONResponse{badRequest(err.Error())}, nil
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
		return GenerateImage400JSONResponse{badRequest(err.Error())}, nil
	}
	return GenerateImage200JSONResponse(imageGenerationOf(generation, err)), nil
}

// imageRequestOf reads what to draw out of the request.
func imageRequestOf(body *ImageGenerationRequest) (imagegen.Request, error) {
	sent := value(body.Options)
	if sent.N != nil && *sent.N < 1 {
		return imagegen.Request{}, fmt.Errorf("n must be between 1 and %d", imagegen.MaxImages)
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
			return imagegen.Request{}, fmt.Errorf("size %q is not a width and a height such as 1024x1024", size)
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
