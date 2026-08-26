package api

import (
	"context"
	"errors"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
)

// noVoices is what the voice paths say on a deployment that cannot hold a recording.
const noVoices = "voices of your own are not available: this deployment has no object storage configured"

// unknownVoice is what a caller is told about a voice that is not theirs, or not there.
const unknownVoice = "there is no such voice"

// ListVoices returns the calling customer's own voices, newest first.
func (s *Server) ListVoices(ctx context.Context, _ ListVoicesRequestObject) (ListVoicesResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListVoices401JSONResponse{missingCustomer()}, nil
	}
	if s.voices == nil {
		return ListVoices400JSONResponse{badRequest(noVoices)}, nil
	}

	stored, err := s.store.CustomerVoices(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]Voice, 0, len(stored))
	for _, voice := range stored {
		described, err := s.describeVoice(ctx, voice)
		if err != nil {
			return nil, err
		}
		listed = append(listed, described)
	}
	return ListVoices200JSONResponse(listed), nil
}

// CreateVoice names a new, empty voice.
func (s *Server) CreateVoice(ctx context.Context, request CreateVoiceRequestObject) (CreateVoiceResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateVoice401JSONResponse{missingCustomer()}, nil
	}
	if s.voices == nil {
		return CreateVoice400JSONResponse{badRequest(noVoices)}, nil
	}
	if request.Body == nil {
		return CreateVoice400JSONResponse{badRequest("a request body is required")}, nil
	}
	if strings.TrimSpace(request.Body.Name) == "" {
		return CreateVoice400JSONResponse{badRequest("a voice needs a name")}, nil
	}

	voice, err := s.voices.Create(ctx, customerID, request.Body.Name, text(request.Body.Description))
	if err != nil {
		return CreateVoice400JSONResponse{badRequest(err.Error())}, nil
	}

	described, err := s.describeVoice(ctx, voice)
	if err != nil {
		return nil, err
	}
	return CreateVoice201JSONResponse(described), nil
}

// GetVoice returns one voice with its recordings and provider bindings.
func (s *Server) GetVoice(ctx context.Context, request GetVoiceRequestObject) (GetVoiceResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetVoice401JSONResponse{missingCustomer()}, nil
	}
	if s.voices == nil {
		return GetVoice400JSONResponse{badRequest(noVoices)}, nil
	}

	voice, err := s.store.Voice(ctx, customerID, request.Id)
	if err != nil {
		return GetVoice404JSONResponse{NotFoundJSONResponse{Error: unknownVoice}}, nil
	}

	described, err := s.describeVoice(ctx, voice)
	if err != nil {
		return nil, err
	}
	return GetVoice200JSONResponse(described), nil
}

// UpdateVoice renames a voice.
func (s *Server) UpdateVoice(ctx context.Context, request UpdateVoiceRequestObject) (UpdateVoiceResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return UpdateVoice401JSONResponse{missingCustomer()}, nil
	}
	if s.voices == nil {
		return UpdateVoice400JSONResponse{badRequest(noVoices)}, nil
	}
	if request.Body == nil {
		return UpdateVoice400JSONResponse{badRequest("a request body is required")}, nil
	}
	if strings.TrimSpace(request.Body.Name) == "" {
		return UpdateVoice400JSONResponse{badRequest("a voice needs a name")}, nil
	}

	voice := store.Voice{
		ID:          request.Id,
		CustomerID:  customerID,
		Name:        request.Body.Name,
		Description: text(request.Body.Description),
	}
	if err := s.store.UpdateVoice(ctx, &voice); err != nil {
		if errors.Is(err, store.ErrNoVoice) {
			return UpdateVoice404JSONResponse{NotFoundJSONResponse{Error: unknownVoice}}, nil
		}
		return UpdateVoice400JSONResponse{badRequest(err.Error())}, nil
	}

	described, err := s.describeVoice(ctx, voice)
	if err != nil {
		return nil, err
	}
	return UpdateVoice200JSONResponse(described), nil
}

// DeleteVoice takes the voice off every provider and then forgets it.
func (s *Server) DeleteVoice(ctx context.Context, request DeleteVoiceRequestObject) (DeleteVoiceResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return DeleteVoice401JSONResponse{missingCustomer()}, nil
	}
	if s.voices == nil {
		return DeleteVoice400JSONResponse{badRequest(noVoices)}, nil
	}

	if err := s.voices.Delete(ctx, customerID, request.Id); err != nil {
		if errors.Is(err, store.ErrNoVoice) {
			return DeleteVoice404JSONResponse{NotFoundJSONResponse{Error: unknownVoice}}, nil
		}
		return DeleteVoice400JSONResponse{badRequest(err.Error())}, nil
	}
	return DeleteVoice204Response{}, nil
}

// AddVoiceSample stores a recording against a voice.
func (s *Server) AddVoiceSample(ctx context.Context, request AddVoiceSampleRequestObject) (AddVoiceSampleResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return AddVoiceSample401JSONResponse{missingCustomer()}, nil
	}
	if s.voices == nil {
		return AddVoiceSample400JSONResponse{badRequest(noVoices)}, nil
	}
	if request.Body == nil {
		return AddVoiceSample400JSONResponse{badRequest("a request body is required")}, nil
	}

	sample := voices.Sample{
		Name:        text(request.Body.Filename),
		ContentType: text(request.Body.ContentType),
		Content:     request.Body.Audio,
		Transcript:  text(request.Body.Transcript),
	}
	if err := s.voices.AddSample(ctx, customerID, request.Id, sample); err != nil {
		if errors.Is(err, store.ErrNoVoice) {
			return AddVoiceSample404JSONResponse{NotFoundJSONResponse{Error: unknownVoice}}, nil
		}
		return AddVoiceSample400JSONResponse{badRequest(err.Error())}, nil
	}

	voice, err := s.store.Voice(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}
	described, err := s.describeVoice(ctx, voice)
	if err != nil {
		return nil, err
	}
	return AddVoiceSample201JSONResponse(described), nil
}

// PrepareVoice teaches the voice to the text-to-speech providers.
func (s *Server) PrepareVoice(ctx context.Context, request PrepareVoiceRequestObject) (PrepareVoiceResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return PrepareVoice401JSONResponse{missingCustomer()}, nil
	}
	if s.voices == nil {
		return PrepareVoice400JSONResponse{badRequest(noVoices)}, nil
	}

	var providers []string
	if request.Body != nil && request.Body.Providers != nil {
		providers = *request.Body.Providers
	}

	if err := s.voices.Prepare(ctx, customerID, request.Id, providers); err != nil {
		if errors.Is(err, store.ErrNoVoice) {
			return PrepareVoice404JSONResponse{NotFoundJSONResponse{Error: unknownVoice}}, nil
		}
		return PrepareVoice400JSONResponse{badRequest(err.Error())}, nil
	}

	voice, err := s.store.Voice(ctx, customerID, request.Id)
	if err != nil {
		return nil, err
	}
	described, err := s.describeVoice(ctx, voice)
	if err != nil {
		return nil, err
	}
	return PrepareVoice200JSONResponse(described), nil
}

// describeVoice reads a voice back with its recordings and bindings, which is what every
// voice path answers with. The audio itself is never sent back: the caller uploaded it.
func (s *Server) describeVoice(ctx context.Context, voice store.Voice) (Voice, error) {
	samples, err := s.voices.Samples(ctx, voice.ID)
	if err != nil {
		return Voice{}, err
	}
	bindings, err := s.voices.Bindings(ctx, voice.ID)
	if err != nil {
		return Voice{}, err
	}

	described := make([]VoiceSample, 0, len(samples))
	for _, sample := range samples {
		described = append(described, VoiceSample{
			Id:          sample.ID,
			Filename:    optional(lastSegment(sample.ObjectKey)),
			ContentType: optional(sample.ContentType),
			Bytes:       &sample.Bytes,
			Transcript:  optional(sample.Transcript),
			CreatedAt:   sample.CreatedAt,
		})
	}

	bound := make([]VoiceBinding, 0, len(bindings))
	for _, binding := range bindings {
		bound = append(bound, VoiceBinding{
			Provider:   binding.Provider,
			ExternalId: optional(binding.ExternalID),
			State:      VoiceBindingState(binding.State),
			Error:      optional(binding.Error),
			UpdatedAt:  &binding.UpdatedAt,
		})
	}

	return Voice{
		Id:          voice.ID,
		Name:        voice.Name,
		Description: optional(voice.Description),
		Samples:     &described,
		Bindings:    &bound,
		CreatedAt:   voice.CreatedAt,
		UpdatedAt:   voice.UpdatedAt,
	}, nil
}

// lastSegment is the filename part of an object key.
func lastSegment(key string) string {
	if index := strings.LastIndex(key, "/"); index >= 0 {
		return key[index+1:]
	}
	return key
}

// text reads an optional string a caller may have left out.
func text(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}
