package api

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
	"github.com/danielgtaylor/huma/v2"
)

// noVoices is what the voice paths say on a deployment that cannot hold a recording.
const noVoices = "voices of your own are not available: this deployment has no object storage configured"

// noLibrary is what the library path says where no provider publishes a catalogue here.
const noLibrary = "no speech provider configured here publishes a voice library, so a voice is whatever id the provider knows it by"

// unknownVoice is what a caller is told about a voice that is not theirs, or not there.
const unknownVoice = "there is no such voice"

// previewLine is what a voice says when the caller did not choose a line.
const previewLine = "Hi there! This is how I will sound when I answer your calls."

type Voice struct {
	Id          string          `json:"id"`
	Name        string          `json:"name"`
	Description *string         `json:"description,omitempty"`
	Samples     *[]VoiceSample  `json:"samples,omitempty"`
	Bindings    *[]VoiceBinding `json:"bindings,omitempty"`
	CreatedAt   time.Time       `json:"created_at"`
	UpdatedAt   time.Time       `json:"updated_at"`
}

type VoiceSample struct {
	Id          string    `json:"id"`
	Filename    *string   `json:"filename,omitempty"`
	ContentType *string   `json:"content_type,omitempty"`
	Bytes       *int64    `json:"bytes,omitempty"`
	Transcript  *string   `json:"transcript,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
}

type VoiceBindingState string

const (
	VoiceBindingStateFailed  VoiceBindingState = "failed"
	VoiceBindingStatePending VoiceBindingState = "pending"
	VoiceBindingStateReady   VoiceBindingState = "ready"
)

// Valid indicates whether the value is a known member of the VoiceBindingState enum.
func (e VoiceBindingState) Valid() bool {
	switch e {
	case VoiceBindingStateFailed:
		return true
	case VoiceBindingStatePending:
		return true
	case VoiceBindingStateReady:
		return true
	default:
		return false
	}
}

type VoiceBinding struct {
	Provider   string            `json:"provider"`
	ExternalId *string           `json:"external_id,omitempty" doc:"What this provider calls the voice."`
	State      VoiceBindingState `json:"state" enum:"pending,ready,failed"`
	Error      *string           `json:"error,omitempty" doc:"Why the provider would not take the recordings, when it would not."`
	UpdatedAt  *time.Time        `json:"updated_at,omitempty"`
	SyncedAt   *time.Time        `json:"synced_at,omitempty" doc:"When this provider last came back with a voice that can be spoken in, absent until one does. It is not updated_at, which moves again when a binding goes back to pending."`
}

type VoiceRequest struct {
	Name        string  `json:"name" doc:"What an agent config names the voice by, which is unique among the customer's own voices."`
	Description *string `json:"description,omitempty" doc:"A note for whoever reads the voice back, and for the provider's dashboard."`
}

type VoiceSampleRequest struct {
	Audio       []byte  `json:"audio" doc:"The recording, base64 encoded. Thirty seconds of clean speech is plenty, and every provider here clones from less." format:"byte"`
	Filename    *string `json:"filename,omitempty" doc:"What to call the file upstream. The extension is how a provider knows what it was given, so send one."`
	ContentType *string `json:"content_type,omitempty"`
	Transcript  *string `json:"transcript,omitempty" doc:"What is said in the recording. Optional, and the providers that use one clone more faithfully with it."`
}

type PrepareVoiceRequest struct {
	Providers *[]string `json:"providers,omitempty" doc:"Which providers to teach the voice to. Empty means every provider this deployment can clone with."`
}

type VoicePreviewRequest struct {
	Provider string  `json:"provider" doc:"Which provider's copy of the voice to hear."`
	Text     *string `json:"text,omitempty" doc:"What to say. Omitted says a short greeting."`
}

type VoicePreview struct {
	Provider    string `json:"provider"`
	ContentType string `json:"content_type" example:"audio/mpeg"`
	Audio       []byte `json:"audio" doc:"The spoken line, base64 encoded." format:"byte"`
}

type VoiceProviders struct {
	Providers []string `json:"providers"`
}

type LibraryVoices struct {
	Voices      []LibraryVoice `json:"voices"`
	Providers   []string       `json:"providers" doc:"The providers that publish a library, sorted by name."`
	Unavailable *[]string      `json:"unavailable,omitempty" doc:"Providers whose library could not be read just now."`
}

type LibraryVoice struct {
	Provider    string    `json:"provider"`
	Id          string    `json:"id" doc:"What to put in the voice field, in the provider's own terms."`
	Name        string    `json:"name"`
	Description *string   `json:"description,omitempty"`
	Gender      *string   `json:"gender,omitempty"`
	Accent      *string   `json:"accent,omitempty"`
	Language    *string   `json:"language,omitempty"`
	Tags        *[]string `json:"tags,omitempty"`
	Own         *bool     `json:"own,omitempty" doc:"A voice this account made, rather than one from the public library."`
	Preview     *bool     `json:"preview,omitempty" doc:"Whether the voice can be heard."`
}

func (*LibraryVoice) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One voice a provider offers. Everything past the name is what that vendor chose to say " +
		"about it, in its own words, so a field being absent means the vendor did not label it " +
		"rather than that the voice lacks it."
	return schema
}

type voiceListResponse struct {
	Body []Voice
}

type createVoiceRequest struct {
	Body VoiceRequest
}

type voiceResponse struct {
	Body Voice
}

type getVoiceRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type updateVoiceRequest struct {
	ID   string `path:"id" doc:"The resource, as returned when it was created."`
	Body VoiceRequest
}

type deleteVoiceRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type addVoiceSampleRequest struct {
	ID   string `path:"id" doc:"The resource, as returned when it was created."`
	Body VoiceSampleRequest
}

type prepareVoiceRequest struct {
	ID   string `path:"id" doc:"The resource, as returned when it was created."`
	Body PrepareVoiceRequest
}

type previewVoiceRequest struct {
	ID   string `path:"id" doc:"The resource, as returned when it was created."`
	Body VoicePreviewRequest
}

type voicePreviewResponse struct {
	Body VoicePreview
}

type voiceProvidersResponse struct {
	Body VoiceProviders
}

type listLibraryVoicesRequest struct {
	Provider optionalParam[string] `query:"provider" doc:"Only this provider's voices."`
}

type libraryVoicesResponse struct {
	Body LibraryVoices
}

type previewLibraryVoiceRequest struct {
	Provider string `path:"provider"`
	Voice    string `path:"voice" doc:"The voice id, as the provider names it."`
}

func (s *Server) registerVoices(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listVoices",
		Method:      http.MethodGet,
		Path:        "/v1/agents/voices",
		Summary:     "The voices the calling customer has brought with them",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's voices, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listVoices)
	huma.Register(api, huma.Operation{
		OperationID: "createVoice",
		Method:      http.MethodPost,
		Path:        "/v1/agents/voices",
		Summary:     "Name a voice of the customer's own",
		Description: "A voice starts empty. Recordings are added to it one at a time, and then it is " +
			"prepared with the text-to-speech providers that should be able to speak in it. " +
			"An agent config names this voice rather than any provider's id for it, so the " +
			"router can still fail over between providers mid-call.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The voice was stored"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.createVoice)
	huma.Register(api, huma.Operation{
		OperationID: "getVoice",
		Method:      http.MethodGet,
		Path:        "/v1/agents/voices/{id}",
		Summary:     "One voice, with its recordings and what each provider made of them",
		Responses: map[string]*huma.Response{
			"200": {Description: "The voice"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getVoice)
	huma.Register(api, huma.Operation{
		OperationID: "updateVoice",
		Method:      http.MethodPut,
		Path:        "/v1/agents/voices/{id}",
		Summary:     "Rename a voice",
		Description: "Only what the voice is called changes. The recordings and the provider bindings " +
			"are what it sounds like, and renaming it does not make it sound different.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The voice as it now is"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.updateVoice)
	huma.Register(api, huma.Operation{
		OperationID: "deleteVoice",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/voices/{id}",
		Summary:     "Delete a voice",
		Description: "The voice is taken off every provider it was prepared with, so a deleted voice " +
			"stops being billed for as well as stops being usable. Calls that spoke in it " +
			"keep naming it.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"204": {Description: "The voice is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteVoice)
	huma.Register(api, huma.Operation{
		OperationID: "addVoiceSample",
		Method:      http.MethodPost,
		Path:        "/v1/agents/voices/{id}/samples",
		Summary:     "Add a recording to a voice",
		Description: "The audio is stored in the deployment's object bucket and the voice keeps a " +
			"reference to it, so recordings can be re-sent to a provider that is added later " +
			"without asking the customer for them again. Adding a recording does not " +
			"re-prepare the voice.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The voice with the recording added"},
		},
		Errors:       []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
		MaxBodyBytes: largeBody,
	}, s.addVoiceSample)
	huma.Register(api, huma.Operation{
		OperationID: "prepareVoice",
		Method:      http.MethodPost,
		Path:        "/v1/agents/voices/{id}/prepare",
		Summary:     "Teach the text-to-speech providers this voice",
		Description: "Each provider is sent the recordings and hands back an id of its own, which is " +
			"remembered so a session can ask for this voice by name. Providers are prepared " +
			"independently: one refusing the recordings leaves the others usable, and the " +
			"binding says why it refused.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The voice with a binding per provider"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.prepareVoice)
	huma.Register(api, huma.Operation{
		OperationID: "previewVoice",
		Method:      http.MethodPost,
		Path:        "/v1/agents/voices/{id}/preview",
		Summary:     "Hear a voice through one provider",
		Description: "Says a short line in the voice with the provider's own copy of it, so what a " +
			"caller will hear can be checked before an agent speaks in it. The provider must " +
			"have the voice ready.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The line, spoken"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.previewVoice)
	huma.Register(api, huma.Operation{
		OperationID: "listVoiceProviders",
		Method:      http.MethodGet,
		Path:        "/v1/agents/voices/providers",
		Summary:     "The providers a voice can be prepared with",
		Description: "Only the providers this deployment holds a key for and knows how to clone with. " +
			"A voice with no binding for one of them has not been prepared there yet.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The providers, sorted by name"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listVoiceProviders)
	huma.Register(api, huma.Operation{
		OperationID: "listLibraryVoices",
		Method:      http.MethodGet,
		Path:        "/v1/agents/voices/library",
		Summary:     "The voices the speech providers offer",
		Description: "The catalogue each provider publishes, so a voice can be picked by name rather " +
			"than by pasting an id. Only providers this deployment holds a key for and that " +
			"publish a library appear; for the others a voice is still whatever the vendor's " +
			"own terms call one, and has to be typed. A provider that cannot be reached is " +
			"reported in `unavailable` rather than emptying the list.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The voices, by provider and then name"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listLibraryVoices)
	huma.Register(api, huma.Operation{
		OperationID: "previewLibraryVoice",
		Method:      http.MethodGet,
		Path:        "/v1/agents/voices/library/{provider}/{voice}/preview",
		Summary:     "Hear a voice from a provider's library",
		Description: "The sample the vendor already published, fetched through the router because two " +
			"of them want the deployment's key to hand it over. Nothing is synthesised, so " +
			"browsing a library spends no credits.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The sample"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.previewLibraryVoice)
}

// listVoices returns the calling customer's own voices, newest first.
func (s *Server) listVoices(ctx context.Context, _ *struct{}) (*voiceListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.voices == nil {
		return nil, huma.Error400BadRequest(noVoices)
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
	return &voiceListResponse{Body: listed}, nil
}

// createVoice names a new, empty voice.
func (s *Server) createVoice(ctx context.Context, request *createVoiceRequest) (*voiceResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.voices == nil {
		return nil, huma.Error400BadRequest(noVoices)
	}
	if strings.TrimSpace(request.Body.Name) == "" {
		return nil, huma.Error400BadRequest("a voice needs a name")
	}

	voice, err := s.voices.Create(ctx, customerID, request.Body.Name, text(request.Body.Description))
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	described, err := s.describeVoice(ctx, voice)
	if err != nil {
		return nil, err
	}
	return &voiceResponse{Body: described}, nil
}

// getVoice returns one voice with its recordings and provider bindings.
func (s *Server) getVoice(ctx context.Context, request *getVoiceRequest) (*voiceResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.voices == nil {
		return nil, huma.Error400BadRequest(noVoices)
	}

	voice, err := s.store.Voice(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownVoice)
	}

	described, err := s.describeVoice(ctx, voice)
	if err != nil {
		return nil, err
	}
	return &voiceResponse{Body: described}, nil
}

// updateVoice renames a voice.
func (s *Server) updateVoice(ctx context.Context, request *updateVoiceRequest) (*voiceResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.voices == nil {
		return nil, huma.Error400BadRequest(noVoices)
	}
	if strings.TrimSpace(request.Body.Name) == "" {
		return nil, huma.Error400BadRequest("a voice needs a name")
	}

	voice := store.Voice{
		ID:          request.ID,
		CustomerID:  customerID,
		Name:        request.Body.Name,
		Description: text(request.Body.Description),
	}
	if err := s.store.UpdateVoice(ctx, &voice); err != nil {
		if errors.Is(err, store.ErrNoVoice) {
			return nil, huma.Error404NotFound(unknownVoice)
		}
		return nil, huma.Error400BadRequest(err.Error())
	}

	described, err := s.describeVoice(ctx, voice)
	if err != nil {
		return nil, err
	}
	return &voiceResponse{Body: described}, nil
}

// deleteVoice takes the voice off every provider and then forgets it.
func (s *Server) deleteVoice(ctx context.Context, request *deleteVoiceRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.voices == nil {
		return nil, huma.Error400BadRequest(noVoices)
	}

	if err := s.voices.Delete(ctx, customerID, request.ID); err != nil {
		if errors.Is(err, store.ErrNoVoice) {
			return nil, huma.Error404NotFound(unknownVoice)
		}
		return nil, huma.Error400BadRequest(err.Error())
	}
	return nil, nil
}

// addVoiceSample stores a recording against a voice.
func (s *Server) addVoiceSample(ctx context.Context, request *addVoiceSampleRequest) (*voiceResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.voices == nil {
		return nil, huma.Error400BadRequest(noVoices)
	}

	sample := voices.Sample{
		Name:        text(request.Body.Filename),
		ContentType: text(request.Body.ContentType),
		Content:     request.Body.Audio,
		Transcript:  text(request.Body.Transcript),
	}
	if err := s.voices.AddSample(ctx, customerID, request.ID, sample); err != nil {
		if errors.Is(err, store.ErrNoVoice) {
			return nil, huma.Error404NotFound(unknownVoice)
		}
		return nil, huma.Error400BadRequest(err.Error())
	}

	voice, err := s.store.Voice(ctx, customerID, request.ID)
	if err != nil {
		return nil, err
	}
	described, err := s.describeVoice(ctx, voice)
	if err != nil {
		return nil, err
	}
	return &voiceResponse{Body: described}, nil
}

// prepareVoice teaches the voice to the text-to-speech providers.
func (s *Server) prepareVoice(ctx context.Context, request *prepareVoiceRequest) (*voiceResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.voices == nil {
		return nil, huma.Error400BadRequest(noVoices)
	}

	var providers []string
	if request.Body.Providers != nil {
		providers = *request.Body.Providers
	}

	if err := s.voices.Prepare(ctx, customerID, request.ID, providers); err != nil {
		if errors.Is(err, store.ErrNoVoice) {
			return nil, huma.Error404NotFound(unknownVoice)
		}
		return nil, huma.Error400BadRequest(err.Error())
	}

	voice, err := s.store.Voice(ctx, customerID, request.ID)
	if err != nil {
		return nil, err
	}
	described, err := s.describeVoice(ctx, voice)
	if err != nil {
		return nil, err
	}
	return &voiceResponse{Body: described}, nil
}

// listVoiceProviders reports which providers a voice can be prepared with.
func (s *Server) listVoiceProviders(ctx context.Context, _ *struct{}) (*voiceProvidersResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.voices == nil {
		return nil, huma.Error400BadRequest(noVoices)
	}
	return &voiceProvidersResponse{Body: VoiceProviders{Providers: s.voices.Providers()}}, nil
}

// listLibraryVoices returns the voices the speech providers themselves offer.
func (s *Server) listLibraryVoices(ctx context.Context, request *listLibraryVoicesRequest) (*libraryVoicesResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.library == nil {
		return nil, huma.Error400BadRequest(noLibrary)
	}

	provider := text(request.Provider.Ptr())
	found, err := s.library.List(ctx, provider)
	// Some providers answering is enough to fill a picker, so a failure only refuses the
	// request when it left nothing to show.
	if len(found) == 0 && err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	if err != nil {
		s.logger.Warn("a voice library could not be read", "error", err)
	}

	listed := make([]LibraryVoice, 0, len(found))
	for _, voice := range found {
		listed = append(listed, LibraryVoice{
			Provider:    voice.Provider,
			Id:          voice.ID,
			Name:        voice.Name,
			Description: optional(voice.Description),
			Gender:      optional(voice.Gender),
			Accent:      optional(voice.Accent),
			Language:    optional(voice.Language),
			Tags:        &voice.Tags,
			Own:         &voice.Own,
			Preview:     &voice.Preview,
		})
	}
	answered := map[string]struct{}{}
	for _, voice := range found {
		answered[voice.Provider] = struct{}{}
	}
	providers := s.library.Providers()
	var unavailable []string
	for _, name := range providers {
		if _, ok := answered[name]; !ok && (provider == "" || provider == name) {
			unavailable = append(unavailable, name)
		}
	}
	return &libraryVoicesResponse{Body: LibraryVoices{
		Voices:      listed,
		Providers:   providers,
		Unavailable: &unavailable,
	}}, nil
}

// previewLibraryVoice hands back the sample a provider published for one of its voices.
func (s *Server) previewLibraryVoice(ctx context.Context, request *previewLibraryVoiceRequest) (*voicePreviewResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.library == nil {
		return nil, huma.Error400BadRequest(noLibrary)
	}

	spoken, err := s.library.Preview(ctx, request.Provider, request.Voice)
	if err != nil {
		return nil, huma.Error404NotFound(err.Error())
	}
	return &voicePreviewResponse{Body: VoicePreview{
		Provider:    request.Provider,
		ContentType: spoken.ContentType,
		Audio:       spoken.Audio,
	}}, nil
}

// previewVoice says a short line in the voice through one provider.
func (s *Server) previewVoice(ctx context.Context, request *previewVoiceRequest) (*voicePreviewResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.voices == nil {
		return nil, huma.Error400BadRequest(noVoices)
	}
	if strings.TrimSpace(request.Body.Provider) == "" {
		return nil, huma.Error400BadRequest("name the provider to hear the voice through")
	}
	if _, err := s.store.Voice(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error404NotFound(unknownVoice)
	}

	provider := request.Body.Provider
	line := text(request.Body.Text)
	if strings.TrimSpace(line) == "" {
		line = previewLine
	}
	speech, err := s.voices.Speak(ctx, customerID, request.ID, provider, line)
	if errors.Is(err, store.ErrNoVoice) {
		return nil, huma.Error400BadRequest("the voice is not ready with " + provider + " yet")
	}
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &voicePreviewResponse{Body: VoicePreview{
		Provider:    provider,
		ContentType: speech.ContentType,
		Audio:       speech.Audio,
	}}, nil
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
			SyncedAt:   binding.SyncedAt,
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
