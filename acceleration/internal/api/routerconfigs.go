package api

import (
	"context"
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// noRouterConfigs is what the router config paths say on a deployment without a database.
const noRouterConfigs = "router configs are not available: no database configured"

// unknownRouterConfig is what a caller is told about a config that is not theirs, which is
// the same thing they are told about one that never existed.
const unknownRouterConfig = "no such router config"

// ListRouterConfigs returns the calling customer's router configs, newest first.
func (s *Server) ListRouterConfigs(ctx context.Context, _ ListRouterConfigsRequestObject) (ListRouterConfigsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListRouterConfigs401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ListRouterConfigs400JSONResponse{badRequest(noRouterConfigs)}, nil
	}

	stored, err := s.store.CustomerRouterConfigs(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]RouterConfig, 0, len(stored))
	for _, config := range stored {
		listed = append(listed, routerConfigOf(config))
	}
	return ListRouterConfigs200JSONResponse(listed), nil
}

// CreateRouterConfig stores a named set of per-modality routing options.
func (s *Server) CreateRouterConfig(ctx context.Context, request CreateRouterConfigRequestObject) (CreateRouterConfigResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateRouterConfig401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return CreateRouterConfig400JSONResponse{badRequest(noRouterConfigs)}, nil
	}
	if request.Body == nil {
		return CreateRouterConfig400JSONResponse{badRequest("a request body is required")}, nil
	}
	if message, ok := routerConfigComplaint(*request.Body); !ok {
		return CreateRouterConfig400JSONResponse{badRequest(message)}, nil
	}

	config := storedRouterConfig(*request.Body, customerID)
	if err := s.store.CreateRouterConfig(ctx, &config); err != nil {
		return CreateRouterConfig400JSONResponse{badRequest(err.Error())}, nil
	}
	return CreateRouterConfig201JSONResponse(routerConfigOf(config)), nil
}

// GetRouterConfig returns one router config.
func (s *Server) GetRouterConfig(ctx context.Context, request GetRouterConfigRequestObject) (GetRouterConfigResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetRouterConfig401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetRouterConfig400JSONResponse{badRequest(noRouterConfigs)}, nil
	}

	config, err := s.store.RouterConfig(ctx, customerID, request.Id)
	if err != nil {
		return GetRouterConfig404JSONResponse{NotFoundJSONResponse{Error: unknownRouterConfig}}, nil
	}
	return GetRouterConfig200JSONResponse(routerConfigOf(config)), nil
}

// UpdateRouterConfig replaces a router config with what it now is.
func (s *Server) UpdateRouterConfig(ctx context.Context, request UpdateRouterConfigRequestObject) (UpdateRouterConfigResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return UpdateRouterConfig401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return UpdateRouterConfig400JSONResponse{badRequest(noRouterConfigs)}, nil
	}
	if request.Body == nil {
		return UpdateRouterConfig400JSONResponse{badRequest("a request body is required")}, nil
	}
	if message, ok := routerConfigComplaint(*request.Body); !ok {
		return UpdateRouterConfig400JSONResponse{badRequest(message)}, nil
	}

	existing, err := s.store.RouterConfig(ctx, customerID, request.Id)
	if err != nil {
		return UpdateRouterConfig404JSONResponse{NotFoundJSONResponse{Error: unknownRouterConfig}}, nil
	}

	config := storedRouterConfig(*request.Body, customerID)
	config.ID = existing.ID
	config.CreatedAt = existing.CreatedAt
	if err := s.store.UpdateRouterConfig(ctx, &config); err != nil {
		return UpdateRouterConfig400JSONResponse{badRequest(err.Error())}, nil
	}
	return UpdateRouterConfig200JSONResponse(routerConfigOf(config)), nil
}

// DeleteRouterConfig stops a router config being usable.
func (s *Server) DeleteRouterConfig(ctx context.Context, request DeleteRouterConfigRequestObject) (DeleteRouterConfigResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return DeleteRouterConfig401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return DeleteRouterConfig400JSONResponse{badRequest(noRouterConfigs)}, nil
	}

	if err := s.store.DeleteRouterConfig(ctx, customerID, request.Id); err != nil {
		return DeleteRouterConfig404JSONResponse{NotFoundJSONResponse{Error: unknownRouterConfig}}, nil
	}
	return DeleteRouterConfig204Response{}, nil
}

// routerConfigComplaint reports what is wrong with a router config, if anything. The tags
// and the keyterms are checked here rather than left to the request that uses the config,
// because a config nothing can be routed under is worth hearing about while it is being
// written and not once a socket is open.
func routerConfigComplaint(request RouterConfigRequest) (string, bool) {
	if strings.TrimSpace(request.Name) == "" {
		return "a router config needs a name", false
	}
	if request.Tags != nil {
		if err := routing.Tags(*request.Tags).Validate(); err != nil {
			return err.Error(), false
		}
	}
	if request.Stt != nil && request.Stt.Keyterms != nil && len(*request.Stt.Keyterms) > stt.MaxKeyterms {
		return fmt.Sprintf("a config may name at most %d keyterms", stt.MaxKeyterms), false
	}
	return "", true
}

// storedRouterConfig turns a request into a row. The customer comes from the trusted
// header rather than the body, the same way an agent config's does.
func storedRouterConfig(request RouterConfigRequest, customerID string) store.RouterConfig {
	config := store.RouterConfig{
		CustomerID: customerID,
		Name:       strings.TrimSpace(request.Name),
		STT:        sttOptionsOf(request.Stt),
		TTS:        ttsOptionsOf(request.Tts),
		LLM:        llmOptionsOf(request.Llm),
		Search:     searchOptionsOf(request.Search),
	}
	if request.Tags != nil {
		config.Tags = *request.Tags
	}
	config.STT.Keyterms = stt.CleanKeyterms(config.STT.Keyterms)
	return config
}

// routerConfigOf renders a config for the wire.
func routerConfigOf(config store.RouterConfig) RouterConfig {
	rendered := RouterConfig{
		Id:        config.ID,
		Name:      config.Name,
		Stt:       sttOptionsFor(config.STT),
		Tts:       ttsOptionsFor(config.TTS),
		Llm:       llmOptionsFor(config.LLM),
		Search:    searchOptionsFor(config.Search),
		CreatedAt: config.CreatedAt,
		UpdatedAt: config.UpdatedAt,
	}
	if len(config.Tags) > 0 {
		tags := config.Tags
		rendered.Tags = &tags
	}
	return rendered
}

// routerOptions reads a stored config, if one was named, and writes the per-call options
// over it. Everything a config holds is a default; a keyword on the call overrides that
// one field of it.
//
// A config nobody can find is an error rather than an empty default: a caller that named
// one meant it, and transcribing at whatever the fallback happens to be is not what they
// asked for. It is found by id first and by name second, so a caller can say either.
func (s *Server) routerOptions(ctx context.Context, customerID, configID string) (store.RouterConfig, error) {
	if configID == "" {
		return store.RouterConfig{}, nil
	}
	if s.store == nil {
		return store.RouterConfig{}, fmt.Errorf("%s", noRouterConfigs)
	}

	if config, err := s.store.RouterConfig(ctx, customerID, configID); err == nil {
		return config, nil
	}
	config, found, err := s.store.RouterConfigByName(ctx, customerID, configID)
	if err != nil {
		return store.RouterConfig{}, err
	}
	if !found {
		return store.RouterConfig{}, fmt.Errorf("%s: %s", unknownRouterConfig, configID)
	}
	return config, nil
}

// tagsUnder are the labels a request is billed with: the config's own, with the request's
// written over them, so a caller can add a label to one job without restating the rest.
func tagsUnder(config store.RouterConfig, sent *map[string]string) routing.Tags {
	tags := routing.Tags{}
	for key, value := range config.Tags {
		tags[key] = value
	}
	if sent != nil {
		for key, value := range *sent {
			tags[key] = value
		}
	}
	return tags
}

// These are what a block that names no target falls back to, which is what a session with
// nothing configured falls back to.
const (
	sttDefaultTarget = "en-low-latency"
	ttsDefaultTarget = "en-low-latency"
	llmDefaultTarget = "llm-fast"
)

// targeted returns the options with a target filled in, since routing has to be told
// where to go and a caller that said nothing meant the usual place.
func targeted(held options.STT) options.STT {
	if held.Target == "" {
		held.Target = sttDefaultTarget
	}
	return held
}

// recordedTarget is where a job with no target of its own goes: the recorded aliases,
// which are the batch models rather than the live ones. A recording streamed at a socket
// would cost more and transcribe worse, so a caller who only said "transcribe this file"
// is not sent there.
func recordedTarget(languages []string) string {
	for _, language := range languages {
		if language != "" && !strings.HasPrefix(language, "en") {
			return "multilingual-recorded"
		}
	}
	return "en-recorded"
}
