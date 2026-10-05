package api

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/danielgtaylor/huma/v2"
)

// noRouterConfigs is what the router config paths say on a deployment without a database.
const noRouterConfigs = "router configs are not available: no database configured"

// unknownRouterConfig is what a caller is told about a config that is not theirs, which is
// the same thing they are told about one that never existed.
const unknownRouterConfig = "no such router config"

// listRouterConfigs returns the calling customer's router configs, newest first.
func (s *Server) listRouterConfigs(ctx context.Context, _ *listRouterConfigsRequest) (*listRouterConfigsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRouterConfigs)
	}

	stored, err := s.store.CustomerRouterConfigs(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]RouterConfig, 0, len(stored))
	for _, config := range stored {
		listed = append(listed, routerConfigOf(config))
	}
	return &listRouterConfigsResponse{Body: listed}, nil
}

// createRouterConfig stores a named set of per-modality routing options.
func (s *Server) createRouterConfig(ctx context.Context, request *createRouterConfigRequest) (*createRouterConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRouterConfigs)
	}
	if request.Body == nil {
		return nil, huma.Error400BadRequest("a request body is required")
	}
	if message, ok := s.routerConfigComplaint(*request.Body); !ok {
		return nil, huma.Error400BadRequest(message)
	}

	config := storedRouterConfig(*request.Body, customerID)
	if err := s.configs.CreateRouterConfig(ctx, &config); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &createRouterConfigResponse{Body: routerConfigOf(config)}, nil
}

// getRouterConfig returns one router config.
func (s *Server) getRouterConfig(ctx context.Context, request *getRouterConfigRequest) (*getRouterConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRouterConfigs)
	}

	config, err := s.configs.RouterConfig(ctx, customerID, request.Id)
	if err != nil {
		return nil, huma.Error404NotFound(unknownRouterConfig)
	}
	return &getRouterConfigResponse{Body: routerConfigOf(config)}, nil
}

// updateRouterConfig replaces a router config with what it now is.
func (s *Server) updateRouterConfig(ctx context.Context, request *updateRouterConfigRequest) (*updateRouterConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRouterConfigs)
	}
	if request.Body == nil {
		return nil, huma.Error400BadRequest("a request body is required")
	}
	if message, ok := s.routerConfigComplaint(*request.Body); !ok {
		return nil, huma.Error400BadRequest(message)
	}

	existing, err := s.configs.RouterConfig(ctx, customerID, request.Id)
	if err != nil {
		return nil, huma.Error404NotFound(unknownRouterConfig)
	}

	config := storedRouterConfig(*request.Body, customerID)
	config.ID = existing.ID
	config.CreatedAt = existing.CreatedAt
	if err := s.configs.UpdateRouterConfig(ctx, &config); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &updateRouterConfigResponse{Body: routerConfigOf(config)}, nil
}

// deleteRouterConfig stops a router config being usable.
func (s *Server) deleteRouterConfig(ctx context.Context, request *deleteRouterConfigRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRouterConfigs)
	}

	if err := s.configs.DeleteRouterConfig(ctx, customerID, request.Id); err != nil {
		return nil, huma.Error404NotFound(unknownRouterConfig)
	}
	return nil, nil
}

// routerConfigComplaint reports what is wrong with a router config, if anything. The
// keyterms are checked here rather than left to the request that uses the config, because
// a config nothing can be routed under is worth hearing about while it is being written
// and not once a socket is open.
func (s *Server) routerConfigComplaint(request RouterConfigRequest) (string, bool) {
	if strings.TrimSpace(request.Name) == "" {
		return "a router config needs a name", false
	}
	if request.Stt != nil && request.Stt.Keyterms != nil && len(*request.Stt.Keyterms) > stt.MaxKeyterms {
		return fmt.Sprintf("a config may name at most %d keyterms", stt.MaxKeyterms), false
	}

	held := sttOptionsOf(request.Stt)
	if err := held.Validate(); err != nil {
		return err.Error(), false
	}
	if message, ok := s.sttComplaint(held); !ok {
		return message, false
	}

	voice := ttsOptionsOf(request.Tts)
	if err := voice.Validate(); err != nil {
		return err.Error(), false
	}
	if message, ok := s.ttsComplaint(voice); !ok {
		return message, false
	}

	conversation := stsOptionsOf(request.Sts)
	if err := conversation.Validate(); err != nil {
		return err.Error(), false
	}
	// A config decides where a conversation goes, not what is said in it. The agent
	// holding it has instructions of its own and sends them when it opens the session,
	// which is the only moment they are known.
	if conversation.Instructions != "" {
		return "a router config carries no instructions: the agent sends its own when it opens the session", false
	}
	if message, ok := s.stsComplaint(conversation); !ok {
		return message, false
	}
	if message, ok := s.chainComplaint(routing.LLM, llmOptionsOf(request.Llm).Providers); !ok {
		return message, false
	}
	if message, ok := s.chainComplaint(routing.Search, searchOptionsOf(request.Search).Providers); !ok {
		return message, false
	}
	return "", true
}

// chainComplaint is the priority list half of sttComplaint, for the modalities whose
// configs hold nothing else only their router can check.
func (s *Server) chainComplaint(modality routing.Modality, providers []string) (string, bool) {
	routed, ok := s.routerFor(Modality(modality))
	if !ok {
		return "", true
	}
	config := routed.Config()
	for _, target := range providers {
		if !config.Names(target) {
			return fmt.Sprintf(
				"%q is not a provider, a provider/model or a capability shortcut this deployment offers", target), false
		}
	}
	return "", true
}

// sttComplaint reports what this deployment could never route, which is the half of a
// config's validity that only the router knows.
//
// The same reasoning as the keyterm limit, one step further: a config naming a provider
// this build has never heard of, or a data policy none of its models meet, would fail
// every request made under it. Saying so once, while it is being written, beats saying it
// on every call afterwards.
func (s *Server) sttComplaint(held options.STT) (string, bool) {
	speech, ok := s.routerFor(Modality(routing.STT))
	if !ok {
		return "", true
	}
	config := speech.Config()

	for _, target := range held.Providers {
		if !config.Names(target) {
			return fmt.Sprintf(
				"%q is not a provider, a provider/model or a capability shortcut this deployment offers", target), false
		}
	}
	if !config.Meets(held.DataPolicy) {
		return "no provider this deployment offers meets that data policy", false
	}
	if !config.Expresses(held.Terms()) {
		return "no provider this deployment offers can serve every option in this config", false
	}
	// An overwrite for a vendor that does not exist is a typo, and the alternative to
	// reporting it is a setting that was stored, sent nowhere and never mentioned again.
	// Being a candidate for one particular request is not asked: which provider a call
	// lands on is the router's business, and a config that prepares for several is doing
	// the right thing.
	for vendor := range held.Overwrites {
		if !config.Declares(vendor) {
			return fmt.Sprintf("there are overwrites for %q, which this deployment has no provider for", vendor), false
		}
	}
	return "", true
}

// ttsComplaint is sttComplaint for the voice half, asked of the voice router's own config.
func (s *Server) ttsComplaint(held options.TTS) (string, bool) {
	voice, ok := s.routerFor(Modality(routing.TTS))
	if !ok {
		return "", true
	}
	config := voice.Config()

	for _, target := range held.Providers {
		if !config.Names(target) {
			return fmt.Sprintf(
				"%q is not a provider, a provider/model or a capability shortcut this deployment offers", target), false
		}
	}
	if !config.Meets(held.DataPolicy) {
		return "no voice this deployment offers meets that data policy", false
	}
	if !config.Expresses(held.Terms()) {
		return "no voice this deployment offers can serve every option in this config", false
	}
	for vendor := range held.Overwrites {
		if !config.Declares(vendor) {
			return fmt.Sprintf("there are overwrites for %q, which this deployment has no voice for", vendor), false
		}
	}
	return "", true
}

// stsComplaint is sttComplaint for the speech-to-speech half, asked of that router's own
// config. Frames are checked here too: a config that says it will send them and names no
// model that sees would fail every request made under it.
func (s *Server) stsComplaint(held options.STS) (string, bool) {
	conversing, ok := s.routerFor(Modality(routing.STS))
	if !ok {
		return "", true
	}
	config := conversing.Config()

	for _, target := range held.Providers {
		if !config.Names(target) {
			return fmt.Sprintf(
				"%q is not a provider, a provider/model or a capability shortcut this deployment offers", target), false
		}
	}
	if !config.Meets(held.DataPolicy) {
		return "no speech-to-speech model this deployment offers meets that data policy", false
	}
	if !config.Expresses(held.Terms()) {
		return "no speech-to-speech model this deployment offers can serve every option in this config", false
	}
	if modalities := held.InputModalities(); len(modalities) > 0 && !slices.ContainsFunc(config.Providers, func(provider routing.ProviderConfig) bool {
		return provider.Sees(modalities)
	}) {
		return "no speech-to-speech model this deployment offers sees images", false
	}
	for vendor := range held.Overwrites {
		if !config.Declares(vendor) {
			return fmt.Sprintf("there are overwrites for %q, which this deployment has no speech-to-speech model for", vendor), false
		}
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
		STS:        stsOptionsOf(request.Sts),
		Search:     searchOptionsOf(request.Search),
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
		Sts:       stsOptionsFor(config.STS),
		Search:    searchOptionsFor(config.Search),
		CreatedAt: config.CreatedAt,
		UpdatedAt: config.UpdatedAt,
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

	if config, err := s.configs.RouterConfig(ctx, customerID, configID); err == nil {
		return config, nil
	}
	config, found, err := s.configs.RouterConfigByName(ctx, customerID, configID)
	if err != nil {
		return store.RouterConfig{}, err
	}
	if !found {
		return store.RouterConfig{}, fmt.Errorf("%s: %s", unknownRouterConfig, configID)
	}
	return config, nil
}

// tagsSent are the labels a request is billed with, which are the caller's own and
// nobody else's: a router config says where to route, not who to bill.
func tagsSent(sent *map[string]string) routing.Tags {
	tags := routing.Tags{}
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
	stsDefaultTarget = "sts-fast"
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

// registerRouterconfigs declares the operations served in routerconfigs.go.
func (s *Server) registerRouterconfigs(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listRouterConfigs",
		Method:      http.MethodGet,
		Path:        "/v1/router/configs",
		Summary:     "The router configs the calling customer holds",
		Description: "A router config is what an agent config is for a session, for a caller that routes one " +
			"modality at a time: the target, the language and every per-modality option, decided " +
			"once and named. It is separate from an agent config because it configures transcribing, " +
			"speaking, answering and searching on their own, with no conversation behind them.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's router configs, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listRouterConfigs)
	huma.Register(api, huma.Operation{
		OperationID: "createRouterConfig",
		Method:      http.MethodPost,
		Path:        "/v1/router/configs",
		Summary:     "Store a named set of per-modality routing options",
		Description: "A modality block that names no target falls back to what a session falls back to, so a " +
			"config only has to say what it wants changed.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The config was stored"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.createRouterConfig)
	huma.Register(api, huma.Operation{
		OperationID: "getRouterConfig",
		Method:      http.MethodGet,
		Path:        "/v1/router/configs/{id}",
		Summary:     "One router config",
		Responses: map[string]*huma.Response{
			"200": {Description: "The config"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getRouterConfig)
	huma.Register(api, huma.Operation{
		OperationID: "updateRouterConfig",
		Method:      http.MethodPut,
		Path:        "/v1/router/configs/{id}",
		Summary:     "Replace a router config",
		Description: "Every field is written, so the body is what the config now is rather than what changed " +
			"about it. Sockets already open keep the options they were started with.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The config as it now is"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.updateRouterConfig)
	huma.Register(api, huma.Operation{
		OperationID: "deleteRouterConfig",
		Method:      http.MethodDelete,
		Path:        "/v1/router/configs/{id}",
		Summary:     "Delete a router config",
		Description: "The requests that ran under it still name it, so the config stops being usable rather " +
			"than stops having existed.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The config is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteRouterConfig)
}

type listRouterConfigsRequest struct{}

type listRouterConfigsResponse struct {
	Body []RouterConfig `nullable:"false"`
}

type createRouterConfigRequest struct {
	Body *RouterConfigRequest `required:"true"`
}

type createRouterConfigResponse struct {
	Body RouterConfig
}

type getRouterConfigRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getRouterConfigResponse struct {
	Body RouterConfig
}

type updateRouterConfigRequest struct {
	Id   string               `path:"id" doc:"The resource, as returned when it was created."`
	Body *RouterConfigRequest `required:"true"`
}

type updateRouterConfigResponse struct {
	Body RouterConfig
}

type deleteRouterConfigRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

// RouterConfigRequest is the RouterConfigRequest schema.
type RouterConfigRequest struct {
	Llm    *LlmOptions    `json:"llm,omitempty"`
	Name   string         `json:"name" doc:"What the config is called, which is unique among the customer's own."`
	Search *SearchOptions `json:"search,omitempty"`
	Sts    *StsOptions    `json:"sts,omitempty"`
	Stt    *SttOptions    `json:"stt,omitempty"`
	Tts    *TtsOptions    `json:"tts,omitempty"`
}
