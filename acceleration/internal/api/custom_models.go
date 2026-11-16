package api

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
)

// errNoCustomModels is what the model paths say on a deployment without a database.
var errNoCustomModels = notConfigured("models of your own are not available: no database configured")

// errNoModelKeys is a model given an api key on a deployment that could only store it in
// the clear.
var errNoModelKeys = notConfigured("a model's api key cannot be stored: no key encryption key configured")

// errUnknownCustomModel is what a caller is told about a model that is not theirs, or not there.
var errUnknownCustomModel = APIError{
	Type: ErrorTypeNotFound, Code: codeModelNotFound,
	Message: "there is no such model",
}

// errModelNameTaken is a create or a rename to a name another of the customer's models has.
var errModelNameTaken = APIError{
	Type: ErrorTypeConflict, Code: codeNameTaken,
	Message: "a model with this name already exists",
}

// listCustomModels returns a page of the calling customer's own models, newest first.
func (s *Server) listCustomModels(ctx context.Context, request *listCustomModelsRequest) (*listCustomModelsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.configs == nil {
		return nil, errNoCustomModels
	}

	after, err := decodeCursor[store.CustomModelPosition](request.Cursor.ptr())
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	limit := store.CustomModelLimit(value(request.Limit.ptr()))
	rows, err := s.store.CustomModels(ctx, customerID, limit, after)
	if err != nil {
		return nil, err
	}

	rows, more := page(rows, limit)
	listed := CustomModelPage{Items: make([]CustomModel, 0, len(rows)), HasMore: more}
	for _, row := range rows {
		listed.Items = append(listed.Items, customModelOf(row))
	}
	if more {
		last := rows[len(rows)-1]
		listed.NextCursor = encodeCursor(store.CustomModelPosition{CreatedAt: last.CreatedAt, ID: last.ID})
	}
	return &listCustomModelsResponse{Body: listed}, nil
}

// createCustomModel stores a model the customer serves themselves.
func (s *Server) createCustomModel(ctx context.Context, request *createCustomModelRequest) (*createCustomModelResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.configs == nil {
		return nil, errNoCustomModels
	}

	model := store.CustomModel{CustomerID: customerID}
	if failure := s.fillCustomModel(ctx, &model, request.Body); failure != nil {
		return nil, failure
	}
	if err := s.configs.CreateCustomModel(ctx, &model); err != nil {
		return nil, storeFailure(err, errModelNameTaken)
	}
	return &createCustomModelResponse{Body: customModelOf(model)}, nil
}

// getCustomModel returns one of the customer's models.
func (s *Server) getCustomModel(ctx context.Context, request *getCustomModelRequest) (*getCustomModelResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.configs == nil {
		return nil, errNoCustomModels
	}

	model, err := s.store.CustomModel(ctx, customerID, request.Id)
	if errors.Is(err, store.ErrNoCustomModel) {
		return nil, errUnknownCustomModel
	}
	if err != nil {
		return nil, err
	}
	return &getCustomModelResponse{Body: customModelOf(model)}, nil
}

// updateCustomModel replaces a model. An api key left out is kept.
func (s *Server) updateCustomModel(ctx context.Context, request *updateCustomModelRequest) (*updateCustomModelResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.configs == nil {
		return nil, errNoCustomModels
	}

	model, err := s.store.CustomModel(ctx, customerID, request.Id)
	if errors.Is(err, store.ErrNoCustomModel) {
		return nil, errUnknownCustomModel
	}
	if err != nil {
		return nil, err
	}
	if failure := s.fillCustomModel(ctx, &model, request.Body); failure != nil {
		return nil, failure
	}
	if err := s.configs.UpdateCustomModel(ctx, &model); err != nil {
		if errors.Is(err, store.ErrNoCustomModel) {
			return nil, errUnknownCustomModel
		}
		return nil, storeFailure(err, errModelNameTaken)
	}
	return &updateCustomModelResponse{Body: customModelOf(model)}, nil
}

// deleteCustomModel forgets a model. Sessions naming it are refused from then on.
func (s *Server) deleteCustomModel(ctx context.Context, request *deleteCustomModelRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.configs == nil {
		return nil, errNoCustomModels
	}

	if err := s.configs.DeleteCustomModel(ctx, customerID, request.Id); err != nil {
		if errors.Is(err, store.ErrNoCustomModel) {
			return nil, errUnknownCustomModel
		}
		return nil, err
	}
	return nil, nil
}

// fillCustomModel writes a request over a model, checking what Huma's tags cannot: where
// the endpoint is, and what the data policy says.
func (s *Server) fillCustomModel(ctx context.Context, model *store.CustomModel, request *CustomModelRequest) error {
	if request == nil {
		return invalidRequest("a request body is required")
	}
	if failure := s.endpointComplaint(ctx, request.BaseUrl); failure != nil {
		return failure
	}
	handling := options.DataHandling{
		TrainsOnData: options.Claim(text(request.TrainsOnData)),
		Retention:    options.Retention(text(request.Retention)),
	}
	if (handling.TrainsOnData != "" || handling.Retention != "") && !handling.Valid() {
		return invalidRequest("trains_on_data and retention are declared together: yes, no or unknown, " +
			"and none or a duration such as 30d")
	}

	if request.ApiKey != nil {
		model.APIKeySealed, model.KEKVersion = nil, 0
		if *request.ApiKey != "" {
			if s.secrets == nil {
				return errNoModelKeys
			}
			sealed, err := s.secrets.SealWithAAD(*request.ApiKey, []byte(model.CustomerID))
			if err != nil {
				return err
			}
			model.APIKeySealed, model.KEKVersion = sealed, s.secrets.CurrentVersion()
		}
	}
	model.Name = request.Name
	model.BaseURL = request.BaseUrl
	model.Model = request.Model
	model.ContextWindow = value(request.ContextWindow)
	model.InputModalities = value(request.InputModalities)
	model.PerMillionInputTokens = value(request.PerMillionInputTokens)
	model.PerMillionOutputTokens = value(request.PerMillionOutputTokens)
	model.TrainsOnData = string(handling.TrainsOnData)
	model.Retention = string(handling.Retention)
	return nil
}

// endpointComplaint refuses an endpoint the router should not be dialing. A shared router
// only reaches public https addresses, so one tenant cannot point it at another's network;
// a self-hosted one that allows private endpoints takes any http or https URL.
func (s *Server) endpointComplaint(ctx context.Context, raw string) error {
	if !s.privateModels {
		if err := egress.ValidatePublicHTTPSURL(ctx, raw); err != nil {
			return invalidRequest("base_url must be a public https URL: " + err.Error())
		}
		return nil
	}
	parsed, err := url.Parse(raw)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return invalidRequest("base_url must be an http or https URL")
	}
	return nil
}

// customModelOf renders a model for the wire. The api key never goes back out.
func customModelOf(model store.CustomModel) CustomModel {
	return CustomModel{
		Id:                     model.ID,
		Name:                   model.Name,
		Target:                 routing.CustomProvider + "/" + model.Name,
		BaseUrl:                model.BaseURL,
		Model:                  model.Model,
		HasApiKey:              len(model.APIKeySealed) > 0,
		ContextWindow:          model.ContextWindow,
		InputModalities:        append([]string{}, model.InputModalities...),
		PerMillionInputTokens:  model.PerMillionInputTokens,
		PerMillionOutputTokens: model.PerMillionOutputTokens,
		TrainsOnData:           optional(model.TrainsOnData),
		Retention:              optional(model.Retention),
		CreatedAt:              model.CreatedAt,
		UpdatedAt:              model.UpdatedAt,
	}
}

// registerCustomModels declares the operations served in custom_models.go.
func (s *Server) registerCustomModels(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listCustomModels",
		Method:      http.MethodGet,
		Path:        "/v1/agents/models",
		Summary:     "The language models the calling customer serves themselves",
		Description: "Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's models, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listCustomModels)
	huma.Register(api, huma.Operation{
		OperationID: "createCustomModel",
		Method:      http.MethodPost,
		Path:        "/v1/agents/models",
		Summary:     "Add a model of the customer's own",
		Description: "A model the customer serves behind an OpenAI-compatible chat completions endpoint: a " +
			"fine-tune on Baseten, a vLLM or SGLang deployment, a provider the router does not " +
			"route. A router config or session names it as `custom/<name>`, and the router calls it " +
			"as it calls the models it routes. A shared router only dials public https endpoints.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The model was stored"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusConflict, http.StatusNotImplemented},
	}, s.createCustomModel)
	huma.Register(api, huma.Operation{
		OperationID: "getCustomModel",
		Method:      http.MethodGet,
		Path:        "/v1/agents/models/{id}",
		Summary:     "One of the customer's models",
		Description: "Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The model"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getCustomModel)
	huma.Register(api, huma.Operation{
		OperationID: "updateCustomModel",
		Method:      http.MethodPut,
		Path:        "/v1/agents/models/{id}",
		Summary:     "Replace a model",
		Description: "Everything is replaced but the api key, which is kept when left out and removed when " +
			"sent empty. Renaming a model breaks every config that names it by its old name.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The model as it now is"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict, http.StatusNotImplemented},
	}, s.updateCustomModel)
	huma.Register(api, huma.Operation{
		OperationID: "deleteCustomModel",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/models/{id}",
		Summary:     "Delete a model",
		Description: "Sessions that name it are refused from then on.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The model is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteCustomModel)
}

type listCustomModelsRequest struct {
	Limit  optionalParam[int]    `query:"limit" doc:"Up to 200. Omitted is 25." minimum:"1" maximum:"200"`
	Cursor optionalParam[string] "query:\"cursor\" doc:\"The `next_cursor` of the previous page. Omitted is the first page.\""
}

type listCustomModelsResponse struct {
	Body CustomModelPage
}

type createCustomModelRequest struct {
	Body *CustomModelRequest `required:"true"`
}

type createCustomModelResponse struct {
	Body CustomModel
}

type getCustomModelRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getCustomModelResponse struct {
	Body CustomModel
}

type updateCustomModelRequest struct {
	Id   string              `path:"id" doc:"The resource, as returned when it was created."`
	Body *CustomModelRequest `required:"true"`
}

type updateCustomModelResponse struct {
	Body CustomModel
}

type deleteCustomModelRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

// CustomModelRequest is the CustomModelRequest schema.
type CustomModelRequest struct {
	Name                   string    "json:\"name\" doc:\"What a config names the model by, as `custom/<name>`. Unique among the customer's models.\" pattern:\"^[a-z0-9][a-z0-9._-]{0,63}$\" example:\"support-qwen\""
	BaseUrl                string    `json:"base_url" doc:"The endpoint root, up to and including /v1, that chat completions are posted under." minLength:"1" example:"https://model-abc123.api.baseten.co/environments/production/sync/v1"`
	Model                  string    `json:"model" doc:"The id the endpoint serves the weights under." minLength:"1" example:"Qwen/Qwen3.8-27B"`
	ApiKey                 *string   `json:"api_key,omitempty" doc:"Sent as a bearer token. Stored sealed and never returned. Left out of an update keeps the one stored; empty removes it." writeOnly:"true"`
	ContextWindow          *int64    `json:"context_window,omitempty" doc:"Tokens the model accepts, so the router can refuse a conversation that would not fit. Omitted is unknown." minimum:"0"`
	InputModalities        *[]string `json:"input_modalities,omitempty" doc:"Input kinds beyond text the model accepts, e.g. image."`
	PerMillionInputTokens  *float64  `json:"per_million_input_tokens,omitempty" doc:"What the host bills per million input tokens in USD, so usage can say what a conversation cost. Omitted is free." minimum:"0"`
	PerMillionOutputTokens *float64  `json:"per_million_output_tokens,omitempty" doc:"What the host bills per million output tokens in USD. Omitted is free." minimum:"0"`
	TrainsOnData           *string   `json:"trains_on_data,omitempty" doc:"Whether the host trains on what it is sent. Declared with retention, or a session with a data policy is never routed here." enum:"yes,no,unknown"`
	Retention              *string   `json:"retention,omitempty" doc:"How long the host keeps what it is sent: none, or a duration such as 30d or 24h." example:"none"`
}

// CustomModel is the CustomModel schema.
type CustomModel struct {
	Id                     string    `json:"id"`
	Name                   string    `json:"name"`
	Target                 string    `json:"target" doc:"What a router config or session names the model by." example:"custom/support-qwen"`
	BaseUrl                string    `json:"base_url"`
	Model                  string    `json:"model"`
	HasApiKey              bool      `json:"has_api_key" doc:"Whether a key is stored. The key itself is never returned."`
	ContextWindow          int64     `json:"context_window"`
	InputModalities        []string  `json:"input_modalities" nullable:"false"`
	PerMillionInputTokens  float64   `json:"per_million_input_tokens"`
	PerMillionOutputTokens float64   `json:"per_million_output_tokens"`
	TrainsOnData           *string   `json:"trains_on_data,omitempty"`
	Retention              *string   `json:"retention,omitempty"`
	CreatedAt              time.Time `json:"created_at"`
	UpdatedAt              time.Time `json:"updated_at"`
}

// CustomModelPage is the CustomModelPage schema.
type CustomModelPage struct {
	HasMore    bool          `json:"has_more"`
	Items      []CustomModel `json:"items" nullable:"false"`
	NextCursor *string       "json:\"next_cursor,omitempty\" doc:\"Pass as `cursor` for the next page. Absent on the last one.\""
}
