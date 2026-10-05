package api

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const noConnectors = "connector definitions are not available: no database configured"

// mcpSource is both the tool source a custom definition runs and the endpoint role it runs
// against, as the built-in manifests write them (internal/connectors/providers/slack.yaml
// and linear.yaml: sources: [{kind: mcp, endpoint: mcp}]). MCP is the only source so far
// (architecture doc, «Layers and interfaces», the Source registry's first member).
const mcpSource = "mcp"

// Connector is one connector as the catalog shows it and a connection is created from: the
// revision of its definition (store.ConnectorDefinition) the request read. It is the
// non-secret part of the manifest: what a caller chooses between (schemes, inputs, scopes,
// who owns the OAuth client). Everything the router reads to connect is left out: endpoints, vars, authorize and token parameters, capture and identity rules,
// refresh and rate limits, sources, hooks and the operator's client variables.
type Connector struct {
	ID          string           `json:"id" doc:"Unique among the built-ins and the app's own. A custom definition's starts with custom_, and a built-in's never does."`
	Revision    int              `json:"revision" readOnly:"true" doc:"The manifest's revision. A connection is created from the newest one and keeps reading it until it is reconnected."`
	Name        string           `json:"name"`
	Category    string           `json:"category,omitempty"`
	Description string           `json:"description,omitempty"`
	Custom      bool             `json:"custom" readOnly:"true" doc:"The app's own definition rather than a built-in."`
	Schemes     []string         `json:"schemes" doc:"How a connection may authenticate, such as oauth2_code."`
	Inputs      []ConnectorInput `json:"inputs" doc:"What a connection is created with, such as a region or a shop."`
	Scopes      []string         `json:"scopes" doc:"The scopes a consent asks for."`
	Client      ConnectorClient  `json:"client"`
	CreatedAt   time.Time        `json:"created_at" readOnly:"true" doc:"When this revision was stored."`
}

func (*Connector) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A connector: an account elsewhere an agent may reach, built in or the " +
		"app's own. Only what a caller chooses between is shown. Endpoints, how an account is " +
		"recognised, refresh and rate limits stay with the router."
	return schema
}

// ConnectorInput is one value a connection is created with.
type ConnectorInput struct {
	Name    string   `json:"name"`
	Enum    []string `json:"enum,omitempty" doc:"The values it may take. Absent when a pattern decides."`
	Pattern string   `json:"pattern,omitempty" doc:"A regular expression the whole value must match."`
	Default string   `json:"default,omitempty" doc:"Used when the connection gives no value. An input without one is required."`
}

// ConnectorClient is how the OAuth client a connection uses is registered, and how that client
// authenticates.
type ConnectorClient struct {
	Registration []ConnectorClientRegistration `json:"registration,omitempty" uniqueItems:"true" doc:"The client registration mechanisms the connector allows, tried as the scheme orders them. Empty when the connector needs no OAuth client."`
	AuthMethod   ConnectorClientAuthMethod     `json:"auth_method,omitempty"`
	// The algorithms core.Manifest.Validate accepts (assertionAlgs in
	// internal/connectors/core/manifest.go).
	Alg string `json:"alg,omitempty" enum:"RS256,PS256" doc:"How a private_key_jwt assertion is signed, and set only for it."`
}

func (*ConnectorClient) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "How the OAuth client a connection uses is registered, and how the client " +
		"authenticates at the token endpoint."
	schema.AdditionalProperties = false
	return schema
}

// ConnectorClientRegistration is one way an OAuth client is registered.
type ConnectorClientRegistration string

func (ConnectorClientRegistration) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectorClientRegistration",
		"operator is this deployment's own client, customer one the app registered, dcr one "+
			"registered on the fly (RFC 7591) and cimd one named by a metadata document.",
		string(core.ClientOperator), string(core.ClientCustomer), string(core.ClientDCR), string(core.ClientCIMD))
}

// ConnectorClientAuthMethod is how an OAuth client authenticates at the token endpoint.
type ConnectorClientAuthMethod string

func (ConnectorClientAuthMethod) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectorClientAuthMethod",
		"How the OAuth client authenticates at the token endpoint, as the IANA OAuth token "+
			"endpoint authentication methods registry spells it.",
		string(core.AuthNone), string(core.AuthClientSecretPost), string(core.AuthClientSecretBasic),
		string(core.AuthPrivateKeyJWT), string(core.AuthTLSClientAuth))
}

// ConnectorPage is a page of connectors.
type ConnectorPage struct {
	Items      []Connector `json:"items"`
	HasMore    bool        `json:"has_more"`
	NextCursor *string     `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page, with the same q. Absent on the last one."`
}

// CustomConnectorRequest is a custom MCP server for the app's agents to connect to.
//
// The limits on name, category and description are the prototype's
// (CreateConnectorDefinitionRequest in api/openapi.yaml on codex/connector-support at
// cf62af0d). The id is its pattern without the hyphen, which core.Manifest.Validate refuses
// in an id: custom_ and up to 57 more characters, 64 in all. The prototype set no length on
// the endpoint; 2048 is a guess at a URL nobody needs to exceed, not a measured one. 100
// scopes is well over the longest list shipped, Slack's 29 (providers/slack.yaml), and is
// not measured either. A scope is RFC 6749 section 3.3's scope-token, %x21 / %x23-5B /
// %x5D-7E, so it can hold neither a space nor a quote. The schemes have no count: each must
// be a registered one and none may repeat, which bounds them.
type CustomConnectorRequest struct {
	ID          string           `json:"id" pattern:"^custom_[a-z][a-z0-9_]{0,56}$" patternDescription:"custom_ then a lowercase letter, then up to 56 lowercase letters, digits or underscores" doc:"Starts with custom_, which no built-in does, so a custom definition never stands in for one. Creating an id that exists stores the next revision, unless the newest already says the same."`
	Name        string           `json:"name" minLength:"1" maxLength:"120"`
	Category    string           `json:"category,omitempty" maxLength:"80"`
	Description string           `json:"description,omitempty" maxLength:"1000"`
	Endpoint    string           `json:"endpoint" maxLength:"2048" doc:"The MCP server, over Streamable HTTP: a public https URL without userinfo, query or fragment. An address on a private network, loopback or link-local is refused."`
	Schemes     []string         `json:"schemes" minItems:"1" doc:"How a connection may authenticate. Each must be a scheme this deployment has."`
	Scopes      []string         `json:"scopes,omitempty" maxItems:"100" uniqueItems:"true" pattern:"^[!#-\\[\\]-~]+$" patternDescription:"an RFC 6749 scope token" doc:"The scopes a consent asks for, each an RFC 6749 scope token."`
	Client      *ConnectorClient `json:"client,omitempty"`
}

func (*CustomConnectorRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A custom MCP server for the app's agents to connect to. An unknown " +
		"field is refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

type listConnectorsRequest struct {
	// 120 is the longest name a custom definition may have, which is what a picker's
	// search box looks for.
	Q string `query:"q" maxLength:"120" doc:"Keeps the connectors whose id, name, category or description holds this, ignoring case."`
	// 200 and 25 are store.ConnectorDefinitionLimit's.
	Limit  int    `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 25."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
}

type listConnectorsResponse struct {
	Body ConnectorPage
}

type getConnectorRequest struct {
	ID string `path:"id" doc:"The connector, such as slack or custom_crm."`
}

type createConnectorRequest struct {
	Body CustomConnectorRequest
}

type connectorResponse struct {
	Body Connector
}

// registerConnectors declares the connector definition operations. All three are
// server-side only: the catalog is what an app's backend and dashboard choose from when
// they set connectors up, and an end user is sent to a consent by that backend rather
// than picking a connector themselves.
func (s *Server) registerConnectors(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listConnectors",
		Method:      http.MethodGet,
		Path:        "/v1/agents/connectors",
		Summary:     "List or search connectors",
		Description: "The built-ins first, then the app's own, each by id and at its newest " +
			"revision. `q` keeps the ones whose id, name, category or description holds it.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "A page of connectors"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listConnectors)
	huma.Register(api, huma.Operation{
		OperationID: "getConnector",
		Method:      http.MethodGet,
		Path:        "/v1/agents/connectors/{id}",
		Summary:     "Read a connector",
		Description: "A built-in or one of the app's own, at its newest revision. Another " +
			"app's custom connector is not found.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The connector"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getConnector)
	huma.Register(api, huma.Operation{
		OperationID: "createConnector",
		Method:      http.MethodPost,
		Path:        "/v1/agents/connectors",
		Summary:     "Add a custom MCP connector",
		Description: "Stores a custom MCP server as one of the app's connectors. Sending an " +
			"id the app already has stores its next revision, which connections pick up when " +
			"they reconnect; sending the same definition again changes nothing. A built-in " +
			"cannot be changed this way.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The connector as stored"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.createConnector)
}

// listConnectors lists the built-ins and the caller's own, a page at a time.
func (s *Server) listConnectors(ctx context.Context, request *listConnectorsRequest) (*listConnectorsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConnectors)
	}
	cursor, err := decodeCursor[store.ConnectorDefinitionPosition](&request.Cursor)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	found, err := s.store.ListConnectorDefinitions(ctx, customerID, store.ConnectorDefinitionFilter{
		Text:   strings.TrimSpace(request.Q),
		Cursor: cursor,
		Limit:  request.Limit,
	})
	if err != nil {
		return nil, err
	}

	kept, more := page(found, store.ConnectorDefinitionLimit(request.Limit))
	listed := ConnectorPage{Items: make([]Connector, 0, len(kept)), HasMore: more}
	for _, definition := range kept {
		listed.Items = append(listed.Items, connectorOf(definition))
	}
	if more {
		last := kept[len(kept)-1]
		listed.NextCursor = encodeCursor(store.ConnectorDefinitionPosition{
			Custom: last.CustomerID != store.BuiltinCustomer,
			ID:     last.ID,
		})
	}
	return &listConnectorsResponse{Body: listed}, nil
}

// getConnector reads a built-in or one of the caller's own.
func (s *Server) getConnector(ctx context.Context, request *getConnectorRequest) (*connectorResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConnectors)
	}
	definition, err := s.store.LatestConnectorDefinition(ctx, customerID, request.ID)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		return nil, huma.Error404NotFound("no such connector")
	}
	if err != nil {
		return nil, err
	}
	return &connectorResponse{Body: connectorOf(definition)}, nil
}

// createConnector stores a custom MCP definition as the caller's own.
func (s *Server) createConnector(ctx context.Context, request *createConnectorRequest) (*connectorResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConnectors)
	}
	manifest, err := s.customManifest(ctx, request.Body)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	// Every refusal a caller can cause is above, so what the store refuses here is a bug.
	definition, err := s.store.CreateConnectorDefinition(ctx, customerID, manifest)
	if err != nil {
		return nil, err
	}
	return &connectorResponse{Body: connectorOf(definition)}, nil
}

// customManifest is the manifest a custom MCP definition is stored as, or why it cannot be.
func (s *Server) customManifest(ctx context.Context, sent CustomConnectorRequest) (core.Manifest, error) {
	known := slices.Sorted(maps.Keys(s.connectors.Schemes))
	for _, scheme := range sent.Schemes {
		if !slices.Contains(known, scheme) {
			if len(known) == 0 {
				return core.Manifest{}, fmt.Errorf("scheme %q is not one this deployment has: it has none yet", scheme)
			}
			return core.Manifest{}, fmt.Errorf("scheme %q is not one this deployment has (%s)",
				scheme, strings.Join(known, ", "))
		}
	}
	var client core.ClientPolicy
	if sent.Client != nil {
		for _, registration := range sent.Client.Registration {
			// An operator client is the deployment's own, read from the variables a
			// built-in's client.env names, and the deployment has none registered with an
			// app's own server.
			if core.ClientRegistration(registration) == core.ClientOperator {
				return core.Manifest{}, errors.New("client.registration cannot be operator for a custom connector: this deployment has no client registered with it")
			}
			client.Registration = append(client.Registration, core.ClientRegistration(registration))
		}
		client.AuthMethod = core.ClientAuthMethod(sent.Client.AuthMethod)
		client.Alg = sent.Client.Alg
	}
	// oauth2_code tries only the mechanisms client.registration names and fails with ErrNoClient when it
	// names none (pickClient in internal/connectors/schemes/oauth2code/client.go), so such a
	// definition could be stored and never connected.
	if slices.Contains(sent.Schemes, "oauth2_code") && len(client.Registration) == 0 {
		return core.Manifest{}, errors.New("client.registration is required with oauth2_code: name how the OAuth client is registered (customer, cimd or dcr)")
	}
	manifest := core.Manifest{
		ID: sent.ID,
		// Validated at the first revision; the store numbers the one it is stored as.
		Revision:    1,
		Name:        strings.TrimSpace(sent.Name),
		Category:    strings.TrimSpace(sent.Category),
		Description: strings.TrimSpace(sent.Description),
		Endpoints:   map[string]string{mcpSource: sent.Endpoint},
		Schemes:     sent.Schemes,
		Client:      client,
		Scopes:      core.ScopePolicy{List: sent.Scopes},
		Sources:     []core.SourceRule{{Kind: mcpSource, Endpoint: mcpSource}},
	}
	if err := manifest.Validate(); err != nil {
		return core.Manifest{}, err
	}
	// Last, since it resolves the host: the checks above cost nothing.
	// One answer for every refusal: egress's own errors tell a name that does not resolve from
	// one that resolves to a private address, which would let a caller probe the names the
	// router's resolver knows.
	if err := egress.ValidatePublicHTTPSURL(ctx, sent.Endpoint); err != nil {
		return core.Manifest{}, errors.New("endpoint must be a public https URL without userinfo, query or fragment")
	}
	return manifest, nil
}

// connectorOf is the part of a stored definition a caller is shown. Each field is
// copied by name, so a field added to the manifest stays hidden until it is added here.
func connectorOf(definition store.ConnectorDefinition) Connector {
	manifest := definition.Manifest
	inputs := make([]ConnectorInput, 0, len(manifest.Inputs))
	for _, in := range manifest.Inputs {
		inputs = append(inputs, ConnectorInput{Name: in.Name, Enum: in.Enum, Pattern: in.Pattern, Default: in.Default})
	}
	registrations := make([]ConnectorClientRegistration, 0, len(manifest.Client.Registration))
	for _, registration := range manifest.Client.Registration {
		registrations = append(registrations, ConnectorClientRegistration(registration))
	}
	return Connector{
		ID:          definition.ID,
		Revision:    definition.Revision,
		Name:        definition.Name,
		Category:    definition.Category,
		Description: definition.Description,
		Custom:      definition.CustomerID != store.BuiltinCustomer,
		Schemes:     append([]string{}, manifest.Schemes...),
		Inputs:      inputs,
		Scopes:      append([]string{}, manifest.Scopes.List...),
		Client: ConnectorClient{
			Registration: registrations,
			AuthMethod:   ConnectorClientAuthMethod(manifest.Client.AuthMethod),
			Alg:          manifest.Client.Alg,
		},
		CreatedAt: definition.CreatedAt,
	}
}
