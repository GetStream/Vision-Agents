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
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

var errNoConnectors = notConfigured("connector definitions are not available: no database configured")

// errNoCustomConnector answers a delete of an id the app has no custom connector under. A
// built-in is answered the same, since an app cannot delete one.
var errNoCustomConnector = notFound("the app has no custom connector with this id; a built-in cannot be deleted")

// usesNamed is how many connections and how many bindings a refused delete names, the rest
// counted. Ten keeps the message a line a person reads; it is not measured.
const usesNamed = 10

// mcpSource is both the tool source a custom definition runs and the endpoint role it runs
// against, as the built-in manifests write them (internal/connectors/providers/slack.yaml
// and linear.yaml: sources: [{kind: mcp, endpoint: mcp}]). MCP is the only source so far
// (architecture doc, «Layers and interfaces», the Source registry's first member).
const mcpSource = "mcp"

// Connector is one connector as the catalog shows it and a connection is created from: the
// revision of its definition (store.ConnectorDefinition) the request read. It is the
// non-secret part of the manifest: what a caller chooses between (schemes, inputs, scopes,
// who owns the OAuth client). Everything the router reads to connect is left out: a built-in's
// endpoints, vars, authorize and token parameters, capture and identity rules, refresh and
// rate limits, sources, hooks and the operator's client variables. A custom connector's
// endpoint is left out too: an MCP URL can carry a secret in its path (AI-837).
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
	Setup       *ConnectorSetup  `json:"setup,omitempty" doc:"What a person does at the provider before the first consent, such as registering an OAuth client. Absent when the manifest says nothing."`
	RedirectURI string           `json:"redirect_uri,omitempty" readOnly:"true" format:"uri" doc:"The redirect URI an OAuth client registered for this connector has to list: where every consent of this deployment sends the browser back to, ROUTER_PUBLIC_URL followed by /v1/agents/connectors/oauth/callback. Only on a connector that connects with oauth2_code, and absent when ROUTER_PUBLIC_URL is not set, since no consent can start then."`
	CreatedAt   time.Time        `json:"created_at" readOnly:"true" doc:"When this revision was stored."`
}

// ConnectorSetup is a provider's setup page and the steps a person takes there, as a dashboard
// shows them beside the form a client is pasted into.
type ConnectorSetup struct {
	URL   string               `json:"url,omitempty" format:"uri" doc:"Where the steps start, a page of the provider's."`
	Steps []ConnectorSetupStep `json:"steps" nullable:"false" doc:"In order."`
}

func (*ConnectorSetup) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What a person does at the provider before the first consent."
	schema.AdditionalProperties = false
	return schema
}

// ConnectorSetupStep is one step of a provider's setup.
type ConnectorSetupStep struct {
	Title       string `json:"title"`
	Description string `json:"description"`
}

func (*ConnectorSetupStep) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One step of a provider's setup."
	schema.AdditionalProperties = false
	return schema
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
	Registration []ConnectorClientRegistrationMethod `json:"registration,omitempty" uniqueItems:"true" doc:"The client registration mechanisms the connector allows, tried as the scheme orders them. Empty when the connector needs no OAuth client."`
	AuthMethod   ConnectorClientAuthMethod           `json:"auth_method,omitempty"`
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

// ConnectorClientRegistrationMethod is one way an OAuth client is registered.
type ConnectorClientRegistrationMethod string

func (ConnectorClientRegistrationMethod) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectorClientRegistrationMethod",
		"operator is this deployment's own client, customer one the app registered, managed "+
			"one the router created for the app (PUT /v1/agents/connectors/{id}/provider-app), "+
			"dcr one registered on the fly (RFC 7591) and cimd one named by a metadata document.",
		string(core.ClientOperator), string(core.ClientCustomer), string(core.ClientManaged), string(core.ClientDCR), string(core.ClientCIMD))
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

type deleteConnectorRequest struct {
	ID    string `path:"id" doc:"The app's custom connector, such as custom_crm."`
	Force bool   `query:"force" doc:"Delete it even while connections or agent config bindings use it. Its connections are deleted with it, and the bindings are left in place, naming a connector that no longer exists."`
}

type createConnectorRequest struct {
	Body CustomConnectorRequest
}

type connectorResponse struct {
	Body Connector
}

// registerConnectors declares the connector definition operations. All four are
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
	huma.Register(api, huma.Operation{
		OperationID:   "deleteConnector",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/connectors/{id}",
		Summary:       "Delete a custom connector",
		DefaultStatus: http.StatusNoContent,
		Description: "Deletes one of the app's own connectors, every revision of it, with the " +
			"app's OAuth client for it. A built-in cannot be deleted and is not found. A " +
			"connector a live connection was made from, or an agent config binds, is refused " +
			"with a 409 naming them, unless force is set: then its connections are deleted as a " +
			"forced connection delete deletes one, credentials dropped at once, and the " +
			"bindings are left in place. The same id may be created again, from revision 1.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"204": {Description: "The connector is deleted"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict},
	}, s.deleteConnector)
}

// listConnectors lists the built-ins and the caller's own, a page at a time.
func (s *Server) listConnectors(ctx context.Context, request *listConnectorsRequest) (*listConnectorsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnectors
	}
	cursor, err := decodeCursor[store.ConnectorDefinitionPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
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
		listed.Items = append(listed.Items, connectorOf(definition, s.publicURL))
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
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnectors
	}
	definition, err := s.store.LatestConnectorDefinition(ctx, customerID, request.ID)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		return nil, notFound("no such connector")
	}
	if err != nil {
		return nil, err
	}
	return &connectorResponse{Body: connectorOf(definition, s.publicURL)}, nil
}

// createConnector stores a custom MCP definition as the caller's own.
func (s *Server) createConnector(ctx context.Context, request *createConnectorRequest) (*connectorResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnectors
	}
	manifest, err := s.customManifest(ctx, request.Body)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	// Every refusal a caller can cause is above, so what the store refuses here is a bug.
	definition, err := s.store.CreateConnectorDefinition(ctx, customerID, manifest)
	if err != nil {
		return nil, err
	}
	return &connectorResponse{Body: connectorOf(definition, s.publicURL)}, nil
}

// deleteConnector deletes one of the caller's own definitions, refused while something uses
// it unless forced (store.DeleteConnectorDefinition).
func (s *Server) deleteConnector(ctx context.Context, request *deleteConnectorRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnectors
	}
	deleted, err := s.store.DeleteConnectorDefinition(ctx, customerID, request.ID, request.Force)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		return nil, errNoCustomConnector
	}
	// 409: the request conflicts with the state of the resource, which the caller can change
	// and retry (RFC 9110 section 15.5.10), as a bound connection's delete is answered.
	if errors.Is(err, store.ErrConnectorDefinitionInUse) {
		return nil, connectorInUse(request.ID, deleted.Uses)
	}
	if err != nil {
		return nil, err
	}
	for _, connection := range deleted.Connections {
		s.connectionDeleted(ctx, customerID, connection)
	}
	return nil, nil
}

// connectorInUse is the refusal of an unforced delete of a connector uses still name: the
// connections by id and the bindings by config name and alias, usesNamed of each.
func connectorInUse(id string, uses store.ConnectorUses) APIError {
	var users []string
	if len(uses.Connections) > 0 {
		users = append(users, "connections "+namedFew(uses.Connections))
	}
	if len(uses.Bindings) > 0 {
		bindings := make([]string, 0, len(uses.Bindings))
		for _, binding := range uses.Bindings {
			bindings = append(bindings, fmt.Sprintf("%q as %s", binding.ConfigName, binding.Binding))
		}
		users = append(users, "agent config bindings "+namedFew(bindings))
	}
	return conflict(fmt.Sprintf("%s is used by %s: delete or unbind them first, or delete with force=true",
		id, strings.Join(users, " and by ")))
}

// namedFew is the first usesNamed of names, and how many more there are.
func namedFew(names []string) string {
	if len(names) <= usesNamed {
		return strings.Join(names, ", ")
	}
	return fmt.Sprintf("%s and %d more", strings.Join(names[:usesNamed], ", "), len(names)-usesNamed)
}

// customManifest is the manifest a custom MCP definition is stored as, or why it cannot be.
func (s *Server) customManifest(ctx context.Context, sent CustomConnectorRequest) (core.Manifest, error) {
	known := slices.Sorted(maps.Keys(s.connectors.Schemes))
	for _, scheme := range sent.Schemes {
		if !slices.Contains(known, scheme) {
			if len(known) == 0 {
				return core.Manifest{}, stack.Wrap(fmt.Errorf("scheme %q is not one this deployment has: it has none yet", scheme))
			}
			return core.Manifest{}, stack.Wrap(fmt.Errorf("scheme %q is not one this deployment has (%s)",
				scheme, strings.Join(known, ", ")))
		}
	}
	var client core.ClientPolicy
	if sent.Client != nil {
		for _, registration := range sent.Client.Registration {
			// An operator client is the deployment's own, read from the variables a
			// built-in's client.env names, and the deployment has none registered with an
			// app's own server.
			if core.ClientRegistrationMethod(registration) == core.ClientOperator {
				return core.Manifest{}, stack.Wrap(errors.New("client.registration cannot be operator for a custom connector: this deployment has no client registered with it"))
			}
			client.Registration = append(client.Registration, core.ClientRegistrationMethod(registration))
		}
		client.AuthMethod = core.ClientAuthMethod(sent.Client.AuthMethod)
		client.Alg = sent.Client.Alg
	}
	// oauth2_code tries only the mechanisms client.registration names and fails with ErrNoClient when it
	// names none (pickClient in internal/connectors/schemes/oauth2code/client.go), so such a
	// definition could be stored and never connected.
	if slices.Contains(sent.Schemes, "oauth2_code") && len(client.Registration) == 0 {
		return core.Manifest{}, stack.Wrap(errors.New("client.registration is required with oauth2_code: name how the OAuth client is registered (customer, cimd or dcr)"))
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
		return core.Manifest{}, stack.Wrap(errors.New("endpoint must be a public https URL without userinfo, query or fragment"))
	}
	return manifest, nil
}

// connectorOf is the part of a stored definition a caller is shown. Each field is
// copied by name, so a field added to the manifest stays hidden until it is added here.
// publicURL is the router's, which an oauth2_code connector's redirect URI is built from as a
// consent builds it (connectorCallbackURL).
func connectorOf(definition store.ConnectorDefinition, publicURL string) Connector {
	manifest := definition.Manifest
	inputs := make([]ConnectorInput, 0, len(manifest.Inputs))
	for _, in := range manifest.Inputs {
		inputs = append(inputs, ConnectorInput{Name: in.Name, Enum: in.Enum, Pattern: in.Pattern, Default: in.Default})
	}
	registrations := make([]ConnectorClientRegistrationMethod, 0, len(manifest.Client.Registration))
	for _, registration := range manifest.Client.Registration {
		registrations = append(registrations, ConnectorClientRegistrationMethod(registration))
	}
	var setup *ConnectorSetup
	if manifest.Setup.URL != "" || len(manifest.Setup.Steps) > 0 {
		setup = &ConnectorSetup{URL: manifest.Setup.URL, Steps: make([]ConnectorSetupStep, 0, len(manifest.Setup.Steps))}
		for _, step := range manifest.Setup.Steps {
			setup.Steps = append(setup.Steps, ConnectorSetupStep{Title: step.Title, Description: step.Description})
		}
	}
	custom := definition.CustomerID != store.BuiltinCustomer
	var redirectURI string
	if slices.Contains(manifest.Schemes, oauth2code.Name) {
		redirectURI = connectorCallbackURL(publicURL)
	}
	return Connector{
		Setup:       setup,
		RedirectURI: redirectURI,
		ID:          definition.ID,
		Revision:    definition.Revision,
		Name:        definition.Name,
		Category:    definition.Category,
		Description: definition.Description,
		Custom:      custom,
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
