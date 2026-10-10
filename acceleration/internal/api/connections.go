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

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

var errNoConnections = notConfigured("connections are not available: no database configured")

// errConnectorsOff is the answer to a create on a deployment with no scheme registered, which is
// what cmd/router builds with connectors.enabled off (newConnectorRegistry).
var errConnectorsOff = notConfigured("connections cannot be created: connectors are not enabled on this deployment")

// errNoSuchConnection is the one answer for a connection the caller may not have: none was
// made, it is another app's, another user's, or it was deleted. One answer, so a guessed id
// learns nothing (architecture doc, PolicyContract: «a guessed connection id gives
// not-found»).
var errNoSuchConnection = APIError{
	Type: ErrorTypeNotFound, Code: codeConnectionNotFound,
	Message: "no such connection",
}

// Connection is one account at one connector, as a caller is shown it. Its stored
// credentials are never part of it.
type Connection struct {
	ID                     string                     `json:"id" readOnly:"true"`
	ConnectorID            string                     `json:"connector_id"`
	DefinitionRevision     int                        `json:"definition_revision" readOnly:"true" doc:"The connector's revision the connection reads: the one its grant was made on. Every consent runs on the connector's latest revision, and one that connects the connection moves it there. Saving a bearer or api_key connection's token or key again (PUT .../credentials) moves it there too, when that revision still takes the connection's scheme and inputs. Until then it keeps this one."`
	DefinitionStatus       ConnectionDefinitionStatus `json:"definition_status" readOnly:"true"`
	DefinitionBrokenReason string                     `json:"definition_broken_reason,omitempty" readOnly:"true" doc:"Why the connector marked definition_revision broken. Present only when definition_status is broken."`
	Owner                  ConnectionOwner            `json:"owner"`
	AuthScheme             string                     `json:"auth_scheme" doc:"How the connection authenticates, one of its connector's schemes."`
	Inputs                 map[string]string          `json:"inputs" doc:"What the connection was created with, the connector's defaults filled in."`
	Metadata               map[string]string          `json:"metadata" readOnly:"true" doc:"What the provider said about the account when it was connected, such as a workspace id. Empty until then."`
	Label                  string                     `json:"label,omitempty"`
	AccountID              string                     `json:"account_id,omitempty" readOnly:"true" doc:"The provider account, known once it is connected."`
	Status                 ConnectionStatus           `json:"status"`
	GrantedScopes          []string                   `json:"granted_scopes" readOnly:"true"`
	Revision               int                        `json:"revision" readOnly:"true" doc:"Advances with every new credential, starting at 1."`
	ExpiresAt              *time.Time                 `json:"expires_at,omitempty" readOnly:"true" doc:"When the current credential expires. Absent when there is none or it does not."`
	CreatedAt              time.Time                  `json:"created_at" readOnly:"true"`
	UpdatedAt              time.Time                  `json:"updated_at" readOnly:"true"`
	UsedBy                 []ConnectionUse            `json:"used_by" readOnly:"true" doc:"The agent config bindings that name this connection as their fixed connection, which deleting it would break. A binding a session fills with the caller's own connection names none, so it is never listed."`
	Client                 *ConnectionClient          `json:"client,omitempty" readOnly:"true" doc:"The OAuth client the connection's grant was issued to. Absent for a scheme without one, before the first consent, and for a connection last consented before the router kept it."`
	LastValidation         *ConnectionLastValidation  `json:"last_validation,omitempty" readOnly:"true" doc:"What the last validate (POST .../validate) of the connection's current credentials found. Absent until the first one, and again once new credentials are stored (a token saved, a consent finished, a refresh)."`
}

// ConnectionLastValidation is what a connection's last validate found
// (store.ConnectorConnectionValidation, AI-1052).
type ConnectionLastValidation struct {
	Status    ConnectionValidationStatus `json:"status"`
	Code      string                     `json:"code,omitempty" doc:"The validate's code (connector_credential_rejected, connector_scope_required) when it had one. Otherwise, when the provider's last answer was an HTTP error, its status, such as 400 or 503. Absent when neither applies."`
	Error     string                     `json:"error,omitempty" doc:"Why the status is not connected, for a person to read: the validate's error with every value the credential is sent as cut out, and cut at 1 KiB. A provider's own error text in it can still hold anything else the provider wrote."`
	CheckedAt time.Time                  `json:"checked_at" doc:"When the validate ran."`
}

func (*ConnectionLastValidation) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What a connection's last validate found, kept so it is still shown after the " +
		"validate's answer is gone. A validate whose provider refused a bearer or api_key credential " +
		"with any 4xx but 429 also moves the connection to needs_reauthorization."
	return schema
}

// ConnectionClient is the OAuth client a connection's grant was issued to
// (store.ConnectorConnectionClient, AI-990 F16).
type ConnectionClient struct {
	Registration ConnectorClientRegistrationMethod `json:"registration"`
	ClientID     string                            `json:"client_id" doc:"The client identifier, which is not a secret (RFC 6749 section 2.2). For dcr, the one the provider issued when the router registered at the consent."`
}

func (*ConnectionClient) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Which OAuth client a connection's grant was issued to, so a client the router " +
		"registered on the fly (RFC 7591) can be found at the provider. Its secret is never shown."
	return schema
}

// ConnectionUse is one binding of an agent config that names a connection as its fixed
// connection.
type ConnectionUse struct {
	ConfigID   string `json:"config_id"`
	ConfigName string `json:"config_name"`
	Binding    string `json:"binding" doc:"The alias the config binds the connection under."`
}

func (*Connection) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One account at one connector, owned by the app or by one of its " +
		"users. Credentials are never shown."
	return schema
}

// ConnectionOwner is whose a connection is.
type ConnectionOwner struct {
	Type   ConnectionOwnerType `json:"type"`
	UserID string              `json:"user_id,omitempty" doc:"The user, for a user-owned connection only. It must be the user the backend acts for, named by X-Stream-User-Id."`
}

func (*ConnectionOwner) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Whose a connection is: the app's, which any of its agents may be " +
		"bound to, or one user's."
	schema.AdditionalProperties = false
	return schema
}

// ConnectionOwnerType is the two owners a connection can have (architecture doc,
// one-way door 8).
type ConnectionOwnerType string

func (ConnectionOwnerType) Schema(registry huma.Registry) *huma.Schema {
	ref := namedEnum(registry, "ConnectionOwnerType",
		"app is the app's own account, user one user's.", store.OwnerApp, store.OwnerUser)
	// AgentLogSource has user too. When two enums share a constant name oapi-codegen prefixes
	// both with their type (enumsConflict, pkg/codegen/codegen.go:1553 in v2.8.0), which would
	// rename the Go SDK's User, Agent, System and Tool. Naming these constants keeps the
	// others as they were, as PhoneOperation does in api/legacy.yaml.
	registry.Map()["ConnectionOwnerType"].Extensions = map[string]any{
		"x-enum-varnames": []string{"ConnectionOwnerTypeApp", "ConnectionOwnerTypeUser"},
	}
	return ref
}

// ConnectionDefinitionStatus is how the revision a connection reads compares with its
// connector's latest (store.DefinitionStatus).
type ConnectionDefinitionStatus string

func (ConnectionDefinitionStatus) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectionDefinitionStatus",
		"current when the connection reads its connector's latest revision, outdated when a later "+
			"one exists, and broken when a later one marked it as not working: the connection is "+
			"given no credential until it moves to the latest revision, by a consent that connects "+
			"it again or, for a bearer or api_key connection, by saving its token or key again.",
		store.DefinitionCurrent, store.DefinitionOutdated, store.DefinitionBroken)
}

// ConnectionStatus is where a connection is in its life.
type ConnectionStatus string

func (ConnectionStatus) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectionStatus",
		"pending until an account is connected, then connected, needs_reauthorization once the "+
			"provider stops accepting its credential, and disconnected when it is deleted.",
		store.ConnectionPending, store.ConnectionConnected, store.ConnectionNeedsReauthorization,
		store.ConnectionDisconnected)
}

// ConnectionRequest is a connection to create.
//
// The label's 120 is the prototype's (CreateConnectorConnectionRequest in api/openapi.yaml
// on codex/connector-support at cf62af0d). Inputs have no count or length of their own: each
// must be one the connector declares and match its enum or pattern, which bounds them.
type ConnectionRequest struct {
	ConnectorID string            `json:"connector_id" minLength:"1" doc:"A built-in, such as slack, or one of the app's own."`
	Owner       ConnectionOwner   `json:"owner"`
	AuthScheme  string            `json:"auth_scheme,omitempty" doc:"One of the connector's schemes. Omitted is its only one, or else its only one that is not a static token or key (bearer, api_key), such as oauth2_code for github; a connector with several others needs it named."`
	Inputs      map[string]string `json:"inputs,omitempty" doc:"Values for the connector's inputs, such as a region. One without a default is required, and each must match the connector's enum or pattern."`
	Label       string            `json:"label,omitempty" maxLength:"120" doc:"A name to tell connections apart by."`
}

func (*ConnectionRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A connection to create, pending until an account is connected. An " +
		"unknown field is refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

// ConnectionPage is a page of connections.
type ConnectionPage struct {
	Items      []Connection `json:"items"`
	HasMore    bool         `json:"has_more"`
	NextCursor *string      `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page, with the same owner_type and connector_id. Absent on the last one."`
}

type listConnectionsRequest struct {
	OwnerType   ConnectionOwnerType `query:"owner_type" required:"true" doc:"app lists the app's own; user lists those of the user the backend acts for, named by X-Stream-User-Id."`
	ConnectorID string              `query:"connector_id" doc:"Keeps one connector's."`
	// 200 and 25 are store.ConnectionLimit's.
	Limit  int    `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 25."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
}

type listConnectionsResponse struct {
	Body ConnectionPage
}

type connectionRequest struct {
	ID string `path:"id" doc:"The connection, as returned when it was created."`
}

type deleteConnectionRequest struct {
	ID    string `path:"id" doc:"The connection, as returned when it was created."`
	Force bool   `query:"force" doc:"Delete it even while an agent config binds it as its fixed connection. The binding is left in place, naming a connection that no longer exists."`
}

type createConnectionRequest struct {
	Body ConnectionRequest
}

type connectionResponse struct {
	Body Connection
}

// registerConnections declares the connection operations. All five are server-side only:
// a user-owned connection is made by the app's backend for the user it acts for, and the
// end user's device only ever sees the consent that backend sends it to (architecture doc,
// one-way door 7: owner identity comes only from the trusted principal).
func (s *Server) registerConnections(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID:   "createConnection",
		Method:        http.MethodPost,
		Path:          "/v1/agents/connections",
		Summary:       "Create a connection",
		DefaultStatus: http.StatusCreated,
		Description: "A pending connection to one account at a connector, made from the " +
			"connector's newest revision. An app-owned connection is the app's, for any of its " +
			"agents. A user-owned one is the user's the backend acts for: owner.user_id must be " +
			"the user X-Stream-User-Id names. Credentials are added afterwards. A deployment " +
			"with connectors off refuses every create.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"201": {Description: "The pending connection"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.createConnection)
	huma.Register(api, huma.Operation{
		OperationID: "listConnections",
		Method:      http.MethodGet,
		Path:        "/v1/agents/connections",
		Summary:     "List connections",
		Description: "One owner's connections, newest first: the app's own, or those of the " +
			"user the backend acts for.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "A page of connections"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listConnections)
	huma.Register(api, huma.Operation{
		OperationID: "getConnection",
		Method:      http.MethodGet,
		Path:        "/v1/agents/connections/{id}",
		Summary:     "Read a connection",
		Description: "An app-owned connection, or a user-owned one of the user the backend " +
			"acts for. Another user's is not found, the same as one that does not exist.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The connection"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getConnection)
	huma.Register(api, huma.Operation{
		OperationID: "exportConnectionToken",
		Method:      http.MethodPost,
		Path:        "/v1/agents/connections/{id}/token",
		Summary:     "Export a connection's access token",
		Description: "The connection's current access credential, for the app's backend to call the " +
			"provider with directly: an OAuth access token, renewed first when it is about to " +
			"expire, or an API key. A refresh token is never exported. Only the customer's own " +
			"provider app exports: an oauth2_code connection exports when its grant was issued " +
			"to the client the app registered itself, and that client is still the connector's; " +
			"a grant issued to Stream's app, or to one the router created, is refused with a 403. " +
			"An api_key connection always exports, since the key is the app's own. Other schemes " +
			"are refused. Each export is recorded in the connector audit as token_export. Who " +
			"may export it is who may read it.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The access credential"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict, http.StatusServiceUnavailable},
	}, s.exportConnectionToken)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteConnection",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/connections/{id}",
		Summary:       "Delete a connection",
		DefaultStatus: http.StatusNoContent,
		Description: "Disconnects the account and drops its credentials at once, so nothing " +
			"can use it from here on. The provider is not asked to revoke what it issued. A " +
			"connection an agent config binds as its fixed connection is refused with a 409 " +
			"unless force is set. Who may delete it is who may read it.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"204": {Description: "The connection is deleted"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict},
	}, s.deleteConnection)
}

// createConnection records a pending connection for the app or the user the
// backend acts for.
func (s *Server) createConnection(ctx context.Context, request *createConnectionRequest) (*connectionResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnections
	}
	// Before the body is read, so a deployment that cannot connect anything says so rather
	// than naming an input or a scheme the caller never chose.
	if len(s.connectors.Schemes) == 0 {
		return nil, errConnectorsOff
	}
	sent := request.Body
	ownerID, err := ownerOf(ctx, sent.Owner)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	definition, err := s.store.LatestConnectorDefinition(ctx, customerID, sent.ConnectorID)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		return nil, invalidRequest(fmt.Sprintf("no such connector: %q", sent.ConnectorID))
	}
	if err != nil {
		return nil, err
	}
	scheme := sent.AuthScheme
	if scheme == "" {
		chosen, found := defaultScheme(definition.Manifest, s.connectors.Schemes)
		if !found {
			return nil, invalidRequest(fmt.Sprintf("auth_scheme is required: %s allows %s",
				definition.ID, strings.Join(definition.Manifest.Schemes, ", ")))
		}
		scheme = chosen
	}
	// Resolve is what every later use of the connection reads it through, so an input it
	// refuses here is one the connection could never be used with. Its errors name the input.
	profile, err := definition.Manifest.Resolve(scheme, sent.Inputs, nil)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	connection := store.ConnectorConnection{
		CustomerID:         customerID,
		ConnectorID:        definition.ID,
		DefinitionRevision: definition.Revision,
		OwnerType:          string(sent.Owner.Type),
		OwnerID:            ownerID,
		AuthScheme:         scheme,
		Inputs:             profile.Inputs,
		Label:              strings.TrimSpace(sent.Label),
	}
	err = s.store.CreateConnectorConnection(ctx, s.connectors, &connection)
	// Deleted since it was read above (DeleteConnectorDefinition), so answered as one never made.
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		return nil, invalidRequest(fmt.Sprintf("no such connector: %q", sent.ConnectorID))
	}
	if errors.Is(err, store.ErrUnregisteredScheme) {
		known := slices.Sorted(maps.Keys(s.connectors.Schemes))
		return nil, invalidRequest(fmt.Sprintf("auth_scheme %q is not one this deployment has (%s)",
			scheme, strings.Join(known, ", ")))
	}
	// Every other refusal a caller can cause is above, so what the store refuses here is a bug.
	if err != nil {
		return nil, err
	}
	definitions, err := s.store.ConnectorDefinitionStatuses(ctx, customerID, []store.ConnectorConnection{connection})
	if err != nil {
		return nil, err
	}
	// A new connection is bound by nothing yet.
	return &connectionResponse{Body: connectionOf(connection, nil, definitions[connection.ID], nil, nil)}, nil
}

// defaultScheme is the scheme a connection to m gets when nobody names one: m's only scheme,
// or else its only one that is not core.Static. A connector that takes a consent and a static
// token beside it (github: oauth2_code and bearer, AI-990) so connects by consent, as it did
// before it took the token. found is false when that leaves none or several.
func defaultScheme(m core.Manifest, schemes map[string]core.Scheme) (string, bool) {
	if len(m.Schemes) == 1 {
		return m.Schemes[0], true
	}
	var others []string
	for _, name := range m.Schemes {
		if !core.IsStatic(schemes, name) {
			others = append(others, name)
		}
	}
	if len(others) != 1 {
		return "", false
	}
	return others[0], true
}

// listConnections lists one owner's connections, a page at a time.
func (s *Server) listConnections(ctx context.Context, request *listConnectionsRequest) (*listConnectionsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnections
	}
	ownerID := ""
	if request.OwnerType == store.OwnerUser {
		ownerID = actingUser(ctx)
		if ownerID == "" {
			return nil, invalidRequest("owner_type user lists the connections of the user this backend acts for: name them with " + auth.UserHeader)
		}
	}
	cursor, err := decodeCursor[store.ConnectionPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	found, err := s.store.ConnectorConnectionsByOwner(ctx, customerID, store.ConnectionFilter{
		OwnerType:   string(request.OwnerType),
		OwnerID:     ownerID,
		ConnectorID: request.ConnectorID,
		Limit:       request.Limit,
		After:       cursor,
	})
	if err != nil {
		return nil, err
	}

	kept, more := page(found, store.ConnectionLimit(request.Limit))
	ids := make([]string, 0, len(kept))
	for _, connection := range kept {
		ids = append(ids, connection.ID)
	}
	// One read for the whole page, and two for the revisions.
	uses, err := s.store.ConnectorConnectionUses(ctx, customerID, ids)
	if err != nil {
		return nil, err
	}
	definitions, err := s.store.ConnectorDefinitionStatuses(ctx, customerID, kept)
	if err != nil {
		return nil, err
	}
	clients, err := s.store.ConnectorConnectionClients(ctx, ids)
	if err != nil {
		return nil, err
	}
	validations, err := s.store.ConnectorConnectionValidations(ctx, ids)
	if err != nil {
		return nil, err
	}
	listed := ConnectionPage{Items: make([]Connection, 0, len(kept)), HasMore: more}
	for _, connection := range kept {
		listed.Items = append(listed.Items, connectionOf(connection, uses[connection.ID], definitions[connection.ID],
			clientOf(clients, connection.ID), lastValidationOf(validations, connection)))
	}
	if more {
		last := kept[len(kept)-1]
		listed.NextCursor = encodeCursor(store.ConnectionPosition{CreatedAt: last.CreatedAt, ID: last.ID})
	}
	return &listConnectionsResponse{Body: listed}, nil
}

// getConnection reads one connection the caller may have.
func (s *Server) getConnection(ctx context.Context, request *connectionRequest) (*connectionResponse, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	uses, err := s.store.ConnectorConnectionUses(ctx, connection.CustomerID, []string{connection.ID})
	if err != nil {
		return nil, err
	}
	definitions, err := s.store.ConnectorDefinitionStatuses(ctx, connection.CustomerID, []store.ConnectorConnection{connection})
	if err != nil {
		return nil, err
	}
	clients, err := s.store.ConnectorConnectionClients(ctx, []string{connection.ID})
	if err != nil {
		return nil, err
	}
	validations, err := s.store.ConnectorConnectionValidations(ctx, []string{connection.ID})
	if err != nil {
		return nil, err
	}
	return &connectionResponse{Body: connectionOf(connection, uses[connection.ID], definitions[connection.ID],
		clientOf(clients, connection.ID), lastValidationOf(validations, connection))}, nil
}

// deleteConnection soft deletes one connection the caller may have, unless an
// agent config still binds it and the caller did not force it.
func (s *Server) deleteConnection(ctx context.Context, request *deleteConnectionRequest) (*struct{}, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	// Unforced, the store locks the connection and then checks for a binding in the statement
	// that deletes. A config save locks the connections it binds while it writes, so of a
	// bind and a delete one always waits for the other and sees it.
	if request.Force {
		err = s.store.DeleteConnectorConnection(ctx, connection.CustomerID, connection.ID)
	} else {
		err = s.store.DeleteUnboundConnectorConnection(ctx, connection.CustomerID, connection.ID)
	}
	if errors.Is(err, store.ErrNoConnectorConnection) {
		return nil, errNoSuchConnection
	}
	// 409: the request conflicts with the state of the resource, which the caller can
	// change and retry (RFC 9110 section 15.5.10).
	if errors.Is(err, store.ErrConnectorConnectionBound) {
		return nil, conflict("an agent config binds this connection as its fixed connection: " +
			"unbind it first, or delete with force=true")
	}
	if err != nil {
		return nil, err
	}
	s.connectionDeleted(ctx, connection.CustomerID, store.DeletedConnection{
		ID: connection.ID, ConnectorID: connection.ConnectorID, OwnerType: connection.OwnerType,
		HadGrant: len(connection.CredentialsSealed) > 0,
	})
	return nil, nil
}

// connectionDeleted lets go of what a connection the store just soft deleted still has
// outside its row, as deleteConnection and deleteConnector both leave it.
func (s *Server) connectionDeleted(ctx context.Context, customerID string, connection store.DeletedConnection) {
	// Its outbound client goes too. A session holding a copy is refused by the resolver, and
	// by its dispatcher's check before every call.
	if s.connectorTransports != nil {
		s.connectorTransports.Close(core.ConnectionRef{CustomerID: customerID, ConnectionID: connection.ID})
	}
	// So do its MCP event subscriptions: a delivery to one is answered 410 from here on. The
	// connection is deleted whatever happens here, so a failure is logged, not answered with a
	// 500 a retry would turn into a 404: a row left behind goes at its next delivery or its
	// next look, which comes a day later at most (mcpevents.refreshAt, retryWait, waitUntil).
	if s.mcpEvents != nil {
		if err := s.mcpEvents.Stop(ctx, customerID, connection.ID); err != nil {
			s.logger.Error("could not drop a deleted connection's MCP event subscriptions", "connection", connection.ID, "error", err)
		}
	}
	// The delete dropped its credentials. One that held none, pending since it was made, had
	// no grant to revoke.
	if connection.HadGrant {
		s.auditGrant(ctx, customerID, connection.ID, connection.ConnectorID, connection.OwnerType,
			store.AuditGrantRevoked, store.AuditReasonDeleted, 0, "", core.CredentialChange{})
	}
}

// auditGrant records one grant the API created or revoked (T47), with the request's id
// (core.CorrelationOf) and the tokens change names (AI-990), and logs it as one line, as the
// resolver logs a refresh. The change is committed when it is called, so a row that cannot be
// written is logged and the change stands. revision is the connection's once it committed, 0
// when the change names none. change is zero when no token is known, as on a delete, whose
// credentials went sealed with the row.
func (s *Server) auditGrant(ctx context.Context, customerID, connectionID, connectorID, ownerType, action, reason string, revision int, attemptID string, change core.CredentialChange) {
	s.logger.Info("connector credential event", append([]any{"event", action,
		"connection", connectionID, "connector", connectorID, "revision", revision, "reason", reason},
		change.LogAttrs()...)...)
	event := &store.ConnectorAuditEvent{
		CustomerID: customerID, ConnectionID: connectionID, ConnectorID: connectorID, OwnerType: ownerType,
		Action: action, Reason: reason, Revision: revision, RequestID: core.CorrelationOf(ctx).RequestID,
		AttemptID: attemptID,
	}
	if change != (core.CredentialChange{}) {
		event.Credential = store.AuditCredential(change)
	}
	err := s.store.RecordConnectorAudit(ctx, event)
	if err != nil {
		s.logger.Error("could not record a connector audit row", "connection", connectionID, "action", action, "error", err)
	}
}

// recordClient keeps which OAuth client a consent's credentials were issued to, for a scheme
// that names it (core.ClientNamer), so the connection's reads show it (AI-990 F16). The consent
// is committed when it is called, so a row that cannot be written is logged and the consent
// stands.
func (s *Server) recordClient(ctx context.Context, connectionID string, credentials core.StoredCredentials) {
	client, ok := core.ClientOf(s.connectors.Schemes, credentials)
	if !ok {
		return
	}
	err := s.store.PutConnectorConnectionClient(ctx, &store.ConnectorConnectionClient{
		ConnectionID: connectionID, Registration: client.Registration, ClientID: client.ID,
	})
	if err != nil {
		s.logger.Error("could not record a connection's OAuth client", "connection", connectionID, "error", err)
	}
}

// reachableConnection is the live connection id names, if the caller may have it, and
// otherwise the one not-found every other missing connection gets.
func (s *Server) reachableConnection(ctx context.Context, id string) (store.ConnectorConnection, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return store.ConnectorConnection{}, errMissingCustomer
	}
	if s.store == nil {
		return store.ConnectorConnection{}, errNoConnections
	}
	connection, err := s.store.ConnectorConnection(ctx, customerID, id)
	if errors.Is(err, store.ErrNoConnectorConnection) || err == nil && !mayReach(ctx, connection) {
		return store.ConnectorConnection{}, errNoSuchConnection
	}
	if err != nil {
		return store.ConnectorConnection{}, err
	}
	return connection, nil
}

// mayReach is the prototype's owner rule (connectionOwnerMatches in internal/api/connectors.go
// on codex/connector-support at cf62af0d): the app's backend reaches the app's connections,
// and a user's only while it acts for that user.
func mayReach(ctx context.Context, connection store.ConnectorConnection) bool {
	if connection.OwnerType == store.OwnerApp {
		return ServerSideFrom(ctx)
	}
	user := actingUser(ctx)
	return connection.OwnerType == store.OwnerUser && user != "" && user == connection.OwnerID
}

// ownerOf is the owner id a new connection is filed under, or why the owner cannot be it.
func ownerOf(ctx context.Context, owner ConnectionOwner) (string, error) {
	userID := strings.TrimSpace(owner.UserID)
	switch owner.Type {
	case store.OwnerApp:
		if userID != "" {
			return "", stack.Wrap(errors.New("an app-owned connection has no owner.user_id"))
		}
		return "", nil
	case store.OwnerUser:
		if userID == "" {
			return "", stack.Wrap(errors.New("owner.user_id is required for a user-owned connection"))
		}
		// The user is the one the backend acts for, never one the body names alone: the
		// body is the caller's to write, the acting user is the principal's.
		if acting := actingUser(ctx); acting == "" || acting != userID {
			return "", stack.Wrap(errors.New("owner.user_id must be the user this backend acts for, named by " + auth.UserHeader))
		}
		return userID, nil
	default:
		return "", stack.Wrap(fmt.Errorf("owner.type must be %s or %s", store.OwnerApp, store.OwnerUser))
	}
}

// actingUser is the user a server-side caller acts for, and empty for anyone else. A device
// never reaches these operations, which are server-side only; the check is kept so the
// owner rule does not lean on that alone.
func actingUser(ctx context.Context) string {
	if !ServerSideFrom(ctx) {
		return ""
	}
	return CallerFrom(ctx).UserID
}

// connectionOf is the part of a stored connection a caller is shown. Each field is
// copied by name, so a column added to the row stays hidden until it is added here. Sealed
// credentials, cached tools and last_error are left out: the first is never shown, and the
// other two are for the operations that write them (T18, T12). uses are the bindings that
// name it (store.ConnectorConnectionUses), definition how its revision compares with its
// connector's (store.ConnectorDefinitionStatuses), and client the OAuth client its grant was
// issued to, nil when none is recorded (store.ConnectorConnectionClients), and validation what
// its last validate found, nil before the first (store.ConnectorConnectionValidations).
func connectionOf(connection store.ConnectorConnection, uses []store.ConnectionUse, definition store.DefinitionStatus,
	client *ConnectionClient, validation *ConnectionLastValidation) Connection {
	usedBy := make([]ConnectionUse, 0, len(uses))
	for _, use := range uses {
		usedBy = append(usedBy, ConnectionUse{ConfigID: use.ConfigID, ConfigName: use.ConfigName, Binding: use.Binding})
	}
	return Connection{
		ID:                 connection.ID,
		ConnectorID:        connection.ConnectorID,
		DefinitionRevision: connection.DefinitionRevision,
		DefinitionStatus:   ConnectionDefinitionStatus(definition.Status),
		// Empty unless broken.
		DefinitionBrokenReason: definition.Reason,
		Owner: ConnectionOwner{
			Type:   ConnectionOwnerType(connection.OwnerType),
			UserID: connection.OwnerID,
		},
		AuthScheme:     connection.AuthScheme,
		Inputs:         maps.Clone(connection.Inputs),
		Metadata:       maps.Clone(connection.Metadata),
		Label:          connection.Label,
		AccountID:      connection.AccountID,
		Status:         ConnectionStatus(connection.Status),
		GrantedScopes:  append([]string{}, connection.GrantedScopes...),
		Revision:       connection.Revision,
		ExpiresAt:      connection.ExpiresAt,
		CreatedAt:      connection.CreatedAt,
		UpdatedAt:      connection.UpdatedAt,
		UsedBy:         usedBy,
		Client:         client,
		LastValidation: validation,
	}
}

// lastValidationOf is the recorded last validate of connection, nil when there is none or when
// it no longer describes the connection's grant: new credentials since (a token saved, a
// consent, a refresh) moved the connection's revision past the validate's, or a grant began
// after it ran. The second is a token saved again as it was, which leaves the revision
// (pgsealed's commit seals only changed credentials anew) but connects a connection that was
// not connected from then on (ConnectedAt), and a consent, which always begins a grant.
func lastValidationOf(validations map[string]store.ConnectorConnectionValidation, connection store.ConnectorConnection) *ConnectionLastValidation {
	validation, ok := validations[connection.ID]
	if !ok || validation.Revision < connection.Revision ||
		connection.ConnectedAt != nil && validation.CheckedAt.Before(*connection.ConnectedAt) {
		return nil
	}
	return &ConnectionLastValidation{Status: ConnectionValidationStatus(validation.Status), Code: validation.Code,
		Error: validation.Error, CheckedAt: validation.CheckedAt}
}

// clientOf is the recorded client of the connection id names, nil when there is none.
func clientOf(clients map[string]store.ConnectorConnectionClient, id string) *ConnectionClient {
	client, ok := clients[id]
	if !ok {
		return nil
	}
	return &ConnectionClient{Registration: ConnectorClientRegistrationMethod(client.Registration), ClientID: client.ClientID}
}
