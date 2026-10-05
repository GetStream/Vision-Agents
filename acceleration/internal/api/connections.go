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
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const noConnections = "connections are not available: no database configured"

// connectorsOff is the answer to a create on a deployment with no scheme registered, which is
// what cmd/router builds with connectors.enabled off (newConnectorRegistry).
const connectorsOff = "connections cannot be created: connectors are not enabled on this deployment"

// noSuchConnection is the one answer for a connection the caller may not have: none was
// made, it is another app's, another user's, or it was deleted. One answer, so a guessed id
// learns nothing (architecture doc, PolicyContract: «a guessed connection id gives
// not-found»).
const noSuchConnection = "no such connection"

// ConnectorConnection is one account at one connector, as a caller is shown it. Credential
// material is never part of it.
type ConnectorConnection struct {
	ID                 string                    `json:"id" readOnly:"true"`
	ConnectorID        string                    `json:"connector_id"`
	DefinitionRevision int                       `json:"definition_revision" readOnly:"true" doc:"The connector's revision when the connection was made, which it keeps reading until it is reconnected."`
	Owner              ConnectorConnectionOwner  `json:"owner"`
	AuthScheme         string                    `json:"auth_scheme" doc:"How the connection authenticates, one of its connector's schemes."`
	Inputs             map[string]string         `json:"inputs" doc:"What the connection was created with, the connector's defaults filled in."`
	Metadata           map[string]string         `json:"metadata" readOnly:"true" doc:"What the provider said about the account when it was connected, such as a workspace id. Empty until then."`
	Label              string                    `json:"label,omitempty"`
	AccountID          string                    `json:"account_id,omitempty" readOnly:"true" doc:"The provider account, known once it is connected."`
	Status             ConnectorConnectionStatus `json:"status"`
	GrantedScopes      []string                  `json:"granted_scopes" readOnly:"true"`
	Revision           int                       `json:"revision" readOnly:"true" doc:"Advances with every new credential, starting at 1."`
	ExpiresAt          *time.Time                `json:"expires_at,omitempty" readOnly:"true" doc:"When the current credential expires. Absent when there is none or it does not."`
	CreatedAt          time.Time                 `json:"created_at" readOnly:"true"`
	UpdatedAt          time.Time                 `json:"updated_at" readOnly:"true"`
}

func (*ConnectorConnection) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One account at one connector, owned by the app or by one of its " +
		"users. Credentials are never shown."
	return schema
}

// ConnectorConnectionOwner is whose a connection is.
type ConnectorConnectionOwner struct {
	Type   ConnectorConnectionOwnerType `json:"type"`
	UserID string                       `json:"user_id,omitempty" doc:"The user, for a user-owned connection only. It must be the user the backend acts for, named by X-Stream-User-Id."`
}

func (*ConnectorConnectionOwner) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Whose a connection is: the app's, which any of its agents may be " +
		"bound to, or one user's."
	schema.AdditionalProperties = false
	return schema
}

// ConnectorConnectionOwnerType is the two owners a connection can have (architecture doc,
// one-way door 8).
type ConnectorConnectionOwnerType string

func (ConnectorConnectionOwnerType) Schema(registry huma.Registry) *huma.Schema {
	ref := namedEnum(registry, "ConnectorConnectionOwnerType",
		"app is the app's own account, user one user's.", store.OwnerApp, store.OwnerUser)
	// AgentLogSource has user too. When two enums share a constant name oapi-codegen prefixes
	// both with their type (enumsConflict, pkg/codegen/codegen.go:1553 in v2.8.0), which would
	// rename the Go SDK's User, Agent, System and Tool. Naming these constants keeps the
	// others as they were, as PhoneOperation does in api/legacy.yaml.
	registry.Map()["ConnectorConnectionOwnerType"].Extensions = map[string]any{
		"x-enum-varnames": []string{"ConnectorConnectionOwnerTypeApp", "ConnectorConnectionOwnerTypeUser"},
	}
	return ref
}

// ConnectorConnectionStatus is where a connection is in its life.
type ConnectorConnectionStatus string

func (ConnectorConnectionStatus) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectorConnectionStatus",
		"pending until an account is connected, then connected, needs_reauthorization once the "+
			"provider stops accepting its credential, and disconnected when it is deleted.",
		store.ConnectionPending, store.ConnectionConnected, store.ConnectionNeedsReauthorization,
		store.ConnectionDisconnected)
}

// ConnectorConnectionRequest is a connection to create.
//
// The label's 120 is the prototype's (CreateConnectorConnectionRequest in api/openapi.yaml
// on codex/connector-support at cf62af0d). Inputs have no count or length of their own: each
// must be one the connector declares and match its enum or pattern, which bounds them.
type ConnectorConnectionRequest struct {
	ConnectorID string                   `json:"connector_id" minLength:"1" doc:"A built-in, such as slack, or one of the app's own."`
	Owner       ConnectorConnectionOwner `json:"owner"`
	AuthScheme  string                   `json:"auth_scheme,omitempty" doc:"One of the connector's schemes. Omitted is its only one; a connector with several needs it named."`
	Inputs      map[string]string        `json:"inputs,omitempty" doc:"Values for the connector's inputs, such as a region. One without a default is required, and each must match the connector's enum or pattern."`
	Label       string                   `json:"label,omitempty" maxLength:"120" doc:"A name to tell connections apart by."`
}

func (*ConnectorConnectionRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A connection to create, pending until an account is connected. An " +
		"unknown field is refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

// ConnectorConnectionPage is a page of connections.
type ConnectorConnectionPage struct {
	Items      []ConnectorConnection `json:"items"`
	HasMore    bool                  `json:"has_more"`
	NextCursor *string               `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page, with the same owner_type and connector_id. Absent on the last one."`
}

type listConnectorConnectionsRequest struct {
	OwnerType   ConnectorConnectionOwnerType `query:"owner_type" required:"true" doc:"app lists the app's own; user lists those of the user the backend acts for, named by X-Stream-User-Id."`
	ConnectorID string                       `query:"connector_id" doc:"Keeps one connector's."`
	// 200 and 25 are store.ConnectionLimit's.
	Limit  int    `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 25."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
}

type listConnectorConnectionsResponse struct {
	Body ConnectorConnectionPage
}

type connectorConnectionRequest struct {
	ID string `path:"id" doc:"The connection, as returned when it was created."`
}

type deleteConnectorConnectionRequest struct {
	ID    string `path:"id" doc:"The connection, as returned when it was created."`
	Force bool   `query:"force" doc:"Delete it even while an agent config binds it as its fixed connection. The binding is left in place, naming a connection that no longer exists."`
}

type createConnectorConnectionRequest struct {
	Body ConnectorConnectionRequest
}

type connectorConnectionResponse struct {
	Body ConnectorConnection
}

// registerConnections declares the connection operations. All four are server-side only:
// a user-owned connection is made by the app's backend for the user it acts for, and the
// end user's device only ever sees the consent that backend sends it to (architecture doc,
// one-way door 7: owner identity comes only from the trusted principal).
func (s *Server) registerConnections(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID:   "createConnectorConnection",
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
	}, s.createConnectorConnection)
	huma.Register(api, huma.Operation{
		OperationID: "listConnectorConnections",
		Method:      http.MethodGet,
		Path:        "/v1/agents/connections",
		Summary:     "List connections",
		Description: "One owner's connections, newest first: the app's own, or those of the " +
			"user the backend acts for.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "A page of connections"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listConnectorConnections)
	huma.Register(api, huma.Operation{
		OperationID: "getConnectorConnection",
		Method:      http.MethodGet,
		Path:        "/v1/agents/connections/{id}",
		Summary:     "Read a connection",
		Description: "An app-owned connection, or a user-owned one of the user the backend " +
			"acts for. Another user's is not found, the same as one that does not exist.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The connection"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getConnectorConnection)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteConnectorConnection",
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
	}, s.deleteConnectorConnection)
}

// createConnectorConnection records a pending connection for the app or the user the
// backend acts for.
func (s *Server) createConnectorConnection(ctx context.Context, request *createConnectorConnectionRequest) (*connectorConnectionResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConnections)
	}
	// Before the body is read, so a deployment that cannot connect anything says so rather
	// than naming an input or a scheme the caller never chose.
	if len(s.connectors.Schemes) == 0 {
		return nil, huma.Error400BadRequest(connectorsOff)
	}
	sent := request.Body
	ownerID, err := ownerOf(ctx, sent.Owner)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	definition, err := s.store.LatestConnectorDefinition(ctx, customerID, sent.ConnectorID)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		return nil, huma.Error400BadRequest(fmt.Sprintf("no such connector: %q", sent.ConnectorID))
	}
	if err != nil {
		return nil, err
	}
	scheme := sent.AuthScheme
	if scheme == "" {
		if len(definition.Manifest.Schemes) != 1 {
			return nil, huma.Error400BadRequest(fmt.Sprintf("auth_scheme is required: %s allows %s",
				definition.ID, strings.Join(definition.Manifest.Schemes, ", ")))
		}
		scheme = definition.Manifest.Schemes[0]
	}
	// Resolve is what every later use of the connection reads it through, so an input it
	// refuses here is one the connection could never be used with. Its errors name the input.
	profile, err := definition.Manifest.Resolve(scheme, sent.Inputs, nil)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
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
	if errors.Is(err, store.ErrUnregisteredScheme) {
		known := slices.Sorted(maps.Keys(s.connectors.Schemes))
		return nil, huma.Error400BadRequest(fmt.Sprintf("auth_scheme %q is not one this deployment has (%s)",
			scheme, strings.Join(known, ", ")))
	}
	// Every other refusal a caller can cause is above, so what the store refuses here is a bug.
	if err != nil {
		return nil, err
	}
	return &connectorConnectionResponse{Body: connectorConnectionOf(connection)}, nil
}

// listConnectorConnections lists one owner's connections, a page at a time.
func (s *Server) listConnectorConnections(ctx context.Context, request *listConnectorConnectionsRequest) (*listConnectorConnectionsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConnections)
	}
	ownerID := ""
	if request.OwnerType == store.OwnerUser {
		ownerID = actingUser(ctx)
		if ownerID == "" {
			return nil, huma.Error400BadRequest("owner_type user lists the connections of the user this backend acts for: name them with " + auth.UserHeader)
		}
	}
	cursor, err := decodeCursor[store.ConnectionPosition](&request.Cursor)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
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
	listed := ConnectorConnectionPage{Items: make([]ConnectorConnection, 0, len(kept)), HasMore: more}
	for _, connection := range kept {
		listed.Items = append(listed.Items, connectorConnectionOf(connection))
	}
	if more {
		last := kept[len(kept)-1]
		listed.NextCursor = encodeCursor(store.ConnectionPosition{CreatedAt: last.CreatedAt, ID: last.ID})
	}
	return &listConnectorConnectionsResponse{Body: listed}, nil
}

// getConnectorConnection reads one connection the caller may have.
func (s *Server) getConnectorConnection(ctx context.Context, request *connectorConnectionRequest) (*connectorConnectionResponse, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	return &connectorConnectionResponse{Body: connectorConnectionOf(connection)}, nil
}

// deleteConnectorConnection soft deletes one connection the caller may have, unless an
// agent config still binds it and the caller did not force it.
func (s *Server) deleteConnectorConnection(ctx context.Context, request *deleteConnectorConnectionRequest) (*struct{}, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	// The check and the delete are two statements. Nothing writes bindings yet, so no bind
	// can land between them; once one does (AI-842), make the unforced delete one statement
	// that refuses a bound row, or a bind in between leaves the binding this 409 prevents.
	if !request.Force {
		referenced, err := s.store.ConnectorConnectionReferenced(ctx, connection.CustomerID, connection.ID)
		if errors.Is(err, store.ErrNoConnectorConnection) {
			return nil, huma.Error404NotFound(noSuchConnection)
		}
		if err != nil {
			return nil, err
		}
		// 409: the request conflicts with the state of the resource, which the caller can
		// change and retry (RFC 9110 section 15.5.10).
		if referenced {
			return nil, huma.Error409Conflict("an agent config binds this connection as its fixed connection: " +
				"unbind it first, or delete with force=true")
		}
	}
	err = s.store.DeleteConnectorConnection(ctx, connection.CustomerID, connection.ID)
	if errors.Is(err, store.ErrNoConnectorConnection) {
		return nil, huma.Error404NotFound(noSuchConnection)
	}
	if err != nil {
		return nil, err
	}
	return nil, nil
}

// reachableConnection is the live connection id names, if the caller may have it, and
// otherwise the one not-found every other missing connection gets.
func (s *Server) reachableConnection(ctx context.Context, id string) (store.ConnectorConnection, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return store.ConnectorConnection{}, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return store.ConnectorConnection{}, huma.Error400BadRequest(noConnections)
	}
	connection, err := s.store.ConnectorConnection(ctx, customerID, id)
	if errors.Is(err, store.ErrNoConnectorConnection) || err == nil && !mayReach(ctx, connection) {
		return store.ConnectorConnection{}, huma.Error404NotFound(noSuchConnection)
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
func ownerOf(ctx context.Context, owner ConnectorConnectionOwner) (string, error) {
	userID := strings.TrimSpace(owner.UserID)
	switch owner.Type {
	case store.OwnerApp:
		if userID != "" {
			return "", errors.New("an app-owned connection has no owner.user_id")
		}
		return "", nil
	case store.OwnerUser:
		if userID == "" {
			return "", errors.New("owner.user_id is required for a user-owned connection")
		}
		// The user is the one the backend acts for, never one the body names alone: the
		// body is the caller's to write, the acting user is the principal's.
		if acting := actingUser(ctx); acting == "" || acting != userID {
			return "", errors.New("owner.user_id must be the user this backend acts for, named by " + auth.UserHeader)
		}
		return userID, nil
	default:
		return "", fmt.Errorf("owner.type must be %s or %s", store.OwnerApp, store.OwnerUser)
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

// connectorConnectionOf is the part of a stored connection a caller is shown. Each field is
// copied by name, so a column added to the row stays hidden until it is added here. Sealed
// material, cached tools and last_error are left out: the first is never shown, and the
// other two are for the operations that write them (T18, T12).
func connectorConnectionOf(connection store.ConnectorConnection) ConnectorConnection {
	return ConnectorConnection{
		ID:                 connection.ID,
		ConnectorID:        connection.ConnectorID,
		DefinitionRevision: connection.DefinitionRevision,
		Owner: ConnectorConnectionOwner{
			Type:   ConnectorConnectionOwnerType(connection.OwnerType),
			UserID: connection.OwnerID,
		},
		AuthScheme:    connection.AuthScheme,
		Inputs:        maps.Clone(connection.Inputs),
		Metadata:      maps.Clone(connection.Metadata),
		Label:         connection.Label,
		AccountID:     connection.AccountID,
		Status:        ConnectorConnectionStatus(connection.Status),
		GrantedScopes: append([]string{}, connection.GrantedScopes...),
		Revision:      connection.Revision,
		ExpiresAt:     connection.ExpiresAt,
		CreatedAt:     connection.CreatedAt,
		UpdatedAt:     connection.UpdatedAt,
	}
}
