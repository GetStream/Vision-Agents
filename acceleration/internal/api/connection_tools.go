package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// errConnectionToolsOff is the answer to a validate on a deployment with no resolver or no
// transports, which is what cmd/router builds with connectors off.
var errConnectionToolsOff = notConfigured("connections cannot be validated: connectors are not enabled on this deployment")

// errStaleRevision is the answer to a credentials write whose expected_revision the
// connection has moved past. 409: the request conflicts with the state of the resource, which
// the caller can read again and retry (RFC 9110 section 15.5.10).
var errStaleRevision = conflict("the connection's credentials changed since expected_revision: read it again and retry")

// ConnectionCredentials is what a credentials write sends. Values never come back: no
// response carries them.
type ConnectionCredentials struct {
	ExpectedRevision int               `json:"expected_revision" minimum:"1" doc:"The connection's revision as last read. A connection that has moved past it is refused with a 409, so two writers never replace each other's credentials unseen."`
	Values           map[string]string `json:"values,omitempty" doc:"What the connection's auth_scheme takes, write-only. api_key: api_key and header. bearer: token. none: nothing, which activates the connection. oauth2_client_credentials: client_id and client_secret, which are tried at the token endpoint at once. oauth2_code: a grant the provider already issued, as access_token, refresh_token (optional), expires_at (RFC 3339) and scope (the granted scopes joined as the connector's scopes are); its endpoints and client are the connector's, never the caller's."`
}

func (*ConnectionCredentials) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Credentials for a connection, under the revision the caller last read. An " +
		"unknown field is refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

type putConnectionCredentialsRequest struct {
	ID   string `path:"id" doc:"The connection, as returned when it was created."`
	Body ConnectionCredentials
}

// ConnectionValidationStatus is what a validate found.
type ConnectionValidationStatus string

const (
	validationConnected            = "connected"
	validationPending              = "pending"
	validationNeedsReauthorization = "needs_reauthorization"
	validationFailed               = "failed"
)

func (ConnectionValidationStatus) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectionValidationStatus",
		"connected: the credential works and the tools were listed. pending: no credentials yet. "+
			"needs_reauthorization: the provider no longer takes the credential, so only a reconnect "+
			"helps. failed: the provider could not be reached or listed nothing usable; error says why.",
		validationConnected, validationPending, validationNeedsReauthorization, validationFailed)
}

// ConnectionValidation is what a validate found.
type ConnectionValidation struct {
	ConnectionID string                     `json:"connection_id"`
	Status       ConnectionValidationStatus `json:"status"`
	Error        string                     `json:"error,omitempty" doc:"Why the status is not connected, for a person to read."`
	ToolsDigest  string                     `json:"tools_digest,omitempty" doc:"The digest of the tools the connection offers, as GET .../tools shows them. Absent until a validate listed them."`
	CheckedAt    *time.Time                 `json:"checked_at,omitempty" doc:"When the tools were listed. Absent until a validate listed them."`
}

func (*ConnectionValidation) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Whether a connection's credential works, found by asking the provider for its tools."
	return schema
}

type validationResponse struct {
	Body ConnectionValidation
}

// ConnectionTool is one tool a connection offered when it was last validated.
type ConnectionTool struct {
	Name         string         `json:"name" doc:"The tool's name at the provider. An agent config grants it by this name."`
	Description  string         `json:"description"`
	InputSchema  map[string]any `json:"input_schema" doc:"The JSON Schema of its arguments."`
	SchemaDigest string         `json:"schema_digest" doc:"The SHA-256 of its name, description and input schema. A grant pins it, so a tool whose schema changes is not offered until it is granted again."`
}

// ConnectionTools is the tools a connection offered when it was last validated.
type ConnectionTools struct {
	ConnectionID string           `json:"connection_id"`
	Tools        []ConnectionTool `json:"tools"`
	Digest       string           `json:"digest,omitempty" doc:"The digest of the whole list. Absent until a validate listed it."`
	CheckedAt    *time.Time       `json:"checked_at,omitempty" doc:"When the list was read. Absent until a validate listed it."`
}

func (*ConnectionTools) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	// Not a page: it is one connection's snapshot, kept in its row and bounded by what one
	// tools/list response may hold.
	schema.Description = "The tools a connection offered when it was last validated, in one " +
		"piece: the provider's own list, not a page of one."
	return schema
}

type connectionToolsResponse struct {
	Body ConnectionTools
}

// registerConnectionTools declares the operations that give a connection its credentials,
// check them against the provider, and read back the tools it found. All three are
// server-side only, like every connection operation: credentials are secret, and a user's
// connection is the backend's to manage for the user it acts for.
func (s *Server) registerConnectionTools(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "putConnectionCredentials",
		Method:      http.MethodPut,
		Path:        "/v1/agents/connections/{id}/credentials",
		Summary:     "Set a connection's credentials",
		Description: "Stores the credentials a connection's scheme takes, sealed, and connects it: an " +
			"API key, a bearer token, an OAuth client for client credentials, an OAuth grant the " +
			"provider already issued, or nothing for a connector that needs none. expected_revision " +
			"must be the connection's revision as last read; a connection that moved past it is a " +
			"409. The values are never shown again. Who may set them is who may read the " +
			"connection.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The connection, connected"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict},
	}, s.putConnectionCredentials)
	huma.Register(api, huma.Operation{
		OperationID: "validateConnection",
		Method:      http.MethodPost,
		Path:        "/v1/agents/connections/{id}/validate",
		Summary:     "Validate a connection",
		Description: "Gets the connection's credential, renewing it when it must, and asks the " +
			"provider for its tools, which GET .../tools then shows. A connection that needs a " +
			"reconnect says so without the provider being asked. Who may validate it is who may " +
			"read it.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "What the validate found"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.validateConnection)
	huma.Register(api, huma.Operation{
		OperationID: "listConnectionTools",
		Method:      http.MethodGet,
		Path:        "/v1/agents/connections/{id}/tools",
		Summary:     "List a connection's tools",
		Description: "The tools the connection offered when it was last validated, each with the " +
			"schema digest an agent config's grant pins. Empty until a validate listed them. Who " +
			"may read them is who may read the connection.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The connection's tools"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listConnectionTools)
}

// putConnectionCredentials completes the connection's scheme with the values sent and stores
// what it gives under the lock, at the revision the caller read.
func (s *Server) putConnectionCredentials(ctx context.Context, request *putConnectionCredentialsRequest) (*connectionResponse, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	if s.credentials == nil {
		return nil, notConfigured("credentials cannot be stored: connectors are not enabled on this deployment")
	}
	sent := request.Body
	// Checked before the scheme runs, so a stale write asks no token endpoint. Checked again
	// under the lock, which is the check that holds.
	if sent.ExpectedRevision != connection.Revision {
		return nil, errStaleRevision
	}
	scheme, found := s.connectors.Schemes[connection.AuthScheme]
	if !found {
		return nil, invalidRequest(fmt.Sprintf("auth_scheme %q is not one this deployment has", connection.AuthScheme))
	}
	manifest, err := s.connectionManifest(ctx, connection, connection.DefinitionRevision)
	if err != nil {
		return nil, err
	}
	values := sent.Values
	if values == nil {
		values = map[string]string{}
	}
	ref := core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}
	credentials, account, err := scheme.Complete(ctx, core.CompleteInput{Ref: ref, Manifest: manifest, Supplied: values})
	if err != nil {
		// A scheme's error names what is wrong, never the value (core AGENTS.md, «Secrets never
		// print»; contracttest's TestAnErrorAboutAnUnusableValueDoesNotQuoteIt).
		return nil, invalidRequest("the credentials were refused: " + err.Error())
	}
	stale := false
	err = s.credentials.Update(ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		if state.Revision != sent.ExpectedRevision {
			stale = true
			return false, nil
		}
		state.Credentials = credentials
		state.Status = store.ConnectionConnected
		state.LastError = ""
		// A static credential learns no account; what an earlier consent found stays.
		if account.AccountID != "" {
			state.AccountID = account.AccountID
		}
		if len(account.Metadata) > 0 {
			state.Metadata = account.Metadata
		}
		state.Scopes = account.Scopes
		// The resolver sets it the first time it retrieves an access credential, as after a
		// consent (authorizations.go).
		state.ExpiresAt = time.Time{}
		return true, nil
	})
	if errors.Is(err, store.ErrNoConnectorConnection) {
		return nil, errNoSuchConnection
	}
	if err != nil {
		return nil, err
	}
	if stale {
		return nil, errStaleRevision
	}
	return s.getConnection(ctx, &connectionRequest{ID: connection.ID})
}

// validateConnection resolves the connection's credential, lists its tools through each of its
// sources, and stores the list.
func (s *Server) validateConnection(ctx context.Context, request *connectionRequest) (*validationResponse, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	if s.connectorResolver == nil || s.connectorTransports == nil {
		return nil, errConnectionToolsOff
	}
	ref := core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}
	// The resolver refuses a connection that is not connected before anything is sent, and
	// moves one whose renewal the provider refused; either way the row says what to do.
	if _, err := s.connectorResolver.Resolve(ctx, ref, core.CredentialRequest{}); err != nil {
		return s.validationAfter(ctx, connection, err)
	}
	scheme, found := s.connectors.Schemes[connection.AuthScheme]
	if !found {
		return nil, invalidRequest(fmt.Sprintf("auth_scheme %q is not one this deployment has", connection.AuthScheme))
	}
	manifest, err := s.connectionManifest(ctx, connection, connection.DefinitionRevision)
	if err != nil {
		return nil, err
	}
	binding := core.ResolvedBinding{
		Connection: coreConnection(connection),
		Manifest:   manifest,
		HTTP:       s.connectorTransports.Client(ref, scheme),
	}
	var specs []core.ToolSpec
	for _, kind := range sourceKinds(manifest) {
		source, found := s.connectors.ToolSources[kind]
		if !found {
			return failed(connection, fmt.Sprintf("this deployment has no %s tool source", kind)), nil
		}
		listed, err := source.Discover(ctx, binding)
		if err != nil {
			return s.validationAfter(ctx, connection, err)
		}
		specs = append(specs, listed...)
	}

	tools := make([]store.ConnectorTool, 0, len(specs))
	for _, spec := range specs {
		tools = append(tools, store.ConnectorTool{Name: spec.Name, Description: spec.Description,
			InputSchema: spec.InputSchema, SchemaDigest: spec.SchemaDigest})
	}
	digest, err := toolsDigest(tools)
	if err != nil {
		return nil, err
	}
	checked := time.Now().UTC().Truncate(time.Microsecond)
	err = s.store.SetConnectorConnectionTools(ctx, connection.CustomerID, connection.ID, tools, digest, checked)
	if errors.Is(err, store.ErrNoConnectorConnection) {
		return nil, errNoSuchConnection
	}
	if err != nil {
		return nil, err
	}
	return &validationResponse{Body: ConnectionValidation{ConnectionID: connection.ID,
		Status: validationConnected, ToolsDigest: digest, CheckedAt: &checked}}, nil
}

// validationAfter is what a validate reports after the resolver or a source failed: the
// connection's status when the failure moved it off connected (a refused credential, which
// the transport invalidated), and failed with why otherwise.
func (s *Server) validationAfter(ctx context.Context, connection store.ConnectorConnection, cause error) (*validationResponse, error) {
	now, err := s.store.ConnectorConnection(ctx, connection.CustomerID, connection.ID)
	if errors.Is(err, store.ErrNoConnectorConnection) {
		return nil, errNoSuchConnection
	}
	if err != nil {
		return nil, err
	}
	switch now.Status {
	case store.ConnectionPending:
		return &validationResponse{Body: ConnectionValidation{ConnectionID: now.ID, Status: validationPending,
			Error: "no credentials yet: start a consent or set its credentials"}}, nil
	case store.ConnectionNeedsReauthorization:
		return &validationResponse{Body: ConnectionValidation{ConnectionID: now.ID, Status: validationNeedsReauthorization,
			Error: now.LastError}}, nil
	}
	// The error says what failed and where, never a credential: the transport applies those
	// below everything that writes an error (core AGENTS.md, «Secrets never print»).
	return failed(now, cause.Error()), nil
}

// listConnectionTools reads back the tools a validate stored.
func (s *Server) listConnectionTools(ctx context.Context, request *connectionRequest) (*connectionToolsResponse, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	tools := make([]ConnectionTool, 0, len(connection.CachedTools))
	for _, tool := range connection.CachedTools {
		tools = append(tools, ConnectionTool{Name: tool.Name, Description: tool.Description,
			InputSchema: maps.Clone(tool.InputSchema), SchemaDigest: tool.SchemaDigest})
	}
	return &connectionToolsResponse{Body: ConnectionTools{ConnectionID: connection.ID, Tools: tools,
		Digest: connection.ToolsDigest, CheckedAt: connection.ToolsCheckedAt}}, nil
}

// failed is a validate that could not list the tools, saying why.
func failed(connection store.ConnectorConnection, why string) *validationResponse {
	return &validationResponse{Body: ConnectionValidation{ConnectionID: connection.ID, Status: validationFailed, Error: why}}
}

// sourceKinds are the kinds of tool source a manifest lists, each once, in its order.
func sourceKinds(m core.ResolvedManifest) []string {
	var kinds []string
	for _, rule := range m.Sources {
		if !slices.Contains(kinds, rule.Kind) {
			kinds = append(kinds, rule.Kind)
		}
	}
	return kinds
}

// toolsDigest is the SHA-256, in hex, of the tools' JSON: the prototype's connectorToolsDigest
// (internal/api/connectors.go:1246 on codex/connector-support at cf62af0d).
func toolsDigest(tools []store.ConnectorTool) (string, error) {
	data, err := json.Marshal(tools)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

// coreConnection is a stored connection as a tool source reads it.
func coreConnection(connection store.ConnectorConnection) core.Connection {
	return core.Connection{
		ID:                 connection.ID,
		ConnectorID:        connection.ConnectorID,
		DefinitionRevision: connection.DefinitionRevision,
		OwnerType:          connection.OwnerType,
		OwnerID:            connection.OwnerID,
		AccountID:          connection.AccountID,
		Inputs:             maps.Clone(connection.Inputs),
		Metadata:           maps.Clone(connection.Metadata),
		Status:             connection.Status,
	}
}
