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
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

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
	validationNeedsScopes          = "needs_scopes"
	validationFailed               = "failed"
)

// codeScopeRequired is the code of a validate whose grant lacks a scope a tool needs: the
// connector_scope_required the architecture doc names («Add» item 5, from
// connector-design.md:389 on codex/connector-support).
const codeScopeRequired = "connector_scope_required"

// codeCredentialRejected is the code of a validate whose connection holds a token or key the
// provider no longer takes (a core.Static scheme: bearer, api_key). New (AI-990): a reconnect
// cannot help such a connection, only new credentials, so a program needs to tell it apart
// from an OAuth grant that needs a reconnect. It is also the code of one that reads a revision
// marked broken (AI-1002): saving its credentials again moves it, the same remedy, so one code.
const codeCredentialRejected = "connector_credential_rejected"

func (ConnectionValidationStatus) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectionValidationStatus",
		"connected: the credential works and the tools were listed. pending: no credentials yet. "+
			"needs_reauthorization: the provider no longer takes the credential, so only a reconnect "+
			"helps, or, with code connector_credential_rejected, saving credentials again. needs_scopes: the tools were listed, and the grant lacks scopes they need; "+
			"missing_scopes names them, and a consent that asks for them helps. failed: the provider "+
			"could not be reached or listed nothing usable; error says why.",
		validationConnected, validationPending, validationNeedsReauthorization, validationNeedsScopes, validationFailed)
}

// ConnectionValidationRequest is what a validate may be asked to check beyond the credential.
type ConnectionValidationRequest struct {
	Tools []string `json:"tools,omitempty" doc:"The tools to check the granted scopes against, by name: those an agent config will grant. Left out, every tool the connection offers."`
}

func (*ConnectionValidationRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What a validate checks the grant's scopes against. An unknown field is " +
		"refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

type validateConnectionRequest struct {
	ID   string `path:"id" doc:"The connection, as returned when it was created."`
	Body *ConnectionValidationRequest
}

// ConnectionValidation is what a validate found.
type ConnectionValidation struct {
	ConnectionID  string                     `json:"connection_id"`
	Status        ConnectionValidationStatus `json:"status"`
	Code          string                     `json:"code,omitempty" doc:"What a program branches on when the status is not connected: connector_scope_required with needs_scopes; connector_credential_rejected with needs_reauthorization, for a bearer or api_key connection whose token or key the provider rejected, or that reads a connector revision marked broken, which only saving credentials (PUT .../credentials) fixes. More may be added."`
	MissingScopes []string                   `json:"missing_scopes,omitempty" doc:"With needs_scopes: the scopes the checked tools need that the grant lacks, sorted."`
	Error         string                     `json:"error,omitempty" doc:"Why the status is not connected, for a person to read."`
	ToolsDigest   string                     `json:"tools_digest,omitempty" doc:"The digest of the tools the connection offers, as GET .../tools shows them. Absent until a validate listed them."`
	CheckedAt     *time.Time                 `json:"checked_at,omitempty" doc:"When the tools were listed. Absent until a validate listed them."`
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
	NeedsScopes  []string       `json:"needs_scopes,omitempty" doc:"The scopes a call of the tool needs, as the connector says. Absent when it says none."`
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
			"A bearer or api_key connection given its token or key again moves to its connector's " +
			"latest revision, as a consent moves an OAuth one, when that revision takes the " +
			"connection's scheme and inputs. Otherwise it keeps its own revision, and a 400 says " +
			"why when that one is marked broken.\n\n" +
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
			"reconnect says so without the provider being asked. The granted scopes are then " +
			"checked against what the tools need (all of them, or those the body names): a " +
			"grant that lacks some is needs_scopes with code connector_scope_required and the " +
			"missing scopes. Who may validate it is who may read it.\n\n" +
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
	// A token or key saved again is the static scheme's reconnect: no consent ever runs for one,
	// so this write is what moves it to the connector's latest revision, off one marked broken
	// since (AI-1002), as a consent moves an OAuth connection (begin, completeConsent). An
	// imported OAuth grant stays on the revision it was made on.
	revision := connection.DefinitionRevision
	static := core.IsStatic(s.connectors.Schemes, connection.AuthScheme)
	var manifest core.ResolvedManifest
	moved := false
	if static {
		latest, err := s.store.LatestConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID)
		if err != nil {
			return nil, err
		}
		// The latest revision may no longer take the connection's scheme or inputs, which no
		// write changes after create. A connection on a revision that still works keeps it, as
		// before AI-1002; one on a broken revision has none it can read.
		latestManifest, unfit := definitionManifest(latest, connection)
		if unfit == nil {
			revision, manifest, moved = latest.Revision, latestManifest, true
		} else {
			reason, broken, err := s.store.BrokenConnectorRevision(ctx, connection.ConnectorID, connection.DefinitionRevision)
			if err != nil {
				return nil, err
			}
			if broken {
				return nil, invalidRequest(fmt.Sprintf("revision %d of %s is marked broken (%s), and its latest "+
					"revision %d does not take this connection (%s); create a new connection",
					connection.DefinitionRevision, connection.ConnectorID, reason, latest.Revision, unfit))
			}
		}
	}
	if !moved {
		var err error
		manifest, err = s.connectionManifest(ctx, connection, revision)
		if err != nil {
			return nil, err
		}
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
	var committed *core.CredentialState
	// The tokens a write replaces, and those it stores, for a scheme that names them.
	change := core.CredentialChange{Current: core.FingerprintsOf(s.connectors.Schemes, credentials)}
	err = s.credentials.Update(ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		if state.Revision != sent.ExpectedRevision {
			stale = true
			return false, nil
		}
		// The credential store leaves the revision it committed here (core.CredentialStore).
		committed = state
		change.Previous = core.FingerprintsOf(s.connectors.Schemes, state.Credentials)
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
		// A move to another revision begins a new grant, as a reconnect's does: the tools pinned
		// for the old one (store.ConnectorToolPin) are pinned again on the next session.
		if static && state.DefinitionRevision != revision {
			state.DefinitionRevision = revision
			state.ConnectedAt = time.Now().UTC()
		}
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
	s.auditGrant(ctx, connection.CustomerID, connection.ID, connection.ConnectorID, connection.OwnerType,
		store.AuditGrantCreated, store.AuditReasonCredentials, committed.Revision, "", change)
	return s.getConnection(ctx, &connectionRequest{ID: connection.ID})
}

// validateConnection checks the connection (checkConnection) and records what it found as the
// connection's last validation (AI-1052), which GET and list show.
func (s *Server) validateConnection(ctx context.Context, request *validateConnectionRequest) (*validationResponse, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	if s.connectorResolver == nil || s.connectorTransports == nil {
		return nil, errConnectionToolsOff
	}
	ref := core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}
	answered := &lastAnswer{}
	// The revision the validate checks: the credential's, or, when none was resolved, the
	// connection's as read before, so a credential saved since is never shown with this result.
	revision := connection.Revision
	// The resolver refuses a connection that is not connected before anything is sent, and
	// moves one whose renewal the provider refused; either way the row says what to do.
	credential, err := s.connectorResolver.Resolve(ctx, ref, core.CredentialRequest{})
	var response *validationResponse
	if err != nil {
		response, err = s.validationAfter(ctx, connection, err)
	} else {
		revision = credential.Revision
		response, err = s.checkConnection(ctx, connection, credential, request.Body, answered)
	}
	if err != nil {
		return nil, err
	}
	validation := response.Body
	record := &store.ConnectorConnectionValidation{ConnectionID: connection.ID, Revision: revision, Status: string(validation.Status),
		Code: validation.Code, Error: storedError(validation.Error, s.connectors.Schemes[connection.AuthScheme], credential),
		CheckedAt: time.Now().UTC().Truncate(time.Microsecond)}
	if validation.CheckedAt != nil {
		record.CheckedAt = *validation.CheckedAt
	}
	if status := answered.status(); record.Code == "" && status >= http.StatusBadRequest {
		record.Code = strconv.Itoa(status)
	}
	if err := s.store.PutConnectorConnectionValidation(ctx, record); err != nil {
		return nil, err
	}
	return response, nil
}

// checkConnection lists the connection's tools through each of its sources with credential,
// the one Resolve gave, stores the list, and compares the granted scopes with what the tools
// need. answered is told the provider's answers.
func (s *Server) checkConnection(ctx context.Context, connection store.ConnectorConnection, credential core.AccessCredential,
	body *ConnectionValidationRequest, answered *lastAnswer) (*validationResponse, error) {
	ref := core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}
	scheme, found := s.connectors.Schemes[connection.AuthScheme]
	if !found {
		return nil, invalidRequest(fmt.Sprintf("auth_scheme %q is not one this deployment has", connection.AuthScheme))
	}
	manifest, err := s.connectionManifest(ctx, connection, connection.DefinitionRevision)
	if err != nil {
		return nil, err
	}
	// A copy of the connection's client, on the same transport and so the same pool, whose
	// answers the validate reads.
	client := *s.connectorTransports.Client(ref, scheme)
	answered.base, client.Transport = client.Transport, answered
	binding := core.ResolvedBinding{
		Connection: coreConnection(connection),
		Manifest:   manifest,
		HTTP:       &client,
	}
	var specs []core.ToolSpec
	for _, kind := range sourceKinds(manifest) {
		source, found := s.connectors.ToolSources[kind]
		if !found {
			return failed(connection, fmt.Sprintf("this deployment has no %s tool source", kind)), nil
		}
		listed, err := source.Discover(ctx, binding)
		if err != nil {
			if refusedStatic(s.connectors.Schemes, connection.AuthScheme, answered.status()) {
				// The 401 path's move (core.Transports invalidates a refused credential nothing
				// renews), so the row and the validate say the same: replace the token or key.
				// Invalidate leaves a connection whose credentials changed since Resolve alone.
				moveErr := s.connectorResolver.Invalidate(ctx, ref, credential, core.Outcome{Kind: core.OutcomeInvalidGrant})
				if moveErr != nil {
					return nil, moveErr
				}
			}
			return s.validationAfter(ctx, connection, err)
		}
		specs = append(specs, listed...)
	}

	var checked []string
	if body != nil {
		checked = body.Tools
	}
	missing, err := missingScopes(specs, checked, connection.GrantedScopes)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	tools := make([]store.ConnectorTool, 0, len(specs))
	for _, spec := range specs {
		tools = append(tools, store.ConnectorTool{Name: spec.Name, Description: spec.Description,
			InputSchema: spec.InputSchema, SchemaDigest: spec.SchemaDigest, NeedsScopes: spec.NeedsScopes})
	}
	digest, err := toolsDigest(tools)
	if err != nil {
		return nil, err
	}
	listedAt := time.Now().UTC().Truncate(time.Microsecond)
	err = s.store.SetConnectorConnectionTools(ctx, connection.CustomerID, connection.ID, tools, digest, listedAt)
	if errors.Is(err, store.ErrNoConnectorConnection) {
		return nil, errNoSuchConnection
	}
	if err != nil {
		return nil, err
	}
	// The credential works, so the events its bindings declare are subscribed to, off the
	// request (internal/mcpevents). Nil with connectors off.
	// The connection is read again, after Resolve took the credential lock: the copy read
	// before it can be another router's renewal in flight, which Reconcile leaves alone.
	if s.mcpEvents != nil {
		resolved, err := s.store.ConnectorConnection(ctx, connection.CustomerID, connection.ID)
		if errors.Is(err, store.ErrNoConnectorConnection) {
			return nil, errNoSuchConnection
		}
		if err != nil {
			return nil, err
		}
		if err := s.mcpEvents.Reconcile(ctx, resolved); err != nil {
			return nil, err
		}
	}
	validation := ConnectionValidation{ConnectionID: connection.ID, Status: validationConnected,
		ToolsDigest: digest, CheckedAt: &listedAt}
	if len(missing) > 0 {
		validation.Status, validation.Code, validation.MissingScopes = validationNeedsScopes, codeScopeRequired, missing
		validation.Error = "the grant lacks scopes its tools need: " + strings.Join(missing, ", ") +
			"; start a consent that asks for them"
	}
	return &validationResponse{Body: validation}, nil
}

// missingScopes is the scopes the checked tools need that granted lacks, sorted (architecture
// doc, «Add» item 11: «granted_scopes against the union of needs_scopes of the granted
// tools»). No names checks every tool; a name the connection does not offer is an error.
func missingScopes(specs []core.ToolSpec, names, granted []string) ([]string, error) {
	needs := map[string][]string{}
	for _, spec := range specs {
		needs[spec.Name] = spec.NeedsScopes
	}
	if names == nil {
		names = slices.Collect(maps.Keys(needs))
	}
	var missing []string
	for _, name := range names {
		scopes, offered := needs[name]
		if !offered {
			return nil, fmt.Errorf("tools: the connection offers no tool named %q", name)
		}
		for _, scope := range scopes {
			if !slices.Contains(granted, scope) && !slices.Contains(missing, scope) {
				missing = append(missing, scope)
			}
		}
	}
	slices.Sort(missing)
	return missing, nil
}

// lastAnswer is the provider's status for the last request of a validate that was answered, 0
// when none was: the answer the validate failed on, since an MCP client stops at the first
// failed request (go-sdk v1.8.0, MaxRetries -1 in sources/mcp). The DELETE that ends an MCP
// session is left out: a source sends it when it closes the session after the failure, and a
// server «MAY respond to this request with HTTP 405 Method Not Allowed» (MCP 2025-11-25,
// Transports, «Session Management»).
type lastAnswer struct {
	base   http.RoundTripper
	mu     sync.Mutex
	answer int
}

func (a *lastAnswer) RoundTrip(request *http.Request) (*http.Response, error) {
	response, err := a.base.RoundTrip(request)
	if err != nil || request.Method == http.MethodDelete {
		return response, err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.answer = response.StatusCode
	return response, err
}

func (a *lastAnswer) status() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.answer
}

// maxStoredErrorBytes caps the error a last validation keeps. A provider's error body reaches
// it whole: go-sdk v1.8.0 copies the body of a non-transient non-2xx answer into the error
// (mcp/streamable.go checkResponse), up to the 4 MiB the mcp source reads
// (sources/mcp/source.go maxResponseBytes), and a list page of up to 200 connections
// (listConnectionsRequest.Limit) would serve 800 MiB of them. 1 KiB is a choice, unverified
// and not measured: it keeps every error the router writes itself whole (the longest, a
// broken revision's, is under 300 bytes before the manifest's reason) and the start of a
// provider's, which is what a person reads first.
const maxStoredErrorBytes = 1 << 10

// storedErrorCut ends an error that was cut at maxStoredErrorBytes.
const storedErrorCut = " [cut]"

// storedRedacted stands for a credential value cut out of an error.
const storedRedacted = "[redacted]"

// storedError is text as a last validation keeps it: every value scheme sends credential as cut
// out, since a provider can echo a request in its error (an Authorization header in a JSON-RPC
// error message), then at most maxStoredErrorBytes of what is left, never splitting a UTF-8
// sequence. The validate's own answer is left as it is.
func storedError(text string, scheme core.Scheme, credential core.AccessCredential) string {
	for _, value := range sentValues(scheme, credential) {
		text = strings.ReplaceAll(text, value, storedRedacted)
	}
	if len(text) <= maxStoredErrorBytes {
		return text
	}
	cut := maxStoredErrorBytes - len(storedErrorCut)
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	return text[:cut] + storedErrorCut
}

// sentValues are the header values scheme's Wrap puts on a request for credential, longest
// first, and each one's part after its auth scheme (the token in "Bearer <token>", RFC 6750
// section 2.1), since a provider may echo either. Wrap is asked rather than the credential read,
// since only the scheme that issued it reads its secret (core.AccessCredential.Secret). Nothing
// is sent: the request stops at headerCatch. A zero credential or no scheme sends none.
func sentValues(scheme core.Scheme, credential core.AccessCredential) []string {
	if scheme == nil || credential.Scheme == "" {
		return nil
	}
	caught := &headerCatch{}
	request, err := http.NewRequest(http.MethodGet, "https://validate.invalid/", nil)
	if err != nil {
		return nil
	}
	_, _ = scheme.Wrap(caught, credential).RoundTrip(request) //nolint:bodyclose // headerCatch returns no response
	var values []string
	for _, sent := range caught.header {
		for _, value := range sent {
			if value != "" {
				values = append(values, value)
			}
			if _, token, found := strings.Cut(value, " "); found && token != "" {
				values = append(values, token)
			}
		}
	}
	slices.SortFunc(values, func(a, b string) int { return len(b) - len(a) })
	return values
}

// headerCatch is a RoundTripper that keeps the header of the request it is handed and sends
// nothing.
type headerCatch struct {
	header http.Header
}

func (c *headerCatch) RoundTrip(request *http.Request) (*http.Response, error) {
	c.header = request.Header.Clone()
	return nil, errors.New("api: a caught request is not sent")
}

// refusedStatic says a validate whose provider last answered status moves a connection of
// scheme to needs_reauthorization (Kanat, 2026-10-10, AI-1052): a bearer or api_key credential
// answered with any 4xx (RFC 9110 section 15.5, client errors) but 429. A provider can refuse a
// token it does not take with something other than 401: GitHub's MCP server answers a wrong
// token with 400 Bad Request (E2E F60). A 429 (RFC 6585 section 4) says to wait, not that the
// credential is wrong. A 5xx, a timeout or no answer says nothing about the credential, and
// keeps the status. OAuth grants keep the 401-only rule, as tool calls do (core.Transports):
// a reconnect, not a new token, is their remedy, and a 4xx on validate does not prove one is due.
func refusedStatic(schemes map[string]core.Scheme, scheme string, status int) bool {
	return core.IsStatic(schemes, scheme) && status >= http.StatusBadRequest && status < http.StatusInternalServerError &&
		status != http.StatusTooManyRequests
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
	// The resolver gives a connection on a revision marked broken no credential and leaves it
	// connected. A token or key is moved off it by saving it again (putConnectionCredentials),
	// never by a consent, so the validate says that, with the code of a rejected one: the
	// remedy a program branches on is the same (AI-1002). An OAuth connection fails as before.
	if now.Status == store.ConnectionConnected && core.IsStatic(s.connectors.Schemes, now.AuthScheme) {
		reason, broken, err := s.store.BrokenConnectorRevision(ctx, now.ConnectorID, now.DefinitionRevision)
		if err != nil {
			return nil, err
		}
		if broken {
			return &validationResponse{Body: ConnectionValidation{ConnectionID: now.ID, Status: validationNeedsReauthorization,
				Code: codeCredentialRejected, Error: fmt.Sprintf("Revision %d of %s is marked broken (%s); save the token or key "+
					"again with PUT /v1/agents/connections/{id}/credentials, which moves the connection to the latest revision",
					now.DefinitionRevision, now.ConnectorID, reason)}}, nil
		}
	}
	switch now.Status {
	case store.ConnectionPending:
		return &validationResponse{Body: ConnectionValidation{ConnectionID: now.ID, Status: validationPending,
			Error: "no credentials yet: start a consent or set its credentials"}}, nil
	case store.ConnectionNeedsReauthorization:
		validation := ConnectionValidation{ConnectionID: now.ID, Status: validationNeedsReauthorization, Error: now.LastError}
		if core.IsStatic(s.connectors.Schemes, now.AuthScheme) {
			validation.Code = codeCredentialRejected
		}
		return &validationResponse{Body: validation}, nil
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
			InputSchema: maps.Clone(tool.InputSchema), SchemaDigest: tool.SchemaDigest,
			NeedsScopes: slices.Clone(tool.NeedsScopes)})
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
