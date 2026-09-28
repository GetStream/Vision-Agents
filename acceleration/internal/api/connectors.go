package api

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const connectorAuthorizationLifetime = 10 * time.Minute
const connectorOAuthLaunchPath = "/v1/agents/connectors/oauth/launch/"
const unknownConnector = "no such connector"

var connectorOAuthLaunchPage = template.Must(template.New("connector-oauth-launch").Parse(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="referrer" content="no-referrer"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Continue connector authorization</title></head>
<body><p id="status">Connecting securely to the provider…</p><script nonce="{{.Nonce}}">
const dashboardOrigin = {{.DashboardOrigin}};
const status = document.getElementById('status');
let handoffStarted = false;
window.addEventListener('message', async (event) => {
  if (handoffStarted || event.source !== window.opener || event.origin !== dashboardOrigin || !event.data || event.data.type !== 'va.connector.oauth.handoff' || typeof event.data.handoff_token !== 'string') return;
  handoffStarted = true;
  try {
    const response = await fetch(window.location.pathname, { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ handoff_token: event.data.handoff_token }) });
    if (!response.ok) throw new Error('Authorization handoff expired. Start again from the dashboard.');
    const authorization = await response.json();
    window.opener = null;
    window.location.replace(authorization.authorization_url);
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : 'Could not start provider authorization.';
  }
});
if (window.opener) window.opener.postMessage({ type: 'va.connector.oauth.ready' }, dashboardOrigin);
else status.textContent = 'Start authorization from the dashboard so this browser can be verified.';
</script></body></html>`))

var connectorAliasPattern = regexp.MustCompile(`^[a-z][a-z0-9_-]{0,62}$`)
var customConnectorIDPattern = regexp.MustCompile(`^custom_[a-z][a-z0-9_-]{0,56}$`)
var connectorToolDigestPattern = regexp.MustCompile(`^[a-f0-9]{64}$`)

// ListConnectors searches the built-in MCP provider catalog.
func (s *Server) ListConnectors(ctx context.Context, request ListConnectorsRequestObject) (ListConnectorsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListConnectors401JSONResponse{missingCustomer()}, nil
	}
	query := strings.ToLower(strings.TrimSpace(value(request.Params.Q)))
	found := mcp.Search(query)
	listed := make([]ConnectorDefinition, 0, len(found))
	for _, connector := range found {
		listed = append(listed, connectorDefinitionOf(connector))
	}
	if s.store != nil {
		definitions, err := s.store.ConnectorDefinitions(ctx, customerID)
		if err != nil {
			return nil, err
		}
		for _, definition := range definitions {
			if query != "" && !strings.Contains(strings.ToLower(definition.ID+" "+definition.Name+" "+definition.Category+" "+definition.Description), query) {
				continue
			}
			listed = append(listed, connectorDefinitionOf(connectorFromDefinition(definition)))
		}
	}
	return ListConnectors200JSONResponse(listed), nil
}

// CreateConnectorDefinition registers a fixed public MCP endpoint and auth policy for an app.
func (s *Server) CreateConnectorDefinition(ctx context.Context, request CreateConnectorDefinitionRequestObject) (CreateConnectorDefinitionResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateConnectorDefinition401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return CreateConnectorDefinition400JSONResponse{badRequest(noConfigs)}, nil
	}
	if request.Body == nil {
		return CreateConnectorDefinition400JSONResponse{badRequest("a request body is required")}, nil
	}
	body := request.Body
	if !customConnectorIDPattern.MatchString(body.Id) {
		return CreateConnectorDefinition400JSONResponse{badRequest("custom connector ids must start with custom_ and use lowercase letters, digits, hyphens or underscores")}, nil
	}
	if _, exists := mcp.Lookup(body.Id); exists {
		return CreateConnectorDefinition400JSONResponse{badRequest("connector id is already reserved")}, nil
	}
	name := strings.TrimSpace(body.Name)
	if name == "" {
		return CreateConnectorDefinition400JSONResponse{badRequest("connector name is required")}, nil
	}
	if err := egress.ValidatePublicHTTPSURL(ctx, body.Endpoint); err != nil {
		return CreateConnectorDefinition400JSONResponse{badRequest("connector endpoints must be public HTTPS services without query strings")}, nil
	}
	authType := ""
	apiKeyHeader := strings.TrimSpace(value(body.ApiKeyHeader))
	switch body.AuthMode {
	case CreateConnectorDefinitionRequestAuthModeOauthDcr:
		authType = connectors.AuthOAuth2
	case CreateConnectorDefinitionRequestAuthModeNone:
		authType = connectors.AuthNone
	case CreateConnectorDefinitionRequestAuthModeBearer:
		authType = connectors.AuthBearer
	case CreateConnectorDefinitionRequestAuthModeApiKey:
		authType = connectors.AuthAPIKey
		header := http.CanonicalHeaderKey(apiKeyHeader)
		if header == "" || forbiddenAPIKeyHeader(header) {
			return CreateConnectorDefinition400JSONResponse{badRequest("api_key_header must be a safe fixed header name")}, nil
		}
		apiKeyHeader = header
	default:
		return CreateConnectorDefinition400JSONResponse{badRequest("unsupported connector auth mode")}, nil
	}
	if authType != connectors.AuthAPIKey && apiKeyHeader != "" {
		return CreateConnectorDefinition400JSONResponse{badRequest("api_key_header is only valid for api_key auth")}, nil
	}
	definition := store.ConnectorDefinition{
		CustomerID:  customerID,
		ID:          body.Id,
		Name:        name,
		Category:    strings.TrimSpace(value(body.Category)),
		Description: strings.TrimSpace(value(body.Description)),
		Endpoint:    body.Endpoint,
		AuthType:    authType,
		AuthHeader:  apiKeyHeader,
	}
	if err := s.store.CreateConnectorDefinition(ctx, &definition); err != nil {
		return CreateConnectorDefinition400JSONResponse{badRequest("could not create connector definition")}, nil
	}
	return CreateConnectorDefinition201JSONResponse(connectorDefinitionOf(connectorFromDefinition(definition))), nil
}

// GetConnectorDefinition reads one built-in or app-owned connector definition.
func (s *Server) GetConnectorDefinition(ctx context.Context, request GetConnectorDefinitionRequestObject) (GetConnectorDefinitionResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetConnectorDefinition401JSONResponse{missingCustomer()}, nil
	}
	if connector, exists := mcp.Lookup(string(request.Id)); exists {
		return GetConnectorDefinition200JSONResponse(connectorDefinitionOf(connector)), nil
	}
	if s.store == nil {
		return GetConnectorDefinition404JSONResponse{NotFoundJSONResponse{Error: "no such connector"}}, nil
	}
	definition, err := s.store.ConnectorDefinition(ctx, customerID, string(request.Id))
	if err != nil {
		return GetConnectorDefinition404JSONResponse{NotFoundJSONResponse{Error: "no such connector"}}, nil
	}
	return GetConnectorDefinition200JSONResponse(connectorDefinitionOf(connectorFromDefinition(definition))), nil
}

// ListConnectorConnections lists an app's reusable account connections.
func (s *Server) ListConnectorConnections(ctx context.Context, request ListConnectorConnectionsRequestObject) (ListConnectorConnectionsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListConnectorConnections401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ListConnectorConnections400JSONResponse{badRequest(noConfigs)}, nil
	}
	found, err := s.store.ConnectorConnections(ctx, customerID)
	if err != nil {
		return nil, err
	}
	listed := make([]ConnectorConnection, 0, len(found))
	for _, connection := range found {
		if !connectionOwnerMatches(ctx, connection) {
			continue
		}
		if request.Params.ConnectorId != nil && connection.ConnectorID != *request.Params.ConnectorId {
			continue
		}
		if request.Params.OwnerType != nil && connection.OwnerType != string(*request.Params.OwnerType) {
			continue
		}
		if request.Params.OwnerId != nil && connection.OwnerID != *request.Params.OwnerId {
			continue
		}
		listed = append(listed, connectorConnectionOf(connection))
	}
	return ListConnectorConnections200JSONResponse(listed), nil
}

// CreateConnectorConnection creates a reusable pending connection separate from any agent.
func (s *Server) CreateConnectorConnection(ctx context.Context, request CreateConnectorConnectionRequestObject) (CreateConnectorConnectionResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateConnectorConnection401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return CreateConnectorConnection400JSONResponse{badRequest(noConfigs)}, nil
	}
	if request.Body == nil {
		return CreateConnectorConnection400JSONResponse{badRequest("a request body is required")}, nil
	}
	connector, err := s.connectorDefinition(ctx, customerID, request.Body.ConnectorId)
	if err != nil {
		return CreateConnectorConnection400JSONResponse{badRequest(unknownConnector)}, nil
	}
	ownerType, ownerID := string(request.Body.Owner.Type), ""
	switch request.Body.Owner.Type {
	case ConnectorOwnerTypeApp:
	case ConnectorOwnerTypeUser:
		if !ServerSideFrom(ctx) {
			return CreateConnectorConnection400JSONResponse{badRequest("user-owned connections must be created by the authenticated backend")}, nil
		}
		ownerID = strings.TrimSpace(value(request.Body.Owner.UserId))
		if ownerID == "" {
			return CreateConnectorConnection400JSONResponse{badRequest("a user-owned connection needs owner.user_id")}, nil
		}
		if callerID := CallerFrom(ctx).UserID; callerID == "" || ownerID != callerID {
			return CreateConnectorConnection400JSONResponse{badRequest("owner.user_id must match the verified user this backend is acting for")}, nil
		}
	default:
		return CreateConnectorConnection400JSONResponse{badRequest("connection owner must be app or user")}, nil
	}
	instance := strings.TrimSpace(value(request.Body.Instance))
	endpoint, err := connector.Endpoint(instance)
	if err != nil {
		return CreateConnectorConnection400JSONResponse{badRequest(err.Error())}, nil
	}
	if err := egress.ValidatePublicHTTPSURL(ctx, endpoint); err != nil {
		return CreateConnectorConnection400JSONResponse{badRequest("connector endpoints must be public HTTPS services")}, nil
	}
	connection := store.ConnectorConnection{
		CustomerID:  customerID,
		ConnectorID: connector.ID,
		OwnerType:   ownerType,
		OwnerID:     ownerID,
		Endpoint:    endpoint,
		Instance:    instance,
		Label:       strings.TrimSpace(value(request.Body.Label)),
		AuthType:    connectorAuthType(connector),
		AuthHeader:  connector.AuthHeader,
	}
	if err := s.store.CreateConnectorConnection(ctx, &connection); err != nil {
		return CreateConnectorConnection400JSONResponse{badRequest(err.Error())}, nil
	}
	return CreateConnectorConnection201JSONResponse(connectorConnectionOf(connection)), nil
}

// GetConnectorConnection reads one connection's nonsecret metadata.
func (s *Server) GetConnectorConnection(ctx context.Context, request GetConnectorConnectionRequestObject) (GetConnectorConnectionResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetConnectorConnection401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetConnectorConnection400JSONResponse{badRequest(noConfigs)}, nil
	}
	connection, err := s.store.ConnectorConnection(ctx, customerID, string(request.Id))
	if err != nil || !connectionOwnerMatches(ctx, connection) {
		return GetConnectorConnection404JSONResponse{NotFoundJSONResponse{Error: "no such connection"}}, nil
	}
	return GetConnectorConnection200JSONResponse(connectorConnectionOf(connection)), nil
}

// DeleteConnectorConnection disconnects one reusable account and erases its local grant.
func (s *Server) DeleteConnectorConnection(ctx context.Context, request DeleteConnectorConnectionRequestObject) (DeleteConnectorConnectionResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return DeleteConnectorConnection401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return DeleteConnectorConnection400JSONResponse{badRequest(noConfigs)}, nil
	}
	connection, err := s.store.ConnectorConnection(ctx, customerID, string(request.Id))
	if err != nil || !connectionOwnerMatches(ctx, connection) {
		return DeleteConnectorConnection404JSONResponse{NotFoundJSONResponse{Error: "no such connection"}}, nil
	}
	if err := s.store.DeleteConnectorConnection(ctx, customerID, string(request.Id)); err != nil {
		return DeleteConnectorConnection404JSONResponse{NotFoundJSONResponse{Error: "no such connection"}}, nil
	}
	return DeleteConnectorConnection204Response{}, nil
}

// AuthorizeConnectorConnection starts a provider-specific OAuth authorization attempt.
func (s *Server) AuthorizeConnectorConnection(ctx context.Context, request AuthorizeConnectorConnectionRequestObject) (AuthorizeConnectorConnectionResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return AuthorizeConnectorConnection401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return AuthorizeConnectorConnection400JSONResponse{badRequest(noConfigs)}, nil
	}
	if s.credentialSealer == nil {
		return AuthorizeConnectorConnection400JSONResponse{badRequest("encrypted credential storage is unavailable; configure a key encryption key")}, nil
	}
	connection, err := s.store.ConnectorConnection(ctx, customerID, string(request.Id))
	if err != nil {
		return AuthorizeConnectorConnection404JSONResponse{NotFoundJSONResponse{Error: "no such connection"}}, nil
	}
	if !connectionOwnerMatches(ctx, connection) {
		return AuthorizeConnectorConnection404JSONResponse{NotFoundJSONResponse{Error: "no such connection"}}, nil
	}
	connector, err := s.connectorDefinition(ctx, customerID, connection.ConnectorID)
	if err != nil {
		return AuthorizeConnectorConnection404JSONResponse{NotFoundJSONResponse{Error: "no such connector"}}, nil
	}
	if connectorAuthType(connector) != connectors.AuthOAuth2 {
		return AuthorizeConnectorConnection400JSONResponse{badRequest("this connector does not use OAuth")}, nil
	}
	clientID, clientSecret := "", ""
	if request.Body != nil {
		clientID = value(request.Body.OauthClientId)
		clientSecret = value(request.Body.OauthClientSecret)
	}
	if request.Body != nil && request.Body.Scopes != nil && len(*request.Body.Scopes) > 0 {
		allowedScopes := make(map[string]struct{}, len(connector.Scopes))
		for _, scope := range connector.Scopes {
			allowedScopes[scope] = struct{}{}
		}
		seenScopes := make(map[string]struct{}, len(*request.Body.Scopes))
		for _, scope := range *request.Body.Scopes {
			if _, allowed := allowedScopes[scope]; !allowed {
				return AuthorizeConnectorConnection400JSONResponse{badRequest("requested scope is not supported by this connector")}, nil
			}
			if _, duplicate := seenScopes[scope]; duplicate {
				return AuthorizeConnectorConnection400JSONResponse{badRequest("requested scopes must be unique")}, nil
			}
			seenScopes[scope] = struct{}{}
		}
		connector.Scopes = append([]string(nil), (*request.Body.Scopes)...)
	} else if connector.ID == "slack" {
		return AuthorizeConnectorConnection400JSONResponse{badRequest("choose the minimum Slack scopes this connection needs")}, nil
	}
	if connector.OAuthMode == "customer_confidential" {
		if clientID == "" || clientSecret == "" {
			return AuthorizeConnectorConnection400JSONResponse{badRequest("this connector requires the customer-registered OAuth client ID and secret")}, nil
		}
	} else if connector.OAuthMode == "customer_dcr" {
		if (clientID == "") != (clientSecret == "") {
			return AuthorizeConnectorConnection400JSONResponse{badRequest("provide both customer-registered OAuth credentials or leave both blank for dynamic registration")}, nil
		}
	} else if clientID != "" || clientSecret != "" {
		return AuthorizeConnectorConnection400JSONResponse{badRequest("this connector uses server-managed OAuth credentials")}, nil
	}
	if strings.TrimSpace(s.publicURL) == "" {
		return AuthorizeConnectorConnection400JSONResponse{badRequest("the public router URL must be configured before starting OAuth")}, nil
	}
	pending, err := s.oauth.StartAuthorizeWithClient(ctx, connector, connection.Instance, clientID, clientSecret)
	if err != nil {
		if errors.Is(err, mcp.ErrOAuthClientNotConfigured) {
			return AuthorizeConnectorConnection400JSONResponse{badRequest(mcp.ErrOAuthClientNotConfigured.Error())}, nil
		}
		return AuthorizeConnectorConnection400JSONResponse{badRequest("could not start provider authorization")}, nil
	}
	attemptID, err := randomConnectorID()
	if err != nil {
		return nil, err
	}
	browserBinding, err := randomConnectorID()
	if err != nil {
		return nil, err
	}
	attempt := connectors.AuthorizationAttempt{
		ConnectionID:   connection.ID,
		Revision:       connection.Revision,
		ConnectorID:    connector.ID,
		BrowserBinding: browserBinding,
		Pending:        pending,
	}
	sealed, err := connectors.SealAuthorizationAttempt(s.credentialSealer, attemptID, attempt)
	if err != nil {
		return nil, err
	}
	expiresAt := time.Now().UTC().Add(connectorAuthorizationLifetime)
	if err := s.store.CreateConnectorAuthorizationAttempt(ctx, &store.ConnectorAuthorizationAttempt{
		ID:            attemptID,
		CustomerID:    customerID,
		ConnectionID:  connection.ID,
		StateHash:     store.OAuthStateHash(pending.State),
		AttemptSealed: sealed,
		KEKVersion:    s.credentialSealer.CurrentVersion(),
		ExpiresAt:     expiresAt,
	}); err != nil {
		return nil, err
	}
	return AuthorizeConnectorConnection200JSONResponse(ConnectorAuthorization{
		AuthorizationId:  attemptID,
		AuthorizationUrl: strings.TrimRight(s.publicURL, "/") + connectorOAuthLaunchPath + attemptID,
		HandoffToken:     browserBinding,
		ExpiresAt:        expiresAt,
	}), nil
}

func (s *Server) connectorOAuthLaunchPageHandler(w http.ResponseWriter, r *http.Request) {
	if r.PathValue("id") == "" {
		http.NotFound(w, r)
		return
	}
	nonce, err := randomConnectorID()
	if err != nil {
		http.Error(w, "could not start authorization", http.StatusInternalServerError)
		return
	}
	dashboardURL := s.dashboardURL
	if dashboardURL == "" {
		dashboardURL = "http://localhost:3000"
	}
	dashboardOrigin, err := connectorOrigin(dashboardURL)
	if err != nil {
		http.Error(w, "dashboard origin is not configured", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; script-src 'nonce-"+nonce+"'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := connectorOAuthLaunchPage.Execute(w, struct {
		Nonce           string
		DashboardOrigin string
	}{Nonce: nonce, DashboardOrigin: dashboardOrigin}); err != nil {
		s.logger.Error("render connector OAuth launch page", "error", err)
	}
}

// connectorOAuthClientMetadataHandler publishes the router's public OAuth client identity
// for authorization servers that support Client ID Metadata Documents.
func (s *Server) connectorOAuthClientMetadataHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Cache-Control", "public, max-age=300")
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	if err := json.NewEncoder(w).Encode(s.oauth.ClientMetadataDocument()); err != nil {
		http.Error(w, "could not serve OAuth client metadata", http.StatusInternalServerError)
	}
}

func (s *Server) connectorOAuthLaunchHandoff(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" || s.store == nil || s.credentialSealer == nil {
		http.Error(w, "authorization handoff is invalid or expired", http.StatusBadRequest)
		return
	}
	publicOrigin, err := connectorOrigin(s.publicURL)
	if err != nil || r.Header.Get("Origin") != publicOrigin {
		http.Error(w, "authorization handoff origin is invalid", http.StatusForbidden)
		return
	}
	if r.Body == nil {
		http.Error(w, "authorization handoff is invalid", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()
	var body struct {
		HandoffToken string `json:"handoff_token"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&body); err != nil || body.HandoffToken == "" {
		http.Error(w, "authorization handoff is invalid", http.StatusBadRequest)
		return
	}
	row, err := s.store.ConnectorAuthorizationAttemptByID(r.Context(), id)
	if err != nil {
		http.Error(w, "authorization handoff is invalid or expired", http.StatusBadRequest)
		return
	}
	attempt, err := connectors.OpenAuthorizationAttempt(s.credentialSealer, row.ID, row.KEKVersion, row.AttemptSealed)
	if err != nil || subtle.ConstantTimeCompare([]byte(body.HandoffToken), []byte(attempt.BrowserBinding)) != 1 {
		http.Error(w, "authorization handoff is invalid", http.StatusForbidden)
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Type", "application/json")
	http.SetCookie(w, s.connectorAuthorizationCookie(row.ID, attempt.BrowserBinding, row.ExpiresAt))
	if err := json.NewEncoder(w).Encode(struct {
		AuthorizationURL string `json:"authorization_url"`
	}{AuthorizationURL: attempt.Pending.AuthorizeURL}); err != nil {
		s.logger.Error("write connector OAuth handoff response", "error", err)
	}
}

// PutConnectorCredentials writes a static credential, imports an OAuth grant, or activates an anonymous MCP connection.
func (s *Server) PutConnectorCredentials(ctx context.Context, request PutConnectorCredentialsRequestObject) (PutConnectorCredentialsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return PutConnectorCredentials401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return PutConnectorCredentials400JSONResponse{badRequest(noConfigs)}, nil
	}
	if request.Body == nil {
		return PutConnectorCredentials400JSONResponse{badRequest("a request body is required")}, nil
	}
	connection, err := s.store.ConnectorConnection(ctx, customerID, string(request.Id))
	if err != nil || !connectionOwnerMatches(ctx, connection) {
		return PutConnectorCredentials404JSONResponse{NotFoundJSONResponse{Error: "no such connection"}}, nil
	}
	if request.Body.ExpectedRevision != connection.Revision {
		return PutConnectorCredentials409JSONResponse{ConflictJSONResponse{Error: "connection revision changed"}}, nil
	}
	if connection.AuthType == connectors.AuthOAuth2 {
		return s.putImportedOAuthGrant(ctx, customerID, connection, request.Body)
	}
	if request.Body.AccessToken != nil || request.Body.RefreshToken != nil || request.Body.ExpiresAt != nil ||
		request.Body.GrantedScopes != nil || request.Body.OauthClientId != nil || request.Body.OauthClientSecret != nil {
		return PutConnectorCredentials400JSONResponse{badRequest("OAuth grant fields are only valid for OAuth connections")}, nil
	}
	nextRevision := connection.Revision + 1
	var sealed []byte
	switch connection.AuthType {
	case connectors.AuthNone:
		if request.Body.BearerToken != nil || request.Body.ApiKey != nil {
			return PutConnectorCredentials400JSONResponse{badRequest("no-auth connections do not accept a credential")}, nil
		}
		sealed = []byte{}
	case connectors.AuthBearer:
		if request.Body.BearerToken == nil || strings.TrimSpace(*request.Body.BearerToken) == "" || request.Body.ApiKey != nil {
			return PutConnectorCredentials400JSONResponse{badRequest("bearer auth requires only bearer_token")}, nil
		}
		if s.credentialSealer == nil {
			return PutConnectorCredentials400JSONResponse{badRequest("encrypted credential storage is unavailable")}, nil
		}
		sealed, err = connectors.SealCredentials(s.credentialSealer, customerID, connection.ID, nextRevision,
			connectors.Credentials{AuthType: connectors.AuthBearer, AccessToken: *request.Body.BearerToken})
	case connectors.AuthAPIKey:
		if request.Body.ApiKey == nil || strings.TrimSpace(*request.Body.ApiKey) == "" || request.Body.BearerToken != nil {
			return PutConnectorCredentials400JSONResponse{badRequest("api_key auth requires only api_key")}, nil
		}
		if s.credentialSealer == nil {
			return PutConnectorCredentials400JSONResponse{badRequest("encrypted credential storage is unavailable")}, nil
		}
		sealed, err = connectors.SealCredentials(s.credentialSealer, customerID, connection.ID, nextRevision,
			connectors.Credentials{AuthType: connectors.AuthAPIKey, APIKey: *request.Body.ApiKey})
	default:
		return PutConnectorCredentials400JSONResponse{badRequest("unsupported connection auth mode")}, nil
	}
	if err != nil {
		return nil, err
	}
	connection.CredentialSealed = sealed
	if s.credentialSealer != nil {
		connection.CredentialKEKVersion = s.credentialSealer.CurrentVersion()
	}
	connection.ExpiresAt = nil
	connection.GrantedScopes = []string{}
	connection.CachedTools = []store.ConnectorTool{}
	connection.ToolsDigest = ""
	connection.ToolsCheckedAt = nil
	connection.LastError = ""
	connection.Status = store.ConnectorConnected
	connection.Revision = nextRevision
	if err := s.store.SaveConnectorConnectionAtRevision(ctx, &connection, request.Body.ExpectedRevision); err != nil {
		if errors.Is(err, store.ErrConnectorConnectionChanged) {
			return PutConnectorCredentials409JSONResponse{ConflictJSONResponse{Error: "connection revision changed"}}, nil
		}
		return nil, err
	}
	return PutConnectorCredentials200JSONResponse(connectorConnectionOf(connection)), nil
}

func (s *Server) putImportedOAuthGrant(
	ctx context.Context,
	customerID string,
	connection store.ConnectorConnection,
	body *PutConnectorCredentialsRequest,
) (PutConnectorCredentialsResponseObject, error) {
	if body.AccessToken == nil || strings.TrimSpace(*body.AccessToken) == "" || body.ExpiresAt == nil ||
		body.BearerToken != nil || body.ApiKey != nil {
		return PutConnectorCredentials400JSONResponse{badRequest("OAuth grant import requires access_token and expires_at only")}, nil
	}
	if s.credentialSealer == nil {
		return PutConnectorCredentials400JSONResponse{badRequest("encrypted credential storage is unavailable")}, nil
	}
	connector, err := s.connectorDefinition(ctx, customerID, connection.ConnectorID)
	if err != nil {
		return PutConnectorCredentials404JSONResponse{NotFoundJSONResponse{Error: "no such connector"}}, nil
	}
	clientID, clientSecret := value(body.OauthClientId), value(body.OauthClientSecret)
	pending, err := s.oauth.OAuthClientForImport(ctx, connector, connection.Instance, clientID, clientSecret)
	if err != nil {
		return PutConnectorCredentials400JSONResponse{badRequest("OAuth grant does not match the connector's configured client")}, nil
	}
	scopes := []string{}
	if body.GrantedScopes != nil {
		allowedScopes := make(map[string]struct{}, len(connector.Scopes))
		for _, scope := range connector.Scopes {
			allowedScopes[scope] = struct{}{}
		}
		seenScopes := make(map[string]struct{}, len(*body.GrantedScopes))
		for _, scope := range *body.GrantedScopes {
			if _, allowed := allowedScopes[scope]; !allowed {
				return PutConnectorCredentials400JSONResponse{badRequest("imported OAuth scope is not supported by this connector")}, nil
			}
			if _, duplicate := seenScopes[scope]; duplicate {
				return PutConnectorCredentials400JSONResponse{badRequest("imported OAuth scopes must be unique")}, nil
			}
			seenScopes[scope] = struct{}{}
			scopes = append(scopes, scope)
		}
	} else if len(connector.Scopes) > 0 {
		return PutConnectorCredentials400JSONResponse{badRequest("granted_scopes is required for this connector")}, nil
	}
	nextRevision := connection.Revision + 1
	credentials, err := connectors.SealCredentials(s.credentialSealer, customerID, connection.ID, nextRevision, connectors.Credentials{
		AuthType:          connectors.AuthOAuth2,
		AccessToken:       *body.AccessToken,
		RefreshToken:      value(body.RefreshToken),
		OAuthClientID:     pending.ClientID,
		OAuthClientSecret: pending.ClientSecret,
		ClientAuthMethod:  pending.ClientAuthMethod,
		OAuthIssuer:       pending.Issuer,
		TokenEndpoint:     pending.TokenEndpoint,
		RefreshEndpoint:   pending.RefreshEndpoint,
		Resource:          pending.Resource,
	})
	if err != nil {
		return nil, err
	}
	connection.CredentialSealed = credentials
	connection.CredentialKEKVersion = s.credentialSealer.CurrentVersion()
	connection.ExpiresAt = body.ExpiresAt
	connection.GrantedScopes = scopes
	connection.CachedTools = []store.ConnectorTool{}
	connection.ToolsDigest = ""
	connection.ToolsCheckedAt = nil
	connection.LastError = ""
	connection.Status = store.ConnectorConnected
	connection.Revision = nextRevision
	if err := s.store.SaveConnectorConnectionAtRevision(ctx, &connection, body.ExpectedRevision); err != nil {
		if errors.Is(err, store.ErrConnectorConnectionChanged) {
			return PutConnectorCredentials409JSONResponse{ConflictJSONResponse{Error: "connection revision changed"}}, nil
		}
		return nil, err
	}
	return PutConnectorCredentials200JSONResponse(connectorConnectionOf(connection)), nil
}

// ListConnectorTools returns the last explicitly validated tool snapshot.
func (s *Server) ListConnectorTools(ctx context.Context, request ListConnectorToolsRequestObject) (ListConnectorToolsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListConnectorTools401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ListConnectorTools400JSONResponse{badRequest(noConfigs)}, nil
	}
	connection, err := s.store.ConnectorConnection(ctx, customerID, string(request.Id))
	if err != nil || !connectionOwnerMatches(ctx, connection) {
		return ListConnectorTools404JSONResponse{NotFoundJSONResponse{Error: "no such connection"}}, nil
	}
	tools := connectorToolsOf(connection.CachedTools)
	return ListConnectorTools200JSONResponse(ConnectorTools{
		ConnectionId: connection.ID,
		Tools:        tools,
		Digest:       connection.ToolsDigest,
		CheckedAt:    connection.ToolsCheckedAt,
	}), nil
}

// ValidateConnectorConnection refreshes OAuth when needed, opens MCP, and caches its schemas.
func (s *Server) ValidateConnectorConnection(ctx context.Context, request ValidateConnectorConnectionRequestObject) (ValidateConnectorConnectionResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ValidateConnectorConnection401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ValidateConnectorConnection400JSONResponse{badRequest(noConfigs)}, nil
	}
	connection, err := s.store.ConnectorConnection(ctx, customerID, string(request.Id))
	if err != nil || !connectionOwnerMatches(ctx, connection) {
		return ValidateConnectorConnection404JSONResponse{NotFoundJSONResponse{Error: "no such connection"}}, nil
	}
	connector, err := s.connectorDefinition(ctx, customerID, connection.ConnectorID)
	if err != nil {
		return ValidateConnectorConnection404JSONResponse{NotFoundJSONResponse{Error: "no such connector"}}, nil
	}
	if _, err := connectors.ResolveCredentials(ctx, s.store, s.credentialSealer, customerID, connection.ID, s.oauth); err != nil {
		updated, loadErr := s.store.ConnectorConnection(ctx, customerID, connection.ID)
		if loadErr == nil {
			connection = updated
		}
		return ValidateConnectorConnection200JSONResponse(ConnectorValidation{
			ConnectionId: connection.ID,
			Status:       ConnectorValidationStatusNeedsReauthorization,
			ToolsDigest:  connection.ToolsDigest,
			Error:        optional("reauthorization required"),
		}), nil
	}
	connection, err = s.store.ConnectorConnection(ctx, customerID, connection.ID)
	if err != nil {
		return nil, err
	}
	runtime, discovered, failures := mcp.Open(ctx, []mcp.Connection{{
		ConnectorID: connector.ID,
		Endpoint:    connection.Endpoint,
		Authorize: func(callCtx context.Context, request *http.Request) error {
			return connectors.AuthorizeRequest(callCtx, s.store, s.credentialSealer,
				customerID, connection.ID, s.oauth, request)
		},
	}}, nil)
	if runtime != nil {
		runtime.Close()
	}
	connection, err = s.store.ConnectorConnection(ctx, customerID, connection.ID)
	if err != nil {
		return nil, err
	}
	if len(failures) > 0 && len(discovered) == 0 {
		connection.LastError = "MCP validation failed; check provider access and reconnect if needed"
		_ = s.store.SaveConnectorConnection(ctx, &connection)
		return ValidateConnectorConnection200JSONResponse(ConnectorValidation{
			ConnectionId: connection.ID,
			Status:       ConnectorValidationStatusFailed,
			ToolsDigest:  connection.ToolsDigest,
			Error:        optional("MCP validation failed"),
		}), nil
	}
	tools := make([]store.ConnectorTool, 0, len(discovered))
	for _, tool := range discovered {
		_, name, ok := mcp.Split(tool.Name)
		if !ok {
			continue
		}
		tools = append(tools, store.ConnectorTool{
			Name:         name,
			Description:  tool.Description,
			InputSchema:  tool.Parameters,
			SchemaDigest: tool.SchemaDigest,
		})
	}
	digest, err := connectorToolsDigest(tools)
	if err != nil {
		return nil, err
	}
	checkedAt := time.Now().UTC()
	connection.CachedTools = tools
	connection.ToolsDigest = digest
	connection.ToolsCheckedAt = &checkedAt
	connection.LastError = ""
	connection.Status = store.ConnectorConnected
	if err := s.store.SaveConnectorConnection(ctx, &connection); err != nil {
		return nil, err
	}
	return ValidateConnectorConnection200JSONResponse(ConnectorValidation{
		ConnectionId: connection.ID,
		Status:       ConnectorValidationStatusConnected,
		ToolsDigest:  digest,
		CheckedAt:    &checkedAt,
	}), nil
}

// finishConnectorLogin is the unauthenticated provider redirect endpoint.
func (s *Server) finishConnectorLogin(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	state := query.Get("state")
	if state == "" || s.store == nil || s.credentialSealer == nil {
		http.Error(w, "authorization state is invalid", http.StatusBadRequest)
		return
	}
	pendingRow, err := s.store.ConnectorAuthorizationAttemptByState(r.Context(), state)
	if err != nil {
		http.Error(w, "authorization state is invalid, expired or already used", http.StatusBadRequest)
		return
	}
	attempt, err := connectors.OpenAuthorizationAttempt(s.credentialSealer, pendingRow.ID, pendingRow.KEKVersion, pendingRow.AttemptSealed)
	if err != nil || attempt.Pending.State != state || attempt.ConnectionID != pendingRow.ConnectionID || attempt.BrowserBinding == "" {
		http.Error(w, "authorization state is invalid", http.StatusBadRequest)
		return
	}
	browserCookie, err := r.Cookie(connectorAuthorizationCookieName(pendingRow.ID))
	if err != nil || subtle.ConstantTimeCompare([]byte(browserCookie.Value), []byte(attempt.BrowserBinding)) != 1 {
		http.Error(w, "authorization must finish in the browser that started it", http.StatusForbidden)
		return
	}
	attemptRow, err := s.store.ConsumeConnectorAuthorizationAttempt(r.Context(), state)
	if err != nil || attemptRow.ID != pendingRow.ID {
		http.Error(w, "authorization state is invalid, expired or already used", http.StatusBadRequest)
		return
	}
	s.clearConnectorAuthorizationCookie(w, attemptRow.ID)
	connection, err := s.store.ConnectorConnection(r.Context(), attemptRow.CustomerID, attempt.ConnectionID)
	if err != nil || connection.Revision != attempt.Revision || connection.ConnectorID != attempt.ConnectorID {
		http.Error(w, "the connection changed during authorization", http.StatusConflict)
		return
	}
	responseIssuer := query.Get("iss")
	if err := attempt.Pending.ValidateAuthorizationResponseIssuer(responseIssuer); err != nil {
		if connection.Status == store.ConnectorPending {
			connection.Status = store.ConnectorFailed
			connection.LastError = "authorization server identity could not be verified"
			_ = s.store.SaveConnectorConnection(r.Context(), &connection)
		}
		s.redirectAfterConnectorLogin(w, connection.ID, "failed")
		return
	}
	if providerError := query.Get("error"); providerError != "" {
		if connection.Status == store.ConnectorPending {
			connection.Status = store.ConnectorFailed
			connection.LastError = "provider authorization was declined"
			_ = s.store.SaveConnectorConnection(r.Context(), &connection)
		}
		s.redirectAfterConnectorLogin(w, connection.ID, "failed")
		return
	}
	code := query.Get("code")
	if code == "" {
		http.Error(w, "authorization code is missing", http.StatusBadRequest)
		return
	}
	authenticator := s.oauth
	token, err := authenticator.ExchangeWithIssuer(r.Context(), attempt.Pending, code, responseIssuer)
	if err != nil {
		if connection.Status == store.ConnectorPending {
			connection.Status = store.ConnectorFailed
			connection.LastError = "provider token exchange failed"
			_ = s.store.SaveConnectorConnection(r.Context(), &connection)
		}
		s.redirectAfterConnectorLogin(w, connection.ID, "failed")
		return
	}
	if connection.AccountID != "" && connection.AccountID != token.AccountID {
		connection.LastError = "authorization returned a different or unverified provider account; the existing connection was preserved; create a new connection for this account"
		if err := s.store.SaveConnectorConnectionAtRevision(r.Context(), &connection, attempt.Revision); err != nil {
			if errors.Is(err, store.ErrConnectorConnectionChanged) {
				http.Error(w, "the connection changed during authorization", http.StatusConflict)
				return
			}
			http.Error(w, "could not preserve the existing connection", http.StatusInternalServerError)
			return
		}
		s.redirectAfterConnectorLogin(w, connection.ID, "failed")
		return
	}
	nextRevision := connection.Revision + 1
	credentials, err := connectors.SealCredentials(s.credentialSealer, attemptRow.CustomerID, connection.ID, nextRevision, connectors.Credentials{
		AccessToken:       token.AccessToken,
		RefreshToken:      token.RefreshToken,
		OAuthClientID:     attempt.Pending.ClientID,
		OAuthClientSecret: attempt.Pending.ClientSecret,
		ClientAuthMethod:  attempt.Pending.ClientAuthMethod,
		OAuthIssuer:       attempt.Pending.Issuer,
		TokenEndpoint:     attempt.Pending.TokenEndpoint,
		RefreshEndpoint:   attempt.Pending.RefreshEndpoint,
		Resource:          attempt.Pending.Resource,
	})
	if err != nil {
		http.Error(w, "could not store provider credentials", http.StatusInternalServerError)
		return
	}
	connection.CredentialSealed = credentials
	connection.CredentialKEKVersion = s.credentialSealer.CurrentVersion()
	connection.ExpiresAt = token.ExpiresAt
	connection.GrantedScopes = token.Scopes
	connection.AccountID = token.AccountID
	if len(connection.GrantedScopes) == 0 {
		connection.GrantedScopes = append([]string(nil), attempt.Pending.Scopes...)
	}
	connection.Status = store.ConnectorConnected
	connection.LastError = ""
	connection.Revision = nextRevision
	if err := s.store.SaveConnectorConnectionAtRevision(r.Context(), &connection, attempt.Revision); err != nil {
		if errors.Is(err, store.ErrConnectorConnectionChanged) {
			http.Error(w, "the connection changed during authorization", http.StatusConflict)
			return
		}
		http.Error(w, "could not save the connection", http.StatusInternalServerError)
		return
	}
	s.redirectAfterConnectorLogin(w, connection.ID, "connected")
}

func (s *Server) redirectAfterConnectorLogin(w http.ResponseWriter, connectionID, status string) {
	base := s.dashboardURL
	if base == "" {
		base = "http://localhost:3000"
	}
	destination, err := url.Parse(base)
	if err != nil {
		http.Error(w, "dashboard URL is invalid", http.StatusInternalServerError)
		return
	}
	query := destination.Query()
	query.Set("connection_id", connectionID)
	query.Set("status", status)
	destination.RawQuery = query.Encode()
	w.Header().Set("Location", destination.String())
	w.WriteHeader(http.StatusFound)
}

func connectorOrigin(rawURL string) (string, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.User != nil {
		return "", errors.New("invalid connector origin")
	}
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "http" && scheme != "https" {
		return "", errors.New("invalid connector origin")
	}
	hostname := strings.ToLower(parsed.Hostname())
	if hostname == "" {
		return "", errors.New("invalid connector origin")
	}
	port := parsed.Port()
	if (scheme == "https" && port == "443") || (scheme == "http" && port == "80") {
		port = ""
	}
	if port != "" {
		return scheme + "://" + net.JoinHostPort(hostname, port), nil
	}
	if strings.Contains(hostname, ":") {
		hostname = "[" + hostname + "]"
	}
	return scheme + "://" + hostname, nil
}

func (s *Server) connectorAuthorizationCookie(attemptID, value string, expiresAt time.Time) *http.Cookie {
	secure := false
	sameSite := http.SameSiteLaxMode
	if publicURL, err := url.Parse(s.publicURL); err == nil && publicURL.Scheme == "https" {
		secure = true
		sameSite = http.SameSiteNoneMode
	}
	maxAge := int(time.Until(expiresAt).Seconds())
	if maxAge < 1 {
		maxAge = 1
	}
	return &http.Cookie{
		Name:     connectorAuthorizationCookieName(attemptID),
		Value:    value,
		Path:     mcp.CallbackPath,
		Expires:  expiresAt,
		MaxAge:   maxAge,
		HttpOnly: true,
		Secure:   secure,
		SameSite: sameSite,
	}
}

func (s *Server) clearConnectorAuthorizationCookie(w http.ResponseWriter, attemptID string) {
	secure := false
	sameSite := http.SameSiteLaxMode
	if publicURL, err := url.Parse(s.publicURL); err == nil && publicURL.Scheme == "https" {
		secure = true
		sameSite = http.SameSiteNoneMode
	}
	http.SetCookie(w, &http.Cookie{
		Name:     connectorAuthorizationCookieName(attemptID),
		Value:    "",
		Path:     mcp.CallbackPath,
		MaxAge:   -1,
		HttpOnly: true,
		Secure:   secure,
		SameSite: sameSite,
	})
}

func connectorAuthorizationCookieName(attemptID string) string {
	return "va_connector_oauth_" + attemptID
}

func connectorDefinitionOf(connector mcp.Connector) ConnectorDefinition {
	endpoint, _ := connector.Endpoint("")
	authMode := ConnectorDefinitionAuthModeOauthDcr
	switch connector.AuthMode {
	case connectors.AuthNone:
		authMode = ConnectorDefinitionAuthModeNone
	case connectors.AuthBearer:
		authMode = ConnectorDefinitionAuthModeBearer
	case connectors.AuthAPIKey:
		authMode = ConnectorDefinitionAuthModeApiKey
	default:
		switch connector.OAuthMode {
		case "confidential":
			authMode = ConnectorDefinitionAuthModeOauthPreconfigured
		case "customer_confidential", "customer_dcr":
			authMode = ConnectorDefinitionAuthModeOauthCustomerCredentials
		}
	}
	definition := ConnectorDefinition{
		Id:          connector.ID,
		Name:        connector.Name,
		Category:    connector.Category,
		Description: connector.Description,
		Endpoint:    endpoint,
		AuthMode:    authMode,
	}
	if connector.AuthHeader != "" {
		definition.ApiKeyHeader = optional(connector.AuthHeader)
	}
	if connector.InstanceRequired {
		required := true
		definition.InstanceRequired = &required
	}
	definition.InstanceHint = optional(connector.InstanceHint)
	if len(connector.Scopes) > 0 {
		scopes := append([]string(nil), connector.Scopes...)
		definition.Scopes = &scopes
	}
	return definition
}

func (s *Server) connectorDefinition(ctx context.Context, customerID, id string) (mcp.Connector, error) {
	if connector, exists := mcp.Lookup(id); exists {
		return connector, nil
	}
	if s.store == nil {
		return mcp.Connector{}, store.ErrConnectorDefinitionNotFound
	}
	definition, err := s.store.ConnectorDefinition(ctx, customerID, id)
	if err != nil {
		return mcp.Connector{}, err
	}
	return connectorFromDefinition(definition), nil
}

func connectorFromDefinition(definition store.ConnectorDefinition) mcp.Connector {
	authType := definition.AuthType
	oauthMode := ""
	if authType == connectors.AuthOAuth2 {
		authType = "oauth"
		oauthMode = "dcr"
	}
	return mcp.Connector{
		ID:          definition.ID,
		Name:        definition.Name,
		Category:    definition.Category,
		Description: definition.Description,
		URL:         definition.Endpoint,
		AuthMode:    authType,
		AuthHeader:  definition.AuthHeader,
		OAuthMode:   oauthMode,
	}
}

func connectorAuthType(connector mcp.Connector) string {
	if connector.AuthMode == "oauth" {
		return connectors.AuthOAuth2
	}
	return connector.AuthMode
}

func forbiddenAPIKeyHeader(header string) bool {
	switch header {
	case "Authorization", "Cookie", "Host", "Content-Length", "Connection", "Proxy-Authorization", "Proxy-Authenticate", "Transfer-Encoding":
		return true
	default:
		return strings.HasPrefix(header, "Proxy-")
	}
}

func (s *Server) connectorDefinitionComplaint(ctx context.Context, customerID string, bindings *[]AgentConnectorBinding) (string, error) {
	if bindings == nil {
		return "", nil
	}
	seen := make(map[string]struct{})
	for _, binding := range *bindings {
		if _, builtin := mcp.Lookup(binding.ConnectorId); builtin {
			continue
		}
		if _, alreadyChecked := seen[binding.ConnectorId]; alreadyChecked {
			continue
		}
		seen[binding.ConnectorId] = struct{}{}
		if s.store == nil {
			return "no connector named " + binding.ConnectorId, nil
		}
		if _, err := s.store.ConnectorDefinition(ctx, customerID, binding.ConnectorId); err != nil {
			if errors.Is(err, store.ErrConnectorDefinitionNotFound) {
				return "no connector named " + binding.ConnectorId, nil
			}
			return "", err
		}
	}
	return "", nil
}

func connectionOwnerMatches(ctx context.Context, connection store.ConnectorConnection) bool {
	if connection.OwnerType == "app" {
		return true
	}
	return connection.OwnerType == "user" && ServerSideFrom(ctx) && CallerFrom(ctx).UserID != "" &&
		CallerFrom(ctx).UserID == connection.OwnerID
}

func connectorConnectionOf(connection store.ConnectorConnection) ConnectorConnection {
	rendered := ConnectorConnection{
		Id:            connection.ID,
		ConnectorId:   connection.ConnectorID,
		OwnerType:     ConnectorConnectionOwnerType(connection.OwnerType),
		Endpoint:      connection.Endpoint,
		AuthType:      ConnectorConnectionAuthType(connection.AuthType),
		Status:        ConnectorConnectionStatus(connection.Status),
		GrantedScopes: append([]string(nil), connection.GrantedScopes...),
		Revision:      connection.Revision,
	}
	if connection.OwnerID != "" {
		rendered.OwnerId = optional(connection.OwnerID)
	}
	if connection.Instance != "" {
		rendered.Instance = optional(connection.Instance)
	}
	if connection.Label != "" {
		rendered.Label = optional(connection.Label)
	}
	if connection.AccountID != "" {
		rendered.AccountId = optional(connection.AccountID)
	}
	if connection.ExpiresAt != nil {
		rendered.ExpiresAt = connection.ExpiresAt
	}
	if connection.ToolsDigest != "" {
		rendered.ToolsDigest = optional(connection.ToolsDigest)
	}
	if connection.ToolsCheckedAt != nil {
		rendered.ToolsCheckedAt = connection.ToolsCheckedAt
	}
	if connection.LastError != "" {
		rendered.LastError = optional(connection.LastError)
	}
	createdAt, updatedAt := connection.CreatedAt, connection.UpdatedAt
	rendered.CreatedAt = &createdAt
	rendered.UpdatedAt = &updatedAt
	return rendered
}

func connectorToolsOf(tools []store.ConnectorTool) []ConnectorTool {
	listed := make([]ConnectorTool, 0, len(tools))
	for _, tool := range tools {
		listed = append(listed, ConnectorTool{
			Name:         tool.Name,
			Description:  tool.Description,
			InputSchema:  tool.InputSchema,
			SchemaDigest: tool.SchemaDigest,
		})
	}
	return listed
}

func connectorBindingsComplaint(bindings *[]AgentConnectorBinding) (string, bool) {
	if bindings == nil {
		return "", true
	}
	seenAliases := make(map[string]struct{}, len(*bindings))
	for _, binding := range *bindings {
		name := strings.TrimSpace(binding.Name)
		if !connectorAliasPattern.MatchString(name) || strings.Contains(name, "__") {
			return "connector binding names must start with a lowercase letter and use only lowercase letters, numbers, dashes, or underscores", false
		}
		if _, duplicate := seenAliases[name]; duplicate {
			return "connector binding names must be unique within an agent config", false
		}
		seenAliases[name] = struct{}{}
		if _, exists := mcp.Lookup(binding.ConnectorId); !exists && !customConnectorIDPattern.MatchString(binding.ConnectorId) {
			return "no connector named " + binding.ConnectorId, false
		}
		switch binding.Connection.Type {
		case AgentConnectorSelectionTypeFixed:
			if strings.TrimSpace(value(binding.Connection.ConnectionId)) == "" {
				return "a fixed connector binding needs connection_id", false
			}
		case AgentConnectorSelectionTypeSession:
			if binding.Connection.ConnectionId != nil {
				return "a session connector binding selects its connection at session creation", false
			}
		default:
			return "connector connection type must be fixed or session", false
		}
		if binding.TimeoutMs != nil && (*binding.TimeoutMs < 1 || *binding.TimeoutMs > 30000) {
			return "connector timeout_ms must be between 1 and 30000", false
		}
		seenTools := make(map[string]struct{}, len(binding.Tools))
		for _, tool := range binding.Tools {
			toolName := strings.TrimSpace(tool.Name)
			if toolName == "" || !connectorToolDigestPattern.MatchString(tool.SchemaDigest) {
				return "a connector tool grant needs a name and a valid schema_digest", false
			}
			if _, duplicate := seenTools[toolName]; duplicate {
				return "connector tool grants must be unique", false
			}
			seenTools[toolName] = struct{}{}
		}
	}
	return "", true
}

func connectorBindingsFromAPI(bindings []AgentConnectorBinding) []store.ConnectorBinding {
	converted := make([]store.ConnectorBinding, 0, len(bindings))
	for _, binding := range bindings {
		tools := make([]store.ToolGrant, 0, len(binding.Tools))
		for _, tool := range binding.Tools {
			tools = append(tools, store.ToolGrant{Name: strings.TrimSpace(tool.Name), SchemaDigest: tool.SchemaDigest})
		}
		converted = append(converted, store.ConnectorBinding{
			Name:        strings.TrimSpace(binding.Name),
			ConnectorID: binding.ConnectorId,
			Connection: store.ConnectionBinding{
				Type:         string(binding.Connection.Type),
				ConnectionID: strings.TrimSpace(value(binding.Connection.ConnectionId)),
			},
			Tools:     tools,
			Required:  value(binding.Required),
			TimeoutMs: value(binding.TimeoutMs),
		})
	}
	return converted
}

func connectorBindingsToAPI(bindings []store.ConnectorBinding) []AgentConnectorBinding {
	converted := make([]AgentConnectorBinding, 0, len(bindings))
	for _, binding := range bindings {
		tools := make([]ConnectorToolGrant, 0, len(binding.Tools))
		for _, tool := range binding.Tools {
			tools = append(tools, ConnectorToolGrant{Name: tool.Name, SchemaDigest: tool.SchemaDigest})
		}
		required := binding.Required
		var timeout *int
		if binding.TimeoutMs > 0 {
			value := binding.TimeoutMs
			timeout = &value
		}
		selection := AgentConnectorSelection{Type: AgentConnectorSelectionType(binding.Connection.Type)}
		if binding.Connection.ConnectionID != "" {
			connectionID := binding.Connection.ConnectionID
			selection.ConnectionId = &connectionID
		}
		converted = append(converted, AgentConnectorBinding{
			Name:        binding.Name,
			ConnectorId: binding.ConnectorID,
			Connection:  selection,
			Tools:       tools,
			Required:    &required,
			TimeoutMs:   timeout,
		})
	}
	return converted
}

func connectorToolsDigest(tools []store.ConnectorTool) (string, error) {
	data, err := json.Marshal(tools)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

func randomConnectorID() (string, error) {
	raw := make([]byte, 16)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("api: create connector id: %w", err)
	}
	return hex.EncodeToString(raw), nil
}
