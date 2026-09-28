package mcp

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	mcpgoauth "github.com/modelcontextprotocol/go-sdk/auth"
)

var ErrOAuthInvalidGrant = errors.New("mcp: OAuth grant is invalid")
var ErrOAuthRefreshUncertain = errors.New("mcp: OAuth refresh outcome is unknown")
var ErrOAuthClientNotConfigured = errors.New("provider OAuth client credentials are not configured")
var defaultHTTPClient = egress.NewPublicHTTPClient(10 * time.Second)

// OAuthClient is the OAuth 2.1 + PKCE client used to connect a hosted MCP server.
type OAuthClient struct {
	HTTP      *http.Client
	PublicURL string
}

// PendingAuthorization is what authorize has to remember so the callback can finish the login.
type PendingAuthorization struct {
	ConnectorID      string
	State            string
	CodeVerifier     string
	ClientID         string
	ClientSecret     string
	ClientAuthMethod string
	Issuer           string
	RequireIssuer    bool
	TokenEndpoint    string
	RefreshEndpoint  string
	AuthorizeURL     string
	Resource         string
	Scopes           []string
}

// Token is what the callback stores.
type Token struct {
	AccessToken  string
	RefreshToken string
	AccountID    string
	Scopes       []string
	ExpiresAt    *time.Time
}

type protectedResource struct {
	Resource             string   `json:"resource"`
	AuthorizationServers []string `json:"authorization_servers"`
}

type authServer struct {
	Issuer                            string   `json:"issuer"`
	AuthorizationEndpoint             string   `json:"authorization_endpoint"`
	TokenEndpoint                     string   `json:"token_endpoint"`
	JWKSURI                           string   `json:"jwks_uri"`
	ResponseTypesSupported            []string `json:"response_types_supported"`
	RegistrationEndpoint              string   `json:"registration_endpoint"`
	TokenEndpointAuthMethods          []string `json:"token_endpoint_auth_methods_supported"`
	CodeChallengeMethods              []string `json:"code_challenge_methods_supported"`
	ClientIDMetadataSupported         bool     `json:"client_id_metadata_document_supported"`
	AuthorizationResponseIssSupported bool     `json:"authorization_response_iss_parameter_supported"`
}

type registration struct {
	ClientID                string `json:"client_id"`
	ClientSecret            string `json:"client_secret"`
	TokenEndpointAuthMethod string `json:"token_endpoint_auth_method"`
}

// ClientMetadataDocument describes the router as a public OAuth client.
type ClientMetadataDocument struct {
	ClientID                string   `json:"client_id"`
	ClientName              string   `json:"client_name"`
	RedirectURIs            []string `json:"redirect_uris"`
	GrantTypes              []string `json:"grant_types"`
	ResponseTypes           []string `json:"response_types"`
	TokenEndpointAuthMethod string   `json:"token_endpoint_auth_method"`
}

type tokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ID           string `json:"id"`
	Owner        string `json:"owner"`
	Organization string `json:"organization"`
	ExpiresIn    int    `json:"expires_in"`
	Scope        string `json:"scope"`
	Error        string `json:"error"`
	ErrorDesc    string `json:"error_description"`
	OK           bool   `json:"ok"`
	AuthedUser   struct {
		ID           string `json:"id"`
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ExpiresIn    int    `json:"expires_in"`
		Scope        string `json:"scope"`
	} `json:"authed_user"`
	Team struct {
		ID string `json:"id"`
	} `json:"team"`
}

// CallbackPath is the unauthenticated path the provider redirects to.
const CallbackPath = "/v1/agents/connectors/oauth/callback"

// ClientMetadataPath is the public Client ID Metadata Document used by ASs that support CIMD.
const ClientMetadataPath = "/.well-known/oauth-client-metadata"

const maxOAuthResponseBytes = 1 << 20

// StartAuthorize discovers the provider and returns the URL the browser should open.
func (a *OAuthClient) StartAuthorize(ctx context.Context, connector Connector, instance string) (PendingAuthorization, error) {
	return a.StartAuthorizeWithClient(ctx, connector, instance, "", "")
}

// StartAuthorizeWithClient starts a provider-specific OAuth flow. Gong supports both
// administrator-approved dynamic registration and manually supplied client credentials.
// Other providers use the operator-managed credentials or dynamic registration declared by
// the catalog.
func (a *OAuthClient) StartAuthorizeWithClient(ctx context.Context, connector Connector, instance, suppliedID, suppliedSecret string) (PendingAuthorization, error) {
	endpoint, err := connector.Endpoint(instance)
	if err != nil {
		return PendingAuthorization{}, err
	}
	transport := a.client()
	meta := authServer{
		AuthorizationEndpoint: connector.AuthorizationEndpoint,
		TokenEndpoint:         connector.TokenEndpoint,
		RegistrationEndpoint:  connector.RegistrationEndpoint,
	}
	resourceID := connector.Resource
	isDCR := connector.OAuthMode == "dcr" || connector.OAuthMode == "customer_dcr" || connector.OAuthMode == ""
	clientID, clientSecret := suppliedID, suppliedSecret
	if clientID == "" && connector.ClientEnv != "" {
		clientID, clientSecret = envClient(connector.ClientEnv)
	}
	if (connector.OAuthMode == "confidential" || connector.OAuthMode == "customer_confidential") && (clientID == "" || clientSecret == "") {
		return PendingAuthorization{}, ErrOAuthClientNotConfigured
	}
	if connector.OAuthMode == "customer_dcr" && (clientID == "") != (clientSecret == "") {
		return PendingAuthorization{}, fmt.Errorf("mcp: %s needs both customer-registered OAuth credentials or neither", connector.Name)
	}
	meta = providerOAuthEndpoints(connector, instance, meta)
	needsDiscovery := meta.AuthorizationEndpoint == "" || meta.TokenEndpoint == "" || isDCR && clientID == ""
	issuer := connector.Issuer
	if needsDiscovery {
		if issuer == "" {
			resource, err := a.discoverResource(ctx, transport, endpoint)
			if err != nil {
				return PendingAuthorization{}, err
			}
			if resource.Resource != "" {
				resourceID = resource.Resource
			}
			issuer = first(resource.AuthorizationServers)
		}
		if issuer == "" {
			issuer = originOf(endpoint)
		}
		discovered, err := a.discoverServer(ctx, transport, issuer)
		if err != nil {
			return PendingAuthorization{}, err
		}
		meta = mergeAuthServer(meta, discovered)
	}
	if meta.AuthorizationEndpoint == "" || meta.TokenEndpoint == "" {
		return PendingAuthorization{}, fmt.Errorf("mcp: %s did not advertise oauth endpoints", connector.ID)
	}
	if len(meta.CodeChallengeMethods) > 0 && !slicesContains(meta.CodeChallengeMethods, "S256") {
		return PendingAuthorization{}, fmt.Errorf("mcp: %s does not support OAuth PKCE with S256", connector.Name)
	}
	if resourceID == "" && isDCR {
		resourceID = endpoint
	}
	if resourceID != "" {
		if err := validateResourceIdentity(endpoint, resourceID); err != nil {
			return PendingAuthorization{}, fmt.Errorf("mcp: %s has an invalid oauth resource identity", connector.ID)
		}
	}
	for _, endpoint := range []string{meta.AuthorizationEndpoint, meta.TokenEndpoint, meta.RegistrationEndpoint} {
		if endpoint == "" {
			continue
		}
		if err := a.validateOAuthTarget(ctx, endpoint); err != nil {
			return PendingAuthorization{}, fmt.Errorf("mcp: %s has an unsafe oauth endpoint", connector.ID)
		}
	}

	registeredAuthMethod := ""
	if clientID == "" && isDCR && meta.ClientIDMetadataSupported {
		clientID = a.clientMetadataURL()
		registeredAuthMethod = "none"
	} else if clientID == "" && isDCR && meta.RegistrationEndpoint != "" {
		requestedAuthMethod := dcrTokenAuthMethod(meta.TokenEndpointAuthMethods)
		if requestedAuthMethod == "" {
			return PendingAuthorization{}, fmt.Errorf("mcp: %s advertises no supported OAuth client authentication method", connector.Name)
		}
		registered, err := a.register(ctx, transport, meta.RegistrationEndpoint, requestedAuthMethod)
		if err != nil {
			return PendingAuthorization{}, err
		}
		clientID = registered.ClientID
		clientSecret = registered.ClientSecret
		registeredAuthMethod = registered.TokenEndpointAuthMethod
		if registeredAuthMethod == "" {
			registeredAuthMethod = requestedAuthMethod
		}
		if !supportedDCRTokenAuthMethod(registeredAuthMethod) ||
			len(meta.TokenEndpointAuthMethods) > 0 && !slicesContains(meta.TokenEndpointAuthMethods, registeredAuthMethod) {
			return PendingAuthorization{}, fmt.Errorf("mcp: %s registered an unsupported OAuth client authentication method", connector.Name)
		}
		if registeredAuthMethod != "none" && clientSecret == "" {
			return PendingAuthorization{}, fmt.Errorf("mcp: %s registration omitted its OAuth client secret", connector.Name)
		}
	}
	if clientID == "" {
		return PendingAuthorization{}, ErrOAuthClientNotConfigured
	}
	if meta.Issuer != "" {
		issuer = meta.Issuer
	} else if issuer == "" {
		issuer = connector.Issuer
	}
	if issuer == "" {
		issuer = originOf(meta.AuthorizationEndpoint)
	}

	verifier, challenge, err := pkce()
	if err != nil {
		return PendingAuthorization{}, err
	}
	state, err := randomHex(16)
	if err != nil {
		return PendingAuthorization{}, err
	}

	query := url.Values{}
	query.Set("response_type", "code")
	query.Set("client_id", clientID)
	query.Set("redirect_uri", a.callbackURL())
	query.Set("state", state)
	query.Set("code_challenge", challenge)
	query.Set("code_challenge_method", "S256")
	if resourceID != "" {
		query.Set("resource", resourceID)
	}
	if len(connector.Scopes) > 0 {
		query.Set("scope", strings.Join(connector.Scopes, " "))
	}
	if connector.ID == "slack" && len(connector.Scopes) > 0 {
		query.Set("scope", strings.Join(connector.Scopes, ","))
	}
	authorize, err := url.Parse(meta.AuthorizationEndpoint)
	if err != nil {
		return PendingAuthorization{}, fmt.Errorf("mcp: invalid authorization endpoint: %w", err)
	}
	authorizeQuery := authorize.Query()
	for key, values := range query {
		for _, value := range values {
			authorizeQuery.Add(key, value)
		}
	}
	authorize.RawQuery = authorizeQuery.Encode()
	clientAuthMethod := registeredAuthMethod
	if clientAuthMethod == "" && clientSecret != "" {
		if connector.TokenEndpointAuthMethod != "" {
			if len(meta.TokenEndpointAuthMethods) > 0 && !slicesContains(meta.TokenEndpointAuthMethods, connector.TokenEndpointAuthMethod) {
				return PendingAuthorization{}, fmt.Errorf("mcp: %s does not support the configured OAuth client credentials", connector.Name)
			}
			clientAuthMethod = connector.TokenEndpointAuthMethod
		}
		if clientAuthMethod == "" {
			if len(meta.TokenEndpointAuthMethods) > 0 &&
				!slicesContains(meta.TokenEndpointAuthMethods, "client_secret_basic") &&
				!slicesContains(meta.TokenEndpointAuthMethods, "client_secret_post") {
				return PendingAuthorization{}, fmt.Errorf("mcp: %s does not support the configured OAuth client credentials", connector.Name)
			}
			clientAuthMethod = "client_secret_post"
			if slicesContains(meta.TokenEndpointAuthMethods, "client_secret_basic") {
				clientAuthMethod = "client_secret_basic"
			}
		}
	} else if clientAuthMethod == "" {
		clientAuthMethod = "none"
	}
	refreshEndpoint := connector.RefreshEndpoint
	if refreshEndpoint == "" {
		refreshEndpoint = meta.TokenEndpoint
	}

	return PendingAuthorization{
		ConnectorID:      connector.ID,
		State:            state,
		CodeVerifier:     verifier,
		ClientID:         clientID,
		ClientSecret:     clientSecret,
		ClientAuthMethod: clientAuthMethod,
		Issuer:           issuer,
		RequireIssuer:    meta.AuthorizationResponseIssSupported,
		TokenEndpoint:    meta.TokenEndpoint,
		RefreshEndpoint:  refreshEndpoint,
		AuthorizeURL:     authorize.String(),
		Resource:         resourceID,
		Scopes:           append([]string(nil), connector.Scopes...),
	}, nil
}

// OAuthClientForImport resolves trusted OAuth endpoints and client authentication for a
// provider-issued grant. It never accepts token or resource endpoints from the caller.
func (a *OAuthClient) OAuthClientForImport(ctx context.Context, connector Connector, instance, suppliedID, suppliedSecret string) (PendingAuthorization, error) {
	endpoint, err := connector.Endpoint(instance)
	if err != nil {
		return PendingAuthorization{}, err
	}
	transport := a.client()
	meta := authServer{
		AuthorizationEndpoint: connector.AuthorizationEndpoint,
		TokenEndpoint:         connector.TokenEndpoint,
		RegistrationEndpoint:  connector.RegistrationEndpoint,
	}
	resourceID := connector.Resource
	isDCR := connector.OAuthMode == "dcr" || connector.OAuthMode == "customer_dcr"
	clientID, clientSecret := suppliedID, suppliedSecret
	if connector.ClientEnv != "" {
		if suppliedID != "" || suppliedSecret != "" {
			return PendingAuthorization{}, fmt.Errorf("mcp: %s uses an operator-managed OAuth client", connector.Name)
		}
		clientID, clientSecret = envClient(connector.ClientEnv)
	}
	meta = providerOAuthEndpoints(connector, instance, meta)
	issuer := connector.Issuer
	if meta.AuthorizationEndpoint == "" || meta.TokenEndpoint == "" {
		if issuer == "" {
			resource, err := a.discoverResource(ctx, transport, endpoint)
			if err != nil {
				return PendingAuthorization{}, err
			}
			if resource.Resource != "" {
				resourceID = resource.Resource
			}
			issuer = first(resource.AuthorizationServers)
		}
		if issuer == "" {
			issuer = originOf(endpoint)
		}
		discovered, err := a.discoverServer(ctx, transport, issuer)
		if err != nil {
			return PendingAuthorization{}, err
		}
		meta = mergeAuthServer(meta, discovered)
	}
	if meta.AuthorizationEndpoint == "" || meta.TokenEndpoint == "" {
		return PendingAuthorization{}, fmt.Errorf("mcp: %s did not advertise oauth endpoints", connector.ID)
	}
	if resourceID == "" && isDCR {
		resourceID = endpoint
	}
	if resourceID != "" {
		if err := validateResourceIdentity(endpoint, resourceID); err != nil {
			return PendingAuthorization{}, fmt.Errorf("mcp: %s has an invalid oauth resource identity", connector.ID)
		}
	}
	for _, target := range []string{meta.AuthorizationEndpoint, meta.TokenEndpoint, meta.RegistrationEndpoint} {
		if target != "" {
			if err := a.validateOAuthTarget(ctx, target); err != nil {
				return PendingAuthorization{}, fmt.Errorf("mcp: %s has an unsafe oauth endpoint", connector.ID)
			}
		}
	}
	if clientID == "" {
		return PendingAuthorization{}, fmt.Errorf("mcp: %s needs an OAuth client ID to import a grant", connector.Name)
	}
	if (connector.OAuthMode == "customer_confidential" || connector.OAuthMode == "customer_dcr") &&
		(clientID == "" || clientSecret == "") {
		return PendingAuthorization{}, fmt.Errorf("mcp: %s needs both customer-registered OAuth credentials to import a grant", connector.Name)
	}
	clientAuthMethod := connector.TokenEndpointAuthMethod
	if clientAuthMethod == "" {
		switch {
		case clientSecret != "" && slicesContains(meta.TokenEndpointAuthMethods, "client_secret_basic"):
			clientAuthMethod = "client_secret_basic"
		case clientSecret != "" && slicesContains(meta.TokenEndpointAuthMethods, "client_secret_post"):
			clientAuthMethod = "client_secret_post"
		case clientSecret != "" && len(meta.TokenEndpointAuthMethods) == 0 && connector.OAuthMode == "confidential":
			clientAuthMethod = "client_secret_post"
		case clientSecret == "" && (len(meta.TokenEndpointAuthMethods) == 0 || slicesContains(meta.TokenEndpointAuthMethods, "none")):
			clientAuthMethod = "none"
		default:
			return PendingAuthorization{}, fmt.Errorf("mcp: %s does not support the supplied OAuth client credentials", connector.Name)
		}
	}
	if !supportedDCRTokenAuthMethod(clientAuthMethod) ||
		(len(meta.TokenEndpointAuthMethods) > 0 && !slicesContains(meta.TokenEndpointAuthMethods, clientAuthMethod)) ||
		(clientAuthMethod == "none" && clientSecret != "") ||
		(clientAuthMethod != "none" && clientSecret == "") {
		return PendingAuthorization{}, fmt.Errorf("mcp: %s does not support the supplied OAuth client credentials", connector.Name)
	}
	refreshEndpoint := connector.RefreshEndpoint
	if refreshEndpoint == "" {
		refreshEndpoint = meta.TokenEndpoint
	}
	if err := a.validateOAuthTarget(ctx, refreshEndpoint); err != nil {
		return PendingAuthorization{}, fmt.Errorf("mcp: %s has an unsafe refresh endpoint", connector.ID)
	}
	if meta.Issuer != "" {
		issuer = meta.Issuer
	} else if issuer == "" {
		issuer = connector.Issuer
	}
	if issuer == "" {
		issuer = originOf(meta.AuthorizationEndpoint)
	}
	return PendingAuthorization{
		ConnectorID:      connector.ID,
		ClientID:         clientID,
		ClientSecret:     clientSecret,
		ClientAuthMethod: clientAuthMethod,
		Issuer:           issuer,
		RequireIssuer:    meta.AuthorizationResponseIssSupported,
		TokenEndpoint:    meta.TokenEndpoint,
		RefreshEndpoint:  refreshEndpoint,
		Resource:         resourceID,
		Scopes:           append([]string(nil), connector.Scopes...),
	}, nil
}

func validateResourceIdentity(endpoint, resource string) error {
	endpointURL, err := url.Parse(endpoint)
	if err != nil {
		return err
	}
	resourceURL, err := url.Parse(resource)
	if err != nil || resourceURL.Scheme != endpointURL.Scheme || resourceURL.Host == "" || resourceURL.User != nil ||
		resourceURL.RawQuery != "" || resourceURL.ForceQuery || resourceURL.Fragment != "" ||
		!strings.EqualFold(endpointURL.Host, resourceURL.Host) {
		return fmt.Errorf("mcp: oauth resource does not identify the configured MCP host")
	}
	return nil
}

// Exchange finishes the login with the code the provider sent back.
func (a *OAuthClient) Exchange(ctx context.Context, pending PendingAuthorization, code string) (Token, error) {
	return a.ExchangeWithIssuer(ctx, pending, code, "")
}

// ExchangeWithIssuer validates the authorization-server issuer before redeeming a code.
func (a *OAuthClient) ExchangeWithIssuer(ctx context.Context, pending PendingAuthorization, code, responseIssuer string) (Token, error) {
	if err := pending.ValidateAuthorizationResponseIssuer(responseIssuer); err != nil {
		return Token{}, err
	}
	if err := a.validateOAuthTarget(ctx, pending.TokenEndpoint); err != nil {
		return Token{}, fmt.Errorf("mcp: unsafe token endpoint")
	}
	form := url.Values{}
	form.Set("grant_type", "authorization_code")
	form.Set("code", code)
	form.Set("redirect_uri", a.callbackURL())
	form.Set("code_verifier", pending.CodeVerifier)
	if pending.Resource != "" {
		form.Set("resource", pending.Resource)
	}
	request, err := a.tokenRequest(ctx, pending, form)
	if err != nil {
		return Token{}, err
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.Header.Set("Accept", "application/json")

	response, err := a.tokenClient().Do(request)
	if err != nil {
		return Token{}, fmt.Errorf("mcp: token: %w", err)
	}
	defer response.Body.Close()
	raw, err := readOAuthResponse(response.Body)
	if err != nil {
		return Token{}, err
	}
	var body tokenResponse
	if err := json.Unmarshal(raw, &body); err != nil {
		return Token{}, fmt.Errorf("mcp: token: %w", err)
	}
	if body.Error != "" {
		return Token{}, fmt.Errorf("mcp: token endpoint rejected exchange (%s)", body.Error)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return Token{}, fmt.Errorf("mcp: token endpoint returned HTTP %d", response.StatusCode)
	}
	normalizeProviderToken(&body)
	if body.AccessToken == "" {
		return Token{}, fmt.Errorf("mcp: token: no access token")
	}
	token := Token{
		AccessToken:  body.AccessToken,
		RefreshToken: body.RefreshToken,
		AccountID:    providerAccountID(pending.ConnectorID, pending.TokenEndpoint, body),
		Scopes:       parseScopes(body.Scope),
	}
	if body.ExpiresIn > 0 {
		at := time.Now().UTC().Add(time.Duration(body.ExpiresIn) * time.Second)
		token.ExpiresAt = &at
	}
	return token, nil
}

// ValidateAuthorizationResponseIssuer enforces RFC 9207 when the provider advertises it
// and rejects any supplied issuer that does not match the discovered or configured issuer.
func (p PendingAuthorization) ValidateAuthorizationResponseIssuer(responseIssuer string) error {
	if p.RequireIssuer && responseIssuer == "" {
		return fmt.Errorf("mcp: authorization response omitted its issuer")
	}
	if responseIssuer != "" && (p.Issuer == "" || responseIssuer != p.Issuer) {
		return fmt.Errorf("mcp: authorization response issuer does not match the configured issuer")
	}
	return nil
}

// Refresh renews an access token. Empty refresh token is a no-op miss.
func (a *OAuthClient) Refresh(ctx context.Context, tokenEndpoint, clientID, refreshToken string) (Token, error) {
	return a.RefreshWithClient(ctx, PendingAuthorization{TokenEndpoint: tokenEndpoint, ClientID: clientID}, refreshToken)
}

// RefreshWithClient renews an OAuth token using the same client authentication method as
// the original authorization grant.
func (a *OAuthClient) RefreshWithClient(ctx context.Context, pending PendingAuthorization, refreshToken string) (Token, error) {
	if refreshToken == "" || (pending.TokenEndpoint == "" && pending.RefreshEndpoint == "") {
		return Token{}, fmt.Errorf("mcp: nothing to refresh")
	}
	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("refresh_token", refreshToken)
	if pending.Resource != "" {
		form.Set("resource", pending.Resource)
	}
	if pending.RefreshEndpoint != "" {
		pending.TokenEndpoint = pending.RefreshEndpoint
	}
	if err := a.validateOAuthTarget(ctx, pending.TokenEndpoint); err != nil {
		return Token{}, fmt.Errorf("mcp: unsafe token endpoint")
	}
	request, err := a.tokenRequest(ctx, pending, form)
	if err != nil {
		return Token{}, err
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response, err := a.tokenClient().Do(request)
	if err != nil {
		return Token{}, ErrOAuthRefreshUncertain
	}
	defer response.Body.Close()
	raw, err := readOAuthResponse(response.Body)
	if err != nil {
		return Token{}, ErrOAuthRefreshUncertain
	}
	var body tokenResponse
	if err := json.Unmarshal(raw, &body); err != nil {
		return Token{}, ErrOAuthRefreshUncertain
	}
	if body.Error == "invalid_grant" || body.Error == "invalid_refresh_token" {
		return Token{}, ErrOAuthInvalidGrant
	}
	if body.Error == "internal_error" || body.Error == "fatal_error" {
		return Token{}, ErrOAuthRefreshUncertain
	}
	if body.Error != "" {
		return Token{}, fmt.Errorf("mcp: token endpoint rejected refresh (%s)", body.Error)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return Token{}, fmt.Errorf("%w (HTTP %d)", ErrOAuthRefreshUncertain, response.StatusCode)
	}
	normalizeProviderToken(&body)
	if body.AccessToken == "" {
		return Token{}, ErrOAuthRefreshUncertain
	}
	token := Token{AccessToken: body.AccessToken, RefreshToken: or(body.RefreshToken, refreshToken), Scopes: parseScopes(body.Scope)}
	if body.ExpiresIn > 0 {
		at := time.Now().UTC().Add(time.Duration(body.ExpiresIn) * time.Second)
		token.ExpiresAt = &at
	}
	return token, nil
}

func (a *OAuthClient) tokenRequest(ctx context.Context, pending PendingAuthorization, form url.Values) (*http.Request, error) {
	form.Set("client_id", pending.ClientID)
	if pending.ClientSecret != "" {
		switch pending.ClientAuthMethod {
		case "client_secret_basic":
			form.Del("client_id")
			request, err := http.NewRequestWithContext(ctx, http.MethodPost, pending.TokenEndpoint, strings.NewReader(form.Encode()))
			if err != nil {
				return nil, err
			}
			request.SetBasicAuth(pending.ClientID, pending.ClientSecret)
			request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
			request.Header.Set("Accept", "application/json")
			return request, nil
		case "client_secret_post", "":
			form.Set("client_secret", pending.ClientSecret)
		default:
			return nil, fmt.Errorf("mcp: unsupported token endpoint auth method %q", pending.ClientAuthMethod)
		}
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, pending.TokenEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.Header.Set("Accept", "application/json")
	return request, nil
}

func normalizeProviderToken(body *tokenResponse) {
	if body.AccessToken == "" && body.AuthedUser.AccessToken != "" {
		body.AccessToken = body.AuthedUser.AccessToken
		body.RefreshToken = body.AuthedUser.RefreshToken
		body.ExpiresIn = body.AuthedUser.ExpiresIn
		body.Scope = body.AuthedUser.Scope
	}
}

func parseScopes(raw string) []string {
	return strings.FieldsFunc(raw, func(value rune) bool {
		return value == ',' || value == ' ' || value == '\t' || value == '\n' || value == '\r'
	})
}

func (a *OAuthClient) callbackURL() string {
	base := strings.TrimRight(a.PublicURL, "/")
	if base == "" {
		base = "http://localhost:8080"
	}
	return base + CallbackPath
}

func (a *OAuthClient) clientMetadataURL() string {
	base := strings.TrimRight(a.PublicURL, "/")
	if base == "" {
		base = "http://localhost:8080"
	}
	return base + ClientMetadataPath
}

// ClientMetadataDocument is fetched by authorization servers that support CIMD.
func (a *OAuthClient) ClientMetadataDocument() ClientMetadataDocument {
	return ClientMetadataDocument{
		ClientID:                a.clientMetadataURL(),
		ClientName:              "Vision Agents",
		RedirectURIs:            []string{a.callbackURL()},
		GrantTypes:              []string{"authorization_code", "refresh_token"},
		ResponseTypes:           []string{"code"},
		TokenEndpointAuthMethod: "none",
	}
}

func (a *OAuthClient) client() *http.Client {
	if a != nil && a.HTTP != nil {
		return a.HTTP
	}
	return defaultHTTPClient
}

func (a *OAuthClient) validateOAuthTarget(ctx context.Context, raw string) error {
	if a != nil && a.HTTP != nil {
		parsed, err := url.Parse(raw)
		if err == nil && parsed.Scheme == "http" && parsed.User == nil && parsed.RawQuery == "" && parsed.Fragment == "" {
			host := parsed.Hostname()
			if host == "localhost" || strings.HasSuffix(host, ".localhost") {
				return nil
			}
			if ip, parseErr := netip.ParseAddr(host); parseErr == nil && ip.IsLoopback() {
				return nil
			}
		}
	}
	return egress.ValidatePublicHTTPSURL(ctx, raw)
}

func (a *OAuthClient) tokenClient() *http.Client {
	client := *a.client()
	client.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}
	return &client
}

func (a *OAuthClient) discoverResource(ctx context.Context, transport *http.Client, endpoint string) (protectedResource, error) {
	var meta protectedResource
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return meta, nil
	}
	wellKnown := parsed.Scheme + "://" + parsed.Host + "/.well-known/oauth-protected-resource" + parsed.Path
	if err := getJSON(ctx, transport, wellKnown, &meta); err != nil {
		// A server that has not published metadata yet can still do DCR against its own
		// origin, so a miss here is not fatal.
		return meta, nil
	}
	return meta, nil
}

func (a *OAuthClient) discoverServer(ctx context.Context, transport *http.Client, issuer string) (authServer, error) {
	if err := a.validateOAuthTarget(ctx, issuer); err != nil {
		return authServer{}, fmt.Errorf("mcp: invalid oauth issuer %q", issuer)
	}
	discovered, err := mcpgoauth.GetAuthServerMetadata(ctx, issuer, transport)
	if err != nil {
		return authServer{}, fmt.Errorf("mcp: oauth discovery: %w", err)
	}
	if discovered == nil {
		return authServer{}, nil
	}
	return authServer{
		Issuer:                            discovered.Issuer,
		AuthorizationEndpoint:             discovered.AuthorizationEndpoint,
		TokenEndpoint:                     discovered.TokenEndpoint,
		RegistrationEndpoint:              discovered.RegistrationEndpoint,
		TokenEndpointAuthMethods:          discovered.TokenEndpointAuthMethodsSupported,
		CodeChallengeMethods:              discovered.CodeChallengeMethodsSupported,
		ClientIDMetadataSupported:         discovered.ClientIDMetadataDocumentSupported,
		AuthorizationResponseIssSupported: discovered.AuthorizationResponseIssParameterSupported,
	}, nil
}

func (a *OAuthClient) register(ctx context.Context, transport *http.Client, endpoint, tokenAuthMethod string) (registration, error) {
	payload, err := json.Marshal(map[string]any{
		"client_name":                "Vision Agents",
		"application_type":           "web",
		"redirect_uris":              []string{a.callbackURL()},
		"grant_types":                []string{"authorization_code", "refresh_token"},
		"response_types":             []string{"code"},
		"token_endpoint_auth_method": tokenAuthMethod,
	})
	if err != nil {
		return registration{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(string(payload)))
	if err != nil {
		return registration{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := transport.Do(request)
	if err != nil {
		return registration{}, fmt.Errorf("mcp: register: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode >= 300 {
		return registration{}, fmt.Errorf("mcp: register: HTTP %d", response.StatusCode)
	}
	raw, err := readOAuthResponse(response.Body)
	if err != nil {
		return registration{}, err
	}
	var body registration
	if err := json.Unmarshal(raw, &body); err != nil {
		return registration{}, err
	}
	if body.ClientID == "" {
		return registration{}, fmt.Errorf("mcp: register: no client id")
	}
	return body, nil
}

func dcrTokenAuthMethod(supported []string) string {
	if len(supported) == 0 || slicesContains(supported, "none") {
		return "none"
	}
	if slicesContains(supported, "client_secret_basic") {
		return "client_secret_basic"
	}
	if slicesContains(supported, "client_secret_post") {
		return "client_secret_post"
	}
	return ""
}

func supportedDCRTokenAuthMethod(method string) bool {
	return method == "none" || method == "client_secret_basic" || method == "client_secret_post"
}

func envClient(connectorID string) (id, secret string) {
	prefix := strings.ToUpper(connectorID) + "_MCP_"
	return os.Getenv(prefix + "CLIENT_ID"), os.Getenv(prefix + "CLIENT_SECRET")
}

func mergeAuthServer(preferred, discovered authServer) authServer {
	if preferred.Issuer == "" {
		preferred.Issuer = discovered.Issuer
	}
	if preferred.AuthorizationEndpoint == "" {
		preferred.AuthorizationEndpoint = discovered.AuthorizationEndpoint
	}
	if preferred.TokenEndpoint == "" {
		preferred.TokenEndpoint = discovered.TokenEndpoint
	}
	if preferred.RegistrationEndpoint == "" {
		preferred.RegistrationEndpoint = discovered.RegistrationEndpoint
	}
	if len(preferred.TokenEndpointAuthMethods) == 0 {
		preferred.TokenEndpointAuthMethods = discovered.TokenEndpointAuthMethods
	}
	if len(preferred.CodeChallengeMethods) == 0 {
		preferred.CodeChallengeMethods = discovered.CodeChallengeMethods
	}
	preferred.ClientIDMetadataSupported = preferred.ClientIDMetadataSupported || discovered.ClientIDMetadataSupported
	preferred.AuthorizationResponseIssSupported = preferred.AuthorizationResponseIssSupported || discovered.AuthorizationResponseIssSupported
	return preferred
}

func providerOAuthEndpoints(connector Connector, instance string, meta authServer) authServer {
	if connector.ID == "salesforce" && (strings.EqualFold(strings.TrimSpace(instance), "sandbox") ||
		strings.EqualFold(strings.TrimSpace(instance), "test")) {
		meta.AuthorizationEndpoint = "https://test.salesforce.com/services/oauth2/authorize"
		meta.TokenEndpoint = "https://test.salesforce.com/services/oauth2/token"
	}
	return meta
}

func providerAccountID(connectorID, tokenEndpoint string, body tokenResponse) string {
	switch connectorID {
	case "slack":
		if body.Team.ID != "" && body.AuthedUser.ID != "" {
			return "slack:" + body.Team.ID + ":" + body.AuthedUser.ID
		}
	case "calendly":
		return calendlyAccountID(body.Owner, body.Organization)
	case "salesforce":
		return salesforceAccountID(tokenEndpoint, body.ID)
	}
	return ""
}

func calendlyAccountID(owner, organization string) string {
	ownerID := calendlyResourceID(owner, "users")
	if ownerID == "" {
		return ""
	}
	organizationID := calendlyResourceID(organization, "organizations")
	if organization != "" && organizationID == "" {
		return ""
	}
	if organizationID == "" {
		return "calendly:" + ownerID
	}
	return "calendly:" + ownerID + ":" + organizationID
}

func calendlyResourceID(raw, resourceType string) string {
	parsed, err := url.Parse(raw)
	if err != nil || !strings.EqualFold(parsed.Scheme, "https") || !strings.EqualFold(parsed.Host, "api.calendly.com") ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return ""
	}
	parts := strings.Split(strings.Trim(parsed.Path, "/"), "/")
	if len(parts) != 2 || parts[0] != resourceType || parts[1] == "" {
		return ""
	}
	return parts[1]
}

func salesforceAccountID(tokenEndpoint, identityURL string) string {
	endpoint, endpointErr := url.Parse(tokenEndpoint)
	identity, identityErr := url.Parse(identityURL)
	if endpointErr != nil || identityErr != nil || !strings.EqualFold(endpoint.Scheme, "https") ||
		!strings.EqualFold(identity.Scheme, "https") || !strings.EqualFold(endpoint.Host, identity.Host) ||
		identity.User != nil || identity.RawQuery != "" || identity.Fragment != "" {
		return ""
	}
	parts := strings.Split(strings.Trim(identity.Path, "/"), "/")
	if len(parts) != 3 || parts[0] != "id" || parts[1] == "" || parts[2] == "" {
		return ""
	}
	return "salesforce:" + parts[1] + ":" + parts[2]
}

func slicesContains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func getJSON(ctx context.Context, transport *http.Client, url string, target any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	request.Header.Set("Accept", "application/json")
	response, err := transport.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode >= 300 {
		return fmt.Errorf("%s: %s", url, response.Status)
	}
	raw, err := readOAuthResponse(response.Body)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, target)
}

func readOAuthResponse(body io.Reader) ([]byte, error) {
	raw, err := io.ReadAll(io.LimitReader(body, maxOAuthResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("mcp: read OAuth response: %w", err)
	}
	if len(raw) > maxOAuthResponseBytes {
		return nil, fmt.Errorf("mcp: OAuth response exceeded size limit")
	}
	return raw, nil
}

func pkce() (verifier, challenge string, err error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", "", err
	}
	verifier = base64.RawURLEncoding.EncodeToString(raw)
	sum := sha256.Sum256([]byte(verifier))
	return verifier, base64.RawURLEncoding.EncodeToString(sum[:]), nil
}

func randomHex(n int) (string, error) {
	raw := make([]byte, n)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	return hex.EncodeToString(raw), nil
}

func originOf(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return raw
	}
	return parsed.Scheme + "://" + parsed.Host
}

func first(values []string) string {
	if len(values) == 0 {
		return ""
	}
	return values[0]
}
