package plugins

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// Auth is the OAuth 2.1 + PKCE client used to connect a hosted MCP server.
type Auth struct {
	HTTP         *http.Client
	PublicURL    string
	DashboardURL string
}

// Pending is what authorize has to remember so the callback can finish the login.
type Pending struct {
	State         string
	CodeVerifier  string
	ClientID      string
	TokenEndpoint string
	AuthorizeURL  string
}

// Token is what the callback stores.
type Token struct {
	AccessToken  string
	RefreshToken string
	ExpiresAt    *time.Time
}

type protectedResource struct {
	AuthorizationServers []string `json:"authorization_servers"`
}

type authServer struct {
	AuthorizationEndpoint string `json:"authorization_endpoint"`
	TokenEndpoint         string `json:"token_endpoint"`
	RegistrationEndpoint  string `json:"registration_endpoint"`
}

type registration struct {
	ClientID     string `json:"client_id"`
	ClientSecret string `json:"client_secret"`
}

type tokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
	Error        string `json:"error"`
	ErrorDesc    string `json:"error_description"`
}

// CallbackPath is the unauthenticated path the provider redirects to.
const CallbackPath = "/v1/agents/plugins/callback"

// StartAuthorize discovers the provider and returns the URL the browser should open.
func (a *Auth) StartAuthorize(ctx context.Context, plugin Plugin, instance string) (Pending, error) {
	endpoint, err := plugin.Endpoint(instance)
	if err != nil {
		return Pending{}, err
	}
	transport := a.client()
	resource, err := a.discoverResource(ctx, transport, endpoint)
	if err != nil {
		return Pending{}, err
	}
	issuer := first(resource.AuthorizationServers)
	if issuer == "" {
		issuer = originOf(endpoint)
	}
	meta, err := a.discoverServer(ctx, transport, issuer)
	if err != nil {
		return Pending{}, err
	}
	if meta.AuthorizationEndpoint == "" || meta.TokenEndpoint == "" {
		return Pending{}, fmt.Errorf("plugins: %s did not advertise oauth endpoints", plugin.ID)
	}

	clientID, clientSecret := envClient(plugin.ID)
	if clientID == "" && meta.RegistrationEndpoint != "" {
		registered, err := a.register(ctx, transport, meta.RegistrationEndpoint)
		if err != nil {
			return Pending{}, err
		}
		clientID = registered.ClientID
		clientSecret = registered.ClientSecret
	}
	if clientID == "" {
		return Pending{}, fmt.Errorf(
			"plugins: %s needs %s_MCP_CLIENT_ID (it does not advertise dynamic registration)",
			plugin.Name, strings.ToUpper(plugin.ID),
		)
	}
	_ = clientSecret

	verifier, challenge, err := pkce()
	if err != nil {
		return Pending{}, err
	}
	state, err := randomHex(16)
	if err != nil {
		return Pending{}, err
	}

	query := url.Values{}
	query.Set("response_type", "code")
	query.Set("client_id", clientID)
	query.Set("redirect_uri", a.callbackURL())
	query.Set("state", state)
	query.Set("code_challenge", challenge)
	query.Set("code_challenge_method", "S256")
	query.Set("resource", endpoint)

	return Pending{
		State:         state,
		CodeVerifier:  verifier,
		ClientID:      clientID,
		TokenEndpoint: meta.TokenEndpoint,
		AuthorizeURL:  meta.AuthorizationEndpoint + "?" + query.Encode(),
	}, nil
}

// Exchange finishes the login with the code the provider sent back.
func (a *Auth) Exchange(ctx context.Context, pending Pending, code string) (Token, error) {
	form := url.Values{}
	form.Set("grant_type", "authorization_code")
	form.Set("code", code)
	form.Set("redirect_uri", a.callbackURL())
	form.Set("client_id", pending.ClientID)
	form.Set("code_verifier", pending.CodeVerifier)

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, pending.TokenEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return Token{}, err
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.Header.Set("Accept", "application/json")

	response, err := a.client().Do(request)
	if err != nil {
		return Token{}, fmt.Errorf("plugins: token: %w", err)
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return Token{}, err
	}
	var body tokenResponse
	if err := json.Unmarshal(raw, &body); err != nil {
		return Token{}, fmt.Errorf("plugins: token: %w", err)
	}
	if body.Error != "" {
		return Token{}, fmt.Errorf("plugins: token: %s", or(body.ErrorDesc, body.Error))
	}
	if body.AccessToken == "" {
		return Token{}, fmt.Errorf("plugins: token: no access token")
	}
	token := Token{AccessToken: body.AccessToken, RefreshToken: body.RefreshToken}
	if body.ExpiresIn > 0 {
		at := time.Now().UTC().Add(time.Duration(body.ExpiresIn) * time.Second)
		token.ExpiresAt = &at
	}
	return token, nil
}

// Refresh renews an access token. Empty refresh token is a no-op miss.
func (a *Auth) Refresh(ctx context.Context, tokenEndpoint, clientID, refreshToken string) (Token, error) {
	if refreshToken == "" || tokenEndpoint == "" {
		return Token{}, fmt.Errorf("plugins: nothing to refresh")
	}
	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("refresh_token", refreshToken)
	form.Set("client_id", clientID)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return Token{}, err
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response, err := a.client().Do(request)
	if err != nil {
		return Token{}, err
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return Token{}, err
	}
	var body tokenResponse
	if err := json.Unmarshal(raw, &body); err != nil {
		return Token{}, err
	}
	if body.AccessToken == "" {
		return Token{}, fmt.Errorf("plugins: refresh: %s", or(body.ErrorDesc, "no access token"))
	}
	token := Token{AccessToken: body.AccessToken, RefreshToken: or(body.RefreshToken, refreshToken)}
	if body.ExpiresIn > 0 {
		at := time.Now().UTC().Add(time.Duration(body.ExpiresIn) * time.Second)
		token.ExpiresAt = &at
	}
	return token, nil
}

// DashboardRedirect is where the browser should land after the callback.
func (a *Auth) DashboardRedirect(configID string) string {
	base := strings.TrimRight(a.DashboardURL, "/")
	if base == "" {
		base = "http://localhost:3000"
	}
	return base + "/agents/" + configID
}

func (a *Auth) callbackURL() string {
	base := strings.TrimRight(a.PublicURL, "/")
	if base == "" {
		base = "http://localhost:8080"
	}
	return base + CallbackPath
}

func (a *Auth) client() *http.Client {
	if a != nil && a.HTTP != nil {
		return a.HTTP
	}
	return http.DefaultClient
}

func (a *Auth) discoverResource(ctx context.Context, transport *http.Client, endpoint string) (protectedResource, error) {
	var meta protectedResource
	wellKnown := originOf(endpoint) + "/.well-known/oauth-protected-resource"
	if err := getJSON(ctx, transport, wellKnown, &meta); err != nil {
		// A server that has not published metadata yet can still do DCR against its own
		// origin, so a miss here is not fatal.
		return meta, nil
	}
	return meta, nil
}

func (a *Auth) discoverServer(ctx context.Context, transport *http.Client, issuer string) (authServer, error) {
	var meta authServer
	wellKnown := strings.TrimRight(issuer, "/") + "/.well-known/oauth-authorization-server"
	if err := getJSON(ctx, transport, wellKnown, &meta); err != nil {
		return authServer{}, fmt.Errorf("plugins: oauth discovery: %w", err)
	}
	return meta, nil
}

func (a *Auth) register(ctx context.Context, transport *http.Client, endpoint string) (registration, error) {
	payload, err := json.Marshal(map[string]any{
		"client_name":                "Vision Agents",
		"redirect_uris":              []string{a.callbackURL()},
		"grant_types":                []string{"authorization_code", "refresh_token"},
		"response_types":             []string{"code"},
		"token_endpoint_auth_method": "none",
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
		return registration{}, fmt.Errorf("plugins: register: %w", err)
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return registration{}, err
	}
	if response.StatusCode >= 300 {
		return registration{}, fmt.Errorf("plugins: register: %s", strings.TrimSpace(string(raw)))
	}
	var body registration
	if err := json.Unmarshal(raw, &body); err != nil {
		return registration{}, err
	}
	if body.ClientID == "" {
		return registration{}, fmt.Errorf("plugins: register: no client id")
	}
	return body, nil
}

func envClient(pluginID string) (id, secret string) {
	prefix := strings.ToUpper(pluginID) + "_MCP_"
	return os.Getenv(prefix + "CLIENT_ID"), os.Getenv(prefix + "CLIENT_SECRET")
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
	return json.NewDecoder(response.Body).Decode(target)
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
