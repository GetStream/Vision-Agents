package plugins

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
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrClientRequired is a plugin that registers no OAuth client on the fly, logged into for
// an agent that set none, on a deployment that has none of its own either.
var ErrClientRequired = errors.New("needs an OAuth client set for this agent")

// Client is an OAuth client registered with a provider in advance.
type Client struct {
	ID     string
	Secret string
}

// Owner is the agent config a login is made for.
type Owner struct {
	CustomerID string
	ConfigID   string
}

// ClientLookup finds the client an agent config set for a plugin. found is false when it
// set none, which is not an error.
type ClientLookup func(ctx context.Context, owner Owner, pluginID string) (client Client, found bool, err error)

// Auth is the OAuth 2.1 + PKCE client used to connect a hosted MCP server.
type Auth struct {
	// HTTP reaches the servers and their auth servers. Nil reaches only public hosts.
	HTTP         *http.Client
	PublicURL    string
	DashboardURL string
	// Clients finds the client an agent set for a plugin, which comes before this
	// deployment's own. Nil means no agent has one.
	Clients ClientLookup
}

// Pending is what authorize has to remember so the callback can finish the login.
type Pending struct {
	PluginID      string
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
	ScopesSupported      []string `json:"scopes_supported"`
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
func (a *Auth) StartAuthorize(ctx context.Context, owner Owner, plugin Plugin, instance string) (Pending, error) {
	endpoint, err := plugin.Endpoint(instance)
	if err != nil {
		return Pending{}, err
	}
	transport := a.client()
	resource, meta, err := a.discover(ctx, transport, endpoint)
	if err != nil {
		return Pending{}, err
	}
	if meta.AuthorizationEndpoint == "" || meta.TokenEndpoint == "" {
		return Pending{}, stack.Wrap(fmt.Errorf("plugins: %s did not advertise oauth endpoints", plugin.ID))
	}

	clientID := ""
	if !plugin.ByURL {
		preregistered, err := a.preregistered(ctx, owner, plugin.ID)
		if err != nil {
			return Pending{}, err
		}
		clientID = preregistered.ID
	}
	// A plugin that needs a client may still advertise registration, as Gong does for the
	// clients it has approved: registering would only fail, less clearly than saying so.
	if clientID == "" && meta.RegistrationEndpoint != "" && !plugin.ClientRequired {
		registered, err := a.register(ctx, transport, meta.RegistrationEndpoint)
		if err != nil {
			return Pending{}, err
		}
		clientID = registered.ClientID
	}
	if clientID == "" && plugin.ByURL {
		return Pending{}, stack.Wrap(fmt.Errorf("plugins: %s does not advertise dynamic client registration", plugin.Name))
	}
	if clientID == "" {
		return Pending{}, stack.Wrap(fmt.Errorf("plugins: %s %w", plugin.Name, ErrClientRequired))
	}

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
	query.Set("redirect_uri", a.CallbackURL())
	query.Set("state", state)
	query.Set("code_challenge", challenge)
	query.Set("code_challenge_method", "S256")
	query.Set("resource", resourceOf(endpoint))
	scopes := plugin.Scopes
	if len(scopes) == 0 && plugin.ByURL {
		scopes = resource.ScopesSupported
	}
	if len(scopes) > 0 {
		query.Set("scope", strings.Join(scopes, " "))
	}
	for name, value := range plugin.AuthorizeParams {
		query.Set(name, value)
	}

	return Pending{
		PluginID:      plugin.ID,
		State:         state,
		CodeVerifier:  verifier,
		ClientID:      clientID,
		TokenEndpoint: meta.TokenEndpoint,
		AuthorizeURL:  meta.AuthorizationEndpoint + "?" + query.Encode(),
	}, nil
}

// Exchange finishes the login with the code the provider sent back.
func (a *Auth) Exchange(ctx context.Context, owner Owner, pending Pending, code string) (Token, error) {
	form := url.Values{}
	form.Set("grant_type", "authorization_code")
	form.Set("code", code)
	form.Set("redirect_uri", a.CallbackURL())
	form.Set("client_id", pending.ClientID)
	form.Set("code_verifier", pending.CodeVerifier)
	if err := a.setClientSecret(ctx, form, owner, pending.PluginID, pending.ClientID); err != nil {
		return Token{}, err
	}

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
func (a *Auth) Refresh(ctx context.Context, owner Owner, pluginID, tokenEndpoint, clientID, refreshToken string) (Token, error) {
	if refreshToken == "" || tokenEndpoint == "" {
		return Token{}, stack.Wrap(fmt.Errorf("plugins: nothing to refresh"))
	}
	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("refresh_token", refreshToken)
	form.Set("client_id", clientID)
	if err := a.setClientSecret(ctx, form, owner, pluginID, clientID); err != nil {
		return Token{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return Token{}, stack.Wrap(err)
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response, err := a.client().Do(request)
	if err != nil {
		return Token{}, stack.Wrap(err)
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return Token{}, stack.Wrap(err)
	}
	var body tokenResponse
	if err := json.Unmarshal(raw, &body); err != nil {
		return Token{}, stack.Wrap(err)
	}
	if body.AccessToken == "" {
		return Token{}, stack.Wrap(fmt.Errorf("plugins: refresh: %s", or(body.ErrorDesc, "no access token")))
	}
	token := Token{AccessToken: body.AccessToken, RefreshToken: or(body.RefreshToken, refreshToken)}
	if body.ExpiresIn > 0 {
		at := time.Now().UTC().Add(time.Duration(body.ExpiresIn) * time.Second)
		token.ExpiresAt = &at
	}
	return token, nil
}

// DashboardRedirect is where the browser should land after the callback. plugin_connected
// tells a dashboard that opened the login in a popup to close it.
func (a *Auth) DashboardRedirect(configID, pluginID string) string {
	base := strings.TrimRight(a.DashboardURL, "/")
	if base == "" {
		base = "http://localhost:3000"
	}
	return base + "/agents/" + configID + "?plugin_connected=" + url.QueryEscape(pluginID)
}

// CallbackURL is where a provider sends the browser back to, which an OAuth client
// registered in advance has to list as a redirect URI.
func (a *Auth) CallbackURL() string {
	return a.base() + CallbackPath
}

// LogoURL is where this deployment serves a plugin's logo, for a card to draw it.
func (a *Auth) LogoURL(pluginID string) string {
	return a.base() + LogoPath(pluginID)
}

func (a *Auth) base() string {
	if base := strings.TrimRight(a.PublicURL, "/"); base != "" {
		return base
	}
	return "http://localhost:8080"
}

func (a *Auth) client() *http.Client {
	if a != nil && a.HTTP != nil {
		return a.HTTP
	}
	return publicClient
}

// CheckLogin reports why the MCP server at endpoint cannot be logged into with a client the
// router registers itself, or nil when it can. A server that could not be reached at all
// fails with a *url.Error.
func (a *Auth) CheckLogin(ctx context.Context, endpoint string) error {
	_, meta, err := a.discover(ctx, a.client(), endpoint)
	if err != nil {
		return err
	}
	if meta.AuthorizationEndpoint == "" || meta.TokenEndpoint == "" {
		return fmt.Errorf("plugins: %s did not advertise oauth endpoints", endpoint)
	}
	if meta.RegistrationEndpoint == "" {
		return fmt.Errorf("plugins: %s does not advertise dynamic client registration", endpoint)
	}
	return nil
}

// NeedsLogin reports whether the MCP server at endpoint requires an OAuth login: it
// publishes protected-resource metadata naming an authorization server, or refuses a request
// without a token with a 401 that says how to authenticate. A server that could not be
// reached fails with a *url.Error, and one failing on its side says nothing either way.
func (a *Auth) NeedsLogin(ctx context.Context, endpoint string) (bool, error) {
	transport := a.client()
	for _, candidate := range resourceCandidates(endpoint) {
		var meta protectedResource
		if getJSON(ctx, transport, candidate, &meta) == nil && len(meta.AuthorizationServers) > 0 {
			return true, nil
		}
	}
	response, err := unauthenticated(ctx, transport, endpoint)
	if err != nil {
		return false, stack.Wrap(err)
	}
	defer response.Body.Close()
	if response.StatusCode >= http.StatusInternalServerError {
		return false, stack.Wrap(fmt.Errorf("plugins: %s: %s", endpoint, response.Status))
	}
	return response.StatusCode == http.StatusUnauthorized && response.Header.Get("WWW-Authenticate") != "", nil
}

// discover reads the server's protected-resource metadata and then its authorization
// server's, which is its own origin when the server names none.
func (a *Auth) discover(ctx context.Context, transport *http.Client, endpoint string) (protectedResource, authServer, error) {
	resource, err := a.discoverResource(ctx, transport, endpoint)
	if err != nil {
		return protectedResource{}, authServer{}, err
	}
	issuer := first(resource.AuthorizationServers)
	if issuer == "" {
		issuer = originOf(endpoint)
	}
	meta, err := a.discoverServer(ctx, transport, issuer)
	return resource, meta, err
}

// discoverResource reads RFC 9728 metadata, at the endpoint's path first (section 3.1, and
// where Sentry and Google Calendar publish it), then at the origin, then wherever the
// server's WWW-Authenticate points when it refuses a request without a login (section 5.1).
func (a *Auth) discoverResource(ctx context.Context, transport *http.Client, endpoint string) (protectedResource, error) {
	var meta protectedResource
	for _, candidate := range resourceCandidates(endpoint) {
		if err := getJSON(ctx, transport, candidate, &meta); err == nil {
			return meta, nil
		}
	}
	if link := resourceMetadataLink(ctx, transport, endpoint); link != "" {
		if err := getJSON(ctx, transport, link, &meta); err == nil {
			return meta, nil
		}
	}
	// A server that has not published metadata yet can still do DCR against its own
	// origin, so a miss here is not fatal.
	return protectedResource{}, nil
}

// resourceCandidates are where RFC 9728 metadata for endpoint may be, in the order to try.
func resourceCandidates(endpoint string) []string {
	wellKnown := originOf(endpoint) + "/.well-known/oauth-protected-resource"
	if parsed, err := url.Parse(endpoint); err == nil && strings.Trim(parsed.Path, "/") != "" {
		return []string{wellKnown + "/" + strings.Trim(parsed.Path, "/"), wellKnown}
	}
	return []string{wellKnown}
}

// discoverServer reads the authorization server's metadata: RFC 8414 first, then OpenID
// Connect discovery, which is all GitHub publishes for github.com/login/oauth.
func (a *Auth) discoverServer(ctx context.Context, transport *http.Client, issuer string) (authServer, error) {
	base := strings.TrimRight(issuer, "/")
	var failure error
	for _, wellKnown := range []string{
		base + "/.well-known/oauth-authorization-server",
		base + "/.well-known/openid-configuration",
	} {
		var meta authServer
		if err := getJSON(ctx, transport, wellKnown, &meta); err != nil {
			failure = err
			continue
		}
		return meta, nil
	}
	return authServer{}, stack.Wrap(fmt.Errorf("plugins: oauth discovery: %w", failure))
}

func (a *Auth) register(ctx context.Context, transport *http.Client, endpoint string) (registration, error) {
	payload, err := json.Marshal(map[string]any{
		"client_name":                "Vision Agents",
		"redirect_uris":              []string{a.CallbackURL()},
		"grant_types":                []string{"authorization_code", "refresh_token"},
		"response_types":             []string{"code"},
		"token_endpoint_auth_method": "none",
	})
	if err != nil {
		return registration{}, stack.Wrap(err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(string(payload)))
	if err != nil {
		return registration{}, stack.Wrap(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := transport.Do(request)
	if err != nil {
		return registration{}, stack.Wrap(fmt.Errorf("plugins: register: %w", err))
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return registration{}, stack.Wrap(err)
	}
	if response.StatusCode >= 300 {
		return registration{}, stack.Wrap(fmt.Errorf("plugins: register: %s", strings.TrimSpace(string(raw))))
	}
	var body registration
	if err := json.Unmarshal(raw, &body); err != nil {
		return registration{}, stack.Wrap(err)
	}
	if body.ClientID == "" {
		return registration{}, stack.Wrap(fmt.Errorf("plugins: register: no client id"))
	}
	return body, nil
}

// preregistered is the client a login to a plugin goes through when the provider registers
// none on the fly: the one the agent set, then this deployment's own. Empty when neither.
func (a *Auth) preregistered(ctx context.Context, owner Owner, pluginID string) (Client, error) {
	if a != nil && a.Clients != nil {
		client, found, err := a.Clients(ctx, owner, pluginID)
		if err != nil {
			return Client{}, err
		}
		if found {
			return client, nil
		}
	}
	return envClient(pluginID), nil
}

// HasClient reports whether a login to the plugin for owner has a client registered in
// advance, the agent's own or the deployment's.
func (a *Auth) HasClient(ctx context.Context, owner Owner, pluginID string) (bool, error) {
	client, err := a.preregistered(ctx, owner, pluginID)
	return client.ID != "", err
}

func envClient(pluginID string) Client {
	prefix := strings.ToUpper(pluginID) + "_MCP_"
	return Client{ID: os.Getenv(prefix + "CLIENT_ID"), Secret: os.Getenv(prefix + "CLIENT_SECRET")}
}

// setClientSecret authenticates a preregistered client at the token endpoint
// (client_secret_post), which is how a provider that registers none on the fly, such as
// Google, wants it. The secret is looked up again each time, so a rotated one is used at
// once. A client registered on the fly is public and sends none.
func (a *Auth) setClientSecret(ctx context.Context, form url.Values, owner Owner, pluginID, clientID string) error {
	client, err := a.preregistered(ctx, owner, pluginID)
	if err != nil {
		return err
	}
	if client.ID != "" && client.ID == clientID && client.Secret != "" {
		form.Set("client_secret", client.Secret)
	}
	return nil
}

func getJSON(ctx context.Context, transport *http.Client, url string, target any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return stack.Wrap(err)
	}
	request.Header.Set("Accept", "application/json")
	response, err := transport.Do(request)
	if err != nil {
		return stack.Wrap(err)
	}
	defer response.Body.Close()
	if response.StatusCode >= 300 {
		return stack.Wrap(fmt.Errorf("%s: %s", url, response.Status))
	}
	return stack.Wrap(json.NewDecoder(response.Body).Decode(target))
}

// resourceMetadataLink is the resource_metadata the server's WWW-Authenticate names when it
// refuses a request without a login, or empty.
func resourceMetadataLink(ctx context.Context, transport *http.Client, endpoint string) string {
	response, err := unauthenticated(ctx, transport, endpoint)
	if err != nil {
		return ""
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusUnauthorized {
		return ""
	}
	_, link, ok := strings.Cut(response.Header.Get("WWW-Authenticate"), `resource_metadata="`)
	if !ok {
		return ""
	}
	link, _, _ = strings.Cut(link, `"`)
	return link
}

// unauthenticated is how the server answers a ping sent without a token.
func unauthenticated(ctx context.Context, transport *http.Client, endpoint string) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint,
		strings.NewReader(`{"jsonrpc":"2.0","id":0,"method":"ping"}`))
	if err != nil {
		return nil, stack.Wrap(err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json, text/event-stream")
	response, err := transport.Do(request)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return response, nil
}

func pkce() (verifier, challenge string, err error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", "", stack.Wrap(err)
	}
	verifier = base64.RawURLEncoding.EncodeToString(raw)
	sum := sha256.Sum256([]byte(verifier))
	return verifier, base64.RawURLEncoding.EncodeToString(sum[:]), nil
}

func randomHex(n int) (string, error) {
	raw := make([]byte, n)
	if _, err := rand.Read(raw); err != nil {
		return "", stack.Wrap(err)
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

// resourceOf is the server an endpoint is on, without the query that picks its toolsets,
// so a login is for the server and changing toolsets needs no new one.
func resourceOf(endpoint string) string {
	resource, _, _ := strings.Cut(endpoint, "?")
	return resource
}

func first(values []string) string {
	if len(values) == 0 {
		return ""
	}
	return values[0]
}
