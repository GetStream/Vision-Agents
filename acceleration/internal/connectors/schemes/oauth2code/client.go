package oauth2code

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// clientName is how the router names itself to an authorization server, in a registration
// and in its client metadata document, as the prototype did (internal/mcp/oauth.go:661,740
// on codex/connector-support at cf62af0d).
const clientName = "Vision Agents"

// grantTypes and responseTypes are what a registered client asks for: the code grant and
// its refresh (RFC 7591 section 2, grant_types and response_types).
var (
	grantTypes    = []string{"authorization_code", "refresh_token"}
	responseTypes = []string{"code"}
)

// supportedMethods are the token endpoint authentication methods this package implements
// (RFC 7591 section 2 defines all three). PrivateKeyJWT builds a private_key_jwt assertion
// but is not listed until a client record can hold a private key (T19).
var supportedMethods = []core.ClientAuthMethod{core.AuthNone, core.AuthClientSecretBasic, core.AuthClientSecretPost}

// Client is a preregistered OAuth client: the operator's app or a customer's own.
type Client struct {
	ID     string
	Secret string
	// AuthMethod overrides the manifest's client.auth_method when set.
	AuthMethod core.ClientAuthMethod
}

// ClientLookup finds the client source registered for this connection's connector. found
// is false when that source has none, which is not an error: the next source in
// client.from is tried.
type ClientLookup func(ctx context.Context, ref core.ConnectionRef, m core.ResolvedManifest, source core.ClientSource) (c Client, found bool, err error)

// EnvClients finds the operator's client in the environment, as <env>_MCP_CLIENT_ID and
// <env>_MCP_CLIENT_SECRET where env is the manifest's client.env: the prototype's names
// (internal/mcp/oauth.go:794-797 at cf62af0d) and T19's «operator environment». It answers
// for the operator only; customer clients are records (T19).
func EnvClients(getenv func(string) string) ClientLookup {
	return func(_ context.Context, _ core.ConnectionRef, m core.ResolvedManifest, source core.ClientSource) (Client, bool, error) {
		if source != core.ClientOperator || m.Client.Env == "" {
			return Client{}, false, nil
		}
		prefix := m.Client.Env + "_MCP_"
		c := Client{ID: getenv(prefix + "CLIENT_ID"), Secret: getenv(prefix + "CLIENT_SECRET")}
		return c, c.ID != "", nil
	}
}

// client is the client one attempt and then one connection use.
type client struct {
	// Source keeps the JSON name owner: it is sealed into attempts and stored credentials,
	// and payloadVersion 1 payloads already hold it under that name.
	Source     core.ClientSource     `json:"owner"`
	ID         string                `json:"id"`
	AuthMethod core.ClientAuthMethod `json:"auth_method"`
	// Secret is kept only for a client this scheme registered (dcr), which has nowhere
	// else to live. A preregistered client's secret is looked up again each time, so a
	// rotated one is used at once and lives in one place (T19).
	Secret string `json:"secret,omitempty"`
}

// sourceOrder is the order client.from's sources are tried in, whatever order the manifest
// lists them in. A preregistered client comes first and CIMD before DCR, as MCP 2025-11-25
// «Client Registration Approaches» orders them; a customer's client before the operator's,
// because a customer who registered its own app chose it over ours (T19).
var sourceOrder = []core.ClientSource{core.ClientCustomer, core.ClientOperator, core.ClientCIMD, core.ClientDCR}

// pickClient is the first client the manifest's client.from allows that is available.
func (s *Scheme) pickClient(ctx context.Context, ref core.ConnectionRef, m core.ResolvedManifest, d server, redirectURI string) (client, error) {
	for _, source := range sourceOrder {
		if !slices.Contains(m.Client.From, source) {
			continue
		}
		switch source {
		case core.ClientCustomer, core.ClientOperator:
			if s.cfg.Clients == nil {
				continue
			}
			found, ok, err := s.cfg.Clients(ctx, ref, m, source)
			if err != nil {
				return client{}, fmt.Errorf("oauth2code: %s client: %w", source, err)
			}
			if !ok {
				continue
			}
			method, err := preregisteredMethod(m, d, found)
			if err != nil {
				return client{}, err
			}
			return client{Source: source, ID: found.ID, AuthMethod: method}, nil
		case core.ClientCIMD:
			if s.cfg.ClientMetadataURL == "" || !d.CIMD {
				continue
			}
			// CIMD section 4.1: no shared secret, so the client is public here.
			// private_key_jwt, which §4.1 also allows, waits for a key store (PrivateKeyJWT).
			return client{Source: source, ID: s.cfg.ClientMetadataURL, AuthMethod: core.AuthNone}, nil
		case core.ClientDCR:
			if d.Registration == "" {
				continue
			}
			return s.register(ctx, m, d, redirectURI)
		}
	}
	return client{}, fmt.Errorf("%w (client.from %v)", ErrNoClient, m.Client.From)
}

// clientSecret is c with the secret the token request needs: a preregistered client's is
// looked up again, so the attempt never carried it.
func (s *Scheme) clientSecret(ctx context.Context, ref core.ConnectionRef, m core.ResolvedManifest, c client) (client, error) {
	if c.Source != core.ClientCustomer && c.Source != core.ClientOperator {
		return c, nil
	}
	if s.cfg.Clients == nil {
		return client{}, ErrNoClient
	}
	found, ok, err := s.cfg.Clients(ctx, ref, m, c.Source)
	if err != nil {
		return client{}, fmt.Errorf("oauth2code: %s client: %w", c.Source, err)
	}
	if !ok || found.ID != c.ID {
		return client{}, fmt.Errorf("oauth2code: the %s client changed during the consent; start it again", c.Source)
	}
	c.Secret = found.Secret
	if c.AuthMethod != core.AuthNone && c.Secret == "" {
		return client{}, fmt.Errorf("oauth2code: the %s client has no secret for %s", c.Source, c.AuthMethod)
	}
	return c, nil
}

// preregisteredMethod is how a preregistered client authenticates: the lookup's method,
// then the manifest's, then none for a client without a secret, then what the server
// advertises. client_secret_basic is preferred, since RFC 6749 section 2.3.1 makes it the
// one every server «MUST support» and calls the body form «NOT RECOMMENDED».
func preregisteredMethod(m core.ResolvedManifest, d server, c Client) (core.ClientAuthMethod, error) {
	method := firstMethod(c.AuthMethod, m.Client.AuthMethod)
	if method == "" {
		switch {
		case c.Secret == "":
			method = core.AuthNone
		case len(d.AuthMethods) == 0 || slices.Contains(d.AuthMethods, string(core.AuthClientSecretBasic)):
			method = core.AuthClientSecretBasic
		case slices.Contains(d.AuthMethods, string(core.AuthClientSecretPost)):
			method = core.AuthClientSecretPost
		default:
			return "", fmt.Errorf("oauth2code: the authorization server supports neither client_secret_basic nor client_secret_post, only %v", d.AuthMethods)
		}
	}
	return method, checkMethod(method, d)
}

// register is RFC 7591 dynamic client registration for this attempt's redirect URI.
func (s *Scheme) register(ctx context.Context, m core.ResolvedManifest, d server, redirectURI string) (client, error) {
	method := m.Client.AuthMethod
	if method == "" {
		// A public client (none) keeps no secret to store, so it is asked for when the
		// server allows it; RFC 7591 section 2 defines none as a public client. Then the two
		// secret methods, basic first for the reason preregisteredMethod gives. The order is
		// the prototype's (internal/mcp/oauth.go:777-788 at cf62af0d).
		switch {
		case len(d.AuthMethods) == 0 || slices.Contains(d.AuthMethods, string(core.AuthNone)):
			method = core.AuthNone
		case slices.Contains(d.AuthMethods, string(core.AuthClientSecretBasic)):
			method = core.AuthClientSecretBasic
		case slices.Contains(d.AuthMethods, string(core.AuthClientSecretPost)):
			method = core.AuthClientSecretPost
		default:
			return client{}, fmt.Errorf("oauth2code: the authorization server offers no client authentication this scheme has: %v", d.AuthMethods)
		}
	}
	if err := checkMethod(method, d); err != nil {
		return client{}, err
	}
	// RFC 7591 section 2: the client metadata of the registration request.
	body, err := json.Marshal(map[string]any{
		"client_name":                clientName,
		"redirect_uris":              []string{redirectURI},
		"grant_types":                grantTypes,
		"response_types":             responseTypes,
		"token_endpoint_auth_method": method,
	})
	if err != nil {
		return client{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, d.Registration, bytes.NewReader(body))
	if err != nil {
		return client{}, err
	}
	// RFC 7591 section 3.1: a JSON body.
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	response, err := s.cfg.HTTP.Do(request)
	if err != nil {
		return client{}, fmt.Errorf("oauth2code: register: %w", err)
	}
	defer response.Body.Close()
	// RFC 7591 section 3.2.1 answers 201 Created. Any 2xx is taken, as the prototype did
	// (internal/mcp/oauth.go:760), since the body is what carries the client.
	if response.StatusCode < 200 || response.StatusCode > 299 {
		// The body is not quoted: a registration error may echo what was sent.
		return client{}, fmt.Errorf("oauth2code: register: HTTP %d", response.StatusCode)
	}
	raw, err := read(response.Body)
	if err != nil {
		return client{}, err
	}
	var registered struct {
		ClientID     string                `json:"client_id"`
		ClientSecret string                `json:"client_secret"`
		Method       core.ClientAuthMethod `json:"token_endpoint_auth_method"`
	}
	if err := json.Unmarshal(raw, &registered); err != nil {
		return client{}, fmt.Errorf("oauth2code: register: %w", err)
	}
	if registered.ClientID == "" {
		// RFC 7591 section 3.2.1: client_id is REQUIRED.
		return client{}, errors.New("oauth2code: register: no client_id in the response")
	}
	// RFC 7591 section 3.2.1: the server may replace a value it was sent, so the method it
	// answers is the one the client has.
	if registered.Method != "" {
		method = registered.Method
	}
	if !slices.Contains(supportedMethods, method) {
		return client{}, fmt.Errorf("oauth2code: register: the server assigned %q, which this scheme does not implement", method)
	}
	if method != core.AuthNone && registered.ClientSecret == "" {
		return client{}, fmt.Errorf("oauth2code: register: %s without a client_secret", method)
	}
	return client{Source: core.ClientDCR, ID: registered.ClientID, Secret: registered.ClientSecret, AuthMethod: method}, nil
}

// checkMethod is whether this scheme implements method and the server, when it lists its
// methods (RFC 8414 section 2), accepts it.
func checkMethod(method core.ClientAuthMethod, d server) error {
	if !slices.Contains(supportedMethods, method) {
		return fmt.Errorf("oauth2code: client authentication %q is not implemented (only %v)", method, supportedMethods)
	}
	if len(d.AuthMethods) > 0 && !slices.Contains(d.AuthMethods, string(method)) {
		return fmt.Errorf("oauth2code: the authorization server does not accept %s, only %v", method, d.AuthMethods)
	}
	return nil
}

func firstMethod(methods ...core.ClientAuthMethod) core.ClientAuthMethod {
	for _, m := range methods {
		if m != "" {
			return m
		}
	}
	return ""
}

// ClientMetadata is the router's client metadata document under CIMD
// (draft-ietf-oauth-client-id-metadata-document-02, section 4): what an authorization
// server fetches from the client_id URL. Its fields are RFC 7591 section 2 client metadata.
type ClientMetadata struct {
	ClientID                string   `json:"client_id"`
	ClientName              string   `json:"client_name"`
	RedirectURIs            []string `json:"redirect_uris"`
	GrantTypes              []string `json:"grant_types"`
	ResponseTypes           []string `json:"response_types"`
	TokenEndpointAuthMethod string   `json:"token_endpoint_auth_method"`
}

// ClientMetadataDocument is the document to serve at clientID, the Config's
// ClientMetadataURL, naming the redirect URIs Begin is given. client_id must be the URL
// itself (CIMD section 4); MCP 2025-11-25 «Client ID Metadata Documents» requires
// client_id, client_name and redirect_uris; the method is none, since the document cannot
// carry a shared secret (CIMD section 4.1). The prototype served the same fields
// (internal/mcp/oauth.go:657-667 at cf62af0d).
func ClientMetadataDocument(clientID string, redirectURIs []string) ClientMetadata {
	return ClientMetadata{
		ClientID:                clientID,
		ClientName:              clientName,
		RedirectURIs:            slices.Clone(redirectURIs),
		GrantTypes:              slices.Clone(grantTypes),
		ResponseTypes:           slices.Clone(responseTypes),
		TokenEndpointAuthMethod: string(core.AuthNone),
	}
}

// checkClientIdentifierURL is CIMD section 3: https, no userinfo, a path that is not only
// «/» (NOT RECOMMENDED there, refused here), no dot segments, no fragment. A query, which
// §3 says SHOULD NOT be there, is refused too, so the URL compares the same everywhere.
func checkClientIdentifierURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "https" || u.Host == "" || u.User != nil || u.RawQuery != "" || u.ForceQuery ||
		u.Fragment != "" || strings.Contains(raw, "#") || u.Path == "" || u.Path == "/" ||
		slices.ContainsFunc(strings.Split(u.Path, "/"), func(seg string) bool { return seg == "." || seg == ".." }) {
		return fmt.Errorf("oauth2code: ClientMetadataURL %q is not a CIMD client identifier URL: https, a path, no userinfo, dot segments, query or fragment", raw)
	}
	return nil
}
