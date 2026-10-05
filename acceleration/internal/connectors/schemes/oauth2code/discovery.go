package oauth2code

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// maxResponseBytes caps what is read from any discovery, registration or token response.
// It is the prototype's limit (internal/mcp/oauth.go:124 on codex/connector-support at
// cf62af0d); a metadata document or a token response is a few kilobytes.
const maxResponseBytes = 1 << 20

// server is the authorization server one attempt talks to: what the manifest pinned, with
// what discovery found filled in under it.
type server struct {
	Issuer       string
	Authorize    string
	Token        string
	Registration string
	Revocation   string
	// Resource is the RFC 8707 resource indicator sent on authorize and on the exchange.
	Resource string
	// AuthMethods, CodeChallengeMethods, CIMD and IssParameter are RFC 8414 section 2's
	// token_endpoint_auth_methods_supported, code_challenge_methods_supported, CIMD
	// section 6's client_id_metadata_document_supported and RFC 9207 section 3's
	// authorization_response_iss_parameter_supported.
	AuthMethods          []string
	CodeChallengeMethods []string
	CIMD                 bool
	IssParameter         bool
	// HasMetadata is whether authorization server metadata was read at all.
	HasMetadata bool
	// ViaResource is whether the server was found through an MCP endpoint's protected
	// resource metadata, which is when the MCP authorization rules apply.
	ViaResource bool
}

// discover is the manifest's endpoints first, then what metadata says for anything left.
// Endpoint roles read: authorize, token, revoke, issuer, resource, mcp.
//
//   - With issuer set, its metadata is read (RFC 8414).
//   - Without issuer, and with authorize or token missing or a client.registration that
//     registers a client (cimd, dcr), the issuer is found from the mcp endpoint's protected resource
//     metadata (RFC 9728), as MCP 2025-11-25 «Authorization Server Discovery» does.
//   - Without either, nothing is fetched: the endpoints are static, as for a provider
//     whose manifest pins them.
func (s *Scheme) discover(ctx context.Context, m core.ResolvedManifest) (server, error) {
	d := server{
		Issuer:     m.Endpoints["issuer"],
		Authorize:  m.Endpoints["authorize"],
		Token:      m.Endpoints["token"],
		Revocation: m.Endpoints["revoke"],
		Resource:   m.Endpoints["resource"],
	}
	registers := slices.Contains(m.Client.Registration, core.ClientCIMD) || slices.Contains(m.Client.Registration, core.ClientDCR)
	if d.Issuer == "" && (d.Authorize == "" || d.Token == "" || registers) {
		mcp := m.Endpoints["mcp"]
		if mcp == "" {
			return server{}, errors.New("oauth2code: the manifest has no authorize and token endpoints, and no issuer or mcp endpoint to discover them from")
		}
		resource, err := s.protectedResource(ctx, mcp)
		if err != nil {
			return server{}, err
		}
		// RFC 9728 section 7.6 leaves the choice among several to the client; the first
		// is the one the resource lists first.
		d.Issuer = resource.AuthorizationServers[0]
		if d.Resource == "" {
			d.Resource = resource.Resource
		}
		d.ViaResource = true
	}
	if d.Issuer != "" {
		metadata, err := s.authorizationServer(ctx, d.Issuer)
		if err != nil {
			return server{}, err
		}
		// What the manifest pinned wins over what the server advertises.
		d.Authorize = firstSet(d.Authorize, metadata.AuthorizationEndpoint)
		d.Token = firstSet(d.Token, metadata.TokenEndpoint)
		d.Revocation = firstSet(d.Revocation, metadata.RevocationEndpoint)
		d.Registration = metadata.RegistrationEndpoint
		d.AuthMethods = metadata.TokenEndpointAuthMethods
		d.CodeChallengeMethods = metadata.CodeChallengeMethods
		d.CIMD = metadata.ClientIDMetadataDocumentSupported
		d.IssParameter = metadata.AuthorizationResponseIssParameterSupported
		d.HasMetadata = true
	}
	if d.Authorize == "" || d.Token == "" {
		return server{}, errors.New("oauth2code: no authorize or token endpoint in the manifest or the authorization server metadata")
	}
	// Every endpoint, pinned or discovered, is held to the egress policy before it is used.
	// The authorize URL goes to the browser, so egress never dials it and this is its only
	// check; the others leave through the caller's client, which is egress in the router,
	// and are checked here too so a scheme given another client still refuses them. The
	// prototype did the same (internal/mcp/oauth.go:676 on codex/connector-support at
	// cf62af0d).
	for _, endpoint := range []string{d.Authorize, d.Token, d.Registration, d.Revocation} {
		if endpoint == "" {
			continue
		}
		if err := s.checkEndpoint(ctx, endpoint); err != nil {
			return server{}, err
		}
	}
	if d.Issuer == "" {
		// RFC 9207 section 2.4: without metadata, the issuer an iss is compared with comes
		// from «deployment-specific ways (for example, a static configuration)». With no
		// issuer endpoint in the manifest that is the authorize endpoint's origin, as the
		// prototype did (internal/mcp/oauth.go:239-241 at cf62af0d).
		d.Issuer = origin(d.Authorize)
	}
	return d, nil
}

// checkEndpoint refuses an authorization server endpoint that is not https, carries
// userinfo or a fragment, or does not reach a public address. RFC 6749 sections 3.1 and
// 3.2: both endpoints require TLS, «MAY include an "application/x-www-form-urlencoded"
// formatted query component ... which MUST be retained», and «MUST NOT include a fragment
// component». So the query is kept on the URL and only left out of what PublicEndpoint
// sees, since egress refuses a query.
func (s *Scheme) checkEndpoint(ctx context.Context, endpoint string) error {
	u, err := url.Parse(endpoint)
	if err != nil || u.Scheme != "https" || u.Host == "" || u.User != nil || u.Fragment != "" {
		return fmt.Errorf("oauth2code: endpoint %q is not an https URL without userinfo or fragment", endpoint)
	}
	bare := *u
	bare.RawQuery, bare.ForceQuery = "", false
	if err := s.cfg.PublicEndpoint(ctx, bare.String()); err != nil {
		return fmt.Errorf("oauth2code: endpoint %q: %w", endpoint, err)
	}
	return nil
}

// checkPKCE refuses a server that cannot do S256.
func (d server) checkPKCE() error {
	switch {
	case len(d.CodeChallengeMethods) > 0 && !slices.Contains(d.CodeChallengeMethods, "S256"):
		// RFC 9700 section 2.1.1: S256 is the only method that does not expose the verifier.
		return errors.New("oauth2code: the authorization server does not support PKCE S256")
	case d.ViaResource && len(d.CodeChallengeMethods) == 0:
		// MCP 2025-11-25 «Authorization Code Protection»: if code_challenge_methods_supported
		// is absent, «MCP clients MUST refuse to proceed».
		return errors.New("oauth2code: the MCP server's authorization server does not advertise PKCE")
	}
	// Otherwise PKCE is sent anyway: RFC 7636 section 5, clients «SHOULD send the
	// additional parameters ... to all servers», and a server without it ignores them.
	return nil
}

type protectedResourceMetadata struct {
	Resource             string   `json:"resource"`
	AuthorizationServers []string `json:"authorization_servers"`
}

// protectedResource reads RFC 9728 metadata for an MCP endpoint, at the path-inserted
// well-known URL first and then at the root, which is MCP 2025-11-25's order («Protected
// Resource Metadata Discovery Requirements»). The WWW-Authenticate route MCP also names
// needs an unauthenticated request to the MCP server and is not taken here.
func (s *Scheme) protectedResource(ctx context.Context, endpoint string) (protectedResourceMetadata, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return protectedResourceMetadata{}, fmt.Errorf("oauth2code: mcp endpoint: %w", err)
	}
	root := u.Scheme + "://" + u.Host
	// RFC 9728 section 3.1: a terminating slash after the host is removed before the
	// well-known suffix goes in.
	path := strings.TrimSuffix(u.EscapedPath(), "/")
	type candidate struct{ url, identifier string }
	candidates := []candidate{{root + "/.well-known/oauth-protected-resource", root}}
	if path != "" {
		candidates = slices.Insert(candidates, 0, candidate{root + "/.well-known/oauth-protected-resource" + path, endpoint})
	}
	// A candidate that cannot be used, for any reason, sends the search on to the next one:
	// MCP's order is a list to try, and RFC 9728 section 3.3 says only that a mismatched
	// document «MUST NOT be used», not that the search ends. A refused redirect is such a
	// reason too: Slack's path-inserted URL answers 302 to mcp-9827.slack.com, which egress
	// does not follow, while its root document answers 200 (both GET at 2026-10-02T19:57Z).
	var failures []error
	for _, c := range candidates {
		var metadata protectedResourceMetadata
		found, err := s.getJSON(ctx, c.url, &metadata)
		switch {
		case ctx.Err() != nil:
			return protectedResourceMetadata{}, ctx.Err()
		case err != nil:
			failures = append(failures, err)
		case !found:
			// 404 or 410: try the next URL.
		case metadata.Resource != c.identifier:
			// RFC 9728 section 3.3: resource «MUST be identical» to the identifier the
			// well-known suffix was inserted into, or the document «MUST NOT be used».
			failures = append(failures, fmt.Errorf("oauth2code: protected resource metadata at %s names resource %q, not %q", c.url, metadata.Resource, c.identifier))
		case len(metadata.AuthorizationServers) == 0:
			failures = append(failures, fmt.Errorf("oauth2code: protected resource metadata at %s names no authorization server", c.url))
		default:
			return metadata, nil
		}
	}
	return protectedResourceMetadata{}, errors.Join(append([]error{fmt.Errorf("oauth2code: no usable protected resource metadata for %s", endpoint)}, failures...)...)
}

type authorizationServerMetadata struct {
	Issuer                                     string   `json:"issuer"`
	AuthorizationEndpoint                      string   `json:"authorization_endpoint"`
	TokenEndpoint                              string   `json:"token_endpoint"`
	RegistrationEndpoint                       string   `json:"registration_endpoint"`
	RevocationEndpoint                         string   `json:"revocation_endpoint"`
	TokenEndpointAuthMethods                   []string `json:"token_endpoint_auth_methods_supported"`
	CodeChallengeMethods                       []string `json:"code_challenge_methods_supported"`
	ClientIDMetadataDocumentSupported          bool     `json:"client_id_metadata_document_supported"`
	AuthorizationResponseIssParameterSupported bool     `json:"authorization_response_iss_parameter_supported"`
}

// authorizationServer reads an issuer's metadata at the URLs MCP 2025-11-25 «Authorization
// Server Metadata Discovery» lists, in its order: RFC 8414 section 3.1 with the path
// inserted, then OpenID Connect Discovery with the path inserted, then with it appended.
func (s *Scheme) authorizationServer(ctx context.Context, issuer string) (authorizationServerMetadata, error) {
	u, err := url.Parse(issuer)
	if err != nil || u.Scheme != "https" || u.Host == "" || u.RawQuery != "" || u.Fragment != "" {
		// RFC 8414 section 2: an issuer is an https URL with no query or fragment.
		return authorizationServerMetadata{}, fmt.Errorf("oauth2code: issuer %q is not an https URL without query or fragment", issuer)
	}
	root := u.Scheme + "://" + u.Host
	// RFC 8414 section 3.1: «any terminating "/" MUST be removed».
	path := strings.TrimSuffix(u.EscapedPath(), "/")
	candidates := []string{
		root + "/.well-known/oauth-authorization-server" + path,
		root + "/.well-known/openid-configuration" + path,
	}
	if path != "" {
		candidates = append(candidates, root+path+"/.well-known/openid-configuration")
	}
	// As for protected resource metadata, a candidate that cannot be used sends the search on.
	var failures []error
	for _, candidate := range candidates {
		var metadata authorizationServerMetadata
		found, err := s.getJSON(ctx, candidate, &metadata)
		switch {
		case ctx.Err() != nil:
			return authorizationServerMetadata{}, ctx.Err()
		case err != nil:
			failures = append(failures, err)
		case !found:
			// 404 or 410: try the next URL.
		case metadata.Issuer != issuer:
			// RFC 8414 section 3.3 and OpenID Connect Discovery 1.0 section 4.3: the issuer
			// «MUST be identical» to the one the URL was built from, or the metadata is not used.
			failures = append(failures, fmt.Errorf("oauth2code: metadata at %s names issuer %q, not %q", candidate, metadata.Issuer, issuer))
		default:
			return metadata, nil
		}
	}
	return authorizationServerMetadata{}, errors.Join(append([]error{fmt.Errorf("oauth2code: no usable authorization server metadata for %s", issuer)}, failures...)...)
}

// getJSON reads one metadata document. found is false for a 404 or 410, which is how a
// server says it does not publish that document; any other failure is an error, which the
// callers record and then try the next well-known URL.
func (s *Scheme) getJSON(ctx context.Context, target string, into any) (found bool, err error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return false, err
	}
	request.Header.Set("Accept", "application/json")
	response, err := s.cfg.HTTP.Do(request)
	if err != nil {
		return false, fmt.Errorf("oauth2code: GET %s: %w", target, err)
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusNotFound || response.StatusCode == http.StatusGone {
		return false, nil
	}
	// RFC 8414 section 3.2 and RFC 9728 section 3.2: a 200 with a JSON object.
	if response.StatusCode != http.StatusOK {
		return false, fmt.Errorf("oauth2code: GET %s: HTTP %d", target, response.StatusCode)
	}
	raw, err := read(response.Body)
	if err != nil {
		return false, err
	}
	if err := json.Unmarshal(raw, into); err != nil {
		return false, fmt.Errorf("oauth2code: GET %s: %w", target, err)
	}
	return true, nil
}

// read reads a whole response body up to maxResponseBytes, and fails past it rather than
// parsing a cut document.
func read(body io.Reader) ([]byte, error) {
	raw, err := io.ReadAll(io.LimitReader(body, maxResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("oauth2code: read response: %w", err)
	}
	if len(raw) > maxResponseBytes {
		return nil, errors.New("oauth2code: response is larger than 1 MiB")
	}
	return bytes.TrimSpace(raw), nil
}

// firstSet is the first of its arguments that is set.
func firstSet(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func origin(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return u.Scheme + "://" + u.Host
}
