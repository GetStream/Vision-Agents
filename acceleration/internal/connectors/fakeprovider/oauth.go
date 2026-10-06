package fakeprovider

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"sort"
	"strconv"
	"strings"
)

func (s *Server) protectedResource(w http.ResponseWriter, _ *http.Request) {
	// RFC 9728 §2: resource and authorization_servers.
	writeJSON(w, http.StatusOK, map[string]any{
		"resource":              s.resource(),
		"authorization_servers": []string{s.URL},
	})
}

func (s *Server) authorizationServer(w http.ResponseWriter, _ *http.Request) {
	s.mu.Lock()
	cimd := s.is(ClientMetadataDocuments)
	s.mu.Unlock()
	metadata := map[string]any{
		// RFC 8414 §2: issuer, authorization_endpoint, token_endpoint and
		// response_types_supported are required; the rest are optional and say what this
		// server does.
		"issuer":                   s.URL,
		"authorization_endpoint":   s.URL + PathAuthorize,
		"token_endpoint":           s.URL + PathToken,
		"registration_endpoint":    s.URL + PathRegister,
		"revocation_endpoint":      s.URL + PathRevoke,
		"response_types_supported": []string{"code"},
		"grant_types_supported":    []string{"authorization_code", "refresh_token"},
		// S256 only: RFC 9700 §2.1.1 «Currently, S256 is the only such method», and MCP
		// clients «MUST use the S256 code challenge method» (authorization spec 2025-11-25).
		"code_challenge_methods_supported": []string{"S256"},
		// RFC 7591 §2 defines all three.
		"token_endpoint_auth_methods_supported": []string{"none", "client_secret_basic", "client_secret_post"},
		// RFC 9207 §3.
		"authorization_response_iss_parameter_supported": true,
	}
	if cimd {
		// CIMD §6.
		metadata["client_id_metadata_document_supported"] = true
	}
	writeJSON(w, http.StatusOK, metadata)
}

// register is RFC 7591 dynamic client registration.
func (s *Server) register(w http.ResponseWriter, r *http.Request) {
	var request struct {
		RedirectURIs []string `json:"redirect_uris"`
		Method       string   `json:"token_endpoint_auth_method"`
	}
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_client_metadata"})
		return
	}
	// RFC 7591 §3.2.2: invalid_redirect_uri; the code flow needs at least one.
	if len(request.RedirectURIs) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_redirect_uri"})
		return
	}
	// RFC 7591 §2: «If unspecified or omitted, the default is "client_secret_basic"».
	method := request.Method
	if method == "" {
		method = "client_secret_basic"
	}
	if !slices.Contains([]string{"none", "client_secret_basic", "client_secret_post"}, method) {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_client_metadata"})
		return
	}
	c := &client{id: synthetic("client"), methods: []string{method}, redirects: request.RedirectURIs}
	if method != "none" {
		c.secret = synthetic("secret")
	}
	s.mu.Lock()
	s.clients[c.id] = c
	issued := s.now().Unix()
	s.mu.Unlock()
	body := map[string]any{
		"client_id": c.id, "client_id_issued_at": issued,
		"redirect_uris": c.redirects, "token_endpoint_auth_method": method,
	}
	if c.secret != "" {
		body["client_secret"] = c.secret
	}
	// RFC 7591 §3.2.1: 201 Created.
	writeJSON(w, http.StatusCreated, body)
}

// authorize approves at once, as a user who clicks Allow, unless ConsentDenied is on.
func (s *Server) authorize(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	clientID := q.Get("client_id")
	s.mu.Lock()
	fetch := s.is(ClientMetadataDocuments) && s.clients[clientID] == nil && strings.HasPrefix(clientID, "https://")
	metadataClient := s.metadataClient
	s.mu.Unlock()
	if fetch {
		// Fetched without the lock: whatever serves the document may be slow.
		c, err := fetchClientMetadata(metadataClient, clientID)
		if err != nil {
			// CIMD §5.1: a failed fetch aborts the request; RFC 6749 §4.1.2.1: an invalid
			// client is not redirected back.
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		s.mu.Lock()
		s.clients[c.id] = c
		s.mu.Unlock()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	c := s.clients[clientID]
	redirectURI := q.Get("redirect_uri")
	// RFC 6749 §4.1.2.1: with an unknown client or redirect URI the server «MUST NOT
	// automatically redirect the user-agent».
	if c == nil || !slices.Contains(c.redirects, redirectURI) {
		http.Error(w, "unknown client or redirect_uri", http.StatusBadRequest)
		return
	}
	back := url.Values{}
	if state := q.Get("state"); state != "" {
		back.Set("state", state)
	}
	// RFC 9207 §2: iss in every authorization response, error responses included.
	back.Set("iss", s.URL)
	if s.is(ForeignIssuer) {
		back.Set("iss", ForeignIssuerURL)
	}
	fail := func(code string) {
		back.Set("error", code)
		redirect(w, redirectURI, back)
	}
	separator := " " // RFC 6749 §3.3: «space-delimited».
	if s.is(CommaScopes) {
		// Slack: «user scopes should be added as user_scope=…», comma-separated
		// (docs.slack.dev/authentication/installing-with-oauth).
		separator = ","
	}
	scopes := split(q.Get("scope"), separator)
	userScopes := split(q.Get("user_scope"), separator)
	switch {
	case q.Get("response_type") != "code":
		fail("unsupported_response_type") // RFC 6749 §4.1.2.1
	case q.Get("code_challenge") == "":
		fail("invalid_request") // RFC 7636 §4.4.1: code challenge required
	case q.Get("code_challenge_method") != "S256":
		fail("invalid_request") // RFC 7636 §4.4.1: transform not supported
	case q.Get("resource") != "" && q.Get("resource") != s.resource():
		fail("invalid_target") // RFC 8707 §2
	case s.is(CommaScopes) && slices.ContainsFunc(slices.Concat(scopes, userScopes), func(v string) bool { return strings.Contains(v, " ") }):
		// A space-separated list sent to a comma-separated provider is one malformed scope.
		// What Slack itself shows for it is unverified; this is RFC 6749 §4.1.2.1.
		fail("invalid_scope")
	case s.is(ConsentDenied):
		fail("access_denied") // RFC 6749 §4.1.2.1
	default:
		code := synthetic("code")
		s.codes[code] = &authorizationCode{
			clientID: c.id, redirectURI: redirectURI, challenge: q.Get("code_challenge"),
			scopes: scopes, userScopes: userScopes, claims: q.Get("claims"),
			expires: s.now().Add(codeTTL),
		}
		back.Set("code", code)
		if s.is(CallbackRealmID) {
			back.Set("realmId", s.RealmID)
		}
		if s.is(SignedCallback) {
			// Shopify's callback carries shop, host and timestamp beside code and state
			// (shopify.dev «Authorization code grant»). host is opaque to the signature, so
			// it is synthetic here.
			back.Set("shop", s.Shop)
			back.Set("host", base64.RawURLEncoding.EncodeToString([]byte(s.Shop+"/admin")))
			back.Set("timestamp", strconv.FormatInt(s.now().Unix(), 10))
			back.Set("hmac", Sign(back, c.secret))
		}
		redirect(w, redirectURI, back)
	}
}

// maxClientMetadataBytes is CIMD §8.7: «The recommended maximum size to read is 5
// kilobytes».
const maxClientMetadataBytes = 5 * 1024

// fetchClientMetadata reads the client metadata document at clientID and turns it into a
// public client, or says why it cannot.
func fetchClientMetadata(fetcher *http.Client, clientID string) (*client, error) {
	if fetcher == nil {
		return nil, errors.New("fakeprovider: ClientMetadataDocuments needs FetchClientMetadataWith")
	}
	response, err := fetcher.Get(clientID)
	if err != nil {
		return nil, fmt.Errorf("fakeprovider: client metadata: %w", err)
	}
	defer response.Body.Close()
	// CIMD §5: «MUST be served with a 200 OK»; any other status is an error.
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fakeprovider: client metadata answered %d", response.StatusCode)
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, maxClientMetadataBytes+1))
	if err != nil || len(raw) > maxClientMetadataBytes {
		return nil, errors.New("fakeprovider: client metadata unreadable or over 5 KB")
	}
	var document struct {
		ClientID     string   `json:"client_id"`
		ClientName   string   `json:"client_name"`
		RedirectURIs []string `json:"redirect_uris"`
		Method       string   `json:"token_endpoint_auth_method"`
		Secret       *string  `json:"client_secret"`
	}
	if err := json.Unmarshal(raw, &document); err != nil {
		return nil, fmt.Errorf("fakeprovider: client metadata: %w", err)
	}
	switch {
	case document.ClientID != clientID:
		// CIMD §4: client_id matches the URL it was fetched from, by simple string comparison.
		return nil, errors.New("fakeprovider: client metadata names another client_id")
	case document.Secret != nil || (document.Method != "" && document.Method != "none"):
		// CIMD §4.1: no client_secret and no method based on a shared secret. The key-based
		// methods §4.1 still allows, such as private_key_jwt, this server does not offer.
		return nil, errors.New("fakeprovider: client metadata asks for a shared secret")
	case document.ClientName == "" || len(document.RedirectURIs) == 0:
		// What MCP 2025-11-25 («Client ID Metadata Documents») requires beside client_id.
		return nil, errors.New("fakeprovider: client metadata lacks client_name or redirect_uris")
	}
	// CIMD §4.2: the document's redirect_uris are the registered ones.
	return &client{id: clientID, methods: []string{"none"}, redirects: document.RedirectURIs}, nil
}

// Sign is SignedCallback's signature, as shopify.dev «Authorization code grant» specifies
// it: drop hmac, sort the rest by key, join them as key=value with &, HMAC-SHA256 with the
// client secret, hex. A test or a verifier recomputes it to check a callback.
func Sign(query url.Values, secret string) string {
	keys := make([]string, 0, len(query))
	for k := range query {
		if k != "hmac" {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	pairs := make([]string, 0, len(keys))
	for _, k := range keys {
		pairs = append(pairs, k+"="+query.Get(k))
	}
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(strings.Join(pairs, "&")))
	return hex.EncodeToString(mac.Sum(nil))
}

func (s *Server) token(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		s.tokenError(w, http.StatusBadRequest, "invalid_request", "")
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if r.PostForm.Get("grant_type") == "refresh_token" {
		s.refreshes++
		_, s.refreshScopeSent = r.PostForm["scope"]
		s.refreshScope = r.PostForm.Get("scope")
		if s.is(RateLimited) {
			// RFC 9110 §10.2.3: delay-seconds.
			w.Header().Set("Retry-After", strconv.Itoa(int(RetryAfter.Seconds())))
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		if s.is(CutOffRefusal) {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Content-Length", "64")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":`))
			w.(http.Flusher).Flush()
			// net/http closes the connection for this panic value, 55 bytes short.
			panic(http.ErrAbortHandler)
		}
	}
	if r.PostForm.Get("grant_type") == "client_credentials" {
		s.clientCredentialsGrants++
	}
	if s.is(Unavailable) {
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}
	c, basic := s.authenticate(r)
	if c == nil {
		// RFC 6749 §5.2: invalid_client; 401 with WWW-Authenticate when the client used the
		// Authorization header.
		if basic {
			w.Header().Set("WWW-Authenticate", `Basic realm="fakeprovider"`)
		}
		// Slack lists invalid_client_id and bad_client_secret; which one a wrong secret
		// gets is unverified, so both cases answer invalid_client_id.
		s.tokenError(w, http.StatusUnauthorized, "invalid_client", "invalid_client_id")
		return
	}
	switch r.PostForm.Get("grant_type") {
	case "authorization_code":
		s.exchange(w, r, c)
	case "refresh_token":
		s.refreshGrant(w, r, c)
	case "client_credentials":
		if !s.is(ClientCredentials) {
			s.tokenError(w, http.StatusBadRequest, "unsupported_grant_type", "invalid_grant_type")
			return
		}
		s.clientCredentialsGrant(w, r, c)
	default:
		// RFC 6749 §5.2; Slack names it invalid_grant_type (oauth.v2.access errors).
		s.tokenError(w, http.StatusBadRequest, "unsupported_grant_type", "invalid_grant_type")
	}
}

// authenticate finds the client a token request comes from (RFC 6749 §2.3.1 for a secret in
// the Authorization header or the body, §3.2.1 for a public client naming itself). basic
// reports whether the Authorization header was used. Call it with mu held.
func (s *Server) authenticate(r *http.Request) (c *client, basic bool) {
	id, secret, basic := r.BasicAuth()
	method := "client_secret_basic"
	if !basic {
		id, secret = r.PostForm.Get("client_id"), r.PostForm.Get("client_secret")
		method = "client_secret_post"
		if secret == "" {
			method = "none"
		}
	}
	c = s.clients[id]
	if c == nil || !slices.Contains(c.methods, method) ||
		subtle.ConstantTimeCompare([]byte(c.secret), []byte(secret)) != 1 {
		return nil, basic
	}
	return c, basic
}

// exchange is the authorization_code grant, RFC 6749 §4.1.3. Call it with mu held.
func (s *Server) exchange(w http.ResponseWriter, r *http.Request, c *client) {
	form := r.PostForm
	code := s.codes[form.Get("code")]
	if code == nil {
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_code")
		return
	}
	// Checked before expiry, so a code presented again after codeTTL still ends its grant.
	if code.used {
		// RFC 6749 §4.1.2: a code used twice is refused, and the server «SHOULD revoke
		// (when possible) all tokens previously issued based on that authorization code».
		code.grant.revoked = true
		if code.userGrant != nil {
			code.userGrant.revoked = true
		}
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_code")
		return
	}
	if s.now().After(code.expires) {
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_code")
		return
	}
	// RFC 6749 §4.1.3: the code is bound to the client and the redirect URI. RFC 7636 §4.6:
	// BASE64URL(SHA256(code_verifier)) must equal the challenge, or invalid_grant. The Slack
	// names are from the oauth.v2.access error list: bad_redirect_uri «did not match the
	// redirect_uri in the original request», invalid_code_verifier «The code_verifier is
	// invalid». A code bound to another client keeps invalid_code, which is unverified.
	digest := sha256.Sum256([]byte(form.Get("code_verifier")))
	switch {
	case code.clientID != c.id:
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_code")
		return
	case form.Get("redirect_uri") != code.redirectURI:
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "bad_redirect_uri")
		return
	case base64.RawURLEncoding.EncodeToString(digest[:]) != code.challenge:
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_code_verifier")
		return
	}
	if form.Get("resource") != "" && form.Get("resource") != s.resource() {
		s.tokenError(w, http.StatusBadRequest, "invalid_target", "") // RFC 8707 §2.2
		return
	}
	code.used = true
	code.grant = &grant{clientID: c.id, scopes: code.scopes, claims: code.claims, account: s.account}
	body := s.shape(s.issue(code.grant, !s.is(NoRefreshToken)), code.grant)
	if s.is(CommaScopes) && len(code.userScopes) > 0 {
		// The user token is a grant of its own, so it refreshes on its own refresh token.
		code.userGrant = &grant{clientID: c.id, scopes: code.userScopes, user: true}
		issued := s.issue(code.userGrant, !s.is(NoRefreshToken))
		// The user token in authed_user carries expires_in and refresh_token beside it
		// (docs.slack.dev/authentication/using-token-rotation, «If you make use of a user
		// token»).
		user := body["authed_user"].(map[string]any)
		user["access_token"] = issued["access_token"]
		user["scope"] = strings.Join(code.userScopes, ",")
		user["token_type"] = "user"
		user["expires_in"] = issued["expires_in"]
		if rt, ok := issued["refresh_token"]; ok {
			user["refresh_token"] = rt
		}
	}
	writeToken(w, body)
}

// refreshGrant is RFC 6749 §6. Call it with mu held.
func (s *Server) refreshGrant(w http.ResponseWriter, r *http.Request, c *client) {
	if s.is(InvalidGrant) {
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_refresh_token")
		return
	}
	presented := r.PostForm.Get("refresh_token")
	rt := s.refresh[presented]
	if rt == nil || rt.grant.revoked || rt.grant.clientID != c.id {
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_refresh_token")
		return
	}
	if !rt.rotatedAt.IsZero() {
		if !s.is(RotatingRefreshWithGrace) {
			// RFC 9700 §4.14.2: a rotated refresh token presented again means one of two
			// parties holds a stolen copy, so the server «will revoke the active refresh token».
			rt.grant.revoked = true
			s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_refresh_token")
			return
		}
		if !s.now().Before(rt.rotatedAt.Add(Grace)) {
			// Past the window the old token has only expired. QuickBooks: «previous refresh
			// tokens expire 24 hours after you receive a new one» (oauth-jsclient README);
			// Slack: «the refresh token you used is revoked after a short grace period»
			// (docs.slack.dev/authentication/using-token-rotation). Neither ends the grant, so
			// the token the rotation issued keeps working.
			s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_refresh_token")
			return
		}
	}
	if requested, sent := r.PostForm["scope"]; sent {
		separator := " " // RFC 6749 §3.3
		if s.is(CommaScopes) {
			separator = ","
		}
		// RFC 6749 §6: the scope «MUST NOT include any scope not originally granted by the
		// resource owner»; §5.2 names the error invalid_scope.
		for _, scope := range split(requested[0], separator) {
			if !slices.Contains(rt.grant.scopes, scope) {
				s.tokenError(w, http.StatusBadRequest, "invalid_scope", "")
				return
			}
		}
	}
	body := s.shape(s.issue(rt.grant, !s.is(NonRotatingRefresh)), rt.grant)
	if s.is(ServerError) {
		// The rotation is committed and the answer says only that something failed.
		s.tokenError(w, http.StatusInternalServerError, "server_error", "internal_error")
		return
	}
	if s.is(LostResponse) || (s.is(LostResponseOnce) && !s.lostOnce) {
		s.lostOnce = true
		// The rotation is committed; the answer never leaves. net/http closes the connection
		// without a response for this panic value.
		panic(http.ErrAbortHandler)
	}
	writeToken(w, body)
}

// clientCredentialsGrant is RFC 6749 §4.4. Each one is a grant of its own, so revoking its
// access token (RFC 7009) ends that token and no other. Call it with mu held.
func (s *Server) clientCredentialsGrant(w http.ResponseWriter, r *http.Request, c *client) {
	if c.secret == "" {
		// §4.4: «The client credentials grant type MUST only be used by confidential
		// clients»; §5.2 names the refusal unauthorized_client.
		s.tokenError(w, http.StatusBadRequest, "unauthorized_client", "")
		return
	}
	if resource := r.PostForm.Get("resource"); resource != "" && resource != s.resource() {
		s.tokenError(w, http.StatusBadRequest, "invalid_target", "") // RFC 8707 §2.2
		return
	}
	// §4.4.2: scope is OPTIONAL. The fake has no client policy to narrow it by, so what is
	// asked for is granted.
	g := &grant{clientID: c.id, scopes: split(r.PostForm.Get("scope"), " "), account: s.account}
	writeToken(w, s.shape(s.issue(g, false), g))
}

// issue hands out an access token for g and, when rotate is set, a new refresh token that
// retires the current one. Call it with mu held.
func (s *Server) issue(g *grant, rotate bool) map[string]any {
	ttl := AccessTTL
	if s.is(CommaScopes) {
		ttl = SlackAccessTTL
	}
	access := synthetic("access")
	s.access[access] = &accessToken{grant: g, scopes: g.scopes, expires: s.now().Add(ttl)}
	// RFC 6749 §5.1.
	body := map[string]any{"access_token": access, "token_type": "Bearer", "expires_in": int(ttl.Seconds())}
	if len(g.scopes) > 0 {
		body["scope"] = strings.Join(g.scopes, " ")
	}
	if rotate {
		if old := s.refresh[g.current]; old != nil {
			old.rotatedAt = s.now()
		}
		g.current = synthetic("refresh")
		s.refresh[g.current] = &refreshToken{grant: g}
		body["refresh_token"] = g.current
	}
	return body
}

// shape adds what a vendor-shaped personality puts in every token response, the code
// exchange and a refresh alike. Call it with mu held.
func (s *Server) shape(body map[string]any, g *grant) map[string]any {
	if s.is(CommaScopes) {
		body = s.slackShape(body, g)
	}
	if s.is(IdentityURL) {
		body["id"] = s.IdentityURL
	}
	if s.is(NoExpiresIn) {
		delete(body, "expires_in")
	}
	if s.is(CallbackRealmID) {
		// The refresh token's lifetime in seconds. oauth-jsclient README lists
		// x_refresh_token_expires_in in its token object, the shape every token response
		// fills; the value is the README's example response.
		body["x_refresh_token_expires_in"] = 8726400
	}
	return body
}

// slackShape rewrites a token response as oauth.v2.access answers
// (docs.slack.dev/reference/methods/oauth.v2.access), which is also the refresh call
// (docs.slack.dev/authentication/using-token-rotation, «Refresh a token»): ok, a bot token
// with comma scopes, team, enterprise and authed_user. A refresh of a user token answers
// token_type user with its comma scopes; what else Slack puts beside them is unverified.
func (s *Server) slackShape(body map[string]any, g *grant) map[string]any {
	body["ok"] = true
	body["scope"] = strings.Join(g.scopes, ",")
	if g.user {
		body["token_type"] = "user"
		return body
	}
	body["token_type"] = "bot"
	body["bot_user_id"] = "U0000BOT"
	body["app_id"] = "A0000APP"
	body["team"] = map[string]string{"name": "Fake", "id": s.TeamID}
	body["enterprise"] = nil
	// On the exchange authed_user is oauth.v2.access's; that a refresh repeats it is
	// unverified, and it is kept so a manifest's capture reads the same on both.
	body["authed_user"] = map[string]any{"id": g.account}
	return body
}

// revoke is RFC 7009. Revoking any token of a grant ends the whole grant (§2.1 lets a server
// do so), and the answer is 200 whether the token was known or not (§2.2).
func (s *Server) revoke(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if c, _ := s.authenticate(r); c == nil {
		s.tokenError(w, http.StatusUnauthorized, "invalid_client", "")
		return
	}
	token := r.PostForm.Get("token")
	if s.is(AccessTokenNotRevocable) && s.access[token] != nil {
		s.tokenError(w, http.StatusBadRequest, "unsupported_token_type", "")
		return
	}
	if a := s.access[token]; a != nil {
		a.grant.revoked = true
	}
	if rt := s.refresh[token]; rt != nil {
		rt.grant.revoked = true
	}
	w.WriteHeader(http.StatusOK)
}

// tokenError answers a token request with an RFC 6749 §5.2 error, or under CommaScopes as
// Slack does: «The Web API only responds with 200 (yes, even for errors) or 429»
// (docs.slack.dev/tools/node-slack-sdk/web-api) with {"ok":false,"error":…}. slackCode is
// the Slack name from the oauth.v2.access error list; empty passes the RFC code through,
// which is unverified for Slack.
func (s *Server) tokenError(w http.ResponseWriter, status int, code, slackCode string) {
	if s.is(CommaScopes) {
		if slackCode == "" {
			slackCode = code
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": false, "error": slackCode})
		return
	}
	writeJSON(w, status, map[string]string{"error": code})
}

// writeToken adds the headers RFC 6749 §5.1 requires on a token response.
func writeToken(w http.ResponseWriter, body map[string]any) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Pragma", "no-cache")
	writeJSON(w, http.StatusOK, body)
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// redirect sends the browser to redirectURI with values added to its query, which RFC 6749
// §3.1.2 says must be kept.
func redirect(w http.ResponseWriter, redirectURI string, values url.Values) {
	target, _ := url.Parse(redirectURI)
	query := target.Query()
	for k, v := range values {
		query[k] = v
	}
	target.RawQuery = query.Encode()
	w.Header().Set("Location", target.String())
	w.WriteHeader(http.StatusFound)
}

func split(value, separator string) []string {
	if value == "" {
		return nil
	}
	return strings.Split(value, separator)
}
