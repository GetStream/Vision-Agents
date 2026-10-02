package fakeprovider

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
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
	writeJSON(w, http.StatusOK, map[string]any{
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
	})
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
	s.mu.Lock()
	defer s.mu.Unlock()
	c := s.clients[q.Get("client_id")]
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
		fail("invalid_scope") // RFC 6749 §4.1.2.1
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
		s.tokenError(w, http.StatusUnauthorized, "invalid_client", "invalid_client_id")
		return
	}
	switch r.PostForm.Get("grant_type") {
	case "authorization_code":
		s.exchange(w, r, c)
	case "refresh_token":
		s.refreshGrant(w, r, c)
	default:
		s.tokenError(w, http.StatusBadRequest, "unsupported_grant_type", "") // RFC 6749 §5.2
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
	if code == nil || s.now().After(code.expires) {
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_code")
		return
	}
	if code.used {
		// RFC 6749 §4.1.2: a code used twice is refused, and the server «SHOULD revoke
		// (when possible) all tokens previously issued based on that authorization code».
		code.grant.revoked = true
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_code")
		return
	}
	// RFC 6749 §4.1.3: the code is bound to the client and the redirect URI. RFC 7636 §4.6:
	// BASE64URL(SHA256(code_verifier)) must equal the challenge, or invalid_grant. Slack's
	// page does not say which code a PKCE mismatch gets; invalid_code is assumed (unverified).
	digest := sha256.Sum256([]byte(form.Get("code_verifier")))
	if code.clientID != c.id || form.Get("redirect_uri") != code.redirectURI ||
		base64.RawURLEncoding.EncodeToString(digest[:]) != code.challenge {
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_code")
		return
	}
	if form.Get("resource") != "" && form.Get("resource") != s.resource() {
		s.tokenError(w, http.StatusBadRequest, "invalid_target", "") // RFC 8707 §2.2
		return
	}
	code.used = true
	code.grant = &grant{clientID: c.id, scopes: code.scopes, claims: code.claims}
	body := s.issue(code.grant, !s.is(NoRefreshToken))
	if s.is(CommaScopes) {
		body = s.slackShape(body, code)
	}
	if s.is(CallbackRealmID) {
		// The refresh token's lifetime in seconds; the value is oauth-jsclient README's
		// example response.
		body["x_refresh_token_expires_in"] = 8726400
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
	if !rt.rotatedAt.IsZero() && !(s.is(RotatingRefreshWithGrace) && s.now().Before(rt.rotatedAt.Add(Grace))) {
		// RFC 9700 §4.14.2: a rotated refresh token presented again means one of two parties
		// holds a stolen copy, so the server «will revoke the active refresh token».
		rt.grant.revoked = true
		s.tokenError(w, http.StatusBadRequest, "invalid_grant", "invalid_refresh_token")
		return
	}
	body := s.issue(rt.grant, !s.is(NonRotatingRefresh))
	if s.is(LostResponse) {
		// The rotation is committed; the answer never leaves. net/http closes the connection
		// without a response for this panic value.
		panic(http.ErrAbortHandler)
	}
	writeToken(w, body)
}

// issue mints an access token for g and, when rotate is set, a new refresh token that
// retires the current one. Call it with mu held.
func (s *Server) issue(g *grant, rotate bool) map[string]any {
	access := synthetic("access")
	s.access[access] = &accessToken{grant: g, scopes: g.scopes, expires: s.now().Add(AccessTTL)}
	// RFC 6749 §5.1.
	body := map[string]any{"access_token": access, "token_type": "Bearer", "expires_in": int(AccessTTL.Seconds())}
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

// slackShape rewrites a token response as oauth.v2.access answers
// (docs.slack.dev/reference/methods/oauth.v2.access): ok, a bot token with comma scopes,
// team, enterprise and authed_user, which carries a user token when user_scope was asked.
func (s *Server) slackShape(body map[string]any, code *authorizationCode) map[string]any {
	body["ok"] = true
	body["token_type"] = "bot"
	body["scope"] = strings.Join(code.scopes, ",")
	body["bot_user_id"] = "U0000BOT"
	body["app_id"] = "A0000APP"
	body["team"] = map[string]string{"name": "Fake", "id": s.TeamID}
	body["enterprise"] = nil
	user := map[string]any{"id": s.UserID}
	if len(code.userScopes) > 0 {
		token := synthetic("user")
		s.access[token] = &accessToken{grant: code.grant, scopes: code.userScopes, expires: s.now().Add(AccessTTL)}
		user["access_token"] = token
		user["scope"] = strings.Join(code.userScopes, ",")
		user["token_type"] = "user"
	}
	body["authed_user"] = user
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
