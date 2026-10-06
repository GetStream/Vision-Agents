package api

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The browser routes of a consent. The launch path and the callback path are the prototype's
// (connectorOAuthLaunchPath in internal/api/connectors.go and CallbackPath in
// internal/mcp/oauth.go:119 on codex/connector-support at cf62af0d), so a redirect URI
// registered with a provider for the prototype still matches.
const (
	// ConnectorCallbackPath is where a provider sends the browser back to: the redirect URI
	// is ROUTER_PUBLIC_URL followed by it.
	ConnectorCallbackPath = "/v1/agents/connectors/oauth/callback"
	// ConnectorClientMetadataPath is where the router's OAuth Client ID Metadata Document is
	// served. CIMD (draft-ietf-oauth-client-id-metadata-document-02 section 3) lets the
	// client choose any https URL with a path; this one is the prototype's ClientMetadataPath
	// (internal/mcp/oauth.go:122 at cf62af0d). It is not a registered well-known URI
	// (RFC 8615 section 3.1), which is unverified as a problem for any server.
	ConnectorClientMetadataPath = "/.well-known/oauth-client-metadata"
	connectorLaunchPath         = "/v1/agents/connectors/oauth/launch/"
)

const (
	// attemptLifetime is how long a consent may take, from authorize to callback. It is RFC
	// 6749 section 4.1.2's recommended maximum authorization code lifetime (10 minutes), the
	// prototype's connectorAuthorizationLifetime and oauth2code's attemptTTL, so no attempt
	// outlives the code it is waiting for and the scheme's own expiry agrees with the row's.
	attemptLifetime = 10 * time.Minute
	// handoffBytes is the size of the handoff token, which is also the browser cookie's
	// value. 32 random octets, as oauth2code's state: a guess succeeds with probability
	// 2^-256, below the 2^-128 RFC 6749 section 10.10 requires and the 2^-160 it recommends
	// for a token an attacker must not guess.
	handoffBytes = 32
	// nonceBytes is the launch page's CSP nonce: 128 bits, the minimum CSP Level 3 section
	// 7.1 («Nonce Reuse») says a nonce SHOULD have, fresh for every response.
	nonceBytes = 16
	// maxHandoffBody bounds the handoff's JSON, which holds one 43-character token. 4096 is
	// the prototype's io.LimitReader in connectorOAuthLaunchHandoff; not measured.
	maxHandoffBody = 4096
	// attemptCookiePrefix names the cookie that binds an attempt to the browser that began
	// it, one cookie per attempt so two consents in one browser do not overwrite each other.
	// The prototype's name (connectorAuthorizationCookieName at cf62af0d).
	attemptCookiePrefix = "va_connector_oauth_"
	// clientMetadataMaxAge is how long an authorization server may cache the client metadata
	// document; CIMD section 5.2 has it follow HTTP cache headers (RFC 9111). 300 seconds is
	// the prototype's (connectorOAuthClientMetadataHandler at cf62af0d); unverified, not
	// measured.
	clientMetadataMaxAge = "public, max-age=300"
	// accountSwitchError is the connection's last_error after a consent for another account.
	// The prototype's wording, shortened (finishConnectorLogin at cf62af0d).
	accountSwitchError = "A consent returned a different provider account; the existing grant was kept. " +
		"Create a new connection for that account."
)

// Where a consent ended, sent to the dashboard as the status query parameter.
const (
	consentConnected       = "connected"
	consentDenied          = "denied"
	consentFailed          = "failed"
	consentAccountMismatch = "account_mismatch"
)

// noAttempt is the one answer for a callback or a handoff whose attempt cannot be used:
// unknown, expired, already used, sealed for another row, or for a deleted connection. One
// answer, as store.ErrNoAuthorizationAttempt is one error, so a probe learns nothing.
const noAttempt = "this consent is unknown, expired or already finished: start it again"

// AuthorizationKind is why an attempt was started.
type AuthorizationKind string

func (AuthorizationKind) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "AuthorizationKind",
		"consent for a connection no account was connected to yet, reconnect for one that has "+
			"been connected before, which must come back with the same provider account.",
		store.AttemptConsent, store.AttemptReconnect)
}

// Authorization is a consent in flight, as the backend that started it is shown it.
type Authorization struct {
	ID           string            `json:"id" doc:"The attempt."`
	Kind         AuthorizationKind `json:"kind"`
	LaunchURL    string            `json:"launch_url" doc:"The router's page to open in a popup from the dashboard. It asks the opener for handoff_token and then sends the browser to the provider."`
	HandoffToken string            `json:"handoff_token" doc:"Handed to the launch page by postMessage, never put in a URL. It binds the attempt to the browser that opens launch_url."`
	ExpiresAt    time.Time         `json:"expires_at" doc:"When the attempt ends, 10 minutes after it began. A callback after that is refused."`
}

func (*Authorization) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A consent in flight for one connection: the page that starts it in a " +
		"browser and the token that binds it to that browser."
	return schema
}

type createAuthorizationRequest struct {
	ID string `path:"id" doc:"The connection, as returned when it was created."`
}

type authorizationResponse struct {
	Body Authorization
}

// attempt is what an authorization attempt seals: everything the handoff and the callback
// need, so neither trusts a value the browser sends beyond the state and the cookie.
type attempt struct {
	ConnectorID        string `json:"connector_id"`
	DefinitionRevision int    `json:"definition_revision"`
	Scheme             string `json:"scheme"`
	// Handoff is the handoff token and the cookie value.
	Handoff      string `json:"handoff"`
	AuthorizeURL string `json:"authorize_url"`
	// State is the scheme's BeginOutput.State, handed back to Complete.
	State json.RawMessage `json:"scheme_state"`
}

// registerAuthorizations declares the one Huma operation of the consent flow. It is
// server-side only, like every connection operation: a user-owned connection's consent is
// started by the backend for the user it acts for (architecture doc, one-way door 7), and
// the end user's browser only ever sees the launch page that backend hands it.
func (s *Server) registerAuthorizations(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID:   "createAuthorization",
		Method:        http.MethodPost,
		Path:          "/v1/agents/connections/{id}/authorizations",
		Summary:       "Start a consent",
		DefaultStatus: http.StatusCreated,
		Description: "Starts the provider's consent for a connection: a consent for a pending one, " +
			"a reconnect for one connected before. Open launch_url in a popup from the dashboard " +
			"and post it handoff_token when it says it is ready; the browser then goes to the " +
			"provider and comes back to the router, which stores the grant and sends the browser " +
			"to the dashboard with connection_id and status (connected, denied, failed or " +
			"account_mismatch). A reconnect that comes back with another provider account keeps " +
			"the old grant. Who may start it is who may read the connection. Needs " +
			"ROUTER_PUBLIC_URL, where the provider sends the browser back to.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"201": {Description: "The consent is waiting for its browser"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.createAuthorization)
}

// createAuthorization begins a consent for a connection the caller may have.
func (s *Server) createAuthorization(ctx context.Context, request *createAuthorizationRequest) (*authorizationResponse, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	if s.connectorSecrets == nil || s.credentials == nil {
		return nil, huma.Error400BadRequest("consents cannot be started: connectors are not enabled on this deployment")
	}
	// Checked here, not at the callback: a consent the provider cannot send back would
	// otherwise be found out only after the user approved it.
	public := strings.TrimRight(s.publicURL, "/")
	if public == "" {
		return nil, huma.Error400BadRequest("consents cannot be started: ROUTER_PUBLIC_URL is not set, " +
			"so the provider has nowhere to send the browser back to")
	}
	scheme, found := s.connectors.Schemes[connection.AuthScheme]
	if !found {
		return nil, huma.Error400BadRequest(fmt.Sprintf("auth_scheme %q is not one this deployment has", connection.AuthScheme))
	}
	manifest, err := s.connectionManifest(ctx, connection, connection.DefinitionRevision)
	if err != nil {
		return nil, err
	}
	ref := core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}
	begun, err := scheme.Begin(ctx, core.BeginInput{Ref: ref, Manifest: manifest, RedirectURI: public + ConnectorCallbackPath})
	if err != nil {
		// A scheme's error carries no secret (core AGENTS.md, «Secrets never print»), and it
		// is what the backend needs to fix: a missing client, an unreachable server.
		return nil, huma.Error400BadRequest("the provider's consent could not be started: " + err.Error())
	}
	if begun.Done {
		return nil, huma.Error400BadRequest(fmt.Sprintf("auth_scheme %q needs no consent", connection.AuthScheme))
	}
	state, err := authorizeState(begun.AuthorizeURL)
	if err != nil {
		return nil, err
	}
	handoff, err := randomToken(handoffBytes)
	if err != nil {
		return nil, err
	}

	kind := store.AttemptReconnect
	if connection.Status == store.ConnectionPending {
		kind = store.AttemptConsent
	}
	id := store.NewID()
	raw, err := json.Marshal(attempt{
		ConnectorID:        connection.ConnectorID,
		DefinitionRevision: connection.DefinitionRevision,
		Scheme:             connection.AuthScheme,
		Handoff:            handoff,
		AuthorizeURL:       begun.AuthorizeURL,
		State:              begun.State,
	})
	if err != nil {
		return nil, err
	}
	sealed, err := s.connectorSecrets.SealWithAAD(string(raw), attemptAAD(connection.CustomerID, connection.ID, id))
	if err != nil {
		return nil, err
	}
	expires := time.Now().UTC().Add(attemptLifetime).Truncate(time.Microsecond)
	err = s.store.CreateConnectorAuthorizationAttempt(ctx, &store.ConnectorAuthorizationAttempt{
		ID:            id,
		CustomerID:    connection.CustomerID,
		ConnectionID:  connection.ID,
		Kind:          kind,
		StateHash:     store.AuthorizationStateHash(state),
		AttemptSealed: sealed,
		KEKVersion:    s.connectorSecrets.CurrentVersion(),
		ExpiresAt:     expires,
	})
	// Deleted between the read above and the insert.
	if errors.Is(err, store.ErrNoConnectorConnection) {
		return nil, huma.Error404NotFound(noSuchConnection)
	}
	if err != nil {
		return nil, err
	}
	return &authorizationResponse{Body: Authorization{
		ID:           id,
		Kind:         AuthorizationKind(kind),
		LaunchURL:    public + connectorLaunchPath + id,
		HandoffToken: handoff,
		ExpiresAt:    expires,
	}}, nil
}

// launchPage is the router-hosted page a consent starts on. It waits for the dashboard that
// opened it to post the handoff token, trades the token for the authorize URL (which sets
// the browser cookie on the way), and goes to the provider. The prototype's page
// (connectorOAuthLaunchPage at cf62af0d), unchanged but for the wording.
var launchPage = template.Must(template.New("connector-launch").Parse(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="referrer" content="no-referrer"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Connecting your account</title></head>
<body><p id="status">Connecting securely to the provider…</p><script nonce="{{.Nonce}}">
const dashboardOrigin = {{.DashboardOrigin}};
const status = document.getElementById('status');
let handoffStarted = false;
window.addEventListener('message', async (event) => {
  if (handoffStarted || event.source !== window.opener || event.origin !== dashboardOrigin || !event.data || event.data.type !== 'va.connector.oauth.handoff' || typeof event.data.handoff_token !== 'string') return;
  handoffStarted = true;
  try {
    const response = await fetch(window.location.pathname, { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ handoff_token: event.data.handoff_token }) });
    if (!response.ok) throw new Error('This consent has expired. Start again from the dashboard.');
    const authorization = await response.json();
    window.opener = null;
    window.location.replace(authorization.authorization_url);
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : 'Could not reach the provider.';
  }
});
if (window.opener) window.opener.postMessage({ type: 'va.connector.oauth.ready' }, dashboardOrigin);
else status.textContent = 'Start from the dashboard, so this browser can be checked.';
</script></body></html>`))

// launchPolicy is the launch page's Content-Security-Policy (CSP Level 3): nothing loads but
// the one script carrying the nonce, it may fetch only its own origin (the handoff), no page
// may frame it (frame-ancestors, section 6.4.2), and it has no base URL or form to redirect.
// The prototype's policy (connectorOAuthLaunchPageHandler at cf62af0d).
const launchPolicy = "default-src 'none'; script-src 'nonce-%s'; connect-src 'self'; " +
	"frame-ancestors 'none'; base-uri 'none'; form-action 'none'"

// serveConnectorLaunch renders the launch page. It names no attempt and checks none: the
// handoff that follows is what is checked, so the page alone gives nothing away.
func (s *Server) serveConnectorLaunch(w http.ResponseWriter, _ *http.Request) {
	dashboard, err := originOf(s.dashboardURL)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "consents cannot be launched: DASHBOARD_BASE_URL is not an http(s) URL")
		return
	}
	nonce, err := randomToken(nonceBytes)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "could not start the consent")
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Security-Policy", fmt.Sprintf(launchPolicy, nonce))
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	// For a browser without frame-ancestors (CSP Level 3 section 6.4.2.2).
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := launchPage.Execute(w, struct{ Nonce, DashboardOrigin string }{nonce, dashboard}); err != nil {
		s.logger.Error("could not render the connector launch page", "error", err)
	}
}

// handOffConnectorLaunch trades the handoff token for the authorize URL and binds the
// attempt to this browser with a cookie, which the callback then requires.
//
// Origin must be the router's own: the launch page's fetch sends it on a POST (Fetch
// Standard, «append a request Origin header»), so a form or a script on another site that
// learned a handoff token cannot make the router set the cookie in its own browser.
func (s *Server) handOffConnectorLaunch(w http.ResponseWriter, r *http.Request) {
	if s.store == nil || s.connectorSecrets == nil {
		writeError(w, http.StatusBadRequest, noAttempt)
		return
	}
	public, err := originOf(s.publicURL)
	if err != nil || r.Header.Get("Origin") != public {
		writeError(w, http.StatusForbidden, "the handoff must come from the launch page")
		return
	}
	var body struct {
		HandoffToken string `json:"handoff_token"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, maxHandoffBody)).Decode(&body); err != nil || body.HandoffToken == "" {
		writeError(w, http.StatusBadRequest, "the handoff needs a handoff_token")
		return
	}
	row, sealed, err := s.openAttemptByID(r.Context(), r.PathValue("id"))
	if errors.Is(err, store.ErrNoAuthorizationAttempt) {
		writeError(w, http.StatusBadRequest, noAttempt)
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "could not read the consent")
		return
	}
	if subtle.ConstantTimeCompare([]byte(body.HandoffToken), []byte(sealed.Handoff)) != 1 {
		writeError(w, http.StatusForbidden, "that is not this consent's handoff token")
		return
	}
	http.SetCookie(w, s.attemptCookie(row.ID, sealed.Handoff, row.ExpiresAt))
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(struct {
		AuthorizationURL string `json:"authorization_url"`
	}{sealed.AuthorizeURL})
}

// finishConnectorConsent is the redirect URI. Its order is what makes it safe:
//
//  1. the state names an open attempt, read but not used up;
//  2. the browser holds that attempt's cookie (RFC 6749 section 10.12, RFC 9700 section
//     2.1: the state is bound to the user agent), so a callback in another browser is
//     refused and the right browser can still finish;
//  3. the attempt is used up, once (store.ConsumeConnectorAuthorizationAttempt), so a
//     replayed callback never reaches the provider again;
//  4. the manifest's before_complete hook sees the whole callback;
//  5. Scheme.Complete checks state, iss (RFC 9207 section 2.4) and the provider's error
//     before it exchanges the code;
//  6. under the connection's lock, a consent for another account than the connection has
//     keeps the old grant and says so (architecture doc, one-way door 1).
func (s *Server) finishConnectorConsent(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	query := r.URL.Query()
	states := query["state"]
	if len(states) != 1 || states[0] == "" {
		writeError(w, http.StatusBadRequest, "the callback needs exactly one state")
		return
	}
	if s.store == nil || s.connectorSecrets == nil || s.credentials == nil {
		writeError(w, http.StatusBadRequest, noAttempt)
		return
	}
	row, sealed, err := s.openAttemptByState(ctx, states[0])
	if errors.Is(err, store.ErrNoAuthorizationAttempt) {
		writeError(w, http.StatusBadRequest, noAttempt)
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "could not read the consent")
		return
	}
	cookie, err := r.Cookie(attemptCookiePrefix + row.ID)
	if err != nil || subtle.ConstantTimeCompare([]byte(cookie.Value), []byte(sealed.Handoff)) != 1 {
		writeError(w, http.StatusForbidden, "finish the consent in the browser that started it")
		return
	}
	consumed, err := s.store.ConsumeConnectorAuthorizationAttempt(ctx, states[0])
	if errors.Is(err, store.ErrNoAuthorizationAttempt) || err == nil && consumed.ID != row.ID {
		writeError(w, http.StatusBadRequest, noAttempt)
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, "could not read the consent")
		return
	}
	http.SetCookie(w, s.attemptCookie(row.ID, "", time.Time{}))

	outcome := s.completeConsent(ctx, row, sealed, query)
	destination, err := url.Parse(s.dashboardURL)
	if err != nil || s.dashboardURL == "" {
		writeError(w, http.StatusInternalServerError, "the consent ended as "+outcome+", and DASHBOARD_BASE_URL is not set to go back to")
		return
	}
	back := destination.Query()
	back.Set("connection_id", row.ConnectionID)
	back.Set("status", outcome)
	destination.RawQuery = back.Encode()
	http.Redirect(w, r, destination.String(), http.StatusFound)
}

// completeConsent exchanges the callback through the scheme and stores what it got, and
// says how the consent ended.
func (s *Server) completeConsent(ctx context.Context, row store.ConnectorAuthorizationAttempt, sealed attempt, query url.Values) string {
	ref := core.ConnectionRef{CustomerID: row.CustomerID, ConnectionID: row.ConnectionID}
	scheme, found := s.connectors.Schemes[sealed.Scheme]
	if !found {
		s.logger.Error("a consent finished for a scheme this deployment no longer has", "scheme", sealed.Scheme)
		return consentFailed
	}
	connection, err := s.store.ConnectorConnection(ctx, row.CustomerID, row.ConnectionID)
	if err != nil {
		return consentFailed
	}
	manifest, err := s.connectionManifest(ctx, connection, sealed.DefinitionRevision)
	if err != nil {
		s.logger.Error("could not resolve a connection's manifest for its consent", "connection", row.ConnectionID, "error", err)
		return consentFailed
	}
	if err := s.beforeComplete(ctx, manifest, query); err != nil {
		s.logger.Info("a before_complete hook refused a consent", "connection", row.ConnectionID, "error", err)
		return consentFailed
	}
	credentials, account, err := scheme.Complete(ctx, core.CompleteInput{Ref: ref, Manifest: manifest, State: sealed.State, Query: query})
	var refused *oauth2code.AuthorizationError
	if errors.As(err, &refused) && refused.Code == "access_denied" {
		// RFC 6749 section 4.1.2.1: «The resource owner or authorization server denied the
		// request». Nothing is written: the connection stays as it was.
		return consentDenied
	}
	if err != nil {
		s.logger.Info("a consent did not complete", "connection", row.ConnectionID, "error", err)
		return consentFailed
	}

	switched := false
	err = s.credentials.Update(ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		if state.AccountID != "" && state.AccountID != account.AccountID {
			switched = true
			state.LastError = accountSwitchError
			return true, nil
		}
		state.Credentials = credentials
		state.Status = store.ConnectionConnected
		state.LastError = ""
		state.AccountID = account.AccountID
		state.Metadata = account.Metadata
		state.Scopes = account.Scopes
		// Unknown here: the expiry is inside the scheme's stored credentials, and the
		// resolver (T12) sets it the first time it retrieves an access credential.
		state.ExpiresAt = time.Time{}
		return true, nil
	})
	if err != nil {
		s.logger.Error("could not store a consent's credentials", "connection", row.ConnectionID, "error", err)
		return consentFailed
	}
	if switched {
		return consentAccountMismatch
	}
	return consentConnected
}

// beforeComplete runs the hook the manifest names at before_complete, if any, on a copy of
// the full callback query (core.HookBeforeComplete), such as a signed callback's check. A
// hook the manifest names that is not registered fails the consent rather than skipping a
// check the manifest asked for.
func (s *Server) beforeComplete(ctx context.Context, manifest core.ResolvedManifest, query url.Values) error {
	name, named := manifest.Hooks[core.HookBeforeComplete]
	if !named {
		return nil
	}
	hook, found := s.connectors.Hooks[name]
	if !found {
		return fmt.Errorf("hook %q is not registered", name)
	}
	return hook(ctx, &core.HookContext{Point: core.HookBeforeComplete, Manifest: manifest, Query: cloneValues(query)})
}

// serveConnectorClientMetadata serves the router's OAuth Client ID Metadata Document, which
// a server that supports CIMD fetches at the client_id URL. With no https
// ROUTER_PUBLIC_URL there is no client_id URL (CIMD section 3), so there is no document.
func (s *Server) serveConnectorClientMetadata(w http.ResponseWriter, _ *http.Request) {
	clientID := ConnectorClientMetadataURL(s.publicURL)
	if clientID == "" {
		writeError(w, http.StatusNotFound, "no client metadata document: ROUTER_PUBLIC_URL is not an https URL")
		return
	}
	document := oauth2code.ClientMetadataDocument(clientID,
		[]string{strings.TrimRight(s.publicURL, "/") + ConnectorCallbackPath})
	w.Header().Set("Cache-Control", clientMetadataMaxAge)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	_ = json.NewEncoder(w).Encode(document)
}

// ConnectorClientMetadataURL is the router's client_id under CIMD for publicURL, and empty
// when publicURL is not https, which CIMD section 3 requires of a client_id URL.
func ConnectorClientMetadataURL(publicURL string) string {
	base := strings.TrimRight(publicURL, "/")
	if !strings.HasPrefix(base, "https://") {
		return ""
	}
	return base + ConnectorClientMetadataPath
}

// attemptCookie is the cookie binding an attempt to one browser, or, with no value, the one
// that clears it. Its attributes, from draft-ietf-httpbis-rfc6265bis-22:
//
//   - HttpOnly (section 4.1.2.6): no script reads it, the launch page's included.
//   - Secure (section 4.1.2.5) whenever ROUTER_PUBLIC_URL is https, so it never travels in
//     the clear; a local http router leaves it off, or no browser would send it back.
//   - SameSite=Lax (section 5.6.7.1): sent with the provider's redirect, which is a
//     cross-site top-level navigation with a safe method, and with nothing cross-site
//     else. Strict would not be sent with that redirect, so every consent would fail.
//   - Path is the callback alone (section 4.1.2.4), so no other route ever receives it;
//     the section says a path is not a security boundary, and nothing here relies on it.
//   - Expires and Max-Age (sections 4.1.2.1, 4.1.2.2) end it with the attempt.
func (s *Server) attemptCookie(id, value string, expires time.Time) *http.Cookie {
	cookie := &http.Cookie{
		Name:     attemptCookiePrefix + id,
		Value:    value,
		Path:     ConnectorCallbackPath,
		HttpOnly: true,
		Secure:   strings.HasPrefix(s.publicURL, "https://"),
		SameSite: http.SameSiteLaxMode,
		MaxAge:   -1,
	}
	if value != "" {
		cookie.Expires = expires
		cookie.MaxAge = max(int(time.Until(expires).Seconds()), 1)
	}
	return cookie
}

// connectionManifest is the connection's definition at revision, resolved for its inputs
// and what earlier consents captured.
func (s *Server) connectionManifest(ctx context.Context, connection store.ConnectorConnection, revision int) (core.ResolvedManifest, error) {
	definition, err := s.store.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, revision)
	if err != nil {
		return core.ResolvedManifest{}, err
	}
	return definition.Manifest.Resolve(connection.AuthScheme, connection.Inputs, connection.Metadata)
}

// openAttemptByID and openAttemptByState read an open attempt and open what it sealed. A
// blob that does not open for this row is the same as no attempt.
func (s *Server) openAttemptByID(ctx context.Context, id string) (store.ConnectorAuthorizationAttempt, attempt, error) {
	if id == "" {
		return store.ConnectorAuthorizationAttempt{}, attempt{}, store.ErrNoAuthorizationAttempt
	}
	row, err := s.store.ConnectorAuthorizationAttemptByID(ctx, id)
	return s.openedAttempt(row, err)
}

func (s *Server) openAttemptByState(ctx context.Context, state string) (store.ConnectorAuthorizationAttempt, attempt, error) {
	row, err := s.store.ConnectorAuthorizationAttemptByState(ctx, state)
	return s.openedAttempt(row, err)
}

func (s *Server) openedAttempt(row store.ConnectorAuthorizationAttempt, err error) (store.ConnectorAuthorizationAttempt, attempt, error) {
	if err != nil {
		return row, attempt{}, err
	}
	plain, err := s.connectorSecrets.OpenWithAADVersion(row.AttemptSealed, attemptAAD(row.CustomerID, row.ConnectionID, row.ID), row.KEKVersion)
	var opened attempt
	if err != nil || json.Unmarshal([]byte(plain), &opened) != nil || opened.Handoff == "" {
		return row, attempt{}, store.ErrNoAuthorizationAttempt
	}
	return row, opened, nil
}

// attemptAAD binds a sealed attempt to its tenant, connection and id, so a blob copied onto
// another row does not open. Each part is length-prefixed, as pgsealed's credentialsAAD is,
// so no two triples give the same bytes. v1 changes with the layout.
func attemptAAD(customerID, connectionID, id string) []byte {
	return fmt.Appendf(nil, "accelerate:connector-attempt:v1:%d:%s:%d:%s:%d:%s",
		len(customerID), customerID, len(connectionID), connectionID, len(id), id)
}

// authorizeState is the state parameter of an authorize URL: RFC 6749 section 4.1.1 has the
// client send it there, and section 4.1.2 has the server return it unchanged in the
// callback, which is how the callback finds its attempt.
func authorizeState(authorizeURL string) (string, error) {
	parsed, err := url.Parse(authorizeURL)
	if err != nil {
		return "", fmt.Errorf("api: the scheme's authorize URL does not parse: %w", err)
	}
	state := parsed.Query()["state"]
	if len(state) != 1 || state[0] == "" {
		return "", errors.New("api: the scheme's authorize URL carries no single state")
	}
	return state[0], nil
}

// originOf is scheme://host[:port] of an http(s) URL in the form a browser's Origin header
// has: scheme and host lowercased and the scheme's default port dropped, as the URL
// Standard's parser leaves them.
func originOf(raw string) (string, error) {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.User != nil || parsed.Host == "" {
		return "", errors.New("api: not an http(s) URL with a host")
	}
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "http" && scheme != "https" {
		return "", errors.New("api: not an http(s) URL with a host")
	}
	host := strings.ToLower(parsed.Hostname())
	if strings.Contains(host, ":") {
		host = "[" + host + "]"
	}
	if port := parsed.Port(); port != "" && !(scheme == "https" && port == "443") && !(scheme == "http" && port == "80") {
		host += ":" + port
	}
	return scheme + "://" + host, nil
}

// randomToken is size bytes from crypto/rand, base64url without padding, which is safe in a
// cookie value (RFC 6265bis section 4.1.1, cookie-octet) and in JSON.
func randomToken(size int) (string, error) {
	raw := make([]byte, size)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(raw), nil
}

func cloneValues(values url.Values) url.Values {
	cloned := make(url.Values, len(values))
	for key, value := range values {
		cloned[key] = append([]string(nil), value...)
	}
	return cloned
}
