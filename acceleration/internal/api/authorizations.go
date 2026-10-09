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
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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
	// handoffBytes is the size of the handoff token and of the browser cookie's value, two
	// separate tokens. 32 random octets, as oauth2code's state: a guess succeeds with
	// probability 2^-256, below the 2^-128 RFC 6749 section 10.10 requires and the 2^-160 it
	// recommends for a token an attacker must not guess.
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

// errNoAttempt is the one answer for a callback or a handoff whose attempt cannot be used:
// unknown, expired, already used, sealed for another row, or for a deleted connection. One
// answer, as store.ErrNoAuthorizationAttempt is one error, so a probe learns nothing.
var errNoAttempt = invalidRequest("this consent is unknown, expired or already finished: start it again")

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
	HandoffToken string            `json:"handoff_token" doc:"Handed to the launch page by postMessage, never put in a URL. It binds the attempt to the first browser that opens launch_url and hands it off; a second handoff is refused."`
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
	// Handoff is the handoff token the backend was answered.
	Handoff string `json:"handoff"`
	// Cookie is the browser cookie's value, chosen at the handoff and empty before it. It is
	// not the handoff token, which the backend and the dashboard's script also hold.
	Cookie       string `json:"cookie,omitempty"`
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
			"the old grant. The consent runs on the connector's latest revision, and the " +
			"connection reads that revision once the consent connects it. Who may start it is who " +
			"may read the connection. Needs " +
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
		return nil, notConfigured("consents cannot be started: connectors are not enabled on this deployment")
	}
	begun, err := s.consents().begin(ctx, connection, nil)
	if err != nil {
		return nil, err
	}
	return &authorizationResponse{Body: begun}, nil
}

// consents begins consents for the backend that asks for one (createAuthorization) and for a
// session whose end user's own connection a tool call needs (ConnectorConsents). One path, so
// a consent begun from the chat is the same single-use attempt, bound to the first browser
// that hands it off, as one the backend begins.
type consents struct {
	store     *store.Store
	registry  core.Registry
	secrets   *auth.Sealer
	publicURL string
}

func (s *Server) consents() consents {
	return consents{store: s.store, registry: s.connectors, secrets: s.connectorSecrets, publicURL: s.publicURL}
}

// ConnectorConsents is how a session begins a consent for its caller's own connection, when
// a tool call needs one (session.Consents): the attempt createAuthorization begins, on the
// connection chosen, or else on the caller's newest connection to the connector in any
// status, or else on a new one. Nil when connectors are off or ROUTER_PUBLIC_URL is
// unset, so no consent could begin: a session binding with no usable connection is then left
// out of the session, as before.
func ConnectorConsents(records *store.Store, registry core.Registry, secrets *auth.Sealer, publicURL string) session.Consents {
	if records == nil || secrets == nil || len(registry.Schemes) == 0 || strings.TrimRight(publicURL, "/") == "" {
		return nil
	}
	c := consents{store: records, registry: registry, secrets: secrets, publicURL: publicURL}
	return func(ctx context.Context, request session.ConsentRequest) (session.Consent, error) {
		connection, err := c.connectionFor(ctx, request)
		if err != nil {
			return session.Consent{}, err
		}
		definition, err := records.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
		if err != nil {
			return session.Consent{}, err
		}
		begun, err := c.begin(ctx, connection, request.StepUp)
		if err != nil {
			return session.Consent{}, err
		}
		return session.Consent{
			ConnectionID: connection.ID, AuthorizationID: begun.ID, LaunchURL: begun.LaunchURL,
			HandoffToken: begun.HandoffToken, ExpiresAt: begun.ExpiresAt, Name: definition.Manifest.Name,
		}, nil
	}
}

// connectionFor is the caller's own connection a consent from the chat is for. One chosen
// must be theirs and to the binding's connector, as mayReach and the session's owner rule
// (session.mayUse) require.
func (c consents) connectionFor(ctx context.Context, request session.ConsentRequest) (store.ConnectorConnection, error) {
	if request.CustomerID == "" || request.ConnectorID == "" || request.UserID == "" {
		return store.ConnectorConnection{}, stack.Wrap(errors.New("a consent from the chat needs a customer, a connector and the caller"))
	}
	mine := func(connection store.ConnectorConnection) bool {
		return connection.OwnerType == store.OwnerUser && connection.OwnerID == request.UserID &&
			connection.ConnectorID == request.ConnectorID
	}
	if request.ConnectionID != "" {
		connection, err := c.store.ConnectorConnection(ctx, request.CustomerID, request.ConnectionID)
		if err != nil {
			return store.ConnectorConnection{}, err
		}
		if !mine(connection) {
			return store.ConnectorConnection{}, stack.Wrap(errors.New("the connection chosen is not the caller's own to the binding's connector"))
		}
		return connection, nil
	}
	held, err := c.store.ConnectorConnectionsByOwner(ctx, request.CustomerID, store.ConnectionFilter{
		OwnerType: store.OwnerUser, OwnerID: request.UserID, ConnectorID: request.ConnectorID,
	})
	if err != nil {
		return store.ConnectorConnection{}, err
	}
	// The caller's newest connection to the connector, in any status, so a new chat asking
	// again does not leave a row and a grant each time. The person still consents in this
	// chat: one connected before gets a reconnect (begin), which keeps the old grant when it
	// comes back with another account (completeConsent, account_mismatch), and the session
	// opens it only once that consent connected it anew (session.openOrAsk). A session still
	// uses only the connection chosen for it (T22): this one, by the person's own consent.
	for _, connection := range held {
		if mine(connection) {
			return connection, nil
		}
	}
	definition, err := c.store.LatestConnectorDefinition(ctx, request.CustomerID, request.ConnectorID)
	if err != nil {
		return store.ConnectorConnection{}, err
	}
	// The chat cannot ask for a scheme or an input, so only a connector that needs neither is
	// connected from it, as createConnection takes one with neither named.
	if len(definition.Manifest.Schemes) != 1 {
		return store.ConnectorConnection{}, stack.Wrap(fmt.Errorf("%s allows %d schemes, and the chat cannot choose one",
			definition.ID, len(definition.Manifest.Schemes)))
	}
	scheme := definition.Manifest.Schemes[0]
	profile, err := definition.Manifest.Resolve(scheme, nil, nil)
	if err != nil {
		return store.ConnectorConnection{}, stack.Wrap(err)
	}
	connection := store.ConnectorConnection{
		CustomerID: request.CustomerID, ConnectorID: definition.ID, DefinitionRevision: definition.Revision,
		OwnerType: store.OwnerUser, OwnerID: request.UserID, AuthScheme: scheme, Inputs: profile.Inputs,
	}
	if err := c.store.CreateConnectorConnection(ctx, c.registry, &connection); err != nil {
		return store.ConnectorConnection{}, err
	}
	return connection, nil
}

// begin starts the provider's consent for connection and seals its attempt. stepUp, when set,
// is what the provider asked for on a call of the connection (session.ConsentRequest.StepUp):
// the attempt is a step-up that asks for that access. Its callback is any consent's
// (finishConnectorConsent), so the grant is replaced only once the provider granted it, for
// the same account, and a step-up denied, failed or never finished leaves the old grant as
// it is.
func (c consents) begin(ctx context.Context, connection store.ConnectorConnection, stepUp *core.Outcome) (Authorization, error) {
	// Checked here, not at the callback: a consent the provider cannot send back would
	// otherwise be found out only after the user approved it.
	public := strings.TrimRight(c.publicURL, "/")
	if public == "" {
		return Authorization{}, notConfigured("consents cannot be started: ROUTER_PUBLIC_URL is not set, " +
			"so the provider has nowhere to send the browser back to")
	}
	scheme, found := c.registry.Schemes[connection.AuthScheme]
	if !found {
		return Authorization{}, invalidRequest(fmt.Sprintf("auth_scheme %q is not one this deployment has", connection.AuthScheme))
	}
	// Every consent runs on the connector's latest revision, whatever the connection reads
	// now: a pending one made before a fix, or one connected on a revision marked broken
	// since, gets the fixed manifest. The connection moves to it only once the consent
	// connects it (completeConsent), so a consent denied, failed or for another account
	// leaves it reading what its grant was made with.
	latest, err := c.store.LatestConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID)
	if err != nil {
		return Authorization{}, err
	}
	manifest, err := connectionManifest(ctx, c.store, connection, latest.Revision)
	if err != nil {
		return Authorization{}, err
	}
	kind := store.AttemptReconnect
	if connection.Status == store.ConnectionPending {
		kind = store.AttemptConsent
	}
	if stepUp != nil {
		kind = store.AttemptStepUp
		manifest = steppedUp(manifest, connection, *stepUp)
	}
	ref := core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}
	begun, err := scheme.Begin(ctx, core.BeginInput{Ref: ref, Manifest: manifest, RedirectURI: public + ConnectorCallbackPath})
	// The client comes from the app's own record or the operator's environment
	// (ConnectorClients), so a connector that takes the app's own and finds none says where
	// to put it.
	if errors.Is(err, oauth2code.ErrNoClient) && slices.Contains(manifest.Client.Registration, core.ClientCustomer) {
		return Authorization{}, invalidRequest(fmt.Sprintf("the provider's consent could not be started: %v; "+
			"put the app's own OAuth client with PUT /v1/agents/connectors/%s/oauth-client", err, connection.ConnectorID))
	}
	if err != nil {
		// A scheme's error carries no secret (core AGENTS.md, «Secrets never print»), and it
		// is what the backend needs to fix: a missing client, an unreachable server.
		return Authorization{}, invalidRequest("the provider's consent could not be started: " + err.Error())
	}
	if begun.Done {
		return Authorization{}, invalidRequest(fmt.Sprintf("auth_scheme %q needs no consent", connection.AuthScheme))
	}
	state, err := authorizeState(begun.AuthorizeURL)
	if err != nil {
		return Authorization{}, err
	}
	handoff, err := randomToken(handoffBytes)
	if err != nil {
		return Authorization{}, err
	}

	id := store.NewID()
	raw, err := json.Marshal(attempt{
		ConnectorID:        connection.ConnectorID,
		DefinitionRevision: latest.Revision,
		Scheme:             connection.AuthScheme,
		Handoff:            handoff,
		AuthorizeURL:       begun.AuthorizeURL,
		State:              begun.State,
	})
	if err != nil {
		return Authorization{}, err
	}
	sealed, err := c.secrets.SealWithAAD(string(raw), attemptAAD(connection.CustomerID, connection.ID, id))
	if err != nil {
		return Authorization{}, err
	}
	expires := time.Now().UTC().Add(attemptLifetime).Truncate(time.Microsecond)
	err = c.store.CreateConnectorAuthorizationAttempt(ctx, &store.ConnectorAuthorizationAttempt{
		ID:            id,
		CustomerID:    connection.CustomerID,
		ConnectionID:  connection.ID,
		Kind:          kind,
		StateHash:     store.AuthorizationStateHash(state),
		AttemptSealed: sealed,
		KEKVersion:    c.secrets.CurrentVersion(),
		ExpiresAt:     expires,
	})
	// Deleted between the read above and the insert.
	if errors.Is(err, store.ErrNoConnectorConnection) {
		return Authorization{}, errNoSuchConnection
	}
	if err != nil {
		return Authorization{}, err
	}
	return Authorization{
		ID:           id,
		Kind:         AuthorizationKind(kind),
		LaunchURL:    public + connectorLaunchPath + id,
		HandoffToken: handoff,
		ExpiresAt:    expires,
	}, nil
}

// steppedUp is manifest asking for what a step-up needs (AI-854):
//
//   - the scopes the provider asked for (RFC 6750 section 3.1, insufficient_scope's scope),
//     with, when the manifest's scopes.step_up_union is set, the ones it lists and the ones
//     the grant has, for a provider whose new grant replaces the old one rather than adds to
//     it. MCP's «Scope Challenge Handling» (2025-11-25, Authorization) has a client ask for
//     the union too. Asked for none, the manifest's own list stays;
//   - a claims challenge as the authorize request's claims parameter: OpenID Connect Core 1.0
//     section 5.5, which Microsoft's «Claims challenges, claims requests and client
//     capabilities» has a client send back as it was decoded
//     (learn.microsoft.com/en-us/entra/identity-platform/claims-challenge).
func steppedUp(manifest core.ResolvedManifest, connection store.ConnectorConnection, asked core.Outcome) core.ResolvedManifest {
	scopes := asked.Scopes
	if manifest.Scopes.StepUpUnion {
		scopes = slices.Concat(manifest.Scopes.List, connection.GrantedScopes, asked.Scopes)
	}
	var unique []string
	for _, scope := range scopes {
		if !slices.Contains(unique, scope) {
			unique = append(unique, scope)
		}
	}
	if len(unique) > 0 {
		manifest.Scopes.List = unique
	}
	if asked.Claims != "" {
		params := maps.Clone(manifest.AuthorizeParams)
		if params == nil {
			params = map[string]string{}
		}
		params["claims"] = asked.Claims
		manifest.AuthorizeParams = params
	}
	return manifest
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
func (s *Server) serveConnectorLaunch(w http.ResponseWriter, r *http.Request) {
	dashboard, err := originOf(s.dashboardURL)
	if err != nil {
		writeFailure(w, r, fmt.Errorf("consents cannot be launched: DASHBOARD_BASE_URL is not an http(s) URL: %w", err))
		return
	}
	nonce, err := randomToken(nonceBytes)
	if err != nil {
		writeFailure(w, r, err)
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
// attempt to this browser with a cookie of a fresh value, which the callback then requires.
// A token is traded once: the first handoff wins, so a token that leaked is useless after
// the launch page used it, and if it was used first elsewhere the launch page fails where
// the person can see it. RFC 9700 section 2.1 asks for «one-time use CSRF tokens carried in
// the state parameter that are securely bound to the user agent»; this cookie is that
// binding, so only one user agent may ever hold it.
//
// Origin must be the router's own: the launch page's fetch sends it on a POST (Fetch
// Standard, «append a request Origin header»), so a form or a script on another site that
// learned a handoff token cannot make the router set the cookie in its own browser. That
// holds for browsers only: any other client sends whatever Origin it likes, which is why
// the token is single-use.
func (s *Server) handOffConnectorLaunch(w http.ResponseWriter, r *http.Request) {
	if s.store == nil || s.connectorSecrets == nil {
		writeError(w, errNoAttempt)
		return
	}
	public, err := originOf(s.publicURL)
	if err != nil || r.Header.Get("Origin") != public {
		writeError(w, forbidden("the handoff must come from the launch page"))
		return
	}
	var body struct {
		HandoffToken string `json:"handoff_token"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, maxHandoffBody)).Decode(&body); err != nil || body.HandoffToken == "" {
		writeError(w, invalidRequest("the handoff needs a handoff_token"))
		return
	}
	row, sealed, err := s.openAttemptByID(r.Context(), r.PathValue("id"))
	if errors.Is(err, store.ErrNoAuthorizationAttempt) {
		writeError(w, errNoAttempt)
		return
	}
	if err != nil {
		writeFailure(w, r, err)
		return
	}
	if subtle.ConstantTimeCompare([]byte(body.HandoffToken), []byte(sealed.Handoff)) != 1 {
		writeError(w, forbidden("that is not this consent's handoff token"))
		return
	}
	if sealed.Cookie != "" {
		writeError(w, errNoAttempt)
		return
	}
	sealed.Cookie, err = randomToken(handoffBytes)
	if err != nil {
		writeFailure(w, r, err)
		return
	}
	raw, err := json.Marshal(sealed)
	if err != nil {
		writeFailure(w, r, err)
		return
	}
	resealed, err := s.connectorSecrets.SealWithAAD(string(raw), attemptAAD(row.CustomerID, row.ConnectionID, row.ID))
	if err != nil {
		writeFailure(w, r, err)
		return
	}
	err = s.store.HandOffConnectorAuthorizationAttempt(r.Context(), row.ID, row.AttemptSealed, resealed, s.connectorSecrets.CurrentVersion())
	if errors.Is(err, store.ErrNoAuthorizationAttempt) {
		writeError(w, errNoAttempt)
		return
	}
	if err != nil {
		writeFailure(w, r, err)
		return
	}
	http.SetCookie(w, s.attemptCookie(row.ID, sealed.Cookie, row.ExpiresAt))
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
		writeError(w, invalidRequest("the callback needs exactly one state"))
		return
	}
	if s.store == nil || s.connectorSecrets == nil || s.credentials == nil {
		writeError(w, errNoAttempt)
		return
	}
	row, sealed, err := s.openAttemptByState(ctx, states[0])
	if errors.Is(err, store.ErrNoAuthorizationAttempt) {
		writeError(w, errNoAttempt)
		return
	}
	if err != nil {
		writeFailure(w, r, err)
		return
	}
	// An attempt not handed off yet has no cookie value, and an empty cookie must not match it.
	cookie, err := r.Cookie(attemptCookiePrefix + row.ID)
	if err != nil || sealed.Cookie == "" || subtle.ConstantTimeCompare([]byte(cookie.Value), []byte(sealed.Cookie)) != 1 {
		writeError(w, forbidden("finish the consent in the browser that started it"))
		return
	}
	consumed, err := s.store.ConsumeConnectorAuthorizationAttempt(ctx, states[0])
	if errors.Is(err, store.ErrNoAuthorizationAttempt) || err == nil && consumed.ID != row.ID {
		writeError(w, errNoAttempt)
		return
	}
	if err != nil {
		writeFailure(w, r, err)
		return
	}
	http.SetCookie(w, s.attemptCookie(row.ID, "", time.Time{}))

	outcome := s.completeConsent(ctx, row, sealed, query)
	// A consent a session asked for in its conversation carries on there. The attempt names
	// the session's own login, so no other session is handed it.
	if outcome == consentConnected && s.sessions != nil {
		s.sessions.ConnectorConsentFinished(ctx, row.CustomerID, row.ConnectionID, row.ID)
	}
	destination, err := url.Parse(s.dashboardURL)
	if err != nil || s.dashboardURL == "" {
		writeFailure(w, r, errors.Join(fmt.Errorf("the consent ended as %s, and DASHBOARD_BASE_URL is not set to go back to", outcome), err))
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
	var committed *core.CredentialState
	// The tokens a reconnect replaces, and those the consent got.
	change := core.CredentialChange{Current: core.FingerprintsOf(s.connectors.Schemes, credentials)}
	err = s.credentials.Update(ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		if state.AccountID != "" && state.AccountID != account.AccountID {
			switched = true
			state.LastError = accountSwitchError
			return true, nil
		}
		// The credential store leaves the revision it committed here (core.CredentialStore).
		committed = state
		change.Previous = core.FingerprintsOf(s.connectors.Schemes, state.Credentials)
		state.Credentials = credentials
		state.Status = store.ConnectionConnected
		state.LastError = ""
		state.AccountID = account.AccountID
		state.Metadata = account.Metadata
		state.Scopes = account.Scopes
		// The connection reads the revision its grant was made on from now on.
		state.DefinitionRevision = sealed.DefinitionRevision
		// Unknown here: the expiry is inside the scheme's stored credentials, and the
		// resolver (T12) sets it the first time it retrieves an access credential.
		state.ExpiresAt = time.Time{}
		// The grant begins here; a provider's signal about an older one leaves it alone
		// (Resolver.Revoke).
		state.ConnectedAt = time.Now().UTC()
		return true, nil
	})
	if err != nil {
		s.logger.Error("could not store a consent's credentials", "connection", row.ConnectionID, "error", err)
		return consentFailed
	}
	if switched {
		return consentAccountMismatch
	}
	s.auditGrant(ctx, connection.CustomerID, connection.ID, connection.ConnectorID, connection.OwnerType,
		store.AuditGrantCreated, store.AuditReasonConsent, committed.Revision, row.ID, change)
	s.recordClient(ctx, connection.ID, credentials)
	// A reconnect brings back the MCP event subscriptions its bindings declare, as a validate
	// does: one a disconnect or a long wait dropped is made again. The consent itself is done,
	// so a failure here is logged and the next validate tries again.
	if s.mcpEvents != nil {
		connected, err := s.store.ConnectorConnection(ctx, row.CustomerID, row.ConnectionID)
		if err == nil {
			err = s.mcpEvents.Reconcile(ctx, connected)
		}
		if err != nil {
			s.logger.Error("could not subscribe a reconnected connection's MCP events", "connection", row.ConnectionID, "error", err)
		}
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
		writeError(w, notFound("no client metadata document: ROUTER_PUBLIC_URL is not an https URL"))
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
	return connectionManifest(ctx, s.store, connection, revision)
}

// connectionManifest keeps only the captured values revision still captures. A consent runs
// on the latest revision (begin), which may have dropped a capture rule the connection's own
// revision had; no template of revision can name that value, and Resolve refuses one it does
// not capture. At the connection's own revision every value is one it captured, so nothing
// is left out.
func connectionManifest(ctx context.Context, records *store.Store, connection store.ConnectorConnection, revision int) (core.ResolvedManifest, error) {
	definition, err := records.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, revision)
	if err != nil {
		return core.ResolvedManifest{}, err
	}
	metadata := maps.Clone(connection.Metadata)
	maps.DeleteFunc(metadata, func(name, _ string) bool {
		return !slices.ContainsFunc(definition.Manifest.Capture, func(rule core.CaptureRule) bool { return rule.Name == name })
	})
	return definition.Manifest.Resolve(connection.AuthScheme, connection.Inputs, metadata)
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
