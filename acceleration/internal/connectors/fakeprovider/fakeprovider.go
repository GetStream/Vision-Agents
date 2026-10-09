// Package fakeprovider is one httptest server that plays an OAuth authorization server and
// an MCP endpoint for connector tests, with personalities that change how it behaves.
//
// It is for _test.go files only. It listens on a loopback address, which the egress client
// refuses by design, so a test that drives a scheme or a source against it hands that code
// Server.Client() instead of an egress client. Tests of the egress policy itself keep using
// egress's own seams; nothing here weakens or bypasses egress.
package fakeprovider

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"
)

// Paths the server answers on. Every URL is Server.URL plus one of these.
const (
	// RFC 9728 §3.1: the well-known suffix inserted before the resource's path (/mcp).
	PathProtectedResource = "/.well-known/oauth-protected-resource/mcp"
	// RFC 8414 §3: an issuer with no path serves metadata here.
	PathAuthorizationServer = "/.well-known/oauth-authorization-server"
	PathRegister            = "/register"
	PathAuthorize           = "/authorize"
	PathToken               = "/token"
	PathRevoke              = "/revoke"
	PathMCP                 = "/mcp"
)

const (
	// AccessTTL is how long an access token works. Synthetic; it is QuickBooks' one hour
	// (oauth-jsclient README «Access tokens are valid for 3600 seconds»), and a test moves
	// past it with Advance, so the number itself does not matter.
	AccessTTL = time.Hour
	// Grace is how long a rotated refresh token keeps working under RotatingRefreshWithGrace.
	// Synthetic, like AccessTTL: providers differ (QuickBooks 24 hours, oauth-jsclient
	// README; Slack «brief», docs.slack.dev/authentication/using-token-rotation) and a test
	// crosses it with Advance.
	Grace = 30 * time.Minute
	// SlackAccessTTL is how long an access token works under CommaScopes: Slack's rotated
	// tokens «will always expire in 43,200 seconds, which is 12 hours»
	// (docs.slack.dev/authentication/using-token-rotation), and core's recorded Slack
	// response has expires_in 43200.
	SlackAccessTTL = 12 * time.Hour
	// codeTTL is RFC 6749 §4.1.2: «A maximum authorization code lifetime of 10 minutes is
	// RECOMMENDED».
	codeTTL = 10 * time.Minute
	// RetryAfter is what RateLimited asks for. 30 seconds is T28's acceptance case in the
	// AI-816 subtasks, not a vendor value.
	RetryAfter = 30 * time.Second
	// RequiredScope is the scope tools/call needs under InsufficientScope. The name is the
	// MCP authorization spec's own example (2025-11-25 «Scope Challenge Handling»).
	RequiredScope = "files:write"
	// RedirectURI is registered for the preregistered client. client.example is the RFC 6749
	// example client host; the browser is simulated by Consent, so nothing dials it.
	RedirectURI = "https://client.example/callback"
	// ForeignIssuerURL is the issuer ForeignIssuer puts in the callback: a mix-up attacker
	// (RFC 9207 §1). Any https URL other than Server.URL works.
	ForeignIssuerURL = "https://attacker.example"
)

// Personality is one way the server departs from plain OAuth and MCP. With none set it is a
// strict authorization server: PKCE S256 required, iss in every authorization response
// (RFC 9207), and refresh tokens rotated on every use with no grace (RFC 9700 §4.14.2).
type Personality string

const (
	// RotatingRefreshWithGrace keeps a rotated refresh token valid for Grace after the
	// rotation, as QuickBooks does («previous refresh tokens expire 24 hours after you
	// receive a new one», oauth-jsclient README). After Grace the old token gets invalid_grant
	// and the grant, with the token the rotation issued, stays.
	RotatingRefreshWithGrace Personality = "rotating_refresh_with_grace"
	// NonRotatingRefresh answers a refresh without a refresh_token, so the client keeps the
	// one it has (RFC 6749 §6: the server «MAY issue a new refresh token»).
	NonRotatingRefresh Personality = "non_rotating_refresh"
	// NoRefreshToken issues an access token alone (RFC 6749 §5.1: refresh_token is
	// OPTIONAL), so once it expires only a new consent helps.
	NoRefreshToken Personality = "no_refresh_token"
	// InvalidGrant refuses every refresh with invalid_grant (RFC 6749 §5.2). Only the refresh
	// is refused: access tokens already issued keep working until they expire. A test that
	// needs the grant gone as well revokes a token at PathRevoke.
	InvalidGrant Personality = "invalid_grant"
	// LostResponse performs a refresh, rotation included, then closes the connection before
	// answering. The old refresh token is spent and the new one never arrives. It changes how
	// a refresh is delivered, not what it does, so it combines with RotatingRefreshWithGrace
	// (the spent token still works for Grace) and NonRotatingRefresh (nothing is spent).
	LostResponse Personality = "lost_response"
	// LostResponseOnce is LostResponse for the next refresh only; the refreshes after it
	// answer. With RotatingRefreshWithGrace it is the case a grace retry exists for: the
	// rotation happened, its answer was lost, and the old token is presented again inside
	// the window.
	LostResponseOnce Personality = "lost_response_once"
	// Unavailable answers the token endpoint with 503 (RFC 9110 §15.6.4) and changes nothing.
	Unavailable Personality = "unavailable"
	// ServerError performs a refresh, rotation included, then answers 500 (RFC 9110 §15.6.1)
	// with error server_error (RFC 6749 §4.1.2.1's name for a 500). A 500 does not say the
	// request had no effect; Slack's internal_error, which CommaScopes answers instead, says
	// so outright: «It's possible some aspect of the operation succeeded before the error
	// was raised» (docs.slack.dev/reference/methods/oauth.v2.access).
	ServerError Personality = "server_error"
	// CutOffRefusal answers a refresh with 400 and a body cut off midway: the headers arrive,
	// Content-Length promises more than is sent, and the connection closes. The refresh is
	// refused before anything changes, so the status is all a client can go by. Not a vendor
	// behaviour: a transport failure after the status line, which any provider can have.
	CutOffRefusal Personality = "cut_off_refusal"
	// AccessTokenNotRevocable refuses to revoke an access token with 400
	// unsupported_token_type, which RFC 7009 section 2.2.1 defines for «the client tried to
	// revoke an access token on a server not supporting this feature» (section 2: a server
	// MUST revoke refresh tokens and only SHOULD revoke access tokens). Refresh tokens are
	// revoked as before.
	AccessTokenNotRevocable Personality = "access_token_not_revocable"
	// InsufficientScope answers tools/call with 403 and error="insufficient_scope" (RFC 6750
	// §3.1) unless the token carries RequiredScope.
	InsufficientScope Personality = "insufficient_scope"
	// ClaimsChallenge answers MCP requests with 401 and a claims challenge (Microsoft
	// «Claims challenges, claims requests and client capabilities») until the token came
	// from a consent that passed those claims.
	ClaimsChallenge Personality = "claims_challenge"
	// BareChallenge answers an MCP request whose token it does not take with 401 and a Bearer
	// challenge that holds resource_metadata alone, with no error code, as MCP servers do:
	// «Invalid or expired tokens MUST receive a HTTP 401 response», and the spec's 401 names
	// the resource metadata, not an error (MCP 2026-07-28, Authorization, «Token Handling»
	// and the example under «Scope Selection Strategy»). Without it the fake answers
	// error="invalid_token" (RFC 6750 §3.1).
	BareChallenge Personality = "bare_challenge"
	// RateLimited answers MCP requests and refresh grants with 429 (RFC 6585 §4) and
	// Retry-After (RFC 9110 §10.2.3) of RetryAfter. A refresh it refuses changes nothing.
	RateLimited Personality = "rate_limited"
	// CommaScopes plays Slack's OAuth v2: scope and user_scope are comma-separated, the token
	// response, refresh included, is oauth.v2.access's shape with team and authed_user, access
	// tokens live SlackAccessTTL, and token errors come as HTTP 200 with ok false.
	CommaScopes Personality = "comma_scopes"
	// SlackUserToken, beside CommaScopes, plays oauth.v2.user.access, the token endpoint of
	// Slack's user-token consent (providers/slack.yaml): the token response is the user token
	// itself, with user_id, team and enterprise at the top and no authed_user. Its keys are
	// what a live exchange answered on 2026-10-08 with token rotation on: access_token,
	// app_id, enterprise (null), expires_in, is_enterprise_install, ok, refresh_token, scope,
	// team {id, name}, token_type and user_id. The example on
	// docs.slack.dev/reference/methods/oauth.v2.user.access shows authed_user instead. The
	// values are synthetic; token_type "user" and a refresh answering the same keys are
	// unverified. Only Slack has this shape, so it is named for Slack. Its MCP endpoint names
	// the user who consented in every tool's description, as Slack's does (describedFor).
	SlackUserToken Personality = "slack_user_token"
	// CallbackRealmID adds realmId to the callback, as QuickBooks does: Intuit's SDK reads it
	// from the redirect (oauth-jsclient src/OAuthClient.js createToken, params.realmId).
	CallbackRealmID Personality = "callback_realm_id"
	// SignedCallback adds shop, host, timestamp and an hmac over the callback, as Shopify
	// does (shopify.dev «Authorization code grant»). Sign says how it is signed.
	SignedCallback Personality = "signed_callback"
	// ForeignIssuer names ForeignIssuerURL as iss in the callback, which a client must reject
	// (RFC 9207 §2.4).
	ForeignIssuer Personality = "foreign_issuer"
	// ConsentDenied sends the browser back with error=access_denied (RFC 6749 §4.1.2.1).
	ConsentDenied Personality = "consent_denied"
	// ClientMetadataDocuments makes the server accept an https URL as client_id, fetching
	// the client's metadata from it at authorize, as draft-ietf-oauth-client-id-metadata-
	// document-02 («CIMD» below) describes: it advertises
	// client_id_metadata_document_supported (§6), fetches with the client
	// FetchClientMetadataWith set, without following redirects (§5), and checks the
	// document's client_id (§4), its redirect_uris (§4.2) and that it holds no shared secret
	// (§4.1). The document is fetched again on every authorize; §5.2 only allows caching.
	ClientMetadataDocuments Personality = "client_metadata_documents"
	// ClientCredentials lets the token endpoint take grant_type=client_credentials (RFC 6749
	// §4.4) from a client that authenticates with its secret (§4.4.2: «The client MUST
	// authenticate»; §4.4: only for confidential clients), and answers an access token alone
	// (§4.4.3: «A refresh token SHOULD NOT be included»). Without it the grant gets
	// unsupported_grant_type, as from a server that does not offer it (§5.2).
	ClientCredentials Personality = "client_credentials"
	// NoExpiresIn leaves expires_in out of every token response, which RFC 6749 §5.1 only
	// RECOMMENDS, as Salesforce does (its Mobile SDK reads none: OAuth2.java:1325-1336 at
	// 863835e9). Tokens still end after AccessTTL.
	NoExpiresIn Personality = "no_expires_in"
	// IdentityURL adds id, the identity URL IdentityURL holds, to every token response, as
	// Salesforce does: its Mobile SDK reads id from each token endpoint response
	// (SalesforceMobileSDK-Android OAuth2.java:1332 at 863835e9, cited in
	// providers/salesforce.yaml), and core's recorded response has one
	// (core/testdata/recorded/salesforce.token.json). That the client credentials response
	// carries it too is unverified.
	IdentityURL Personality = "identity_url"
	// SlackChannel plays Slack as an inbound channel (channel.go): Deliver posts an Events API
	// event signed as Slack signs it (docs.slack.dev/authentication/verifying-requests-from-slack),
	// and PathChatPostMessage takes a reply with a bot token InstallBot issued, answering as
	// chat.postMessage does, a refusal included, which comes as HTTP 200 with ok false
	// (docs.slack.dev/reference/methods/chat.postMessage). Without it PathChatPostMessage is
	// 404. Only Slack has these shapes, so it is named for Slack.
	SlackChannel Personality = "slack_channel"
	// MCPEvents adds MCP Events' webhook delivery to the MCP endpoint (events.go): the events
	// capability in server/discover, events/subscribe, which verifies the callback with a
	// signed challenge before it subscribes, and events/unsubscribe; Emit and DeliverEvent
	// post signed occurrences. The draft is experimental-ext-triggers-events at 6682596d.
	// Without it those methods are not found and the capability is absent.
	MCPEvents Personality = "mcp_events"
	// SlowConfigRotation holds every tooling.tokens.rotate for slowRotationDelay before it
	// answers, so two callers that rotate one configuration token without a lock between
	// them overlap at the server. Not a Slack behaviour: a slow network, which any caller can
	// meet.
	SlowConfigRotation Personality = "slow_config_rotation"
)

// tokenEndpoint are the personalities that decide what the token endpoint does; at most one
// can be on.
var tokenEndpoint = []Personality{RotatingRefreshWithGrace, NonRotatingRefresh, NoRefreshToken, InvalidGrant, Unavailable, ServerError, CutOffRefusal}

// Server is the fake provider. Its fields are fixed when New returns.
type Server struct {
	// URL is the issuer and the origin of every endpoint: https on a loopback address.
	URL string
	// ClientID and ClientSecret are a confidential client registered ahead of time, as an
	// operator's app is. It may authenticate with client_secret_basic or client_secret_post
	// (RFC 6749 §2.3.1) and redirect to RedirectURI or what AllowRedirect added.
	ClientID     string
	ClientSecret string
	// TeamID and UserID are the workspace and user CommaScopes reports; RealmID is what
	// CallbackRealmID reports; Shop is the shop SignedCallback names; IdentityURL is the id
	// IdentityURL reports. All synthetic.
	TeamID      string
	UserID      string
	RealmID     string
	Shop        string
	IdentityURL string

	t      testing.TB
	server *httptest.Server

	mu            sync.Mutex
	personalities map[Personality]bool
	offset        time.Duration
	clients       map[string]*client
	codes         map[string]*authorizationCode
	access        map[string]*accessToken
	refresh       map[string]*refreshToken
	hits          map[string]int
	refreshes     int
	// clientCredentialsGrants counts the client_credentials grants that reached the token
	// endpoint.
	clientCredentialsGrants int
	// refreshScope is the scope parameter of the last refresh grant, and refreshScopeSent
	// whether it had one at all.
	refreshScope     string
	refreshScopeSent bool
	// lostOnce is set when LostResponseOnce has dropped its one answer.
	lostOnce bool
	// metadataClient fetches client metadata documents under ClientMetadataDocuments.
	metadataClient *http.Client
	// posts are the messages chat.postMessage took under SlackChannel.
	posts []Post
	// failPosts is how many more posts answer 503 (FailPosts).
	failPosts int
	// refusedPosts is how many more posts answer refusal in an HTTP 200 (RefusePosts).
	refusedPosts int
	refusal      string
	// garbledPosts is how many more posts are taken and answered with a body that is not
	// JSON (GarblePosts).
	garbledPosts int
	// account is the user the next consent is by: UserID until SwitchAccount.
	account string

	// The fake Slack (slack.go): configuration tokens by token, refresh tokens with whether
	// they were spent, successful rotations, and apps by id in the order they were made.
	configTokens    map[string]*configToken
	configRefresh   map[string]bool
	configRotations int
	slackApps       map[string]*SlackApp
	slackOrder      []string

	// eventSubs are the MCP Events subscriptions under MCPEvents, by their key (eventSlot).
	eventSubs map[string]*eventSub
	// eventsTTL is what GrantEventsFor set; zero is EventsTTL.
	eventsTTL time.Duration
}

type client struct {
	id        string
	secret    string
	methods   []string
	redirects []string
}

// grant is what one consent produced. Revoking it ends every token it issued.
type grant struct {
	clientID string
	scopes   []string
	claims   string
	revoked  bool
	// current is the refresh token in use; a rotated one stays in refresh with rotatedAt set.
	current string
	// user marks the grant of the Slack user token CommaScopes issues beside the bot token's
	// when user_scope was asked.
	user bool
	// account is the user who consented, which CommaScopes reports as authed_user.id, and
	// SlackUserToken as user_id, on the exchange and on every refresh of this grant.
	account string
}

type authorizationCode struct {
	clientID    string
	redirectURI string
	challenge   string
	scopes      []string
	userScopes  []string
	claims      string
	expires     time.Time
	used        bool
	grant       *grant
	userGrant   *grant
}

type accessToken struct {
	grant   *grant
	scopes  []string
	expires time.Time
}

type refreshToken struct {
	grant     *grant
	rotatedAt time.Time
}

// New starts a server with the given personalities and closes it when the test ends.
func New(t testing.TB, personalities ...Personality) *Server {
	t.Helper()
	s := &Server{
		ClientID:     synthetic("client"),
		ClientSecret: synthetic("secret"),
		TeamID:       "T" + strings.ToUpper(synthetic("team")[5:15]),
		UserID:       "U" + strings.ToUpper(synthetic("user")[5:15]),
		RealmID:      syntheticDigits(16),
		Shop:         "fake-" + synthetic("shop")[5:13] + ".myshopify.com",
		// The shape of core's recorded identity URL (core/testdata/recorded/salesforce.token.json):
		// login.salesforce.com, then an org id (00D) and a user id (005), 18 characters each.
		// Nothing dials it; a manifest only captures it.
		IdentityURL: "https://login.salesforce.com/id/00D" + strings.ToUpper(synthetic("org")[4:19]) +
			"/005" + strings.ToUpper(synthetic("user")[5:20]),
		t:             t,
		personalities: map[Personality]bool{},
		clients:       map[string]*client{},
		codes:         map[string]*authorizationCode{},
		access:        map[string]*accessToken{},
		refresh:       map[string]*refreshToken{},
		hits:          map[string]int{},
		configTokens:  map[string]*configToken{},
		configRefresh: map[string]bool{},
		slackApps:     map[string]*SlackApp{},
	}
	s.account = s.UserID
	s.clients[s.ClientID] = &client{
		id: s.ClientID, secret: s.ClientSecret,
		methods:   []string{"client_secret_basic", "client_secret_post"},
		redirects: []string{RedirectURI},
	}
	s.Use(personalities...)
	s.server = httptest.NewTLSServer(s.routes())
	t.Cleanup(s.server.Close)
	s.URL = s.server.URL
	return s
}

// Use replaces the personalities. Call it from the test goroutine.
func (s *Server) Use(personalities ...Personality) {
	s.t.Helper()
	set := map[Personality]bool{}
	var token []Personality
	for _, p := range personalities {
		set[p] = true
		for _, q := range tokenEndpoint {
			if p == q {
				token = append(token, p)
			}
		}
	}
	if len(token) > 1 {
		s.t.Fatalf("fakeprovider: %v all decide the token endpoint; use one", token)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.personalities = set
	s.lostOnce = false
}

// Client trusts the server's certificate. Hand it to the code under test in place of an
// egress client, which refuses loopback.
func (s *Server) Client() *http.Client {
	return s.server.Client()
}

// AllowRedirect registers one more redirect URI for the preregistered client, such as a
// router's callback URL in an API test.
func (s *Server) AllowRedirect(uri string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c := s.clients[s.ClientID]
	c.redirects = append(c.redirects, uri)
}

// FetchClientMetadataWith sets the client ClientMetadataDocuments fetches a client_id URL
// with, such as the Client of the test's TLS server that serves the document. CIMD §8.6
// allows a server on loopback, in testing only, to fetch from loopback.
func (s *Server) FetchClientMetadataWith(c *http.Client) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fetch := *c
	// CIMD §5: «MUST NOT automatically follow HTTP redirects».
	fetch.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	s.metadataClient = &fetch
}

// SwitchAccount makes every later consent be by another user of the same workspace, as a
// person who signs in to the provider as someone else between two consents is. It returns
// that user's id, synthetic like UserID, which keeps naming the first one. Grants already
// made keep the user who made them. Call it from the test goroutine.
func (s *Server) SwitchAccount() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.account = "U" + strings.ToUpper(synthetic("user")[5:15])
	return s.account
}

// Advance moves the server's clock, so expiry and grace windows pass without sleeping.
func (s *Server) Advance(d time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.offset += d
}

// Hits is how many requests reached path, whatever their outcome.
func (s *Server) Hits(path string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.hits[path]
}

// Refreshes is how many refresh_token grants reached the token endpoint.
func (s *Server) Refreshes() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.refreshes
}

// ClientCredentialsGrants is how many client_credentials grants reached the token endpoint,
// whatever their outcome.
func (s *Server) ClientCredentialsGrants() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.clientCredentialsGrants
}

// RefreshScope is the scope parameter the last refresh grant carried, and whether it
// carried one (RFC 6749 §6: scope is OPTIONAL there).
func (s *Server) RefreshScope() (scope string, sent bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.refreshScope, s.refreshScopeSent
}

// Consent plays the browser: it opens authorizeURL as a user who approves and returns where
// the server sent the browser back, with code, state and iss (or error) in its query. An
// authorization request the server cannot redirect (unknown client, unregistered redirect
// URI) is an error, as RFC 6749 §4.1.2.1 forbids redirecting it.
func (s *Server) Consent(authorizeURL string) (*url.URL, error) {
	browser := *s.server.Client()
	browser.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	response, err := browser.Get(authorizeURL)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusFound {
		return nil, fmt.Errorf("fakeprovider: authorize answered %d, not a redirect", response.StatusCode)
	}
	return url.Parse(response.Header.Get("Location"))
}

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET "+PathProtectedResource, s.protectedResource)
	mux.HandleFunc("GET "+PathAuthorizationServer, s.authorizationServer)
	mux.HandleFunc("POST "+PathRegister, s.register)
	mux.HandleFunc("GET "+PathAuthorize, s.authorize)
	mux.HandleFunc("POST "+PathToken, s.token)
	mux.HandleFunc("POST "+PathRevoke, s.revoke)
	// GET and DELETE on the MCP endpoint get 405 from the mux, which is what MCP 2026-07-28
	// asks of a server without the GET stream or sessions (Streamable HTTP, «Earlier
	// Streamable HTTP Revisions»).
	mux.HandleFunc("POST "+PathMCP, s.mcp)
	mux.HandleFunc("POST "+PathChatPostMessage, s.chatPostMessage)
	s.slackRoutes(mux)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.hits[r.URL.Path]++
		s.mu.Unlock()
		mux.ServeHTTP(w, r)
	})
}

// now is the server's clock. Call it with mu held.
func (s *Server) now() time.Time {
	return time.Now().Add(s.offset)
}

func (s *Server) is(p Personality) bool {
	return s.personalities[p]
}

// isOn is is for a caller that does not hold mu.
func (s *Server) isOn(p Personality) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.is(p)
}

func (s *Server) resource() string {
	return s.URL + PathMCP
}

// syntheticDigits is a random decimal string, the shape of a QuickBooks realmId
// (9130350000000001 in core/testdata/recorded/quickbooks.callback).
func syntheticDigits(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	for i := range b {
		b[i] = '0' + b[i]%10
	}
	return string(b)
}

// synthetic is a fresh random value with a readable prefix, so a test never sees the same
// token twice and no real credential is ever involved.
func synthetic(prefix string) string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return prefix + "-" + hex.EncodeToString(b)
}
