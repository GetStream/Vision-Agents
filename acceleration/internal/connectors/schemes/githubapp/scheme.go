// Package githubapp is the github_app scheme: a GitHub App acting as one of its installations
// (the bot), with an installation access token minted from the app's private key.
//
// Begin is Done, since nobody consents. Complete takes the app id, the private key and the
// installation id the developer supplies, mints a first token and lists one repository with
// it, so a wrong key or installation is refused at once, and seals all of it. Retrieve hands
// the token out until it is due, then mints another: an installation token lasts an hour and
// has no refresh token. Wrap is a bearer token, which both the MCP server and git's Basic
// password (through the sandbox brokers) take. Revoke revokes the token; the app and its key
// live on at GitHub.
//
// Sources, opened November 9, 2026:
//
//	[jwt]     https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-a-json-web-token-jwt-for-a-github-app
//	[install] https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-an-installation-access-token-for-a-github-app
//	[create]  https://docs.github.com/en/rest/apps/apps#create-an-installation-access-token-for-an-app
//	[list]    https://docs.github.com/en/rest/apps/installations#list-repositories-accessible-to-the-app-installation
//	[revoke]  https://docs.github.com/en/rest/apps/installations#revoke-an-installation-access-token
//	[limits]  https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api#exceeding-the-rate-limit
//	[headers] https://docs.github.com/en/rest/using-the-rest-api/getting-started-with-the-rest-api#headers
package githubapp

import (
	"bytes"
	"context"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Name is the registry name manifests list under schemes, as the design's GitHub manifest
// spells it (core/testdata/manifests/github.yaml, stress-test row 6).
const Name = "github_app"

// The keys of core.CompleteInput.Supplied this scheme reads.
const (
	// SuppliedAppID is the JWT's iss: «The client ID or application ID of your GitHub App»
	// [jwt].
	SuppliedAppID = "app_id"
	// SuppliedPrivateKey is the app's private key, the PEM GitHub downloads [jwt].
	SuppliedPrivateKey = "private_key"
	// SuppliedInstallationID is the installation the token acts as [install].
	SuppliedInstallationID = "installation_id"
	// SuppliedPermissions narrows the token: a JSON object of permission to access level, the
	// body's permissions [create]. Optional.
	SuppliedPermissions = "permissions"
	// SuppliedRepositories narrows the token to these repositories of the installation, by
	// name, comma-separated: the body's repositories [create]. Optional.
	SuppliedRepositories = "repositories"
)

const (
	// payloadVersion is the shape of payload. A new shape is a new version.
	payloadVersion = 1
	// defaultMargin is how long before expires_at a token is replaced when the manifest sets
	// no refresh.margin. A choice: a token lasts an hour [install], and five minutes leaves
	// a sandbox command or a tool call started on the old one time to finish.
	defaultMargin = 5 * time.Minute
	// jwtBackdate and jwtLifetime: «set this 60 seconds in the past to allow for clock
	// drift», and exp «no more than 10 minutes into the future» [jwt]. Nine minutes keeps
	// a margin under ten for a router clock that runs a little fast.
	jwtBackdate = 60 * time.Second
	jwtLifetime = 9 * time.Minute
	// maxRepositories is the most repositories a token may name: «You can specify up to 500
	// repositories» [create].
	maxRepositories = 500
	// apiVersion is the REST API version every request names [headers].
	apiVersion = "2022-11-28"
	// maxBody bounds what is read of an answer: a token response is well under it.
	maxBody = 1 << 20
)

var (
	// appID is a numeric app id or a client id, such as Iv23li…: letters, digits and dots
	// [jwt]. It goes into the JWT, never a URL.
	appID = regexp.MustCompile(`^[A-Za-z0-9.]{1,64}$`)
	// installationID is the integer in /app/installations/{installation_id} [create].
	installationID = regexp.MustCompile(`^[1-9][0-9]{0,19}$`)
	// permissionName and permissionLevel are what the body's permissions object holds:
	// snake_case names such as pull_requests, each read, write or admin [create].
	permissionName  = regexp.MustCompile(`^[a-z][a-z_]{0,63}$`)
	permissionLevel = []string{"read", "write", "admin"}
	// repositoryName is a repository's name without its owner: GitHub keeps letters,
	// digits, '-', '_' and '.' (the form that creates one replaces anything else).
	repositoryName = regexp.MustCompile(`^[A-Za-z0-9._-]{1,100}$`)
)

// ErrNoAPIBase is a manifest with no endpoints.api_base, which every request here is built on.
var ErrNoAPIBase = errors.New("githubapp: the manifest names no endpoints.api_base")

// Config is what the scheme depends on.
type Config struct {
	// HTTP carries every request to GitHub. In the router it is egress.NewClient(timeout,
	// nil); tests pass an httptest server's client.
	HTTP *http.Client
	// Now is the clock the JWT and a token's expiry are read by; nil is time.Now.
	Now func() time.Time
	// PublicEndpoint checks every URL before anything is sent; nil is
	// egress.ValidatePublicHTTPSURL.
	PublicEndpoint func(ctx context.Context, raw string) error
}

// Scheme is the github_app scheme. It holds no state of its own and is safe for concurrent use.
type Scheme struct {
	http           *http.Client
	now            func() time.Time
	publicEndpoint func(ctx context.Context, raw string) error
}

var (
	_ core.Scheme        = (*Scheme)(nil)
	_ core.Static        = (*Scheme)(nil)
	_ core.Fingerprinter = (*Scheme)(nil)
)

// New checks cfg and returns the scheme.
func New(cfg Config) (*Scheme, error) {
	if cfg.HTTP == nil {
		return nil, stack.Wrap(errors.New("githubapp: Config.HTTP is required"))
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.PublicEndpoint == nil {
		cfg.PublicEndpoint = egress.ValidatePublicHTTPSURL
	}
	return &Scheme{http: cfg.HTTP, now: cfg.Now, publicEndpoint: cfg.PublicEndpoint}, nil
}

// Name is Name.
func (*Scheme) Name() string {
	return Name
}

// StaticCredential marks the scheme core.Static: the key is supplied, nobody consents, and a
// key GitHub refuses is fixed only by supplying another with PUT .../credentials. The tokens
// minted from it are renewed here, which Static does not forbid.
func (*Scheme) StaticCredential() {}

// payload is the sealed payload of a connection's StoredCredentials: the app's key and the
// installation, which are the long-lived credential, and the token last minted with them.
type payload struct {
	AppID          string            `json:"app_id"`
	PrivateKey     string            `json:"private_key"`
	InstallationID string            `json:"installation_id"`
	Permissions    map[string]string `json:"permissions,omitempty"`
	Repositories   []string          `json:"repositories,omitempty"`
	Token          string            `json:"token"`
	ExpiresAt      time.Time         `json:"expires_at"`
}

// Begin is Done: the key is supplied, nobody consents.
func (*Scheme) Begin(_ context.Context, in core.BeginInput) (core.BeginOutput, error) {
	if _, err := apiBase(in.Manifest); err != nil {
		return core.BeginOutput{}, err
	}
	return core.BeginOutput{Done: true}, nil
}

// Complete checks the supplied values, mints a first token, lists one repository with it and
// seals them all, so a connection is never made with a key, an installation or a narrowing
// GitHub refuses. Its errors name what is wrong, never the value.
func (s *Scheme) Complete(ctx context.Context, in core.CompleteInput) (core.StoredCredentials, core.AccountInfo, error) {
	app, err := supplied(in.Supplied)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	next, err := s.mint(ctx, in.Manifest, app)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	if err := s.check(ctx, in.Manifest, next.Token); err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	stored, err := seal(next)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	return stored, core.AccountInfo{}, nil
}

// Retrieve hands out the stored token until it is inside the margin of its expiry or expires
// at or before opts.ValidUntil, and then mints another with the stored key. A refused one
// (opts.Refused) is replaced whatever its expiry. opts.Checkpoint is never called: a mint
// spends nothing, so a lost answer is safe to follow with another. A failed mint returns a
// *core.OutcomeError and no StoredCredentials, with the old token beside it while that has
// not expired and was not refused.
func (s *Scheme) Retrieve(ctx context.Context, stored core.StoredCredentials, m core.ResolvedManifest, opts core.RetrieveOptions) (core.AccessCredential, core.StoredCredentials, error) {
	current, err := open(stored)
	if err != nil {
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	due := s.now().Add(margin(m))
	if opts.ValidUntil.After(due) {
		due = opts.ValidUntil
	}
	if !opts.Refused && due.Before(current.ExpiresAt) {
		return credential(current), stored, nil
	}
	next, err := s.mint(ctx, m, current)
	if err != nil {
		if !opts.Refused && s.now().Before(current.ExpiresAt) {
			return credential(current), core.StoredCredentials{}, err
		}
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	renewed, err := seal(next)
	if err != nil {
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	return credential(next), renewed, nil
}

// Wrap puts the token on every request as «Authorization: Bearer» ([install]: «use the
// installation access token in the Authorization header»). A credential this scheme did not
// issue fails each request instead.
func (*Scheme) Wrap(base http.RoundTripper, c core.AccessCredential) http.RoundTripper {
	var secret struct {
		Token string `json:"token"`
	}
	if c.Scheme != Name || json.Unmarshal(c.Secret(), &secret) != nil {
		secret.Token = ""
	}
	return oauth2code.Bearer(base, secret.Token)
}

// Classify is oauth2code's: a 401 to a request that carried the token is InvalidGrant, which
// the transport answers by minting another (core.Transports).
func (*Scheme) Classify(resp *http.Response, body []byte, err error) core.Outcome {
	return oauth2code.ClassifyStatic(resp, body, err)
}

// Revoke revokes the stored token: DELETE /installation/token with it, which answers 204
// [revoke]. The key lives on at GitHub, so whoever holds it can still mint a token until the
// app's owner deletes it there.
func (s *Scheme) Revoke(ctx context.Context, stored core.StoredCredentials, m core.ResolvedManifest) error {
	current, err := open(stored)
	if err != nil {
		return err
	}
	response, body, err := s.send(ctx, m, http.MethodDelete, "/installation/token", "Bearer "+current.Token, nil)
	if err == nil && response.StatusCode == http.StatusNoContent {
		return nil
	}
	return refused("revoke", response, body, err, current.Token)
}

// Fingerprints names the token last minted in stored, by core.Fingerprint, with its expiry.
// The key is the connection's credential, not a token, so it is not named.
func (*Scheme) Fingerprints(stored core.StoredCredentials) (core.CredentialFingerprints, error) {
	p, err := open(stored)
	if err != nil {
		return core.CredentialFingerprints{}, err
	}
	return core.CredentialFingerprints{Access: core.Fingerprint(p.Token), AccessExpiresAt: p.ExpiresAt}, nil
}

// mint is POST /app/installations/{installation_id}/access_tokens with an app JWT [create],
// and the payload it leaves: app with the new token. A failure is a *core.OutcomeError.
func (s *Scheme) mint(ctx context.Context, m core.ResolvedManifest, app payload) (payload, error) {
	key, err := parseKey(app.PrivateKey)
	if err != nil {
		return payload{}, err
	}
	now := s.now()
	// [jwt]: iat, exp and iss, signed RS256. Nothing else is asked for.
	jwt, err := oauth2code.PrivateKeyJWT{Key: key, Alg: "RS256"}.Sign(map[string]any{
		"iat": now.Add(-jwtBackdate).Unix(),
		"exp": now.Add(jwtLifetime).Unix(),
		"iss": app.AppID,
	})
	if err != nil {
		return payload{}, stack.Wrap(fmt.Errorf("githubapp: sign the app JWT: %w", err))
	}
	var narrowing []byte
	if len(app.Permissions) > 0 || len(app.Repositories) > 0 {
		narrowing, err = json.Marshal(struct {
			Repositories []string          `json:"repositories,omitempty"`
			Permissions  map[string]string `json:"permissions,omitempty"`
		}{app.Repositories, app.Permissions})
		if err != nil {
			return payload{}, stack.Wrap(err)
		}
	}
	response, body, err := s.send(ctx, m, http.MethodPost, "/app/installations/"+app.InstallationID+"/access_tokens", "Bearer "+jwt, narrowing)
	if err == nil && response.StatusCode == http.StatusCreated {
		var token struct {
			Token     string    `json:"token"`
			ExpiresAt time.Time `json:"expires_at"`
		}
		if json.Unmarshal(body, &token) == nil && token.Token != "" && !token.ExpiresAt.IsZero() {
			next := app
			next.Token, next.ExpiresAt = token.Token, token.ExpiresAt
			return next, nil
		}
		// [create] answers 201 with token and expires_at. Nothing was spent, so asking again
		// may do better.
		return payload{}, &core.OutcomeError{Outcome: core.Outcome{Kind: core.OutcomeTransient},
			Err: stack.Wrap(errors.New("githubapp: mint: HTTP 201 without a readable token and expires_at"))}
	}
	return payload{}, refused("mint", response, body, err, jwt)
}

// check lists one repository with token, GET /installation/repositories?per_page=1 [list], so
// a token that cannot read its installation fails the connection before it is made.
func (s *Scheme) check(ctx context.Context, m core.ResolvedManifest, token string) error {
	response, body, err := s.send(ctx, m, http.MethodGet, "/installation/repositories?per_page=1", "Bearer "+token, nil)
	if err == nil && response.StatusCode == http.StatusOK {
		return nil
	}
	return refused("list repositories", response, body, err, token)
}

// send is one request to the manifest's api_base, with the headers [headers] asks for. The
// URL passes publicEndpoint first. body is JSON, or nil for none.
func (s *Scheme) send(ctx context.Context, m core.ResolvedManifest, method, path, authorization string, body []byte) (*http.Response, []byte, error) {
	base, err := apiBase(m)
	if err != nil {
		return nil, nil, err
	}
	endpoint := base + path
	if err := s.publicEndpoint(ctx, strings.SplitN(endpoint, "?", 2)[0]); err != nil {
		return nil, nil, stack.Wrap(fmt.Errorf("githubapp: %s is not a public https endpoint: %w", method, err))
	}
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	request, err := http.NewRequestWithContext(ctx, method, endpoint, reader)
	if err != nil {
		return nil, nil, stack.Wrap(err)
	}
	request.Header.Set("Authorization", authorization)
	request.Header.Set("Accept", "application/vnd.github+json")
	request.Header.Set("X-GitHub-Api-Version", apiVersion)
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := s.http.Do(request)
	if err != nil {
		return nil, nil, err
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(response.Body, maxBody))
	if err != nil {
		return response, nil, err
	}
	return response, raw, nil
}

// refused is the *core.OutcomeError for a request GitHub did not answer as hoped. The text
// is the status and GitHub's message with what the request carried redacted, never the key,
// the JWT or a token.
//
// For a request here: 401 is a JWT GitHub could not verify (a wrong key or app id) or a token
// it does not take; 404 an installation or token it does not know; 422 a narrowing it will not
// grant; 403 without a rate limit an installation that is suspended or lacks a permission
// asked for. Only a new key, installation or narrowing helps each, so each is InvalidGrant.
// A 403 or 429 with a rate limit [limits] is RateLimited; anything else is Transient, since a
// mint spends nothing.
func refused(what string, response *http.Response, body []byte, err error, secrets ...string) error {
	if response == nil {
		return &core.OutcomeError{Outcome: core.Outcome{Kind: core.OutcomeTransient}, Err: stack.Wrap(fmt.Errorf("githubapp: %s: %w", what, err))}
	}
	var answer struct {
		Message string `json:"message"`
	}
	_ = json.Unmarshal(body, &answer)
	message := core.LoggedDescription(answer.Message, secrets...)
	outcome := core.Outcome{Kind: core.OutcomeTransient, Description: message}
	switch status := response.StatusCode; {
	case status == http.StatusTooManyRequests || (status == http.StatusForbidden && rateLimited(response, answer.Message)):
		classified := oauth2code.ClassifyStatic(response, body, nil)
		outcome.Kind, outcome.RetryAfter = core.OutcomeRateLimited, classified.RetryAfter
	case status == http.StatusUnauthorized, status == http.StatusForbidden, status == http.StatusNotFound,
		status == http.StatusUnprocessableEntity:
		outcome.Kind = core.OutcomeInvalidGrant
	}
	text := fmt.Sprintf("githubapp: %s: GitHub answered HTTP %d", what, response.StatusCode)
	if message != "" {
		text += ": " + message
	}
	return &core.OutcomeError{Outcome: outcome, Err: stack.Wrap(errors.New(text))}
}

// rateLimited is a 403 [limits] says is a rate limit: x-ratelimit-remaining 0, a retry-after,
// or a secondary rate limit, which may carry neither header and says so in its message («an
// error message that indicates that you exceeded a secondary rate limit»).
func rateLimited(response *http.Response, message string) bool {
	return response.Header.Get("X-RateLimit-Remaining") == "0" || response.Header.Get("Retry-After") != "" ||
		strings.Contains(strings.ToLower(message), "rate limit")
}

// supplied reads and checks what Complete was given. Its errors name what is wrong, never the
// value.
func supplied(values map[string]string) (payload, error) {
	for name := range values {
		switch name {
		case SuppliedAppID, SuppliedPrivateKey, SuppliedInstallationID, SuppliedPermissions, SuppliedRepositories:
		default:
			return payload{}, stack.Wrap(fmt.Errorf("githubapp: %q is not a value %s takes; it takes %s, %s, %s, and optionally %s and %s",
				name, Name, SuppliedAppID, SuppliedPrivateKey, SuppliedInstallationID, SuppliedPermissions, SuppliedRepositories))
		}
	}
	p := payload{AppID: values[SuppliedAppID], PrivateKey: values[SuppliedPrivateKey], InstallationID: values[SuppliedInstallationID]}
	if raw := values[SuppliedPermissions]; raw != "" {
		if json.Unmarshal([]byte(raw), &p.Permissions) != nil {
			return payload{}, stack.Wrap(fmt.Errorf("githubapp: %s is not a JSON object of permission names to access levels", SuppliedPermissions))
		}
	}
	if raw := values[SuppliedRepositories]; raw != "" {
		for name := range strings.SplitSeq(raw, ",") {
			p.Repositories = append(p.Repositories, strings.TrimSpace(name))
		}
	}
	if err := p.check(); err != nil {
		return payload{}, err
	}
	return p, nil
}

// check refuses an app this scheme cannot mint with. Its errors name what is wrong, never the
// value.
func (p payload) check() error {
	switch {
	case !appID.MatchString(p.AppID):
		return stack.Wrap(fmt.Errorf("githubapp: %s is required: the app's numeric id or its client id", SuppliedAppID))
	case !installationID.MatchString(p.InstallationID):
		return stack.Wrap(fmt.Errorf("githubapp: %s is required: the installation's numeric id", SuppliedInstallationID))
	case len(p.Repositories) > maxRepositories:
		return stack.Wrap(fmt.Errorf("githubapp: %s names more than %d repositories", SuppliedRepositories, maxRepositories))
	}
	if _, err := parseKey(p.PrivateKey); err != nil {
		return err
	}
	for name, level := range p.Permissions {
		if !permissionName.MatchString(name) || !slices.Contains(permissionLevel, level) {
			return stack.Wrap(fmt.Errorf("githubapp: %s holds an entry that is not a permission name with read, write or admin", SuppliedPermissions))
		}
	}
	for _, name := range p.Repositories {
		if !repositoryName.MatchString(name) {
			return stack.Wrap(fmt.Errorf("githubapp: %s holds a name that is not a repository's name without its owner", SuppliedRepositories))
		}
	}
	return nil
}

// parseKey reads the app's private key: the PKCS#1 «RSA PRIVATE KEY» GitHub downloads [jwt],
// or the same key as PKCS#8. Its errors never quote the key.
func parseKey(raw string) (*rsa.PrivateKey, error) {
	block, _ := pem.Decode([]byte(strings.TrimSpace(raw)))
	if block == nil {
		return nil, stack.Wrap(fmt.Errorf("githubapp: %s is not a PEM private key", SuppliedPrivateKey))
	}
	if key, err := x509.ParsePKCS1PrivateKey(block.Bytes); err == nil {
		return key, nil
	}
	if parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes); err == nil {
		if key, ok := parsed.(*rsa.PrivateKey); ok {
			return key, nil
		}
	}
	return nil, stack.Wrap(fmt.Errorf("githubapp: %s is not an RSA private key in PKCS#1 or PKCS#8", SuppliedPrivateKey))
}

// apiBase is the manifest's endpoints.api_base, without a trailing slash.
func apiBase(m core.ResolvedManifest) (string, error) {
	base := strings.TrimSuffix(m.Endpoints["api_base"], "/")
	if base == "" {
		return "", stack.Wrap(ErrNoAPIBase)
	}
	return base, nil
}

// margin is the manifest's refresh.margin, else defaultMargin.
func margin(m core.ResolvedManifest) time.Duration {
	if m.Refresh.Margin > 0 {
		return time.Duration(m.Refresh.Margin)
	}
	return defaultMargin
}

// open reads the payload this scheme sealed. Its errors never quote the payload.
func open(stored core.StoredCredentials) (payload, error) {
	if stored.Scheme != Name || stored.Version != payloadVersion {
		return payload{}, stack.Wrap(fmt.Errorf("githubapp: stored credentials are %q version %d, not %q version %d", stored.Scheme, stored.Version, Name, payloadVersion))
	}
	var p payload
	if json.Unmarshal(stored.Payload, &p) != nil || p.Token == "" || p.ExpiresAt.IsZero() {
		return payload{}, stack.Wrap(errors.New("githubapp: stored credentials payload is unreadable or has no token"))
	}
	if err := p.check(); err != nil {
		return payload{}, err
	}
	return p, nil
}

func seal(p payload) (core.StoredCredentials, error) {
	raw, err := json.Marshal(p)
	if err != nil {
		return core.StoredCredentials{}, stack.Wrap(err)
	}
	return core.StoredCredentials{Scheme: Name, Version: payloadVersion, Payload: raw}, nil
}

// credential is the token in p as a core.AccessCredential, read back by Wrap. The key stays in
// the stored credentials: a request needs only the token.
func credential(p payload) core.AccessCredential {
	secret, _ := json.Marshal(map[string]string{"token": p.Token})
	return core.NewAccessCredential(Name, p.ExpiresAt, secret)
}
