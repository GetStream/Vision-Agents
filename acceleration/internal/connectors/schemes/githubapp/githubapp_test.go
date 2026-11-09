package githubapp_test

import (
	"bytes"
	"context"
	"crypto/rsa"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core/contracttest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/githubapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

func TestGitHubAppSchemeContract(t *testing.T) {
	suite.Run(t, &contracttest.SchemeContract{New: func(t *testing.T, _ *slog.Logger) contracttest.Subject {
		key, pemKey := newKey(t)
		clock := &clock{now: time.Now()}
		gh := newGitHub(t, key, clock)
		scheme, err := githubapp.New(githubapp.Config{HTTP: gh.Client(), Now: clock.Now, PublicEndpoint: loopbackOrPublic})
		if err != nil {
			t.Fatal(err)
		}
		return contracttest.Subject{
			Scheme:    scheme,
			Manifest:  manifest(gh),
			Supplied:  supplied(pemKey),
			Transport: gh.Client().Transport,
			Call:      func() *http.Request { return call(gh) },
			Secrets: func(stored core.StoredCredentials) []string {
				return []string{token(stored), pemKey, keyBody(pemKey)}
			},
			Expire: func() { clock.Advance(tokenLifetime) },
			// The first mint is Complete's; a renewal is every one after it.
			Renewals: func() int { return gh.mintCount() - 1 },
		}
	}})
}

// GitHubAppSuite covers what is the scheme's own: the app JWT, when a token is minted, what
// narrows it, what each refusal means and that no secret leaves.
type GitHubAppSuite struct {
	suite.Suite
	ctx    context.Context
	clock  *clock
	key    *rsa.PrivateKey
	pemKey string
	gh     *github
	scheme *githubapp.Scheme
	logs   *bytes.Buffer
}

func TestGitHubAppSuite(t *testing.T) {
	suite.Run(t, new(GitHubAppSuite))
}

func (s *GitHubAppSuite) SetupTest() {
	s.ctx = context.Background()
	s.clock = &clock{now: time.Date(2026, 11, 9, 12, 0, 0, 0, time.UTC)}
	s.key, s.pemKey = newKey(s.T())
	s.gh = newGitHub(s.T(), s.key, s.clock)
	scheme, err := githubapp.New(githubapp.Config{HTTP: s.gh.Client(), Now: s.clock.Now, PublicEndpoint: loopbackOrPublic})
	s.Require().NoError(err)
	s.scheme = scheme
	s.logs = &bytes.Buffer{}
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(s.logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	s.T().Cleanup(func() { slog.SetDefault(previous) })
}

func (s *GitHubAppSuite) TestCompleteMintsATokenWithAnAppJWTAsGitHubAsksForIt() {
	stored := s.connect(supplied(s.pemKey))

	_, claims := s.gh.lastMint()
	s.Equal(appID, claims["iss"])
	s.InDelta(s.clock.Now().Add(-time.Minute).Unix(), claims["iat"], 0, "iat 60 s in the past, for clock drift")
	s.LessOrEqual(claims["exp"], float64(s.clock.Now().Add(10*time.Minute).Unix()), "exp at most 10 minutes out")
	s.Len(claims, 3, "only iat, exp and iss")

	credential, _, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(s.clock.Now().Add(tokenLifetime), credential.ExpiresAt, "the token's expires_at")
	s.Equal(http.StatusOK, s.status(credential))
}

func (s *GitHubAppSuite) TestAKeyInPKCS8IsTaken() {
	der, err := x509PKCS8(s.key)
	s.Require().NoError(err)
	values := supplied(der)
	_, _, err = s.scheme.Complete(s.ctx, core.CompleteInput{Manifest: manifest(s.gh), Supplied: values})
	s.NoError(err)
}

func (s *GitHubAppSuite) TestATokenOutsideTheMarginIsHandedOutAsStored() {
	stored := s.connect(supplied(s.pemKey))
	s.clock.Advance(tokenLifetime - 6*time.Minute)

	credential, again, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(stored, again, "nothing to persist")
	s.Equal(1, s.gh.mintCount(), "no second mint")
	s.Equal(token(stored), secretOf(credential))
}

func (s *GitHubAppSuite) TestATokenInsideTheMarginIsMintedAgainBeforeItExpires() {
	stored := s.connect(supplied(s.pemKey))
	s.clock.Advance(tokenLifetime - 4*time.Minute)

	credential, renewed, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(2, s.gh.mintCount())
	s.NotEqual(token(stored), token(renewed))
	s.Equal(token(renewed), secretOf(credential))
	s.Equal(s.clock.Now().Add(tokenLifetime), credential.ExpiresAt)
	s.Equal(http.StatusOK, s.status(credential))
}

func (s *GitHubAppSuite) TestATokenThatExpiresBeforeValidUntilIsMintedAgain() {
	stored := s.connect(supplied(s.pemKey))

	_, renewed, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{ValidUntil: s.clock.Now().Add(2 * tokenLifetime)})
	s.Require().NoError(err)
	s.NotEqual(token(stored), token(renewed))
}

func (s *GitHubAppSuite) TestARefusedTokenIsMintedAgainWhateverItsExpiry() {
	stored := s.connect(supplied(s.pemKey))

	_, renewed, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{Refused: true})
	s.Require().NoError(err)
	s.NotEqual(token(stored), token(renewed))
}

func (s *GitHubAppSuite) TestAKeyGitHubDoesNotTakeIsInvalidGrantAtComplete() {
	_, otherKey := newKey(s.T())

	_, _, err := s.scheme.Complete(s.ctx, core.CompleteInput{Manifest: manifest(s.gh), Supplied: supplied(otherKey)})
	s.Equal(core.OutcomeInvalidGrant, outcome(err))
	s.Contains(err.Error(), "HTTP 401")
	s.Contains(err.Error(), "A JSON web token could not be decoded", "GitHub's message, for the caller")
	s.NotContains(err.Error(), "eyJ", "never the JWT GitHub echoed")
}

func (s *GitHubAppSuite) TestAKeyRotatedAwayAtGitHubIsInvalidGrantWhenTheTokenIsDue() {
	stored := s.connect(supplied(s.pemKey))
	rotated, _ := newKey(s.T())
	s.gh.rotate(rotated)
	s.clock.Advance(tokenLifetime)

	credential, renewed, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{})
	s.Equal(core.OutcomeInvalidGrant, outcome(err), "only a new key helps, so the connection needs one")
	s.Empty(credential.Secret(), "the old token has expired, so none is handed out")
	s.Zero(renewed)
}

func (s *GitHubAppSuite) TestAFailedMintHandsOutTheOldTokenWhileItWorks() {
	stored := s.connect(supplied(s.pemKey))
	s.gh.limit()
	s.clock.Advance(tokenLifetime - time.Minute)

	credential, renewed, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{})
	s.Equal(core.OutcomeRateLimited, outcome(err), "a 403 with x-ratelimit-remaining 0 is a rate limit")
	s.Equal(token(stored), secretOf(credential))
	s.Zero(renewed)
}

func (s *GitHubAppSuite) TestASecondaryRateLimitWithoutHeadersIsRateLimitedNotARefusedKey() {
	stored := s.connect(supplied(s.pemKey))
	s.gh.limitSecondary()
	s.clock.Advance(tokenLifetime)

	_, _, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{})
	s.Equal(core.OutcomeRateLimited, outcome(err), "a reconnect would not help")
}

func (s *GitHubAppSuite) TestASuspendedInstallationIsInvalidGrant() {
	stored := s.connect(supplied(s.pemKey))
	s.gh.suspend()
	s.clock.Advance(tokenLifetime)

	_, _, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{})
	s.Equal(core.OutcomeInvalidGrant, outcome(err))
	s.Contains(err.Error(), "HTTP 403: This installation has been suspended")
}

func (s *GitHubAppSuite) TestAnInstallationGitHubDoesNotKnowIsInvalidGrant() {
	values := supplied(s.pemKey)
	values[githubapp.SuppliedInstallationID] = "99"

	_, _, err := s.scheme.Complete(s.ctx, core.CompleteInput{Manifest: manifest(s.gh), Supplied: values})
	s.Equal(core.OutcomeInvalidGrant, outcome(err))
	s.Contains(err.Error(), "HTTP 404")
}

func (s *GitHubAppSuite) TestPermissionsAndRepositoriesNarrowTheToken() {
	values := supplied(s.pemKey)
	values[githubapp.SuppliedPermissions] = `{"contents":"read","pull_requests":"write"}`
	values[githubapp.SuppliedRepositories] = "hello"
	stored := s.connect(values)

	body, _ := s.gh.lastMint()
	s.JSONEq(`{"repositories":["hello"],"permissions":{"contents":"read","pull_requests":"write"}}`, string(body))

	s.clock.Advance(tokenLifetime)
	_, _, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{})
	s.Require().NoError(err)
	again, _ := s.gh.lastMint()
	s.JSONEq(string(body), string(again), "a renewal narrows as Complete did")
}

func (s *GitHubAppSuite) TestWithoutNarrowingTheMintSendsNoBody() {
	s.connect(supplied(s.pemKey))

	body, _ := s.gh.lastMint()
	s.Empty(body, "the token gets every permission and repository of the installation")
}

func (s *GitHubAppSuite) TestANarrowingGitHubWillNotGrantIsRefusedAtComplete() {
	values := supplied(s.pemKey)
	values[githubapp.SuppliedRepositories] = "hello, not-installed"

	_, _, err := s.scheme.Complete(s.ctx, core.CompleteInput{Manifest: manifest(s.gh), Supplied: values})
	s.Equal(core.OutcomeInvalidGrant, outcome(err))
	s.Contains(err.Error(), "HTTP 422")
}

func (s *GitHubAppSuite) TestValuesItCannotUseAreRefusedBeforeAnythingIsSent() {
	for name, value := range map[string]string{
		githubapp.SuppliedAppID:          "42/../../users",
		githubapp.SuppliedInstallationID: "1337/../../x",
		githubapp.SuppliedPrivateKey:     "not a key",
		githubapp.SuppliedPermissions:    `{"contents":"everything"}`,
		githubapp.SuppliedRepositories:   "octo/hello",
	} {
		values := supplied(s.pemKey)
		values[name] = value
		_, _, err := s.scheme.Complete(s.ctx, core.CompleteInput{Manifest: manifest(s.gh), Supplied: values})
		s.ErrorContains(err, name, "the error names %s", name)
		s.NotContains(err.Error(), value)
	}
	values := supplied(s.pemKey)
	values["client_secret"] = "x"
	_, _, err := s.scheme.Complete(s.ctx, core.CompleteInput{Manifest: manifest(s.gh), Supplied: values})
	s.ErrorContains(err, `"client_secret" is not a value github_app takes`)
	s.Zero(s.gh.mintCount())
}

func (s *GitHubAppSuite) TestRevokeEndsTheToken() {
	stored := s.connect(supplied(s.pemKey))
	credential, _, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{})
	s.Require().NoError(err)

	s.Require().NoError(s.scheme.Revoke(s.ctx, stored, manifest(s.gh)))
	s.Equal(http.StatusUnauthorized, s.status(credential))
}

func (s *GitHubAppSuite) TestTheSandboxGetsTheTokenAsABearerToken() {
	stored := s.connect(supplied(s.pemKey))
	credential, _, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{})
	s.Require().NoError(err)

	sent, err := sandbox.BearerToken(s.ctx, s.scheme, credential, "api.github.com")
	s.Require().NoError(err)
	s.Equal(token(stored), sent, "the token the brokers put in GITHUB_TOKEN's place, as Bearer or as git's Basic password")
}

func (s *GitHubAppSuite) TestNoSecretReachesALogAnErrorOrAPrintedValue() {
	stored := s.connect(supplied(s.pemKey))
	credential, _, err := s.scheme.Retrieve(s.ctx, stored, manifest(s.gh), core.RetrieveOptions{})
	s.Require().NoError(err)
	_, otherKey := newKey(s.T())
	_, _, refused := s.scheme.Complete(s.ctx, core.CompleteInput{Manifest: manifest(s.gh), Supplied: supplied(otherKey)})
	s.Require().Error(refused)
	fingerprints, err := s.scheme.Fingerprints(stored)
	s.Require().NoError(err)
	slog.Info("connected", "stored", stored, "credential", credential, "fingerprints", fingerprints, "error", refused)

	_, claims := s.gh.lastMint()
	s.Require().NotEmpty(claims)
	printed := strings.Join([]string{
		s.logs.String(), refused.Error(),
		fmt.Sprintf("%v %+v %#v %s", stored, stored, stored, stored),
		fmt.Sprintf("%v %+v %#v %s", credential, credential, credential, credential),
		fmt.Sprintf("%v %+v", fingerprints, fingerprints),
	}, "\n")
	for _, secret := range []string{token(stored), keyBody(s.pemKey), keyBody(otherKey)} {
		s.NotContains(printed, secret)
	}
	s.NotContains(printed, "eyJ", "no JWT: its header always starts so")
}

func (s *GitHubAppSuite) TestTheGitHubManifestTakesGitHubAppAndKeepsOAuthAsItsDefault() {
	raw, err := fs.ReadFile(providers.FS, "github.yaml")
	s.Require().NoError(err)
	parsed, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	s.Equal("oauth2_code", parsed.Schemes[0], "a connection naming no scheme still consents")

	resolved, err := parsed.Resolve(githubapp.Name, nil, nil)
	s.Require().NoError(err)
	s.Equal("https://api.github.com", resolved.Endpoints["api_base"], "where the scheme mints")
	s.Equal("https://api.githubcopilot.com/mcp/", resolved.Endpoints["mcp"])
	s.Equal("GITHUB_TOKEN", resolved.Sandbox.Env)
	out, err := s.scheme.Begin(s.ctx, core.BeginInput{Manifest: resolved})
	s.Require().NoError(err)
	s.True(out.Done)
}

// connect completes the scheme with values against the fake and returns what it stored.
func (s *GitHubAppSuite) connect(values map[string]string) core.StoredCredentials {
	out, err := s.scheme.Begin(s.ctx, core.BeginInput{Manifest: manifest(s.gh)})
	s.Require().NoError(err)
	s.Require().True(out.Done)
	stored, account, err := s.scheme.Complete(s.ctx, core.CompleteInput{Manifest: manifest(s.gh), Supplied: values})
	s.Require().NoError(err)
	s.Equal(core.AccountInfo{}, account)
	return stored
}

// status is what the fake answers a request wrapped with credential.
func (s *GitHubAppSuite) status(credential core.AccessCredential) int {
	client := &http.Client{Transport: s.scheme.Wrap(s.gh.Client().Transport, credential)}
	response, err := client.Do(call(s.gh))
	s.Require().NoError(err)
	defer response.Body.Close()
	return response.StatusCode
}

func manifest(gh *github) core.ResolvedManifest {
	return core.ResolvedManifest{ConnectorID: "custom_github", Scheme: githubapp.Name, Endpoints: map[string]string{"api_base": gh.URL}}
}

func supplied(pemKey string) map[string]string {
	return map[string]string{
		githubapp.SuppliedAppID:          appID,
		githubapp.SuppliedPrivateKey:     pemKey,
		githubapp.SuppliedInstallationID: installationID,
	}
}

func call(gh *github) *http.Request {
	request, _ := http.NewRequest(http.MethodGet, gh.URL+"/repos/octo/hello", nil)
	return request
}

func token(stored core.StoredCredentials) string {
	var p struct {
		Token string `json:"token"`
	}
	_ = json.Unmarshal(stored.Payload, &p)
	return p.Token
}

func secretOf(c core.AccessCredential) string {
	var secret struct {
		Token string `json:"token"`
	}
	_ = json.Unmarshal(c.Secret(), &secret)
	return secret.Token
}

// keyBody is one line of the key's base64, which any copy of the key holds whatever its
// line breaks became.
func keyBody(pemKey string) string {
	return strings.Split(pemKey, "\n")[3]
}

func outcome(err error) core.OutcomeKind {
	var refused *core.OutcomeError
	if errors.As(err, &refused) {
		return refused.Outcome.Kind
	}
	return ""
}

// loopbackOrPublic lets the local fake's loopback address through and checks everything else
// as the router does.
func loopbackOrPublic(ctx context.Context, raw string) error {
	if u, err := url.Parse(raw); err == nil {
		if ip, err := netip.ParseAddr(u.Hostname()); err == nil && ip.IsLoopback() && u.User == nil {
			return nil
		}
	}
	return egress.ValidatePublicHTTPSURL(ctx, raw)
}
