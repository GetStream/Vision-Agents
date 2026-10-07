// Package contracttest holds the suites every adapter of one kind runs, so what the core
// relies on is proved once and checked for each adapter. SchemeContract is the one for
// core.Scheme, SourceContract the one for core.ToolSource. It is test code: import it only
// from _test.go files.
package contracttest

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// Subject is one scheme under SchemeContract, with what the contract cannot know about it:
// how to get through acquisition, where to send a request that needs the credential, which
// values are secret.
type Subject struct {
	Scheme   core.Scheme
	Manifest core.ResolvedManifest
	// Ref is the connection the contract acquires for; zero is a fixed one.
	Ref core.ConnectionRef
	// RedirectURI is what Begin gets, for an interactive scheme.
	RedirectURI string
	// Supplied is what Complete gets, for a scheme whose Begin is Done.
	Supplied map[string]string
	// Consent is the browser, for an interactive scheme: it follows the authorize URL Begin
	// returned and returns the callback query. Nil when Begin is Done.
	Consent func(authorizeURL string) (url.Values, error)
	// Transport is what a request leaves through after Wrap and the contract's look at it:
	// the client transport of the provider the Call goes to.
	Transport http.RoundTripper
	// Call is a request to the provider that it answers with a 2xx when the credential is on
	// it, and without one when it is not (unless Anonymous).
	Call func() *http.Request
	// Anonymous says the scheme applies nothing: its access credential holds no secret and
	// Wrap leaves the request as it is.
	Anonymous bool
	// Secrets are every secret behind stored: what Supplied gave and what the provider
	// issued. Nothing the scheme returns, sends in a URL or logs may contain one.
	Secrets func(stored core.StoredCredentials) []string
	// Expire makes the access credential due for renewal, moving the scheme's and the
	// provider's clocks. Nil for a credential that never expires.
	Expire func()
	// Renewals counts the renewals that reached the provider; nil when there are none to
	// count.
	Renewals func() int
	// RevokeErr is what Revoke returns for this subject, matched with errors.Is; nil means
	// it revokes, so the provider refuses the credential afterwards.
	RevokeErr error
}

// SchemeContract is the suite every core.Scheme passes. Run it with suite.Run and a New that
// builds a fresh Subject for each test; logger is the one the scheme should log to, and is
// also slog's default while the test runs, so every line the scheme writes is read for
// secrets.
type SchemeContract struct {
	suite.Suite
	New func(t *testing.T, logger *slog.Logger) Subject

	ctx      context.Context
	subject  Subject
	logs     *lockedBuffer
	previous *slog.Logger
	// secrets are every secret the test has seen, and public every text that must hold none:
	// errors, URLs. Both are checked once the test is over, since a provider issues some
	// secrets after the URL that led to them was built.
	secrets []string
	public  []string
}

// concurrency is how many resolves the concurrency test runs at once: enough to overlap, few
// enough to stay fast. A choice, not a measured number.
const concurrency = 8

// identifier is the shape a scheme name must have to be listed under a manifest's schemes
// (identifier in core/manifest.go).
var identifier = regexp.MustCompile(`^[a-z][a-z0-9_]*$`)

func (s *SchemeContract) SetupTest() {
	s.ctx = context.Background()
	s.logs = &lockedBuffer{}
	s.secrets, s.public = nil, nil
	// Both handlers, since the JSON one marshals a value and the text one formats it, and a
	// secret can slip out through either.
	logger := slog.New(slog.NewMultiHandler(
		slog.NewJSONHandler(s.logs, &slog.HandlerOptions{Level: slog.LevelDebug}),
		slog.NewTextHandler(s.logs, &slog.HandlerOptions{Level: slog.LevelDebug}),
	))
	s.previous = slog.Default()
	slog.SetDefault(logger)
	s.subject = s.New(s.T(), logger)
	s.Require().NotNil(s.subject.Scheme, "Subject.Scheme")
	s.Require().NotNil(s.subject.Transport, "Subject.Transport")
	s.Require().NotNil(s.subject.Call, "Subject.Call")
	s.Require().NotNil(s.subject.Secrets, "Subject.Secrets")
	if s.subject.Ref == (core.ConnectionRef{}) {
		s.subject.Ref = core.ConnectionRef{CustomerID: "contract", ConnectionID: "contract-1"}
	}
}

// TearDownTest is where the secrets are looked for, in each of their forms: in every error,
// URL and log line the test produced.
func (s *SchemeContract) TearDownTest() {
	slog.SetDefault(s.previous)
	logs := s.logs.String()
	for _, secret := range s.secrets {
		for _, form := range forms(secret) {
			s.NotContains(logs, form, "a log line holds a secret")
			for _, text := range s.public {
				s.NotContains(text, form, "an error or a URL holds a secret")
			}
		}
	}
}

func (s *SchemeContract) TestARoundTripPutsTheCredentialOnTheRequest() {
	s.Regexp(identifier, s.subject.Scheme.Name(), "a manifest can list it under schemes")
	stored := s.connect()
	s.Equal(s.subject.Scheme.Name(), stored.Scheme)

	credential, again := s.retrieve(stored)
	s.Equal(s.subject.Scheme.Name(), credential.Scheme)
	s.True(same(stored, again), "a credential that is not due comes back as stored, so the resolver persists nothing")
	s.Equal(s.subject.Expire == nil, credential.ExpiresAt.IsZero(), "a credential has an expiry exactly when it can be due")

	status, sent := s.wrapped(credential)
	s.True(ok(status), "the provider takes the wrapped request, got %d", status)
	s.Require().Len(sent, 1)
	if s.subject.Anonymous {
		s.Empty(credential.Secret(), "a scheme that applies nothing holds nothing")
		s.Equal(s.subject.Call().Header, sent[0].Header, "Wrap left the request as it was")
		return
	}
	s.NotEmpty(credential.Secret())
	s.False(ok(s.status(s.subject.Transport, s.subject.Call())), "without Wrap the provider refuses, so the credential is what made it pass")
}

func (s *SchemeContract) TestConcurrentResolvesCommitAtMostOneNewRevision() {
	stored := s.connect()
	due := s.subject.Expire != nil
	if due {
		s.subject.Expire()
	}
	store := &lockedStore{state: core.CredentialState{Revision: 1, Status: "connected", Credentials: stored}}

	var wg sync.WaitGroup
	credentials := make([]core.AccessCredential, concurrency)
	errs := make([]error, concurrency)
	for i := range concurrency {
		wg.Go(func() {
			errs[i] = store.Update(s.ctx, s.subject.Ref, func(state *core.CredentialState, checkpoint func() error) (bool, error) {
				credential, next, err := s.subject.Scheme.Retrieve(s.ctx, state.Credentials, s.subject.Manifest, core.RetrieveOptions{Checkpoint: checkpoint})
				if err != nil {
					return false, err
				}
				credentials[i] = credential
				if same(state.Credentials, next) {
					return false, nil
				}
				state.Credentials, state.Revision = next, state.Revision+1
				return true, nil
			})
		})
	}
	wg.Wait()
	s.remember(store.state.Credentials)
	for i := range concurrency {
		s.Require().NoError(errs[i])
		status, _ := s.wrapped(credentials[i])
		s.True(ok(status), "every resolve got a credential the provider takes, got %d", status)
	}
	if !due {
		s.Equal(1, store.state.Revision, "a credential that is not due is never rewritten")
		return
	}
	s.Equal(2, store.state.Revision, "one renewal, and the resolves after it found it done")
	if s.subject.Renewals != nil {
		s.Equal(1, s.subject.Renewals(), "one renewal reached the provider")
	}
}

func (s *SchemeContract) TestConcurrentRetrievesOfACredentialThatIsNotDueAllHandItOutAsStored() {
	stored := s.connect()

	var wg sync.WaitGroup
	unchanged := make([]bool, concurrency)
	errs := make([]error, concurrency)
	for i := range concurrency {
		wg.Go(func() {
			_, again, err := s.subject.Scheme.Retrieve(s.ctx, stored, s.subject.Manifest, core.RetrieveOptions{})
			unchanged[i], errs[i] = same(stored, again), err
		})
	}
	wg.Wait()
	for i := range concurrency {
		s.Require().NoError(errs[i])
		s.True(unchanged[i], "nothing to persist, whoever asks first")
	}
}

// The resolver asks for a credential that still works when the call ends (ValidUntil). One
// that expires before then is renewed, with at most one checkpoint, before the renewal
// reached the provider; one that never expires outlives any call and comes back as stored.
func (s *SchemeContract) TestACredentialThatExpiresBeforeValidUntilIsRenewed() {
	stored := s.connect()
	credential, _ := s.retrieve(stored)
	validUntil := credential.ExpiresAt.Add(time.Second)
	if credential.ExpiresAt.IsZero() {
		// No time is past an expiry that never comes; a year from now stands in. A choice.
		validUntil = time.Now().AddDate(1, 0, 0)
	}
	checkpoints, renewalsAtCheckpoint := 0, 0
	renewed, next, err := s.subject.Scheme.Retrieve(s.ctx, stored, s.subject.Manifest, core.RetrieveOptions{
		ValidUntil: validUntil,
		Checkpoint: func() error {
			checkpoints++
			if s.subject.Renewals != nil {
				renewalsAtCheckpoint = s.subject.Renewals()
			}
			return nil
		},
	})
	s.Require().NoError(err)
	s.remember(next)
	if s.subject.Expire == nil {
		s.True(same(stored, next), "a credential that never expires is not renewed")
		s.Zero(checkpoints, "nothing that cannot be taken back was sent")
		return
	}
	s.False(same(stored, next), "renewed, so the resolver persists the new stored credentials")
	s.LessOrEqual(checkpoints, 1, "at most one checkpoint for one renewal")
	if s.subject.Renewals != nil {
		s.Equal(1, s.subject.Renewals(), "one renewal reached the provider")
		s.Zero(renewalsAtCheckpoint, "the checkpoint came before the renewal left")
	}
	status, _ := s.wrapped(renewed)
	s.True(ok(status), "the provider takes the renewed credential, got %d", status)
}

func (s *SchemeContract) TestStoredCredentialsItCannotReadAreRefusedWithoutQuotingThem() {
	stored := s.connect()
	unreadable := map[string]core.StoredCredentials{
		"another scheme's":         {Scheme: "contract_other", Version: stored.Version, Payload: stored.Payload},
		"a version it never wrote": {Scheme: stored.Scheme, Version: stored.Version + 1, Payload: stored.Payload},
		"a cut-off payload":        {Scheme: stored.Scheme, Version: stored.Version, Payload: stored.Payload[:len(stored.Payload)-1]},
	}
	for name, bad := range unreadable {
		_, _, err := s.subject.Scheme.Retrieve(s.ctx, bad, s.subject.Manifest, core.RetrieveOptions{})
		s.Require().Error(err, "Retrieve of %s stored credentials", name)
		s.say(err)
		err = s.subject.Scheme.Revoke(s.ctx, bad, s.subject.Manifest)
		s.Require().Error(err, "Revoke of %s stored credentials", name)
		s.say(err)
	}
}

func (s *SchemeContract) TestAnErrorAboutAnUnusableValueDoesNotQuoteIt() {
	stored := s.connect()
	// A supplied value with CR LF after it cannot be a header value (RFC 9110 section 5.5),
	// so a scheme that puts it in one refuses it. Whether it refuses or not, an error does
	// not repeat it.
	for name, value := range s.subject.Supplied {
		supplied := make(map[string]string, len(s.subject.Supplied))
		for k, v := range s.subject.Supplied {
			supplied[k] = v
		}
		supplied[name] = value + "\r\n"
		out, err := s.subject.Scheme.Begin(s.ctx, s.begin())
		s.Require().NoError(err)
		_, _, err = s.subject.Scheme.Complete(s.ctx, core.CompleteInput{Ref: s.subject.Ref, Manifest: s.subject.Manifest, State: out.State, Supplied: supplied})
		s.say(err)
	}
	if s.subject.Consent == nil {
		return
	}
	// A callback made of the secrets, as a forged one could be.
	forged := url.Values{}
	for _, secret := range s.subject.Secrets(stored) {
		forged.Add("state", secret)
		forged.Add("code", secret)
	}
	out, err := s.subject.Scheme.Begin(s.ctx, s.begin())
	s.Require().NoError(err)
	_, _, err = s.subject.Scheme.Complete(s.ctx, core.CompleteInput{Ref: s.subject.Ref, Manifest: s.subject.Manifest, State: out.State, Query: forged})
	s.Require().Error(err, "a callback that is not the provider's is refused")
	s.say(err)
}

func (s *SchemeContract) TestWrapAppliesOnlyACredentialThisSchemeIssued() {
	stored := s.connect()
	credential, _ := s.retrieve(stored)
	// Characters forms writes differently, so the scan below finds it in any of them.
	foreign := `contract-foreign~+/=;,@:" \` + random()
	s.secrets = append(s.secrets, foreign)
	shaped, err := json.Marshal(map[string]string{
		"access_token": foreign, "token": foreign, "key": foreign, "api_key": foreign, "header": "X-Contract-Foreign",
	})
	s.Require().NoError(err)

	for _, other := range []core.AccessCredential{
		// This scheme's own secret, under another scheme's name.
		core.NewAccessCredential("contract_other", time.Time{}, credential.Secret()),
		// A secret in every shape a scheme here might read, under another scheme's name.
		core.NewAccessCredential("contract_other", time.Time{}, shaped),
	} {
		sent := &recorder{next: s.subject.Transport}
		response, err := (&http.Client{Transport: s.subject.Scheme.Wrap(sent, other)}).Do(s.subject.Call())
		if err == nil {
			s.Require().NoError(drain(response))
		}
		s.say(err)
		for _, r := range sent.requests() {
			for _, secret := range s.secrets {
				for _, form := range forms(secret) {
					s.NotContains(r.URL.String(), form, "another scheme's credential went on the wire")
				}
				for _, values := range r.Header {
					for _, value := range values {
						s.NotContains(value, secret, "another scheme's credential went on the wire")
					}
				}
			}
		}
	}
}

func (s *SchemeContract) TestClassifyMakesASuccessOK() {
	s.Equal(core.Outcome{Kind: core.OutcomeOK}, s.classify(s.answer("/ok")))
}

// RFC 6750 section 3.1: invalid_token is a token «expired, revoked, malformed, or invalid
// for other reasons», answered with 401. Only a reconnect helps.
func (s *SchemeContract) TestClassifyMakesAnInvalidTokenChallengeInvalidGrant() {
	s.Equal(core.Outcome{Kind: core.OutcomeInvalidGrant}, s.classify(s.answer("/invalid-token")))
}

// A 401 to a request that carried the scheme's own credential, with a challenge that names
// no error: what an MCP server answers a token it does not take («Invalid or expired tokens
// MUST receive a HTTP 401 response», with resource_metadata in WWW-Authenticate; MCP
// 2026-07-28, Authorization, «Token Handling» and the example under «Scope Selection
// Strategy»). The provider refused the credential, so only a renewed or new one helps.
func (s *SchemeContract) TestClassifyMakesABare401ToTheCredentialInvalidGrant() {
	credential, _ := s.retrieve(s.connect())
	server := httptest.NewServer(http.HandlerFunc(answers))
	s.T().Cleanup(server.Close)
	request, err := http.NewRequestWithContext(s.ctx, http.MethodPost, server.URL+"/bare-challenge", strings.NewReader(`{}`))
	s.Require().NoError(err)
	// A connection of its own, as answer has it.
	client := &http.Client{Transport: s.subject.Scheme.Wrap(&http.Transport{DisableKeepAlives: true}, credential)}
	response, err := client.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)

	s.Equal(core.Outcome{Kind: core.OutcomeInvalidGrant}, s.classify(response, body, err))
}

// RFC 6750 section 3.1: insufficient_scope means «the request requires higher privileges
// than provided by the access token», answered with 403, and scope names what is needed.
func (s *SchemeContract) TestClassifyMakesAnInsufficientScopeChallengeScopeRequired() {
	s.Equal(core.Outcome{Kind: core.OutcomeScopeRequired, Scopes: []string{"files:read", "files:write"}}, s.classify(s.answer("/insufficient-scope")))
}

// RFC 6585 section 4 for 429, RFC 9110 section 10.2.3 for Retry-After in delay-seconds.
func (s *SchemeContract) TestClassifyMakesA429RateLimitedWithItsRetryAfter() {
	s.Equal(core.Outcome{Kind: core.OutcomeRateLimited, RetryAfter: 30 * time.Second}, s.classify(s.answer("/rate-limited")))
}

// RFC 9110 section 15.6.4: a 503 server «is currently unable to handle the request». A
// refused dial never reached anyone.
func (s *SchemeContract) TestClassifyMakesAFailureThatChangedNothingTransient() {
	s.Equal(core.OutcomeTransient, s.classify(s.answer("/unavailable")).Kind, "a 503")

	closed := httptest.NewServer(http.NotFoundHandler())
	closed.Close()
	response, err := closed.Client().Get(closed.URL)
	s.Equal(core.OutcomeTransient, s.classify(response, nil, err).Kind, "a refused dial")
}

// RFC 9110 section 15.6.1: a 500 does not say nothing happened. A connection that closes
// after the request reached the server loses an answer to something that may have been done.
func (s *SchemeContract) TestClassifyMakesAnAnswerThatMayHaveTakenEffectUncertain() {
	s.Equal(core.OutcomeUncertain, s.classify(s.answer("/server-error")).Kind, "a 500")
	s.Equal(core.OutcomeUncertain, s.classify(s.answer("/lost")).Kind, "a connection closed after the request")
}

// Revoke is honest: nil means the provider no longer takes the credential, and an error
// that says nothing was revoked leaves a credential the provider still takes.
func (s *SchemeContract) TestRevokeSaysWhatItDid() {
	stored := s.connect()
	credential, _ := s.retrieve(stored)

	err := s.subject.Scheme.Revoke(s.ctx, stored, s.subject.Manifest)
	s.say(err)
	status, _ := s.wrapped(credential)
	if s.subject.RevokeErr == nil {
		s.Require().NoError(err)
		if !s.subject.Anonymous {
			s.False(ok(status), "Revoke said it revoked, so the provider refuses the credential, got %d", status)
		}
		return
	}
	s.Require().ErrorIs(err, s.subject.RevokeErr)
	var outcome *core.OutcomeError
	s.False(errors.As(err, &outcome), "an answer no retry changes is not an outcome to retry")
	s.True(ok(status), "Revoke said it revoked nothing, so the provider still takes the credential, got %d", status)
}

// connect is a whole acquisition: Begin, the browser when there is one, Complete.
func (s *SchemeContract) connect() core.StoredCredentials {
	out, err := s.subject.Scheme.Begin(s.ctx, s.begin())
	s.Require().NoError(err)
	in := core.CompleteInput{Ref: s.subject.Ref, Manifest: s.subject.Manifest, State: out.State, Supplied: s.subject.Supplied}
	if s.subject.Consent == nil {
		s.Require().True(out.Done, "Begin is Done for a scheme with no Consent")
		s.Require().Empty(out.AuthorizeURL)
	} else {
		s.Require().False(out.Done, "Begin is not Done for a scheme with a Consent")
		s.Require().NotEmpty(out.AuthorizeURL)
		s.public = append(s.public, out.AuthorizeURL)
		in.Query, err = s.subject.Consent(out.AuthorizeURL)
		s.Require().NoError(err)
	}
	stored, _, err := s.subject.Scheme.Complete(s.ctx, in)
	s.Require().NoError(err)
	s.remember(stored)
	return stored
}

func (s *SchemeContract) begin() core.BeginInput {
	return core.BeginInput{Ref: s.subject.Ref, Manifest: s.subject.Manifest, RedirectURI: s.subject.RedirectURI}
}

func (s *SchemeContract) retrieve(stored core.StoredCredentials) (core.AccessCredential, core.StoredCredentials) {
	credential, again, err := s.subject.Scheme.Retrieve(s.ctx, stored, s.subject.Manifest, core.RetrieveOptions{})
	s.Require().NoError(err)
	return credential, again
}

// wrapped sends the subject's Call through Wrap and returns its status and what left Wrap. It
// also proves Wrap leaves the caller's request alone.
func (s *SchemeContract) wrapped(credential core.AccessCredential) (int, []*http.Request) {
	sent := &recorder{next: s.subject.Transport}
	request := s.subject.Call()
	before := request.Header.Clone()
	status := s.status(s.subject.Scheme.Wrap(sent, credential), request)
	s.Equal(before, request.Header, "net/http: «RoundTrip should not modify the request»")
	// RFC 6750 section 2.3: a credential in the URI «SHOULD NOT be used», since a URL is
	// logged and cached where a header is not. What left Wrap is looked through at the end.
	for _, r := range sent.requests() {
		s.public = append(s.public, r.URL.String())
	}
	return status, sent.requests()
}

func (s *SchemeContract) status(transport http.RoundTripper, request *http.Request) int {
	response, err := (&http.Client{Transport: transport}).Do(request)
	s.Require().NoError(err)
	s.Require().NoError(drain(response))
	return response.StatusCode
}

// remember adds the secrets behind stored to the ones looked for at the end.
func (s *SchemeContract) remember(stored core.StoredCredentials) {
	for _, secret := range s.subject.Secrets(stored) {
		if secret != "" {
			s.secrets = append(s.secrets, secret)
		}
	}
}

// say keeps err's text, if any, to be looked through for secrets.
func (s *SchemeContract) say(err error) {
	if err != nil {
		s.public = append(s.public, err.Error())
	}
}

func (s *SchemeContract) classify(response *http.Response, body []byte, err error) core.Outcome {
	return s.subject.Scheme.Classify(response, body, err)
}

// answer is what a provider's server answers at path, read as a tool source hands it to
// Classify: the response, its body, and the error of reading them.
func (s *SchemeContract) answer(path string) (*http.Response, []byte, error) {
	server := httptest.NewServer(http.HandlerFunc(answers))
	s.T().Cleanup(server.Close)
	// A POST on a connection of its own, so net/http retries nothing (it retries only
	// idempotent requests on a reused connection).
	client := &http.Client{Transport: &http.Transport{DisableKeepAlives: true}}
	response, err := client.Post(server.URL+path, "application/json", strings.NewReader(`{}`))
	if err != nil {
		return nil, nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	return response, body, err
}

// answers are the provider answers the Classify tests read, by path.
func answers(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/ok":
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	case "/invalid-token":
		w.Header().Set("WWW-Authenticate", `Bearer error="invalid_token"`)
		w.WriteHeader(http.StatusUnauthorized)
	case "/bare-challenge":
		// MCP 2026-07-28's 401 shape, the resource metadata alone: RFC 9728 section 5.1's
		// example challenge.
		w.Header().Set("WWW-Authenticate", `Bearer resource_metadata="https://resource.example.com/.well-known/oauth-protected-resource"`)
		w.WriteHeader(http.StatusUnauthorized)
	case "/insufficient-scope":
		w.Header().Set("WWW-Authenticate", `Bearer error="insufficient_scope", scope="files:read files:write"`)
		w.WriteHeader(http.StatusForbidden)
	case "/rate-limited":
		w.Header().Set("Retry-After", "30")
		w.WriteHeader(http.StatusTooManyRequests)
	case "/unavailable":
		w.WriteHeader(http.StatusServiceUnavailable)
	case "/server-error":
		w.WriteHeader(http.StatusInternalServerError)
	case "/lost":
		_, _ = io.Copy(io.Discard, r.Body)
		conn, _, err := http.NewResponseController(w).Hijack()
		if err == nil {
			_ = conn.Close()
		}
	default:
		http.NotFound(w, r)
	}
}

// same reports whether two StoredCredentials are the same bytes, which is what decides
// whether the resolver has something to persist.
func same(a, b core.StoredCredentials) bool {
	return a.Scheme == b.Scheme && a.Version == b.Version && bytes.Equal(a.Payload, b.Payload)
}

// forms are the texts secret is written as: itself; percent-encoded as net/url writes it in a
// query value (QueryEscape, which url.Values.Encode uses), a path segment (PathEscape), and a
// URL's path, fragment and userinfo (what URL.String writes for each); and escaped as slog
// writes a string value, with strconv.Quote in the text handler and JSON string escapes
// without HTML escaping in the JSON one (log/slog handler.go:572-577 and json_handler.go:181-185,
// go1.27). Texts are not decoded instead, since a log line can hold a URL and cannot be
// decoded as a whole.
func forms(secret string) []string {
	u := url.URL{Path: secret, Fragment: secret, User: url.User(secret)}
	quoted := strconv.Quote(secret)
	var escaped bytes.Buffer
	encoder := json.NewEncoder(&escaped)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(secret)
	jsoned := strings.TrimSuffix(escaped.String(), "\n")
	all := []string{
		secret,
		url.QueryEscape(secret),
		url.PathEscape(secret),
		u.EscapedPath(),
		u.EscapedFragment(),
		u.User.String(),
		quoted[1 : len(quoted)-1],
		jsoned[1 : len(jsoned)-1],
	}
	slices.Sort(all)
	return slices.Compact(all)
}

func ok(status int) bool {
	return status >= 200 && status < 300
}

func drain(response *http.Response) error {
	_, _ = io.Copy(io.Discard, response.Body)
	return response.Body.Close()
}

func random() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// recorder is the transport under Wrap: it keeps a copy of each request Wrap produced, which
// is what egress would see at dial, and sends it on.
type recorder struct {
	next http.RoundTripper
	mu   sync.Mutex
	sent []*http.Request
}

func (r *recorder) RoundTrip(request *http.Request) (*http.Response, error) {
	r.mu.Lock()
	r.sent = append(r.sent, request.Clone(request.Context()))
	r.mu.Unlock()
	return r.next.RoundTrip(request)
}

func (r *recorder) requests() []*http.Request {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]*http.Request(nil), r.sent...)
}

// lockedStore is a core.CredentialStore in memory: one lock, the state behind it. It is the
// least a resolver needs to commit a renewal once.
type lockedStore struct {
	mu    sync.Mutex
	state core.CredentialState
}

func (l *lockedStore) Update(_ context.Context, _ core.ConnectionRef, fn func(state *core.CredentialState, checkpoint func() error) (bool, error)) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	state := l.state
	changed, err := fn(&state, func() error {
		l.state = state
		return nil
	})
	if err != nil {
		return err
	}
	if changed {
		l.state = state
	}
	return nil
}

var _ core.CredentialStore = (*lockedStore)(nil)

// lockedBuffer is a bytes.Buffer two slog handlers can write to at once.
type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
