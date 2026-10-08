package bearer_test

import (
	"context"
	"crypto/subtle"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core/contracttest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
)

// token is synthetic: no provider issued it. Its characters are all b64token's.
const token = "contract-bearer.token_0123~456+789/abc=="

func TestBearerSchemeContract(t *testing.T) {
	suite.Run(t, &contracttest.SchemeContract{New: func(t *testing.T, _ *slog.Logger) contracttest.Subject {
		provider := provider(t)
		return contracttest.Subject{
			Scheme:    bearer.New(),
			Supplied:  map[string]string{bearer.SuppliedToken: token},
			Transport: provider.Client().Transport,
			Call: func() *http.Request {
				request, _ := http.NewRequest(http.MethodGet, provider.URL+"/mcp", nil)
				return request
			},
			Secrets:   func(core.StoredCredentials) []string { return []string{token} },
			RevokeErr: bearer.ErrNotRevocable,
		}
	}})
}

// BearerSuite covers what is the bearer scheme's own: which tokens it takes.
type BearerSuite struct {
	suite.Suite
	scheme *bearer.Scheme
}

func TestBearerSuite(t *testing.T) {
	suite.Run(t, new(BearerSuite))
}

func (s *BearerSuite) SetupTest() {
	s.scheme = bearer.New()
}

// RFC 6750 section 2.1: b64token = 1*( ALPHA / DIGIT / "-" / "." / "_" / "~" / "+" / "/" )
// *"=".
func (s *BearerSuite) TestATokenThatIsNotAB64TokenIsRefused() {
	for _, value := range []string{"", "=abc", "a b", "a,b", `a"b`, token + "\r\n", "abc=def"} {
		_, _, err := s.scheme.Complete(context.Background(), core.CompleteInput{Supplied: map[string]string{bearer.SuppliedToken: value}})
		s.ErrorContains(err, "bearer: the token", "%q", value)
	}
}

func (s *BearerSuite) TestASuppliedValueItDoesNotTakeIsRefused() {
	_, _, err := s.scheme.Complete(context.Background(), core.CompleteInput{Supplied: map[string]string{
		bearer.SuppliedToken: token, "api_key": token,
	}})
	s.ErrorContains(err, `"api_key" is not a value bearer takes`)
	s.NotContains(err.Error(), token)
}

// AI-863: the Linq line a token sends from is an input, and the manifest's identity, so the
// bridge finds the connection of a line by it.
func (s *BearerSuite) TestAnIdentityOfInputsIsTheAccount() {
	_, account, err := s.scheme.Complete(context.Background(), core.CompleteInput{
		Manifest: s.resolved(`
inputs:
  - name: line
    pattern: "[+][0-9]+"
identity: [line]`, map[string]string{"line": "+12025551234"}),
		Supplied: map[string]string{bearer.SuppliedToken: token},
	})

	s.Require().NoError(err)
	s.Equal(core.AccountInfo{AccountID: "+12025551234"}, account)
}

func (s *BearerSuite) TestAManifestWithoutAnIdentityHasNoAccount() {
	_, account, err := s.scheme.Complete(context.Background(), core.CompleteInput{
		Manifest: s.resolved(`
inputs:
  - name: line
    pattern: "[+][0-9]+"`, map[string]string{"line": "+12025551234"}),
		Supplied: map[string]string{bearer.SuppliedToken: token},
	})

	s.Require().NoError(err)
	s.Zero(account)
}

// A captured value is something only a consent returns, so a token supplied for a connector
// that captures one learns no account, as before AI-863.
func (s *BearerSuite) TestAnIdentityACaptureMakesIsNoAccountForASuppliedToken() {
	_, account, err := s.scheme.Complete(context.Background(), core.CompleteInput{
		Manifest: s.resolved(`
inputs:
  - name: line
    pattern: "[+][0-9]+"
capture:
  - name: team
    from: token_response
    path: $.team.id
identity: [line, team]`, map[string]string{"line": "+12025551234"}),
		Supplied: map[string]string{bearer.SuppliedToken: token},
	})

	s.Require().NoError(err)
	s.Zero(account)
}

// resolved is a bearer manifest with the inputs, identity and capture rules in rules, resolved
// with inputs.
func (s *BearerSuite) resolved(rules string, inputs map[string]string) core.ResolvedManifest {
	manifest, err := core.ParseManifest([]byte(`
id: acme_line
revision: 1
name: Acme
endpoints:
  mcp: https://mcp.acme.example/mcp
schemes: [bearer]
sources:
  - kind: mcp
    endpoint: mcp` + rules))
	s.Require().NoError(err)
	resolved, err := manifest.Resolve(bearer.Name, inputs, nil)
	s.Require().NoError(err)
	return resolved
}

// A 401 without an RFC 6750 challenge still means the token no longer works.
func (s *BearerSuite) TestABare401IsInvalidGrant() {
	recorder := httptest.NewRecorder()
	recorder.WriteHeader(http.StatusUnauthorized)
	s.Equal(core.Outcome{Kind: core.OutcomeInvalidGrant}, s.scheme.Classify(recorder.Result(), nil, nil))
}

// provider is a TLS server that answers 200 to a request carrying Authorization: Bearer
// token, exactly, and 401 to anything else.
func provider(t *testing.T) *httptest.Server {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if values := r.Header.Values("Authorization"); len(values) != 1 || subtle.ConstantTimeCompare([]byte(values[0]), []byte("Bearer "+token)) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer error="invalid_token"`)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(server.Close)
	return server
}
