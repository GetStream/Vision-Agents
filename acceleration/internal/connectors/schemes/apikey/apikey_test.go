package apikey_test

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
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/apikey"
)

// key is synthetic: no provider issued it.
const key = "contract-api-key-0123456789abcdef"

func TestAPIKeySchemeContract(t *testing.T) {
	suite.Run(t, &contracttest.SchemeContract{New: func(t *testing.T, _ *slog.Logger) contracttest.Subject {
		provider := provider(t, "X-Api-Key")
		return contracttest.Subject{
			Scheme:    apikey.New(),
			Supplied:  map[string]string{apikey.SuppliedKey: key, apikey.SuppliedHeader: "X-Api-Key"},
			Transport: provider.Client().Transport,
			Call: func() *http.Request {
				request, _ := http.NewRequest(http.MethodGet, provider.URL+"/v1/contacts?limit=1", nil)
				return request
			},
			Secrets:   func(core.StoredCredentials) []string { return []string{key} },
			RevokeErr: apikey.ErrNotRevocable,
		}
	}})
}

// APIKeySuite covers what is the api_key scheme's own: which headers it takes.
type APIKeySuite struct {
	suite.Suite
	scheme *apikey.Scheme
}

func TestAPIKeySuite(t *testing.T) {
	suite.Run(t, new(APIKeySuite))
}

func (s *APIKeySuite) SetupTest() {
	s.scheme = apikey.New()
}

func (s *APIKeySuite) TestTheKeyGoesInTheHeaderTheDeveloperNamedInAnyCase() {
	provider := provider(s.T(), "X-Shop-Access-Token")
	stored := s.complete(map[string]string{apikey.SuppliedKey: key, apikey.SuppliedHeader: "x-shop-access-token"})
	credential, _, err := s.scheme.Retrieve(context.Background(), stored, core.ResolvedManifest{})
	s.Require().NoError(err)

	client := &http.Client{Transport: s.scheme.Wrap(provider.Client().Transport, credential)}
	response, err := client.Get(provider.URL)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Equal(http.StatusOK, response.StatusCode)
}

func (s *APIKeySuite) TestAHeaderTheRouterOwnsOrAProxyReadsIsRefused() {
	for _, header := range []string{
		"Authorization", "authorization", "Cookie", "Host", "Content-Length", "Connection",
		"Transfer-Encoding", "Proxy-Authorization", "Proxy-Authenticate", "proxy-connection",
		"Keep-Alive", "te", "Trailer", "upgrade", "Expect",
	} {
		_, _, err := s.scheme.Complete(context.Background(), core.CompleteInput{Supplied: map[string]string{apikey.SuppliedKey: key, apikey.SuppliedHeader: header}})
		s.ErrorContains(err, "the header is one the router sets itself", header)
	}
}

func (s *APIKeySuite) TestAHeaderThatIsNotAFieldNameIsRefused() {
	for _, header := range []string{"", "X Api Key", "X-Api-Key:", "X-Api-Key\r\nX-Other"} {
		_, _, err := s.scheme.Complete(context.Background(), core.CompleteInput{Supplied: map[string]string{apikey.SuppliedKey: key, apikey.SuppliedHeader: header}})
		s.Error(err, "%q", header)
	}
}

func (s *APIKeySuite) TestAKeyThatWouldNotArriveAsSuppliedIsRefused() {
	for _, value := range []string{"", " " + key, key + " ", key + "\n", "a\x00b"} {
		_, _, err := s.scheme.Complete(context.Background(), core.CompleteInput{Supplied: map[string]string{apikey.SuppliedKey: value, apikey.SuppliedHeader: "X-Api-Key"}})
		s.ErrorContains(err, "apikey: the key")
	}
}

func (s *APIKeySuite) TestASuppliedValueItDoesNotTakeIsRefused() {
	_, _, err := s.scheme.Complete(context.Background(), core.CompleteInput{Supplied: map[string]string{
		apikey.SuppliedKey: key, apikey.SuppliedHeader: "X-Api-Key", "token": key,
	}})
	s.ErrorContains(err, `"token" is not a value api_key takes`)
	s.NotContains(err.Error(), key)
}

// A 401 without an RFC 6750 challenge, as an API key's provider answers it, still means the
// key no longer works.
func (s *APIKeySuite) TestABare401IsInvalidGrant() {
	recorder := httptest.NewRecorder()
	recorder.WriteHeader(http.StatusUnauthorized)
	s.Equal(core.Outcome{Kind: core.OutcomeInvalidGrant}, s.scheme.Classify(recorder.Result(), nil, nil))
}

func (s *APIKeySuite) complete(supplied map[string]string) core.StoredCredentials {
	stored, account, err := s.scheme.Complete(context.Background(), core.CompleteInput{Supplied: supplied})
	s.Require().NoError(err)
	s.Equal(core.AccountInfo{}, account, "a key says nothing about whose it is")
	return stored
}

// provider is a TLS server that answers 200 to a request carrying key in header, and 401 to
// anything else.
func provider(t *testing.T, header string) *httptest.Server {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if values := r.Header.Values(header); len(values) != 1 || subtle.ConstantTimeCompare([]byte(values[0]), []byte(key)) != 1 {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(server.Close)
	return server
}
