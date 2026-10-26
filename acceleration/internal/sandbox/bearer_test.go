package sandbox

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

type BearerTokenSuite struct {
	suite.Suite
}

func TestBearerTokenSuite(t *testing.T) {
	suite.Run(t, new(BearerTokenSuite))
}

// hostBearer is a scheme that sends the host it is asked for as its bearer token, so a test
// sees which host BearerToken asked its scheme to wrap a request for.
type hostBearer struct{ core.Scheme }

func (hostBearer) Wrap(base http.RoundTripper, _ core.AccessCredential) http.RoundTripper {
	return wrapped(func(r *http.Request) (*http.Response, error) {
		r = r.Clone(r.Context())
		r.Header.Set("Authorization", "Bearer "+r.URL.Host)
		return base.RoundTrip(r)
	})
}

type wrapped func(*http.Request) (*http.Response, error)

func (f wrapped) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// TestAWildcardHostIsAskedAsItsDomain: a grant's host *.crm.example is a request to
// crm.example, a name a scheme can build a URL for (an instance's host, a signed host).
func (s *BearerTokenSuite) TestAWildcardHostIsAskedAsItsDomain() {
	for host, asked := range map[string]string{"api.crm.example": "api.crm.example", "*.crm.example": "crm.example",
		"crm.example": "crm.example", "*.*.crm.example": "*.crm.example"} {
		token, err := BearerToken(s.T().Context(), hostBearer{}, core.AccessCredential{}, host)

		s.Require().NoError(err, host)
		s.Equal(asked, token, host)
	}
}
