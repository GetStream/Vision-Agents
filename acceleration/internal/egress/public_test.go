package egress

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type EgressSuite struct {
	suite.Suite
}

func TestEgressSuite(t *testing.T) {
	suite.Run(t, new(EgressSuite))
}

func (s *EgressSuite) TestAPublicHTTPSURLPasses() {
	for _, raw := range []string{
		"https://8.8.8.8/mcp",
		"https://8.8.8.8:8443/mcp",
		"https://[2606:4700:4700::1111]/mcp",
	} {
		s.NoErrorf(ValidatePublicHTTPSURL(s.T().Context(), raw), "%s should pass", raw)
	}
}

func (s *EgressSuite) TestAURLThatIsNotHTTPSIsRefused() {
	s.requireRefused("http://8.8.8.8/mcp", "javascript:alert(1)", "ftp://8.8.8.8/mcp", "//8.8.8.8/mcp")
}

func (s *EgressSuite) TestUserinfoQueryAndFragmentAreRefused() {
	s.requireRefused(
		"https://user:password@8.8.8.8/mcp",
		"https://8.8.8.8/mcp?token=secret",
		"https://8.8.8.8/mcp?",
		"https://8.8.8.8/mcp#fragment",
	)
}

func (s *EgressSuite) TestPrivateAddressesAreRefused() {
	s.requireRefused(
		"https://10.2.3.4/mcp",
		"https://172.16.0.1/mcp",
		"https://192.168.1.1/mcp",
		"https://100.64.0.1/mcp",
		"https://[fc00::1]/mcp",
		"https://[fd12::1]/mcp",
	)
}

func (s *EgressSuite) TestLoopbackIsRefusedInEveryForm() {
	s.requireRefused(
		"https://127.0.0.1/mcp",
		"https://[::1]/mcp",
		"https://[::ffff:127.0.0.1]/mcp",
		"https://[::127.0.0.1]/mcp",
		"https://[::ffff:0:127.0.0.1]/mcp",
		"https://[64:ff9b::127.0.0.1]/mcp",
		"https://0.0.0.0/mcp",
		"https://[::]/mcp",
	)
}

func (s *EgressSuite) TestLinkLocalAndMetadataAddressesAreRefused() {
	s.requireRefused(
		"https://169.254.169.254/latest/meta-data",
		"https://[::ffff:169.254.169.254]/latest/meta-data",
		"https://[fd00:ec2::254]/latest/meta-data",
		"https://100.100.100.200/latest/meta-data",
		"https://169.254.1.1/mcp",
		"https://[fe80::1]/mcp",
	)
}

func (s *EgressSuite) TestAZonedAddressIsRefused() {
	s.requireRefused(
		"https://[fe80::1%25en0]/mcp",
		"https://[64:ff9b::127.0.0.1%25en0]/mcp",
		"https://[2606:4700:4700::1111%25en0]/mcp",
	)
}

func (s *EgressSuite) TestAHostNameIsCheckedAfterResolution() {
	err := ValidatePublicHTTPSURL(s.T().Context(), "https://localhost/mcp")

	s.ErrorContains(err, "resolved to a non-public address")
}

func (s *EgressSuite) TestTheClientRefusesToConnectToALoopbackServer() {
	var hits atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	defer server.Close()
	client := NewPublicHTTPClient(5 * time.Second)

	// By name too: a host that passed validation and now resolves to loopback is
	// refused when the connection is opened.
	for _, target := range []string{server.URL, strings.Replace(server.URL, "127.0.0.1", "localhost", 1)} {
		_, err := client.Get(target)
		s.ErrorContainsf(err, "non-public address", "%s should be refused", target)
	}
	s.Zero(hits.Load())
}

func (s *EgressSuite) TestTheClientRefusesPlainHTTP() {
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	defer server.Close()

	_, err := NewPublicHTTPClient(5 * time.Second).Get(server.URL)

	s.ErrorContains(err, "only HTTPS")
	s.Zero(hits.Load())
}

func (s *EgressSuite) TestTheClientRefusesARedirectToAnotherHost() {
	var hits atomic.Int32
	target := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	defer target.Close()
	origin := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+"/stolen", http.StatusFound)
	}))
	defer origin.Close()

	_, err := s.redirectClient(origin).Get(origin.URL + "/mcp")

	s.ErrorIs(err, errCrossHostRedirect)
	s.Zero(hits.Load())
}

func (s *EgressSuite) TestTheClientFollowsARedirectOnTheSameHost() {
	origin := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/mcp" {
			http.Redirect(w, r, "/mcp/", http.StatusTemporaryRedirect)
			return
		}
		_, _ = io.WriteString(w, r.URL.Path)
	}))
	defer origin.Close()

	response, err := s.redirectClient(origin).Get(origin.URL + "/mcp")
	s.Require().NoError(err)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)

	s.Require().NoError(err)
	s.Equal("/mcp/", string(body))
}

func (s *EgressSuite) requireRefused(raws ...string) {
	for _, raw := range raws {
		s.Errorf(ValidatePublicHTTPSURL(s.T().Context(), raw), "%s should be refused", raw)
	}
}

// redirectClient is the public client with its redirect policy, over a transport that
// trusts the test server. The public transport would refuse the loopback test server
// before any redirect is seen.
func (s *EgressSuite) redirectClient(server *httptest.Server) *http.Client {
	client := NewPublicHTTPClient(5 * time.Second)
	client.Transport = server.Client().Transport
	return client
}
