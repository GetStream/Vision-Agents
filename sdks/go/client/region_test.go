package client

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

type regionSuite struct {
	suite.Suite
	httpClient *http.Client
}

func TestRegionalClients(t *testing.T) {
	suite.Run(t, new(regionSuite))
}

func (s *regionSuite) SetupTest() {
	s.T().Setenv(stream.URLEnv, "")
	s.T().Setenv(stream.RegionEnv, "")
	s.T().Setenv(stream.CustomerEnv, "")
	s.T().Setenv(stream.APIKeyEnv, "test-key")
	s.T().Setenv(stream.APISecretEnv, "test-secret")

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/settings/app", func(w http.ResponseWriter, r *http.Request) {
		if r.Host != "accelerate-us-east4.gcp.stream-io-api.com" ||
			r.Header.Get(stream.APIKeyHeader) != "test-key" ||
			r.Header.Get(stream.AuthTypeHeader) != stream.AuthTypeJWT ||
			r.Header.Get(stream.AuthorizationHeader) == "" {
			http.Error(w, "expected an authenticated regional request", http.StatusUnauthorized)
			return
		}
		answer(w, http.StatusOK, acceleration.AppSettings{Stream: acceleration.StreamSettings{
			Tenancy: "app", WritesInto: "customer_app", ChannelType: "present", CallType: "present",
		}})
	})
	server := httptest.NewTLSServer(mux)
	s.T().Cleanup(server.Close)
	s.httpClient = server.Client()
	transport := s.httpClient.Transport.(*http.Transport)
	transport.TLSClientConfig.ServerName = "example.com"
	// Keep the regional URL and TLS request, but connect only to the test server.
	transport.DialContext = func(ctx context.Context, network, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
	}
}

func (s *regionSuite) readSettings(backend stream.Backend) {
	backend.HTTPClient = s.httpClient
	api, err := New(backend)
	s.Require().NoError(err)
	settings, err := api.Settings().App(s.T().Context())
	s.Require().NoError(err)
	s.Equal(acceleration.StreamWritesInto("customer_app"), settings.Stream.WritesInto)
}

func (s *regionSuite) TestTheDefaultClientReadsAppSettings() {
	s.readSettings(stream.Backend{})
}

func (s *regionSuite) TestAnExplicitRegionReadsAppSettings() {
	s.readSettings(stream.Backend{Region: "us-east4"})
}

func (s *regionSuite) TestARegionFromTheEnvironmentReadsAppSettings() {
	s.T().Setenv(stream.RegionEnv, "us-east4")
	s.readSettings(stream.Backend{})
}

func (s *regionSuite) TestARegionsOwnURLWithATrailingSlashReadsAppSettings() {
	s.readSettings(stream.Backend{Region: "us-east4", URL: stream.DefaultURL + "/"})
}

func (s *regionSuite) TestAConflictingURLAndRegionAreRefused() {
	_, err := New(stream.Backend{Region: "europe-west4", URL: stream.DefaultURL})
	s.Require().EqualError(err, "stream: set a URL or a region, not both")
}

func (s *regionSuite) TestAnInvalidRegionWithAURLIsRefused() {
	_, err := New(stream.Backend{Region: "us-east4.evil.com/x", URL: stream.DefaultURL})
	s.Require().EqualError(err, `stream: "us-east4.evil.com/x" is not a region, such as us-east4`)
}

func (s *regionSuite) TestAResolvedRegionalBackendCanBePointedElsewhere() {
	s.T().Setenv(stream.RegionEnv, "us-east4")
	api, err := New(stream.Backend{})
	s.Require().NoError(err)
	moved := api.Backend()
	moved.URL = "http://localhost:8080"
	resolved, err := moved.Resolve()
	s.Require().NoError(err)
	s.Equal("http://localhost:8080", resolved.URL)
}

func (s *regionSuite) TestARegionalClientCanActForAUser() {
	s.actForAUser(stream.Backend{Region: "us-east4"})
}

func (s *regionSuite) TestARegionFromTheEnvironmentCanActForAUser() {
	s.T().Setenv(stream.RegionEnv, "us-east4")
	s.actForAUser(stream.Backend{})
}

// actForAUser checks the user client both resolves and reaches the regional router.
func (s *regionSuite) actForAUser(backend stream.Backend) {
	backend.HTTPClient = s.httpClient
	api, err := New(backend)
	s.Require().NoError(err)
	user, err := api.SetUser("jean", "user-token")
	s.Require().NoError(err)
	s.Equal(stream.DefaultURL, user.Backend().URL)
	s.Equal("jean", user.Backend().UserID)
	s.Equal("user-token", user.Backend().Token)
	_, err = user.Settings().App(s.T().Context())
	s.Require().NoError(err)
}
