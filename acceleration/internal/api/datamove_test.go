package api

import (
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// dataMovePaths are the three ways a customer's data goes in or out, which are refused
// together or not at all: an export somebody may not have is no safer if the change feed
// hands the same rows over one at a time.
var dataMovePaths = []struct {
	method string
	path   string
}{
	{http.MethodGet, "/v1/data/export"},
	{http.MethodPost, "/v1/data/import"},
	{http.MethodGet, "/v1/data/changes"},
}

// movingData builds a handler in api_key mode over one key, which is the only mode the
// data-move endpoints answer in.
func (s *ServerSuite) movingData(key, secret string) http.Handler {
	server, err := NewServer(Options{
		Routers:  map[routing.Modality]routing.Inspector{routing.STT: s.speech()},
		AuthMode: auth.APIKey,
		Auth:     s.keyAuth(key, auth.App{AppID: "acme", Secret: secret}),
	})
	s.Require().NoError(err)
	return server.Handler()
}

func (s *ServerSuite) TestADeviceMayNotMoveACustomersData() {
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.movingData(key, secret)

	for _, endpoint := range dataMovePaths {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(endpoint.method, endpoint.path, strings.NewReader(""))
		handler.ServeHTTP(recorder, s.asUser(request, key, secret))

		s.Equal(http.StatusForbidden, recorder.Code, endpoint.path)
		s.Contains(recorder.Body.String(), "server-side only", endpoint.path)
	}
}

func (s *ServerSuite) TestMovingDataIsRefusedWhileNobodyIsAuthenticated() {
	// In noauth the customer is a header, so an export would hand any caller any
	// customer's rows for the price of naming them.
	server, err := NewServer(Options{
		Routers:  map[routing.Modality]routing.Inspector{routing.STT: s.speech()},
		AuthMode: auth.NoAuth,
	})
	s.Require().NoError(err)
	handler := server.Handler()

	for _, endpoint := range dataMovePaths {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(endpoint.method, endpoint.path, strings.NewReader(""))
		request.Header.Set(CustomerHeader, "acme")
		handler.ServeHTTP(recorder, request)

		s.Equal(http.StatusForbidden, recorder.Code, endpoint.path)
		s.Contains(recorder.Body.String(), "without authentication", endpoint.path)
	}
}

func (s *ServerSuite) TestMovingDataNeedsACaller() {
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.movingData(key, secret)

	for _, endpoint := range dataMovePaths {
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, httptest.NewRequest(endpoint.method, endpoint.path, strings.NewReader("")))

		s.Equal(http.StatusUnauthorized, recorder.Code, endpoint.path)
	}
}

func (s *ServerSuite) TestABackendIsToldThereIsNoDatabaseToMove() {
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.movingData(key, secret)

	for _, endpoint := range dataMovePaths {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(endpoint.method, endpoint.path, strings.NewReader(""))
		handler.ServeHTTP(recorder, s.asBackend(request, key, secret))

		s.Equal(http.StatusServiceUnavailable, recorder.Code, endpoint.path)
		s.Contains(recorder.Body.String(), "no database", endpoint.path)
	}
}
