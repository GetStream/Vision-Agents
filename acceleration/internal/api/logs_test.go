package api

import (
	"net/http"
	"net/http/httptest"
)

func (s *ServerSuite) TestAgentLogsRequireBackendCredentials() {
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)
	for _, path := range []string{"/v1/agents/logs", "/v1/agents/logs/1", "/v1/agents/logs/stream?cursor=MA"} {
		s.Run(path, func() {
			user := httptest.NewRecorder()
			handler.ServeHTTP(user, s.asUser(httptest.NewRequest(http.MethodGet, path, nil), key, secret))
			s.Equal(http.StatusForbidden, user.Code)
			backend := httptest.NewRecorder()
			handler.ServeHTTP(backend, s.asBackend(httptest.NewRequest(http.MethodGet, path, nil), key, secret))
			// No store in this test: a backend reaches the handler and gets unavailable.
			s.Equal(http.StatusServiceUnavailable, backend.Code)
		})
	}
}
