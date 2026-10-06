package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/getsentry/sentry-go"
	"github.com/stretchr/testify/suite"
)

// SentrySuite covers what a request hands Sentry.
type SentrySuite struct {
	suite.Suite
}

func TestSentrySuite(t *testing.T) {
	suite.Run(t, new(SentrySuite))
}

func (s *SentrySuite) TestSentryCollectsNoRequestBodies() {
	// Sentry copies the body of a request it watches. A body carrying an app's Stream
	// secrets is never handed to it.
	watched := map[string]bool{}
	handler := withSentry(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		watched[r.URL.Path] = sentry.GetHubFromContext(r.Context()) != nil
		w.WriteHeader(http.StatusNoContent)
	}))

	for _, path := range []string{"/v1/settings/app", streamCredentialsPath + "credentials", streamCredentialsPath + "check"} {
		request := httptest.NewRequest(http.MethodPut, path, strings.NewReader(`{"api_secret":"a-long-stream-secret"}`))
		handler.ServeHTTP(httptest.NewRecorder(), request)
	}

	s.True(watched["/v1/settings/app"], "everything else is reported as before")
	s.False(watched[streamCredentialsPath+"credentials"])
	s.False(watched[streamCredentialsPath+"check"])
}
