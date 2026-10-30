package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
)

type ReplicateSourceSuite struct {
	suite.Suite
}

func TestReplicateSourceSuite(t *testing.T) {
	suite.Run(t, new(ReplicateSourceSuite))
}

const (
	sourceKey    = "the-source-key"
	sourceSecret = "the-source-secret"
	exportBody   = `{"table":"agents","row":{}}`
)

// serve starts a source that answers the export to a request its check lets through.
func (s *ReplicateSourceSuite) serve(check func(r *http.Request) (int, string)) *deployment {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		status, body := check(r)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	s.T().Cleanup(server.Close)
	return &deployment{url: server.URL, key: sourceKey, secret: sourceSecret, client: server.Client()}
}

// read asks the source for path the way runReplicate does and returns what it answered.
func (s *ReplicateSourceSuite) read(source *deployment, path string, query map[string]string) (string, error) {
	body, err := source.get(context.Background(), path, query)
	if err != nil {
		return "", err
	}
	defer body.Close()
	read, err := io.ReadAll(body)
	return string(read), err
}

// gateway is a source behind the Stream gateway in front of a hosted router, which reads
// the key from the api_key header or query parameter only, as chat's
// auth.GetApiKeyFromRequest does, and turns away the rest. It answers with the query it
// passed on.
func (s *ReplicateSourceSuite) gateway() *deployment {
	return s.serve(func(r *http.Request) (int, string) {
		key := r.Header.Get("api_key")
		if key == "" {
			key = r.URL.Query().Get("api_key")
		}
		switch key {
		case "":
			return http.StatusUnauthorized, `{"code":2,"message":"api_key is required"}`
		case sourceKey:
			return http.StatusOK, r.URL.Query().Get("after")
		default:
			return http.StatusUnauthorized, `{"code":2,"message":"api_key not found"}`
		}
	})
}

func (s *ReplicateSourceSuite) TestASourceBehindTheStreamGatewayIsExported() {
	_, err := s.read(s.gateway(), "/v1/data/export", nil)

	s.Require().NoError(err)
}

func (s *ReplicateSourceSuite) TestASourceBehindTheStreamGatewayIsAskedForChangesSinceTheCursor() {
	after, err := s.read(s.gateway(), "/v1/data/changes", map[string]string{"after": "42"})

	s.Require().NoError(err)
	s.Require().Equal("42", after)
}

// A router reached directly checks the key and the server token itself.
func (s *ReplicateSourceSuite) TestASourceReachedDirectlyIsCopied() {
	authenticator, err := auth.New(auth.APIKey, func(_ context.Context, key string) (auth.App, error) {
		if key != sourceKey {
			return auth.App{}, auth.ErrUnauthenticated
		}
		return auth.App{AppID: "the-source-app", Secret: sourceSecret}, nil
	})
	s.Require().NoError(err)
	source := s.serve(func(r *http.Request) (int, string) {
		principal, err := authenticator.Authenticate(r.Context(), r)
		if err != nil || !principal.ServerSide {
			return http.StatusUnauthorized, `{"error":{"message":"unauthenticated"}}`
		}
		return http.StatusOK, exportBody
	})

	body, err := s.read(source, "/v1/data/export", nil)

	s.Require().NoError(err)
	s.Require().Equal(exportBody, body)
}
