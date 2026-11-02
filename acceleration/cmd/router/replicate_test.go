package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/golang-jwt/jwt/v5"
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

// serve starts a source that answers a request the way check says and returns its URL.
func (s *ReplicateSourceSuite) serve(check func(r *http.Request) (int, string)) string {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		status, body := check(r)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	s.T().Cleanup(server.Close)
	return server.URL
}

// source is the deployment runReplicate builds from these flags, after the source's URL.
func (s *ReplicateSourceSuite) source(sourceURL string, flags ...string) *deployment {
	options, err := parseReplicate(append([]string{
		"--from", sourceURL + "/", "--api-key", sourceKey, "--api-secret", sourceSecret,
	}, flags...))
	s.Require().NoError(err)
	return options.source
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

// gateway is the Stream gateway in front of a hosted router, which lets a server-side
// request through only as GetStream/chat checks it, and answers with the query it passed
// on. The checks, in order:
//   - the key comes from the api_key header or query parameter only
//     (monolith/auth/auth.go:739-760, GetApiKeyFromRequest);
//   - Stream-Auth-Type must be jwt; anonymous and anything else are refused
//     (projects/stream-accelerate/proxy.go:540-585, branch thierry/accelerate-gke);
//   - the token in Authorization, "Bearer " trimmed (auth.go:784), must be HS256 and
//     verify with the app secret (auth.go:861, GetAuthPayloadFromRequest); one naming no user_id is
//     server-side (proxy.go:563-566), one naming a user is not.
func (s *ReplicateSourceSuite) gateway() string {
	return s.serve(func(r *http.Request) (int, string) {
		key := r.Header.Get("api_key")
		if key == "" {
			key = r.URL.Query().Get("api_key")
		}
		switch key {
		case "":
			return http.StatusUnauthorized, `{"code":2,"message":"api_key is required"}`
		case sourceKey:
		default:
			return http.StatusUnauthorized, `{"code":2,"message":"api_key not found"}`
		}

		authType := r.Header.Get("stream-auth-type")
		if authType == "" {
			authType = r.URL.Query().Get("stream-auth-type")
		}
		switch authType {
		case "jwt":
		case "anonymous":
			return http.StatusUnauthorized, `{"code":2,"message":"anonymous access is not available on the acceleration api"}`
		default:
			return http.StatusUnauthorized, `{"code":2,"message":"stream-auth-type missing or invalid"}`
		}

		raw := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		claims := jwt.MapClaims{}
		_, err := jwt.ParseWithClaims(raw, claims, func(*jwt.Token) (any, error) {
			return []byte(sourceSecret), nil
		}, jwt.WithValidMethods([]string{jwt.SigningMethodHS256.Name}))
		if err != nil {
			return http.StatusUnauthorized, `{"code":5,"message":"token is invalid"}`
		}
		if _, user := claims["user_id"]; user {
			return http.StatusForbidden, `{"code":17,"message":"not a server-side token"}`
		}
		return http.StatusOK, r.URL.RawQuery
	})
}

// direct is a router reached with no gateway in front, checking the key and the server
// token itself.
func (s *ReplicateSourceSuite) direct() string {
	authenticator, err := auth.New(auth.APIKey, func(_ context.Context, key string) (auth.App, error) {
		if key != sourceKey {
			return auth.App{}, auth.ErrUnauthenticated
		}
		return auth.App{AppID: "the-source-app", Secret: sourceSecret}, nil
	})
	s.Require().NoError(err)
	return s.serve(func(r *http.Request) (int, string) {
		principal, err := authenticator.Authenticate(r.Context(), r)
		if err != nil || !principal.ServerSide {
			return http.StatusUnauthorized, `{"error":{"message":"unauthenticated"}}`
		}
		return http.StatusOK, exportBody
	})
}

func (s *ReplicateSourceSuite) TestASourceBehindTheStreamGatewayIsExportedByDefault() {
	query, err := s.read(s.source(s.gateway()), "/v1/data/export", nil)

	s.Require().NoError(err)
	s.Require().Equal(url.Values{"api_key": {sourceKey}}.Encode(), query)
}

func (s *ReplicateSourceSuite) TestASourceBehindTheStreamGatewayIsAskedForChangesSinceTheCursor() {
	query, err := s.read(s.source(s.gateway()), "/v1/data/changes", map[string]string{"after": "42"})

	s.Require().NoError(err)
	s.Require().Equal(url.Values{"api_key": {sourceKey}, "after": {"42"}}.Encode(), query)
}

func (s *ReplicateSourceSuite) TestASourceBehindTheStreamGatewayRefusingServerNamesTheFlagForIt() {
	_, err := s.read(s.source(s.gateway(), "--source-auth", "server"), "/v1/data/export", nil)

	s.Require().ErrorContains(err, "stream-auth-type missing or invalid")
	s.Require().ErrorContains(err, "--source-auth jwt")
}

func (s *ReplicateSourceSuite) TestASourceReachedDirectlyIsCopiedWithSourceAuthServer() {
	body, err := s.read(s.source(s.direct(), "--source-auth", "server"), "/v1/data/export", nil)

	s.Require().NoError(err)
	s.Require().Equal(exportBody, body)
}

func (s *ReplicateSourceSuite) TestASourceReachedDirectlyRefusingJWTNamesTheFlagForIt() {
	_, err := s.read(s.source(s.direct()), "/v1/data/export", nil)

	s.Require().ErrorContains(err, "401 Unauthorized")
	s.Require().ErrorContains(err, "--source-auth server")
}

func (s *ReplicateSourceSuite) TestAnUnknownSourceAuthIsRefused() {
	_, err := parseReplicate([]string{
		"--from", "http://source", "--api-key", sourceKey, "--api-secret", sourceSecret, "--source-auth", "anonymous",
	})

	s.Require().ErrorContains(err, "--source-auth is jwt or server")
}

func (s *ReplicateSourceSuite) TestTheCutOverNeedsNoSource() {
	options, err := parseReplicate([]string{"--resume", "--as", "acme"})

	s.Require().NoError(err)
	s.Require().True(options.resume)
	s.Require().Equal("acme", options.as)
	s.Require().Nil(options.source)
}

func (s *ReplicateSourceSuite) TestTheCutOverNeedsTheAppTheRowsLandedUnder() {
	_, err := parseReplicate([]string{"--resume"})

	s.Require().ErrorContains(err, "--resume --as <app id>")
}

func (s *ReplicateSourceSuite) TestTheCutOverRefusesToCatchUpFromTheSourceFirst() {
	_, err := parseReplicate([]string{"--resume", "--as", "acme", "--from", "http://source", "--api-key", sourceKey, "--api-secret", sourceSecret})

	s.Require().ErrorContains(err, "--resume --as <app id>")
}
