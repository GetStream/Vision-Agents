package api

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/suite"
)

// OperationNamesSuite covers the names the access line gives operations (operationNames).
type OperationNamesSuite struct {
	suite.Suite
}

func TestOperationNamesSuite(t *testing.T) {
	suite.Run(t, new(OperationNamesSuite))
}

// Defect: an id lower-cased or cut by its first letter, or an empty one panicking at start-up.
func (s *OperationNamesSuite) TestAnIDIsNamedWithItsFirstLetterUpperCased() {
	for id, want := range map[string]string{
		"getSession":         "GetSession",
		"QueryChannels":      "QueryChannels",
		"x":                  "X",
		"X":                  "X",
		"":                   "",
		"proxyConnectionGet": "ProxyConnectionGet",
	} {
		s.Equal(want, pascalCase(id), "%q", id)
	}
}

// Defect: the proxy's operations keyed by their spec path, {path}, which chi never reports,
// so a provider call logs no operation, or one method's name for all.
func (s *OperationNamesSuite) TestTheConnectionProxyIsNamedByThePatternChiMatchesItBy() {
	names := operationNames([]operationSummary{
		{method: http.MethodGet, path: connectionProxyPath + "{path}", id: "proxyConnectionGet"},
		{method: http.MethodPost, path: connectionProxyPath + "{path}", id: "proxyConnectionPost"},
		{method: http.MethodGet, path: "/v1/agents/sessions/{id}", id: "getSession"},
	})

	s.Equal(map[string]string{
		"GET " + connectionProxyPath + "*":  "ProxyConnectionGet",
		"POST " + connectionProxyPath + "*": "ProxyConnectionPost",
		"GET /v1/agents/sessions/{id}":      "GetSession",
	}, names)
}
