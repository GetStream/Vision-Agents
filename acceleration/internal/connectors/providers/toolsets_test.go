package providers_test

import (
	"context"
	"encoding/json"
	"io/fs"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// toolsetsHeader is the header GitHub's MCP server reads its toolsets from (providers/github.yaml,
// [toolsets]).
const toolsetsHeader = "X-MCP-Toolsets"

// ToolsetSuite opens each built-in's mcp source as a session does (mcp.Source.Open, then a
// tools/call), with its mcp endpoint pointed at the fake provider, and reads the headers every
// request reached the fake with.
type ToolsetSuite struct {
	suite.Suite
	ctx context.Context
}

func TestToolsetSuite(t *testing.T) {
	suite.Run(t, new(ToolsetSuite))
}

func (s *ToolsetSuite) SetupTest() {
	s.ctx = context.Background()
}

func (s *ToolsetSuite) TestGitHubAsksForTheDefaultAndActionsToolsetsOnEveryRequest() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	binding := s.binding(srv, "github")
	source := mcp.New()
	specs, err := source.Discover(s.ctx, binding)
	s.Require().NoError(err)
	echo := slices.IndexFunc(specs, func(spec core.ToolSpec) bool { return spec.Name == "echo" })
	s.Require().GreaterOrEqual(echo, 0)

	toolset, err := source.Open(s.ctx, binding, []core.ToolGrant{{Name: "echo", SchemaDigest: specs[echo].SchemaDigest}})
	s.Require().NoError(err)
	defer toolset.Close()
	_, err = toolset.Call(s.ctx, llm.ToolCall{ID: "call-1", Name: "github" + mcp.Separator + "echo", Arguments: `{"text":"hi"}`})
	s.Require().NoError(err)

	sent := srv.MCPHeader(toolsetsHeader)
	s.Require().NotEmpty(sent)
	for _, value := range sent {
		s.Equal("default,actions", value)
	}
}

// The control: on the base, 04a6d74e, no built-in sent the header, github included
// (author-1.md, «Control probe»).
func (s *ToolsetSuite) TestNoOtherBuiltInAsksForToolsets() {
	files, err := fs.Glob(providers.FS, "*.yaml")
	s.Require().NoError(err)
	opened := 0
	for _, file := range files {
		id := strings.TrimSuffix(file, ".yaml")
		if id == "github" || !s.hasMCPSource(id) {
			continue
		}
		srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
		toolset, err := mcp.New().Open(s.ctx, s.binding(srv, id), nil)
		s.Require().NoError(err, id)
		toolset.Close()

		sent := srv.MCPHeader(toolsetsHeader)
		s.Require().NotEmpty(sent, id)
		for _, value := range sent {
			s.Empty(value, id)
		}
		opened++
	}
	s.Positive(opened)
}

func (s *ToolsetSuite) manifest(id string) core.Manifest {
	raw, err := fs.ReadFile(providers.FS, id+".yaml")
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	return manifest
}

func (s *ToolsetSuite) hasMCPSource(id string) bool {
	return slices.ContainsFunc(s.manifest(id).Sources, func(rule core.SourceRule) bool { return rule.Kind == mcp.Kind })
}

// binding is the built-in id resolved with its first scheme, its mcp endpoint the fake's, and
// a client that sends a token from the fake's client credentials grant.
func (s *ToolsetSuite) binding(srv *fakeprovider.Server, id string) core.ResolvedBinding {
	manifest := s.manifest(id)
	var inputs map[string]string
	if id == "shopify" {
		inputs = map[string]string{"shop": "mystore.myshopify.com"}
	}
	resolved, err := manifest.Resolve(manifest.Schemes[0], inputs, nil)
	s.Require().NoError(err, id)
	rule := resolved.Sources[slices.IndexFunc(resolved.Sources, func(rule core.SourceRule) bool { return rule.Kind == mcp.Kind })]
	resolved.Endpoints[rule.Endpoint] = srv.URL + fakeprovider.PathMCP
	client := srv.Client()
	return core.ResolvedBinding{
		Binding:  core.Binding{Name: id},
		Manifest: resolved,
		HTTP:     &http.Client{Transport: authorized{base: client.Transport, token: s.token(srv)}, Timeout: client.Timeout},
	}
}

func (s *ToolsetSuite) token(srv *fakeprovider.Server) string {
	form := url.Values{"grant_type": {"client_credentials"}}
	request, err := http.NewRequestWithContext(s.ctx, http.MethodPost, srv.URL+fakeprovider.PathToken, strings.NewReader(form.Encode()))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.SetBasicAuth(srv.ClientID, srv.ClientSecret)
	response, err := srv.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	var token struct {
		AccessToken string `json:"access_token"`
	}
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&token))
	s.Require().NotEmpty(token.AccessToken)
	return token.AccessToken
}

// authorized sends token as a bearer credential (RFC 6750 section 2.1) on a copy of each
// request, as the connection's transport does.
type authorized struct {
	base  http.RoundTripper
	token string
}

func (a authorized) RoundTrip(request *http.Request) (*http.Response, error) {
	request = request.Clone(request.Context())
	request.Header.Set("Authorization", "Bearer "+a.token)
	return a.base.RoundTrip(request)
}
