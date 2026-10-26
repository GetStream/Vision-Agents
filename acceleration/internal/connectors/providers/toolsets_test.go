package providers_test

import (
	"context"
	"encoding/json"
	"io/fs"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
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

// TestMetabaseOffersOnlyTheToolsItsScopesReach (AI-1394): against a server listing the 21
// tools staging's Metabase listed on 2026-10-16, revision 2 offers the ones that need
// agent:content:read or agent:query:run, which it asks for, and none that needs a scope it
// leaves out (metabase.yaml, [tools]).
func (s *ToolsetSuite) TestMetabaseOffersOnlyTheToolsItsScopesReach() {
	srv := s.listing("metabase", "alert_write", "bookmark_content", "browse_collection", "browse_data",
		"collection_write", "dashboard_write", "document_write", "duplicate_content", "execute_query",
		"execute_sql", "get_content", "get_parameter_values", "glossary", "learn", "measure_write",
		"metric_write", "question_write", "run_saved_question", "search", "segment_write", "subscription_write")
	resolved, err := s.manifest("metabase").Resolve(oauth2code.Name, map[string]string{"host": "metabase.example.com"}, nil)
	s.Require().NoError(err)
	resolved.Endpoints["mcp"] = srv.URL

	specs, err := mcp.New().Discover(s.ctx, core.ResolvedBinding{Binding: core.Binding{Name: "metabase"}, Manifest: resolved, HTTP: srv.Client()})

	s.Require().NoError(err)
	var names []string
	for _, spec := range specs {
		names = append(names, spec.Name)
	}
	s.ElementsMatch([]string{"browse_collection", "browse_data", "execute_query", "get_content", "get_parameter_values",
		"glossary", "learn", "run_saved_question", "search"}, names)
}

// TestAGrantOfDriveMeetsWhatGoogleWriteToolsNeed: Google's tool pages take drive for each write
// tool (google_drive.yaml, google_docs.yaml), so a grant holding drive, such as one moved from a
// plugin that asked for it, needs nothing more; one holding only the read scopes does.
func (s *ToolsetSuite) TestAGrantOfDriveMeetsWhatGoogleWriteToolsNeed() {
	const drive = "https://www.googleapis.com/auth/drive"
	for _, c := range []struct {
		id, needed string
		read       []string
	}{
		{"google_drive", "https://www.googleapis.com/auth/drive.file", []string{"https://www.googleapis.com/auth/drive.readonly"}},
		{"google_docs", "https://www.googleapis.com/auth/documents",
			[]string{"https://www.googleapis.com/auth/documents.readonly", "https://www.googleapis.com/auth/drive.readonly"}},
	} {
		s.Run(c.id, func() {
			resolved, err := s.manifest(c.id).Resolve(oauth2code.Name, nil, nil)
			s.Require().NoError(err)

			s.True(resolved.Scopes.Meets([]string{drive}, c.needed))
			s.False(resolved.Scopes.Meets(c.read, c.needed))
			s.NotContains(resolved.Scopes.List, drive, "the consent still asks for the narrower scope")
		})
	}
}

// TestGoogleDriveAndDocsOfferTheirWriteTools (AI-1933): against a server listing the tools
// Google's tool lists name (google_drive.yaml, google_docs.yaml), each write tool is offered
// to a session granting it and says the scope it needs, which the consent asks for. A
// manifest that stops asking for the scope leaves the tool out (AI-1394).
func (s *ToolsetSuite) TestGoogleDriveAndDocsOfferTheirWriteTools() {
	const driveFile, documents = "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/documents"
	for _, c := range []struct {
		id     string
		tools  []string
		writes map[string]string
	}{
		{"google_drive", []string{"copy_file", "create_file", "download_file_content", "get_file_metadata",
			"get_file_permissions", "list_recent_files", "read_file_content", "search_files"},
			map[string]string{"copy_file": driveFile, "create_file": driveFile}},
		{"google_docs", []string{"read_doc", "update_doc"}, map[string]string{"update_doc": documents}},
	} {
		s.Run(c.id, func() {
			srv := s.listing(c.id, c.tools...)
			resolved, err := s.manifest(c.id).Resolve(oauth2code.Name, nil, nil)
			s.Require().NoError(err)
			resolved.Endpoints["mcp"] = srv.URL
			binding := core.ResolvedBinding{Binding: core.Binding{Name: c.id}, Manifest: resolved, HTTP: srv.Client()}

			specs, err := mcp.New().Discover(s.ctx, binding)
			s.Require().NoError(err)
			needs := map[string][]string{}
			var grants []core.ToolGrant
			for _, spec := range specs {
				needs[spec.Name] = spec.NeedsScopes
				if _, write := c.writes[spec.Name]; write {
					grants = append(grants, core.ToolGrant{Name: spec.Name, SchemaDigest: spec.SchemaDigest})
				}
			}
			s.ElementsMatch(c.tools, slices.Collect(maps.Keys(needs)))
			for tool, scope := range c.writes {
				s.Equal([]string{scope}, needs[tool], tool)
				s.Contains(resolved.Scopes.List, scope, "the consent asks for what %s needs", tool)
			}
			toolset, err := mcp.New().Open(s.ctx, binding, grants)
			s.Require().NoError(err)
			var offered []string
			for _, tool := range toolset.Tools() {
				offered = append(offered, strings.TrimPrefix(tool.Name, c.id+mcp.Separator))
			}
			toolset.Close()
			s.ElementsMatch(slices.Collect(maps.Keys(c.writes)), offered)

			binding.Manifest.Scopes.List = slices.DeleteFunc(slices.Clone(resolved.Scopes.List), func(scope string) bool {
				return slices.Contains(slices.Collect(maps.Values(c.writes)), scope)
			})
			specs, err = mcp.New().Discover(s.ctx, binding)
			s.Require().NoError(err)
			var left []string
			for _, spec := range specs {
				left = append(left, spec.Name)
			}
			s.ElementsMatch(slices.DeleteFunc(slices.Clone(c.tools), func(tool string) bool {
				_, write := c.writes[tool]
				return write
			}), left, "without the scope, the tools that need it are left out")
		})
	}
}

// listing is an MCP server, named name, that lists tools, each taking an object.
func (s *ToolsetSuite) listing(name string, tools ...string) *httptest.Server {
	server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: name, Version: "1"}, nil)
	for _, tool := range tools {
		server.AddTool(&sdkmcp.Tool{Name: tool, InputSchema: map[string]any{"type": "object"}},
			func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
				return &sdkmcp.CallToolResult{}, nil
			})
	}
	srv := httptest.NewTLSServer(sdkmcp.NewStreamableHTTPHandler(func(*http.Request) *sdkmcp.Server { return server },
		&sdkmcp.StreamableHTTPOptions{Stateless: true, JSONResponse: true}))
	s.T().Cleanup(srv.Close)
	return srv
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
	// The built-ins whose endpoints take an input, each with a value its pattern accepts.
	inputs := map[string]map[string]string{
		"shopify":  {"shop": "mystore.myshopify.com"},
		"metabase": {"host": "metabase.example.com"},
	}[id]
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
