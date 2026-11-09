//go:build integration

package providers_test

import (
	"context"
	"io/fs"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/githubapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
)

// liveTimeout bounds the live listing: the router's connectorHTTPTimeout (cmd/router), 10 s,
// three times over for the MCP initialize and every tools/list page. Unverified, not measured.
const liveTimeout = 30 * time.Second

// GitHubLiveSuite checks the built-in github manifest with bearer against GitHub itself
// (AI-990): a personal access token from E2E_GITHUB_PAT, read from the environment only and
// never printed, lists GitHub's MCP tools at the manifest's mcp endpoint. Skipped without it;
// CI sets none.
type GitHubLiveSuite struct {
	suite.Suite
	token string
}

func TestGitHubLiveSuite(t *testing.T) {
	suite.Run(t, new(GitHubLiveSuite))
}

func (s *GitHubLiveSuite) SetupSuite() {
	s.token = os.Getenv("E2E_GITHUB_PAT")
	if s.token == "" {
		s.T().Skip("E2E_GITHUB_PAT is not set")
	}
}

func (s *GitHubLiveSuite) TestAPersonalAccessTokenListsGitHubsTools() {
	ctx, cancel := context.WithTimeout(context.Background(), liveTimeout)
	defer cancel()
	raw, err := fs.ReadFile(providers.FS, "github.yaml")
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	resolved, err := manifest.Resolve(bearer.Name, nil, nil)
	s.Require().NoError(err)
	scheme := bearer.New()
	stored, _, err := scheme.Complete(ctx, core.CompleteInput{Manifest: resolved, Supplied: map[string]string{bearer.SuppliedToken: s.token}})
	s.Require().NoError(err)
	credential, _, err := scheme.Retrieve(ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().NoError(err)
	client := &http.Client{Transport: scheme.Wrap(http.DefaultTransport, credential), Timeout: liveTimeout}

	specs, err := mcp.New().Discover(ctx, core.ResolvedBinding{Manifest: resolved, HTTP: client})

	s.Require().NoError(err)
	names := make([]string, 0, len(specs))
	for _, spec := range specs {
		names = append(names, spec.Name)
	}
	s.Contains(names, "get_me", "github-mcp-server's default toolset")
	s.Contains(names, "actions_list", "the actions toolset, which X-MCP-Toolsets asks for")
}

// GitHubAppLiveSuite checks the built-in github manifest with github_app against GitHub itself:
// an app's id, private key and installation, from E2E_GITHUB_APP_ID, E2E_GITHUB_APP_PRIVATE_KEY
// (the PEM, or a path to it) and E2E_GITHUB_APP_INSTALLATION_ID, read from the environment only
// and never printed, mint an installation token that GitHub's MCP server takes. Use a test app,
// never a production bot's key. Skipped without them; CI sets none.
type GitHubAppLiveSuite struct {
	suite.Suite
	supplied map[string]string
}

func TestGitHubAppLiveSuite(t *testing.T) {
	suite.Run(t, new(GitHubAppLiveSuite))
}

func (s *GitHubAppLiveSuite) SetupSuite() {
	app, key, installation := os.Getenv("E2E_GITHUB_APP_ID"), os.Getenv("E2E_GITHUB_APP_PRIVATE_KEY"), os.Getenv("E2E_GITHUB_APP_INSTALLATION_ID")
	if app == "" || key == "" || installation == "" {
		s.T().Skip("E2E_GITHUB_APP_ID, E2E_GITHUB_APP_PRIVATE_KEY and E2E_GITHUB_APP_INSTALLATION_ID are not all set")
	}
	if !strings.Contains(key, "PRIVATE KEY") {
		raw, err := os.ReadFile(key)
		s.Require().NoError(err, "E2E_GITHUB_APP_PRIVATE_KEY is neither a PEM nor a readable file")
		key = string(raw)
	}
	s.supplied = map[string]string{githubapp.SuppliedAppID: app, githubapp.SuppliedPrivateKey: key, githubapp.SuppliedInstallationID: installation}
}

func (s *GitHubAppLiveSuite) TestAnInstallationTokenListsGitHubsTools() {
	ctx, cancel := context.WithTimeout(context.Background(), liveTimeout)
	defer cancel()
	raw, err := fs.ReadFile(providers.FS, "github.yaml")
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	resolved, err := manifest.Resolve(githubapp.Name, nil, nil)
	s.Require().NoError(err)
	scheme, err := githubapp.New(githubapp.Config{HTTP: &http.Client{Timeout: liveTimeout}})
	s.Require().NoError(err)
	stored, _, err := scheme.Complete(ctx, core.CompleteInput{Manifest: resolved, Supplied: s.supplied})
	s.Require().NoError(err, "the token is minted and lists a repository of the installation")
	credential, _, err := scheme.Retrieve(ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().NoError(err)
	s.WithinDuration(time.Now().Add(time.Hour), credential.ExpiresAt, 5*time.Minute, "an installation token lasts an hour")
	client := &http.Client{Transport: scheme.Wrap(http.DefaultTransport, credential), Timeout: liveTimeout}

	specs, err := mcp.New().Discover(ctx, core.ResolvedBinding{Manifest: resolved, HTTP: client})

	s.Require().NoError(err, "GitHub's MCP server takes an installation token")
	names := make([]string, 0, len(specs))
	for _, spec := range specs {
		names = append(names, spec.Name)
	}
	s.Contains(names, "get_file_contents", "github-mcp-server's default toolset")
	s.NoError(scheme.Revoke(ctx, stored, resolved))
}
