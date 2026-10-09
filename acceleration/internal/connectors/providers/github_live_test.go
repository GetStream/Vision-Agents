//go:build integration

package providers_test

import (
	"context"
	"io/fs"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
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
}
