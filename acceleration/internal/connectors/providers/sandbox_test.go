package providers_test

import (
	"io/fs"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
)

// SandboxSuite reads each built-in's sandbox block as a connection does: parsed, then resolved
// for one of its schemes.
type SandboxSuite struct {
	suite.Suite
}

func TestSandboxSuite(t *testing.T) {
	suite.Run(t, new(SandboxSuite))
}

func (s *SandboxSuite) TestGitHubGivesASandboxItsTokenForTheRESTAPIHostsFromRevisionSix() {
	github := s.manifest("github")
	s.Equal(6, github.Revision)

	for _, scheme := range github.Schemes {
		resolved, err := github.Resolve(scheme, nil, nil)
		s.Require().NoError(err, scheme)

		s.Equal(&core.SandboxRule{Env: "GITHUB_TOKEN", Hosts: []string{"api.github.com", "uploads.github.com"}}, resolved.Sandbox, scheme)
	}
}

// The control: on the base, b7187f62, no built-in had a sandbox block
// (git grep -n '^sandbox' b7187f622 -- acceleration/internal/connectors/providers: nothing).
func (s *SandboxSuite) TestNoOtherBuiltInHasASandbox() {
	files, err := fs.Glob(providers.FS, "*.yaml")
	s.Require().NoError(err)
	s.Require().NotEmpty(files)
	for _, file := range files {
		id := strings.TrimSuffix(file, ".yaml")
		if id == "github" {
			continue
		}
		s.Nil(s.manifest(id).Sandbox, id)
	}
}

func (s *SandboxSuite) manifest(id string) core.Manifest {
	raw, err := fs.ReadFile(providers.FS, id+".yaml")
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	return manifest
}
