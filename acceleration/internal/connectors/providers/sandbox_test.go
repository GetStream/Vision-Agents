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

// From revision 7 github.com too, where git sends the token as Basic auth's password. From
// revision 8 git clone and fetch read there with a POST to git-upload-pack
// (https://git-scm.com/docs/gitprotocol-http); a push's git-receive-pack is no read.
func (s *SandboxSuite) TestGitHubGivesASandboxItsTokenForTheRESTAPIAndGitHosts() {
	github := s.manifest("github")
	s.Equal(8, github.Revision)

	for _, scheme := range github.Schemes {
		resolved, err := github.Resolve(scheme, nil, nil)
		s.Require().NoError(err, scheme)

		s.Equal(&core.SandboxRule{Env: "GITHUB_TOKEN", Hosts: []string{"api.github.com", "uploads.github.com", "github.com"},
			Reads: []core.SandboxRead{{Host: "github.com", Method: "POST", Path: "/*/*/git-upload-pack"}}}, resolved.Sandbox, scheme)
	}
}

// Revision 8 adds reads alone, which a session takes from the latest revision, so a connection
// pinned to revision 7 reads as one on 8 and is not marked outdated.
func (s *SandboxSuite) TestGitHubsReadsNeedNoReconnect() {
	raw, err := fs.ReadFile(providers.FS, "github.yaml")
	s.Require().NoError(err)
	before, _, found := strings.Cut(string(raw), "  reads:\n")
	s.Require().True(found)
	seven, err := core.ParseManifest([]byte(strings.Replace(before, "revision: 8\n", "revision: 7\n", 1)))
	s.Require().NoError(err)
	s.Equal(7, seven.Revision)
	s.Nil(seven.Sandbox.Reads)

	s.True(seven.ReadsAsConnection(s.manifest("github")))
}

// From revision 5, Google Drive's REST API host (AI-1958).
func (s *SandboxSuite) TestGoogleDriveGivesASandboxItsTokenForTheRESTAPIHost() {
	drive := s.manifest("google_drive")
	s.Equal(6, drive.Revision)
	s.Require().NotEmpty(drive.Schemes)

	for _, scheme := range drive.Schemes {
		resolved, err := drive.Resolve(scheme, nil, nil)
		s.Require().NoError(err, scheme)

		s.Equal(&core.SandboxRule{Env: "GOOGLE_DRIVE_TOKEN", Hosts: []string{"www.googleapis.com"}}, resolved.Sandbox, scheme)
	}
}

// The router refuses two sandbox bindings with the same env (internal/api/configs.go), so no two
// built-ins may name one.
func (s *SandboxSuite) TestNoTwoBuiltInsShareASandboxEnv() {
	files, err := fs.Glob(providers.FS, "*.yaml")
	s.Require().NoError(err)
	owner := map[string]string{}
	for _, file := range files {
		id := strings.TrimSuffix(file, ".yaml")
		rule := s.manifest(id).Sandbox
		if rule == nil {
			continue
		}
		s.NotContains(owner, rule.Env, "%s and %s share %s", id, owner[rule.Env], rule.Env)
		owner[rule.Env] = id
	}
}

// The control: on the base, b7187f62, no built-in had a sandbox block
// (git grep -n '^sandbox' b7187f622 -- acceleration/internal/connectors/providers: nothing); on
// 8e132bd37 only github.yaml has one (the same grep).
func (s *SandboxSuite) TestNoOtherBuiltInHasASandbox() {
	files, err := fs.Glob(providers.FS, "*.yaml")
	s.Require().NoError(err)
	s.Require().NotEmpty(files)
	for _, file := range files {
		id := strings.TrimSuffix(file, ".yaml")
		if id == "github" || id == "google_drive" {
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
