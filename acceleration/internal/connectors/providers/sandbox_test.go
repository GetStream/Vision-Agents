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
// (https://git-scm.com/docs/gitprotocol-http); a push's git-receive-pack is no read. From
// revision 9 gh reads with a POST of queries to api.github.com/graphql
// (https://docs.github.com/en/graphql/guides/forming-calls-with-graphql).
func (s *SandboxSuite) TestGitHubGivesASandboxItsTokenForTheRESTAPIAndGitHosts() {
	github := s.manifest("github")
	s.Equal(9, github.Revision)

	for _, scheme := range github.Schemes {
		resolved, err := github.Resolve(scheme, nil, nil)
		s.Require().NoError(err, scheme)

		s.Equal(&core.SandboxRule{Env: "GITHUB_TOKEN", Hosts: []string{"api.github.com", "uploads.github.com", "github.com"},
			Reads: []core.SandboxRead{{Host: "github.com", Method: "POST", Path: "/*/*/git-upload-pack"},
				{Host: "api.github.com", Path: "/graphql", GraphQL: true}}}, resolved.Sandbox, scheme)
	}
}

// Revisions 8 and 9 add reads alone, which a session takes from the latest revision, so a
// connection pinned to revision 7 reads as one on 9 and is not marked outdated.
func (s *SandboxSuite) TestGitHubsReadsNeedNoReconnect() {
	raw, err := fs.ReadFile(providers.FS, "github.yaml")
	s.Require().NoError(err)
	before, _, found := strings.Cut(string(raw), "  reads:\n")
	s.Require().True(found)
	seven, err := core.ParseManifest([]byte(strings.Replace(before, "revision: 9\n", "revision: 7\n", 1)))
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

// From revision 2, Linear's GraphQL API read with queries alone, and its file storage
// (https://linear.app/developers/file-storage-authentication), for linear_app's OAuth token
// (AI-2082).
func (s *SandboxSuite) TestLinearAppGivesASandboxItsTokenForQueriesOfTheGraphQLAPI() {
	linear := s.manifest("linear_app")
	s.Equal(2, linear.Revision)
	s.Require().NotEmpty(linear.Schemes)

	for _, scheme := range linear.Schemes {
		resolved, err := linear.Resolve(scheme, nil, nil)
		s.Require().NoError(err, scheme)

		s.Equal(&core.SandboxRule{Env: "LINEAR_ACCESS_TOKEN", Hosts: []string{"api.linear.app", "uploads.linear.app"},
			Reads: []core.SandboxRead{{Host: "api.linear.app", Path: "/graphql", GraphQL: true}}}, resolved.Sandbox, scheme)
	}
}

// Revision 2 adds the block itself, which a session reads from the revision a connection pins,
// so a connection pinned to revision 1 is outdated and gives the sandbox nothing until it is
// reconnected; the block is all that differs.
func (s *SandboxSuite) TestALinearAppConnectionOfRevisionOneNeedsAReconnect() {
	raw, err := fs.ReadFile(providers.FS, "linear_app.yaml")
	s.Require().NoError(err)
	before, _, found := strings.Cut(string(raw), "\nsandbox:\n")
	s.Require().True(found)
	one, err := core.ParseManifest([]byte(strings.Replace(before, "revision: 2\n", "revision: 1\n", 1)))
	s.Require().NoError(err)
	s.Equal(1, one.Revision)
	s.Nil(one.Sandbox)

	latest := s.manifest("linear_app")
	s.False(one.ReadsAsConnection(latest))
	latest.Sandbox = nil
	s.True(one.ReadsAsConnection(latest))
}

// From revision 4, BigQuery's REST API host, read with GETs and with one SELECT per jobs.query
// POST (AI-2114).
func (s *SandboxSuite) TestBigQueryGivesASandboxItsTokenForSELECTsOfTheRESTAPI() {
	bigquery := s.manifest("bigquery")
	s.Equal(4, bigquery.Revision)
	s.Require().NotEmpty(bigquery.Schemes)

	for _, scheme := range bigquery.Schemes {
		resolved, err := bigquery.Resolve(scheme, nil, nil)
		s.Require().NoError(err, scheme)

		s.Equal(&core.SandboxRule{Env: "BIGQUERY_TOKEN", Hosts: []string{"bigquery.googleapis.com"},
			Reads: []core.SandboxRead{{Host: "bigquery.googleapis.com", Path: "/bigquery/v2/projects/*/queries", SQL: true}}},
			resolved.Sandbox, scheme)
	}
}

// Revision 4 adds the block itself, so a connection pinned to revision 3 is outdated and gives
// the sandbox nothing until it is reconnected; the block is all that differs.
func (s *SandboxSuite) TestABigQueryConnectionOfRevisionThreeNeedsAReconnect() {
	raw, err := fs.ReadFile(providers.FS, "bigquery.yaml")
	s.Require().NoError(err)
	before, _, found := strings.Cut(string(raw), "\nsandbox:\n")
	s.Require().True(found)
	three, err := core.ParseManifest([]byte(strings.Replace(before, "revision: 4\n", "revision: 3\n", 1)))
	s.Require().NoError(err)
	s.Equal(3, three.Revision)
	s.Nil(three.Sandbox)

	latest := s.manifest("bigquery")
	s.False(three.ReadsAsConnection(latest))
	latest.Sandbox = nil
	s.True(three.ReadsAsConnection(latest))
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
// 8e132bd37 only github.yaml has one (the same grep); on b1cfbd31d github.yaml and
// google_drive.yaml (the same grep), and linear_app.yaml adds the third. linear.yaml has none:
// its DCR token gets 401 from api.linear.app/graphql (AI-2082). On e1a71dcc4 those three (the
// same grep), and bigquery.yaml adds the fourth.
func (s *SandboxSuite) TestNoOtherBuiltInHasASandbox() {
	files, err := fs.Glob(providers.FS, "*.yaml")
	s.Require().NoError(err)
	s.Require().NotEmpty(files)
	for _, file := range files {
		id := strings.TrimSuffix(file, ".yaml")
		if id == "github" || id == "google_drive" || id == "linear_app" || id == "bigquery" {
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
