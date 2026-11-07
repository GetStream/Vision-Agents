package main

import (
	"bytes"
	"crypto/rand"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
)

type CommandSuite struct {
	suite.Suite
}

func TestCommandSuite(t *testing.T) {
	suite.Run(t, new(CommandSuite))
}

func (s *CommandSuite) TestANegativeLimitIsRefused() {
	options := options{customer: "acme", namespace: "docs", size: ingest.DefaultChunk,
		limits: ingest.Limits{MinChunk: ingest.DefaultLimits.MinChunk, MaxHeading: -1}, dryRun: true}

	err := run(options, []string{s.T().TempDir()}, slog.New(slog.DiscardHandler))

	s.ErrorContains(err, "cannot be negative")
}

// The logger main builds takes a key out of a failure: a path that names one is not there, and
// the failure is logged without it.
func (s *CommandSuite) TestTheFailureIsLoggedWithoutTheKeyInIt() {
	if path := os.Getenv("KNOWLEDGE_MAIN_TEST_PATH"); path != "" {
		os.Args = []string{"knowledge", "-customer", "acme", "-namespace", "docs", path}
		main()
		return
	}
	key := rand.Text()
	path := filepath.Join(s.T().TempDir(), "Bearer sk-test-"+key)
	command := exec.Command(os.Args[0], "-test.run=^TestCommandSuite$/^TestTheFailureIsLoggedWithoutTheKeyInIt$")
	command.Env = append(os.Environ(), "KNOWLEDGE_MAIN_TEST_PATH="+path)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	s.Require().Error(command.Run(), "knowledge read a path that does not exist")

	s.NotContains(stderr.String(), key)
	s.Contains(stderr.String(), `msg="knowledge stopped" error="stat `+filepath.Dir(path)+`/[redacted] no such file or directory"`)
}
