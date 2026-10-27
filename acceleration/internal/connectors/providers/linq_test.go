package providers_test

import (
	"io/fs"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
)

// LinqSuite reads the built-in linq manifest against the core's recorded Linq message.
type LinqSuite struct {
	suite.Suite
}

func TestLinqSuite(t *testing.T) {
	suite.Run(t, new(LinqSuite))
}

// A build from before text_parts ignores it, so the stored revision's text must still read
// the first part: an empty text would drop every Linq message after a rollback (AI-2092).
func (s *LinqSuite) TestARollbackToABuildWithoutTextPartsStillReadsTheMessage() {
	raw, err := fs.ReadFile(providers.FS, "linq.yaml")
	s.Require().NoError(err)
	m, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	s.Equal(2, m.Revision)
	s.Equal("$.data.parts[*].value", m.Channel.Messages.TextParts)

	body, err := os.ReadFile(filepath.Join("..", "core", "testdata", "recorded", "linq.received.json"))
	s.Require().NoError(err)
	older := *m.Channel
	older.Messages.TextParts = ""
	read, err := older.Read(m.ID, body)
	s.Require().NoError(err)
	s.Require().Len(read.Messages, 1)
	s.Equal("Hi, is my order ready?", read.Messages[0].Text)
}
