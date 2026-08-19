package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type DocumentsSuite struct {
	suite.Suite
}

func TestDocumentsSuite(t *testing.T) {
	suite.Run(t, new(DocumentsSuite))
}

func (s *DocumentsSuite) TestADocumentIsCutAtItsHeadings() {
	content := `# Agents

An agent holds a conversation.

## Skills

A skill is handed to a slower model.
`

	passages := split("agents.md", content, defaultChunk)

	s.Require().Len(passages, 2)
	s.Equal("agents.md > Agents", passages[0].source)
	s.Contains(passages[0].text, "An agent holds a conversation.")
	s.NotContains(passages[0].text, "A skill is handed", "a section belongs to its own heading")
	s.Equal("agents.md > Skills", passages[1].source)
}

func (s *DocumentsSuite) TestALongSectionIsCutAgainAndKeepsItsHeading() {
	// The second half is prose with nothing saying what it is about, which is what
	// full-text search would otherwise have to match on.
	long := strings.Repeat("The router fails over between providers. ", 20)
	content := "## Routing\n\n" + long + "\n\n" + long

	passages := split("routing.md", content, 400)

	s.Require().Greater(len(passages), 1)
	for _, piece := range passages {
		s.Contains(piece.text, "Routing", "every piece says which section it came from")
		s.Equal("routing.md > Routing", piece.source)
	}
}

func (s *DocumentsSuite) TestASectionThatIsOnlyANameIsNotWorthStoring() {
	// A heading with the next heading under it says nothing. Handed back as an answer it
	// gives the model a title where it needed something to read.
	content := "# Events\n\n## Sending\n\nCall send with the event.\n"

	passages := split("events.md", content, defaultChunk)

	s.Require().Len(passages, 1)
	s.Equal("events.md > Sending", passages[0].source)
}

func (s *DocumentsSuite) TestATableIsCutAtItsRowsBecauseItHasNoParagraphs() {
	// A table is one paragraph however long it is, so cutting only at blank lines hands
	// the model the whole thing.
	var table strings.Builder
	table.WriteString("## Providers\n\n| Name | What it serves |\n| --- | --- |\n")
	for range 40 {
		table.WriteString("| deepgram | speech, in eleven languages, streamed |\n")
	}

	passages := split("providers.md", table.String(), 400)

	s.Require().Greater(len(passages), 1)
	for _, piece := range passages {
		s.Less(len(piece.text), 600, "no passage runs away with the whole table")
		s.Equal("providers.md > Providers", piece.source)
	}
}

func (s *DocumentsSuite) TestFencedCodeIsKeptWhole() {
	content := "## Usage\n\n```go\nfunc main() {\n\n\tprintln(\"hello\")\n}\n```\n"

	passages := split("usage.md", content, defaultChunk)

	s.Require().Len(passages, 1)
	s.Contains(passages[0].text, "func main()")
	s.Contains(passages[0].text, `println("hello")`, "a blank line in code is not a new paragraph")
}

func (s *DocumentsSuite) TestPassagesAreIdentifiedByWhereTheyCameFrom() {
	// Ingesting the same documentation twice has to be an update rather than a second
	// copy of it, which is what the id being stable buys.
	directory := s.T().TempDir()
	s.write(directory, "guide.md", "# One\n\nfirst\n\n# Two\n\nsecond\n")

	documents, err := read([]string{directory}, defaultChunk)
	s.Require().NoError(err)

	s.Require().Len(documents, 2)
	s.Equal("guide.md#0", documents[0].ID)
	s.Equal("guide.md#1", documents[1].ID)
	s.Equal("guide.md > One", documents[0].Source)
}

func (s *DocumentsSuite) TestOnlyProseIsIngested() {
	directory := s.T().TempDir()
	s.write(directory, "guide.md", "# One\n\nfirst\n")
	s.write(directory, "logo.png", "not a document")
	s.write(filepath.Join(directory, ".git"), "config", "# One\n\nnot documentation\n")

	documents, err := read([]string{directory}, defaultChunk)
	s.Require().NoError(err)

	s.Require().Len(documents, 1)
	s.Equal("guide.md#0", documents[0].ID)
}

func (s *DocumentsSuite) TestNestedDocumentsKeepThePathTheyWereFoundAt() {
	directory := s.T().TempDir()
	s.write(filepath.Join(directory, "reference"), "api.md", "# API\n\nthe endpoints\n")

	documents, err := read([]string{directory}, defaultChunk)
	s.Require().NoError(err)

	s.Require().Len(documents, 1)
	s.Equal("reference/api.md#0", documents[0].ID)
	s.Equal("reference/api.md > API", documents[0].Source)
}

// write puts a file in a directory, creating it if it is not there yet.
func (s *DocumentsSuite) write(directory, name, content string) {
	s.Require().NoError(os.MkdirAll(directory, 0o755))
	s.Require().NoError(os.WriteFile(filepath.Join(directory, name), []byte(content), 0o644))
}
