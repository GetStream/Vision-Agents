package ingest

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
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

	passages := split("agents.md", content, DefaultChunk)

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

	passages := split("events.md", content, DefaultChunk)

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

	passages := split("usage.md", content, DefaultChunk)

	s.Require().Len(passages, 1)
	s.Contains(passages[0].text, "func main()")
	s.Contains(passages[0].text, `println("hello")`, "a blank line in code is not a new paragraph")
}

func (s *DocumentsSuite) TestPassagesAreIdentifiedByWhereTheyCameFrom() {
	// Ingesting the same documentation twice has to be an update rather than a second
	// copy of it, which is what the id being stable buys.
	directory := s.T().TempDir()
	s.write(directory, "guide.md", "# One\n\nfirst\n\n# Two\n\nsecond\n")

	documents, err := Read([]string{directory}, DefaultChunk)
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

	documents, err := Read([]string{directory}, DefaultChunk)
	s.Require().NoError(err)

	s.Require().Len(documents, 1)
	s.Equal("guide.md#0", documents[0].ID)
}

func (s *DocumentsSuite) TestNestedDocumentsKeepThePathTheyWereFoundAt() {
	directory := s.T().TempDir()
	s.write(filepath.Join(directory, "reference"), "api.md", "# API\n\nthe endpoints\n")

	documents, err := Read([]string{directory}, DefaultChunk)
	s.Require().NoError(err)

	s.Require().Len(documents, 1)
	s.Equal("reference/api.md#0", documents[0].ID)
	s.Equal("reference/api.md > API", documents[0].Source)
}

func (s *DocumentsSuite) TestAnIDThatFitsIsKeptAsItWas() {
	// What the ids were before long sources were hashed: a base filled then is found,
	// trimmed and deleted by the same ids without a re-index.
	s.Equal([]string{"guide.md#0", "guide.md#1"}, IDs("guide.md", 0, 2))
	s.Equal("guide.md#10", IDs("guide.md", 10, 11)[0])
	s.Equal([]string{
		"https://example.com/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa#0",
		"https://example.com/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa#9",
	}, []string{IDs(sixtyOne, 0, 1)[0], IDs(sixtyOne, 9, 10)[0]}, "63 bytes is the longest id kept")
}

func (s *DocumentsSuite) TestEveryIDFitsWhateverTheSourceAndIndex() {
	// turbopuffer refuses an id of 64 bytes or more, which every url over 61 bytes made.
	for _, source := range []string{
		sixtyOne,
		sixtyOne + "a",
		"https://getstream.io/chat/docs/sdk/android/client/guides/push-notifications/",
		"https://example.com/" + strings.Repeat("long/", 400),
	} {
		for _, index := range []int{0, 9, 10, 100_000, math.MaxInt} {
			id := passageID(source, index)
			s.LessOrEqual(len(id), 63, id)
			s.True(strings.HasSuffix(id, "#"+strconv.Itoa(index)), id)
		}
	}
}

func (s *DocumentsSuite) TestALongSourceIsKeyedByAHashOfAllOfIt() {
	push := "https://getstream.io/chat/docs/sdk/android/client/guides/push-notifications/"
	content := "# One\n\nfirst\n\n# Two\n\nsecond\n"

	written := Split(push, content, DefaultChunk)

	s.Require().Len(written, 2)
	s.Equal(IDs(push, 0, 2), []string{written[0].ID, written[1].ID},
		"reading back, trimming and deleting name what was written")
	s.Equal("s9X5PdOacmqTmtGOosmBp2IUcfgci9mQTB0qRg62q1k#0", written[0].ID,
		"a stored base is only found again while the hash stays the same")
	s.Equal(push+" > One", written[0].Source, "a reader still sees the url")

	// Two urls with the same first 60 bytes, as two pages of one docs section have.
	shared := "https://getstream.io/moderation/docs/node/content-moderation/"
	s.NotEqual(IDs(shared+"before-message-send/", 0, 1), IDs(shared+"custom-actions/", 0, 1))
}

func (s *DocumentsSuite) TestFilesCountsADocumentOnceThoughItsIDsAreKeyedTwoWays() {
	// Eleven passages of a 61-byte source: #0 to #9 fit as they are, #10 is hashed.
	var long strings.Builder
	for index := range 11 {
		long.WriteString("# Part " + strconv.Itoa(index) + "\n\ntext\n\n")
	}
	documents := append(Split(sixtyOne, long.String(), DefaultChunk), Split("guide.md", "# One\n\nfirst\n", DefaultChunk)...)
	s.Require().Len(documents, 12)

	s.Equal(2, Files(documents))
}

// sixtyOne is a source whose ids up to #9 are 63 bytes, the longest kept as they are.
var sixtyOne = "https://example.com/" + strings.Repeat("a", 41)

// write puts a file in a directory, creating it if it is not there yet.
func (s *DocumentsSuite) write(directory, name, content string) {
	s.Require().NoError(os.MkdirAll(directory, 0o755))
	s.Require().NoError(os.WriteFile(filepath.Join(directory, name), []byte(content), 0o644))
}
