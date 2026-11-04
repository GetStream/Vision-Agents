package extract

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"image/jpeg"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type ExtractSuite struct {
	suite.Suite
	extractor *Extractor
}

func TestExtractSuite(t *testing.T) {
	suite.Run(t, new(ExtractSuite))
}

func (s *ExtractSuite) SetupSuite() {
	s.extractor = New(Options{})
}

func (s *ExtractSuite) TearDownSuite() {
	s.Require().NoError(s.extractor.Close())
}

func (s *ExtractSuite) fixture(name string) []byte {
	data, err := os.ReadFile(filepath.Join("testdata", name))
	s.Require().NoError(err)
	return data
}

func (s *ExtractSuite) extract(name string) Document {
	document, err := s.extractor.Extract(context.Background(), name, s.fixture(name))
	s.Require().NoError(err)
	return document
}

func (s *ExtractSuite) page(document Document, number int) string {
	for _, page := range document.Pages {
		if page.Number == number {
			return page.Text
		}
	}
	s.FailNow("no such page", "page %d of %d", number, len(document.Pages))
	return ""
}

func (s *ExtractSuite) requireReason(err error, reason Reason) {
	s.Require().Error(err)
	got, ok := ReasonOf(err)
	s.Require().True(ok, "%v says no reason", err)
	s.Equal(reason, got, err.Error())
}

// rezip is a copy of an archive with some of its parts replaced or added.
func (s *ExtractSuite) rezip(data []byte, parts map[string][]byte) []byte {
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	s.Require().NoError(err)
	var out bytes.Buffer
	writer := zip.NewWriter(&out)
	for _, file := range archive.File {
		if _, replaced := parts[file.Name]; replaced {
			continue
		}
		s.Require().NoError(writer.Copy(file))
	}
	for name, content := range parts {
		part, err := writer.Create(name)
		s.Require().NoError(err)
		_, err = part.Write(content)
		s.Require().NoError(err)
	}
	s.Require().NoError(writer.Close())
	return out.Bytes()
}

func (s *ExtractSuite) TestAPDFPageSaysWhatIsOnItUnderItsNumber() {
	document := s.extract("manual.pdf")

	s.Equal(PDF, document.Format)
	s.Len(document.Pages, 12)
	s.Contains(s.page(document, 7), "Fryer oil is changed every 48 hours, or sooner when it smokes or darkens.",
		"the lines of a paragraph are one paragraph")
	s.NotContains(s.page(document, 6), "Fryer oil")
}

func (s *ExtractSuite) TestAPDFRunningHeadAndPageNumbersAreLeftOutButItsChapterTitlesAreNot() {
	document := s.extract("manual.pdf")

	for _, page := range document.Pages {
		s.NotContains(page.Text, "Kitchen Operations", "page %d repeats the running head", page.Number)
		lines := strings.Split(page.Text, "\n")
		s.NotEqual(strconv.Itoa(page.Number), lines[len(lines)-1], "page %d ends with its number", page.Number)
	}
	s.True(strings.HasPrefix(s.page(document, 3), "# Chapter 3: Station 3\n\n"), s.page(document, 3))
}

func (s *ExtractSuite) TestAPDFTableIsATable() {
	text := s.page(s.extract("manual.pdf"), 5)

	s.Contains(text, "| Equipment | Frequency | Owner |\n| --- | --- | --- |\n| Fryer | Daily | Shift lead |\n| Grill | Every shift | Cook |")
	s.Contains(text, "| Ice machine | Weekly | Manager |\n\nLog every cleaning", "the paragraph under the table is not a row of it")
}

func (s *ExtractSuite) TestAPDFBulletedListIsAListUnderItsHeading() {
	text := s.page(s.extract("manual.pdf"), 3)

	s.Contains(text, "## Before you start\n\n- Wash your hands\n- Put on gloves\n- Check the thermometer")
}

func (s *ExtractSuite) TestTwoColumnsAreReadOneAfterTheOther() {
	// The file draws the columns a line of each at a time, so read in its own order they
	// interleave.
	text := s.page(s.extract("columns.pdf"), 1)

	s.Contains(text, "# Newsletter\n\nThe left column starts the story about the new menu, which launches in every store on the first of the month with three new items.")
	s.Contains(text, "The right column carries on with the training plan: every team member watches the video and passes the quiz before the launch day.")
	s.Less(strings.Index(text, "left column"), strings.Index(text, "right column"))
}

func (s *ExtractSuite) TestAScannedPageIsMarkedAndCanBeDrawn() {
	document := s.extract("scanned.pdf")

	s.Equal([]int{2}, document.Scanned(), "the page with a text layer is read, the picture of one is not")
	s.Contains(s.page(document, 1), "This first page has a text layer.")

	drawn, err := document.Render(context.Background(), 2)
	s.Require().NoError(err)
	s.Equal("image/jpeg", drawn.MIME)
	picture, err := jpeg.Decode(bytes.NewReader(drawn.Data))
	s.Require().NoError(err)
	s.Greater(picture.Bounds().Dy(), 1500, "an A4 page drawn finely enough to read")
}

func (s *ExtractSuite) TestAPDFWithAPasswordIsRefusedAsEncrypted() {
	_, err := s.extractor.Extract(context.Background(), "encrypted.pdf", s.fixture("encrypted.pdf"))

	s.requireReason(err, Encrypted)
}

func (s *ExtractSuite) TestABrokenPDFIsRefusedAsCorrupt() {
	_, err := s.extractor.Extract(context.Background(), "corrupt.pdf", s.fixture("corrupt.pdf"))

	s.requireReason(err, Corrupt)
}

func (s *ExtractSuite) TestAPDFWithMorePagesThanAllowedIsRefused() {
	small := New(Options{MaxPages: 5})
	defer small.Close()

	_, err := small.Extract(context.Background(), "manual.pdf", s.fixture("manual.pdf"))

	s.requireReason(err, TooLarge)
}

func (s *ExtractSuite) TestReadingStopsAtTheDeadline() {
	hurried := New(Options{Timeout: time.Nanosecond})
	defer hurried.Close()

	_, err := hurried.Extract(context.Background(), "manual.pdf", s.fixture("manual.pdf"))

	s.Require().ErrorIs(err, context.DeadlineExceeded)
}

func (s *ExtractSuite) TestAWordDocumentKeepsItsHeadingsListsTablesAndPages() {
	document := s.extract("handbook.docx")

	s.Equal(DOCX, document.Format)
	s.Require().Len(document.Pages, 2)
	first := s.page(document, 1)
	s.Contains(first, "# Employee Handbook\n\nWelcome to the team.")
	s.Contains(first, "## Uniforms\n\n- Wear a clean shirt\n- Wear non-slip shoes\n  - Black or brown\n1. Clock in\n1. Wash your hands")
	second := s.page(document, 2)
	s.True(strings.HasPrefix(second, "Page two starts with the break policy."), second)
	s.Contains(second, "| Shift length | Break |\n| --- | --- |\n| 6 hours | 30 minutes |\n| 8 hours | 45 minutes |")
}

func (s *ExtractSuite) TestAWordDocumentThatRecordsNoPagesIsOnePageWithNoNumber() {
	// A break put in by hand is the only one such a file says anything about, and
	// numbering by those would call the thirtieth page the second.
	data := s.fixture("handbook.docx")
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	s.Require().NoError(err)
	body, err := zipFile(archive, "word/document.xml")
	s.Require().NoError(err)
	body = bytes.ReplaceAll(body, []byte("<w:lastRenderedPageBreak/>"), nil)

	document, err := s.extractor.Extract(context.Background(), "handbook.docx", s.rezip(data, map[string][]byte{"word/document.xml": body}))

	s.Require().NoError(err)
	s.Require().Len(document.Pages, 1)
	s.Zero(document.Pages[0].Number)
	s.Contains(document.Pages[0].Text, "Page two starts with the break policy.")
	s.NotContains(document.Markdown(), "<!-- page")
}

func (s *ExtractSuite) TestAWordDocumentWithAPasswordIsRefusedAsEncrypted() {
	_, err := s.extractor.Extract(context.Background(), "encrypted.docx", s.fixture("encrypted.docx"))

	s.requireReason(err, Encrypted)
}

func (s *ExtractSuite) TestAPartThatUnpacksPastTheLimitIsRefused() {
	// Zeros compress a thousand times over, so this archive is small and would unpack to
	// more than the limit.
	bomb := bytes.Repeat([]byte{' '}, maxUnzipped+1)
	data := s.rezip(s.fixture("handbook.docx"), map[string][]byte{"word/document.xml": bomb})
	s.Less(len(data), 1<<20)

	_, err := s.extractor.Extract(context.Background(), "bomb.docx", data)

	s.requireReason(err, TooLarge)
}

func (s *ExtractSuite) TestAnOpenDocumentKeepsItsHeadingsListsTablesAndPages() {
	document := s.extract("handbook.odt")

	s.Equal(ODT, document.Format)
	s.Require().Len(document.Pages, 2)
	s.Contains(s.page(document, 1), "## Uniforms\n\n- Wear a clean shirt\n- Wear non-slip shoes\n  - Black or brown\n1. Clock in\n1. Wash your hands")
	s.Contains(s.page(document, 2), "## Breaks\n\n| Shift length | Break |\n| --- | --- |\n| 6 hours | 30 minutes |")
}

func (s *ExtractSuite) TestAnOpenDocumentWithAPasswordIsRefusedAsEncrypted() {
	manifest := []byte(`<manifest:manifest><manifest:file-entry manifest:full-path="content.xml"><manifest:encryption-data/></manifest:file-entry></manifest:manifest>`)
	data := s.rezip(s.fixture("handbook.odt"), map[string][]byte{"META-INF/manifest.xml": manifest})

	_, err := s.extractor.Extract(context.Background(), "locked.odt", data)

	s.requireReason(err, Encrypted)
}

func (s *ExtractSuite) TestRichTextKeepsItsHeadingsListsAndTables() {
	document := s.extract("handbook.rtf")

	s.Equal(RTF, document.Format)
	s.Require().Len(document.Pages, 1)
	s.Zero(document.Pages[0].Number, "RTF records no pages")
	text := document.Pages[0].Text
	s.True(strings.HasPrefix(text, "# Employee Handbook\n\nWelcome to the team."), text)
	s.Contains(text, "- Wear non-slip shoes\n  - Black or brown\n1. Clock in")
	s.Contains(text, "## Breaks\n\n| Shift length | Break |\n| --- | --- |\n| 6 hours | 30 minutes |\n| 8 hours | 45 minutes |\n\nBreaks are taken in the back room.")
	s.NotContains(text, "Helvetica", "the font table is not text")
}

func (s *ExtractSuite) TestRichTextReadsItsCodePageAndUnicode() {
	// \'e9 is é in Windows-1252, and the ? after \u8364 is what a reader without Unicode
	// would show instead of the euro sign.
	data := []byte(`{\rtf1\ansi\ansicpg1252{\fonttbl{\f0 Arial;}}\pard Caf\'e9 prices in \u8364?.\par}`)

	document, err := s.extractor.Extract(context.Background(), "prices.rtf", data)

	s.Require().NoError(err)
	s.Equal("Café prices in €.", document.Pages[0].Text)
}

func (s *ExtractSuite) TestALegacyWordDocumentKeepsItsParagraphsListsAndTables() {
	document := s.extract("policies.doc")

	s.Equal(DOC, document.Format)
	text := document.Pages[0].Text
	s.Contains(text, "Store Policies\n\nOpening hours are 6am to 10pm.")
	s.Contains(text, "- Wash hands\n- Wear gloves")
	s.Contains(text, "| Item | Price |\n| --- | --- |\n| Sandwich | 4.50 |\n| Lemonade | 2.25 |\n\nRefunds are given within 30 days with a receipt.")
}

func (s *ExtractSuite) TestAWorkbookIsATablePerSheetWithItsValuesAsTheSheetShowsThem() {
	document := s.extract("menu.xlsx")

	s.Equal(XLSX, document.Format)
	text := document.Pages[0].Text
	s.Contains(text, "## Menu\n\n| Item | Price | Calories |\n| --- | --- | --- |\n| Chicken sandwich | $4.50 | 440 |")
	s.Contains(text, "## Hours\n\n| Day | Opens | Closes |")
	s.NotContains(text, "Empty", "a sheet with nothing on it is left out")
}

func (s *ExtractSuite) TestAPresentationIsRefusedAsUnsupported() {
	var out bytes.Buffer
	writer := zip.NewWriter(&out)
	part, err := writer.Create("ppt/presentation.xml")
	s.Require().NoError(err)
	_, err = part.Write([]byte("<p:presentation/>"))
	s.Require().NoError(err)
	s.Require().NoError(writer.Close())

	_, err = s.extractor.Extract(context.Background(), "deck.pptx", out.Bytes())

	s.requireReason(err, Unsupported)
}

func (s *ExtractSuite) TestTextIsReadAsItIsWithoutItsByteOrderMark() {
	document, err := s.extractor.Extract(context.Background(), "notes.md", []byte("\xEF\xBB\xBF# Notes\n\nOpen at six."))

	s.Require().NoError(err)
	s.Equal(Text, document.Format)
	s.Equal([]Page{{Text: "# Notes\n\nOpen at six."}}, document.Pages)
}

func (s *ExtractSuite) TestBytesThatAreNoFormatAreRefusedAsUnsupported() {
	_, err := s.extractor.Extract(context.Background(), "photo.png", []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"))

	s.requireReason(err, Unsupported)
}

func (s *ExtractSuite) TestAPageMarkerNamesThePageItOpens() {
	document := Document{Pages: []Page{{Number: 1, Text: "One."}, {Number: 2, Text: ""}, {Number: 3, Text: "Three."}}}

	s.Equal("<!-- page 1 -->\n\nOne.\n\n<!-- page 2 -->\n\n<!-- page 3 -->\n\nThree.", document.Markdown())
	number, ok := PageOf("  <!-- page 12 -->  ")
	s.True(ok)
	s.Equal(12, number)
	_, ok = PageOf("<!-- page twelve -->")
	s.False(ok)
}

func (s *ExtractSuite) TestAClosedExtractorRefusesAPDF() {
	closed := New(Options{})
	s.Require().NoError(closed.Close())

	_, err := closed.Extract(context.Background(), "manual.pdf", s.fixture("manual.pdf"))

	s.Require().Error(err)
	_, ok := ReasonOf(err)
	s.False(ok, "a closed extractor is not a fault of the file")
	s.False(errors.Is(err, context.DeadlineExceeded))
}

func (s *ExtractSuite) TestEveryFormatThatCanBeReadPassesASniff() {
	for _, name := range []string{"manual.pdf", "handbook.docx", "handbook.odt", "handbook.rtf", "menu.xlsx", "policies.doc"} {
		s.NoError(Sniff(s.fixture(name)[:SniffBytes]), name)
	}
	s.NoError(Sniff([]byte("# Opening hours\n\nNine to five.")))
}

func (s *ExtractSuite) TestAPhotoIsRefusedFromItsFirstBytes() {
	photo := append([]byte{0xFF, 0xD8, 0xFF, 0xE0, 0, 0x10}, []byte("JFIF\x00")...)

	s.requireReason(Sniff(photo), Unsupported)
}

func (s *ExtractSuite) TestTextCutPartWayThroughACharacterPassesASniff() {
	text := bytes.Repeat([]byte("é"), SniffBytes)

	s.NoError(Sniff(text[:SniffBytes-1]))
}
