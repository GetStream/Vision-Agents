//go:build integration

// Package ocrsuite is what every OCR provider is held to against its real API: a page of a
// manual comes back as its words in Markdown, with its heading, its numbered steps and its
// table kept, is counted as one page, and a blank page reads as no text.
//
// A provider suite embeds Suite, says how to build its provider, and inherits those tests.
package ocrsuite

import (
	"bytes"
	"context"
	_ "embed"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
	// The providers need their credentials, which live in the repository's .env rather
	// than in the environment an editor happens to run a test with.
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// page is a page of a fryer manual: a heading, a paragraph, three numbered steps and a
// table of four rows.
//
//go:embed testdata/page.png
var page []byte

// defaultTimeout bounds one page, looser than the router's attempt timeout.
const defaultTimeout = 90 * time.Second

// Suite is the shared behaviour. The fields are set where the suite is constructed.
type Suite struct {
	suite.Suite

	// New builds a provider.
	New func() (ocr.Provider, error)
	// Requires are the environment variables whose absence skips rather than fails.
	Requires []string
	// ByTokens marks a provider billed by the tokens a page took rather than by the page.
	ByTokens bool

	// Provider is built once and closed when the suite ends.
	Provider ocr.Provider
}

func (s *Suite) SetupSuite() {
	s.Require().NotNil(s.New, "a provider suite has to say how to build its provider")
	for _, name := range s.Requires {
		if os.Getenv(name) == "" {
			s.T().Skipf("%s not set", name)
		}
	}
	provider, err := s.New()
	s.Require().NoError(err)
	s.Provider = provider
}

func (s *Suite) TearDownSuite() {
	if s.Provider != nil {
		s.NoError(s.Provider.Close())
	}
}

// Read runs one page on the suite's provider and fails the test unless it succeeds.
func (s *Suite) Read(image []byte) ocr.Result {
	request := ocr.Request{Image: image}
	s.Require().NoError(request.Validate())
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()
	read, err := s.Provider.Read(ctx, request)
	s.Require().NoError(err)
	return read
}

func (s *Suite) TestAPageComesBackAsItsWordsInMarkdown() {
	read := s.Read(page)

	s.Regexp(`(?m)^#+ Fryer cleaning\s*$`, read.Markdown, "the heading is a heading")
	s.Contains(read.Markdown, "once the oil")
	s.Regexp(`(?m)^1\. Drain the oil into the disposal tank\.`, read.Markdown, "the steps are a numbered list")
	s.Regexp(`(?m)^3\. Wipe the vat with a dry cloth\.`, read.Markdown)
	s.Regexp(`(?m)^\|\s*Vat\s*\|\s*week\s*\|\s*boil-out\s*\|`, read.Markdown, "the table is a table")
}

func (s *Suite) TestAPageIsCountedAsOne() {
	read := s.Read(page)

	s.EqualValues(1, read.Pages)
	if s.ByTokens {
		s.Positive(read.InputTokens, "a picture costs tokens to look at")
		s.Positive(read.OutputTokens)
	}
}

func (s *Suite) TestABlankPageReadsAsNoText() {
	// A scanned document's blank pages are what tells a knowledge upload it has no text.
	read := s.Read(blank())

	s.Empty(letters.FindAllString(read.Markdown, -1), "found %q on a blank page", strings.TrimSpace(read.Markdown))
}

var letters = regexp.MustCompile(`\pL`)

// blank is a white page the size of the manual's.
func blank() []byte {
	white := image.NewGray(image.Rect(0, 0, 1000, 820))
	draw.Draw(white, white.Bounds(), &image.Uniform{C: color.White}, image.Point{}, draw.Src)
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, white); err != nil {
		panic(err)
	}
	return encoded.Bytes()
}
