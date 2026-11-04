// Package extract reads the files a business keeps what it knows in - PDFs, Word, OpenDocument
// and RTF documents, spreadsheets - as the Markdown a knowledge base is cut from.
//
// Every format comes out the same way: pages of Markdown, numbered when the file has pages,
// so a passage can say which page it came from. A page that is only a picture of text, as a
// scanner makes it, has nothing to extract. It is marked Scanned, and Render draws it for
// whatever reads pictures.
package extract

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Format is what a file turned out to be, read from its bytes rather than its name.
type Format string

const (
	Text Format = "text"
	PDF  Format = "pdf"
	DOCX Format = "docx"
	DOC  Format = "doc"
	XLSX Format = "xlsx"
	ODT  Format = "odt"
	RTF  Format = "rtf"
)

// Page is one page of a document as Markdown.
type Page struct {
	// Number is the page in the file, counted from 1. It is 0 for a file with no pages to
	// count, which is a spreadsheet, a text file or a document saved without page breaks.
	Number int
	Text   string
	// Scanned says the page is a picture with no text on it, so only reading the picture can
	// say what it holds.
	Scanned bool
}

// Image is a page drawn as a picture.
type Image struct {
	MIME string
	Data []byte
}

// Document is a file read as pages.
type Document struct {
	Format Format
	Pages  []Page
	render func(ctx context.Context, number int) (Image, error)
}

// Scanned are the numbers of the pages that are only a picture.
func (d Document) Scanned() []int {
	var numbers []int
	for _, page := range d.Pages {
		if page.Scanned {
			numbers = append(numbers, page.Number)
		}
	}
	return numbers
}

// Render draws a page as a picture, which is how a scanned page is read. Only a PDF has
// pages to draw.
func (d Document) Render(ctx context.Context, number int) (Image, error) {
	if d.render == nil {
		return Image{}, stack.Wrap(fmt.Errorf("extract: a %s has no pages to draw", d.Format))
	}
	return d.render(ctx, number)
}

// Markdown is the document as one text. Each numbered page opens with PageMarker, so the
// page a passage came from survives the text being stored, edited and posted again.
func (d Document) Markdown() string {
	var b strings.Builder
	for _, page := range d.Pages {
		text := strings.TrimSpace(page.Text)
		if page.Number > 0 {
			if b.Len() > 0 {
				b.WriteString("\n\n")
			}
			b.WriteString(PageMarker(page.Number))
		}
		if text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(text)
	}
	return b.String()
}

const (
	markerOpen  = "<!-- page "
	markerClose = " -->"
)

// PageMarker is the line that opens a page in a document's Markdown. It is an HTML comment,
// so Markdown rendered for a reader shows nothing of it.
func PageMarker(number int) string {
	return markerOpen + strconv.Itoa(number) + markerClose
}

// PageOf reads the page a PageMarker line opens.
func PageOf(line string) (int, bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(trimmed, markerOpen) || !strings.HasSuffix(trimmed, markerClose) {
		return 0, false
	}
	number, err := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(trimmed, markerOpen), markerClose))
	if err != nil || number <= 0 {
		return 0, false
	}
	return number, true
}

// Reason is why a file could not be read.
type Reason string

const (
	// Unsupported is a file in a format nothing here reads.
	Unsupported Reason = "unsupported"
	// Encrypted is a file that needs a password to open.
	Encrypted Reason = "encrypted"
	// Corrupt is a file whose bytes are not the format they claim to be.
	Corrupt Reason = "corrupt"
	// TooLarge is a file with more pages, or more unpacked bytes, than one is allowed.
	TooLarge Reason = "too_large"
)

// Error is a file that could not be read, and why.
type Error struct {
	Reason Reason
	Format Format
	Err    error
}

func (e *Error) Error() string {
	if e.Format == "" {
		return fmt.Sprintf("extract: %s: %v", e.Reason, e.Err)
	}
	return fmt.Sprintf("extract: %s %s: %v", e.Reason, e.Format, e.Err)
}

func (e *Error) Unwrap() error { return e.Err }

// ReasonOf is why err says a file could not be read, when it says.
func ReasonOf(err error) (Reason, bool) {
	var failed *Error
	if errors.As(err, &failed) {
		return failed.Reason, true
	}
	return "", false
}

func fail(reason Reason, format Format, err error) error {
	return stack.Wrap(&Error{Reason: reason, Format: format, Err: err})
}

// DefaultMaxPages is the most pages one file is read for: more than any manual somebody
// uploads, and few enough that a crafted file cannot keep a worker reading for an hour.
const DefaultMaxPages = 2000

// DefaultTimeout is how long reading one file may take.
const DefaultTimeout = 2 * time.Minute

// maxUnzipped is the most one part of an Office or OpenDocument file is read to once
// unpacked. Zip compresses repeated XML by a thousand times or more, so a small file can
// unpack to more memory than the process has.
const maxUnzipped = 64 << 20

// Options configures an Extractor.
type Options struct {
	// MaxPages is the most pages one file is read for. Zero is DefaultMaxPages.
	MaxPages int
	// Timeout is how long reading one file may take. Zero is DefaultTimeout.
	Timeout time.Duration
	// Workers is how many PDFs are read at once. Zero is 2.
	Workers int
	Logger  *slog.Logger
}

// Extractor reads files. It holds the PDF engine, which is compiled the first time a PDF
// is read rather than when the process starts, since most never read one.
type Extractor struct {
	options Options

	once    sync.Once
	pdf     *pdfEngine
	pdfErr  error
	closeMu sync.Mutex
	closed  bool
}

// New returns an Extractor. Close it to release the PDF engine.
func New(options Options) *Extractor {
	if options.MaxPages <= 0 {
		options.MaxPages = DefaultMaxPages
	}
	if options.Timeout <= 0 {
		options.Timeout = DefaultTimeout
	}
	if options.Workers <= 0 {
		options.Workers = 2
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	return &Extractor{options: options}
}

// Close releases the PDF engine, when one was started.
func (e *Extractor) Close() error {
	e.closeMu.Lock()
	defer e.closeMu.Unlock()
	e.closed = true
	if e.pdf == nil {
		return nil
	}
	return e.pdf.close()
}

// Extract reads a file as pages. The name is only used to say what failed: the format is
// read from the bytes, since a name is whatever somebody called the file.
func (e *Extractor) Extract(ctx context.Context, name string, data []byte) (Document, error) {
	ctx, cancel := context.WithTimeout(ctx, e.options.Timeout)
	defer cancel()

	format, err := detect(data)
	if err != nil {
		return Document{}, err
	}
	var document Document
	switch format {
	case Text:
		document, err = readText(data)
	case PDF:
		document, err = e.readPDF(ctx, data)
	case DOCX:
		document, err = readDOCX(data)
	case DOC:
		document, err = readDOC(data)
	case XLSX:
		document, err = readXLSX(data)
	case ODT:
		document, err = readODT(data)
	case RTF:
		document, err = readRTF(data)
	}
	if err != nil {
		return Document{}, err
	}
	if len(document.Pages) > e.options.MaxPages {
		return Document{}, fail(TooLarge, format, fmt.Errorf("%s has %d pages, more than the %d allowed", name, len(document.Pages), e.options.MaxPages))
	}
	document.Format = format
	return document, nil
}

var (
	pdfMagic = []byte("%PDF-")
	zipMagic = []byte("PK\x03\x04")
	oleMagic = []byte{0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1}
	rtfMagic = []byte(`{\rtf`)
	utf8BOM  = []byte{0xEF, 0xBB, 0xBF}
)

// detect reads what a file is from its first bytes and, for the formats that are zip or
// OLE containers, from what the container holds.
func detect(data []byte) (Format, error) {
	// A PDF may carry junk before its header, and readers accept one within the first
	// kilobyte of the file.
	if bytes.Contains(data[:min(len(data), 1024)], pdfMagic) {
		return PDF, nil
	}
	switch {
	case bytes.HasPrefix(data, zipMagic):
		return detectZip(data)
	case bytes.HasPrefix(data, oleMagic):
		return detectOLE(data)
	case bytes.HasPrefix(data, rtfMagic):
		return RTF, nil
	}
	if utf8.Valid(data) && !bytes.ContainsRune(data, 0) {
		return Text, nil
	}
	return "", fail(Unsupported, "", errors.New("the file is not text, and not a format that can be read"))
}

func detectZip(data []byte) (Format, error) {
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return "", fail(Corrupt, "", fmt.Errorf("the zip archive cannot be opened: %w", err))
	}
	for _, file := range archive.File {
		switch file.Name {
		case "word/document.xml":
			return DOCX, nil
		case "xl/workbook.xml":
			return XLSX, nil
		case "ppt/presentation.xml":
			return "", fail(Unsupported, "", errors.New("PowerPoint presentations cannot be read yet"))
		case "mimetype":
			kind, err := readZipFile(file)
			if err != nil {
				return "", err
			}
			if strings.TrimSpace(string(kind)) == "application/vnd.oasis.opendocument.text" {
				return ODT, nil
			}
			return "", fail(Unsupported, "", fmt.Errorf("an OpenDocument %s cannot be read, only a text document", kind))
		}
	}
	return "", fail(Unsupported, "", errors.New("a zip archive that is not a Word, Excel or OpenDocument file"))
}

func readText(data []byte) (Document, error) {
	return Document{Pages: []Page{{Text: string(bytes.TrimPrefix(data, utf8BOM))}}}, nil
}

// readZipFile reads one part of an archive, no further than maxUnzipped.
func readZipFile(file *zip.File) ([]byte, error) {
	reader, err := file.Open()
	if err != nil {
		return nil, fail(Corrupt, "", fmt.Errorf("%s cannot be unpacked: %w", file.Name, err))
	}
	defer reader.Close()
	var b bytes.Buffer
	n, err := b.ReadFrom(io.LimitReader(reader, maxUnzipped+1))
	if err != nil {
		return nil, fail(Corrupt, "", fmt.Errorf("%s cannot be unpacked: %w", file.Name, err))
	}
	if n > maxUnzipped {
		return nil, fail(TooLarge, "", fmt.Errorf("%s unpacks to more than %d bytes", file.Name, maxUnzipped))
	}
	return b.Bytes(), nil
}

// zipFile reads the part of an archive with this name, or nil when it has none.
func zipFile(archive *zip.Reader, name string) ([]byte, error) {
	for _, file := range archive.File {
		if file.Name == name {
			return readZipFile(file)
		}
	}
	return nil, nil
}
