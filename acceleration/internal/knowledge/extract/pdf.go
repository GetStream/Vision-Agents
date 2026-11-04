package extract

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image/jpeg"
	"io"
	"strings"

	"github.com/klippa-app/go-pdfium"
	"github.com/klippa-app/go-pdfium/enums"
	pdfiumerrors "github.com/klippa-app/go-pdfium/errors"
	"github.com/klippa-app/go-pdfium/references"
	"github.com/klippa-app/go-pdfium/requests"
	"github.com/klippa-app/go-pdfium/webassembly"
	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/experimental"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// pdfMemory is the most memory one PDF worker may grow to. A WebAssembly page is 64 KiB.
// The whole file sits in it, so it is ten times the largest upload with room for the
// engine's own structures.
const pdfMemory = 512 << 20

// renderDPI is how finely a scanned page is drawn for reading: enough for 8-point print to
// come out legible, and an A4 page stays under 2,000 pixels on its long side, which is
// what vision models scale a picture down to anyway.
const renderDPI = 160

// renderQuality is the JPEG quality a drawn page is encoded at. A scan is a photograph
// of paper, which JPEG keeps legible at a fraction of PNG's size.
const renderQuality = 85

// pdfEngine is PDFium compiled to WebAssembly and run in process, so reading a PDF needs
// neither cgo nor a program installed next to the router.
type pdfEngine struct {
	pool pdfium.Pool
}

func startPDF(workers int) (*pdfEngine, error) {
	pool, err := webassembly.Init(webassembly.Config{
		MaxIdle:  workers,
		MaxTotal: workers,
		// Nothing is mounted: a document is handed over as bytes, and a crafted one has no
		// file of the host's to reach.
		FSConfig: wazero.NewFSConfig(),
		RuntimeConfig: wazero.NewRuntimeConfig().
			WithCoreFeatures(api.CoreFeaturesV2 | experimental.CoreFeaturesExceptionHandling).
			WithMemoryLimitPages(pdfMemory / (64 << 10)),
		Stdout: io.Discard,
		Stderr: io.Discard,
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("extract: start the PDF engine: %w", err))
	}
	return &pdfEngine{pool: pool}, nil
}

func (p *pdfEngine) close() error {
	return p.pool.Close()
}

func (e *Extractor) engine() (*pdfEngine, error) {
	e.closeMu.Lock()
	defer e.closeMu.Unlock()
	if e.closed {
		return nil, stack.Wrap(errors.New("extract: the extractor is closed"))
	}
	e.once.Do(func() { e.pdf, e.pdfErr = startPDF(e.options.Workers) })
	return e.pdf, e.pdfErr
}

// pdfSession is one worker with one document open in it.
type pdfSession struct {
	instance pdfium.Pdfium
	document references.FPDF_DOCUMENT
	stop     func() bool
}

// open takes a worker and opens the document in it. PDFium takes no context, so a file
// that keeps it busy past the deadline is stopped by killing the worker it runs in.
func (e *Extractor) open(ctx context.Context, data []byte) (*pdfSession, error) {
	engine, err := e.engine()
	if err != nil {
		return nil, err
	}
	instance, err := engine.pool.GetInstanceWithContext(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("extract: a PDF worker: %w", err))
	}
	stop := context.AfterFunc(ctx, func() { _ = instance.Kill() })
	opened, err := instance.OpenDocument(&requests.OpenDocument{File: &data})
	if err != nil {
		stop()
		_ = instance.Close()
		if ctx.Err() != nil {
			return nil, stack.Wrap(fmt.Errorf("extract: open the PDF: %w", ctx.Err()))
		}
		if errors.Is(err, pdfiumerrors.ErrPassword) {
			return nil, fail(Encrypted, PDF, errors.New("the PDF is protected by a password"))
		}
		return nil, fail(Corrupt, PDF, fmt.Errorf("the PDF cannot be opened: %w", err))
	}
	return &pdfSession{instance: instance, document: opened.Document, stop: stop}, nil
}

func (s *pdfSession) close() {
	_, _ = s.instance.FPDF_CloseDocument(&requests.FPDF_CloseDocument{Document: s.document})
	s.stop()
	_ = s.instance.Close()
}

func (s *pdfSession) page(index int) requests.Page {
	return requests.Page{ByIndex: &requests.PageByIndex{Document: s.document, Index: index}}
}

// pdfPage is what one page holds, before it is laid out.
type pdfPage struct {
	runs   []run
	images int
	// unreadable is how many characters have no Unicode mapping, which is what a font
	// embedded without one extracts as. A page of them reads as nothing, like a scan.
	unreadable int
	chars      int
	width      float64
	height     float64
}

// run is text on one line in one font, where the page puts it. PDF coordinates grow
// upwards, so top is greater than bottom.
type run struct {
	text                     string
	left, right, top, bottom float64
	size                     float64
	bold                     bool
}

func (s *pdfSession) read(index int) (pdfPage, error) {
	var page pdfPage
	size, err := s.instance.FPDF_GetPageSizeByIndex(&requests.FPDF_GetPageSizeByIndex{Document: s.document, Index: index})
	if err != nil {
		return page, err
	}
	page.width, page.height = size.Width, size.Height

	text, err := s.instance.GetPageTextStructured(&requests.GetPageTextStructured{
		Page:                   s.page(index),
		Mode:                   requests.GetPageTextStructuredModeRects,
		CollectFontInformation: true,
	})
	if err != nil {
		return page, err
	}
	for _, rect := range text.Rects {
		// PDFium reports the hyphen a line was broken at as U+0002.
		content := strings.ReplaceAll(strings.ReplaceAll(rect.Text, "\x00", ""), "\x02", "-")
		for _, r := range content {
			if r == '\uFFFD' || (r < ' ' && r != '\t' && r != '\r' && r != '\n') {
				page.unreadable++
			} else if r != ' ' {
				page.chars++
			}
		}
		if strings.TrimSpace(content) == "" {
			continue
		}
		piece := run{
			text: content,
			left: rect.PointPosition.Left, right: rect.PointPosition.Right,
			top: rect.PointPosition.Top, bottom: rect.PointPosition.Bottom,
		}
		if font := rect.FontInformation; font != nil {
			piece.size = font.Size
			if font.RenderedSize > 0 {
				piece.size = font.RenderedSize
			}
			piece.bold = font.Weight >= 600 || strings.Contains(strings.ToLower(font.Name), "bold")
		}
		page.runs = append(page.runs, piece)
	}

	loaded, err := s.instance.FPDF_LoadPage(&requests.FPDF_LoadPage{Document: s.document, Index: index})
	if err != nil {
		return page, err
	}
	defer func() { _, _ = s.instance.FPDF_ClosePage(&requests.FPDF_ClosePage{Page: loaded.Page}) }()
	objects, err := s.instance.FPDFPage_CountObjects(&requests.FPDFPage_CountObjects{Page: requests.Page{ByReference: &loaded.Page}})
	if err != nil {
		return page, err
	}
	for i := range objects.Count {
		object, err := s.instance.FPDFPage_GetObject(&requests.FPDFPage_GetObject{Page: requests.Page{ByReference: &loaded.Page}, Index: i})
		if err != nil {
			return page, err
		}
		kind, err := s.instance.FPDFPageObj_GetType(&requests.FPDFPageObj_GetType{PageObject: object.PageObject})
		if err != nil {
			return page, err
		}
		if kind.Type == enums.FPDF_PAGEOBJ_IMAGE {
			page.images++
		}
	}
	return page, nil
}

// readPDF reads every page's text. A page with a picture and next to no text on it is
// marked Scanned, and the document keeps the file so Render can draw it.
func (e *Extractor) readPDF(ctx context.Context, data []byte) (Document, error) {
	session, err := e.open(ctx, data)
	if err != nil {
		return Document{}, err
	}
	defer session.close()

	count, err := session.instance.FPDF_GetPageCount(&requests.FPDF_GetPageCount{Document: session.document})
	if err != nil {
		return Document{}, fail(Corrupt, PDF, fmt.Errorf("the pages cannot be counted: %w", err))
	}
	if count.PageCount > e.options.MaxPages {
		return Document{}, fail(TooLarge, PDF, fmt.Errorf("the PDF has %d pages, more than the %d allowed", count.PageCount, e.options.MaxPages))
	}

	pages := make([]pdfPage, 0, count.PageCount)
	for index := range count.PageCount {
		page, err := session.read(index)
		if err != nil {
			if ctx.Err() != nil {
				return Document{}, stack.Wrap(fmt.Errorf("extract: read the PDF: %w", ctx.Err()))
			}
			return Document{}, fail(Corrupt, PDF, fmt.Errorf("page %d cannot be read: %w", index+1, err))
		}
		pages = append(pages, page)
	}

	document := Document{Pages: layoutPages(pages)}
	document.render = func(ctx context.Context, number int) (Image, error) {
		return e.renderPDF(ctx, data, number)
	}
	return document, nil
}

func (e *Extractor) renderPDF(ctx context.Context, data []byte, number int) (Image, error) {
	ctx, cancel := context.WithTimeout(ctx, e.options.Timeout)
	defer cancel()
	session, err := e.open(ctx, data)
	if err != nil {
		return Image{}, err
	}
	defer session.close()

	drawn, err := session.instance.RenderPageInDPI(&requests.RenderPageInDPI{Page: session.page(number - 1), DPI: renderDPI})
	if err != nil {
		return Image{}, stack.Wrap(fmt.Errorf("extract: draw page %d: %w", number, err))
	}
	defer drawn.Cleanup()
	var b bytes.Buffer
	if err := jpeg.Encode(&b, drawn.Result.Image, &jpeg.Options{Quality: renderQuality}); err != nil {
		return Image{}, stack.Wrap(fmt.Errorf("extract: encode page %d: %w", number, err))
	}
	return Image{MIME: "image/jpeg", Data: b.Bytes()}, nil
}
