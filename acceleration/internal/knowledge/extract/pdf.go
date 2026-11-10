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
	// drawn is where the page's pictures and the shapes of its charts and diagrams are.
	drawn []box
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
	onPage := requests.Page{ByReference: &loaded.Page}
	walked := 0
	err = s.walk(&page, objects{
		count: func() (int, error) {
			counted, err := s.instance.FPDFPage_CountObjects(&requests.FPDFPage_CountObjects{Page: onPage})
			return counted.Count, err
		},
		at: func(i int) (references.FPDF_PAGEOBJECT, error) {
			object, err := s.instance.FPDFPage_GetObject(&requests.FPDFPage_GetObject{Page: onPage, Index: i})
			return object.PageObject, err
		},
	}, identity, 0, &walked)
	return page, err
}

// objects are the objects a page or a form holds.
type objects struct {
	count func() (int, error)
	at    func(int) (references.FPDF_PAGEOBJECT, error)
}

// walk counts the pictures among a page's objects, and keeps where on the page those that
// are part of a figure are: a picture, a shape with a curve or a slant in it, as a pie, a
// plotted line or an arrow has, and a filled box with no text in it, as a bar is. Boxes and
// lines with text in them are a table's, which reads as text already. A form's objects
// are where its matrix puts them.
func (s *pdfSession) walk(page *pdfPage, holder objects, onPage matrix, depth int, walked *int) error {
	count, err := holder.count()
	if err != nil {
		return err
	}
	for i := range count {
		if *walked >= maxPageObjects {
			return nil
		}
		*walked++
		object, err := holder.at(i)
		if err != nil {
			return err
		}
		kind, err := s.instance.FPDFPageObj_GetType(&requests.FPDFPageObj_GetType{PageObject: object})
		if err != nil {
			return err
		}
		switch kind.Type {
		case enums.FPDF_PAGEOBJ_IMAGE:
			page.images++
			bounds, err := s.bounds(object, onPage)
			if err != nil {
				return err
			}
			if page.sized(bounds) {
				page.drawn = append(page.drawn, bounds)
			}
		case enums.FPDF_PAGEOBJ_PATH, enums.FPDF_PAGEOBJ_SHADING:
			bounds, err := s.bounds(object, onPage)
			if err != nil {
				return err
			}
			if !page.sized(bounds) {
				continue
			}
			drawing, err := s.drawing(object, kind.Type, func() bool { return page.written(bounds) })
			if err != nil {
				return err
			}
			if drawing {
				page.drawn = append(page.drawn, bounds)
			}
		case enums.FPDF_PAGEOBJ_FORM:
			if depth >= maxFormDepth {
				continue
			}
			placed, err := s.instance.FPDFPageObj_GetMatrix(&requests.FPDFPageObj_GetMatrix{PageObject: object})
			if err != nil {
				return err
			}
			form := object
			err = s.walk(page, objects{
				count: func() (int, error) {
					counted, err := s.instance.FPDFFormObj_CountObjects(&requests.FPDFFormObj_CountObjects{PageObject: form})
					return counted.Count, err
				},
				at: func(i int) (references.FPDF_PAGEOBJECT, error) {
					child, err := s.instance.FPDFFormObj_GetObject(&requests.FPDFFormObj_GetObject{PageObject: form, Index: uint64(i)})
					return child.PageObject, err
				},
			}, matrixOf(placed.Matrix).then(onPage), depth+1, walked)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// bounds is where an object is on the page.
func (s *pdfSession) bounds(object references.FPDF_PAGEOBJECT, onPage matrix) (box, error) {
	bounds, err := s.instance.FPDFPageObj_GetBounds(&requests.FPDFPageObj_GetBounds{PageObject: object})
	if err != nil {
		return box{}, err
	}
	return onPage.box(box{
		left: float64(bounds.Left), bottom: float64(bounds.Bottom),
		right: float64(bounds.Right), top: float64(bounds.Top),
	}), nil
}

// drawing says a shape is part of a figure rather than a table's or a page's furniture:
// it has a curve or a slanted line in it, or it is filled and has no text in it. A shading
// is a fill.
func (s *pdfSession) drawing(object references.FPDF_PAGEOBJECT, kind enums.FPDF_PAGEOBJ, written func() bool) (bool, error) {
	if kind == enums.FPDF_PAGEOBJ_SHADING {
		return !written(), nil
	}
	curved, err := s.curved(object)
	if err != nil || curved {
		return curved, err
	}
	mode, err := s.instance.FPDFPath_GetDrawMode(&requests.FPDFPath_GetDrawMode{PageObject: object})
	if err != nil {
		return false, err
	}
	return mode.FillMode != enums.FPDF_FILLMODE_NONE && !written(), nil
}

// curved says a shape has a curve in it, or a line that is neither level nor upright,
// among its first maxPathPoints points. A table is drawn from level and upright lines. The
// points are in the shape's own coordinates, which a table is never turned in.
func (s *pdfSession) curved(object references.FPDF_PAGEOBJECT) (bool, error) {
	points, err := s.instance.FPDFPath_CountSegments(&requests.FPDFPath_CountSegments{PageObject: object})
	if err != nil {
		return false, err
	}
	var lastX, lastY float64
	for i := range min(points.Count, maxPathPoints) {
		segment, err := s.instance.FPDFPath_GetPathSegment(&requests.FPDFPath_GetPathSegment{PageObject: object, Index: i})
		if err != nil {
			return false, err
		}
		kind, err := s.instance.FPDFPathSegment_GetType(&requests.FPDFPathSegment_GetType{PathSegment: segment.PathSegment})
		if err != nil {
			return false, err
		}
		if kind.Type == enums.FPDF_SEGMENT_BEZIERTO {
			return true, nil
		}
		at, err := s.instance.FPDFPathSegment_GetPoint(&requests.FPDFPathSegment_GetPoint{PathSegment: segment.PathSegment})
		if err != nil {
			return false, err
		}
		x, y := float64(at.X), float64(at.Y)
		if kind.Type == enums.FPDF_SEGMENT_LINETO && slanted(x-lastX, y-lastY) {
			return true, nil
		}
		lastX, lastY = x, y
	}
	return false, nil
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
