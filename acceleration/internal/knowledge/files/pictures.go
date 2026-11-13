package files

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/extract"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocrrouter"
)

// picturesAtOnce is how many pictures of one file are read at once.
const picturesAtOnce = 4

// PictureLimits bound how much of one file is read from its pictures.
type PictureLimits struct {
	// Scanned is the most scanned pages read. A file with more is refused with a
	// TooManyScansError, since its text is missing without them.
	Scanned int
	// Reads is the most pictures read in all: the scanned pages first, then the pages with
	// figures on them, then the pictures the file holds. The rest are left unread.
	Reads int
}

// TooManyScansError is a file with more scanned pages than its PictureLimits read.
type TooManyScansError struct {
	Scanned, Limit int
}

func (e *TooManyScansError) Error() string {
	return fmt.Sprintf("files: %d scanned pages, more than the %d read", e.Scanned, e.Limit)
}

// look is one picture a model is asked to read: a scanned page, a page with figures on it,
// or a picture an office file holds.
type look struct {
	// page is the page's place in the document's pages, and picture the picture's place
	// in its pictures, whichever the look is of; the other is -1.
	page, picture int
	mode          ocr.Mode
}

// ReadPictures reads a file's pictures with reader, a few at a time, and writes what they say
// into read: a scanned page's text in place of its empty one, a figured page's figures, and
// each picture's text. request is every read's, with its page left to fill.
//
// A scanned page that cannot be read fails the whole read, as does any failure fatal says
// is one (money refused, which reading on would only meet again). A figure that cannot be
// read is left out and logged: every model failed it, which is more likely the picture than
// an outage.
func ReadPictures(ctx context.Context, reader Reader, request ocrrouter.Request, read *extract.Document, limits PictureLimits,
	fatal func(error) bool, logger *slog.Logger,
) error {
	scanned := read.Scanned()
	if len(scanned) > limits.Scanned {
		return &TooManyScansError{Scanned: len(scanned), Limit: limits.Scanned}
	}

	at := make(map[int]int, len(read.Pages))
	for i, page := range read.Pages {
		at[page.Number] = i
	}
	var looks []look
	for _, number := range scanned {
		looks = append(looks, look{page: at[number], picture: -1, mode: ocr.TextAndFigures})
	}
	for _, number := range read.Figured() {
		looks = append(looks, look{page: at[number], picture: -1, mode: ocr.Figures})
	}
	for i := range read.Pictures {
		looks = append(looks, look{page: -1, picture: i, mode: ocr.Figures})
	}
	if len(looks) > limits.Reads {
		logger.Info("a file has more figures than are read, the rest left unsummarised",
			"figures", len(looks), "read", limits.Reads)
		looks = looks[:limits.Reads]
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var (
		mu    sync.Mutex
		first error
		wg    sync.WaitGroup
		slots = make(chan struct{}, picturesAtOnce)
	)
	for _, wanted := range looks {
		if ctx.Err() != nil {
			break
		}
		wg.Add(1)
		slots <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-slots }()
			text, err := readPicture(ctx, reader, request, read, wanted)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if wanted.mode != ocr.Figures || fatal(err) || ctx.Err() != nil {
					if first == nil {
						first = err
						cancel()
					}
					return
				}
				logger.Warn("a figure in a file could not be read, and is left out",
					"page", wanted.page, "picture", wanted.picture, "error", err)
				return
			}
			switch {
			case wanted.picture >= 0:
				read.Pictures[wanted.picture].Text = text
			case wanted.mode == ocr.Figures:
				read.Pages[wanted.page].Figures = unread(text, read.Pages[wanted.page].Text)
			default:
				read.Pages[wanted.page].Text = text
				read.Pages[wanted.page].Scanned = false
			}
		}()
	}
	wg.Wait()
	return first
}

func readPicture(ctx context.Context, reader Reader, request ocrrouter.Request, read *extract.Document, wanted look) (string, error) {
	var picture extract.Image
	what := ""
	if wanted.picture >= 0 {
		picture = extract.Image{MIME: read.Pictures[wanted.picture].MIME, Data: read.Pictures[wanted.picture].Data}
		what = fmt.Sprintf("picture %d", wanted.picture+1)
	} else {
		number := read.Pages[wanted.page].Number
		drawn, err := read.Render(ctx, number)
		if err != nil {
			return "", err
		}
		picture = drawn
		what = fmt.Sprintf("page %d", number)
	}
	page := ocr.Request{Image: picture.Data, MediaType: picture.MIME, Mode: wanted.mode}
	if err := page.Validate(); err != nil {
		return "", fmt.Errorf("files: %s cannot be read: %w", what, err)
	}
	request.Page = page
	reading, err := reader.Read(ctx, request)
	if err != nil {
		return "", fmt.Errorf("files: read %s from its picture: %w", what, err)
	}
	return reading.Markdown, nil
}
