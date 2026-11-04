package extract

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// wordStyle is what a Word paragraph style says about the paragraphs set in it.
type wordStyle struct {
	name    string
	basedOn string
	// outline is the heading level the style sets, from 1, or 0 for body text.
	outline int
	numID   string
	ilvl    int
}

// readDOCX reads a Word document: its headings by their styles, its lists by their
// numbering, and its tables as tables.
//
// Pages are numbered where Word recorded them. Word writes where it last laid each page
// break, and a file saved by anything else has none; counting only the breaks somebody
// put in by hand would number the pages wrong, so such a file is one page with no number.
func readDOCX(data []byte) (Document, error) {
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return Document{}, fail(Corrupt, DOCX, fmt.Errorf("the archive cannot be opened: %w", err))
	}
	body, err := zipFile(archive, "word/document.xml")
	if err != nil {
		return Document{}, err
	}
	stylesXML, err := zipFile(archive, "word/styles.xml")
	if err != nil {
		return Document{}, err
	}
	numberingXML, err := zipFile(archive, "word/numbering.xml")
	if err != nil {
		return Document{}, err
	}
	styles, err := wordStyles(stylesXML)
	if err != nil {
		return Document{}, err
	}
	bullets, err := wordBullets(numberingXML)
	if err != nil {
		return Document{}, err
	}
	reader := &wordReader{styles: styles, bullets: bullets}
	reader.numbered = bytes.Contains(body, []byte("lastRenderedPageBreak"))
	if err := reader.read(body); err != nil {
		return Document{}, fail(Corrupt, DOCX, err)
	}
	return Document{Pages: reader.pages()}, nil
}

func wordStyles(data []byte) (map[string]wordStyle, error) {
	styles := map[string]wordStyle{}
	if data == nil {
		return styles, nil
	}
	decoder := xml.NewDecoder(bytes.NewReader(data))
	var id string
	var current wordStyle
	for {
		token, err := decoder.Token()
		if errors.Is(err, io.EOF) {
			return styles, nil
		}
		if err != nil {
			return nil, fail(Corrupt, DOCX, fmt.Errorf("styles.xml: %w", err))
		}
		switch element := token.(type) {
		case xml.StartElement:
			switch element.Name.Local {
			case "style":
				id, current = attr(element, "styleId"), wordStyle{}
			case "name":
				current.name = strings.ToLower(attr(element, "val"))
			case "basedOn":
				current.basedOn = attr(element, "val")
			case "outlineLvl":
				if level, err := strconv.Atoi(attr(element, "val")); err == nil && level < 9 {
					current.outline = level + 1
				}
			case "numId":
				current.numID = attr(element, "val")
			case "ilvl":
				current.ilvl, _ = strconv.Atoi(attr(element, "val"))
			}
		case xml.EndElement:
			if element.Name.Local == "style" && id != "" {
				styles[id] = current
				id = ""
			}
		}
	}
}

// wordBullets says, for each list and level, whether its items are bulleted rather than
// numbered.
func wordBullets(data []byte) (map[string]map[int]bool, error) {
	out := map[string]map[int]bool{}
	if data == nil {
		return out, nil
	}
	decoder := xml.NewDecoder(bytes.NewReader(data))
	abstract := map[string]map[int]bool{}
	lists := map[string]string{}
	var abstractID, numID string
	level := 0
	for {
		token, err := decoder.Token()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fail(Corrupt, DOCX, fmt.Errorf("numbering.xml: %w", err))
		}
		element, ok := token.(xml.StartElement)
		if !ok {
			continue
		}
		switch element.Name.Local {
		case "abstractNum":
			abstractID, numID = attr(element, "abstractNumId"), ""
			abstract[abstractID] = map[int]bool{}
		case "lvl":
			level, _ = strconv.Atoi(attr(element, "ilvl"))
		case "numFmt":
			if abstractID != "" && numID == "" {
				abstract[abstractID][level] = attr(element, "val") == "bullet"
			}
		case "num":
			numID = attr(element, "numId")
		case "abstractNumId":
			if numID != "" {
				lists[numID] = attr(element, "val")
			}
		}
	}
	for id, abstractID := range lists {
		out[id] = abstract[abstractID]
	}
	return out, nil
}

func attr(element xml.StartElement, name string) string {
	for _, a := range element.Attr {
		if a.Name.Local == name {
			return a.Value
		}
	}
	return ""
}

// wordReader walks a document's body in order.
type wordReader struct {
	writer
	styles  map[string]wordStyle
	bullets map[string]map[int]bool

	text    strings.Builder
	style   string
	numID   string
	ilvl    int
	outline int
	// span is how many grid columns the cell being read covers.
	span int
}

func (w *wordReader) read(body []byte) error {
	decoder := xml.NewDecoder(bytes.NewReader(body))
	// skip is how deep the reader is in something that is not text a reader sees: a
	// deletion kept for review, a field's code, a footnote's number.
	skip := 0
	for {
		token, err := decoder.Token()
		if errors.Is(err, io.EOF) {
			w.flushParagraph()
			return nil
		}
		if err != nil {
			return err
		}
		switch element := token.(type) {
		case xml.StartElement:
			if skip > 0 {
				skip++
				continue
			}
			switch element.Name.Local {
			case "del", "instrText", "delInstrText", "footnoteReference", "endnoteReference":
				skip = 1
			case "p":
				w.style, w.numID, w.ilvl, w.outline = "", "", 0, 0
			case "pStyle":
				w.style = attr(element, "val")
			case "numId":
				w.numID = attr(element, "val")
			case "ilvl":
				w.ilvl, _ = strconv.Atoi(attr(element, "val"))
			case "outlineLvl":
				if level, err := strconv.Atoi(attr(element, "val")); err == nil && level < 9 {
					w.outline = level + 1
				}
			case "tab":
				w.text.WriteByte(' ')
			case "br", "cr":
				if attr(element, "type") == "page" {
					w.flushParagraph()
					w.pageBreak()
				} else {
					w.text.WriteByte(' ')
				}
			case "lastRenderedPageBreak":
				w.flushParagraph()
				w.pageBreak()
			case "noBreakHyphen":
				w.text.WriteByte('-')
			case "tbl":
				w.flushParagraph()
				w.startTable()
			case "tr":
				w.startRow()
			case "tc":
				w.startCell()
				w.span = 1
			case "gridSpan":
				if span, err := strconv.Atoi(attr(element, "val")); err == nil {
					w.span = min(max(span, 1), 64)
				}
			}
		case xml.EndElement:
			if skip > 0 {
				skip--
				continue
			}
			switch element.Name.Local {
			case "p":
				w.flushParagraph()
			case "tc":
				w.flushParagraph()
				// A merged cell is its text and then the columns it covers, so the cells
				// after it stay under their own headers.
				for range w.span - 1 {
					w.startCell()
				}
				w.span = 1
			case "tbl":
				w.flushParagraph()
				w.endTable()
			}
		case xml.CharData:
			if skip == 0 {
				w.text.Write(element)
			}
		}
	}
}

// flushParagraph writes the paragraph read so far as a heading, a list item or a
// paragraph, by its style.
func (w *wordReader) flushParagraph() {
	text := w.text.String()
	w.text.Reset()
	if level := w.headingLevel(); level > 0 {
		w.heading(level, text)
		return
	}
	if numID, ilvl := w.list(); numID != "" && numID != "0" {
		w.item(ilvl, w.bullets[numID][ilvl], text)
		return
	}
	w.paragraph(text)
}

func (w *wordReader) headingLevel() int {
	if w.outline > 0 {
		return w.outline
	}
	for id, hops := w.style, 0; id != "" && hops < 16; hops++ {
		style := w.styles[id]
		if style.outline > 0 {
			return style.outline
		}
		switch {
		case style.name == "title":
			return 1
		case style.name == "subtitle":
			return 2
		case strings.HasPrefix(style.name, "heading "):
			if level, err := strconv.Atoi(strings.TrimPrefix(style.name, "heading ")); err == nil && level > 0 {
				return level
			}
		}
		id = style.basedOn
	}
	return 0
}

// list is the list a paragraph is an item of, named in the paragraph or in its style.
func (w *wordReader) list() (string, int) {
	if w.numID != "" {
		return w.numID, w.ilvl
	}
	for id, hops := w.style, 0; id != "" && hops < 16; hops++ {
		style := w.styles[id]
		if style.numID != "" {
			return style.numID, style.ilvl
		}
		id = style.basedOn
	}
	return "", 0
}
