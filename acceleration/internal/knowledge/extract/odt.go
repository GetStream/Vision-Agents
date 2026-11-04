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

// readODT reads an OpenDocument text: headings by their outline level, lists by their
// list styles, and tables as tables. Pages are numbered where the program that saved it
// recorded them, as LibreOffice does with a soft page break.
func readODT(data []byte) (Document, error) {
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return Document{}, fail(Corrupt, ODT, fmt.Errorf("the archive cannot be opened: %w", err))
	}
	manifest, err := zipFile(archive, "META-INF/manifest.xml")
	if err != nil {
		return Document{}, err
	}
	if bytes.Contains(manifest, []byte("encryption-data")) {
		return Document{}, fail(Encrypted, ODT, errors.New("the document is protected by a password"))
	}
	content, err := zipFile(archive, "content.xml")
	if err != nil {
		return Document{}, err
	}
	if content == nil {
		return Document{}, fail(Corrupt, ODT, errors.New("the document has no content.xml"))
	}
	stylesXML, err := zipFile(archive, "styles.xml")
	if err != nil {
		return Document{}, err
	}
	bullets := map[string]map[int]bool{}
	for _, part := range [][]byte{stylesXML, content} {
		if err := odtListStyles(part, bullets); err != nil {
			return Document{}, err
		}
	}

	reader := &odtReader{bullets: bullets}
	reader.numbered = bytes.Contains(content, []byte("soft-page-break"))
	if err := reader.read(content); err != nil {
		return Document{}, fail(Corrupt, ODT, err)
	}
	return Document{Pages: reader.pages()}, nil
}

// odtListStyles records, for each list style and level, whether its items are bulleted.
func odtListStyles(data []byte, bullets map[string]map[int]bool) error {
	if data == nil {
		return nil
	}
	decoder := xml.NewDecoder(bytes.NewReader(data))
	style := ""
	for {
		token, err := decoder.Token()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return fail(Corrupt, ODT, err)
		}
		element, ok := token.(xml.StartElement)
		if !ok {
			continue
		}
		switch element.Name.Local {
		case "list-style":
			style = attr(element, "name")
			bullets[style] = map[int]bool{}
		case "list-level-style-bullet", "list-level-style-number":
			if level, err := strconv.Atoi(attr(element, "level")); err == nil && style != "" {
				bullets[style][level-1] = element.Name.Local == "list-level-style-bullet"
			}
		}
	}
}

type odtReader struct {
	writer
	bullets map[string]map[int]bool

	text strings.Builder
	// lists are the list styles of the lists being read, outermost first. A list inside
	// another that names no style of its own takes its parent's.
	lists   []string
	heading int
	// paragraphs is how deep the reader is in paragraphs and headings, which can hold a
	// text box with paragraphs of its own.
	paragraphs int
}

func (o *odtReader) read(content []byte) error {
	decoder := xml.NewDecoder(bytes.NewReader(content))
	body := false
	// skip is how deep the reader is in something that is not text a reader sees: a
	// footnote, a comment, a change kept for review.
	skip := 0
	for {
		token, err := decoder.Token()
		if errors.Is(err, io.EOF) {
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
			if element.Name.Local == "body" {
				body = true
			}
			if !body {
				continue
			}
			switch element.Name.Local {
			case "note", "annotation", "tracked-changes":
				skip = 1
			case "h":
				o.flush()
				o.paragraphs++
				o.heading, _ = strconv.Atoi(attr(element, "outline-level"))
				o.heading = max(o.heading, 1)
			case "p":
				o.flush()
				o.paragraphs++
			case "s":
				count, err := strconv.Atoi(attr(element, "c"))
				if err != nil {
					count = 1
				}
				o.text.WriteString(strings.Repeat(" ", min(max(count, 1), 64)))
			case "tab", "line-break":
				o.text.WriteByte(' ')
			case "soft-page-break":
				o.flush()
				o.pageBreak()
			case "list":
				style := attr(element, "style-name")
				if style == "" && len(o.lists) > 0 {
					style = o.lists[len(o.lists)-1]
				}
				o.lists = append(o.lists, style)
			case "table":
				o.flush()
				o.startTable()
			case "table-row":
				o.startRow()
			case "table-cell", "covered-table-cell":
				o.startCell()
				// A run of identical cells is written once with a count, which an empty
				// spreadsheet row sets in the thousands, and a merged cell names how many
				// columns it covers. Both keep the next cell under its own header.
				repeat, _ := strconv.Atoi(attr(element, "number-columns-repeated"))
				spanned, _ := strconv.Atoi(attr(element, "number-columns-spanned"))
				for range min(max(repeat, spanned, 1), 64) - 1 {
					o.startCell()
				}
			}
		case xml.EndElement:
			if skip > 0 {
				skip--
				continue
			}
			if !body {
				continue
			}
			switch element.Name.Local {
			case "h", "p":
				o.flush()
				o.paragraphs--
			case "list":
				o.lists = o.lists[:len(o.lists)-1]
			case "table":
				o.flush()
				o.endTable()
			}
		case xml.CharData:
			if skip == 0 && body && o.paragraphs > 0 {
				o.text.Write(element)
			}
		}
	}
}

// flush writes the paragraph or heading read so far.
func (o *odtReader) flush() {
	text := o.text.String()
	o.text.Reset()
	heading := o.heading
	o.heading = 0
	switch {
	case heading > 0:
		o.writer.heading(heading, text)
	case len(o.lists) > 0:
		depth := len(o.lists) - 1
		o.item(depth, o.bullets[o.lists[depth]][depth], text)
	default:
		o.paragraph(text)
	}
}
