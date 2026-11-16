package extract

import (
	"errors"
	"strconv"
	"strings"
	"unicode/utf16"

	"golang.org/x/text/encoding/charmap"
)

// skipped are the RTF destinations that hold no text a reader sees: tables of fonts,
// colours and styles, metadata, pictures, field codes, headers and footers, footnotes.
var skipped = map[string]bool{
	"fonttbl": true, "colortbl": true, "stylesheet": true, "info": true, "pict": true,
	"object": true, "fldinst": true, "header": true, "headerl": true, "headerr": true,
	"headerf": true, "footer": true, "footerl": true, "footerr": true, "footerf": true,
	"footnote": true, "annotation": true, "themedata": true, "colorschememapping": true,
	"latentstyles": true, "datastore": true, "xmlnstbl": true, "listtable": true,
	"listoverridetable": true, "rsidtbl": true, "generator": true, "mmathPr": true,
	"filetbl": true, "revtbl": true, "userprops": true, "shp": true, "nonshppict": true,
	"bkmkstart": true, "bkmkend": true, "pgdsctbl": true, "xmlopen": true, "atnid": true,
	"atnauthor": true, "comment": true, "template": true, "background": true,
}

// rtfState is what a group carries: RTF scopes formatting to the group it is set in.
type rtfState struct {
	skip    bool
	uc      int
	style   int
	outline int
	indent  int
	intbl   bool
}

// readRTF reads a Rich Text Format document. RTF records no pages, only the breaks
// somebody put in, so its pages are not numbered.
func readRTF(data []byte) (Document, error) {
	headings, err := rtfHeadingStyles(data)
	if err != nil {
		return Document{}, err
	}
	r := &rtfReader{headings: headings, decoder: charmap.Windows1252, state: rtfState{uc: 1}}
	if err := r.read(data); err != nil {
		return Document{}, fail(Corrupt, RTF, err)
	}
	return Document{Pages: r.pages()}, nil
}

type rtfReader struct {
	writer
	headings map[int]int
	decoder  *charmap.Charmap

	text  strings.Builder
	cell  strings.Builder
	row   []string
	rows  [][]string
	state rtfState
	stack []rtfState
	// skipNext is how many characters after a \u are what a reader without Unicode shows
	// instead, which \uc sets.
	skipNext int
}

// rtfToken is one piece of RTF: a control word with its parameter, a control symbol, a
// group's opening or closing brace, or plain text.
type rtfToken struct {
	word      string
	parameter int
	hasParam  bool
	text      string
	open      bool
	close     bool
}

// rtfTokens cuts RTF into tokens.
func rtfTokens(data []byte, each func(rtfToken) error) error {
	for i := 0; i < len(data); {
		c := data[i]
		switch c {
		case '{':
			i++
			if err := each(rtfToken{open: true}); err != nil {
				return err
			}
		case '}':
			i++
			if err := each(rtfToken{close: true}); err != nil {
				return err
			}
		case '\r', '\n':
			i++
		case '\\':
			i++
			if i >= len(data) {
				return nil
			}
			if !isLetter(data[i]) {
				symbol := data[i]
				i++
				if symbol == '\'' && i+2 <= len(data) {
					if value, err := strconv.ParseUint(string(data[i:i+2]), 16, 8); err == nil {
						if err := each(rtfToken{word: "'", parameter: int(value), hasParam: true}); err != nil {
							return err
						}
					}
					i += 2
					continue
				}
				if err := each(rtfToken{word: string(symbol)}); err != nil {
					return err
				}
				continue
			}
			start := i
			for i < len(data) && isLetter(data[i]) {
				i++
			}
			token := rtfToken{word: string(data[start:i])}
			numberStart := i
			if i < len(data) && data[i] == '-' {
				i++
			}
			for i < len(data) && data[i] >= '0' && data[i] <= '9' {
				i++
			}
			if i > numberStart && !(i == numberStart+1 && data[numberStart] == '-') {
				token.parameter, _ = strconv.Atoi(string(data[numberStart:i]))
				token.hasParam = true
			} else {
				i = numberStart
			}
			if i < len(data) && data[i] == ' ' {
				i++
			}
			if token.word == "bin" && token.parameter > 0 {
				i = min(len(data), i+token.parameter)
				continue
			}
			if err := each(token); err != nil {
				return err
			}
		default:
			start := i
			for i < len(data) && data[i] != '\\' && data[i] != '{' && data[i] != '}' && data[i] != '\r' && data[i] != '\n' {
				i++
			}
			if err := each(rtfToken{text: string(data[start:i])}); err != nil {
				return err
			}
		}
	}
	return nil
}

func isLetter(c byte) bool { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') }

// rtfHeadingStyles reads the stylesheet for the styles named as headings, by number.
func rtfHeadingStyles(data []byte) (map[int]int, error) {
	headings := map[int]int{}
	depth, sheet := 0, 0
	style := -1
	var name strings.Builder
	err := rtfTokens(data, func(t rtfToken) error {
		switch {
		case t.open:
			depth++
			if sheet > 0 && depth == sheet+1 {
				style, name = 0, strings.Builder{}
			}
		case t.close:
			if sheet > 0 && depth == sheet+1 && style >= 0 {
				label := strings.ToLower(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(name.String()), ";")))
				if level, ok := strings.CutPrefix(label, "heading "); ok {
					if n, err := strconv.Atoi(level); err == nil && n > 0 {
						headings[style] = n
					}
				}
				style = -1
			}
			if depth == sheet {
				sheet = 0
			}
			depth--
		case t.word == "stylesheet":
			sheet = depth
		case sheet > 0 && t.word == "s" && t.hasParam:
			style = t.parameter
		case sheet > 0 && t.text != "" && depth == sheet+1:
			name.WriteString(t.text)
		}
		return nil
	})
	if depth < 0 {
		return nil, fail(Corrupt, RTF, errors.New("a group is closed that was never opened"))
	}
	return headings, err
}

func (r *rtfReader) read(data []byte) error {
	// ignorable says the group just opened with \*, so a destination it names that is not
	// known here is skipped rather than read as text.
	ignorable := false
	err := rtfTokens(data, func(t rtfToken) error {
		switch {
		case t.open:
			r.stack = append(r.stack, r.state)
			ignorable = false
			return nil
		case t.close:
			if len(r.stack) == 0 {
				return errors.New("a group is closed that was never opened")
			}
			r.state = r.stack[len(r.stack)-1]
			r.stack = r.stack[:len(r.stack)-1]
			return nil
		case t.word == "*":
			ignorable = true
			return nil
		}
		if r.state.skip {
			return nil
		}
		if t.text != "" {
			ignorable = false
			text := t.text
			if r.skipNext > 0 {
				n := min(r.skipNext, len(text))
				text, r.skipNext = text[n:], r.skipNext-n
			}
			r.write(text)
			return nil
		}
		if skipped[t.word] || (ignorable && t.word != "fldrslt") {
			r.state.skip = true
			ignorable = false
			return nil
		}
		ignorable = false
		r.control(t)
		return nil
	})
	if err != nil {
		return err
	}
	r.endParagraph()
	r.flushTable()
	return nil
}

func (r *rtfReader) write(text string) {
	if r.state.intbl {
		r.cell.WriteString(text)
		return
	}
	r.text.WriteString(text)
}

func (r *rtfReader) control(t rtfToken) {
	switch t.word {
	case "ansicpg":
		if decoder := codePage(t.parameter); decoder != nil {
			r.decoder = decoder
		}
	case "uc":
		r.state.uc = max(t.parameter, 0)
	case "'":
		if r.skipNext > 0 {
			r.skipNext--
			return
		}
		r.write(string(r.decoder.DecodeByte(byte(t.parameter))))
	case "u":
		code := t.parameter
		if code < 0 {
			code += 65536
		}
		r.write(string(utf16.Decode([]uint16{uint16(code)})))
		r.skipNext = r.state.uc
	case "par", "sect":
		r.endParagraph()
	case "line", "tab", "~":
		r.write(" ")
	case "_":
		r.write("-")
	case "bullet":
		r.write("•")
	case "endash":
		r.write("–")
	case "emdash":
		r.write("—")
	case "lquote", "rquote":
		r.write("'")
	case "ldblquote", "rdblquote":
		r.write(`"`)
	case "\\", "{", "}":
		r.write(t.word)
	case "pard":
		r.state.style, r.state.outline, r.state.indent, r.state.intbl = 0, 0, 0, false
	case "s":
		r.state.style = t.parameter
	case "outlinelevel":
		r.state.outline = t.parameter + 1
	case "li":
		r.state.indent = t.parameter
	case "intbl":
		r.state.intbl = true
	case "cell":
		r.row = append(r.row, oneLine(r.cell.String()))
		r.cell.Reset()
	case "row":
		if len(r.row) > 0 {
			r.rows = append(r.rows, r.row)
		}
		r.row = nil
	}
}

// endParagraph writes the paragraph read so far: into the cell it is in, or as a heading,
// a list item or a paragraph.
func (r *rtfReader) endParagraph() {
	if r.state.intbl {
		r.cell.WriteByte(' ')
		return
	}
	r.flushTable()
	text := oneLine(r.text.String())
	r.text.Reset()
	if text == "" {
		return
	}
	level := r.state.outline
	if level == 0 {
		level = r.headings[r.state.style]
	}
	if level > 0 {
		r.heading(level, text)
		return
	}
	if mark, item, ok := listItem(text); ok {
		// An item is indented a quarter inch, 360 twips, further for each level it is
		// nested.
		r.item(max(r.state.indent/360-1, 0), mark == "-", item)
		return
	}
	r.paragraph(text)
}

// flushTable writes the table read so far, when one is.
func (r *rtfReader) flushTable() {
	if len(r.row) > 0 {
		r.rows = append(r.rows, r.row)
		r.row = nil
	}
	if len(r.rows) == 0 {
		return
	}
	r.startTable()
	for _, row := range r.rows {
		r.startRow()
		for _, cell := range row {
			r.startCell()
			r.appendCell(cell)
		}
	}
	r.endTable()
	r.rows = nil
}

// codePage is the decoder for a Windows or DOS code page RTF names in \ansicpg.
func codePage(number int) *charmap.Charmap {
	switch number {
	case 437:
		return charmap.CodePage437
	case 850:
		return charmap.CodePage850
	case 1250:
		return charmap.Windows1250
	case 1251:
		return charmap.Windows1251
	case 1252:
		return charmap.Windows1252
	case 1253:
		return charmap.Windows1253
	case 1254:
		return charmap.Windows1254
	case 1255:
		return charmap.Windows1255
	case 1256:
		return charmap.Windows1256
	case 1257:
		return charmap.Windows1257
	case 1258:
		return charmap.Windows1258
	case 10000:
		return charmap.Macintosh
	}
	return nil
}
