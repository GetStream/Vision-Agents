package extract

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode/utf16"

	"github.com/richardlehane/mscfb"
	"golang.org/x/text/encoding/charmap"
)

// detectOLE reads what an OLE compound file holds. Word 97-2003 keeps its text in a
// WordDocument stream. A Word, Excel or PowerPoint file saved with a password is an OLE
// file whatever its format, holding the real file encrypted in an EncryptedPackage
// stream.
func detectOLE(data []byte) (Format, error) {
	streams, err := oleStreams(data, nil)
	if err != nil {
		return "", err
	}
	switch {
	case streams["EncryptedPackage"] || streams["EncryptionInfo"]:
		return "", fail(Encrypted, "", errors.New("the file is protected by a password"))
	case streams["WordDocument"]:
		return DOC, nil
	case streams["Workbook"] || streams["Book"]:
		return "", fail(Unsupported, "", errors.New("Excel 97-2003 workbooks cannot be read, save it as .xlsx"))
	case streams["PowerPoint Document"]:
		return "", fail(Unsupported, "", errors.New("PowerPoint presentations cannot be read yet"))
	}
	return "", fail(Unsupported, "", errors.New("an OLE file that is not a Word document"))
}

// oleStreams lists the streams of an OLE file, and reads the ones named in read.
func oleStreams(data []byte, read map[string][]byte) (map[string]bool, error) {
	file, err := mscfb.New(bytes.NewReader(data))
	if err != nil {
		return nil, fail(Corrupt, "", fmt.Errorf("the OLE file cannot be opened: %w", err))
	}
	names := map[string]bool{}
	for entry, err := file.Next(); err == nil; entry, err = file.Next() {
		names[entry.Name] = true
		if _, wanted := read[entry.Name]; wanted && read[entry.Name] == nil {
			content, err := io.ReadAll(io.LimitReader(entry, int64(len(data))+1))
			if err != nil {
				return nil, fail(Corrupt, DOC, fmt.Errorf("stream %s cannot be read: %w", entry.Name, err))
			}
			read[entry.Name] = content
		}
	}
	return names, nil
}

// The parts of a Word 97-2003 file that say where its text is, by their offsets in the
// File Information Block ([MS-DOC] 2.5.1).
const (
	fibFlags      = 0x0A
	fibEncrypted  = 0x0100
	fibTable1     = 0x0200
	fibObfuscated = 0x8000
	fibCsw        = 0x20
	// clxIndex is the place of fcClx among the FibRgFcLcb97 pairs.
	clxIndex = 33
)

// readDOC reads a Word 97-2003 document's text from its piece table ([MS-DOC] 2.9.177):
// paragraphs, and tables row by row. The file keeps headings and lists as formatting
// rather than text, and records no pages, so neither is read.
func readDOC(data []byte) (Document, error) {
	streams := map[string][]byte{"WordDocument": nil, "0Table": nil, "1Table": nil}
	if _, err := oleStreams(data, streams); err != nil {
		return Document{}, err
	}
	word := streams["WordDocument"]
	if len(word) < 0x22 || binary.LittleEndian.Uint16(word) != 0xA5EC {
		return Document{}, fail(Corrupt, DOC, errors.New("the WordDocument stream has no File Information Block"))
	}
	flags := binary.LittleEndian.Uint16(word[fibFlags:])
	if flags&(fibEncrypted|fibObfuscated) != 0 {
		return Document{}, fail(Encrypted, DOC, errors.New("the document is protected by a password"))
	}
	tableStream := streams["0Table"]
	if flags&fibTable1 != 0 {
		tableStream = streams["1Table"]
	}

	text, err := docText(word, tableStream)
	if err != nil {
		return Document{}, fail(Corrupt, DOC, err)
	}
	return Document{Pages: []Page{{Text: docMarkdown(text)}}}, nil
}

// docText reads the main document's characters out of the pieces they are stored in.
func docText(word, tableStream []byte) ([]rune, error) {
	at := fibCsw
	read16 := func() (int, error) {
		if at+2 > len(word) {
			return 0, errors.New("the File Information Block is cut short")
		}
		value := int(binary.LittleEndian.Uint16(word[at:]))
		at += 2
		return value, nil
	}
	csw, err := read16()
	if err != nil {
		return nil, err
	}
	at += csw * 2
	cslw, err := read16()
	if err != nil {
		return nil, err
	}
	lw := at
	at += cslw * 4
	if cslw < 4 || at > len(word) {
		return nil, errors.New("the File Information Block is cut short")
	}
	// ccpText is how many characters the main document has; footnotes, headers and the
	// rest follow it in the same pieces.
	ccpText := int(binary.LittleEndian.Uint32(word[lw+12:]))
	if _, err := read16(); err != nil {
		return nil, err
	}
	pair := at + clxIndex*8
	if pair+8 > len(word) {
		return nil, errors.New("the File Information Block has no piece table")
	}
	fcClx := int(binary.LittleEndian.Uint32(word[pair:]))
	lcbClx := int(binary.LittleEndian.Uint32(word[pair+4:]))
	if lcbClx <= 0 || fcClx < 0 || fcClx+lcbClx > len(tableStream) {
		return nil, errors.New("the piece table is outside the table stream")
	}
	clx := tableStream[fcClx : fcClx+lcbClx]

	// The piece table is after any number of property runs, each opened by 0x01.
	for len(clx) > 0 && clx[0] == 0x01 {
		if len(clx) < 3 {
			return nil, errors.New("a property run is cut short")
		}
		size := int(int16(binary.LittleEndian.Uint16(clx[1:])))
		if size < 0 || 3+size > len(clx) {
			return nil, errors.New("a property run is cut short")
		}
		clx = clx[3+size:]
	}
	if len(clx) < 5 || clx[0] != 0x02 {
		return nil, errors.New("the piece table is missing")
	}
	size := int(binary.LittleEndian.Uint32(clx[1:]))
	plc := clx[5:]
	if size > len(plc) || (size-4)%12 != 0 {
		return nil, errors.New("the piece table is cut short")
	}
	pieces := (size - 4) / 12
	descriptors := plc[(pieces+1)*4:]

	var text []rune
	for i := range pieces {
		start := int(binary.LittleEndian.Uint32(plc[i*4:]))
		end := min(int(binary.LittleEndian.Uint32(plc[(i+1)*4:])), ccpText)
		if start >= end {
			continue
		}
		fc := binary.LittleEndian.Uint32(descriptors[i*8+2:])
		count := end - start
		if fc&0x40000000 != 0 {
			// A compressed piece is one byte a character, in Windows-1252.
			offset := int(fc&0x3FFFFFFF) / 2
			if offset+count > len(word) {
				return nil, errors.New("a piece is outside the document")
			}
			for _, b := range word[offset : offset+count] {
				text = append(text, charmap.Windows1252.DecodeByte(b))
			}
			continue
		}
		offset := int(fc)
		if offset+count*2 > len(word) {
			return nil, errors.New("a piece is outside the document")
		}
		units := make([]uint16, count)
		for j := range units {
			units[j] = binary.LittleEndian.Uint16(word[offset+j*2:])
		}
		text = append(text, utf16.Decode(units)...)
	}
	return text, nil
}

// docPiece is a run of a document's text up to the mark that ends it: a paragraph mark,
// or the 0x07 that ends a table cell or row.
type docPiece struct {
	text string
	cell bool
}

// docPieces cuts a document's characters at its paragraph and cell marks. A field is its
// code and then what it shows, and only what it shows is kept.
func docPieces(text []rune) []docPiece {
	var pieces []docPiece
	var b strings.Builder
	var fields []bool
	for _, r := range text {
		if n := len(fields); n > 0 && fields[n-1] && r != 0x13 && r != 0x14 && r != 0x15 {
			continue
		}
		switch r {
		case 0x13:
			fields = append(fields, true)
		case 0x14:
			if n := len(fields); n > 0 {
				fields[n-1] = false
			}
		case 0x15:
			if n := len(fields); n > 0 {
				fields = fields[:n-1]
			}
		case '\r', 0x07:
			pieces = append(pieces, docPiece{text: oneLine(b.String()), cell: r == 0x07})
			b.Reset()
		case 0x0B, 0x0C, '\t', 0xA0:
			b.WriteByte(' ')
		case 0x1E:
			b.WriteByte('-')
		case 0x01, 0x02, 0x05, 0x08, 0x1F:
		default:
			if r >= ' ' {
				b.WriteRune(r)
			}
		}
	}
	if b.Len() > 0 {
		pieces = append(pieces, docPiece{text: oneLine(b.String())})
	}
	return pieces
}

// docMarkdown writes a document's pieces as paragraphs and tables.
//
// Word ends each table cell with 0x07 and each row with another, so a row is its cells
// and then a mark of its own; the first row, read to its double mark, says how many
// cells a row has. Which paragraphs are in a table is formatting the text does not
// carry, so a paragraph between rows is taken as the start of the next cell when the
// piece after it ends one, and as the end of the table when it does not.
func docMarkdown(text []rune) string {
	pieces := docPieces(text)
	var w writer
	var row []string
	var rows [][]string
	var prefix []string
	columns := 0

	flushTable := func() {
		if len(row) > 0 {
			rows = append(rows, row)
			row = nil
		}
		if len(rows) > 0 {
			w.startTable()
			for _, cells := range rows {
				w.startRow()
				for _, cell := range cells {
					w.startCell()
					w.appendCell(cell)
				}
			}
			w.endTable()
		}
		rows, columns = nil, 0
	}

	for i, piece := range pieces {
		inTable := len(row) > 0 || len(rows) > 0
		if !piece.cell {
			switch {
			case len(row) > 0:
				prefix = append(prefix, piece.text)
			case inTable && i+1 < len(pieces) && pieces[i+1].cell:
				prefix = append(prefix, piece.text)
			default:
				flushTable()
				for _, held := range prefix {
					w.line(held)
				}
				prefix = nil
				w.line(piece.text)
			}
			continue
		}
		cell := strings.TrimSpace(strings.Join(append(prefix, piece.text), " "))
		prefix = nil
		if (columns > 0 && len(row) == columns) || (columns == 0 && cell == "" && len(row) > 0) {
			if columns == 0 {
				columns = len(row)
			}
			rows = append(rows, row)
			row = nil
			continue
		}
		row = append(row, cell)
	}
	flushTable()
	for _, held := range prefix {
		w.line(held)
	}
	return w.pages()[0].Text
}
