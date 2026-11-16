package extract

import (
	"strings"
)

// writer builds pages of Markdown out of what a word processor's file says, in order:
// headings, paragraphs, list items and tables, and where its pages break.
type writer struct {
	// numbered says the file records where its pages break, so its pages are numbered.
	numbered bool

	done   []Page
	blocks []string
	// listing says the last block is a list, so the next item is added to it.
	listing bool
	// tables are the tables being read, innermost last: a table inside a cell is written
	// into that cell as text.
	tables [][][]string
	// pictures are the pictures placed so far, each written as its marker.
	pictures pictures
}

// paragraph writes a paragraph, or adds it to the cell being read.
func (w *writer) paragraph(text string) {
	text = oneLine(text)
	if text == "" {
		return
	}
	if len(w.tables) > 0 {
		w.appendCell(text)
		return
	}
	w.listing = false
	w.blocks = append(w.blocks, text)
}

// line writes a paragraph whose text says whether it is a list item, by the bullet or
// number it opens with, as a format that keeps lists as formatting writes it out as text.
func (w *writer) line(text string) {
	if mark, item, ok := listItem(oneLine(text)); ok {
		w.item(0, mark == "-", item)
		return
	}
	w.paragraph(text)
}

func (w *writer) heading(level int, text string) {
	text = oneLine(text)
	if text == "" {
		return
	}
	if len(w.tables) > 0 {
		w.appendCell(text)
		return
	}
	w.listing = false
	w.blocks = append(w.blocks, strings.Repeat("#", min(max(level, 1), 6))+" "+text)
}

// item writes a list item at a depth from 0, under the item before it when that was one.
func (w *writer) item(depth int, bulleted bool, text string) {
	text = oneLine(text)
	if text == "" {
		return
	}
	if len(w.tables) > 0 {
		w.appendCell(text)
		return
	}
	mark := "1."
	if bulleted {
		mark = "-"
	}
	item := strings.Repeat("  ", min(depth, 8)) + mark + " " + text
	if n := len(w.blocks); n > 0 && w.listing {
		w.blocks[n-1] += "\n" + item
		return
	}
	w.blocks = append(w.blocks, item)
	w.listing = true
}

// picture writes a picture's marker in the cell being read, or as a block of its own.
func (w *writer) picture(marker string) {
	if marker == "" {
		return
	}
	if len(w.tables) > 0 {
		w.appendCell(marker)
		return
	}
	w.listing = false
	w.blocks = append(w.blocks, marker)
}

func (w *writer) startTable() { w.tables = append(w.tables, nil) }

func (w *writer) startRow() {
	if n := len(w.tables); n > 0 {
		w.tables[n-1] = append(w.tables[n-1], nil)
	}
}

func (w *writer) startCell() {
	n := len(w.tables)
	if n == 0 {
		return
	}
	rows := w.tables[n-1]
	if len(rows) == 0 {
		rows = append(rows, nil)
		w.tables[n-1] = rows
	}
	rows[len(rows)-1] = append(rows[len(rows)-1], "")
}

func (w *writer) appendCell(text string) {
	rows := w.tables[len(w.tables)-1]
	if len(rows) == 0 {
		rows = append(rows, nil)
		w.tables[len(w.tables)-1] = rows
	}
	row := rows[len(rows)-1]
	if len(row) == 0 {
		rows[len(rows)-1] = append(row, text)
		return
	}
	row[len(row)-1] = strings.TrimSpace(row[len(row)-1] + " " + text)
}

// endTable writes the innermost table: as a table, or as text in the cell of the table
// it is inside.
func (w *writer) endTable() {
	n := len(w.tables)
	if n == 0 {
		return
	}
	rows := trimRows(w.tables[n-1])
	w.tables = w.tables[:n-1]
	if len(rows) == 0 {
		return
	}
	if len(w.tables) > 0 {
		for _, row := range rows {
			w.appendCell(strings.Join(row, " "))
		}
		return
	}
	w.listing = false
	w.blocks = append(w.blocks, table(rows))
}

// pageBreak ends the page, unless nothing is on it yet: a file can record where a page
// break was laid as well as the break that made it, and the two are one page. A break
// inside a table is left for the table to end, so a table is not cut in two.
func (w *writer) pageBreak() {
	if !w.numbered || len(w.tables) > 0 || len(w.blocks) == 0 {
		return
	}
	w.flushPage()
}

func (w *writer) flushPage() {
	for len(w.tables) > 0 {
		w.endTable()
	}
	w.done = append(w.done, Page{Text: strings.Join(w.blocks, "\n\n")})
	w.blocks = nil
	w.listing = false
}

// pages ends the last page and numbers them, when the file recorded its pages.
func (w *writer) pages() []Page {
	w.flushPage()
	if w.numbered {
		for i := range w.done {
			w.done[i].Number = i + 1
		}
	}
	return w.done
}

func oneLine(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

// table writes rows as a Markdown table, the first row as its header. Every row is padded
// to the widest, and columns empty in every row are dropped: a spreadsheet's formatting
// often reaches well past its data.
func table(rows [][]string) string {
	width := 0
	for _, row := range rows {
		width = max(width, len(row))
	}
	used := make([]bool, width)
	for _, row := range rows {
		for i, cell := range row {
			if strings.TrimSpace(cell) != "" {
				used[i] = true
			}
		}
	}

	var b strings.Builder
	write := func(cells []string) {
		b.WriteString("|")
		for i := range width {
			if !used[i] {
				continue
			}
			cell := ""
			if i < len(cells) {
				cell = cells[i]
			}
			b.WriteString(" " + tableCell(cell) + " |")
		}
		b.WriteString("\n")
	}
	write(rows[0])
	b.WriteString("|")
	for i := range width {
		if used[i] {
			b.WriteString(" --- |")
		}
	}
	b.WriteString("\n")
	for _, row := range rows[1:] {
		write(row)
	}
	return strings.TrimSuffix(b.String(), "\n")
}

// tableCell is a cell's text as one line, since a Markdown table row cannot break, with
// the pipes that would end the cell escaped.
func tableCell(text string) string {
	return strings.ReplaceAll(oneLine(text), "|", `\|`)
}

// trimRows drops the empty rows at the end of a table and the empty cells at the end of
// each row.
func trimRows(rows [][]string) [][]string {
	for i, row := range rows {
		end := len(row)
		for end > 0 && strings.TrimSpace(row[end-1]) == "" {
			end--
		}
		rows[i] = row[:end]
	}
	end := len(rows)
	for end > 0 && len(rows[end-1]) == 0 {
		end--
	}
	return rows[:end]
}
