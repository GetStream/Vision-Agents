package extract

import (
	"cmp"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// A PDF says where each piece of text is drawn and nothing about what it is: no paragraphs,
// headings, lists or tables, and an order that is whatever the program that wrote it chose.
// What follows rebuilds them from where the text sits and how large it is set, which is
// how a reader tells them apart too.
//
// The page is cut the way a reader's eye cuts it (the recursive XY-cut of Nagy and Seth,
// 1984): first at a gutter, a band of white running the whole height of the text, which
// is what divides columns; failing one, at the white between blocks. Blocks either side
// of a cut that share a gutter are put back together, so a column that pauses between
// paragraphs is still read to its foot before the next column, and a table's rows stay
// one table.

// scannedChars is how few characters a page with a picture on it can have and still be a
// scan: a stamp, a page number or a line of form fields is not a page of text.
const scannedChars = 20

// gutterWidth is how many sizes of white divide two columns. The gutter between columns of
// prose is one to two sizes; the space between words is a quarter of one.
const gutterWidth = 0.8

// blockGap is how many sizes of white divide two blocks. Lines set solid have about a
// fifth of a size between them; a paragraph or a table row set apart has more.
const blockGap = 0.6

// paragraphGap is how many sizes apart the middles of two lines are before they are
// separate paragraphs. Set solid, a line sits about 1.2 sizes under the one above.
const paragraphGap = 1.6

// headingScale is how much larger than the body a line is set for it to be a heading.
const headingScale = 1.15

// cellLength is how many characters long a table's cells are at most, in the middle. A
// line of prose in a column is forty or more; a table's cells are a word or a figure.
const cellLength = 30

// margin is the share of the page at its top and at its bottom that running heads and page
// numbers sit in.
const margin = 0.1

// line is the runs on one line of a page, left to right.
type line struct {
	runs   []run
	top    float64
	bottom float64
	size   float64
	bold   bool
}

func (l line) center() float64 { return (l.top + l.bottom) / 2 }

func (l line) left() float64 { return l.runs[0].left }

func (l line) right() float64 { return l.runs[len(l.runs)-1].right }

func (l line) text() string { return joinRuns(l.runs) }

// layoutPages turns what each page holds into Markdown pages.
func layoutPages(pages []pdfPage) []Page {
	style := styleOf(pages)
	lined := make([][]line, len(pages))
	for i, page := range pages {
		lined[i] = lines(page.runs)
	}
	running := runningLines(pages, lined, style)

	out := make([]Page, 0, len(pages))
	for i, page := range pages {
		var kept []run
		for _, l := range lined[i] {
			if !running[runningKey(page, l, style)] && !pageNumber(page, l) {
				kept = append(kept, l.runs...)
			}
		}
		out = append(out, Page{
			Number: i + 1,
			Text:   style.render(style.cut(kept)),
			Scanned: (page.images > 0 && page.chars < scannedChars) ||
				(page.unreadable > 0 && page.unreadable >= page.chars),
		})
	}
	return out
}

// lines groups runs into lines, top to bottom.
func lines(runs []run) []line {
	sorted := slices.Clone(runs)
	slices.SortStableFunc(sorted, func(a, b run) int {
		return cmp.Compare((b.top+b.bottom)/2, (a.top+a.bottom)/2)
	})

	var grouped [][]run
	for _, r := range sorted {
		if n := len(grouped); n > 0 && sameLine(grouped[n-1], r) {
			grouped[n-1] = append(grouped[n-1], r)
			continue
		}
		grouped = append(grouped, []run{r})
	}

	out := make([]line, 0, len(grouped))
	for _, group := range grouped {
		slices.SortFunc(group, func(a, b run) int { return cmp.Compare(a.left, b.left) })
		l := line{runs: group, top: group[0].top, bottom: group[0].bottom, bold: true}
		weight := 0
		for _, r := range group {
			l.top, l.bottom = max(l.top, r.top), min(l.bottom, r.bottom)
			if chars := utf8.RuneCountInString(strings.TrimSpace(r.text)); chars > weight {
				l.size, weight = r.size, chars
			}
			l.bold = l.bold && r.bold
		}
		out = append(out, l)
	}
	return out
}

// sameLine says a run sits on the line the group is on: its middle is inside the group's
// last run's height, which tolerates a superscript and the glyph boxes of a word with no
// descenders sitting higher than one with them.
func sameLine(group []run, r run) bool {
	last := group[len(group)-1]
	middle := (r.top + r.bottom) / 2
	return middle <= last.top && middle >= last.bottom
}

// joinRuns is runs on one line as text, with a space where white divides them.
func joinRuns(runs []run) string {
	var b strings.Builder
	for i, r := range runs {
		if i > 0 && (r.left-runs[i-1].right > 0.15*max(r.size, 1) || strings.HasSuffix(runs[i-1].text, " ")) {
			b.WriteByte(' ')
		}
		b.WriteString(strings.TrimSpace(r.text))
	}
	return b.String()
}

var digits = regexp.MustCompile(`\d+`)

// runningKey names a line by where it sits and what it says, with its numbers taken out,
// so the same running head on every page has the same key whatever page it numbers. A
// running head is set no larger than the body; a chapter title at the top of every page
// is not one.
func runningKey(page pdfPage, l line, style style) string {
	if !inMargin(page, l) || l.size > style.body {
		return ""
	}
	return strconv.Itoa(int(math.Round(l.center()/4))) + "|" + digits.ReplaceAllString(l.text(), "#")
}

func inMargin(page pdfPage, l line) bool {
	return page.height > 0 && (l.center() > page.height*(1-margin) || l.center() < page.height*margin)
}

// runningLines are the running heads and feet of a document: a line in the margin that
// recurs in the same place on most pages. Repeated in every passage cut from the
// document, they would be the words every search matched.
func runningLines(pages []pdfPage, lined [][]line, style style) map[string]bool {
	if len(pages) < 3 {
		return nil
	}
	seen := map[string]int{}
	for i, page := range pages {
		once := map[string]bool{}
		for _, l := range lined[i] {
			if key := runningKey(page, l, style); key != "" && !once[key] {
				once[key] = true
				seen[key]++
			}
		}
	}
	running := map[string]bool{}
	for key, count := range seen {
		if count*2 > len(pages) {
			running[key] = true
		}
	}
	return running
}

var pageNumberPattern = regexp.MustCompile(`(?i)^\s*[-–—]?\s*(page\s+)?\d+(\s*(of|/)\s*\d+)?\s*[-–—]?\s*$`)

// pageNumber says a line is only the page's number, which the page marker already says.
func pageNumber(page pdfPage, l line) bool {
	return inMargin(page, l) && pageNumberPattern.MatchString(l.text())
}

// style is how a document sets its body and its headings.
type style struct {
	body     float64
	headings []float64
}

// styleOf reads the body size as the one most of the text is set in, and the heading
// sizes as the larger ones, largest first.
func styleOf(pages []pdfPage) style {
	weight := map[float64]int{}
	for _, page := range pages {
		for _, r := range page.runs {
			weight[math.Round(r.size*2)/2] += utf8.RuneCountInString(r.text)
		}
	}
	s := style{body: 10}
	most := 0
	for size, count := range weight {
		if count > most || (count == most && size < s.body) {
			s.body, most = size, count
		}
	}
	for size := range weight {
		if size >= s.body*headingScale {
			s.headings = append(s.headings, size)
		}
	}
	slices.SortFunc(s.headings, func(a, b float64) int { return cmp.Compare(b, a) })
	return s
}

// block is a piece of a page in reading order: lines of text, or a table.
type block struct {
	lines []line
	rows  [][]string
}

// cut reads a region of a page into blocks, in the order a reader takes them.
func (s style) cut(runs []run) []block {
	if len(runs) == 0 {
		return nil
	}
	if columns := s.columnsOf(runs); len(columns) > 1 {
		if rows, ok := s.tableOf(runs, columns); ok {
			return []block{{rows: rows}}
		}
		var blocks []block
		for _, column := range columns {
			blocks = append(blocks, s.cut(column)...)
		}
		return blocks
	}

	strips := s.stripsOf(runs)
	if len(strips) == 1 {
		return []block{{lines: lines(runs)}}
	}
	// Strips either side of a band of white that share their gutters are one block cut in
	// two by the white, such as columns pausing between paragraphs at the same height.
	var merged [][]run
	var gutters []span
	for _, strip := range strips {
		columns := s.columnsOf(strip)
		here := guttersOf(columns)
		if n := len(merged); n > 0 && len(here) > 0 && sameGutters(gutters, here) {
			merged[n-1] = append(merged[n-1], strip...)
			gutters = intersect(gutters, here)
			continue
		}
		merged = append(merged, slices.Clone(strip))
		gutters = here
	}
	if len(merged) == 1 {
		merged = strips
	}
	var blocks []block
	for _, region := range merged {
		if len(merged) == len(strips) && len(region) == len(runs) {
			return []block{{lines: lines(runs)}}
		}
		blocks = append(blocks, s.cut(region)...)
	}
	return blocks
}

type span struct{ left, right float64 }

// columnsOf cuts runs at the gutters no run crosses, left to right. A column of nothing
// but bullets or item numbers is put with the one to its right: a list's markers set
// apart from its items leave white as tall as the list, and are not a column.
func (s style) columnsOf(runs []run) [][]run {
	sorted := slices.Clone(runs)
	slices.SortFunc(sorted, func(a, b run) int { return cmp.Compare(a.left, b.left) })

	var columns [][]run
	right := math.Inf(-1)
	for _, r := range sorted {
		if n := len(columns); n > 0 && r.left-right < gutterWidth*max(r.size, s.body) {
			columns[n-1] = append(columns[n-1], r)
			right = max(right, r.right)
			continue
		}
		columns = append(columns, []run{r})
		right = r.right
	}

	for i := 0; i < len(columns)-1; {
		if markers(columns[i]) {
			columns[i+1] = append(columns[i], columns[i+1]...)
			columns = slices.Delete(columns, i, i+1)
			continue
		}
		i++
	}
	return columns
}

var marker = regexp.MustCompile(`^\(?(\d{1,3}|[a-zA-Z])[.)]$`)

func markers(runs []run) bool {
	for _, r := range runs {
		text := strings.TrimSpace(r.text)
		if bullet(text) == "" && !marker.MatchString(text) {
			return false
		}
	}
	return true
}

func guttersOf(columns [][]run) []span {
	var gutters []span
	for i := 0; i < len(columns)-1; i++ {
		right := math.Inf(-1)
		for _, r := range columns[i] {
			right = max(right, r.right)
		}
		left := math.Inf(1)
		for _, r := range columns[i+1] {
			left = min(left, r.left)
		}
		gutters = append(gutters, span{right, left})
	}
	return gutters
}

func sameGutters(a, b []span) bool {
	if len(a) != len(b) || len(a) == 0 {
		return false
	}
	for i := range a {
		if min(a[i].right, b[i].right) <= max(a[i].left, b[i].left) {
			return false
		}
	}
	return true
}

func intersect(a, b []span) []span {
	out := make([]span, len(a))
	for i := range a {
		out[i] = span{max(a[i].left, b[i].left), min(a[i].right, b[i].right)}
	}
	return out
}

// stripsOf cuts runs at the bands of white across the whole region, top to bottom.
func (s style) stripsOf(runs []run) [][]run {
	sorted := slices.Clone(runs)
	slices.SortFunc(sorted, func(a, b run) int { return cmp.Compare(b.top, a.top) })

	var strips [][]run
	bottom := math.Inf(1)
	for _, r := range sorted {
		if n := len(strips); n > 0 && bottom-r.top < blockGap*max(r.size, s.body) {
			strips[n-1] = append(strips[n-1], r)
			bottom = min(bottom, r.bottom)
			continue
		}
		strips = append(strips, []run{r})
		bottom = r.bottom
	}
	return strips
}

// tableOf reads runs cut into columns as a table, when they are one: most lines have
// something in two columns or more, and what is in them is short. Columns of prose have
// lines too long to be cells.
func (s style) tableOf(runs []run, columns [][]run) ([][]string, bool) {
	gutters := guttersOf(columns)
	var rows [][]string
	var lengths []int
	across := 0
	for _, l := range lines(runs) {
		cells := make([][]run, len(columns))
		for _, r := range l.runs {
			column := 0
			for column < len(gutters) && r.left >= gutters[column].right {
				column++
			}
			cells[column] = append(cells[column], r)
		}
		row := make([]string, len(columns))
		filled := 0
		for i, cell := range cells {
			row[i] = joinRuns(cell)
			if row[i] != "" {
				filled++
				lengths = append(lengths, utf8.RuneCountInString(row[i]))
			}
		}
		if filled >= 2 {
			across++
		}
		rows = append(rows, row)
	}
	if across < 2 || across*2 < len(rows) {
		return nil, false
	}
	slices.Sort(lengths)
	if lengths[len(lengths)/2] > cellLength {
		return nil, false
	}
	return rows, true
}

// level is the heading level a line is set at, or 0 when it is body text.
func (s style) level(l line) int {
	text := l.text()
	if utf8.RuneCountInString(text) > 120 || bullet(text) != "" || !strings.ContainsFunc(text, unicode.IsLetter) {
		return 0
	}
	size := math.Round(l.size*2) / 2
	if size >= s.body*headingScale {
		index := slices.Index(s.headings, size)
		if index < 0 {
			index = len(s.headings) - 1
		}
		return min(index+1, 4)
	}
	if l.bold && utf8.RuneCountInString(text) <= 80 && !strings.HasSuffix(text, ".") {
		return min(len(s.headings)+1, 4)
	}
	return 0
}

var orderedItem = regexp.MustCompile(`^(\d{1,3}|[a-zA-Z])[.)]\s+`)

// bullet is the Markdown a list item opens with, or "" for a line that opens none.
func bullet(text string) string {
	first, size := utf8.DecodeRuneInString(text)
	switch first {
	case '•', '◦', '▪', '▫', '‣', '⁃', '●', '○', '■', '□', '–', '-', '*', '\uF0B7', '\uF0A7':
		if rest := text[size:]; rest == "" || unicode.IsSpace([]rune(rest)[0]) {
			return "-"
		}
	}
	if match := orderedItem.FindString(text); match != "" {
		return strings.TrimSpace(match)
	}
	return ""
}

// listItem cuts a line that opens a list item into its Markdown mark and its text.
func listItem(text string) (string, string, bool) {
	mark := bullet(text)
	if mark == "" {
		return "", "", false
	}
	if mark == "-" {
		return mark, strings.TrimSpace(strings.TrimPrefix(text, strings.Fields(text)[0])), true
	}
	return mark, strings.TrimSpace(strings.TrimPrefix(text, mark)), true
}

// render writes blocks as Markdown: headings, paragraphs, lists and tables.
func (s style) render(blocks []block) string {
	var out []string
	for _, b := range blocks {
		if b.rows != nil {
			out = append(out, table(b.rows))
			continue
		}
		out = append(out, s.paragraphs(b.lines)...)
	}
	return strings.Join(out, "\n\n")
}

func (s style) paragraphs(lines []line) []string {
	var out []string
	var paragraph strings.Builder
	var previous *line
	// listing says the paragraph is a list, so the next item goes on the line under it
	// rather than in a paragraph of its own.
	listing := false
	flush := func() {
		if paragraph.Len() > 0 {
			out = append(out, paragraph.String())
			paragraph.Reset()
		}
		previous = nil
		listing = false
	}
	leftmost := math.Inf(1)
	for _, l := range lines {
		leftmost = min(leftmost, l.left())
	}

	for i := 0; i < len(lines); i++ {
		l := lines[i]
		text := l.text()
		if level := s.level(l); level > 0 && !continued(lines, i) {
			flush()
			// A heading set over two lines is one heading.
			for i+1 < len(lines) && s.level(lines[i+1]) == level && close(l, lines[i+1]) && !continued(lines, i+1) {
				i++
				l = lines[i]
				text += " " + l.text()
			}
			out = append(out, strings.Repeat("#", level)+" "+text)
			continue
		}
		if mark, item, ok := listItem(text); ok {
			if listing {
				paragraph.WriteByte('\n')
			} else {
				flush()
			}
			if l.left()-leftmost > 1.5*max(l.size, 1) {
				paragraph.WriteString("  ")
			}
			paragraph.WriteString(mark + " " + item)
			previous, listing = &lines[i], true
			continue
		}
		if previous != nil && close(*previous, l) {
			joined := paragraph.String()
			if hyphenated(joined, text) {
				paragraph.Reset()
				paragraph.WriteString(strings.TrimSuffix(joined, "-"))
			} else {
				paragraph.WriteByte(' ')
			}
			paragraph.WriteString(text)
			previous = &lines[i]
			continue
		}
		flush()
		paragraph.WriteString(text)
		previous = &lines[i]
	}
	flush()
	return out
}

// close says line b sits close enough under a to be in the same paragraph.
func close(a, b line) bool {
	return a.center()-b.center() <= paragraphGap*max(a.size, b.size, 1)
}

// continued says the sentence on line i carries on into the next, so the line is the
// start of a paragraph however it is set: a bold phrase opening a paragraph is not a
// heading.
func continued(lines []line, i int) bool {
	if i+1 >= len(lines) || !close(lines[i], lines[i+1]) {
		return false
	}
	next, _ := utf8.DecodeRuneInString(lines[i+1].text())
	return unicode.IsLower(next) || strings.HasSuffix(lines[i].text(), "-")
}

// hyphenated says a line ending in a hyphen breaks a word the next line finishes.
func hyphenated(before, after string) bool {
	if !strings.HasSuffix(before, "-") || len(before) < 2 {
		return false
	}
	last, _ := utf8.DecodeLastRuneInString(strings.TrimSuffix(before, "-"))
	next, _ := utf8.DecodeRuneInString(after)
	return unicode.IsLetter(last) && unicode.IsLower(next)
}
