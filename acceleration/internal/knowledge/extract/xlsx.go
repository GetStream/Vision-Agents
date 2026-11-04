package extract

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/xuri/excelize/v2"
)

// maxWorkbook is the most a workbook may unpack to in all. A sheet's XML is several
// times the size of the values in it, so this is far more than a sheet anybody reads.
const maxWorkbook = 4 * maxUnzipped

// readXLSX reads a workbook as a table per sheet, each under its sheet's name, with its
// values as the sheet shows them: a price formatted as money reads as money. A hidden
// sheet is left out, since nobody reading the workbook sees it.
func readXLSX(data []byte) (Document, error) {
	book, err := excelize.OpenReader(bytes.NewReader(data), excelize.Options{UnzipSizeLimit: maxWorkbook})
	if err != nil {
		if strings.Contains(err.Error(), "unzip size") {
			return Document{}, fail(TooLarge, XLSX, err)
		}
		return Document{}, fail(Corrupt, XLSX, fmt.Errorf("the workbook cannot be opened: %w", err))
	}
	defer book.Close()

	var sections []string
	for _, sheet := range book.GetSheetList() {
		if visible, err := book.GetSheetVisible(sheet); err == nil && !visible {
			continue
		}
		rows, err := book.GetRows(sheet)
		if err != nil {
			return Document{}, fail(Corrupt, XLSX, fmt.Errorf("sheet %q cannot be read: %w", sheet, err))
		}
		start := 0
		for start < len(rows) && strings.TrimSpace(strings.Join(rows[start], "")) == "" {
			start++
		}
		rows = trimRows(rows[start:])
		if len(rows) == 0 {
			continue
		}
		sections = append(sections, "## "+oneLine(sheet)+"\n\n"+table(rows))
	}
	return Document{Pages: []Page{{Text: strings.Join(sections, "\n\n")}}}, nil
}
