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
// sheet is left out, since nobody reading the workbook sees it. A sheet's pictures follow
// its table. Its charts are Excel's own drawings, which nothing here draws.
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
	var kept pictures
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
		var blocks []string
		if len(rows) > 0 {
			blocks = append(blocks, table(rows))
		}
		markers, err := sheetPictures(book, sheet, &kept)
		if err != nil {
			return Document{}, err
		}
		blocks = append(blocks, markers...)
		if len(blocks) == 0 {
			continue
		}
		sections = append(sections, "## "+oneLine(sheet)+"\n\n"+strings.Join(blocks, "\n\n"))
	}
	return Document{Pages: []Page{{Text: strings.Join(sections, "\n\n")}}, Pictures: kept.kept}, nil
}

// sheetPictures keeps the pictures of a sheet, by the cells they are placed at, and
// returns their markers.
func sheetPictures(book *excelize.File, sheet string, kept *pictures) ([]string, error) {
	cells, err := book.GetPictureCells(sheet)
	if err != nil {
		return nil, fail(Corrupt, XLSX, fmt.Errorf("the pictures of sheet %q cannot be read: %w", sheet, err))
	}
	var markers []string
	for _, cell := range cells {
		placed, err := book.GetPictures(sheet, cell)
		if err != nil {
			return nil, fail(Corrupt, XLSX, fmt.Errorf("the pictures of sheet %q cannot be read: %w", sheet, err))
		}
		for _, picture := range placed {
			if marker := kept.keep(picture.File); marker != "" {
				markers = append(markers, marker)
			}
		}
	}
	return markers, nil
}
