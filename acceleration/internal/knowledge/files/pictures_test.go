package files

import (
	"context"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/extract"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocrrouter"
)

func TestAFileWithMoreScansThanAreReadIsRefusedBeforeAnyIsRead(t *testing.T) {
	read := extract.Document{Format: extract.PDF}
	for number := 1; number <= 11; number++ {
		read.Pages = append(read.Pages, extract.Page{Number: number, Scanned: true})
	}

	// No reader: a file refused for its scans must not have any of them read and billed.
	err := ReadPictures(context.Background(), nil, ocrrouter.Request{}, &read,
		PictureLimits{Scanned: 10, Reads: 10}, func(error) bool { return false }, slog.Default())

	var tooMany *TooManyScansError
	require.ErrorAs(t, err, &tooMany)
	require.Equal(t, TooManyScansError{Scanned: 11, Limit: 10}, *tooMany)
}
