package channels

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"
)

type SegmentsSuite struct{ suite.Suite }

func TestSegmentsSuite(t *testing.T) { suite.Run(t, new(SegmentsSuite)) }

func (s *SegmentsSuite) TestSegmentsCountsGSM7AndUCS2() {
	for _, each := range []struct {
		text     string
		segments int64
		why      string
	}{
		{"", 0, "nothing is sent"},
		{"hi", 1, "a short text"},
		{strings.Repeat("a", 160), 1, "160 septets fit one segment"},
		{strings.Repeat("a", 161), 2, "one more splits it, at 153 a part"},
		{strings.Repeat("a", 306), 2, "two full parts"},
		{strings.Repeat("a", 307), 3, "one more starts a third"},
		{strings.Repeat("€", 80), 1, "€ is two septets: 80 of them are 160"},
		{strings.Repeat("€", 81), 2, "and 81 are 162"},
		{strings.Repeat("a", 152) + "€" + strings.Repeat("a", 152), 3, "an extension character is not split across parts"},
		{"Ok 👍", 1, "an emoji makes the text UCS-2"},
		{strings.Repeat("a", 69) + "👍", 2, "an emoji is two UTF-16 units: 71 do not fit 70"},
		{strings.Repeat("é", 160), 1, "é is in the GSM-7 alphabet"},
		{strings.Repeat("ç", 70), 1, "ç is not, so 70 are one UCS-2 segment"},
		{strings.Repeat("ç", 71), 2, "and 71 are two"},
		{strings.Repeat("ç", 134), 2, "two UCS-2 parts of 67"},
		{strings.Repeat("ç", 135), 3, "and one more starts a third"},
	} {
		s.Equal(each.segments, Segments(each.text), each.why)
	}
}
