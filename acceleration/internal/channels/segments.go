package channels

import (
	"strings"
	"unicode/utf16"
)

// The sizes of a text message's segments: what a carrier bills by. A text that fits one
// segment has it whole; a longer one is split, and each part gives up some room to the header
// that puts the parts back together.
const (
	gsmSingle  = 160
	gsmPart    = 153
	ucs2Single = 70
	ucs2Part   = 67
)

// gsmBasic is the GSM 03.38 default alphabet, each character one septet. The escape that
// opens the extension table is left out: it is never a character of a text.
const gsmBasic = "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?" +
	"¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"

// gsmExtension is the extension table, each character an escape and itself: two septets.
const gsmExtension = "\f^{}\\[~]|€"

// Segments is how many segments a carrier bills a text message as. A text every character of
// which is in the GSM-7 alphabet is 160 septets in one segment and 153 in each of several, an
// extension character counting two; any other text is sent as UCS-2, 70 UTF-16 code units in
// one segment and 67 in each of several. A character is never split across two segments.
// An empty text is none.
func Segments(text string) int64 {
	if text == "" {
		return 0
	}
	widths, ok := gsmWidths(text)
	single, part := gsmSingle, gsmPart
	if !ok {
		widths, single, part = ucs2Widths(text), ucs2Single, ucs2Part
	}
	total := 0
	for _, width := range widths {
		total += width
	}
	if total <= single {
		return 1
	}
	segments, used := int64(1), 0
	for _, width := range widths {
		if used+width > part {
			segments, used = segments+1, 0
		}
		used += width
	}
	return segments
}

// gsmWidths is each character's septets, or false when one is not in the GSM-7 alphabet.
func gsmWidths(text string) ([]int, bool) {
	widths := make([]int, 0, len(text))
	for _, r := range text {
		switch {
		case strings.ContainsRune(gsmBasic, r):
			widths = append(widths, 1)
		case strings.ContainsRune(gsmExtension, r):
			widths = append(widths, 2)
		default:
			return nil, false
		}
	}
	return widths, true
}

// ucs2Widths is each character's UTF-16 code units: two for one outside the basic plane, such
// as an emoji.
func ucs2Widths(text string) []int {
	widths := make([]int, 0, len(text))
	for _, r := range text {
		widths = append(widths, utf16.RuneLen(r))
	}
	return widths
}
