package files

import (
	"strings"
	"unicode"
)

// echoed is the share of a paragraph's words a page's text has to hold for the paragraph to
// be the page written out again, rather than words only a figure on it shows.
const echoed = 0.8

// unread is what a page's figures were read as, less the page's own text. A model told the
// text was read already still writes out the labels it reads off a chart, and often the
// page around it, which the text layer gave once; twice would only crowd the passages. A
// summary is always kept, and so is a paragraph with words the page's text does not have,
// such as the title printed in a photo.
func unread(figures, text string) string {
	known := map[string]bool{}
	for _, word := range wordsOf(text) {
		known[word] = true
	}
	var kept []string
	for paragraph := range strings.SplitSeq(strings.TrimSpace(figures), "\n\n") {
		paragraph = strings.TrimSpace(paragraph)
		if paragraph == "" {
			continue
		}
		words := wordsOf(paragraph)
		held := 0
		for _, word := range words {
			if known[word] {
				held++
			}
		}
		if !strings.HasPrefix(paragraph, ">") && len(words) > 0 && float64(held) >= echoed*float64(len(words)) {
			continue
		}
		kept = append(kept, paragraph)
	}
	return strings.Join(kept, "\n\n")
}

// wordsOf is text's words, lowercased, with the punctuation and quotes around them left out.
func wordsOf(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}
