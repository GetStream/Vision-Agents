package agent

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSpokenNumbersBecomeDigits(t *testing.T) {
	for said, want := range map[string]string{
		"nine eight two one":  "9821",
		"four four seven one": "4471",
		"fourteen Cedar Lane": "14 Cedar Lane",
		"one four Cedar Lane": "14 Cedar Lane",
		"seventy one":         "71",
		"4471":                "4471",
		"Cedar Lane":          "Cedar Lane",
		"one of us":           "one of us",
		"party of one":        "party of one",
		"Vantrell":            "Vantrell",
		"oh":                  "oh",
		"five one two, five five five, oh one four two": "5125550142",
		"PIN four four seven one, last four":            "PIN 4471 last four",
	} {
		assert.Equal(t, want, digitsFor(said), said)
	}
}
