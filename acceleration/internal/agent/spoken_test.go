package agent

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSpokenNumbersBecomeDigits(t *testing.T) {
	for said, want := range map[string]string{
		"six three five seven":  "6357",
		"three eight two five":  "3825",
		"fourteen Fenwick Road": "14 Fenwick Road",
		"one four Fenwick Road": "14 Fenwick Road",
		"seventy one":           "71",
		"3825":                  "3825",
		"Fenwick Road":          "Fenwick Road",
		"one of us":             "one of us",
		"party of one":          "party of one",
		"Vantrell":              "Vantrell",
		"oh":                    "oh",
		"five one two, five five five, oh one seven eight": "5125550178",
		"PIN three eight two five, last four":              "PIN 3825 last four",
	} {
		assert.Equal(t, want, digitsFor(said), said)
	}
}
