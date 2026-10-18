package agent

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

func TestSpokenNumbersInArgumentsBecomeDigits(t *testing.T) {
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
		"Okafor":              "Okafor",
		"oh":                  "oh",
		"five one two, five five five, oh one four two": "5125550142",
	} {
		assert.Equal(t, want, digitsFor(said), said)
	}
}

func TestOnlyShortStringArgumentsAreRewritten(t *testing.T) {
	calls := withSpokenNumbers([]llm.ToolCall{
		{Name: "verify_account", Arguments: `{"last4":"nine eight two one","pin":"4471","party_size":4}`},
		{Name: "create_ticket", Arguments: `{"reason":"one of us has been without internet since six and the light is red"}`},
		{Name: "broken", Arguments: `{"last4":`},
	})

	assert.JSONEq(t, `{"last4":"9821","pin":"4471","party_size":4}`, calls[0].Arguments)
	assert.JSONEq(t, `{"reason":"one of us has been without internet since six and the light is red"}`, calls[1].Arguments)
	assert.Equal(t, `{"last4":`, calls[2].Arguments)
}
