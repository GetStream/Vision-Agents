package store

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

func TestARowIsPricedByItsPromptSplitIntoFreshCachedAndWritten(t *testing.T) {
	request := Request{
		InputTokens:       2_400,
		CachedInputTokens: 11,
		OutputTokens:      22,
		Meters:            pricing.Quantities{pricing.CacheWriteInputTokens: 2_386, pricing.Pixels: 1_000_000},
	}

	require.Equal(t, pricing.Quantities{
		pricing.InputTokens:           3,
		pricing.CachedInputTokens:     11,
		pricing.CacheWriteInputTokens: 2_386,
		pricing.OutputTokens:          22,
		pricing.Pixels:                1_000_000,
	}, request.Quantities())
}

func TestARowLeavesOutWhatItDidNotUse(t *testing.T) {
	require.Equal(t, pricing.Quantities{pricing.AudioMs: 1_234}, Request{AudioMs: 1_234}.Quantities())
}

func TestARowNeverPricesANegativePrompt(t *testing.T) {
	// A provider that reported more cached tokens than its prompt should not lower the bill.
	used := Request{InputTokens: 100, CachedInputTokens: 1_000}.Quantities()

	require.Equal(t, pricing.Quantities{pricing.CachedInputTokens: 1_000}, used)
}
