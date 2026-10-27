package store

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

var (
	skuNow      = time.Date(2026, 11, 18, 12, 0, 0, 0, time.UTC)
	skuBefore   = skuNow.Add(-time.Hour)
	skuTomorrow = skuNow.Add(24 * time.Hour)
)

func skuWrite(model, sku string) PriceEntryWrite {
	return PriceEntryWrite{Entry: pricing.Entry{Modality: "llm", Provider: "openai", Model: model, SKU: sku}}
}

func skuWithdrawal(model string) PriceEntryWrite {
	write := skuWrite(model, "")
	write.Withdrawn = true
	return write
}

func skusOf(writes []PriceEntryWrite) []string {
	skus := make([]string, 0, len(writes))
	for _, write := range writes {
		skus = append(skus, write.SKU)
	}
	return skus
}

func TestANewItemIsGivenItsBaseSKUNumberedPastAnotherItems(t *testing.T) {
	history := []PriceSKU{{Item: "llm/openai/gpt-4.1", SKU: "llm-openai-gpt-4-1", EffectiveAt: skuBefore}}

	assigned, err := AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("gpt-4-1", ""), skuWrite("GPT 4 1", "")})

	require.NoError(t, err)
	require.Equal(t, []string{"llm-openai-gpt-4-1-2", "llm-openai-gpt-4-1-3"}, skusOf(assigned))
}

func TestAnItemInForceKeepsItsSKUAndRefusesAnother(t *testing.T) {
	history := []PriceSKU{{Item: "llm/openai/a", SKU: "kept", EffectiveAt: skuBefore}}

	assigned, err := AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("a", "")})
	require.NoError(t, err)
	require.Equal(t, []string{"kept"}, skusOf(assigned))
	assigned, err = AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("a", "kept")})
	require.NoError(t, err)
	require.Equal(t, []string{"kept"}, skusOf(assigned))

	_, err = AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("a", "other")})
	require.ErrorIs(t, err, ErrSKUChanged)
}

func TestANewItemTakesASKUOnlyFromAnItemWithdrawnOrOutOfForce(t *testing.T) {
	history := []PriceSKU{
		{Item: "llm/openai/a", SKU: "llm-openai-a", EffectiveAt: skuBefore},
		{Item: "llm/openai/gone", SKU: "llm-openai-gone", EffectiveAt: skuBefore.Add(-time.Hour)},
		{Item: "llm/openai/gone", SKU: "llm-openai-gone", Withdrawn: true, EffectiveAt: skuBefore},
	}

	_, err := AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("a-2026", "llm-openai-a")})
	require.ErrorIs(t, err, ErrSKUInUse, "a is in force and not withdrawn")

	assigned, err := AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("a-2026", "llm-openai-a"), skuWithdrawal("a")})
	require.NoError(t, err)
	require.Equal(t, []string{"llm-openai-a", "llm-openai-a"}, skusOf(assigned), "the withdrawal names the SKU it stops pricing")

	assigned, err = AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("gone-2026", "llm-openai-gone")})
	require.NoError(t, err, "gone is no longer in force")
	require.Equal(t, []string{"llm-openai-gone"}, skusOf(assigned))

	_, err = AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("x", "shared"), skuWrite("y", "shared")})
	require.ErrorIs(t, err, ErrSKUInUse, "one change cannot give one SKU to two items")
}

func TestAnItemPricedAgainTakesItsOldSKUWhenNoOtherHasIt(t *testing.T) {
	history := []PriceSKU{
		{Item: "llm/openai/a", SKU: "llm-openai-a", EffectiveAt: skuBefore.Add(-time.Hour)},
		{Item: "llm/openai/a", SKU: "llm-openai-a", Withdrawn: true, EffectiveAt: skuBefore},
	}

	assigned, err := AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("a", "")})
	require.NoError(t, err)
	require.Equal(t, []string{"llm-openai-a"}, skusOf(assigned))

	renamed := append(history, PriceSKU{Item: "llm/openai/a-2026", SKU: "llm-openai-a", EffectiveAt: skuBefore})
	assigned, err = AssignSKUs(renamed, skuNow, []PriceEntryWrite{skuWrite("a", "")})
	require.NoError(t, err)
	require.Equal(t, []string{"llm-openai-a-2"}, skusOf(assigned), "its successor has it now")
}

// TestASKUAChangeScheduledLaterGivesAwayIsRefused: a change at tomorrow gives b's SKU to
// b-2026, withdrawing b. Giving it to another item today would leave two items with it then.
func TestASKUAChangeScheduledLaterGivesAwayIsRefused(t *testing.T) {
	history := []PriceSKU{
		{Item: "llm/openai/b", SKU: "llm-openai-b", EffectiveAt: skuBefore},
		{Item: "llm/openai/b-2026", SKU: "llm-openai-b", EffectiveAt: skuTomorrow},
		{Item: "llm/openai/b", SKU: "llm-openai-b", Withdrawn: true, EffectiveAt: skuTomorrow},
	}

	_, err := AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("d", "llm-openai-b"), skuWithdrawal("b")})
	require.ErrorIs(t, err, ErrSKUInUse)

	assigned, err := AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("b", "")})
	require.NoError(t, err, "b changing its price before tomorrow keeps its SKU until it is withdrawn")
	require.Equal(t, []string{"llm-openai-b"}, skusOf(assigned))

	assigned, err = AssignSKUs(history, skuNow, []PriceEntryWrite{skuWrite("b-2026", "")})
	require.NoError(t, err)
	require.Equal(t, []string{"llm-openai-b-2026"}, skusOf(assigned), "b still has b's today, so b-2026 is given its own")
}

func TestASeedGivesEveryItemASKU(t *testing.T) {
	flex := skuWrite("a", "")
	flex.When = map[string]string{"service_tier": "flex"}

	assigned, err := AssignSKUs(nil, skuNow, []PriceEntryWrite{skuWrite("a", ""), flex, skuWrite("b", "house-b")})

	require.NoError(t, err)
	require.Equal(t, []string{"llm-openai-a", "llm-openai-a-service-tier-flex", "house-b"}, skusOf(assigned))
}
