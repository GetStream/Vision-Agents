package store

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// What a change's SKUs are refused for.
var (
	// ErrSKUInUse is a SKU given to an item while another item has it in force when the change
	// takes effect, and is not withdrawn by the same change, or while a change scheduled later
	// gives it to another item.
	ErrSKUInUse = errors.New("store: the SKU names another item")
	// ErrSKUChanged is a SKU other than its own given to an item in force: an item keeps its
	// SKU across every change of its price.
	ErrSKUChanged = errors.New("store: an item in force keeps its SKU")
)

// PriceSKU is the SKU one entry of a change not cancelled gave its item.
type PriceSKU struct {
	// Item is the entry's prices.Name.
	Item        string
	SKU         string
	Withdrawn   bool
	EffectiveAt time.Time
}

// priceSKUs is the SKU of every entry of a change not cancelled, in the order the list in
// force reads them: by effective time, change and id.
func priceSKUs(ctx context.Context, db bun.IDB) ([]PriceSKU, error) {
	var rows []PriceEntry
	err := db.NewSelect().Model(&rows).
		Column("pe.modality", "pe.provider", "pe.model", "pe.attributes", "pe.sku", "pe.withdrawn", "pe.effective_at").
		Join("JOIN price_changes AS pc ON pc.name = pe.change AND pc.cancelled_at IS NULL").
		Order("pe.effective_at", "pe.change", "pe.id").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: price skus: %w", err))
	}
	skus := make([]PriceSKU, 0, len(rows))
	for _, row := range rows {
		skus = append(skus, PriceSKU{Item: prices.Name(row.Entry()), SKU: row.SKU, Withdrawn: row.Withdrawn, EffectiveAt: row.EffectiveAt})
	}
	return skus, nil
}

// AssignSKUs gives every entry of a change taking effect at at its SKU, history being every
// entry written before it (priceSKUs). An item in force keeps its own; asked for another, the
// change is ErrSKUChanged. An item not in force may be given a SKU it names (as a model renamed
// by its vendor keeps the old name's): refused with ErrSKUInUse while another item has it in
// force at at, unless the change withdraws that item, and while a change scheduled after at
// gives it to another item. An item given none takes the SKU it last had when that is free,
// else its pricing.BaseSKU, numbered past every SKU another item has ever had. A withdrawal
// carries the SKU it withdraws.
func AssignSKUs(history []PriceSKU, at time.Time, writes []PriceEntryWrite) ([]PriceEntryWrite, error) {
	current := map[string]PriceSKU{}
	later := map[string]PriceSKU{}
	upcoming := map[string][]string{}
	everyone := map[string][]string{}
	for _, row := range history {
		if !slices.Contains(everyone[row.SKU], row.Item) {
			everyone[row.SKU] = append(everyone[row.SKU], row.Item)
		}
		if !row.EffectiveAt.After(at) {
			current[row.Item] = row
			continue
		}
		if _, ok := later[row.Item]; !ok {
			later[row.Item] = row
		}
		if !row.Withdrawn {
			upcoming[row.SKU] = append(upcoming[row.SKU], row.Item)
		}
	}
	holders := map[string]string{}
	for item, row := range current {
		if !row.Withdrawn {
			holders[row.SKU] = item
		}
	}
	withdrawn := map[string]bool{}
	for _, write := range writes {
		if write.Withdrawn {
			withdrawn[prices.Name(write.Entry)] = true
		}
	}
	inForce := func(item string) (string, bool) {
		row, ok := current[item]
		return row.SKU, ok && !row.Withdrawn
	}
	claimed := map[string]string{}
	free := func(sku, item string) error {
		if holder, ok := holders[sku]; ok && holder != item && !withdrawn[holder] {
			return fmt.Errorf("%w: %s is the SKU of %s, in force when the change takes effect: withdraw %s in the same change, "+
				"or give %s another SKU", ErrSKUInUse, sku, holder, holder, item)
		}
		for _, other := range upcoming[sku] {
			if other != item {
				return fmt.Errorf("%w: a change scheduled later gives %s to %s", ErrSKUInUse, sku, other)
			}
		}
		if other, ok := claimed[sku]; ok && other != item {
			return fmt.Errorf("%w: the change gives %s to %s as well", ErrSKUInUse, sku, other)
		}
		return nil
	}

	assigned := slices.Clone(writes)
	for i, write := range assigned {
		if write.Withdrawn || write.SKU == "" {
			continue
		}
		item := prices.Name(write.Entry)
		if own, ok := inForce(item); ok && own != write.SKU {
			return nil, fmt.Errorf("%w: %s has the SKU %s, not %s", ErrSKUChanged, item, own, write.SKU)
		} else if !ok {
			if err := free(write.SKU, item); err != nil {
				return nil, err
			}
		}
		claimed[assigned[i].SKU] = item
	}
	for i, write := range assigned {
		if write.SKU != "" && !write.Withdrawn {
			continue
		}
		item := prices.Name(write.Entry)
		if own, ok := inForce(item); ok {
			assigned[i].SKU = own
			if !write.Withdrawn {
				claimed[own] = item
			}
			continue
		}
		last := current[item]
		if last.SKU == "" {
			last = later[item]
		}
		if write.Withdrawn && last.SKU != "" {
			assigned[i].SKU = last.SKU
			continue
		}
		if last.SKU != "" && free(last.SKU, item) == nil {
			assigned[i].SKU = last.SKU
			claimed[last.SKU] = item
			continue
		}
		base := pricing.BaseSKU(write.Modality, write.Provider, write.Model, write.When)
		sku := base
		taken := func(sku string) bool {
			if other, ok := claimed[sku]; ok && other != item {
				return true
			}
			return slices.ContainsFunc(everyone[sku], func(other string) bool { return other != item })
		}
		for n := 2; taken(sku); n++ {
			sku = pricing.NumberedSKU(base, n)
		}
		assigned[i].SKU = sku
		if !write.Withdrawn {
			claimed[sku] = item
		}
	}
	return assigned, nil
}
