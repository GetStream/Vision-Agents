package phone

import (
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// PriceEntries prices placing and transferring a call through every vendor, a customer's
// own trunk included, at nothing: what a vendor bills for a call is its minutes.
func PriceEntries(config Config) []pricing.Entry {
	vendors := []string{SIPTrunkVendor}
	for _, vendor := range config.Vendors {
		if !slices.Contains(vendors, vendor.Vendor) {
			vendors = append(vendors, vendor.Vendor)
		}
	}
	var entries []pricing.Entry
	for _, vendor := range vendors {
		for _, model := range []string{"call", "transfer"} {
			entries = append(entries, pricing.Entry{Modality: string(routing.Phone), Provider: vendor, Model: model, Free: true})
		}
	}
	return entries
}
