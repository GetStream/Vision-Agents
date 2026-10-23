package phone

import (
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// PriceEntries prices placing a call or a transfer through every vendor, a customer's own
// trunk included, at nothing: what a vendor bills for a call is its minutes, which are priced
// by country and direction in the vendor price list. A number needs no rate either: each month
// is billed at the price the vendor quoted when it was bought, and a purchase that failed
// bought nothing.
func PriceEntries(config Config) []pricing.Entry {
	vendors := []string{SIPTrunkVendor}
	for _, vendor := range config.Vendors {
		if !slices.Contains(vendors, vendor.Vendor) {
			vendors = append(vendors, vendor.Vendor)
		}
	}
	var entries []pricing.Entry
	for _, vendor := range vendors {
		for _, model := range []string{ModelDial, ModelNumber} {
			entries = append(entries, pricing.Entry{Modality: string(routing.Phone), Provider: vendor, Model: model, Free: true})
		}
	}
	return entries
}
