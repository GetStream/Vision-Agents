package phone

import (
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// PriceEntries prices placing a call or a transfer through every vendor, a customer's own
// trunk included, at nothing: what a vendor bills for a call is its minutes, which are priced
// by country and direction in the vendor price list. A number needs no rate: each month is
// billed at the price the vendor quoted, and one with no quote is billable work with no price.
// A number on the customer's own trunk, and its minutes, are billed by the customer's carrier.
func PriceEntries(config Config) []pricing.Entry {
	vendors := []string{SIPTrunkVendor}
	for _, vendor := range config.Vendors {
		if !slices.Contains(vendors, vendor.Vendor) {
			vendors = append(vendors, vendor.Vendor)
		}
	}
	free := func(vendor, model string) pricing.Entry {
		return pricing.Entry{Modality: string(routing.Phone), Provider: vendor, Model: model, Free: true}
	}
	entries := []pricing.Entry{free(SIPTrunkVendor, ModelNumber), free(SIPTrunkVendor, ModelCallMinutes)}
	for _, vendor := range vendors {
		entries = append(entries, free(vendor, ModelDial))
	}
	return entries
}
