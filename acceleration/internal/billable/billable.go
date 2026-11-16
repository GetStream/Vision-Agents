// Package billable lists the work a deployment bills, one item per thing that needs a price,
// so that each can be checked for one: at start, in the price-list API and by
// `router prices show --unpriced`. Work with no price is recorded at nothing and reserves
// nothing, so an item missing here is work that would go unbilled unnoticed.
package billable

import (
	"cmp"
	"maps"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// RuntimeProvider is who runtime rows are billed under: the deployment itself.
const RuntimeProvider = "stream"

// Deployment is what a deployment runs that bills.
type Deployment struct {
	// Routing is the providers the routers serve, each in its region. Keyed reports whether the
	// router of a modality takes a customer key for a provider: its work on that key is billed
	// under a BYOK twin of the provider's item.
	Routing routing.Config
	Keyed   func(modality routing.Modality, provider string) bool
	// Knowledge is the knowledge vendor, empty on a deployment that keeps none, and
	// KnowledgeRegions the regions it keeps knowledge in.
	Knowledge        string
	KnowledgeRegions []string
	// Sandboxes is the sandbox vendor, empty on a deployment that runs none, and
	// SandboxRegions the regions it runs them in.
	Sandboxes      string
	SandboxRegions []string
	// PhoneVendors are the phone vendors the deployment can place and take calls through, and
	// Registrar its 10DLC registrar, empty with none. SMS is whether it sends texts.
	PhoneVendors []string
	Registrar    string
	SMS          bool
	// Runtime is whether sessions run here, billed for the time they are open.
	Runtime bool
}

// Items is one item per piece of work the deployment bills, each with no attributes, in its
// region, beside its BYOK twin where a customer's own key bills it. A customer can keep a
// knowledge and a sandbox key in any region, so both always have their twins.
//
// Left out is work that is never priced from the list: a number's month, which the vendor
// quotes, and WhatsApp and iMessage, which the app's own account bills.
func Items(d Deployment) []pricing.Item {
	items := routing.Items(d.Routing)
	for _, modality := range slices.Sorted(maps.Keys(d.Routing)) {
		for _, provider := range d.Routing[modality].Providers {
			if provider.Provider == routing.CustomProvider || d.Keyed == nil || !d.Keyed(modality, provider.Provider) {
				continue
			}
			twin := provider
			twin.BYOK = true
			items = append(items, routing.ItemOf(modality, twin, nil))
		}
	}
	if d.Knowledge != "" {
		items = append(items, inRegions(routing.Knowledge, d.Knowledge, d.KnowledgeRegions,
			metered.KnowledgeLookup, metered.KnowledgeWrite, metered.KnowledgeStorage)...)
	}
	if d.Sandboxes != "" {
		items = append(items, inRegions(routing.Sandbox, d.Sandboxes, d.SandboxRegions,
			metered.SandboxRunning, metered.SandboxStopped)...)
	}
	for _, vendor := range slices.Sorted(slices.Values(d.PhoneVendors)) {
		items = append(items, item(routing.Phone, vendor, phone.ModelDial), item(routing.Phone, vendor, phone.ModelCallMinutes))
	}
	if d.SMS {
		items = append(items, item(routing.Phone, channels.Vendor(channels.SMS), string(channels.SMS)))
	}
	if d.Registrar != "" {
		items = append(items, item(routing.Phone, d.Registrar, dlc.ModelBrand), item(routing.Phone, d.Registrar, dlc.ModelCampaign),
			item(routing.Phone, d.Registrar, dlc.ModelCampaignMonth))
	}
	if d.Runtime {
		for _, mode := range []string{store.ModalityText, store.ModalityVoice, store.ModalityVideo} {
			items = append(items, item(routing.Runtime, RuntimeProvider, mode))
		}
	}
	return items
}

// InRegion is the items of work done in region: what an agent kept there bills.
func InRegion(items []pricing.Item, region string) []pricing.Item {
	region = cmp.Or(region, pricing.DefaultRegion)
	return slices.DeleteFunc(slices.Clone(items), func(item pricing.Item) bool { return item.Region != region })
}

// inRegions is each model of the vendor in each region, and its BYOK twin.
func inRegions(modality routing.Modality, vendor string, regions []string, models ...string) []pricing.Item {
	if len(regions) == 0 {
		regions = []string{pricing.DefaultRegion}
	}
	var items []pricing.Item
	for _, region := range slices.Sorted(slices.Values(regions)) {
		for _, model := range models {
			for _, byok := range []bool{false, true} {
				items = append(items, pricing.Item{Modality: string(modality), Provider: vendor, Model: model, Region: region, BYOK: byok})
			}
		}
	}
	return items
}

// item is work in the default region, which phone, texts, fees and runtime always are.
func item(modality routing.Modality, provider, model string) pricing.Item {
	return pricing.Item{Modality: string(modality), Provider: provider, Model: model, Region: pricing.DefaultRegion}
}
