package main

import (
	"log/slog"
	"maps"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/billable"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	dlctelnyx "github.com/GetStream/Vision-Agents/acceleration/internal/dlc/telnyx"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona"
)

// billableOf is what this deployment bills, read from its settings alone, so that the router
// at start, its price-list API and `router prices show --unpriced` list the same items.
func billableOf(settings config.Config, capabilities routing.Config, vendorConfig phone.Config) billable.Deployment {
	llm := llmrouter.DefaultRegistry()
	deployment := billable.Deployment{
		Routing: capabilities,
		Keyed: func(modality routing.Modality, provider string) bool {
			return modality == routing.LLM && llm.Honors(provider, routing.OverrideAPIKey)
		},
		Sandboxes:      daytona.Provider,
		SandboxRegions: sandboxRegions(settings),
		PhoneVendors:   vendors.Registry(vendorConfig).Available(),
		Runtime:        true,
	}
	deployment.SMS = slices.Contains(deployment.PhoneVendors, channels.Vendor(channels.SMS))
	if backends, err := knowledgeBackends(settings.Knowledge, 0, slog.New(slog.DiscardHandler)); err == nil {
		deployment.Knowledge = backends[pricing.DefaultRegion].Provider()
		deployment.KnowledgeRegions = slices.Sorted(maps.Keys(backends))
	}
	if registrar, err := dlctelnyx.New(dlctelnyx.Options{}); err == nil {
		deployment.Registrar = registrar.Name()
	}
	return deployment
}

// sandboxRegions is the default region, where sandboxes always run, and every region a
// Daytona target serves.
func sandboxRegions(settings config.Config) []string {
	regions := slices.Collect(maps.Keys(settings.CodeSandbox.Targets))
	if !slices.Contains(regions, pricing.DefaultRegion) {
		regions = append(regions, pricing.DefaultRegion)
	}
	slices.Sort(regions)
	return regions
}

// logUnpriced logs each item the deployment bills that list has no price for, so it can be
// priced, or marked free or local, before a row of it is recorded at nothing.
func logUnpriced(list *pricing.List, items []pricing.Item, logger *slog.Logger) {
	for _, item := range list.Unpriced(items) {
		logger.Error("billable work has no price", "modality", item.Modality, "provider", item.Provider, "model", item.Model,
			"region", item.Region, "byok", item.BYOK)
	}
}
