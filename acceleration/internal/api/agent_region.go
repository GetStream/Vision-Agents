package api

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/providerkeys"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// agentRegionComplaint refuses a region that is not one: a letter, then up to 31 of a-z, 0-9
// and hyphens, as every price and instance names one. Left out is no complaint.
func agentRegionComplaint(region *string) (string, bool) {
	if region == nil || pricing.ValidRegion(*region) {
		return "", true
	}
	return fmt.Sprintf("region is us or eu for now, a letter and up to 31 of a-z, 0-9 and hyphens, not %q", *region), false
}

// regionComplaint refuses an agent whose work could not all be done in its region, naming each
// part that could not: a model it names with no instance there, its knowledge base kept in
// another region or in one this deployment keeps no knowledge in, and its sandbox in a region
// no Daytona target serves. Its sessions would be refused each of them, never served in
// another region, so the save is refused before anything is written instead.
//
// was is the agent before this save, nil for a new one. A model, a knowledge base or a sandbox
// the save leaves as it was, in the region it was in, is not asked about again, so an agent
// stored before the check stays editable, as liveSTTComplaint leaves it. A target this
// deployment does not route at all is left to the session to refuse, as before regions. A read
// that fails is the error, and the caller tries again.
func (s *Server) regionComplaint(ctx context.Context, config store.AgentConfig, was *store.AgentConfig) (string, bool, error) {
	region := cmp.Or(config.Region, pricing.DefaultRegion)
	moved := was == nil || cmp.Or(was.Region, pricing.DefaultRegion) != region
	if was == nil {
		was = &store.AgentConfig{}
	}

	keyed, err := s.keyedIn(ctx, config.CustomerID, region)
	if err != nil {
		return "", false, err
	}
	var unmet []string
	// billed is the work in the region the agent's save asks about, each named by its setting,
	// which must have a price: work with none is recorded at nothing and reserves nothing.
	var billed []settingItem
	type setting struct {
		setting       string
		modality      routing.Modality
		target, stood string
	}
	models := []setting{
		{"llm", routing.LLM, config.LLM, was.LLM},
		{"stt", routing.STT, config.STT, was.STT},
		{"tts", routing.TTS, config.TTS, was.TTS},
		{"sts", routing.STS, config.STS, was.STS},
		{"subagent", routing.LLM, config.Subagent, was.Subagent},
		{"search", routing.Search, config.Search, was.Search},
		{"image", routing.Image, config.Image, was.Image},
	}
	for _, target := range config.LLMFallbacks {
		var stood string
		if slices.Contains(was.LLMFallbacks, target) {
			stood = target
		}
		models = append(models, setting{"llm_fallbacks", routing.LLM, target, stood})
	}
	for _, model := range models {
		if model.target == "" || (!moved && model.target == model.stood) {
			continue
		}
		router, ok := s.routers[model.modality]
		if !ok {
			continue
		}
		served, err := router.InRegion(ctx, model.target, region, nil)
		if err != nil {
			continue
		}
		if !served {
			unmet = append(unmet, fmt.Sprintf("%s: no instance of %s in region %s", model.setting, model.target, region))
			continue
		}
		billed = append(billed, instancesIn(ctx, router, model.setting, model.modality, model.target, region, keyed)...)
	}

	namespace := strings.TrimSpace(config.KnowledgeNamespace)
	if s.knowledge != nil && namespace != "" && (moved || namespace != strings.TrimSpace(was.KnowledgeNamespace)) {
		place, kept, err := s.knowledge.Place(ctx, config.CustomerID, namespace)
		if err != nil {
			return "", false, err
		}
		switch {
		case kept && place.Region != region:
			unmet = append(unmet, fmt.Sprintf("knowledge: the knowledge base %q is kept in %s, and a knowledge base is never moved to another region",
				namespace, place.Region))
		case !kept && !slices.Contains(s.knowledge.Regions(), region):
			unmet = append(unmet, fmt.Sprintf("knowledge: this deployment keeps no knowledge in region %s", region))
		default:
			billed = append(billed, vendorWorkIn("knowledge", routing.Knowledge, s.knowledge.Provider(), region, keyed,
				metered.KnowledgeLookup, metered.KnowledgeWrite, metered.KnowledgeStorage)...)
		}
	}

	if s.sessions != nil && config.Sandbox != "" && (moved || was.Sandbox == "") {
		if s.sessions.SandboxesIn(region) {
			billed = append(billed, vendorWorkIn("sandbox", routing.Sandbox, providerkeys.Daytona, region, keyed,
				metered.SandboxRunning, metered.SandboxStopped)...)
		} else {
			unmet = append(unmet, fmt.Sprintf("sandbox: this deployment runs no sandboxes in region %s", region))
		}
	}

	if list := s.priceList(); list != nil {
		for _, work := range billed {
			if len(list.Unpriced([]pricing.Item{work.item})) > 0 {
				unmet = append(unmet, fmt.Sprintf("%s: %s has no price", work.setting, itemName(work.item)))
			}
		}
	}

	if len(unmet) == 0 {
		return "", true, nil
	}
	return fmt.Sprintf("the agent's work cannot all be done in %s: %s", region, strings.Join(unmet, "; ")), false, nil
}

// settingItem is a piece of work an agent's setting asks for.
type settingItem struct {
	setting string
	item    pricing.Item
}

// instancesIn is the work of each instance target resolves to in region, and its BYOK twin
// where the router takes a customer key for the instance's vendor and keyed holds one. A
// customer's own model bills nothing.
func instancesIn(ctx context.Context, router routing.Inspector, setting string, modality routing.Modality, target, region string,
	keyed map[string]bool) []settingItem {
	candidates, err := router.Resolve(ctx, target, nil)
	if err != nil {
		return nil
	}
	takes, _ := router.(interface{ TakesCustomerKey(provider string) bool })
	var billed []settingItem
	seen := map[string]bool{}
	for _, candidate := range candidates {
		instance := candidate.Config
		if instance.Provider == routing.CustomProvider || cmp.Or(instance.Region, pricing.DefaultRegion) != region {
			continue
		}
		own := routing.ItemOf(modality, instance, nil)
		if seen[itemName(own)] {
			continue
		}
		seen[itemName(own)] = true
		billed = append(billed, settingItem{setting, own})
		if takes != nil && keyed[instance.Provider] && takes.TakesCustomerKey(instance.Provider) {
			instance.BYOK = true
			billed = append(billed, settingItem{setting, routing.ItemOf(modality, instance, nil)})
		}
	}
	return billed
}

// vendorWorkIn is each model of a vendor's work in region, and its BYOK twin where keyed holds
// the customer's key for the vendor.
func vendorWorkIn(setting string, modality routing.Modality, vendor, region string, keyed map[string]bool, models ...string) []settingItem {
	var billed []settingItem
	for _, model := range models {
		work := pricing.Item{Modality: string(modality), Provider: vendor, Model: model, Region: region}
		billed = append(billed, settingItem{setting, work})
		if keyed[vendor] {
			work.BYOK = true
			billed = append(billed, settingItem{setting, work})
		}
	}
	return billed
}

// keyedIn is the vendors the customer keeps a key for in region, read through appconfig. A key
// is not opened: that one is kept is what bills its work under the BYOK twin.
func (s *Server) keyedIn(ctx context.Context, customerID, region string) (map[string]bool, error) {
	keyed := map[string]bool{}
	if s.configs == nil || customerID == "" {
		return keyed, nil
	}
	keys, err := s.configs.ProviderKeys(ctx, customerID)
	if err != nil {
		return nil, err
	}
	for _, key := range keys {
		if key.Region == region {
			keyed[key.Provider] = true
		}
	}
	return keyed, nil
}

// priceList is the price list in force, nil where the server prices by something else.
func (s *Server) priceList() *pricing.List {
	if s.prices != nil {
		return s.prices.List()
	}
	list, _ := s.pricer.(*pricing.List)
	return list
}
