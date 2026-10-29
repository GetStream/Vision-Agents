package api

import (
	"cmp"
	"context"
	"fmt"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
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

	var unmet []string
	for _, model := range []struct {
		setting       string
		modality      routing.Modality
		target, stood string
	}{
		{"llm", routing.LLM, config.LLM, was.LLM},
		{"stt", routing.STT, config.STT, was.STT},
		{"tts", routing.TTS, config.TTS, was.TTS},
		{"sts", routing.STS, config.STS, was.STS},
		{"subagent", routing.LLM, config.Subagent, was.Subagent},
		{"search", routing.Search, config.Search, was.Search},
	} {
		if model.target == "" || (!moved && model.target == model.stood) {
			continue
		}
		router, ok := s.routers[model.modality]
		if !ok {
			continue
		}
		served, err := router.InRegion(ctx, model.target, region)
		if err != nil || served {
			continue
		}
		unmet = append(unmet, fmt.Sprintf("%s: no instance of %s in region %s", model.setting, model.target, region))
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
		}
	}

	if s.sessions != nil && config.Sandbox != "" && (moved || was.Sandbox == "") && !s.sessions.SandboxesIn(region) {
		unmet = append(unmet, fmt.Sprintf("sandbox: this deployment runs no sandboxes in region %s", region))
	}

	if len(unmet) == 0 {
		return "", true, nil
	}
	return fmt.Sprintf("the agent's work cannot all be done in %s: %s", region, strings.Join(unmet, "; ")), false, nil
}
