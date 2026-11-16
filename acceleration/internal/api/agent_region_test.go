package api

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// RegionWorkSuite is the work an agent's model asks for in its region, which its sync checks
// has a price.
type RegionWorkSuite struct {
	suite.Suite
}

func TestRegionWorkSuite(t *testing.T) {
	suite.Run(t, new(RegionWorkSuite))
}

// keyedRouter resolves every target to its instances, and takes a customer key for openai.
type keyedRouter struct {
	routing.Inspector
	instances []routing.ProviderConfig
}

func (r keyedRouter) Resolve(context.Context, string, []string) ([]routing.Candidate, error) {
	candidates := make([]routing.Candidate, 0, len(r.instances))
	for _, instance := range r.instances {
		candidates = append(candidates, routing.Candidate{Config: instance})
	}
	return candidates, nil
}

func (keyedRouter) TakesCustomerKey(provider string) bool { return provider == "openai" }

func (s *RegionWorkSuite) TestAModelOnTheCustomersOwnKeyAsksForItsBYOKTwinInTheRegionAlone() {
	router := keyedRouter{instances: []routing.ProviderConfig{
		{Provider: "openai", Model: "gpt", Region: "eu"},
		{Provider: "openai", Model: "gpt"},
		{Provider: "meta", Model: "llama", Region: "eu"},
	}}

	work := instancesIn(context.Background(), router, "llm", routing.LLM, "fast", "eu", map[string]bool{"openai": true, "meta": true})

	s.Equal([]settingItem{
		{"llm", pricing.Item{Modality: "llm", Provider: "openai", Model: "gpt", Region: "eu"}},
		{"llm", pricing.Item{Modality: "llm", Provider: "openai", Model: "gpt", Region: "eu", BYOK: true}},
		{"llm", pricing.Item{Modality: "llm", Provider: "meta", Model: "llama", Region: "eu"}},
	}, work, "meta takes no customer key, and the us instance is not in the eu")
}

func (s *RegionWorkSuite) TestAVendorsWorkHasItsTwinOnlyWhereTheCustomerKeepsAKey() {
	s.Len(vendorWorkIn("sandbox", routing.Sandbox, "daytona", "eu", nil, "running", "stopped"), 2)
	s.Equal([]settingItem{
		{"sandbox", pricing.Item{Modality: "sandbox", Provider: "daytona", Model: "running", Region: "eu"}},
		{"sandbox", pricing.Item{Modality: "sandbox", Provider: "daytona", Model: "running", Region: "eu", BYOK: true}},
	}, vendorWorkIn("sandbox", routing.Sandbox, "daytona", "eu", map[string]bool{"daytona": true}, "running"))
}
