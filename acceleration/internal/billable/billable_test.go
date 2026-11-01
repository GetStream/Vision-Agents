package billable

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

type BillableSuite struct {
	suite.Suite
}

func TestBillableSuite(t *testing.T) {
	suite.Run(t, new(BillableSuite))
}

func want(modality, provider, model, region string, byok bool) pricing.Item {
	return pricing.Item{Modality: modality, Provider: provider, Model: model, Region: region, BYOK: byok}
}

func (s *BillableSuite) TestItemsListsEveryPieceOfWorkTheDeploymentBills() {
	items := Items(Deployment{
		Routing: routing.Config{
			routing.LLM: {Providers: []routing.ProviderConfig{
				{Provider: "openai", Model: "gpt", Region: "eu"},
				{Provider: "meta", Model: "llama"},
				{Provider: routing.CustomProvider, Model: "theirs"},
			}},
			routing.STT: {Providers: []routing.ProviderConfig{{Provider: "deepgram", Model: "nova"}}},
		},
		Keyed: func(modality routing.Modality, provider string) bool {
			return modality == routing.LLM && provider != "meta"
		},
		Knowledge:        "turbopuffer",
		KnowledgeRegions: []string{"us", "eu"},
		Sandboxes:        "daytona",
		SandboxRegions:   []string{"us"},
		PhoneVendors:     []string{"twilio"},
		Registrar:        "telnyx",
		SMS:              true,
		Runtime:          true,
	})

	for _, billed := range []pricing.Item{
		want("llm", "openai", "gpt", "eu", false),
		want("llm", "openai", "gpt", "eu", true),
		want("llm", "meta", "llama", "us", false),
		want("stt", "deepgram", "nova", "us", false),
		want("knowledge", "turbopuffer", "lookup", "eu", true),
		want("knowledge", "turbopuffer", "storage", "us", false),
		want("sandbox", "daytona", "running", "us", true),
		want("sandbox", "daytona", "stopped", "us", false),
		want("phone", "twilio", "dial", "us", false),
		want("phone", "twilio", "call_minutes", "us", false),
		want("phone", "telnyx", "sms", "us", false),
		want("phone", "telnyx", "dlc_campaign_month", "us", false),
		want("runtime", "stream", "voice", "us", false),
	} {
		s.Contains(items, billed)
	}
	s.NotContains(items, want("llm", "meta", "llama", "us", true), "a vendor that takes no customer key has no twin")
	s.NotContains(items, want("llm", routing.CustomProvider, "theirs", "us", true), "a customer's own model is never billed")
	s.NotContains(items, want("stt", "deepgram", "nova", "us", true))
	s.NotContains(items, want("sandbox", "daytona", "running", "eu", false), "no sandboxes run in eu")
	s.NotContains(items, want("phone", "twilio", "number", "us", false), "a number's month is quoted by its vendor")
}

func (s *BillableSuite) TestADeploymentWithoutAPartBillsNoneOfIt() {
	items := Items(Deployment{})

	s.Empty(items)
}

func (s *BillableSuite) TestInRegionKeepsTheWorkDoneThere() {
	items := []pricing.Item{want("llm", "openai", "gpt", "eu", false), want("llm", "openai", "gpt", "us", false)}

	s.Equal([]pricing.Item{want("llm", "openai", "gpt", "us", false)}, InRegion(items, ""))
	s.Equal([]pricing.Item{want("llm", "openai", "gpt", "eu", false)}, InRegion(items, "eu"))
}
