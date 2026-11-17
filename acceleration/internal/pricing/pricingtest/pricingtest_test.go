package pricingtest

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

type FixturesSuite struct {
	suite.Suite
	list *pricing.List
}

func TestFixturesSuite(t *testing.T) {
	suite.Run(t, new(FixturesSuite))
}

func (s *FixturesSuite) SetupTest() {
	s.list = List(s.T(), "fixtures")
}

// TestEveryFixtureItemIsPricedInTheUSAndTheEUAtPricesOfItsOwn: a suite telling the regions
// apart sees a price that is not the other region's, BYOK or not.
func (s *FixturesSuite) TestEveryFixtureItemIsPricedInTheUSAndTheEUAtPricesOfItsOwn() {
	for _, item := range []func(string, bool) pricing.Item{LLM, Knowledge, Sandbox} {
		for _, byok := range []bool{false, true} {
			us, inUS := s.list.Entry(item(pricing.RegionUS, byok))
			eu, inEU := s.list.Entry(item(pricing.RegionEU, byok))
			s.Require().True(inUS, "%+v", item(pricing.RegionUS, byok))
			s.Require().True(inEU, "%+v", item(pricing.RegionEU, byok))
			s.NotEqual(us.Prices, eu.Prices, "%+v", item(pricing.RegionEU, byok))
			s.NotEqual(us.SKU, eu.SKU)
		}
	}
}

// TestABYOKEntryTheFileWritesIsPricedAsWritten: Sonnet's us BYOK entry and the eu knowledge
// write's are priced outright, below a fifth of their rates, and win over the twins
// vendor.json and the file's own eu entry would make.
func (s *FixturesSuite) TestABYOKEntryTheFileWritesIsPricedAsWritten() {
	sonnet, ok := s.list.Entry(OwnBYOKLLM())
	s.Require().True(ok)
	write, ok := s.list.Entry(Knowledge(pricing.RegionEU, true))
	s.Require().True(ok)

	s.Equal(0.2, sonnet.Prices[pricing.InputTokens].USD, "a tenth of its rate of 2")
	s.Equal(0.3, write.Prices[pricing.BytesWritten].USD, "an eighth of its rate of 2.4")
}

func (s *FixturesSuite) TestEveryEntryOfVendorJSONIsKept() {
	us, ok := s.list.Entry(LLM(pricing.RegionUS, false))
	s.Require().True(ok)

	s.Equal(0.1, us.Rates[pricing.InputTokens].USD)
	s.Equal("llm-anthropic-claude-haiku-5-5-us", us.SKU)
}
