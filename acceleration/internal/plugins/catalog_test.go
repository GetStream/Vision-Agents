package plugins

import (
	"testing"

	"github.com/stretchr/testify/suite"
)

type CatalogSuite struct {
	suite.Suite
}

func TestCatalogSuite(t *testing.T) {
	suite.Run(t, new(CatalogSuite))
}

func (s *CatalogSuite) TestTheFivePluginsAreListed() {
	listed := Catalog()
	s.Len(listed, 5)
	ids := make([]string, 0, len(listed))
	for _, plugin := range listed {
		ids = append(ids, plugin.ID)
	}
	s.Equal([]string{"slack", "calendly", "calcom", "shopify", "salesforce"}, ids)
}

func (s *CatalogSuite) TestSearchMatchesAName() {
	found := Search("cal")
	ids := make([]string, 0, len(found))
	for _, plugin := range found {
		ids = append(ids, plugin.ID)
	}
	s.Equal([]string{"calendly", "calcom"}, ids)
}

func (s *CatalogSuite) TestAnEmptyQueryIsTheWholeCatalog() {
	s.Equal(Catalog(), Search("  "))
}

func (s *CatalogSuite) TestAnUnknownIdIsRefused() {
	_, ok := Lookup("notion")
	s.False(ok)
}

func (s *CatalogSuite) TestShopifyNeedsAnInstance() {
	plugin, ok := Lookup("shopify")
	s.Require().True(ok)
	s.True(plugin.InstanceRequired)

	_, err := plugin.Endpoint("")
	s.ErrorContains(err, "instance url")

	url, err := plugin.Endpoint("https://mystore.myshopify.com/")
	s.Require().NoError(err)
	s.Equal("https://mystore.myshopify.com/api/mcp", url)
}

func (s *CatalogSuite) TestAGlobalPluginIgnoresAnInstance() {
	plugin, ok := Lookup("slack")
	s.Require().True(ok)
	url, err := plugin.Endpoint("ignored")
	s.Require().NoError(err)
	s.Equal("https://mcp.slack.com/mcp", url)
}
