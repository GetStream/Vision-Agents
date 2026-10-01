package mcp

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

func (s *CatalogSuite) TestTheSevenConnectorsAreListed() {
	listed := Catalog()
	s.Len(listed, 7)
	ids := make([]string, 0, len(listed))
	for _, connector := range listed {
		ids = append(ids, connector.ID)
	}
	s.Equal([]string{"slack", "calendly", "calcom", "linear", "github", "gong", "salesforce"}, ids)
}

func (s *CatalogSuite) TestSearchMatchesAName() {
	found := Search("cal")
	ids := make([]string, 0, len(found))
	for _, connector := range found {
		ids = append(ids, connector.ID)
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

func (s *CatalogSuite) TestSalesforceUsesItsHostedProductionAndSandboxEndpoints() {
	connector, ok := Lookup("salesforce")
	s.Require().True(ok)
	s.False(connector.InstanceRequired)

	production, err := connector.Endpoint("")
	s.Require().NoError(err)
	s.Equal("https://api.salesforce.com/platform/mcp/v1/platform/sobject-all", production)

	sandbox, err := connector.Endpoint("sandbox")
	s.Require().NoError(err)
	s.Equal("https://api.salesforce.com/platform/mcp/v1/sandbox/platform/sobject-all", sandbox)
	_, err = connector.Endpoint("https://mycompany.my.salesforce.com")
	s.ErrorContains(err, "production or sandbox")
}

func (s *CatalogSuite) TestAGlobalConnectorIgnoresAnInstance() {
	connector, ok := Lookup("slack")
	s.Require().True(ok)
	url, err := connector.Endpoint("ignored")
	s.Require().NoError(err)
	s.Equal("https://mcp.slack.com/mcp", url)
}

func (s *CatalogSuite) TestSlackUsesItsDocumentedProtectedResourceIdentity() {
	connector, ok := Lookup("slack")
	s.Require().True(ok)
	s.Equal("https://mcp.slack.com", connector.Resource)
	s.Equal("client_secret_post", connector.TokenEndpointAuthMethod)
}

func (s *CatalogSuite) TestGitHubUsesDynamicOAuthWithOptionalStaticClientIDAndGongUsesBasicClientAuth() {
	github, ok := Lookup("github")
	s.Require().True(ok)
	s.Equal("dcr", github.OAuthMode)
	s.Empty(github.AuthorizationEndpoint)
	s.Empty(github.TokenEndpoint)
	s.Equal("GITHUB", github.ClientEnv)
	s.Empty(github.Scopes)

	gong, ok := Lookup("gong")
	s.Require().True(ok)
	s.Equal("customer_dcr", gong.OAuthMode)
	s.Equal("client_secret_basic", gong.TokenEndpointAuthMethod)
}

func (s *CatalogSuite) TestNamedOAuthProvidersKeepTheirDocumentedEndpointsAndScopes() {
	calendly, ok := Lookup("calendly")
	s.Require().True(ok)
	s.Equal("https://mcp.calendly.com", calendly.URL)
	s.Equal("https://mcp.calendly.com/", calendly.Resource)
	s.Equal("https://calendly.com", calendly.Issuer)
	s.Equal("https://calendly.com/oauth/register", calendly.RegistrationEndpoint)
	s.Equal([]string{"mcp:scheduling:read", "mcp:scheduling:write"}, calendly.Scopes)

	calcom, ok := Lookup("calcom")
	s.Require().True(ok)
	s.Equal("https://mcp.cal.com/mcp", calcom.URL)
	s.Equal("dcr", calcom.OAuthMode)
	s.Empty(calcom.Scopes, "do not guess hosted server scopes")

	linear, ok := Lookup("linear")
	s.Require().True(ok)
	s.Equal("https://mcp.linear.app/mcp", linear.URL)
	s.Equal("dcr", linear.OAuthMode)
	s.Equal([]string{"read", "write"}, linear.Scopes)

	salesforce, ok := Lookup("salesforce")
	s.Require().True(ok)
	s.Equal("confidential", salesforce.OAuthMode)
	s.Equal("https://login.salesforce.com/services/oauth2/authorize", salesforce.AuthorizationEndpoint)
	s.Equal("https://login.salesforce.com/services/oauth2/token", salesforce.TokenEndpoint)
	s.Equal("client_secret_post", salesforce.TokenEndpointAuthMethod)
	s.Equal([]string{"mcp_api", "refresh_token"}, salesforce.Scopes)
}
