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

func (s *CatalogSuite) TestTheTwelvePluginsAreListed() {
	listed := Catalog()
	s.Len(listed, 12)
	ids := make([]string, 0, len(listed))
	for _, plugin := range listed {
		ids = append(ids, plugin.ID)
	}
	s.Equal([]string{
		"slack", "calendly", "calcom", "shopify", "salesforce", "sentry", "linear",
		"github", "hubspot", "google_calendar", "google_drive", "google_docs",
	}, ids)
}

func (s *CatalogSuite) TestSearchMatchesAName() {
	found := Search("cal")
	ids := make([]string, 0, len(found))
	for _, plugin := range found {
		ids = append(ids, plugin.ID)
	}
	s.Equal([]string{"calendly", "calcom", "google_calendar"}, ids)
}

func (s *CatalogSuite) TestEveryPluginIsDrawnWithAnSVGOfItsOwn() {
	drawn := map[string]string{}
	for _, plugin := range Catalog() {
		raw, ok := Logo(plugin.ID)
		s.Require().True(ok, plugin.ID)
		s.Contains(string(raw), "<svg", plugin.ID)
		s.NotContains(drawn, string(raw), plugin.ID+" is drawn with another plugin's logo")
		drawn[plugin.ID] = string(raw)
	}
	_, ok := Logo("notion")
	s.False(ok, "a plugin nobody has has no logo")
}

func (s *CatalogSuite) TestALogoIsServedUnderThePluginsOwnId() {
	s.Equal("/v1/agents/plugins/linear/logo", LogoPath("linear"))
}

func (s *CatalogSuite) TestLinearIsReachedWithDynamicRegistration() {
	plugin, ok := Lookup("linear")
	s.Require().True(ok)
	url, err := plugin.Endpoint("")
	s.Require().NoError(err)
	s.Equal("https://mcp.linear.app/mcp", url)
	s.Equal([]string{"read", "write"}, plugin.Scopes)
}

func (s *CatalogSuite) TestGitHubIsReachedAtCopilotsMCPServer() {
	plugin, ok := Lookup("github")
	s.Require().True(ok)
	url, err := plugin.Endpoint("")
	s.Require().NoError(err)
	s.Equal("https://api.githubcopilot.com/mcp", url)
	s.Equal([]string{"repo", "read:org", "read:user"}, plugin.Scopes)
}

func (s *CatalogSuite) TestSlackAsksForTheScopesItsToolsNeed() {
	plugin, ok := Lookup("slack")
	s.Require().True(ok)
	s.Equal([]string{"channels:history", "channels:read", "chat:write", "search:read", "users:read"}, plugin.Scopes)
}

func (s *CatalogSuite) TestDriveAndDocsAreSeparateServersAndBothAskOnlyToRead() {
	for id, wanted := range map[string][]string{
		"google_drive": {"https://www.googleapis.com/auth/drive.readonly"},
		"google_docs": {
			"https://www.googleapis.com/auth/documents.readonly",
			"https://www.googleapis.com/auth/drive.readonly",
		},
	} {
		plugin, ok := Lookup(id)
		s.Require().True(ok, id)
		s.Equal(wanted, plugin.Scopes, id)
		// Google hands back a refresh token only when asked, as it does for Calendar.
		s.Equal(map[string]string{"access_type": "offline", "prompt": "consent"}, plugin.AuthorizeParams, id)
	}
	drive, _ := Lookup("google_drive")
	docs, _ := Lookup("google_docs")
	s.NotEqual(drive.URL, docs.URL)
}

func (s *CatalogSuite) TestGoogleCalendarAsksOnlyToReadAndForARefreshToken() {
	plugin, ok := Lookup("google_calendar")
	s.Require().True(ok)
	s.Equal([]string{"https://www.googleapis.com/auth/calendar.readonly"}, plugin.Scopes)
	s.Equal(map[string]string{"access_type": "offline", "prompt": "consent"}, plugin.AuthorizeParams)
}

func (s *CatalogSuite) TestSentryIsReachedAtItsHostedServer() {
	plugin, ok := Lookup("sentry")
	s.Require().True(ok)
	url, err := plugin.Endpoint("")
	s.Require().NoError(err)
	s.Equal("https://mcp.sentry.dev/mcp", url)
	s.Empty(plugin.Scopes, "Sentry's own consent page decides what the login may do")
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
