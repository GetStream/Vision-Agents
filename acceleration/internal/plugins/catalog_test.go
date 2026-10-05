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

func (s *CatalogSuite) TestReadonlyLinearIsReachedAtItsReadOnlyEndpointAskingOnlyToRead() {
	plugin, ok := Lookup("linear")
	s.Require().True(ok)

	readonly, err := plugin.Configured(Options{Readonly: true})
	s.Require().NoError(err)
	url, err := readonly.Endpoint("")
	s.Require().NoError(err)
	s.Equal("https://mcp.linear.app/mcp/readonly", url)
	s.Equal([]string{"read"}, readonly.Scopes)
	s.Equal([]string{"read", "write"}, plugin.Scopes, "the catalog's own entry is left as it was")
}

func (s *CatalogSuite) TestScopesAConfigGivesReplaceTheCatalogs() {
	plugin, ok := Lookup("linear")
	s.Require().True(ok)

	configured, err := plugin.Configured(Options{Scopes: []string{"read"}})
	s.Require().NoError(err)
	s.Equal("https://mcp.linear.app/mcp", configured.URL)
	s.Equal([]string{"read"}, configured.Scopes)
}

func (s *CatalogSuite) TestDriveCanBeAskedForTheFilesItCreatesAsGooglesGuideDoes() {
	plugin, ok := Lookup("google_drive")
	s.Require().True(ok)
	s.Equal([]string{"https://www.googleapis.com/auth/drive.readonly"}, plugin.Scopes, "read only unless asked")

	wanted := []string{
		"https://www.googleapis.com/auth/drive.readonly",
		"https://www.googleapis.com/auth/drive.file",
	}
	configured, err := plugin.Configured(Options{Scopes: wanted})
	s.Require().NoError(err)
	s.Equal(wanted, configured.Scopes)
}

func (s *CatalogSuite) TestAScopeTheServerDoesNotAdvertiseIsRefused() {
	drive, _ := Lookup("google_drive")
	_, err := drive.Configured(Options{Scopes: []string{"https://www.googleapis.com/auth/gmail.readonly"}})
	s.ErrorContains(err, "gmail.readonly")

	linear, _ := Lookup("linear")
	_, err = linear.Configured(Options{Readonly: true, Scopes: []string{"write"}})
	s.ErrorContains(err, `"write"`, "the read-only endpoint accepts only read")
}

func (s *CatalogSuite) TestAPluginWithNoReadOnlyEndpointCannotBeMadeReadonly() {
	plugin, ok := Lookup("sentry")
	s.Require().True(ok)

	_, err := plugin.Configured(Options{Readonly: true})
	s.ErrorContains(err, "read-only")
}

func (s *CatalogSuite) TestCalcomIsLimitedToTheToolsetsPickedOnItsURL() {
	plugin, ok := Lookup("calcom")
	s.Require().True(ok)

	configured, err := plugin.Configured(Options{Toolsets: []string{"bookings", "availability"}})
	s.Require().NoError(err)
	url, err := configured.Endpoint("")
	s.Require().NoError(err)
	s.Equal("https://mcp.cal.com/mcp?toolsets=bookings,availability", url)
}

func (s *CatalogSuite) TestAToolsetThePluginDoesNotHaveIsRefused() {
	calcom, _ := Lookup("calcom")
	_, err := calcom.Configured(Options{Toolsets: []string{"invoices"}})
	s.ErrorContains(err, `"invoices"`)

	sentry, _ := Lookup("sentry")
	_, err = sentry.Configured(Options{Toolsets: []string{"bookings"}})
	s.ErrorContains(err, `"bookings"`, "a plugin with no toolsets cannot be limited")
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
	s.Equal([]string{"channels:history", "channels:read", "chat:write", "search:read.public", "users:read"}, plugin.Scopes)
}

func (s *CatalogSuite) TestSlackScopesCanBeChosen() {
	plugin, ok := Lookup("slack")
	s.Require().True(ok)

	chosen, err := plugin.Configured(Options{Scopes: []string{"search:read.public", "search:read.private", "channels:history"}})
	s.Require().NoError(err)
	s.Equal([]string{"search:read.public", "search:read.private", "channels:history"}, chosen.Scopes)

	_, err = plugin.Configured(Options{Scopes: []string{"search:read"}})
	s.ErrorContains(err, `does not accept the scope "search:read"`)
}

func (s *CatalogSuite) TestEveryDefaultScopeIsOneTheServerAccepts() {
	for _, plugin := range Catalog() {
		if len(plugin.ScopesSupported) == 0 {
			continue
		}
		for _, scope := range plugin.Scopes {
			s.Contains(plugin.ScopesSupported, scope, plugin.ID)
		}
	}
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

func (s *CatalogSuite) TestSalesforceIsOneHostForEveryOrg() {
	plugin, ok := Lookup("salesforce")
	s.Require().True(ok)
	s.False(plugin.InstanceRequired)

	url, err := plugin.Endpoint("")
	s.Require().NoError(err)
	s.Equal("https://api.salesforce.com/platform/mcp/v1/platform/sobject-reads", url)
	s.Equal([]string{"mcp_api", "refresh_token"}, plugin.Scopes)
}

func (s *CatalogSuite) TestAGlobalPluginIgnoresAnInstance() {
	plugin, ok := Lookup("slack")
	s.Require().True(ok)
	url, err := plugin.Endpoint("ignored")
	s.Require().NoError(err)
	s.Equal("https://mcp.slack.com/mcp", url)
}
