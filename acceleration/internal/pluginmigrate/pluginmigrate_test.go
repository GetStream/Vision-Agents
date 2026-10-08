package pluginmigrate_test

import (
	"bytes"
	"io/fs"
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pluginmigrate"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
)

// CatalogSuite checks the plugin catalog against the built-in connectors a login moves onto,
// without a database: T58's acceptance, every plugin id has a connector id.
type CatalogSuite struct {
	suite.Suite
}

func TestCatalogSuite(t *testing.T) {
	suite.Run(t, new(CatalogSuite))
}

func (s *CatalogSuite) TestEveryCatalogPluginHasABuiltInConnectorOfTheSameID() {
	s.Require().NotEmpty(plugins.Catalog())
	for _, plugin := range plugins.Catalog() {
		manifest := s.manifest(plugin.ID)
		s.Equal(plugin.ID, manifest.ID)
		s.Contains(manifest.Schemes, oauth2code.Name, plugin.ID)
		s.Contains(manifest.Sources, core.SourceRule{Kind: "mcp", Endpoint: "mcp"}, plugin.ID)
	}
}

// A grant is for one server, so a login moves only onto a connector that reaches the server
// the plugin reached. Salesforce's does not: the plugin reaches sobject-reads, the connector
// sobject-all (providers/salesforce.yaml), so its logins are reported and a person logs in again.
func (s *CatalogSuite) TestEachConnectorReachesThePluginsServerButSalesforce() {
	elsewhere := map[string]string{}
	for _, plugin := range plugins.Catalog() {
		instance, inputs := "", map[string]string(nil)
		if plugin.InstanceRequired {
			instance, inputs = "mystore.myshopify.com", map[string]string{"shop": "mystore.myshopify.com"}
		}
		reached, err := plugin.Endpoint(instance)
		s.Require().NoError(err)
		resolved, err := s.manifest(plugin.ID).Resolve(oauth2code.Name, inputs, nil)
		s.Require().NoError(err, plugin.ID)
		if strings.TrimSuffix(reached, "/") != strings.TrimSuffix(resolved.Endpoints["mcp"], "/") {
			elsewhere[plugin.ID] = resolved.Endpoints["mcp"]
		}
	}
	s.Equal(map[string]string{"salesforce": "https://api.salesforce.com/platform/mcp/v1/platform/sobject-all"}, elsewhere)
}

// The four Google plugins each read an env pair of their own; their connectors share one.
func (s *CatalogSuite) TestTheGoogleConnectorsShareOneClientEnv() {
	for _, id := range []string{"google_calendar", "google_drive", "google_docs", "gmail"} {
		s.Equal("GOOGLE", s.manifest(id).Client.Env, id)
	}
}

func (s *CatalogSuite) TestAMovedConnectionIDIsStableAndShapedAsAStoreID() {
	first := pluginmigrate.MovedConnectionID("login-1")
	s.Equal(first, pluginmigrate.MovedConnectionID("login-1"))
	s.NotEqual(first, pluginmigrate.MovedConnectionID("login-2"))
	s.Regexp(regexp.MustCompile(`^[0-9a-f]{32}$`), first)
}

func (s *CatalogSuite) TestTheReportPrintsEveryRowAndWhatADryRunWouldDo() {
	report := pluginmigrate.Report{Rows: []pluginmigrate.Row{
		{Kind: pluginmigrate.KindConnection, Action: pluginmigrate.Planned, Source: "agent_plugin_connections a", Target: "connector_connections b"},
		{Kind: pluginmigrate.KindConnection, Action: pluginmigrate.Skipped, Source: "agent_plugin_connections c", Note: "not connected (pending)"},
		{Kind: pluginmigrate.KindBinding, Action: pluginmigrate.Exists, Source: "agent_configs d"},
	}}
	var out bytes.Buffer

	s.Require().NoError(report.Write(&out))

	s.Contains(out.String(), "agent_plugin_connections a")
	s.Contains(out.String(), "not connected (pending)")
	s.Contains(out.String(), "3 rows: 1 would move (dry run; pass --apply to write), 1 already there, 1 skipped")
	s.Equal(1, report.Count(pluginmigrate.KindConnection, pluginmigrate.Skipped))
}

func (s *CatalogSuite) manifest(id string) core.Manifest {
	raw, err := fs.ReadFile(providers.FS, id+".yaml")
	s.Require().NoError(err, "no built-in connector %s", id)
	manifest, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	return manifest
}
