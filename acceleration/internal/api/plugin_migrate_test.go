//go:build integration

package api

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"strconv"
	"sync"
	"testing"
	"testing/fstest"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pluginmigrate"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// movedPlugin is the plugin the suite's rows name, and the built-in connector they move onto:
// crm, so the connecting model's first call, crm__echo, reaches it (connectorEcho).
const movedPlugin = "crm"

// PluginMigrateSuite is router plugins migrate (T61) on a copy of plugin rows: the plugin
// tables written as the plugin system writes them, moved, then used through the session path
// of #775. The fake provider (T5) is the MCP server and the authorization server, and issues
// the token every login holds, so a call that reaches it went with the moved grant. Each test
// is an app of its own, and each run moves that app's rows only.
type PluginMigrateSuite struct {
	RouterSuite
	provider *fakeprovider.Server
	// token is an access token the fake issued. Synthetic, fresh per suite.
	token string
}

func TestPluginMigrateSuite(t *testing.T) {
	runSuite(t, new(PluginMigrateSuite))
}

func (s *PluginMigrateSuite) SetupSuite() {
	s.provider = fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	code, err := oauth2code.New(oauth2code.Config{HTTP: s.provider.Client(), PublicEndpoint: loopbackOrPublic, Clients: s.clients})
	s.Require().NoError(err)
	s.connectors = core.Registry{
		Schemes:     map[string]core.Scheme{oauth2code.Name: code},
		ToolSources: map[string]core.ToolSource{mcp.Kind: mcp.New()},
	}
	s.connectorHTTP = s.provider.Client()
	s.RouterSuite.SetupSuite()
	s.token = issuedToken(&s.RouterSuite, s.provider)
	s.seedConnectors()
}

func (s *PluginMigrateSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *PluginMigrateSuite) TestADryRunListsEveryRowAndItsTargetAndWritesNothing() {
	fixed := s.config([]store.PluginEntry{{Name: movedPlugin}}, nil)
	s.pluginClient(fixed, s.provider.ClientID, s.provider.ClientSecret)
	app := s.login(fixed, "", s.provider.ClientID)
	personal := s.config(nil, []store.PluginEntry{{Name: movedPlugin}})
	mine := s.login(personal, s.client.userID, s.provider.ClientID)
	before := s.connectorRows()

	report := s.run(false)

	s.Equal(before, s.connectorRows(), "a dry run writes nothing")
	s.Equal(pluginmigrate.Planned, s.row(report, pluginmigrate.KindClient, "agent_plugin_clients "+s.customerID()+"/"+fixed).Action)
	s.Equal(pluginmigrate.Planned, s.row(report, pluginmigrate.KindConnection, app).Action)
	s.Contains(s.row(report, pluginmigrate.KindConnection, app).Target, pluginmigrate.MovedConnectionID(app))
	s.Equal(pluginmigrate.Planned, s.row(report, pluginmigrate.KindConnection, mine).Action)
	s.Equal(pluginmigrate.Planned, s.row(report, pluginmigrate.KindBinding, fixed+" agent_plugins").Action)
	s.Equal(pluginmigrate.Planned, s.row(report, pluginmigrate.KindBinding, personal+" user_plugins").Action)
	s.Len(report.Rows, 5, "every row read is listed")
	s.False(report.Applied)
}

// TestARealRunLeavesTheAgentsToolsWorkingWithNoNewLogin: the app's login, moved and bound
// fixed, answers the connecting model's call, and nobody logged in or registered a client.
func (s *PluginMigrateSuite) TestARealRunLeavesTheAgentsToolsWorkingWithNoNewLogin() {
	config := s.config([]store.PluginEntry{{Name: movedPlugin}}, nil)
	s.pluginClient(config, s.provider.ClientID, s.provider.ClientSecret)
	s.login(config, "", s.provider.ClientID)
	authorized, registered := s.provider.Hits(fakeprovider.PathAuthorize), s.provider.Hits(fakeprovider.PathRegister)

	report := s.run(true)
	opened := s.client.createSession(CreateSessionRequest{ConfigId: &config, Text: pointerTo(true)})
	ran := s.toolRan(s.serverClient.actingFor(s.client), opened.Id)

	s.Zero(report.Count("", pluginmigrate.Skipped), "%v", report.Rows)
	s.Equal(connectorEcho, ran["tool"])
	s.Equal(connectorEchoText, ran["result"])
	s.Equal(authorized, s.provider.Hits(fakeprovider.PathAuthorize), "no new login")
	s.Equal(registered, s.provider.Hits(fakeprovider.PathRegister), "no new client")
}

// TestAPersonsMovedLoginAnswersTheirSessionBinding: a user_plugins entry becomes a session
// binding, and the person's moved login, chosen when they open the session, answers it.
func (s *PluginMigrateSuite) TestAPersonsMovedLoginAnswersTheirSessionBinding() {
	config := s.config(nil, []store.PluginEntry{{Name: movedPlugin}})
	mine := s.login(config, s.client.userID, s.provider.ClientID)

	s.run(true)
	chosen := []SessionConnectorBinding{{Name: movedPlugin, ConnectionId: pluginmigrate.MovedConnectionID(mine)}}
	opened := s.client.createSession(CreateSessionRequest{ConfigId: &config, Text: pointerTo(true), ConnectorBindings: &chosen})
	ran := s.toolRan(s.serverClient.actingFor(s.client), opened.Id)

	s.Equal(connectorEchoText, ran["result"])
	stored := s.binding(config)
	s.Equal("session", stored.Connection.Type)
	s.Empty(stored.Connection.ConnectionID)
	s.NotEmpty(stored.Tools)
}

func (s *PluginMigrateSuite) TestASecondRunChangesNothing() {
	fixed := s.config([]store.PluginEntry{{Name: movedPlugin}}, nil)
	s.pluginClient(fixed, s.provider.ClientID, s.provider.ClientSecret)
	s.login(fixed, "", s.provider.ClientID)
	personal := s.config(nil, []store.PluginEntry{{Name: movedPlugin}})
	s.login(personal, s.client.userID, s.provider.ClientID)
	first := s.run(true)
	after := s.connectorRows()

	second := s.run(true)

	s.Equal(5, first.Count("", pluginmigrate.Written), "%v", first.Rows)
	s.Equal(after, s.connectorRows())
	s.Zero(second.Count("", pluginmigrate.Written), "%v", second.Rows)
	s.Equal(5, second.Count("", pluginmigrate.Exists))
}

// TestARunStoppedBetweenItsTwoWritesIsFinishedByTheNext: the connection was made and its
// grant never saved; the next run saves it rather than making another.
func (s *PluginMigrateSuite) TestARunStoppedBetweenItsTwoWritesIsFinishedByTheNext() {
	config := s.config([]store.PluginEntry{{Name: movedPlugin}}, nil)
	s.pluginClient(config, s.provider.ClientID, s.provider.ClientSecret)
	login := s.login(config, "", s.provider.ClientID)
	definition, err := s.store.LatestConnectorDefinition(context.Background(), s.customerID(), movedPlugin)
	s.Require().NoError(err)
	half := store.ConnectorConnection{CustomerID: s.customerID(), ConnectorID: movedPlugin, DefinitionRevision: definition.Revision,
		OwnerType: store.OwnerApp, AuthScheme: oauth2code.Name}
	s.Require().NoError(s.store.CreateConnectorConnectionWithID(context.Background(), s.connectors, &half, pluginmigrate.MovedConnectionID(login)))

	report := s.run(true)

	s.Equal(pluginmigrate.Written, s.row(report, pluginmigrate.KindConnection, login).Action)
	moved, err := s.store.ConnectorConnection(context.Background(), s.customerID(), pluginmigrate.MovedConnectionID(login))
	s.Require().NoError(err)
	s.Equal(store.ConnectionConnected, moved.Status)
	s.Equal(1, s.count("SELECT count(*) FROM connector_connections WHERE customer_id = ?", s.customerID()))
}

func (s *PluginMigrateSuite) TestThePluginTablesAreByteIdenticalAfterARealRun() {
	fixed := s.config([]store.PluginEntry{{Name: movedPlugin}}, nil)
	s.pluginClient(fixed, s.provider.ClientID, s.provider.ClientSecret)
	s.login(fixed, "", s.provider.ClientID)
	personal := s.config(nil, []store.PluginEntry{{Name: movedPlugin}})
	s.login(personal, s.client.userID, s.provider.ClientID)
	before := s.pluginTables()

	report := s.run(true)

	s.Equal(5, report.Count("", pluginmigrate.Written), "%v", report.Rows)
	s.Equal(before, s.pluginTables())
}

// TestAnAppLoginIsAppOwnedAndAPersonsIsTheirsWithTheTokensSealed: user_id empty is the app's
// login, so its connection has owner app and no owner id; a person's has owner user and
// their id. Neither row holds a token in the clear.
func (s *PluginMigrateSuite) TestAnAppLoginIsAppOwnedAndAPersonsIsTheirsWithTheTokensSealed() {
	config := s.config([]store.PluginEntry{{Name: movedPlugin}}, nil)
	s.pluginClient(config, s.provider.ClientID, s.provider.ClientSecret)
	app := s.login(config, "", s.provider.ClientID)
	mine := s.login(config, s.client.userID, s.provider.ClientID)

	s.run(true)

	ctx := context.Background()
	appOwned, err := s.store.ConnectorConnection(ctx, s.customerID(), pluginmigrate.MovedConnectionID(app))
	s.Require().NoError(err)
	s.Equal(store.OwnerApp, appOwned.OwnerType)
	s.Empty(appOwned.OwnerID)
	personal, err := s.store.ConnectorConnection(ctx, s.customerID(), pluginmigrate.MovedConnectionID(mine))
	s.Require().NoError(err)
	s.Equal(store.OwnerUser, personal.OwnerType)
	s.Equal(s.client.userID, personal.OwnerID)
	for _, moved := range []store.ConnectorConnection{appOwned, personal} {
		s.Equal(store.ConnectionConnected, moved.Status)
		s.Equal(oauth2code.Name, moved.AuthScheme)
		s.NotEmpty(moved.CredentialsSealed)
		s.Equal(s.sealer.CurrentVersion(), moved.CredentialsKEKVersion)
		s.False(bytes.Contains(moved.CredentialsSealed, []byte(s.token)), "the access token is sealed")
		s.False(bytes.Contains(moved.CredentialsSealed, []byte("plugin-refresh-")), "the refresh token is sealed")
		s.NotContains(s.text("SELECT cc::text FROM connector_connections cc WHERE id = ?", moved.ID), s.token, "no column holds it")
		s.NotContains(s.text("SELECT encode(credentials_sealed, 'escape') FROM connector_connections WHERE id = ?", moved.ID), s.token)
	}
}

// TestAnUnmappableRowIsReportedAndSkipped: a login to an MCP server named by its URL, a
// login never finished, and a grant the plugin renewed at another token endpoint each get a
// row saying why, and nothing is made for them.
func (s *PluginMigrateSuite) TestAnUnmappableRowIsReportedAndSkipped() {
	config := s.config([]store.PluginEntry{{Name: movedPlugin}}, nil)
	byURL := s.loginTo(config, "", "my-server", s.provider.ClientID, s.provider.URL+fakeprovider.PathToken, store.PluginConnected)
	pending := s.loginTo(config, "", movedPlugin, s.provider.ClientID, s.provider.URL+fakeprovider.PathToken, store.PluginPending)
	elsewhere := s.loginTo(config, s.client.userID, movedPlugin, s.provider.ClientID, "https://elsewhere.example/token", store.PluginConnected)

	report := s.run(true)

	s.Contains(s.row(report, pluginmigrate.KindConnection, byURL).Note, "not a catalog plugin")
	s.Contains(s.row(report, pluginmigrate.KindConnection, pending).Note, "not connected (pending)")
	s.Contains(s.row(report, pluginmigrate.KindConnection, elsewhere).Note, "renewed at https://elsewhere.example/token")
	s.Contains(s.row(report, pluginmigrate.KindBinding, config+" agent_plugins").Note, "no login of the app's was moved")
	s.Equal(4, report.Count("", pluginmigrate.Skipped), "%v", report.Rows)
	s.Zero(s.count("SELECT count(*) FROM connector_connections WHERE customer_id = ?", s.customerID()))
	s.Empty(s.storedConfig(config).Connectors)
}

// TestTheNewestClientOfAnAppWinsAndTheOtherIsReported: two configs set different clients for
// the plugin; the app gets one record, the one updated last, and a login issued to the other
// is not moved, since that client would not renew it.
func (s *PluginMigrateSuite) TestTheNewestClientOfAnAppWinsAndTheOtherIsReported() {
	older := s.config([]store.PluginEntry{{Name: movedPlugin}}, nil)
	s.pluginClient(older, "older-client", "older-secret")
	stale := s.login(older, "", "older-client")
	newer := s.config([]store.PluginEntry{{Name: movedPlugin}}, nil)
	s.pluginClient(newer, s.provider.ClientID, s.provider.ClientSecret)
	current := s.login(newer, "", s.provider.ClientID)

	report := s.run(true)

	record, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), movedPlugin)
	s.Require().NoError(err)
	s.Equal(s.provider.ClientID, record.ClientID)
	s.Equal(core.ClientCustomer, record.Registration)
	s.False(bytes.Contains(record.SecretSealed, []byte(s.provider.ClientSecret)), "the client secret is sealed")
	lost := s.row(report, pluginmigrate.KindClient, "agent_plugin_clients "+s.customerID()+"/"+older)
	s.Equal(pluginmigrate.Skipped, lost.Action)
	s.Contains(lost.Note, "client older-client differs from config "+newer)
	s.Contains(s.row(report, pluginmigrate.KindConnection, stale).Note, "issued to client older-client")
	s.Equal(pluginmigrate.Written, s.row(report, pluginmigrate.KindConnection, current).Action)
}

// TestARotatingConnectorsGrantIsSkippedUnlessAskedFor: the plugin row keeps its copy of the
// refresh token, so a grant to a connector whose refresh tokens rotate is skipped with why,
// in a dry run and a real one, until --include-rotating asks for it.
func (s *PluginMigrateSuite) TestARotatingConnectorsGrantIsSkippedUnlessAskedFor() {
	config := s.config([]store.PluginEntry{{Name: rotatingPlugin}}, nil)
	login := s.loginTo(config, "", rotatingPlugin, s.provider.ClientID, s.provider.URL+fakeprovider.PathToken, store.PluginConnected)

	dry, real := s.run(false), s.run(true)

	for _, report := range []pluginmigrate.Report{dry, real} {
		row := s.row(report, pluginmigrate.KindConnection, login)
		s.Equal(pluginmigrate.Skipped, row.Action)
		s.Contains(row.Note, "connector "+rotatingPlugin+" rotates refresh tokens")
		s.Contains(row.Note, "--include-rotating")
	}
	s.Zero(s.count("SELECT count(*) FROM connector_connections WHERE customer_id = ?", s.customerID()))

	asked := s.runWith(true, true)

	s.Equal(pluginmigrate.Written, s.row(asked, pluginmigrate.KindConnection, login).Action)
	s.Equal(pluginmigrate.Written, s.row(asked, pluginmigrate.KindBinding, config+" agent_plugins").Action)
}

// TestAGrantWhoseRotationIsUnknownIsSkippedUnlessAskedFor: a manifest that does not say
// whether refresh tokens rotate (as sentry, hubspot and shopify do not) is taken as one that
// may, and a login with no refresh token has nothing to rotate, so it moves.
func (s *PluginMigrateSuite) TestAGrantWhoseRotationIsUnknownIsSkippedUnlessAskedFor() {
	config := s.config([]store.PluginEntry{{Name: unknownPlugin}}, nil)
	login := s.loginTo(config, "", unknownPlugin, s.provider.ClientID, s.provider.URL+fakeprovider.PathToken, store.PluginConnected)
	other := s.config([]store.PluginEntry{{Name: unknownPlugin}}, nil)
	accessOnly := s.loginTo(other, "", unknownPlugin, s.provider.ClientID, s.provider.URL+fakeprovider.PathToken, store.PluginConnected)
	_, err := s.store.DB().ExecContext(context.Background(), "UPDATE agent_plugin_connections SET refresh_token = '' WHERE id = ?", accessOnly)
	s.Require().NoError(err)

	report := s.run(true)

	row := s.row(report, pluginmigrate.KindConnection, login)
	s.Equal(pluginmigrate.Skipped, row.Action)
	s.Contains(row.Note, "refresh rotation unknown")
	s.Contains(row.Note, "--include-rotating")
	s.Equal(pluginmigrate.Written, s.row(report, pluginmigrate.KindConnection, accessOnly).Action)

	asked := s.runWith(true, true)

	s.Equal(pluginmigrate.Written, s.row(asked, pluginmigrate.KindConnection, login).Action)
}

// credentialsAfterAnotherRun is the credential store with another run's save landing just
// before this run's: its first Update runs fn twice, under the lock each time, the first time
// as the other run. Two real pgsealed writers, nothing faked.
type credentialsAfterAnotherRun struct {
	core.CredentialStore
	done bool
}

func (c *credentialsAfterAnotherRun) Update(ctx context.Context, ref core.ConnectionRef, fn func(*core.CredentialState, func() error) (bool, error)) error {
	if !c.done {
		c.done = true
		if err := c.CredentialStore.Update(ctx, ref, fn); err != nil {
			return err
		}
	}
	return c.CredentialStore.Update(ctx, ref, fn)
}

// TestAConnectionFinishedWhileARunWaitsIsAlreadyMoved: a run that read the connection pending,
// and finds it finished by another run once it holds the lock, writes nothing and says so.
func (s *PluginMigrateSuite) TestAConnectionFinishedWhileARunWaitsIsAlreadyMoved() {
	config := s.config([]store.PluginEntry{{Name: movedPlugin}}, nil)
	s.pluginClient(config, s.provider.ClientID, s.provider.ClientSecret)
	login := s.login(config, "", s.provider.ClientID)
	options := s.options(false)
	options.Credentials = &credentialsAfterAnotherRun{CredentialStore: options.Credentials}

	report, err := pluginmigrate.Run(context.Background(), options, true)
	s.Require().NoError(err)

	row := s.row(report, pluginmigrate.KindConnection, login)
	s.Equal(pluginmigrate.Exists, row.Action)
	s.Equal("already moved", row.Note)
	s.Zero(report.Count(pluginmigrate.KindConnection, pluginmigrate.Written))
	moved, err := s.store.ConnectorConnection(context.Background(), s.customerID(), pluginmigrate.MovedConnectionID(login))
	s.Require().NoError(err)
	s.Equal(2, moved.Revision, "one save, the other run's")
}

// TestASkippedLoginNamesTheEnvTheConnectorReads: the plugin read its operator client from
// SHAREDCRM_MCP_*, the connector reads SHARED_MCP_*, and nothing set the second, so the row
// says which to set, as for the four Google connectors.
func (s *PluginMigrateSuite) TestASkippedLoginNamesTheEnvTheConnectorReads() {
	config := s.config([]store.PluginEntry{{Name: sharedEnvPlugin}}, nil)
	login := s.loginTo(config, "", sharedEnvPlugin, "plugin-operator-client", s.provider.URL+fakeprovider.PathToken, store.PluginConnected)

	report := s.run(false)

	row := s.row(report, pluginmigrate.KindConnection, login)
	s.Equal(pluginmigrate.Skipped, row.Action)
	s.Contains(row.Note, "issued to client plugin-operator-client")
	s.Contains(row.Note, "the plugin read SHAREDCRM_MCP_CLIENT_ID and the connector reads SHARED_MCP_CLIENT_ID")
}

// TestTwoRunsAtOnceMoveALoginOnce: two --apply runs started together make one connection and
// one binding, and only one of them reports writing each.
func (s *PluginMigrateSuite) TestTwoRunsAtOnceMoveALoginOnce() {
	config := s.config([]store.PluginEntry{{Name: movedPlugin}}, nil)
	s.pluginClient(config, s.provider.ClientID, s.provider.ClientSecret)
	login := s.login(config, "", s.provider.ClientID)
	options := s.options(false)

	reports := make([]pluginmigrate.Report, 2)
	errs := make([]error, 2)
	var wg sync.WaitGroup
	for i := range reports {
		wg.Add(1)
		go func() {
			defer wg.Done()
			reports[i], errs[i] = pluginmigrate.Run(context.Background(), options, true)
		}()
	}
	wg.Wait()

	s.Require().NoError(errors.Join(errs...))
	written := 0
	for _, report := range reports {
		written += report.Count(pluginmigrate.KindConnection, pluginmigrate.Written)
	}
	s.Equal(1, written, "%v", reports)
	s.Equal(1, s.count("SELECT count(*) FROM connector_connections WHERE customer_id = ?", s.customerID()))
	s.Len(s.storedConfig(config).Connectors, 1)
	moved, err := s.store.ConnectorConnection(context.Background(), s.customerID(), pluginmigrate.MovedConnectionID(login))
	s.Require().NoError(err)
	s.Equal(store.ConnectionConnected, moved.Status)
}

// TestNothingConfiguredMovesNothing: an app with no plugin rows is a run with no row and no
// write, real or dry.
func (s *PluginMigrateSuite) TestNothingConfiguredMovesNothing() {
	s.config(nil, nil)
	before := s.connectorRows()

	dry, real := s.run(false), s.run(true)

	s.Empty(dry.Rows)
	s.Empty(real.Rows)
	s.Equal(before, s.connectorRows())
}

// The other plugins of the suite's catalog, each a connector at the fake as crm is: one whose
// refresh tokens rotate, one whose manifest does not say, and one whose operator client is
// read from a shared env.
const (
	rotatingPlugin  = "rotatingcrm"
	unknownPlugin   = "unknowncrm"
	sharedEnvPlugin = "sharedcrm"
)

// seedConnectors stores the suite's built-in connectors at the fake. crm and sharedcrm say
// their refresh tokens do not rotate, so a grant to them moves by default.
func (s *PluginMigrateSuite) seedConnectors() {
	s.seedConnector(movedPlugin, "client:\n  registration: [customer, dcr]\nrefresh:\n  rotating: false\n")
	s.seedConnector(rotatingPlugin, "client:\n  registration: [customer, dcr]\nrefresh:\n  rotating: true\n")
	s.seedConnector(unknownPlugin, "client:\n  registration: [customer, dcr]\n")
	s.seedConnector(sharedEnvPlugin, "client:\n  registration: [operator]\n  env: SHARED\nrefresh:\n  rotating: false\n")
}

// seedConnector stores the built-in connector id at the fake, as the next revision so a rerun
// of the suite against the same database stores its own fake's address.
func (s *PluginMigrateSuite) seedConnector(id, client string) {
	ctx := context.Background()
	revision := 1
	latest, err := s.store.LatestBuiltinConnectorDefinition(ctx, id)
	if err == nil {
		revision = latest.Revision + 1
	}
	manifest := `
id: ` + id + `
revision: ` + strconv.Itoa(revision) + `
name: CRM
endpoints:
  mcp: ` + s.provider.URL + fakeprovider.PathMCP + `
schemes: [oauth2_code]
` + client + `sources:
  - kind: mcp
    endpoint: mcp
`
	s.Require().NoError(s.store.SeedConnectorDefinitions(ctx, fstest.MapFS{id + ".yaml": {Data: []byte(manifest)}}))
}

// catalog is the plugin catalog with the suite's three in it, hosted MCP servers at the fake.
func (s *PluginMigrateSuite) catalog(id string) (plugins.Plugin, bool) {
	if id == movedPlugin || id == rotatingPlugin || id == unknownPlugin || id == sharedEnvPlugin {
		return plugins.Plugin{ID: id, Name: "CRM", URL: s.provider.URL + fakeprovider.PathMCP, Auth: "oauth"}, true
	}
	return plugins.Lookup(id)
}

// clients is the router's lookup of an app's client (ConnectorClients), over the suite's store
// once there is one.
func (s *PluginMigrateSuite) clients(ctx context.Context, ref core.ConnectionRef, m core.ResolvedManifest, registration core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
	return ConnectorClients(s.store, s.sealer, func(string) string { return "" })(ctx, ref, m, registration)
}

// run is router plugins migrate over the test's app, as cmd/router builds it.
func (s *PluginMigrateSuite) run(apply bool) pluginmigrate.Report {
	return s.runWith(apply, false)
}

// runWith is run, moving grants of rotating connectors when includeRotating.
func (s *PluginMigrateSuite) runWith(apply, includeRotating bool) pluginmigrate.Report {
	report, err := pluginmigrate.Run(context.Background(), s.options(includeRotating), apply)
	s.Require().NoError(err)
	return report
}

// options are the command's, over the suite's router and the test's app.
func (s *PluginMigrateSuite) options(includeRotating bool) pluginmigrate.Options {
	credentials, err := pgsealed.New(s.store, s.sealer)
	s.Require().NoError(err)
	transports, err := core.NewTransports(core.TransportsConfig{Resolver: s.resolver, Timeout: suiteConnectorTimeout,
		NewClient: loopbackClients(s.provider.Client())})
	s.Require().NoError(err)
	return pluginmigrate.Options{
		Customer: s.customerID(), IncludeRotating: includeRotating, Store: s.store, Configs: s.configs, Registry: s.connectors,
		Credentials: credentials, Transports: transports,
		HTTP: s.provider.Client(), PublicEndpoint: loopbackOrPublic, Clients: s.clients,
		PluginClients: session.PluginClients(s.store, s.sealer),
		SealClientSecret: func(customerID, connectorID, secret string) ([]byte, int, error) {
			return SealConnectorOAuthClientSecret(s.sealer, customerID, connectorID, secret)
		},
		Catalog: s.catalog,
	}
}

// config is an agent config of the test's app on the connecting model, naming plugins as the
// plugin endpoints write them.
func (s *PluginMigrateSuite) config(agent, user []store.PluginEntry) string {
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "migrate-" + s.utils.uuid(),
		LLM: "connecting/connector-model", AgentPlugins: agent, UserPlugins: user}
	s.Require().NoError(s.configs.CreateAgentConfig(context.Background(), &config))
	return config.ID
}

// pluginClient sets the OAuth client config logs into the plugin with, its secret sealed as the
// plugin endpoint seals it.
func (s *PluginMigrateSuite) pluginClient(config, id, secret string) {
	sealed, err := session.SealPluginClientSecret(s.sealer, plugins.Owner{CustomerID: s.customerID(), ConfigID: config}, movedPlugin, secret)
	s.Require().NoError(err)
	s.Require().NoError(s.store.SavePluginClient(context.Background(), &store.PluginClient{CustomerID: s.customerID(),
		ConfigID: config, PluginID: movedPlugin, ClientID: id, SecretSealed: sealed, SecretKEKVersion: s.sealer.CurrentVersion()}))
}

// login is a connected plugin login to crm on config, the app's for an empty user, holding the
// fake's token, issued to client.
func (s *PluginMigrateSuite) login(config, user, client string) string {
	return s.loginTo(config, user, movedPlugin, client, s.provider.URL+fakeprovider.PathToken, store.PluginConnected)
}

func (s *PluginMigrateSuite) loginTo(config, user, plugin, client, tokenEndpoint, status string) string {
	expires := time.Now().Add(time.Hour).UTC()
	login := store.PluginConnection{CustomerID: s.customerID(), ConfigID: config, PluginID: plugin, UserID: user,
		AccessToken: s.token, RefreshToken: "plugin-refresh-" + s.utils.uuid(), ExpiresAt: &expires, Status: status,
		ClientID: client, TokenEndpoint: tokenEndpoint}
	s.Require().NoError(s.store.UpsertPluginConnection(context.Background(), &login))
	return login.ID
}

// toolRan is the tool_ran frame of the first turn as asks the session for.
func (s *PluginMigrateSuite) toolRan(as *testClient, id string) map[string]any {
	events := s.client.opens("/v1/agents/sessions/" + id + "/events")
	s.Require().Equal(http.StatusOK, as.do(http.MethodPost, "/v1/agents/sessions/"+id+"/respond",
		RespondRequest{Text: "ask the crm", CommandId: pointerTo(s.utils.uuid())}, nil))
	return s.await(events, "tool_ran")
}

// row is the one row of kind whose source holds source.
func (s *PluginMigrateSuite) row(report pluginmigrate.Report, kind, source string) pluginmigrate.Row {
	var found []pluginmigrate.Row
	for _, row := range report.Rows {
		if row.Kind == kind && bytes.Contains([]byte(row.Source), []byte(source)) {
			found = append(found, row)
		}
	}
	s.Require().Len(found, 1, "rows of %s from %s in %v", kind, source, report.Rows)
	return found[0]
}

func (s *PluginMigrateSuite) storedConfig(id string) store.AgentConfig {
	config, err := s.store.AgentConfig(context.Background(), s.customerID(), id)
	s.Require().NoError(err)
	return config
}

// binding is config's one connector binding.
func (s *PluginMigrateSuite) binding(config string) store.ConnectorBinding {
	bindings := s.storedConfig(config).Connectors
	s.Require().Len(bindings, 1)
	return bindings[0]
}

// connectorRows is everything of the test's app in the tables a run writes, as text.
func (s *PluginMigrateSuite) connectorRows() string {
	return s.text(`SELECT
  (SELECT coalesce(string_agg(cc::text, '|' ORDER BY id), '') FROM connector_connections cc WHERE customer_id = ?) || '#' ||
  (SELECT coalesce(string_agg(coc::text, '|' ORDER BY connector_id), '') FROM connector_oauth_clients coc WHERE customer_id = ?) || '#' ||
  (SELECT coalesce(string_agg(id || ':' || connectors::text || ':' || updated_at::text, '|' ORDER BY id), '') FROM agent_configs WHERE customer_id = ?)`,
		s.customerID(), s.customerID(), s.customerID())
}

// pluginTables is every plugin row of the test's app as text, every column: both plugin
// tables, and the plugin columns of its configs. Only the app's rows, since a run moves only
// those and the earlier suites' workers may still touch their own.
func (s *PluginMigrateSuite) pluginTables() string {
	return s.text(`SELECT
  (SELECT coalesce(string_agg(apc::text, '|' ORDER BY id), '') FROM agent_plugin_connections apc WHERE customer_id = ?) || '#' ||
  (SELECT coalesce(string_agg(apcl::text, '|' ORDER BY config_id, plugin_id), '') FROM agent_plugin_clients apcl WHERE customer_id = ?) || '#' ||
  (SELECT coalesce(string_agg(id || ':' || agent_plugins::text || ':' || user_plugins::text || ':' || plugin_events::text, '|' ORDER BY id), '') FROM agent_configs WHERE customer_id = ?)`,
		s.customerID(), s.customerID(), s.customerID())
}

func (s *PluginMigrateSuite) text(query string, args ...any) string {
	var out string
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(), query, args...).Scan(&out))
	return out
}

func (s *PluginMigrateSuite) count(query string, args ...any) int {
	var n int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(), query, args...).Scan(&n))
	return n
}
