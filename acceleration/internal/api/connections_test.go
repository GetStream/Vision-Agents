//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
)

// ConnectionsSuite is about connections: who may make, read, list and delete one, whose
// each is, and what a connector's manifest lets one be created with.
type ConnectionsSuite struct {
	RouterSuite
}

func TestConnectionsSuite(t *testing.T) {
	runSuite(t, new(ConnectionsSuite))
}

// SetupSuite registers oauth2_code, which the built-ins name, and test_key, so a connector
// can allow two schemes; and seeds the built-ins as a router start does.
func (s *ConnectionsSuite) SetupSuite() {
	s.connectors = core.Registry{Schemes: map[string]core.Scheme{
		"oauth2_code": namedScheme("oauth2_code"),
		"test_key":    namedScheme("test_key"),
	}}
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

// SetupTest gives every test an app of its own, so a list holds only what the test made.
func (s *ConnectionsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ConnectionsSuite) TestOnlyTheAppsBackendMayCreateAConnection() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/connections", appOwned("linear"), nil)
	})
}

func (s *ConnectionsSuite) TestOnlyTheAppsBackendMayListConnections() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connections?owner_type=app", nil, nil)
	})
}

func (s *ConnectionsSuite) TestOnlyTheAppsBackendMayReadAConnection() {
	id := s.create(s.serverClient, appOwned("linear")).ID

	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connections/"+id, nil, nil)
	})
}

func (s *ConnectionsSuite) TestOnlyTheAppsBackendMayDeleteAConnection() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		id := s.create(s.serverClient, appOwned("linear")).ID
		return as.do(http.MethodDelete, "/v1/agents/connections/"+id, nil, nil)
	})
}

func (s *ConnectionsSuite) TestNoDeviceMayCreateAUserOwnedConnectionNotEvenForItself() {
	// Turned away before the body is read: every operation the spec does not open is a 403
	// to a device (PostureSuite), so the owner rule below never has to tell a device apart.
	for _, device := range []*testClient{s.anonymousClient, s.guestClient, s.client} {
		status := device.do(http.MethodPost, "/v1/agents/connections", userOwned("linear", device), nil)
		s.Equal(http.StatusForbidden, status, string(device.kind))
	}
}

func (s *ConnectionsSuite) TestABackendCreatesAUserOwnedConnectionForTheUserItActsFor() {
	alice := s.data.createUser()

	created := s.create(s.serverClient.actingFor(alice), userOwned("linear", alice))

	s.Equal(ConnectionOwner{Type: "user", UserID: alice.userID}, created.Owner)
}

func (s *ConnectionsSuite) TestAUserOwnedConnectionFromABackendActingForNobodyIsRefused() {
	alice := s.data.createUser()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections", userOwned("linear", alice))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "X-Stream-User-Id")
}

func (s *ConnectionsSuite) TestAUserOwnedConnectionForAnotherUserIsRefused() {
	alice, bob := s.data.createUser(), s.data.createUser()

	status, failure := s.serverClient.actingFor(alice).failure(http.MethodPost, "/v1/agents/connections", userOwned("linear", bob))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "X-Stream-User-Id")
	s.Empty(s.list(s.serverClient.actingFor(bob), "user", "").Items, "nothing was made in Bob's name")
}

func (s *ConnectionsSuite) TestAUserOwnedConnectionNamesItsUser() {
	alice := s.data.createUser()
	sent := userOwned("linear", alice)
	sent["owner"] = map[string]any{"type": "user"}

	status, failure := s.serverClient.actingFor(alice).failure(http.MethodPost, "/v1/agents/connections", sent)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "owner.user_id")
}

func (s *ConnectionsSuite) TestAnAppOwnedConnectionNamesNoUser() {
	alice := s.data.createUser()
	sent := appOwned("linear")
	sent["owner"] = map[string]any{"type": "app", "user_id": alice.userID}

	status, failure := s.serverClient.actingFor(alice).failure(http.MethodPost, "/v1/agents/connections", sent)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "owner.user_id")
}

func (s *ConnectionsSuite) TestAliceCannotReadBobsConnection() {
	alice, bob := s.data.createUser(), s.data.createUser()
	bobs := s.create(s.serverClient.actingFor(bob), userOwned("linear", bob))

	s.assertNotFoundLikeAMissingOne(s.serverClient.actingFor(alice), http.MethodGet, bobs.ID)
	s.assertNotFoundLikeAMissingOne(s.serverClient, http.MethodGet, bobs.ID)
	s.Equal(bobs, s.get(s.serverClient.actingFor(bob), bobs.ID), "Bob's backend still reaches it")
}

func (s *ConnectionsSuite) TestAliceCannotDeleteBobsConnection() {
	alice, bob := s.data.createUser(), s.data.createUser()
	bobs := s.create(s.serverClient.actingFor(bob), userOwned("linear", bob))

	s.assertNotFoundLikeAMissingOne(s.serverClient.actingFor(alice), http.MethodDelete, bobs.ID)
	s.assertNotFoundLikeAMissingOne(s.serverClient, http.MethodDelete, bobs.ID)
	s.Equal(bobs, s.get(s.serverClient.actingFor(bob), bobs.ID), "it was not deleted")
}

func (s *ConnectionsSuite) TestAliceDoesNotListBobsConnections() {
	alice, bob := s.data.createUser(), s.data.createUser()
	alices := s.create(s.serverClient.actingFor(alice), userOwned("linear", alice))
	bobs := s.create(s.serverClient.actingFor(bob), userOwned("linear", bob))
	apps := s.create(s.serverClient.actingFor(alice), appOwned("linear"))

	s.Equal([]string{alices.ID}, connectionIDs(s.list(s.serverClient.actingFor(alice), "user", "").Items))
	s.Equal([]string{bobs.ID}, connectionIDs(s.list(s.serverClient.actingFor(bob), "user", "").Items))
	s.Equal([]string{apps.ID}, connectionIDs(s.list(s.serverClient.actingFor(bob), "app", "").Items),
		"the app's own are no user's")
}

func (s *ConnectionsSuite) TestListingAUsersConnectionsNeedsTheUserTheBackendActsFor() {
	status, failure := s.serverClient.failure(http.MethodGet, "/v1/agents/connections?owner_type=user", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "X-Stream-User-Id")
}

func (s *ConnectionsSuite) TestAnOwnerTypeThatIsNeitherIsRefused() {
	status, _ := s.serverClient.failure(http.MethodGet, "/v1/agents/connections?owner_type=agent", nil)
	s.Equal(http.StatusBadRequest, status)

	status, _ = s.serverClient.failure(http.MethodGet, "/v1/agents/connections", nil)
	s.Equal(http.StatusBadRequest, status, "owner_type is required")
}

func (s *ConnectionsSuite) TestAnAppOwnedConnectionIsReachedWhoeverTheBackendActsFor() {
	created := s.create(s.serverClient, appOwned("linear"))

	s.Equal(created, s.get(s.serverClient.actingFor(s.data.createUser()), created.ID))
}

func (s *ConnectionsSuite) TestAnotherAppsConnectionIsNeitherReadNorDeleted() {
	id := s.create(s.serverClient, appOwned("linear")).ID

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connections/"+id, nil, nil)
	})
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodDelete, "/v1/agents/connections/"+id, nil, nil)
	})
	s.get(s.serverClient, id)
}

func (s *ConnectionsSuite) TestANewConnectionIsPendingAtTheConnectorsNewestRevision() {
	linear := s.connector("linear")
	sent := appOwned("linear")
	sent["label"] = "  Support workspace  "

	created := s.create(s.serverClient, sent)

	s.NotEmpty(created.ID)
	s.Equal("linear", created.ConnectorID)
	s.Equal(linear.Revision, created.DefinitionRevision)
	s.Equal(ConnectionOwner{Type: "app"}, created.Owner)
	s.Equal("oauth2_code", created.AuthScheme, "Linear's only scheme")
	s.Equal(ConnectionStatus("pending"), created.Status)
	s.Equal(1, created.Revision)
	s.Equal("Support workspace", created.Label, "trimmed")
	s.Empty(created.AccountID)
	s.Empty(created.GrantedScopes)
	s.Nil(created.ExpiresAt)
	s.Equal(created, s.get(s.serverClient, created.ID), "the answer to the create is the stored row")
}

func (s *ConnectionsSuite) TestAConnectionNeverShowsItsStoredCredentials() {
	created := s.create(s.serverClient, appOwned("linear"))

	status, raw := s.serverClient.call(http.MethodGet, "/v1/agents/connections/"+created.ID, nil)
	s.Require().Equal(http.StatusOK, status)
	var shown map[string]any
	s.Require().NoError(json.Unmarshal(raw, &shown))
	for _, withheld := range []string{"credentials", "credentials_sealed", "credentials_kek_version", "customer_id", "cached_tools", "deleted_at"} {
		s.NotContains(shown, withheld)
	}
}

func (s *ConnectionsSuite) TestAnUnknownConnectorIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections", appOwned("custom_nobody"))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "custom_nobody")
}

func (s *ConnectionsSuite) TestAnotherAppsCustomConnectorIsRefusedLikeAnUnknownOne() {
	mine := s.app
	s.useApp(s.data.createApp())
	theirs := s.customConnector()
	s.create(s.serverClient, withScheme(appOwned(theirs, "shop", "acme"), "test_key"))
	s.useApp(mine)

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections",
		withScheme(appOwned(theirs, "shop", "acme"), "test_key"))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "no such connector")
}

func (s *ConnectionsSuite) TestAnInputTheConnectorRequiresIsRefusedByName() {
	id := s.customConnector()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections", withScheme(appOwned(id), "test_key"))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `"shop"`)
	s.Contains(failure, "required")
}

func (s *ConnectionsSuite) TestAnInputTheConnectorDoesNotDeclareIsRefusedByName() {
	id := s.customConnector()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections",
		withScheme(appOwned(id, "shop", "acme", "tenant", "x"), "test_key"))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `"tenant"`)
}

func (s *ConnectionsSuite) TestAnInputOutsideItsEnumOrPatternIsRefusedByName() {
	id := s.customConnector()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections",
		withScheme(appOwned(id, "shop", "acme", "region", "mars"), "test_key"))
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `"region"`)

	status, failure = s.serverClient.failure(http.MethodPost, "/v1/agents/connections",
		withScheme(appOwned(id, "shop", "../admin"), "test_key"))
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `"shop"`)
}

func (s *ConnectionsSuite) TestAnOmittedInputTakesItsDefault() {
	id := s.customConnector()

	created := s.create(s.serverClient, withScheme(appOwned(id, "shop", "acme"), "test_key"))

	s.Equal(map[string]string{"shop": "acme", "region": "us"}, created.Inputs)
}

func (s *ConnectionsSuite) TestAConnectorWithSeveralSchemesNeedsOneNamed() {
	id := s.customConnector()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections", appOwned(id, "shop", "acme"))
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "auth_scheme")

	sent := appOwned(id, "shop", "acme")
	sent["auth_scheme"] = "test_key"
	s.Equal("test_key", s.create(s.serverClient, sent).AuthScheme)
}

func (s *ConnectionsSuite) TestASchemeTheConnectorDoesNotAllowIsRefused() {
	sent := appOwned("linear")
	sent["auth_scheme"] = "test_key"

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections", sent)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "test_key")
}

func (s *ConnectionsSuite) TestASchemeThisDeploymentDoesNotHaveIsRefused() {
	id := s.customConnector()
	sent := appOwned(id, "shop", "acme")
	sent["auth_scheme"] = "test_absent"

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections", sent)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "test_absent")
	s.Contains(failure, "oauth2_code, test_key", "the ones it has")
}

func (s *ConnectionsSuite) TestAFieldTheRequestDoesNotHaveIsRefused() {
	sent := appOwned("linear")
	sent["endpoint"] = "https://8.8.8.8/mcp"

	status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/connections", sent)

	s.Equal(http.StatusBadRequest, status)
}

func (s *ConnectionsSuite) TestADeletedConnectionIsGone() {
	created := s.create(s.serverClient, appOwned("linear"))

	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+created.ID, nil, nil))

	s.assertNotFoundLikeAMissingOne(s.serverClient, http.MethodGet, created.ID)
	s.assertNotFoundLikeAMissingOne(s.serverClient, http.MethodDelete, created.ID)
	s.Empty(s.list(s.serverClient, "app", "").Items)
}

func (s *ConnectionsSuite) TestABackendDeletesItsUsersConnection() {
	alice := s.data.createUser()
	created := s.create(s.serverClient.actingFor(alice), userOwned("linear", alice))

	s.Equal(http.StatusNoContent, s.serverClient.actingFor(alice).do(http.MethodDelete, "/v1/agents/connections/"+created.ID, nil, nil))
	s.assertNotFoundLikeAMissingOne(s.serverClient.actingFor(alice), http.MethodGet, created.ID)
}

func (s *ConnectionsSuite) TestAConnectionAnAgentBindsIsNotDeletedUnlessForced() {
	created := s.create(s.serverClient, appOwned("linear"))
	s.bindFixed(created.ID)

	status, failure := s.serverClient.failure(http.MethodDelete, "/v1/agents/connections/"+created.ID, nil)
	s.Equal(http.StatusConflict, status)
	s.Contains(failure, "force=true")
	kept := s.get(s.serverClient, created.ID)
	s.Len(kept.UsedBy, 1, "and says what binds it")
	kept.UsedBy = created.UsedBy
	s.Equal(created, kept, "it was not deleted")

	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+created.ID+"?force=true", nil, nil))
	s.assertNotFoundLikeAMissingOne(s.serverClient, http.MethodGet, created.ID)
}

func (s *ConnectionsSuite) TestAConnectionSaysWhichAgentConfigsBindIt() {
	created := s.create(s.serverClient, appOwned("linear"))
	other := s.create(s.serverClient, appOwned("linear"))
	config := s.bindFixed(created.ID)

	read := s.get(s.serverClient, created.ID)
	listed := s.list(s.serverClient, "app", "linear")

	s.Equal([]ConnectionUse{{ConfigID: config.Id, ConfigName: config.Name, Binding: "tracker"}}, read.UsedBy)
	for _, connection := range listed.Items {
		switch connection.ID {
		case created.ID:
			s.Equal(read.UsedBy, connection.UsedBy)
		case other.ID:
			s.Empty(connection.UsedBy)
		}
	}
}

func (s *ConnectionsSuite) TestANewConnectionIsUsedByNothing() {
	created := s.create(s.serverClient, appOwned("linear"))

	s.NotNil(created.UsedBy, "an empty list, not an absent one")
	s.Empty(created.UsedBy)
}

func (s *ConnectionsSuite) TestAnUnboundConnectionIsDeletedWithoutForce() {
	bound := s.create(s.serverClient, appOwned("linear"))
	s.bindFixed(bound.ID)
	created := s.create(s.serverClient, appOwned("linear"))

	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+created.ID, nil, nil))
}

func (s *ConnectionsSuite) TestTheListIsNewestFirstAndNarrowsToOneConnector() {
	custom := s.customConnector()
	first := s.create(s.serverClient, appOwned("linear"))
	second := s.create(s.serverClient, withScheme(appOwned(custom, "shop", "acme"), "test_key"))
	third := s.create(s.serverClient, appOwned("linear"))

	s.Equal([]string{third.ID, second.ID, first.ID}, connectionIDs(s.list(s.serverClient, "app", "").Items))
	s.Equal([]string{third.ID, first.ID}, connectionIDs(s.list(s.serverClient, "app", "linear").Items))
}

func (s *ConnectionsSuite) TestPagingWalksTheWholeListWithoutRepeatingOrSkipping() {
	for range 5 {
		s.create(s.serverClient, appOwned("linear"))
	}
	whole := connectionIDs(s.list(s.serverClient, "app", "").Items)
	s.Require().Len(whole, 5)

	var walked []string
	cursor := ""
	for pages := 0; ; pages++ {
		s.Require().Less(pages, len(whole), "paging does not end")
		var listed ConnectionPage
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
			"/v1/agents/connections?owner_type=app&limit=2&cursor="+url.QueryEscape(cursor), nil, &listed))
		s.LessOrEqual(len(listed.Items), 2)
		walked = append(walked, connectionIDs(listed.Items)...)
		if !listed.HasMore {
			s.Nil(listed.NextCursor)
			break
		}
		s.Require().NotNil(listed.NextCursor)
		cursor = *listed.NextCursor
	}
	s.Equal(whole, walked)
}

func (s *ConnectionsSuite) TestACursorThisListDidNotHandOutIsRefused() {
	status, _ := s.serverClient.failure(http.MethodGet, "/v1/agents/connections?owner_type=app&cursor=not-a-cursor", nil)

	s.Equal(http.StatusBadRequest, status)
}

// ConnectionsWithoutConnectorsSuite is a router with connectors off: no scheme registered,
// which is what cmd/router builds when connectors.enabled is false.
type ConnectionsWithoutConnectorsSuite struct {
	RouterSuite
}

func TestConnectionsWithoutConnectorsSuite(t *testing.T) {
	runSuite(t, new(ConnectionsWithoutConnectorsSuite))
}

func (s *ConnectionsWithoutConnectorsSuite) SetupSuite() {
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *ConnectionsWithoutConnectorsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ConnectionsWithoutConnectorsSuite) TestACreateSaysConnectorsAreNotEnabled() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections", appOwned("linear"))

	s.Equal(http.StatusBadRequest, status)
	s.Equal(errConnectorsOff.Message, failure, "not a scheme or an input the caller never chose")
}

func (s *ConnectionsWithoutConnectorsSuite) TestTheListStillAnswers() {
	var listed ConnectionPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections?owner_type=app", nil, &listed))
	s.Empty(listed.Items)
}

// assertNotFoundLikeAMissingOne checks the caller is told what an id nobody used is told,
// status and body, so the answer does not confirm the connection exists. The time each
// took is left out, since no two answers agree on it.
func (s *ConnectionsSuite) assertNotFoundLikeAMissingOne(as *testClient, method, id string) {
	status, answered := as.call(method, "/v1/agents/connections/"+id, nil)
	missingStatus, missing := as.call(method, "/v1/agents/connections/"+s.utils.uuid(), nil)

	s.Equal(http.StatusNotFound, missingStatus)
	s.Equal(missingStatus, status, "%s %s", method, id)
	s.Equal(withoutDuration(missing), withoutDuration(answered), "%s %s", method, id)
}

// customConnector stores one of the app's own connectors with a required input, an input
// with a default and three schemes, one of which the suite does not register, and returns
// its id. No endpoint can ask for inputs, so it
// is written to the store directly.
func (s *ConnectionsSuite) customConnector() string {
	id := "custom_shop" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte(`
id: ` + id + `
revision: 1
name: Shop
endpoints:
  mcp: https://mcp.shop.example/{shop}/mcp
inputs:
  - name: shop
    pattern: "[a-z0-9-]+"
  - name: region
    enum: [us, eu]
    default: us
schemes: [oauth2_code, test_key, test_absent]
client:
  registration: [dcr]
sources:
  - kind: mcp
    endpoint: mcp
`))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return id
}

// bindFixed makes an agent config of the suite's app bind the connection as its fixed one,
// in the shape store.ConnectorConnectionReferenced matches, and returns the config. The
// column is written directly, so these tests do not depend on what the config endpoints
// accept.
func (s *ConnectionsSuite) bindFixed(connectionID string) AgentConfig {
	config := s.data.createAgentConfig()
	_, err := s.store.DB().ExecContext(context.Background(),
		"UPDATE agent_configs SET connectors = ?::jsonb WHERE id = ?",
		`[{"name": "tracker", "connector_id": "linear", "connection": {"type": "fixed", "connection_id": "`+connectionID+`"}}]`,
		config.Id)
	s.Require().NoError(err)
	return config
}

func (s *ConnectionsSuite) connector(id string) Connector {
	var definition Connector
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connectors/"+id, nil, &definition))
	return definition
}

func (s *ConnectionsSuite) create(as *testClient, body map[string]any) Connection {
	var created Connection
	s.Require().Equal(http.StatusCreated, as.do(http.MethodPost, "/v1/agents/connections", body, &created))
	return created
}

func (s *ConnectionsSuite) get(as *testClient, id string) Connection {
	var connection Connection
	s.Require().Equal(http.StatusOK, as.do(http.MethodGet, "/v1/agents/connections/"+id, nil, &connection))
	return connection
}

func (s *ConnectionsSuite) list(as *testClient, ownerType, connectorID string) ConnectionPage {
	query := url.Values{"owner_type": {ownerType}, "limit": {"200"}}
	if connectorID != "" {
		query.Set("connector_id", connectorID)
	}
	var listed ConnectionPage
	s.Require().Equal(http.StatusOK, as.do(http.MethodGet, "/v1/agents/connections?"+query.Encode(), nil, &listed))
	return listed
}

// appOwned is a request for an app-owned connection to connector, with inputs given as
// name, value pairs.
func appOwned(connector string, inputs ...string) map[string]any {
	sent := map[string]any{"connector_id": connector, "owner": map[string]any{"type": "app"}}
	if len(inputs) > 0 {
		values := map[string]string{}
		for i := 0; i+1 < len(inputs); i += 2 {
			values[inputs[i]] = inputs[i+1]
		}
		sent["inputs"] = values
	}
	return sent
}

// userOwned is a request for a connection to connector owned by user.
func userOwned(connector string, user *testClient) map[string]any {
	return map[string]any{
		"connector_id": connector,
		"owner":        map[string]any{"type": "user", "user_id": user.userID},
	}
}

func withScheme(sent map[string]any, scheme string) map[string]any {
	sent["auth_scheme"] = scheme
	return sent
}

func connectionIDs(connections []Connection) []string {
	ids := make([]string, 0, len(connections))
	for _, connection := range connections {
		ids = append(ids, connection.ID)
	}
	return ids
}
