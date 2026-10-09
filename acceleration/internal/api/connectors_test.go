//go:build integration

package api

import (
	"context"
	"encoding/json"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
)

// ConnectorsSuite is about the connector catalog: the built-ins, an app's own custom MCP
// connectors, and what of a manifest a caller is shown.
type ConnectorsSuite struct {
	RouterSuite
}

func TestConnectorsSuite(t *testing.T) {
	runSuite(t, new(ConnectorsSuite))
}

// SetupSuite registers the scheme the built-ins name, and seeds the built-ins as a router
// start does. Seeding is idempotent, so suites running beside this one see the same rows.
func (s *ConnectorsSuite) SetupSuite() {
	s.connectors = core.Registry{Schemes: map[string]core.Scheme{"oauth2_code": namedScheme("oauth2_code")}}
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

// SetupTest gives every test an app of its own, so a list holds the built-ins and only what
// the test made.
func (s *ConnectorsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ConnectorsSuite) TestABuiltInShowsItsSchemesInputsScopesAndClientRegistrations() {
	slack := s.get("slack")

	s.False(slack.Custom)
	s.Equal("Slack", slack.Name)
	s.GreaterOrEqual(slack.Revision, 1)
	s.Equal([]string{"oauth2_code"}, slack.Schemes)
	s.Empty(slack.Inputs, "Slack is connected with nothing but a consent")
	s.Contains(slack.Scopes, "chat:write")
	s.Equal([]ConnectorClientRegistrationMethod{"operator"}, slack.Client.Registration)
	s.Equal(ConnectorClientAuthMethod("client_secret_post"), slack.Client.AuthMethod)
}

// TestAMovedPluginShowsItsSetupSteps: the steps the plugin catalog shows before a client is
// pasted in come with its connector (T58), and a connector whose manifest has none shows none.
func (s *ConnectorsSuite) TestAMovedPluginShowsItsSetupSteps() {
	calendar := s.get("google_calendar")
	calcom := s.get("calcom")

	s.Require().NotNil(calendar.Setup)
	s.Equal("https://console.cloud.google.com/auth/clients", calendar.Setup.URL)
	s.Len(calendar.Setup.Steps, 5)
	s.Equal("Enable the API", calendar.Setup.Steps[0].Title)
	s.Equal([]ConnectorClientRegistrationMethod{"operator", "customer"}, calendar.Client.Registration)
	s.Nil(calcom.Setup)
}

func (s *ConnectorsSuite) TestWhatTheRouterReadsToConnectIsNeverShown() {
	// slack.yaml has endpoints, capture and identity rules, a refresh and a rate limit,
	// sources, and client.env naming the operator's variables.
	status, raw := s.serverClient.call(http.MethodGet, "/v1/agents/connectors/slack", nil)
	s.Require().Equal(http.StatusOK, status)

	var shown map[string]any
	s.Require().NoError(json.Unmarshal(raw, &shown))
	for _, withheld := range []string{
		"endpoints", "vars", "authorize_params", "token_params", "identity", "capture",
		"refresh", "rate_limit", "sources", "hooks", "manifest", "channel",
	} {
		s.NotContains(shown, withheld)
	}
	client, ok := shown["client"].(map[string]any)
	s.Require().True(ok, string(raw))
	s.NotContains(client, "env")
	for _, value := range []string{"SLACK", "mcp.slack.com", "slack.com/api", "$.team.id"} {
		s.NotContains(string(raw), value)
	}
}

// TestACustomConnectorShowsItsEndpointAndABuiltInNone (AI-1046): the app sent the endpoint and
// edits it, so it is read back; a built-in's endpoints stay with the router.
func (s *ConnectorsSuite) TestACustomConnectorShowsItsEndpointAndABuiltInNone() {
	created := s.create(s.customConnector(s.customID()))

	s.Equal("https://8.8.8.8/mcp", created.Endpoint)
	s.Equal("https://8.8.8.8/mcp", s.get(created.ID).Endpoint)
	listed := s.list(created.ID).Items
	s.Require().Len(listed, 1)
	s.Equal("https://8.8.8.8/mcp", listed[0].Endpoint)
	status, raw := s.serverClient.call(http.MethodGet, "/v1/agents/connectors/slack", nil)
	s.Require().Equal(http.StatusOK, status)
	s.NotContains(string(raw), `"endpoint"`)
	for _, builtin := range s.list("").Items {
		if !builtin.Custom {
			s.Empty(builtin.Endpoint, builtin.ID)
		}
	}
}

// TestABuiltInReadsAsBeforeOnARouterWithoutAPublicURL is the control for AI-1046 and AI-1047:
// the keys a built-in is answered with are the ones accelerate at 26051062 answered with,
// probed there with this suite (pr-w7/author-1.md). A built-in has no endpoint to show, and
// without ROUTER_PUBLIC_URL no redirect URI.
func (s *ConnectorsSuite) TestABuiltInReadsAsBeforeOnARouterWithoutAPublicURL() {
	for _, id := range []string{"slack", "linear", "telnyx"} {
		status, raw := s.serverClient.call(http.MethodGet, "/v1/agents/connectors/"+id, nil)
		s.Require().Equal(http.StatusOK, status)
		var shown map[string]any
		s.Require().NoError(json.Unmarshal(raw, &shown))
		s.Equal([]string{"category", "client", "created_at", "custom", "description", "duration", "id", "inputs",
			"name", "revision", "schemes", "scopes"}, slices.Sorted(maps.Keys(shown)), id)
	}
}

func (s *ConnectorsSuite) TestACustomMCPConnectorIsStoredAndReadBack() {
	id := s.customID()
	sent := s.customConnector(id)
	sent["category"] = "  CRM  "
	sent["scopes"] = []string{"crm.read", "crm.write"}
	sent["client"] = map[string]any{"registration": []string{"dcr", "customer"}, "auth_method": "client_secret_basic"}

	created := s.create(sent)
	s.Equal(id, created.ID)
	s.Equal(1, created.Revision)
	s.True(created.Custom)
	s.Equal("CRM", created.Category, "trimmed")
	s.Equal([]string{"oauth2_code"}, created.Schemes)
	s.Equal([]string{"crm.read", "crm.write"}, created.Scopes)
	s.Equal([]ConnectorClientRegistrationMethod{"dcr", "customer"}, created.Client.Registration)
	s.Equal(ConnectorClientAuthMethod("client_secret_basic"), created.Client.AuthMethod)

	s.Equal(created, s.get(id), "the answer to the create is the stored row")
}

func (s *ConnectorsSuite) TestAnIdThatWouldShadowABuiltInIsRefused() {
	before := s.get("slack")

	status, raw := s.serverClient.call(http.MethodPost, "/v1/agents/connectors", s.customConnector("slack"))

	s.Equal(http.StatusBadRequest, status)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(raw, &answered), "the error envelope: %s", raw)
	s.Contains(answered.Error.Message, "custom_")
	s.Equal(ErrorTypeInvalidRequest, answered.Error.Type)
	s.Equal(before, s.get("slack"), "the built-in is untouched")
}

func (s *ConnectorsSuite) TestAnEndpointOnAPrivateNetworkIsRefused() {
	for _, endpoint := range []string{
		"https://127.0.0.1/mcp",       // loopback
		"https://10.0.0.7/mcp",        // private-use, RFC 1918
		"https://169.254.169.254/mcp", // link-local, where cloud metadata servers answer
		"https://[::1]/mcp",           // IPv6 loopback
	} {
		sent := s.customConnector(s.customID())
		sent["endpoint"] = endpoint

		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connectors", sent)
		s.Equal(http.StatusBadRequest, status, endpoint)
		s.Contains(failure, "endpoint", endpoint)
	}
}

func (s *ConnectorsSuite) TestAnEndpointThatDoesNotResolveIsRefusedLikeAPrivateOne() {
	answers := map[string]string{}
	for _, endpoint := range []string{
		"https://router-probe.invalid/mcp", // .invalid never resolves, RFC 6761 section 6.4
		"https://10.0.0.7/mcp",             // private-use, RFC 1918
	} {
		sent := s.customConnector(s.customID())
		sent["endpoint"] = endpoint

		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connectors", sent)
		s.Equal(http.StatusBadRequest, status, endpoint)
		answers[endpoint] = failure
	}
	s.Equal(answers["https://10.0.0.7/mcp"], answers["https://router-probe.invalid/mcp"],
		"a caller cannot tell which names the router's resolver knows")
}

func (s *ConnectorsSuite) TestAnEndpointThatIsNotPlainHTTPSIsRefused() {
	for _, endpoint := range []string{
		"http://8.8.8.8/mcp",
		"https://8.8.8.8/mcp?token=secret",
		"https://user:secret@8.8.8.8/mcp",
		"https://8.8.8.8/{tenant}/mcp",
	} {
		sent := s.customConnector(s.customID())
		sent["endpoint"] = endpoint

		status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/connectors", sent)
		s.Equal(http.StatusBadRequest, status, endpoint)
	}
}

func (s *ConnectorsSuite) TestASchemeThisDeploymentDoesNotHaveIsRefused() {
	sent := s.customConnector(s.customID())
	sent["schemes"] = []string{"aws_sigv4"}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connectors", sent)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "aws_sigv4")
}

func (s *ConnectorsSuite) TestAnOperatorClientIsRefusedForACustomConnector() {
	sent := s.customConnector(s.customID())
	sent["client"] = map[string]any{"registration": []string{"operator"}}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connectors", sent)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "operator")
}

func (s *ConnectorsSuite) TestAnOAuthConnectorWithoutClientRegistrationsIsRefused() {
	sent := s.customConnector(s.customID())
	delete(sent, "client")

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connectors", sent)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "client.registration")
}

func (s *ConnectorsSuite) TestARepeatedScopeOrClientRegistrationIsRefused() {
	for name, change := range map[string]func(map[string]any){
		"scope":  func(sent map[string]any) { sent["scopes"] = []string{"crm.read", "crm.read"} },
		"source": func(sent map[string]any) { sent["client"] = map[string]any{"registration": []string{"dcr", "dcr"}} },
	} {
		sent := s.customConnector(s.customID())
		change(sent)

		status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/connectors", sent)
		s.Equal(http.StatusBadRequest, status, name)
	}
}

func (s *ConnectorsSuite) TestAFieldTheRequestDoesNotHaveIsRefused() {
	for field, value := range map[string]any{
		"hooks":     map[string]string{"before_complete": "shopify.callback_hmac"},
		"endpoints": map[string]string{"token": "https://8.8.8.8/token"},
	} {
		sent := s.customConnector(s.customID())
		sent[field] = value

		status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/connectors", sent)
		s.Equal(http.StatusBadRequest, status, field)
	}
	sent := s.customConnector(s.customID())
	sent["client"] = map[string]any{"registration": []string{"dcr"}, "env": "SLACK"}
	status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/connectors", sent)
	s.Equal(http.StatusBadRequest, status, "client.env")
}

func (s *ConnectorsSuite) TestAScopeThatIsNotAnRFC6749ScopeTokenIsRefused() {
	sent := s.customConnector(s.customID())
	sent["scopes"] = []string{"read write"}

	status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/connectors", sent)

	s.Equal(http.StatusBadRequest, status)
}

func (s *ConnectorsSuite) TestSendingTheSameConnectorAgainChangesNothingAndAChangeIsTheNextRevision() {
	id := s.customID()
	s.Equal(1, s.create(s.customConnector(id)).Revision)
	s.Equal(1, s.create(s.customConnector(id)).Revision, "the same definition again")

	changed := s.customConnector(id)
	changed["name"] = "Renamed"
	s.Equal(2, s.create(changed).Revision)
	s.Equal("Renamed", s.get(id).Name)
}

func (s *ConnectorsSuite) TestAnotherAppsCustomConnectorIsNeitherListedNorRead() {
	id := s.customID()
	s.create(s.customConnector(id))
	stranger := s.data.backendOfAnotherApp()

	var theirs ConnectorPage
	s.Require().Equal(http.StatusOK, stranger.do(http.MethodGet, "/v1/agents/connectors?limit=200", nil, &theirs))
	s.NotContains(connectorIDs(theirs.Items), id)
	s.Contains(connectorIDs(theirs.Items), "slack", "the built-ins are everybody's")
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connectors/"+id, nil, nil)
	})
}

func (s *ConnectorsSuite) TestAnotherAppMayUseTheSameCustomIdWithoutTouchingThisOne() {
	id := s.customID()
	s.create(s.customConnector(id))

	theirs := s.customConnector(id)
	theirs["name"] = "Theirs"
	var created Connector
	s.Require().Equal(http.StatusOK,
		s.data.backendOfAnotherApp().do(http.MethodPost, "/v1/agents/connectors", theirs, &created))
	s.Equal(1, created.Revision)

	s.Equal("Our CRM", s.get(id).Name)
	s.Equal(1, s.get(id).Revision)
}

func (s *ConnectorsSuite) TestTheListIsTheBuiltInsThenTheAppsOwnEachById() {
	second, first := "custom_b"+s.suffix(), "custom_a"+s.suffix()
	s.create(s.customConnector(second))
	s.create(s.customConnector(first))

	listed := s.list("")
	s.False(listed.HasMore)
	s.Nil(listed.NextCursor)
	custom := slices.IndexFunc(listed.Items, func(d Connector) bool { return d.Custom })
	s.Require().Positive(custom, "the built-ins come first")
	builtIns := connectorIDs(listed.Items[:custom])
	s.True(slices.IsSorted(builtIns), "%v", builtIns)
	s.Contains(builtIns, "linear")
	s.Contains(builtIns, "slack")
	s.Equal([]string{first, second}, connectorIDs(listed.Items[custom:]))
}

func (s *ConnectorsSuite) TestPagingWalksTheWholeListWithoutRepeatingOrSkipping() {
	for range 3 {
		s.create(s.customConnector(s.customID()))
	}
	whole := connectorIDs(s.list("").Items)

	var walked []string
	cursor := ""
	for pages := 0; ; pages++ {
		s.Require().Less(pages, len(whole), "paging does not end")
		var listed ConnectorPage
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
			"/v1/agents/connectors?limit=2&cursor="+url.QueryEscape(cursor), nil, &listed))
		s.LessOrEqual(len(listed.Items), 2)
		walked = append(walked, connectorIDs(listed.Items)...)
		if !listed.HasMore {
			s.Nil(listed.NextCursor)
			break
		}
		s.Require().NotNil(listed.NextCursor)
		cursor = *listed.NextCursor
	}
	s.Equal(whole, walked)
}

func (s *ConnectorsSuite) TestACursorThisListDidNotHandOutIsRefused() {
	status, _ := s.serverClient.failure(http.MethodGet, "/v1/agents/connectors?cursor=not-a-cursor", nil)

	s.Equal(http.StatusBadRequest, status)
}

func (s *ConnectorsSuite) TestTheSearchMatchesTheIdNameCategoryOrDescriptionIgnoringCase() {
	id := s.customID()
	sent := s.customConnector(id)
	sent["category"] = "Ticketing"
	sent["description"] = "Reads the helpdesk queue."
	s.create(sent)

	for _, q := range []string{id, "OUR crm", "ticketing", "HELPDESK"} {
		s.Equal([]string{id}, connectorIDs(s.list(q).Items), q)
	}
	s.Contains(connectorIDs(s.list("slack").Items), "slack", "a built-in is searched too")
	s.Empty(s.list("CRM Ticketing").Items, "the end of the name and the start of the category are two fields")
}

func (s *ConnectorsSuite) TestTheSearchReadsTheNewestRevisionOnly() {
	id := s.customID()
	s.create(s.customConnector(id))
	renamed := s.customConnector(id)
	renamed["name"] = "Ledger"
	s.create(renamed)

	s.Empty(s.list("Our CRM").Items, "revision 1 was called that, and is not the connector any more")
	s.Equal([]string{id}, connectorIDs(s.list("ledger").Items))
}

func (s *ConnectorsSuite) TestASearchForAWildcardMatchesOnlyItself() {
	s.create(s.customConnector(s.customID()))

	s.Empty(s.list("%").Items, "nothing has a percent sign in it")
	s.Empty(s.list("Our_CRM").Items, "as a wildcard, _ would match the space in Our CRM")
}

func (s *ConnectorsSuite) TestAnUnknownConnectorIsNotFound() {
	status, _ := s.serverClient.failure(http.MethodGet, "/v1/agents/connectors/custom_nobody", nil)

	s.Equal(http.StatusNotFound, status)
}

func (s *ConnectorsSuite) TestOnlyTheAppsBackendMayListConnectors() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connectors", nil, nil)
	})
}

func (s *ConnectorsSuite) TestOnlyTheAppsBackendMayReadAConnector() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connectors/slack", nil, nil)
	})
}

func (s *ConnectorsSuite) TestOnlyTheAppsBackendMayAddAConnector() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/connectors", s.customConnector(s.customID()), nil)
	})
}

func (s *ConnectorsSuite) TestOnlyTheAppsBackendMayDeleteAConnector() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodDelete, "/v1/agents/connectors/"+s.create(s.customConnector(s.customID())).ID, nil, nil)
	})
}

func (s *ConnectorsSuite) TestAnUnknownOrBuiltInConnectorIsNotFoundToDelete() {
	unknown, missing := s.serverClient.failure(http.MethodDelete, "/v1/agents/connectors/custom_nobody", nil)
	builtin, answered := s.serverClient.failure(http.MethodDelete, "/v1/agents/connectors/slack?force=true", nil)

	s.Equal(http.StatusNotFound, unknown)
	s.Equal(http.StatusNotFound, builtin)
	s.Equal(missing, answered, "one answer for both")
	s.Equal("Slack", s.get("slack").Name, "the built-in is untouched")
}

func (s *ConnectorsSuite) TestAnotherAppCannotDeleteTheAppsConnector() {
	id := s.create(s.customConnector(s.customID())).ID

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodDelete, "/v1/agents/connectors/"+id+"?force=true", nil, nil)
	})
	s.Equal(id, s.get(id).ID)
}

func (s *ConnectorsSuite) TestAnUnusedConnectorIsDeletedWithItsOAuthClientAndMayBeCreatedAgain() {
	sent := s.customConnector(s.customID())
	sent["client"] = map[string]any{"registration": []string{"customer"}}
	id := s.create(sent).ID
	sent["name"] = "Our CRM, renamed"
	s.Require().Equal(2, s.create(sent).Revision)
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPut, "/v1/agents/connectors/"+id+"/oauth-client",
		map[string]any{"client_id": "crm-client", "client_secret": "crm-secret"}, nil))

	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+id, nil, nil))

	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodGet, "/v1/agents/connectors/"+id, nil, nil))
	s.NotContains(connectorIDs(s.list("").Items), id)
	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodGet, "/v1/agents/connectors/"+id+"/oauth-client", nil, nil),
		"the client and its sealed secret went with it")
	s.Equal(1, s.create(sent).Revision, "created again, it starts over")
}

func (s *ConnectorsSuite) TestAConnectorAConnectionUsesIsRefusedNamingTheConnection() {
	id := s.create(s.customConnector(s.customID())).ID
	var connection Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned(id), &connection))

	status, raw := s.serverClient.call(http.MethodDelete, "/v1/agents/connectors/"+id, nil)

	s.Equal(http.StatusConflict, status)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(raw, &answered), "the error envelope: %s", raw)
	s.Equal(ErrorTypeConflict, answered.Error.Type)
	s.Equal("conflict", answered.Error.Code)
	s.Equal(id+" is used by connections "+connection.ID+": delete or unbind them first, or delete with force=true",
		answered.Error.Message)
	s.Equal(id, s.get(id).ID)
	s.Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+connection.ID, nil, nil))
}

func (s *ConnectorsSuite) TestAConnectorAConfigBindsIsRefusedNamingTheBinding() {
	id := s.create(s.customConnector(s.customID())).ID
	binding := sessionSlack("crm")
	binding["connector_id"] = id
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support-" + s.suffix(), "connectors": []map[string]any{binding}}, &config))

	status, failure := s.serverClient.failure(http.MethodDelete, "/v1/agents/connectors/"+id, nil)

	s.Equal(http.StatusConflict, status)
	s.Equal(id+` is used by agent config bindings "`+config.Name+`" as crm: delete or unbind them first, or delete with force=true`, failure)
}

func (s *ConnectorsSuite) TestARefusalNamesTenUsersOfEachKindAndCountsTheRest() {
	id := s.create(s.customConnector(s.customID())).ID
	var ids []string
	for range usesNamed + 2 {
		var connection Connection
		s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned(id), &connection))
		ids = append(ids, connection.ID)
	}

	status, failure := s.serverClient.failure(http.MethodDelete, "/v1/agents/connectors/"+id, nil)

	s.Equal(http.StatusConflict, status)
	s.Equal(id+" is used by connections "+strings.Join(ids[:usesNamed], ", ")+
		" and 2 more: delete or unbind them first, or delete with force=true", failure)
}

func (s *ConnectorsSuite) TestAForcedDeleteDeletesTheConnectionsAndLeavesTheBindings() {
	id := s.create(s.customConnector(s.customID())).ID
	var connection Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned(id), &connection))
	binding := sessionSlack("crm")
	binding["connector_id"] = id
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support-" + s.suffix(), "connectors": []map[string]any{binding}}, &config))

	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+id+"?force=true", nil, nil))

	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodGet, "/v1/agents/connectors/"+id, nil, nil))
	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+connection.ID, nil, nil))
	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+config.Id, nil, &read))
	s.Require().Len(value(read.Connectors), 1, "left in place, as a forced connection delete leaves its binding")
	s.Equal(id, value(read.Connectors)[0].ConnectorId)
}

// TestAConnectionToAConnectorDeletedMeanwhileIsRefusedAsUnknown: the create read the connector
// before the delete committed and stores the connection after, so the store finds it gone.
func (s *ConnectorsSuite) TestAConnectionToAConnectorDeletedMeanwhileIsRefusedAsUnknown() {
	id := s.create(s.customConnector(s.customID())).ID

	status, failure := s.duringADeleteOf(id, func() (int, string) {
		return s.serverClient.failure(http.MethodPost, "/v1/agents/connections", appOwned(id))
	})

	s.Equal(http.StatusBadRequest, status)
	s.Equal(`no such connector: "`+id+`"`, failure)
}

func (s *ConnectorsSuite) TestAnOAuthClientForAConnectorDeletedMeanwhileIsNotFound() {
	sent := s.customConnector(s.customID())
	sent["client"] = map[string]any{"registration": []string{"customer"}}
	id := s.create(sent).ID

	status, failure := s.duringADeleteOf(id, func() (int, string) {
		return s.serverClient.failure(http.MethodPut, "/v1/agents/connectors/"+id+"/oauth-client",
			map[string]any{"client_id": "crm-client", "client_secret": "crm-secret"})
	})

	s.Equal(http.StatusNotFound, status)
	s.Equal("no such connector", failure)
}

func (s *ConnectorsSuite) TestABindingToAConnectorDeletedMeanwhileIsRefused() {
	id := s.create(s.customConnector(s.customID())).ID
	binding := sessionSlack("crm")
	binding["connector_id"] = id

	status, failure := s.duringADeleteOf(id, func() (int, string) {
		return s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
			map[string]any{"name": "support-" + s.suffix(), "connectors": []map[string]any{binding}})
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, id)
}

// duringADeleteOf makes call while a delete of the app's connector id holds its lock on the
// definition: a SHARE lock on connector_oauth_clients holds the delete at its DELETE there,
// which comes after that lock (store.DeleteConnectorDefinition). The call is let go once it
// waits for the definition, and it answers after the delete committed. Suites beside this one
// may wait on the table meanwhile; the waits counted are the delete's and the call's alone.
func (s *ConnectorsSuite) duringADeleteOf(id string, call func() (int, string)) (int, string) {
	held, err := s.store.DB().BeginTx(context.Background(), nil)
	s.Require().NoError(err)
	defer held.Rollback() //nolint:errcheck // after the commit below there is nothing to roll back
	_, err = held.ExecContext(context.Background(), "LOCK TABLE connector_oauth_clients IN SHARE MODE")
	s.Require().NoError(err)
	deleted := make(chan int, 1)
	go func() { deleted <- s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+id, nil, nil) }()
	// The two seconds and the ten milliseconds are assertTheWaitEnded's (internal/store/credentials_test.go).
	s.Require().Eventually(func() bool { return s.waitingFor("DELETE FROM connector_oauth_clients", id) == 1 },
		2*time.Second, 10*time.Millisecond, "the delete waits at its DELETE")
	type answer struct {
		status  int
		failure string
	}
	answered := make(chan answer, 1)
	go func() {
		status, failure := call()
		answered <- answer{status, failure}
	}()
	s.Require().Eventually(func() bool { return s.waitingFor("FOR KEY SHARE", id) == 1 },
		2*time.Second, 10*time.Millisecond, "the call waits for the delete's lock on the definition")
	s.Require().NoError(held.Commit())
	s.Require().Equal(http.StatusNoContent, <-deleted)
	got := <-answered
	return got.status, got.failure
}

// waitingFor counts the statements in this database waiting for a lock whose text holds both
// statement and the connector id (https://www.postgresql.org/docs/current/monitoring-stats.html#WAIT-EVENT-TABLE).
func (s *ConnectorsSuite) waitingFor(statement, id string) int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(), `
SELECT count(*) FROM pg_stat_activity
WHERE datname = current_database() AND wait_event_type = 'Lock'
  AND strpos(query, ?) > 0 AND strpos(query, ?) > 0`, statement, id).Scan(&count))
	return count
}

// customConnector is a custom MCP connector the router accepts. Its endpoint is a public IP
// literal, so checking it resolves nothing.
func (s *ConnectorsSuite) customConnector(id string) map[string]any {
	return map[string]any{
		"id":       id,
		"name":     "Our CRM",
		"endpoint": "https://8.8.8.8/mcp",
		"schemes":  []string{"oauth2_code"},
		"client":   map[string]any{"registration": []string{"dcr"}},
	}
}

// customID is a custom connector id nothing else has.
func (s *ConnectorsSuite) customID() string { return "custom_t" + s.suffix() }

// suffix is a fresh UUID written as an id may have it.
func (s *ConnectorsSuite) suffix() string { return strings.ReplaceAll(s.utils.uuid(), "-", "") }

func (s *ConnectorsSuite) create(body map[string]any) Connector {
	var created Connector
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/connectors", body, &created))
	return created
}

func (s *ConnectorsSuite) get(id string) Connector {
	var read Connector
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connectors/"+id, nil, &read))
	return read
}

// list is one page of everything, which fits while the test's app has a few of its own.
func (s *ConnectorsSuite) list(q string) ConnectorPage {
	var listed ConnectorPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/connectors?limit=200&q="+url.QueryEscape(q), nil, &listed))
	return listed
}

func connectorIDs(definitions []Connector) []string {
	ids := make([]string, 0, len(definitions))
	for _, definition := range definitions {
		ids = append(ids, definition.ID)
	}
	return ids
}
