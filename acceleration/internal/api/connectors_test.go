//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"testing"

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

func (s *ConnectorsSuite) TestWhatTheRouterReadsToConnectIsNeverShown() {
	// slack.yaml has endpoints, capture and identity rules, a refresh and a rate limit,
	// sources, and client.env naming the operator's variables.
	status, raw := s.serverClient.call(http.MethodGet, "/v1/agents/connectors/slack", nil)
	s.Require().Equal(http.StatusOK, status)

	var shown map[string]any
	s.Require().NoError(json.Unmarshal(raw, &shown))
	for _, withheld := range []string{
		"endpoints", "vars", "authorize_params", "token_params", "identity", "capture",
		"refresh", "rate_limit", "sources", "hooks", "manifest",
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

func (s *ConnectorsSuite) TestACustomConnectorsEndpointIsNeverShown() {
	created := s.create(s.customConnector(s.customID()))

	status, raw := s.serverClient.call(http.MethodGet, "/v1/agents/connectors/"+created.ID, nil)
	s.Require().Equal(http.StatusOK, status)
	s.NotContains(string(raw), "8.8.8.8", "an MCP URL can carry a secret in its path")
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
