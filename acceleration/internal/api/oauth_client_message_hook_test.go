//go:build integration

package api

import (
	"context"
	"net/http"
	"strconv"
	"strings"
	"testing"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ownSlackApp is the app's own Slack app put through the oauth-client PUT: its client, and the
// provider app with the secret Slack signs its events with.
func (s *messageHookHarness) ownSlackApp() ConnectorOAuthClientRequest {
	return ConnectorOAuthClientRequest{
		ClientID: "synthetic-client-" + s.utils.uuid(), ClientSecret: "synthetic-client-secret",
		ProviderAppID: "A" + strings.ToUpper(strings.ReplaceAll(s.utils.uuid(), "-", ""))[20:],
		SigningSecret: "synthetic-signing-" + s.utils.uuid(),
	}
}

// putOwnApp puts the app's own client for a connector, and takes it back when the test ends,
// since a provider app serves one customer.
func (s *messageHookHarness) putOwnApp(connector string, sent ConnectorOAuthClientRequest) (int, []byte) {
	status, payload := s.serverClient.call(http.MethodPut, oauthClientPath(connector), sent)
	s.T().Cleanup(func() { s.serverClient.do(http.MethodDelete, oauthClientPath(connector), nil, nil) })
	return status, payload
}

// warnings are the startup check's warnings about the test customer's app, logged since before.
func (s *messageHookHarness) warnings(before int) string {
	var about []string
	for _, line := range strings.Split(s.logged.String()[before:], "\n") {
		if strings.Contains(line, "stream_app="+strconv.FormatInt(s.appID(), 10)+" ") && strings.Contains(line, "level=WARN") {
			about = append(about, line)
		}
	}
	return strings.Join(about, "\n")
}

// The staging incident of 2026-10-10: the app's backend put its own Slack app through the
// oauth-client PUT, which pointed no hook, so the messages the bridge wrote reached another
// router. The PUT points the pinned app's message hook here, as the provider app PUTs do.
func (s *ProviderAppMessageHookSuite) TestPuttingTheAppsOwnSlackAppPointsThePinnedAppsMessageHook() {
	status, payload := s.putOwnApp("slack_bot", s.ownSlackApp())

	s.Require().Equal(http.StatusCreated, status, string(payload))
	record, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.Require().NoError(err)
	s.Equal(s.appID(), record.StreamAppPK)
	s.Equal([]getstream.EventHook{pointed(s.hookURL(providerAppPublicURL))}, s.chat.EventHooks(s.apiKey))
	s.Zero(s.asked(suiteStreamKey, http.MethodPatch, "/api/v2/app"), "the deployment's own app is left alone")
}

// A client with no provider app is no app whose events the bridge reads: nothing is asked of
// Stream, as on base.
func (s *ProviderAppMessageHookSuite) TestAnOAuthClientWithoutAProviderAppPointsNoHook() {
	sent := s.ownSlackApp()
	sent.ProviderAppID, sent.SigningSecret = "", ""

	status, payload := s.putOwnApp("slack_bot", sent)

	s.Require().Equal(http.StatusCreated, status, string(payload))
	s.Zero(s.asked(s.apiKey, http.MethodPatch, "/api/v2/app"))
	s.Empty(s.chat.EventHooks(s.apiKey))
}

// github has no channel, so a provider app id on its client is no app whose messages the
// bridge writes: nothing is asked of Stream, as on base.
func (s *ProviderAppMessageHookSuite) TestAProviderAppOfAConnectorWithoutAChannelPointsNoHook() {
	status, payload := s.putOwnApp("github", ConnectorOAuthClientRequest{
		ClientID: "synthetic-client-" + s.utils.uuid(), ClientSecret: "synthetic-client-secret", ProviderAppID: "synthetic-github-app",
	})

	s.Require().Equal(http.StatusCreated, status, string(payload))
	s.Zero(s.asked(s.apiKey, http.MethodPatch, "/api/v2/app"))
	s.Empty(s.chat.EventHooks(s.apiKey))
	before := len(s.logged.String())
	s.router.WarnWithoutMessageHooks(context.Background())
	s.Empty(s.warnings(before), "nor is its app's hook checked at startup")
}

// Stream refusing fails the PUT with the provider app PUTs' 503 and keeps the client; a PUT
// again points the hook.
func (s *ProviderAppMessageHookSuite) TestAHookStreamRefusesFailsTheOAuthClientPutAndAPutAgainPointsIt() {
	sent := s.ownSlackApp()
	s.answering(true)

	status, payload := s.putOwnApp("slack_bot", sent)

	s.Equal(http.StatusServiceUnavailable, status, string(payload))
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.Require().NoError(err, "the client is kept")
	s.answering(false)
	status, payload = s.putOwnApp("slack_bot", sent)
	s.Equal(http.StatusOK, status, string(payload))
	s.Equal([]getstream.EventHook{pointed(s.hookURL(providerAppPublicURL))}, s.chat.EventHooks(s.apiKey))
}

// The startup check says nothing of an app whose hook delivers here.
func (s *ProviderAppMessageHookSuite) TestTheStartupCheckSaysNothingOfAPinnedAppThatDeliversHere() {
	status, payload := s.putOwnApp("slack_bot", s.ownSlackApp())
	s.Require().Equal(http.StatusCreated, status, string(payload))
	before := len(s.logged.String())

	s.router.WarnWithoutMessageHooks(context.Background())

	s.Empty(s.warnings(before))
}

// The staging case after the fact: an app whose hook was pointed elsewhere, or that was put
// before the PUT pointed hooks, is warned about once at startup, naming the customer, the app
// and the hook it needs.
func (s *ProviderAppMessageHookSuite) TestTheStartupCheckWarnsOfAPinnedAppWhoseHookIsElsewhere() {
	sent := s.ownSlackApp()
	status, payload := s.putOwnApp("slack_bot", sent)
	s.Require().Equal(http.StatusCreated, status, string(payload))
	client, err := getstream.NewClient(s.apiKey, s.secret, getstream.WithBaseUrl(s.chat.URL))
	s.Require().NoError(err)
	_, err = client.UpdateApp(context.Background(), &getstream.UpdateAppRequest{EventHooks: []getstream.EventHook{
		pointed("https://another-router.example" + chat.MessageHookPath),
	}})
	s.Require().NoError(err)
	before := len(s.logged.String())
	gets, patches := s.asked(s.apiKey, http.MethodGet, "/api/v2/app"), s.asked(s.apiKey, http.MethodPatch, "/api/v2/app")

	s.router.WarnWithoutMessageHooks(context.Background())

	warned := s.warnings(before)
	s.Equal(1, strings.Count(warned, "level=WARN"), warned)
	s.Contains(warned, `msg="stream: a provider app's Stream app sends no new message to this router`)
	s.Contains(warned, "customer="+s.customerID())
	s.Contains(warned, "connector=slack_bot provider_app="+sent.ProviderAppID)
	s.Contains(warned, "hook="+s.hookURL(providerAppPublicURL))
	s.NotContains(warned, s.secret)
	s.NotContains(warned, sent.SigningSecret)
	s.Equal(gets+1, s.asked(s.apiKey, http.MethodGet, "/api/v2/app"), "one read")
	s.Equal(patches, s.asked(s.apiKey, http.MethodPatch, "/api/v2/app"), "and nothing changed")
}

// A record pinned to the deployment's own app is the operator's to point, as pointMessageHook
// leaves it: the startup check asks nothing of it.
func (s *ProviderAppMessageHookSuite) TestTheStartupCheckLeavesAProviderAppPinnedToTheDeploymentsApp() {
	s.useApp(s.data.createApp())
	s.Require().Equal(http.StatusCreated, s.operatorApp())
	record, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.Require().NoError(err)
	s.Require().Equal(int64(suiteStreamApp), record.StreamAppPK)
	asked := s.asked(suiteStreamKey, http.MethodGet, "/api/v2/app")
	before := len(s.logged.String())

	s.router.WarnWithoutMessageHooks(context.Background())

	s.Equal(asked, s.asked(suiteStreamKey, http.MethodGet, "/api/v2/app"))
	s.NotContains(s.logged.String()[before:], "stream_app="+strconv.Itoa(suiteStreamApp)+" ")
}

// Without ROUTER_PUBLIC_URL the oauth-client PUT points no hook and warns once, as the provider
// app PUTs do, and the startup check has no URL to check for.
func (s *ProviderAppMessageHookWithoutPublicURLSuite) TestPuttingTheAppsOwnSlackAppPointsNoHookAndWarnsOnce() {
	before := strings.Count(s.logged.String(), "ROUTER_PUBLIC_URL is not set")
	asked := len(s.chat.Requests(s.apiKey))

	status, payload := s.putOwnApp("slack_bot", s.ownSlackApp())
	s.router.WarnWithoutMessageHooks(context.Background())

	s.Equal(http.StatusCreated, status, string(payload))
	s.Empty(s.chat.EventHooks(s.apiKey))
	s.Len(s.chat.Requests(s.apiKey), asked, "nothing is asked of the customer's Stream app")
	s.Equal(before+1, strings.Count(s.logged.String(), "ROUTER_PUBLIC_URL is not set"))
}

// Connectors off, the control: the oauth-client PUT answers as on base (probed on 599298c6:
// 400 not_configured), and the startup check asks nothing of Stream, even of an app a provider
// app stored while connectors were on is pinned to.
func (s *ProviderAppMessageHookOffSuite) TestTheOAuthClientPutAnswersAsBeforeAndAsksNothingOfStream() {
	// A provider app stored while connectors were on, pinned to the customer's app.
	_, err := s.store.PutConnectorOAuthClient(context.Background(), &store.ConnectorOAuthClient{
		CustomerID: s.customerID(), ConnectorID: "slack_bot", Registration: core.ClientCustomer,
		ClientID: "synthetic-client", ProviderAppID: s.ownSlackApp().ProviderAppID, StreamAppPK: s.appID(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() {
		s.Require().NoError(s.store.DeleteConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot", core.ClientCustomer))
	})
	asked := len(s.chat.Requests(s.apiKey))

	status, payload := s.serverClient.call(http.MethodPut, oauthClientPath("slack_bot"), s.ownSlackApp())
	s.router.WarnWithoutMessageHooks(context.Background())

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), `"code":"not_configured"`)
	s.Contains(string(payload), "OAuth clients cannot be stored: connectors are not enabled on this deployment")
	s.Len(s.chat.Requests(s.apiKey), asked, "nothing is asked of the customer's Stream app")
	s.Empty(s.chat.EventHooks(s.apiKey))
}

// Two provider apps pinned to one Stream app whose hook delivers elsewhere: the startup check
// reads the app once and warns once.
func (s *ProviderAppMessageHookSuite) TestTheStartupCheckReadsAnAppOnce() {
	status, payload := s.putOwnApp("slack_bot", s.ownSlackApp())
	s.Require().Equal(http.StatusCreated, status, string(payload))
	_, err := s.store.PutConnectorOAuthClient(context.Background(), &store.ConnectorOAuthClient{
		CustomerID: s.customerID(), ConnectorID: "linq", Registration: core.ClientCustomer,
		ClientID: "synthetic-linq-client", ProviderAppID: "synthetic-linq-" + s.utils.uuid(), StreamAppPK: s.appID(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() {
		_ = s.store.DeleteConnectorOAuthClient(context.Background(), s.customerID(), "linq", core.ClientCustomer)
	})
	client, err := getstream.NewClient(s.apiKey, s.secret, getstream.WithBaseUrl(s.chat.URL))
	s.Require().NoError(err)
	_, err = client.UpdateApp(context.Background(), &getstream.UpdateAppRequest{EventHooks: []getstream.EventHook{
		pointed("https://another-router.example" + chat.MessageHookPath),
	}})
	s.Require().NoError(err)
	before := len(s.logged.String())
	gets := s.asked(s.apiKey, http.MethodGet, "/api/v2/app")

	s.router.WarnWithoutMessageHooks(context.Background())

	s.Equal(gets+1, s.asked(s.apiKey, http.MethodGet, "/api/v2/app"), "one read per app")
	s.Equal(1, strings.Count(s.warnings(before), "level=WARN"), s.warnings(before))
}

// A client with no provider app takes no events, so the app it is pinned to is not checked,
// wherever its hook delivers.
func (s *ProviderAppMessageHookSuite) TestAClientWithoutAProviderAppIsNotCheckedAtStartup() {
	sent := s.ownSlackApp()
	sent.ProviderAppID, sent.SigningSecret = "", ""
	status, payload := s.putOwnApp("slack_bot", sent)
	s.Require().Equal(http.StatusCreated, status, string(payload))
	before := len(s.logged.String())
	gets := s.asked(s.apiKey, http.MethodGet, "/api/v2/app")

	s.router.WarnWithoutMessageHooks(context.Background())

	s.Equal(gets, s.asked(s.apiKey, http.MethodGet, "/api/v2/app"))
	s.Empty(s.warnings(before))
}

// ProviderAppMessageHookDeploymentSuite is deployment mode with connectors on and a public
// URL: every customer shares the deployment's app, so there is no pinned app to check.
type ProviderAppMessageHookDeploymentSuite struct{ messageHookHarness }

func TestProviderAppMessageHookDeploymentSuite(t *testing.T) {
	runSuite(t, new(ProviderAppMessageHookDeploymentSuite))
}

func (s *ProviderAppMessageHookDeploymentSuite) SetupSuite() {
	s.deployment = true
	s.start(true, providerAppPublicURL)
}

func (s *ProviderAppMessageHookDeploymentSuite) SetupTest() { s.useApp(s.data.createApp()) }

// The startup check asks Stream nothing in deployment mode, even of a record that carries a
// pin.
func (s *ProviderAppMessageHookDeploymentSuite) TestTheStartupCheckAsksNothing() {
	s.Require().False(s.stream.PerApp())
	pin := int64(910000000) + int64(len(s.utils.uuid()))
	_, err := s.store.PutConnectorOAuthClient(context.Background(), &store.ConnectorOAuthClient{
		CustomerID: s.customerID(), ConnectorID: "slack_bot", Registration: core.ClientCustomer,
		ClientID: "synthetic-client", ProviderAppID: "ADEPLOY" + strings.ToUpper(s.utils.uuid()[:6]), StreamAppPK: pin,
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() {
		_ = s.store.DeleteConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot", core.ClientCustomer)
	})
	before := len(s.logged.String())
	asked := len(s.chat.Requests(suiteStreamKey))

	s.router.WarnWithoutMessageHooks(context.Background())

	s.Len(s.chat.Requests(suiteStreamKey), asked)
	s.NotContains(s.logged.String()[before:], "stream_app="+strconv.FormatInt(pin, 10))
	s.NotContains(s.logged.String()[before:], `level=WARN msg="stream: `)
}
