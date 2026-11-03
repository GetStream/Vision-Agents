//go:build integration

package api

import (
	"context"
	"net/http"
	"strings"

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

// The staging incident of 2026-10-10 had the app's backend put its own Slack app through the
// oauth-client PUT. Chat delivery answers its messages now, so the PUT points no hook, whatever
// the client and provider app it stores.
func (s *ProviderAppMessageHookSuite) TestPuttingTheAppsOwnSlackAppLeavesThePinnedAppsHooksAlone() {
	hooks := s.seeded()

	status, payload := s.putOwnApp("slack_bot", s.ownSlackApp())

	s.Require().Equal(http.StatusCreated, status, string(payload))
	record, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.Require().NoError(err)
	s.Equal(s.appID(), record.StreamAppPK)
	s.leftAlone(hooks)
}

// The PUT of a client with no provider app, and of a connector with no channel, were the
// two cases that pointed no hook before; they still point none.
func (s *ProviderAppMessageHookSuite) TestAnOAuthClientWithNoProviderAppOrNoChannelLeavesTheHooksAlone() {
	hooks := s.seeded()
	sent := s.ownSlackApp()
	sent.ProviderAppID, sent.SigningSecret = "", ""

	status, payload := s.putOwnApp("slack_bot", sent)
	s.Require().Equal(http.StatusCreated, status, string(payload))
	status, payload = s.putOwnApp("github", ConnectorOAuthClientRequest{
		ClientID: "synthetic-client-" + s.utils.uuid(), ClientSecret: "synthetic-client-secret", ProviderAppID: "synthetic-github-app",
	})
	s.Require().Equal(http.StatusCreated, status, string(payload))

	s.leftAlone(hooks)
}

// Connectors off, the control: the oauth-client PUT answers as on base (probed on 599298c6:
// 400 not_configured) and asks nothing of Stream, even of an app a provider app stored while
// connectors were on is pinned to.
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
	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), `"code":"not_configured"`)
	s.Contains(string(payload), "OAuth clients cannot be stored: connectors are not enabled on this deployment")
	s.Len(s.chat.Requests(s.apiKey), asked, "nothing is asked of the customer's Stream app")
	s.Empty(s.chat.EventHooks(s.apiKey))
}
