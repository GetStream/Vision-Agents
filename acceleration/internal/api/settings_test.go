//go:build integration

package api

import (
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// SettingsSuite reads what the router does for an app, against the suite's deployment app
// and apps of the customers' own, each a Stream app in memory.
type SettingsSuite struct {
	RouterSuite
}

func TestSettingsSuite(t *testing.T) {
	runSuite(t, new(SettingsSuite))
}

func (s *SettingsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *SettingsSuite) settings() StreamSettings {
	var read AppSettings
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/settings/app", nil, &read))
	return read.Stream
}

func (s *SettingsSuite) TestAppSettingsReportTheDeploymentAppInDeploymentMode() {
	read := s.settings()

	s.Equal(StreamTenancy("deployment"), read.Tenancy)
	s.Equal(WritesIntoDeploymentApp, read.WritesInto, "an app with none of its own writes into the router's")
	s.Equal(StreamTypeState(streamapp.TypePresent), read.ChannelType)
	s.Equal(StreamTypeState(streamapp.TypePresent), read.CallType)
	s.NotNil(read.CheckedAt)
}

func (s *SettingsSuite) TestAppSettingsReportAnAppsOwnStreamApp() {
	own := s.giveApp(s.customerID(), 4242, "own-key")

	read := s.settings()

	s.Equal(WritesIntoThisApp, read.WritesInto)
	s.Equal(1, own.AppReads(), "what the app holds is read from its own app")
}

func (s *SettingsSuite) TestAppSettingsReportAMissingAgentChannelType() {
	own := s.giveApp(s.customerID(), 4242, "own-key")
	own.SetApp(chattest.App{ID: 4242, ChannelTypes: map[string]map[string][]string{"messaging": {}}})

	read := s.settings()

	s.Equal(StreamTypeState(streamapp.TypeMissing), read.ChannelType)
	s.Equal(StreamTypeState(streamapp.TypeMissing), read.CallType)
}

func (s *SettingsSuite) TestCheckReportsAnAgentChannelTypeClientsCanForge() {
	// A user who may update an agent channel may rewrite whose conversation it says it is.
	own := s.giveApp(s.customerID(), 4242, "own-key")
	own.SetApp(chattest.App{
		ID:           4242,
		ChannelTypes: map[string]map[string][]string{"agent": {"user": {"update-channel-owner"}}},
		CallTypes:    []string{"agent"},
	})

	s.Equal(StreamTypeState(streamapp.TypeUnsafe), s.settings().ChannelType)
}

func (s *SettingsSuite) TestAppSettingsCarryNoSecret() {
	s.giveApp(s.customerID(), 918273645, "own-key")

	status, body := s.serverClient.call(http.MethodGet, "/v1/settings/app", nil)
	s.Require().Equal(http.StatusOK, status)

	s.NotContains(string(body), "own-key-secret")
	s.NotContains(string(body), suiteStreamSecret)
	s.NotContains(string(body), "918273645", "v1 names no app by its id")
	var fields map[string]map[string]any
	s.Require().NoError(json.Unmarshal(body, &fields))
	s.ElementsMatch([]string{"tenancy", "writes_into", "channel_type", "call_type", "checked_at"},
		slices.Collect(maps.Keys(fields["stream"])))
}

func (s *SettingsSuite) TestAppSettingsAreServerSideOnly() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/settings/app", nil, nil)
	})
}

func (s *SettingsSuite) TestAppSettingsAnswerWhenStreamCannotBeReached() {
	// The settings are worth reading most when Stream is the trouble.
	closed := httptest.NewServer(nil)
	closed.Close()
	s.apps.mu.Lock()
	s.apps.own[s.customerID()] = streamapp.Identity{
		CustomerID: s.customerID(), StreamApp: 4242, APIKey: "own-key",
		Secret: streamapp.NewSecret("own-key-secret"), BaseURL: closed.URL,
	}
	s.apps.mu.Unlock()
	s.stream.Invalidate(s.customerID())

	read := s.settings()

	s.Equal(WritesIntoThisApp, read.WritesInto)
	s.Equal(StreamTypeState(streamapp.TypeUnknown), read.ChannelType)
	s.Equal(StreamTypeState(streamapp.TypeUnknown), read.CallType)
	s.Nil(read.CheckedAt)
}
