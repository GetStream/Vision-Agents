package slackapps_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/slackapps"
)

// The router's callback and events URLs as a deployment at router.example would have them.
const (
	redirectURL = "https://router.example/v1/agents/connectors/oauth/callback"
	requestURL  = "https://router.example/v1/connectors/events/slack_bot/A0123456789"
)

// SlackAppsSuite runs the client against the fake Slack and builds manifests from the
// built-in Slack connector and core's bot fixture.
type SlackAppsSuite struct {
	suite.Suite
	ctx    context.Context
	slack  *fakeprovider.Server
	client *slackapps.Client
}

func TestSlackAppsSuite(t *testing.T) {
	suite.Run(t, new(SlackAppsSuite))
}

func (s *SlackAppsSuite) SetupTest() {
	s.ctx = context.Background()
	s.slack = fakeprovider.New(s.T())
	client, err := slackapps.New(slackapps.Config{HTTP: s.slack.Client(), BaseURL: s.slack.URL + fakeprovider.PathSlackAPI})
	s.Require().NoError(err)
	s.client = client
}

func (s *SlackAppsSuite) TestARotationGivesATokenThatExpiresInTwelveHoursAndANewRefreshToken() {
	given := s.slack.NewConfigToken()

	rotated, err := s.client.Rotate(s.ctx, given)

	s.Require().NoError(err)
	s.NotEmpty(rotated.Token)
	s.NotEqual(given, rotated.RefreshToken)
	s.WithinDuration(time.Now().Add(12*time.Hour), rotated.ExpiresAt, time.Minute)
}

func (s *SlackAppsSuite) TestASpentRefreshTokenIsRefused() {
	given := s.slack.NewConfigToken()
	_, err := s.client.Rotate(s.ctx, given)
	s.Require().NoError(err)

	_, err = s.client.Rotate(s.ctx, given)

	s.ErrorIs(err, slackapps.ErrInvalidRefreshToken)
	s.Equal(1, s.slack.ConfigTokenRotations())
}

func (s *SlackAppsSuite) TestCreateReturnsTheNewAppsIdAndCredentials() {
	token := s.token()

	created, err := s.client.Create(s.ctx, token, s.manifest(""))

	s.Require().NoError(err)
	apps := s.slack.SlackApps()
	s.Require().Len(apps, 1)
	s.Equal(slackapps.Credentials{
		AppID: apps[0].AppID, ClientID: apps[0].ClientID,
		ClientSecret: apps[0].ClientSecret, SigningSecret: apps[0].SigningSecret,
	}, created)
}

func (s *SlackAppsSuite) TestUpdateReplacesTheAppsManifest() {
	token := s.token()
	created, err := s.client.Create(s.ctx, token, s.manifest(""))
	s.Require().NoError(err)

	s.Require().NoError(s.client.Update(s.ctx, token, created.AppID, s.manifest(requestURL)))

	var sent slackapps.Manifest
	s.Require().NoError(json.Unmarshal(s.slack.SlackApps()[0].Manifest, &sent))
	s.Require().NotNil(sent.Settings.EventSubscriptions)
	s.Equal(requestURL, sent.Settings.EventSubscriptions.RequestURL)
}

func (s *SlackAppsSuite) TestDeletingAnAppTwiceFindsItGoneTheSecondTime() {
	token := s.token()
	created, err := s.client.Create(s.ctx, token, s.manifest(""))
	s.Require().NoError(err)

	s.Require().NoError(s.client.Delete(s.ctx, token, created.AppID))
	err = s.client.Delete(s.ctx, token, created.AppID)

	s.ErrorIs(err, slackapps.ErrAppNotFound)
	s.True(s.slack.SlackApps()[0].Deleted)
}

func (s *SlackAppsSuite) TestAnExpiredConfigTokenIsRefused() {
	token := s.token()
	s.slack.Advance(fakeprovider.ConfigTokenTTL)

	_, err := s.client.Create(s.ctx, token, s.manifest(""))

	s.ErrorIs(err, slackapps.ErrTokenExpired)
	s.Empty(s.slack.SlackApps())
}

func (s *SlackAppsSuite) TestTheBotConnectorsAppAsksForItsBotScopesAndSubscribesToItsEvents() {
	connector := s.connector()

	manifest, err := slackapps.ManifestFor(connector, slackapps.Template{Name: "Acme agent", RedirectURL: redirectURL, RequestURL: requestURL})

	s.Require().NoError(err)
	s.Equal("Acme agent", manifest.DisplayInformation.Name)
	s.Equal(connector.Scopes.List, manifest.OAuthConfig.Scopes.Bot)
	s.Empty(manifest.OAuthConfig.Scopes.User)
	s.Equal(&slackapps.Features{BotUser: &slackapps.BotUser{DisplayName: "Acme agent"}}, manifest.Features)
	s.Equal([]string{redirectURL}, manifest.OAuthConfig.RedirectURLs)
	s.Equal(&slackapps.EventSubscriptions{RequestURL: requestURL, BotEvents: []string{"message.channels", "message.im", "tokens_revoked"}},
		manifest.Settings.EventSubscriptions)
	s.True(manifest.Settings.TokenRotationEnabled)
	s.False(manifest.Settings.OrgDeployEnabled)
	s.False(manifest.Settings.SocketModeEnabled)
}

func (s *SlackAppsSuite) TestAUserTokenConnectorsAppAsksForUserScopesAndHasNoBotUser() {
	raw, err := providers.FS.ReadFile("slack.yaml")
	s.Require().NoError(err)
	connector, err := core.ParseManifest(raw)
	s.Require().NoError(err)

	manifest, err := slackapps.ManifestFor(connector, slackapps.Template{Name: "Acme agent", RedirectURL: redirectURL, RequestURL: requestURL})

	s.Require().NoError(err)
	s.Equal(connector.Scopes.List, manifest.OAuthConfig.Scopes.User)
	s.Empty(manifest.OAuthConfig.Scopes.Bot)
	s.Nil(manifest.Features)
	s.Empty(manifest.Settings.EventSubscriptions.UserEvents, "slack.yaml lists no subscriptions")
}

func (s *SlackAppsSuite) TestWithoutARequestURLTheAppSubscribesToNothing() {
	manifest, err := slackapps.ManifestFor(s.connector(), slackapps.Template{Name: "Acme agent", RedirectURL: redirectURL})

	s.Require().NoError(err)
	s.Nil(manifest.Settings.EventSubscriptions)
}

func (s *SlackAppsSuite) TestANameLongerThanSlacksLimitIsRefused() {
	_, err := slackapps.ManifestFor(s.connector(), slackapps.Template{Name: strings.Repeat("a", 36), RedirectURL: redirectURL})

	s.ErrorContains(err, "1 to 35 characters")
}

func (s *SlackAppsSuite) TestMoreThanTenIPRangesAreRefused() {
	ranges := make([]string, 11)
	for i := range ranges {
		ranges[i] = fmt.Sprintf("203.0.113.%d/32", i)
	}

	_, err := slackapps.ManifestFor(s.connector(), slackapps.Template{Name: "Acme agent", RedirectURL: redirectURL, AllowedIPAddressRanges: ranges})

	s.ErrorContains(err, "at most 10")
}

func (s *SlackAppsSuite) TestAnIPRangeThatIsNotOneIsRefused() {
	_, err := slackapps.ManifestFor(s.connector(), slackapps.Template{Name: "Acme agent", RedirectURL: redirectURL, AllowedIPAddressRanges: []string{"router.example"}})

	s.ErrorContains(err, "not an IP address or a CIDR range")
}

func (s *SlackAppsSuite) TestAConnectorThatDoesNotAuthorizeAtSlackGetsNoSlackApp() {
	connector := s.connector()
	connector.Endpoints = map[string]string{"authorize": "https://slack.com.example/oauth/v2/authorize"}

	s.False(slackapps.Serves(connector))
	_, err := slackapps.ManifestFor(connector, slackapps.Template{Name: "Acme agent", RedirectURL: redirectURL})
	s.ErrorContains(err, "does not authorize at slack.com")
}

func (s *SlackAppsSuite) TestCredentialsAndConfigTokensNeverPrintTheirSecrets() {
	credentials := slackapps.Credentials{AppID: "A0123456789", ClientID: "1.2", ClientSecret: "client-secret-value", SigningSecret: "signing-secret-value"}
	token := slackapps.ConfigToken{Token: "xoxe.xoxp-token-value", RefreshToken: "xoxe-refresh-value", ExpiresAt: time.Now()}
	var text, structured bytes.Buffer
	slog.New(slog.NewTextHandler(&text, nil)).Info("x", "credentials", credentials, "token", token)
	slog.New(slog.NewJSONHandler(&structured, nil)).Info("x", "credentials", credentials, "token", token)

	for _, printed := range []string{
		fmt.Sprintf("%v %v", credentials, token), fmt.Sprintf("%+v %+v", credentials, token),
		fmt.Sprintf("%#v %#v", credentials, token), fmt.Sprintf("%s %s", credentials, token),
		text.String(), structured.String(),
	} {
		for _, secret := range []string{"client-secret-value", "signing-secret-value", "token-value", "refresh-value"} {
			s.NotContains(printed, secret)
		}
		s.Contains(printed, "A0123456789")
	}
}

// token is a fresh configuration token, rotated from one the fake's admin generated.
func (s *SlackAppsSuite) token() string {
	rotated, err := s.client.Rotate(s.ctx, s.slack.NewConfigToken())
	s.Require().NoError(err)
	return rotated.Token
}

// connector is the built-in Slack bot connector, the one a managed app is for, at its latest
// revision.
func (s *SlackAppsSuite) connector() core.Manifest {
	raw, err := providers.FS.ReadFile("slack_bot.yaml")
	s.Require().NoError(err)
	connector, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	return connector
}

// manifest is the Slack bot connector's app manifest, with request as its events URL.
func (s *SlackAppsSuite) manifest(request string) slackapps.Manifest {
	manifest, err := slackapps.ManifestFor(s.connector(), slackapps.Template{Name: "Acme agent", RedirectURL: redirectURL, RequestURL: request})
	s.Require().NoError(err)
	return manifest
}
