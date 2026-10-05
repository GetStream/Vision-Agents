package plugins

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
)

type UserSuite struct {
	suite.Suite
	calendar Plugin
	logo     string
}

func TestUserSuite(t *testing.T) {
	suite.Run(t, new(UserSuite))
}

func (s *UserSuite) SetupTest() {
	calendar, ok := Lookup("google_calendar")
	s.Require().True(ok)
	s.calendar = calendar
	s.logo = (&Auth{PublicURL: "https://router.example"}).LogoURL(calendar.ID)
}

func (s *UserSuite) TestEachUserPluginIsOfferedAsTwoToolsWhoseNamesNeedNoLogin() {
	tools := UserTools([]string{"google_calendar", "carrier-pigeon"})

	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}
	s.Equal([]string{"google_calendar__list_tools", "google_calendar__call_tool"}, names)
	s.NoError(harness.Tools{Tools: tools}.Validate())
}

func (s *UserSuite) TestTheResultAskingForALoginIsReadBackAsTheAttachment() {
	result := AuthorizationResult(s.calendar, "https://accounts.google.com/o/oauth2/v2/auth?state=abc&scope=x", s.logo)

	found, ok := RequestedAuthorization("google_calendar__list_tools", result)

	s.Require().True(ok)
	s.Equal(Authorization{
		Type:         AuthorizationType,
		PluginID:     "google_calendar",
		Title:        "Connect Google Calendar",
		AuthorizeURL: "https://accounts.google.com/o/oauth2/v2/auth?state=abc&scope=x",
		Text:         s.calendar.Description,
		ThumbURL:     s.logo,
		TitleLink:    "https://accounts.google.com/o/oauth2/v2/auth?state=abc&scope=x",
	}, found)
	var read struct {
		Message string `json:"message"`
	}
	s.Require().NoError(json.Unmarshal([]byte(result), &read))
	s.Contains(read.Message, "Tell them to press it", "the model is told what to say")
}

func (s *UserSuite) TestOnlyThePluginsOwnToolMayAskForItsLogin() {
	result := AuthorizationResult(s.calendar, "https://accounts.google.com/auth", s.logo)

	for _, tool := range []string{"sentry__list_tools", "weather", "google_calendar"} {
		_, ok := RequestedAuthorization(tool, result)
		s.False(ok, tool)
	}
}

func (s *UserSuite) TestAnythingButExactlyTheResultAsksForNothing() {
	for _, result := range []string{
		"",
		"please connect at https://accounts.google.com/auth",
		`{"status":"authorization_required","message":"","attachment":{"type":"plugin_authorization","plugin_id":"google_calendar","title":"Connect","authorize_url":"http://accounts.google.com/auth"}}`,
		`{"status":"authorization_required","message":"","attachment":{"type":"plugin_authorization","plugin_id":"google_calendar","title":"Connect","authorize_url":"https://accounts.google.com/auth","extra":1}}`,
		`{"status":"authorization_required","message":"","attachment":{"type":"plugin_authorization","plugin_id":"notion","title":"Connect","authorize_url":"https://accounts.google.com/auth"}}`,
		`{"status":"answered","message":"","attachment":{"type":"plugin_authorization","plugin_id":"google_calendar","title":"Connect","authorize_url":"https://accounts.google.com/auth"}}`,
	} {
		_, ok := RequestedAuthorization("google_calendar__list_tools", result)
		s.False(ok, result)
	}
}

func (s *UserSuite) TestTheCardMayOnlyShowWhatTheCatalogSaysAboutThatPlugin() {
	asked := Authorization{
		Type:         AuthorizationType,
		PluginID:     "google_calendar",
		Title:        "Connect Google Calendar",
		AuthorizeURL: "https://accounts.google.com/auth",
	}
	s.True(ValidAuthorization(asked), "the fields a card adds are all optional")

	for name, wrong := range map[string]func(*Authorization){
		"an image of the server's own choosing": func(a *Authorization) {
			a.ThumbURL = "https://evil.example/pixel.png"
		},
		"another plugin's logo": func(a *Authorization) {
			a.ThumbURL = "https://router.example" + LogoPath("sentry")
		},
		"a logo behind a query it could track with": func(a *Authorization) {
			a.ThumbURL = "https://router.example" + LogoPath("google_calendar") + "?who=someone"
		},
		"a description of its own": func(a *Authorization) {
			a.Text = "Enter your password to continue."
		},
		"a link somewhere other than the login": func(a *Authorization) {
			a.TitleLink = "https://evil.example/login"
		},
	} {
		refused := asked
		wrong(&refused)
		s.False(ValidAuthorization(refused), name)
	}

	allowed := asked
	allowed.ThumbURL = (&Auth{PublicURL: "http://localhost:8080"}).LogoURL("google_calendar")
	s.True(ValidAuthorization(allowed), "a router developed against localhost serves http")
}

func (s *UserSuite) TestListedToolsAreNamedAsCallToolTakesThem() {
	listed := ListedTools("google_calendar", []harness.Tool{{
		Name:        "google_calendar__list_events",
		Description: "Events in a range (via google_calendar)",
		Parameters:  map[string]any{"type": "object"},
	}})

	s.JSONEq(`{"tools":[{"name":"list_events","description":"Events in a range","input_schema":{"type":"object"}}]}`, listed)
}
