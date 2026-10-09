package main

import (
	"bytes"
	"context"
	"log/slog"
	"testing"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
)

// MessageHookWarningSuite is the startup check that the deployment's own Stream app sends
// its new messages to this router (AI-990 F19), against Chat in memory.
type MessageHookWarningSuite struct {
	suite.Suite
	stream   *chattest.Server
	settings config.Config
	logs     bytes.Buffer
}

func TestMessageHookWarningSuite(t *testing.T) {
	suite.Run(t, new(MessageHookWarningSuite))
}

const (
	hookKey    = "deploy-key"
	hookPublic = "https://router.example"
	hookFix    = "go run ./cmd/phone hooks -url " + hookPublic
)

// SetupTest is a router with connectors on and a public URL, whose app has no hooks yet.
func (s *MessageHookWarningSuite) SetupTest() {
	s.stream = chattest.NewServer(s.T())
	s.settings = config.Defaults()
	s.settings.Stream.APIKey, s.settings.Stream.APISecret, s.settings.Stream.BaseURL = hookKey, "deploy-secret", s.stream.URL
	s.settings.Connectors.Enabled = true
	s.settings.PublicURL = hookPublic + "/"
	s.logs.Reset()
}

// hooks gives the deployment's app these hooks, as an operator would have set them.
func (s *MessageHookWarningSuite) hooks(hooks ...getstream.EventHook) {
	client, err := getstream.NewClient(hookKey, "deploy-secret", getstream.WithBaseUrl(s.stream.URL))
	s.Require().NoError(err)
	_, err = client.UpdateApp(context.Background(), &getstream.UpdateAppRequest{EventHooks: hooks})
	s.Require().NoError(err)
}

func (s *MessageHookWarningSuite) check() string {
	warnWithoutMessageHook(context.Background(), s.settings, slog.New(slog.NewTextHandler(&s.logs, nil)))
	return s.logs.String()
}

func hookAt(url string, enabled bool, events ...string) getstream.EventHook {
	return getstream.EventHook{HookType: pointerTo("webhook"), WebhookUrl: &url, Enabled: &enabled, EventTypes: events}
}

func pointerTo(text string) *string { return &text }

func (s *MessageHookWarningSuite) TestAnAppWithOnlyACallHookIsWarnedAboutWithTheCommand() {
	s.hooks(hookAt(hookPublic+"/v1/phone/hooks/stream", true, "call.session_started", "call.session_ended"))

	logged := s.check()

	s.Contains(logged, "level=WARN")
	s.Contains(logged, "never answered")
	s.Contains(logged, hookFix)
	s.Contains(logged, "hook="+hookPublic+"/v1/chat/hooks/stream")
	s.NotContains(logged, "deploy-secret")
}

func (s *MessageHookWarningSuite) TestAMessageHookAtTheRouterIsNotWarnedAbout() {
	s.hooks(hookAt(hookPublic+"/v1/chat/hooks/stream", true, "message.new"))

	s.Empty(s.check())
}

func (s *MessageHookWarningSuite) TestAMessageHookOnTheAppsOwnPathCounts() {
	s.hooks(hookAt(hookPublic+"/v1/chat/hooks/stream/1257545", true, "message.new"))

	s.Empty(s.check())
}

func (s *MessageHookWarningSuite) TestAHookAskingForEveryEventCounts() {
	s.hooks(hookAt(hookPublic+"/v1/chat/hooks/stream", true))

	s.Empty(s.check())
}

func (s *MessageHookWarningSuite) TestAMessageHookThatIsSwitchedOffIsWarnedAbout() {
	s.hooks(hookAt(hookPublic+"/v1/chat/hooks/stream", false, "message.new"))

	s.Contains(s.check(), hookFix)
}

func (s *MessageHookWarningSuite) TestAHookAtTheRouterAskingForOtherEventsIsWarnedAbout() {
	s.hooks(hookAt(hookPublic+"/v1/chat/hooks/stream", true, "reaction.new"))

	s.Contains(s.check(), hookFix)
}

func (s *MessageHookWarningSuite) TestAMessageHookAtAnotherRouterIsWarnedAbout() {
	s.hooks(hookAt("https://gone.trycloudflare.com/v1/chat/hooks/stream", true, "message.new"))

	s.Contains(s.check(), hookFix)
}

func (s *MessageHookWarningSuite) TestAnAppThatCannotBeReadIsAWarningNotAFailure() {
	s.stream.SetAppFor(hookKey, chattest.App{Refuses: true})

	logged := s.check()

	s.Contains(logged, "level=WARN")
	s.Contains(logged, "could not read the Stream app's hooks")
	s.NotContains(logged, hookFix)
}

// With connectors off Stream is not asked anything and nothing is logged: on base the router
// has no such check at startup, so there is nothing for it to ask or say.
func (s *MessageHookWarningSuite) TestWithConnectorsOffStreamIsNotAsked() {
	s.settings.Connectors.Enabled = false

	s.Empty(s.check())
	s.Empty(s.stream.Requests(hookKey))
}

func (s *MessageHookWarningSuite) TestWithoutAPublicURLStreamIsNotAsked() {
	s.settings.PublicURL = ""

	s.Empty(s.check())
	s.Empty(s.stream.Requests(hookKey))
}
