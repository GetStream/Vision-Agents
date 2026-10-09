package main

import (
	"context"
	"net/http"
	"testing"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

// HooksSuite is `phone hooks` changing an app's hooks against Chat in memory, which refuses
// an update holding a hook at a host that does not resolve, as Stream does (AI-990 F22).
type HooksSuite struct {
	suite.Suite
	chat   *chattest.Server
	stream *phone.Stream
}

func TestHooksSuite(t *testing.T) {
	suite.Run(t, new(HooksSuite))
}

const (
	goneBase  = "https://gone.trycloudflare.com"
	otherHook = "https://colleague.ngrok-free.app/v1/phone/hooks/stream"
)

// SetupTest is an app with a gone tunnel's call and message hooks, and a colleague's call
// hook at a host that still resolves, as app 1257545 had on 2026-10-09.
func (s *HooksSuite) SetupTest() {
	s.chat = chattest.NewServer(s.T())
	s.stream = phone.NewStreamFromClient(s.chat.Client)
	s.seed()
	s.chat.Unresolvable("gone.trycloudflare.com")
}

// seed puts the suite's three hooks on the app, and more after them, while every host still
// resolves.
func (s *HooksSuite) seed(more ...getstream.EventHook) {
	hooks := append([]getstream.EventHook{
		hookAt(goneBase+"/v1/phone/hooks/stream", "call.session_started", "call.session_ended"),
		hookAt(goneBase+"/v1/chat/hooks/stream", "message.new"),
		hookAt(otherHook, "call.session_started", "call.session_ended"),
	}, more...)
	_, err := s.chat.Client.UpdateApp(context.Background(), &getstream.UpdateAppRequest{EventHooks: hooks})
	s.Require().NoError(err)
}

// olderTunnelGone builds the app again with a second gone tunnel's call hook on it too.
func (s *HooksSuite) olderTunnelGone() {
	s.chat = chattest.NewServer(s.T())
	s.stream = phone.NewStreamFromClient(s.chat.Client)
	s.seed(hookAt("https://older.trycloudflare.com/v1/phone/hooks/stream"))
	s.chat.Unresolvable("gone.trycloudflare.com")
	s.chat.Unresolvable("older.trycloudflare.com")
}

func (s *HooksSuite) urls() []string {
	var urls []string
	for _, hook := range s.chat.EventHooks("test") {
		urls = append(urls, *hook.WebhookUrl)
	}
	return urls
}

func (s *HooksSuite) updates() int {
	updates := 0
	for _, request := range s.chat.Requests("test") {
		if request.Method == http.MethodPatch {
			updates++
		}
	}
	return updates
}

func hookAt(url string, events ...string) getstream.EventHook {
	enabled, webhook := true, "webhook"
	return getstream.EventHook{HookType: &webhook, WebhookUrl: &url, Enabled: &enabled, EventTypes: events}
}

func (s *HooksSuite) TestRemovingAGoneTunnelDropsBothItsHooks() {
	said, err := changeHooks(context.Background(), s.stream, "", []string{goneBase + "/"}, "")

	s.Require().NoError(err)
	s.Equal([]string{otherHook}, s.urls())
	s.Equal([]string{
		"call events no longer go to " + goneBase + "/v1/phone/hooks/stream",
		"messages no longer go to " + goneBase + "/v1/chat/hooks/stream",
	}, said)
}

func (s *HooksSuite) TestMovingOffAGoneTunnelPointsTheNewOneInTheSameRun() {
	said, err := changeHooks(context.Background(), s.stream, "https://new.ngrok-free.dev", []string{goneBase}, "")

	s.Require().NoError(err)
	s.Equal([]string{
		otherHook,
		"https://new.ngrok-free.dev/v1/phone/hooks/stream",
		"https://new.ngrok-free.dev/v1/chat/hooks/stream",
	}, s.urls())
	s.Equal(2, s.updates(), "the seed and one update")
	events := map[string][]string{}
	for _, hook := range s.chat.EventHooks("test") {
		events[*hook.WebhookUrl] = hook.EventTypes
	}
	s.Equal([]string{"call.session_started", "call.session_ended"}, events["https://new.ngrok-free.dev/v1/phone/hooks/stream"])
	s.Equal([]string{"message.new"}, events["https://new.ngrok-free.dev/v1/chat/hooks/stream"])
	s.Contains(said, "messages now go to https://new.ngrok-free.dev/v1/chat/hooks/stream")
}

func (s *HooksSuite) TestAnotherGoneTunnelLeftInPlaceIsNamedWithTheFix() {
	s.olderTunnelGone()

	_, err := changeHooks(context.Background(), s.stream, "", []string{goneBase}, "")

	s.Require().Error(err)
	s.Contains(err.Error(), "https://older.trycloudflare.com/v1/phone/hooks/stream")
	s.Contains(err.Error(), "-remove <its base url>")
	s.Len(s.urls(), 4, "nothing changed")
}

func (s *HooksSuite) TestRemovingEveryGoneTunnelAtOnceWorks() {
	s.olderTunnelGone()

	_, err := changeHooks(context.Background(), s.stream, "", []string{goneBase, "https://older.trycloudflare.com"}, "")

	s.Require().NoError(err)
	s.Equal([]string{otherHook}, s.urls())
}

func (s *HooksSuite) TestAnAppsOwnPathIsRemovedWithItsSegment() {
	_, err := changeHooks(context.Background(), s.stream, "https://router.example", []string{goneBase}, "")
	s.Require().NoError(err)
	_, err = changeHooks(context.Background(), s.stream, "https://router.example", nil, "/1234")
	s.Require().NoError(err)

	_, err = changeHooks(context.Background(), s.stream, "", []string{"https://router.example"}, "/1234")

	s.Require().NoError(err)
	s.Equal([]string{otherHook, "https://router.example/v1/phone/hooks/stream", "https://router.example/v1/chat/hooks/stream"}, s.urls())
}

// Removing a url nothing delivers to asks Stream only to read, as on base: a probe of
// base's RemoveCallHook against Chat in memory made one request, the GET of the app.
func (s *HooksSuite) TestRemovingWhatIsNotThereChangesNothing() {
	before := s.updates()

	said, err := changeHooks(context.Background(), s.stream, "", []string{"https://nowhere.example"}, "")

	s.Require().NoError(err)
	s.Equal(before, s.updates())
	s.Equal([]string{
		"nothing was delivering to https://nowhere.example/v1/phone/hooks/stream",
		"nothing was delivering to https://nowhere.example/v1/chat/hooks/stream",
	}, said)
}

func (s *HooksSuite) TestPointingAtAURLThatIsNotOneIsRefusedBeforeAnUpdate() {
	before := s.updates()

	_, err := changeHooks(context.Background(), s.stream, "example.ngrok.app", nil, "")

	s.Require().Error(err)
	s.Equal(before, s.updates())
}
