package phone

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
)

const (
	testRouterURL = "https://router.example"
	testApp       = int64(4242)
)

type EnsureCallHookSuite struct {
	suite.Suite
	ctx    context.Context
	server *chattest.Server
	stream *Stream
}

func TestEnsureCallHookSuite(t *testing.T) { suite.Run(t, new(EnsureCallHookSuite)) }

func (s *EnsureCallHookSuite) SetupTest() {
	s.ctx = context.Background()
	s.server = chattest.NewServer(s.T())
	s.stream = NewStreamFromClient(s.server.Client)
}

func (s *EnsureCallHookSuite) rawHook(url string, enabled bool, events ...string) json.RawMessage {
	raw, err := json.Marshal(map[string]any{
		"hook_type": "webhook", "webhook_url": url, "enabled": enabled,
		"event_types": events, "stream_only": "kept",
	})
	s.Require().NoError(err)
	return raw
}

func (s *EnsureCallHookSuite) TestAnAppWithNoHooksGetsOneDeliveringTheCallEvents() {
	wrote, err := s.stream.EnsureCallHook(s.ctx, testRouterURL, testApp)

	s.Require().NoError(err)
	s.True(wrote)
	hooks := s.server.EventHooks(s.server.Key)
	s.Require().Len(hooks, 1)
	s.Equal("https://router.example/v1/phone/hooks/stream/4242", value(hooks[0].WebhookUrl))
	s.True(*hooks[0].Enabled)
	s.ElementsMatch(callHookEvents, hooks[0].EventTypes)
}

func (s *EnsureCallHookSuite) TestAnAppAlreadyDeliveringTheCallEventsIsNotWritten() {
	s.server.SetEventHooks(s.server.Key, s.rawHook("https://router.example/v1/phone/hooks/stream/4242", true, callHookEvents...))
	before := s.server.EventHooksJSON(s.server.Key)

	wrote, err := s.stream.EnsureCallHook(s.ctx, testRouterURL, testApp)

	s.Require().NoError(err)
	s.False(wrote)
	s.Equal(before, s.server.EventHooksJSON(s.server.Key))
}

func (s *EnsureCallHookSuite) TestTheAppsOtherHooksAreKept() {
	other := s.rawHook("https://other.example/hook", true, "message.new")
	s.server.SetEventHooks(s.server.Key, other)

	wrote, err := s.stream.EnsureCallHook(s.ctx, testRouterURL, testApp)

	s.Require().NoError(err)
	s.True(wrote)
	hooks := s.server.EventHooks(s.server.Key)
	s.Require().Len(hooks, 2)
	s.Equal("https://other.example/hook", value(hooks[0].WebhookUrl))
	s.Equal([]string{"message.new"}, hooks[0].EventTypes)
}

func (s *EnsureCallHookSuite) TestTheDeploymentsOwnAppGetsAHookWithNoAppInItsURL() {
	wrote, err := s.stream.EnsureCallHook(s.ctx, testRouterURL+"/", 0)

	s.Require().NoError(err)
	s.True(wrote)
	hooks := s.server.EventHooks(s.server.Key)
	s.Require().Len(hooks, 1)
	s.Equal("https://router.example/v1/phone/hooks/stream", value(hooks[0].WebhookUrl))
}

func (s *EnsureCallHookSuite) TestAHookUnderTheRouterForAnotherAppStillCounts() {
	s.server.SetEventHooks(s.server.Key, s.rawHook("https://router.example/v1/phone/hooks/stream", true, callHookEvents...))
	before := s.server.EventHooksJSON(s.server.Key)

	wrote, err := s.stream.EnsureCallHook(s.ctx, testRouterURL, testApp)

	s.Require().NoError(err)
	s.False(wrote)
	s.Equal(before, s.server.EventHooksJSON(s.server.Key))
}

func (s *EnsureCallHookSuite) TestAHookAtALongerPathIsNotTheRouters() {
	s.server.SetEventHooks(s.server.Key, s.rawHook("https://router.example/v1/phone/hooks/streamX", true, callHookEvents...))

	wrote, err := s.stream.EnsureCallHook(s.ctx, testRouterURL, testApp)

	s.Require().NoError(err)
	s.True(wrote)
	s.Len(s.server.EventHooks(s.server.Key), 2)
}

func (s *EnsureCallHookSuite) TestADisabledHookDoesNotCount() {
	s.server.SetEventHooks(s.server.Key, s.rawHook("https://router.example/v1/phone/hooks/stream/4242", false, callHookEvents...))

	wrote, err := s.stream.EnsureCallHook(s.ctx, testRouterURL, testApp)

	s.Require().NoError(err)
	s.True(wrote)
	hooks := s.server.EventHooks(s.server.Key)
	s.Require().Len(hooks, 1)
	s.True(*hooks[0].Enabled)
}

func (s *EnsureCallHookSuite) TestAPublicURLStreamCannotReachIsRefused() {
	wrote, err := s.stream.EnsureCallHook(s.ctx, "ftp://router.example", testApp)

	s.Error(err)
	s.False(wrote)
	s.Empty(s.server.EventHooks(s.server.Key))
}
