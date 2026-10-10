package chat

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
)

// HooksSuite is the router changing an app's event hooks against Chat in memory, which keeps
// each hook exactly as it was sent.
type HooksSuite struct {
	suite.Suite
	chat   *chattest.Server
	stream *Stream
}

func TestHooksSuite(t *testing.T) {
	suite.Run(t, new(HooksSuite))
}

const routerHook = "https://router.example/v1/chat/hooks/stream/1"

// Hooks shaped as Stream's GET /app writes them (chat monolith/types/event_hook.go), with
// fields the SDK's EventHook does not model, fields Stream sends as null and a time as
// GET /api/v2/app writes one (integer nanoseconds, GetStream/chat
// lib/core/api/encoding/json.go).
var (
	sqsFIFO    = json.RawMessage(`{"id":"h1","hook_type":"sqs","enabled":true,"event_types":null,"sqs_queue_url":"https://sqs.example/q.fifo","sqs_region":"us-east-1","sqs_auth_type":"keys","sqs_key":"AKIA","sqs_secret":"secret","sqs_event_based_message_group_id_enabled":true,"should_send_custom_events":null}`)
	pubSub     = json.RawMessage(`{"id":"h2","hook_type":"gcp_pubsub","enabled":true,"event_types":[],"gcp_pubsub_topic":"projects/p/topics/t1","gcp_pubsub_auth_type":"resource","gcp_pubsub_region":"europe-west4","gcp_pubsub_event_based_ordering_key_enabled":true}`)
	s3Failover = json.RawMessage(`{"id":"h3","hook_type":"webhook","enabled":true,"event_types":["message.new"],"webhook_url":"https://customer.example/hooks","failover_config":{"type":"s3","s3_bucket":"b","s3_path":"p","s3_region":"us-east-1","s3_role_arn":"arn:aws:iam::1:role/r"},"created_at":1759312800123456000}`)
	// s3FailoverSent is s3Failover as the router sends it back: its time as text.
	s3FailoverSent = json.RawMessage(`{"id":"h3","hook_type":"webhook","enabled":true,"event_types":["message.new"],"webhook_url":"https://customer.example/hooks","failover_config":{"type":"s3","s3_bucket":"b","s3_path":"p","s3_region":"us-east-1","s3_role_arn":"arn:aws:iam::1:role/r"},"created_at":"2025-10-01T10:00:00.123456Z"}`)
	otherHooks     = []json.RawMessage{sqsFIFO, pubSub, s3Failover}
	routerOwned    = json.RawMessage(`{"id":"h4","hook_type":"webhook","enabled":false,"event_types":["message.updated"],"webhook_url":"` + routerHook + `","timeout_ms":3000,"failover_config":{"type":"s3","s3_bucket":"mine"}}`)
)

func (s *HooksSuite) SetupTest() {
	s.chat = chattest.NewServer(s.T())
	s.stream = StreamOf(s.chat.Client)
}

// sent is the app's hooks as the last update sent them, each as text.
func (s *HooksSuite) sent() []string {
	var hooks []json.RawMessage
	s.Require().NoError(json.Unmarshal(s.lastUpdate()["event_hooks"], &hooks))
	return texts(hooks...)
}

func texts(hooks ...json.RawMessage) []string {
	var out []string
	for _, hook := range hooks {
		out = append(out, string(hook))
	}
	return out
}

// lastUpdate is the fields the last app update sent, by name.
func (s *HooksSuite) lastUpdate() map[string]json.RawMessage {
	updates := s.chat.AppUpdates("test")
	s.Require().NotEmpty(updates)
	var fields map[string]json.RawMessage
	s.Require().NoError(json.Unmarshal(updates[len(updates)-1], &fields))
	return fields
}

func (s *HooksSuite) TestPointingKeepsEveryOtherHookAsStreamSentIt() {
	s.chat.SetEventHooks("test", otherHooks...)

	updated, err := s.stream.PointMessageHook(context.Background(), routerHook)

	s.Require().NoError(err)
	s.False(updated)
	sent := s.sent()
	s.Require().Len(sent, 4)
	s.Equal(texts(sqsFIFO, pubSub, s3FailoverSent), sent[:3], "every other hook, every field, byte for byte but its times")
	hooks := s.chat.EventHooks("test")
	s.Equal(routerHook, *hooks[3].WebhookUrl)
	s.Equal([]string{"message.new"}, hooks[3].EventTypes)
	s.True(*hooks[3].Enabled)
}

func (s *HooksSuite) TestPointingSendsTheHooksAndNoOtherSetting() {
	s.chat.SetEventHooks("test", otherHooks...)

	_, err := s.stream.PointMessageHook(context.Background(), routerHook)

	s.Require().NoError(err)
	s.Equal([]string{"event_hooks"}, keys(s.lastUpdate()),
		"no grants, user_search_disallowed_roles or webhook_events sent as null")
}

func (s *HooksSuite) TestPointingAgainChangesOnlyWhatTheRouterSets() {
	s.chat.SetEventHooks("test", sqsFIFO, routerOwned, pubSub)

	updated, err := s.stream.PointMessageHook(context.Background(), routerHook)

	s.Require().NoError(err)
	s.True(updated)
	sent := s.sent()
	s.Require().Len(sent, 3)
	s.Equal(string(sqsFIFO), sent[0])
	s.Equal(string(pubSub), sent[2])
	var own map[string]any
	s.Require().NoError(json.Unmarshal([]byte(sent[1]), &own))
	s.Equal(map[string]any{
		"id": "h4", "hook_type": "webhook", "enabled": true, "event_types": []any{"message.new"},
		"webhook_url": routerHook, "timeout_ms": float64(3000),
		"failover_config": map[string]any{"type": "s3", "s3_bucket": "mine"},
	}, own, "its events and switch set, the rest of it kept")
}

func (s *HooksSuite) TestRemovingKeepsEveryOtherHookAsStreamSentIt() {
	s.chat.SetEventHooks("test", sqsFIFO, routerOwned, pubSub, s3Failover)

	removed, err := s.stream.RemoveMessageHook(context.Background(), routerHook)

	s.Require().NoError(err)
	s.True(removed)
	s.Equal(texts(sqsFIFO, pubSub, s3FailoverSent), s.sent())
	s.Equal([]string{"event_hooks"}, keys(s.lastUpdate()))
}

func keys(fields map[string]json.RawMessage) []string {
	var names []string
	for name := range fields {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

// GET /api/v2/app writes a hook's times as integer nanoseconds, and an app update refuses
// anything but text for them (Stream decodes them into a time.Time). They go back as the
// same instant, in RFC 3339, and the rest of the hook as it was.
func (s *HooksSuite) TestAHooksTimesGoBackAsText() {
	read := json.RawMessage(`{"id":"h9","hook_type":"webhook","enabled":true,"event_types":["call.session_started"],"webhook_url":"https://other.example/h","created_at":1791485542458572000,"updated_at":1791588958134941000}`)
	s.chat.SetEventHooks("test", read)

	_, err := s.stream.PointMessageHook(context.Background(), routerHook)

	s.Require().NoError(err)
	sent := s.sent()
	s.Require().Len(sent, 2)
	s.Equal(`{"id":"h9","hook_type":"webhook","enabled":true,"event_types":["call.session_started"],"webhook_url":"https://other.example/h","created_at":"2026-10-08T18:52:22.458572Z","updated_at":"2026-10-09T23:35:58.134941Z"}`, sent[0])
}

// The router's own hook, changed, keeps its times too, as text.
func (s *HooksSuite) TestTheRoutersOwnHookKeepsItsTimesAsText() {
	s.chat.SetEventHooks("test", json.RawMessage(`{"id":"h4","hook_type":"webhook","enabled":false,"event_types":["message.updated"],"webhook_url":"`+routerHook+`","created_at":1791485542458572000}`))

	_, err := s.stream.PointMessageHook(context.Background(), routerHook)

	s.Require().NoError(err)
	var own map[string]any
	s.Require().NoError(json.Unmarshal([]byte(s.sent()[0]), &own))
	s.Equal("2026-10-08T18:52:22.458572Z", own["created_at"])
	s.Equal([]any{"message.new"}, own["event_types"])
}
